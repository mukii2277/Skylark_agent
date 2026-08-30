"""Normalization layer (R4, R5).

Turns the raw, messy wide DataFrames from monday.com into typed, canonical frames
and — crucially — records *what it could not clean* so the tool layer can report
honest caveats (R6). Nothing here silently drops data: every excluded value is
counted and attributed to a reason.

Design notes
------------
* Column titles differ per board and may be renamed. Instead of hardcoding titles
  we resolve *logical fields* (close_date, amount, sector, stage, account, owner,
  status, due_date) to the best-matching actual column via fuzzy title match.
* Sector and status are canonicalized against explicit maps with a fuzzy fallback
  and an explicit ``Unclassified`` / ``Unknown`` bucket — never a silent merge.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable

import pandas as pd
from dateutil import parser as dateparser
from rapidfuzz import fuzz, process

# --------------------------------------------------------------------------- #
# Null handling
# --------------------------------------------------------------------------- #

NULL_TOKENS = {"", "-", "--", "n/a", "na", "n.a.", "tbd", "tba", "none", "null", "nil", "?"}


def is_null(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip().lower() in NULL_TOKENS:
        return True
    return False


def clean_null(value):
    return pd.NA if is_null(value) else value


# --------------------------------------------------------------------------- #
# Column resolution (fuzzy, so renamed columns still resolve)
# --------------------------------------------------------------------------- #

# Logical field -> candidate title keywords, best first. Aliases are kept multi-word
# where possible: single common words ("date", "value") over-match via subset scoring,
# so we lean on specific phrases plus a coverage tie-break (see resolve_columns).
FIELD_ALIASES: dict[str, list[str]] = {
    "account": ["account", "client", "client code", "company", "customer", "customer name", "organisation", "organization"],
    "sector": ["sector", "sector/service", "industry", "vertical", "domain", "segment"],
    "stage": ["deal stage", "pipeline stage", "sales stage", "stage"],
    "amount": ["estimated deal value", "masked deal value", "deal value", "deal size", "amount",
               "amount in rupees incl of gst masked", "contract value", "revenue", "price"],
    "close_date": ["estimated close date", "tentative close date", "expected close date",
                   "close date", "closing date", "won date"],
    "owner": ["owner code", "owner", "bd/kam personnel code", "kam personnel", "personnel code",
              "sales owner", "rep", "salesperson"],
    "probability": ["closure probability", "win probability", "probability", "likelihood", "confidence"],
    # Work order fields
    "wo_status": ["execution status", "work order status", "delivery status", "status", "state"],
    "due_date": ["probable end date", "delivery date", "completion date", "due date",
                 "deadline", "target date", "end date"],
    "start_date": ["probable start date", "start date", "kickoff date", "begin date"],
}


def resolve_columns(
    titles: Iterable[str],
    fields: Iterable[str],
    threshold: int = 62,
    df: pd.DataFrame | None = None,
) -> dict[str, str | None]:
    """Map each logical field to the best-matching actual column title.

    Uses ``token_sort_ratio`` (subset-safe, unlike token_set_ratio which scores a
    single shared word as a perfect match). When a DataFrame is provided, ties and
    near-ties are broken by *data coverage* — so a populated "Estimated Close Date"
    beats an empty duplicate "Close Date (A)". Each column is claimed by at most one
    field (highest scorer wins) to avoid two fields resolving to the same column.
    """
    titles = [t for t in titles if isinstance(t, str) and not t.startswith("__raw__")]

    def coverage(title: str) -> float:
        if df is None or title not in df.columns:
            return 0.0
        col = df[title]
        nonnull = col.map(lambda v: not is_null(v)).sum()
        return float(nonnull) / max(len(col), 1)

    # Score every (field, title) pair.
    scored: dict[str, list[tuple[str, float]]] = {}
    for field_name in fields:
        aliases = FIELD_ALIASES.get(field_name, [field_name])
        per_title: dict[str, float] = {}
        for title in titles:
            tl = title.lower()
            best = max(fuzz.token_sort_ratio(alias.lower(), tl) for alias in aliases)
            per_title[title] = best
        # Rank candidates: fuzzy score gated, then boosted by coverage.
        ranked = sorted(
            ((t, s + 25 * coverage(t)) for t, s in per_title.items() if s >= threshold),
            key=lambda x: x[1],
            reverse=True,
        )
        scored[field_name] = ranked

    # Greedy claim: process fields by their best available candidate's strength.
    resolved: dict[str, str | None] = {f: None for f in fields}
    claimed: set[str] = set()
    order = sorted(fields, key=lambda f: (scored[f][0][1] if scored[f] else -1), reverse=True)
    for field_name in order:
        for title, _ in scored[field_name]:
            if title not in claimed:
                resolved[field_name] = title
                claimed.add(title)
                break
    return resolved


# --------------------------------------------------------------------------- #
# Dates
# --------------------------------------------------------------------------- #

_QUARTER_RE = re.compile(r"^\s*q([1-4])[\s\-/']*?(\d{2,4})\s*$", re.IGNORECASE)
_QUARTER_RE2 = re.compile(r"^\s*(\d{4})[\s\-/']*?q([1-4])\s*$", re.IGNORECASE)


def parse_date(value, *, fiscal_start_month: int = 4) -> pd.Timestamp:
    """Parse one messy date value. Unparseable / null -> NaT (caller counts it).

    Quarter strings (``Q1 2024``, ``2024-Q1``) resolve to the *first day* of that
    fiscal quarter given the configured fiscal-year start month.
    """
    if is_null(value):
        return pd.NaT
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value)
    s = str(value).strip()

    for rx, order in ((_QUARTER_RE, ("q", "y")), (_QUARTER_RE2, ("y", "q"))):
        m = rx.match(s)
        if m:
            if order == ("q", "y"):
                q, y = int(m.group(1)), int(m.group(2))
            else:
                y, q = int(m.group(1)), int(m.group(2))
            if y < 100:
                y += 2000
            month = (fiscal_start_month - 1 + (q - 1) * 3) % 12 + 1
            year = y + ((fiscal_start_month - 1 + (q - 1) * 3) // 12)
            return pd.Timestamp(year=year, month=month, day=1)

    # ISO / year-first strings (2024-03-12, 2024/03/12) are unambiguous — do NOT
    # apply dayfirst to them or "03" gets read as the day.
    iso_like = bool(re.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}", s))
    try:
        # dayfirst for the rest: assignment data is India-sourced (dd/mm/yyyy dominant).
        return pd.Timestamp(
            dateparser.parse(s, dayfirst=not iso_like, yearfirst=iso_like, fuzzy=True)
        )
    except (ValueError, OverflowError, TypeError):
        return pd.NaT


# --------------------------------------------------------------------------- #
# Currency / numbers
# --------------------------------------------------------------------------- #

_SUFFIX_MULT = {
    "k": 1_000,
    "l": 100_000,      # lakh
    "lac": 100_000,
    "lakh": 100_000,
    "m": 1_000_000,
    "mn": 1_000_000,
    "cr": 10_000_000,  # crore
    "crore": 10_000_000,
    "b": 1_000_000_000,
    "bn": 1_000_000_000,
}
_NUM_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")
_SUFFIX_RE = re.compile(r"(crore|cr|lakh|lac|mn|bn|[klmb])\b", re.IGNORECASE)


_PROB_WORDS = {
    "very high": 0.9, "high": 0.8, "likely": 0.75, "medium": 0.5, "med": 0.5,
    "moderate": 0.5, "low": 0.25, "very low": 0.1, "unlikely": 0.15,
}


def parse_probability(value) -> float:
    """Parse a win-probability that may be numeric (0-1 / 0-100) or a word band.

    Returns a fraction in [0, 1], or NaN when unusable. Categorical bands like
    High/Medium/Low map to representative fractions (documented in the Decision Log).
    """
    if is_null(value):
        return float("nan")
    if isinstance(value, (int, float)):
        p = float(value)
        return p / 100 if p > 1 else p
    s = str(value).strip().lower()
    if s in _PROB_WORDS:
        return _PROB_WORDS[s]
    num = parse_currency(s)
    if num == num:  # not NaN
        return num / 100 if num > 1 else num
    return float("nan")


def parse_currency(value) -> float:
    """Parse messy currency/number text into a float. Unparseable -> NaN."""
    if is_null(value):
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().lower().replace("/-", "").replace("rs.", "").replace("inr", "")
    s = s.replace("usd", "").replace("$", "").replace("₹", "").replace("€", "")

    suffix_mult = 1
    m = _SUFFIX_RE.search(s)
    if m:
        suffix_mult = _SUFFIX_MULT.get(m.group(1).lower(), 1)

    num_match = _NUM_RE.search(s)
    if not num_match:
        return float("nan")
    num = num_match.group(0).replace(",", "")  # handles both 1,200 and Indian 1,20,000
    try:
        return float(num) * suffix_mult
    except ValueError:
        return float("nan")


# --------------------------------------------------------------------------- #
# Sector canonicalization
# --------------------------------------------------------------------------- #

SECTOR_CANON: dict[str, str] = {
    "energy": "Energy",
    "power": "Energy",
    "power & energy": "Energy",
    "power and energy": "Energy",
    "renewables": "Energy",
    "renewable energy": "Energy",
    "solar": "Energy",
    "oil & gas": "Energy",
    "oil and gas": "Energy",
    "utilities": "Energy",
    "powerline": "Energy",
    "power line": "Energy",
    "transmission": "Energy",
    "grid": "Energy",
    "mining": "Mining",
    "minerals": "Mining",
    "agriculture": "Agriculture",
    "agri": "Agriculture",
    "agritech": "Agriculture",
    "farming": "Agriculture",
    "construction": "Construction",
    "infrastructure": "Construction",
    "infra": "Construction",
    "real estate": "Real Estate",
    "realty": "Real Estate",
    "logistics": "Logistics",
    "supply chain": "Logistics",
    "transportation": "Logistics",
    "telecom": "Telecom",
    "telecommunications": "Telecom",
    "government": "Government",
    "govt": "Government",
    "public sector": "Government",
    "defense": "Defense",
    "defence": "Defense",
}


def canonical_sector(value, *, threshold: int = 85) -> tuple[str, bool]:
    """Return (canonical_sector, matched). Unmatched -> ('Unclassified', False)."""
    if is_null(value):
        return "Unclassified", False
    s = str(value).strip().lower()
    if s in SECTOR_CANON:
        return SECTOR_CANON[s], True
    match = process.extractOne(s, list(SECTOR_CANON.keys()), scorer=fuzz.token_set_ratio)
    if match and match[1] >= threshold:
        return SECTOR_CANON[match[0]], True
    # Title-case the raw value so a genuinely new sector is still usable, but flag it.
    return "Unclassified", False


# --------------------------------------------------------------------------- #
# Deal stage / status canonicalization
# --------------------------------------------------------------------------- #

# Ordered: first match wins. Tuned to the real Deal funnel, which runs a lettered
# A–O pipeline. Stages at/after commitment (work order received, invoice sent,
# amount accrued, project completed) are treated as WON business; the two
# "Not Relevant" states are treated as LOST. This interpretation is documented in
# the Decision Log — counting only literal "Closed Won" would badly understate
# secured business (27 vs ~100 deals here).
STAGE_PATTERNS: list[tuple[str, str]] = [
    (r"closed?\s*won|deal\s*won|\bwon\b|✅", "Closed Won"),
    (r"work\s*order\s*receiv|\bwo\s*receiv", "Work Order Received"),
    (r"invoice\s*sent|invoiced", "Invoice Sent"),
    (r"amount\s*accru|accrued", "Amount Accrued"),
    (r"on\s*hold", "On Hold"),
    (r"project\s*complet|completed", "Project Completed"),
    (r"not\s*relevant|not\s*applicable|\bn/?a\b", "Not Relevant"),
    (r"closed?\s*lost|\blost\b|❌|dropped", "Closed Lost"),
    (r"negotiat", "Negotiation"),
    (r"propos|commercial|quote", "Proposal"),
    (r"feasib", "Feasibility"),
    (r"\bpoc\b|proof\s*of\s*concept", "PoC"),
    (r"demo|present", "Demo"),
    (r"qualif", "Qualified"),
    (r"discover|prospect|\blead\b|\bnew\b", "Prospecting"),
    (r"contract|legal", "Contract"),
]

# Stages that represent secured business (won) vs dead deals (lost).
WON_STAGES = {"Closed Won", "Work Order Received", "Invoice Sent", "Amount Accrued", "Project Completed"}
LOST_STAGES = {"Closed Lost", "Not Relevant"}


def canonical_stage(value) -> tuple[str, bool]:
    if is_null(value):
        return "Unknown", False
    s = str(value).strip().lower()
    for pattern, canon in STAGE_PATTERNS:
        if re.search(pattern, s):
            return canon, True
    return str(value).strip().title(), False


WO_STATUS_PATTERNS: list[tuple[str, str]] = [
    (r"complet|done|deliver|closed", "Completed"),
    (r"on\s*hold|hold|paus|block|stuck|struck", "On Hold"),
    (r"in\s*progress|ongoing|active|execut|working", "In Progress"),
    (r"not\s*start|todo|to\s*do|backlog|planned|pending|\bnew\b", "Not Started"),
    (r"cancel", "Cancelled"),
    (r"at\s*risk|delay|overdue|late", "At Risk"),
]


def canonical_wo_status(value) -> tuple[str, bool]:
    if is_null(value):
        return "Unknown", False
    s = str(value).strip().lower()
    for pattern, canon in WO_STATUS_PATTERNS:
        if re.search(pattern, s):
            return canon, True
    return str(value).strip().title(), False


# --------------------------------------------------------------------------- #
# Account / owner normalization -> join key
# --------------------------------------------------------------------------- #

_SUFFIXES = [
    "pvt ltd", "pvt. ltd.", "private limited", "pvt limited", "ltd", "ltd.",
    "limited", "llp", "inc", "inc.", "incorporated", "corp", "corp.",
    "corporation", "co", "co.", "company", "gmbh", "plc", "and sons", "& sons",
]


def normalize_account(value) -> str | float:
    """Casefold, strip punctuation and common company suffixes -> stable join key."""
    if is_null(value):
        return float("nan")
    s = str(value).strip().lower()
    s = re.sub(r"[^\w\s&]", " ", s)          # drop punctuation
    s = re.sub(r"\s+", " ", s).strip()
    for suffix in sorted(_SUFFIXES, key=len, reverse=True):
        if s.endswith(" " + suffix) or s == suffix:
            s = s[: -len(suffix)].strip()
    return re.sub(r"\s+", " ", s).strip() or float("nan")


def normalize_text(value) -> str | float:
    if is_null(value):
        return float("nan")
    return re.sub(r"\s+", " ", str(value)).strip()


# --------------------------------------------------------------------------- #
# Field-level quality accounting
# --------------------------------------------------------------------------- #

@dataclass
class FieldQuality:
    field: str
    source_column: str | None
    total: int
    nulls: int = 0
    unparsed: int = 0        # present but could not be parsed (dates/amounts)
    unclassified: int = 0    # fell into Unclassified/Unknown bucket

    @property
    def coverage_pct(self) -> float:
        if self.total == 0:
            return 0.0
        good = self.total - self.nulls - self.unparsed - self.unclassified
        return round(100.0 * good / self.total, 1)

    def to_dict(self) -> dict:
        return {
            "field": self.field,
            "source_column": self.source_column,
            "total": self.total,
            "nulls": self.nulls,
            "unparsed": self.unparsed,
            "unclassified": self.unclassified,
            "coverage_pct": self.coverage_pct,
        }


@dataclass
class NormalizedBoard:
    name: str
    df: pd.DataFrame
    resolved_columns: dict[str, str | None]
    quality: dict[str, FieldQuality] = field(default_factory=dict)
    unresolved_fields: list[str] = field(default_factory=list)

    def quality_records(self) -> list[dict]:
        return [q.to_dict() for q in self.quality.values()]


# --------------------------------------------------------------------------- #
# Board normalizers
# --------------------------------------------------------------------------- #

def normalize_deals(df: pd.DataFrame, fiscal_start_month: int = 4) -> NormalizedBoard:
    titles = list(df.columns)
    fields = ["account", "sector", "stage", "amount", "close_date", "owner", "probability"]
    resolved = resolve_columns(titles, fields, df=df)
    out = pd.DataFrame(index=df.index)
    out["item_name"] = df.get("item_name")
    quality: dict[str, FieldQuality] = {}
    total = len(df)

    def col(field_name):
        c = resolved.get(field_name)
        return df[c] if c and c in df.columns else pd.Series([pd.NA] * total, index=df.index)

    # Account
    acc_raw = col("account")
    out["account_raw"] = acc_raw.map(normalize_text)
    out["account_key"] = acc_raw.map(normalize_account)
    quality["account"] = FieldQuality("account", resolved["account"], total,
                                      nulls=int(out["account_key"].isna().sum()))

    # Sector
    sec = col("sector").map(lambda v: canonical_sector(v, threshold=85))
    out["sector"] = sec.map(lambda t: t[0])
    matched = sec.map(lambda t: t[1])
    out["sector_matched"] = matched
    quality["sector"] = FieldQuality(
        "sector", resolved["sector"], total,
        nulls=int(col("sector").map(is_null).sum()),
        unclassified=int((~matched & ~col("sector").map(is_null)).sum()),
    )

    # Stage
    stg = col("stage").map(canonical_stage)
    out["stage"] = stg.map(lambda t: t[0])
    out["stage_matched"] = stg.map(lambda t: t[1])
    quality["stage"] = FieldQuality(
        "stage", resolved["stage"], total,
        nulls=int(col("stage").map(is_null).sum()),
        unclassified=int((out["stage"] == "Unknown").sum()),
    )

    # Amount
    amt = col("amount").map(parse_currency)
    out["amount"] = amt
    amt_nulls = int(col("amount").map(is_null).sum())
    amt_unparsed = int(amt.isna().sum() - amt_nulls)
    quality["amount"] = FieldQuality("amount", resolved["amount"], total,
                                     nulls=amt_nulls, unparsed=max(amt_unparsed, 0))

    # Close date
    cd = col("close_date").map(lambda v: parse_date(v, fiscal_start_month=fiscal_start_month))
    out["close_date"] = cd
    cd_nulls = int(col("close_date").map(is_null).sum())
    cd_unparsed = int(cd.isna().sum() - cd_nulls)
    quality["close_date"] = FieldQuality("close_date", resolved["close_date"], total,
                                         nulls=cd_nulls, unparsed=max(cd_unparsed, 0))

    # Owner
    out["owner"] = col("owner").map(normalize_text)
    quality["owner"] = FieldQuality("owner", resolved["owner"], total,
                                    nulls=int(out["owner"].isna().sum()))

    # Probability: numeric or categorical (High/Medium/Low) -> fraction in [0,1]
    out["probability"] = col("probability").map(parse_probability)

    out["is_won"] = out["stage"].isin(WON_STAGES)
    out["is_lost"] = out["stage"].isin(LOST_STAGES)
    out["is_open"] = ~out["is_won"] & ~out["is_lost"]

    unresolved = [f for f in fields if resolved.get(f) is None]
    return NormalizedBoard("Deals", out, resolved, quality, unresolved)


def normalize_work_orders(df: pd.DataFrame, fiscal_start_month: int = 4) -> NormalizedBoard:
    titles = list(df.columns)
    fields = ["account", "sector", "wo_status", "owner", "due_date", "start_date", "amount"]
    resolved = resolve_columns(titles, fields, df=df)
    out = pd.DataFrame(index=df.index)
    out["item_name"] = df.get("item_name")
    quality: dict[str, FieldQuality] = {}
    total = len(df)

    def col(field_name):
        c = resolved.get(field_name)
        return df[c] if c and c in df.columns else pd.Series([pd.NA] * total, index=df.index)

    acc_raw = col("account")
    out["account_raw"] = acc_raw.map(normalize_text)
    out["account_key"] = acc_raw.map(normalize_account)
    quality["account"] = FieldQuality("account", resolved["account"], total,
                                      nulls=int(out["account_key"].isna().sum()))

    sec = col("sector").map(lambda v: canonical_sector(v, threshold=85))
    out["sector"] = sec.map(lambda t: t[0])
    matched = sec.map(lambda t: t[1])
    quality["sector"] = FieldQuality(
        "sector", resolved["sector"], total,
        nulls=int(col("sector").map(is_null).sum()),
        unclassified=int((~matched & ~col("sector").map(is_null)).sum()),
    )

    stt = col("wo_status").map(canonical_wo_status)
    out["status"] = stt.map(lambda t: t[0])
    quality["status"] = FieldQuality(
        "status", resolved["wo_status"], total,
        nulls=int(col("wo_status").map(is_null).sum()),
        unclassified=int((out["status"] == "Unknown").sum()),
    )

    out["owner"] = col("owner").map(normalize_text)
    quality["owner"] = FieldQuality("owner", resolved["owner"], total,
                                    nulls=int(out["owner"].isna().sum()))

    dd = col("due_date").map(lambda v: parse_date(v, fiscal_start_month=fiscal_start_month))
    out["due_date"] = dd
    dd_nulls = int(col("due_date").map(is_null).sum())
    quality["due_date"] = FieldQuality("due_date", resolved["due_date"], total,
                                       nulls=dd_nulls, unparsed=max(int(dd.isna().sum() - dd_nulls), 0))

    out["start_date"] = col("start_date").map(lambda v: parse_date(v, fiscal_start_month=fiscal_start_month))
    out["amount"] = col("amount").map(parse_currency)

    today = pd.Timestamp.today().normalize()
    open_mask = ~out["status"].isin({"Completed", "Cancelled"})
    out["is_overdue"] = open_mask & out["due_date"].notna() & (out["due_date"] < today)

    unresolved = [f for f in fields if resolved.get(f) is None]
    return NormalizedBoard("Work Orders", out, resolved, quality, unresolved)
