"""Fiscal-period resolution.

Turns relative/human period phrases ("this quarter", "Q1", "last month", "FY24")
into a concrete [start, end) date window, honouring a configurable fiscal-year
start month. The default is April (Indian fiscal convention) — documented in the
Decision Log. When a phrase is relative and the convention is ambiguous, the agent
is expected to clarify *before* calling a tool; this module still resolves a best
effort so a stated assumption can proceed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

import pandas as pd
from dateutil.relativedelta import relativedelta


@dataclass(frozen=True)
class Period:
    start: pd.Timestamp
    end: pd.Timestamp        # exclusive
    label: str
    assumed: bool = False    # True when we had to guess a relative reference

    def contains(self, ts: pd.Timestamp) -> bool:
        return self.start <= ts < self.end


def _fiscal_year_start(d: date, start_month: int) -> date:
    year = d.year if d.month >= start_month else d.year - 1
    return date(year, start_month, 1)


def _quarter_bounds(fy_start: date, q: int) -> tuple[date, date]:
    start = fy_start + relativedelta(months=3 * (q - 1))
    end = start + relativedelta(months=3)
    return start, end


def resolve_period(
    phrase: str | None,
    *,
    fiscal_start_month: int = 4,
    today: date | None = None,
) -> Period | None:
    """Resolve a period phrase to a window. Returns None if no period is implied."""
    if not phrase:
        return None
    today = today or date.today()
    s = phrase.strip().lower()

    def wrap(start: date, end: date, label: str, assumed: bool = False) -> Period:
        return Period(pd.Timestamp(start), pd.Timestamp(end), label, assumed)

    if s in {"all", "all time", "ever", "overall", "to date"}:
        return None

    # Explicit fiscal quarter: "q1 2024", "fy24 q1", "q1 fy2025"
    m = re.search(r"q([1-4])", s)
    year_m = re.search(r"(?:fy)?\s*'?(\d{4}|\d{2})", s)
    if m:
        q = int(m.group(1))
        if year_m:
            y = int(year_m.group(1))
            y = y + 2000 if y < 100 else y
        else:
            fy = _fiscal_year_start(today, fiscal_start_month)
            y = fy.year
        fy_start = date(y, fiscal_start_month, 1)
        start, end = _quarter_bounds(fy_start, q)
        return wrap(start, end, f"Q{q} FY{str(y)[-2:]}")

    # Fiscal year: "fy24", "fy2025", "this financial year"
    m = re.search(r"fy\s*'?(\d{4}|\d{2})", s)
    if m:
        y = int(m.group(1))
        y = y + 2000 if y < 100 else y
        start = date(y, fiscal_start_month, 1)
        end = start + relativedelta(years=1)
        return wrap(start, end, f"FY{str(y)[-2:]}")

    if "this quarter" in s or s in {"the quarter", "current quarter"}:
        fy_start = _fiscal_year_start(today, fiscal_start_month)
        months_in = (today.year - fy_start.year) * 12 + (today.month - fy_start.month)
        q = months_in // 3 + 1
        start, end = _quarter_bounds(fy_start, q)
        return wrap(start, end, f"Q{q} FY{str(fy_start.year)[-2:]} (this quarter)", assumed=True)

    if "last quarter" in s or "previous quarter" in s:
        fy_start = _fiscal_year_start(today, fiscal_start_month)
        months_in = (today.year - fy_start.year) * 12 + (today.month - fy_start.month)
        q = months_in // 3 + 1
        start, _ = _quarter_bounds(fy_start, q)
        lstart = start - relativedelta(months=3)
        lend = start
        return wrap(lstart, lend, "last quarter", assumed=True)

    if "this year" in s or "this financial year" in s or "this fiscal year" in s:
        start = _fiscal_year_start(today, fiscal_start_month)
        end = start + relativedelta(years=1)
        return wrap(start, end, f"FY{str(start.year)[-2:]}", assumed="financial" not in s and "fiscal" not in s)

    if "last year" in s or "previous year" in s:
        start = _fiscal_year_start(today, fiscal_start_month) - relativedelta(years=1)
        end = start + relativedelta(years=1)
        return wrap(start, end, f"FY{str(start.year)[-2:]}", assumed=True)

    if "this month" in s or s == "month":
        start = today.replace(day=1)
        end = start + relativedelta(months=1)
        return wrap(start, end, start.strftime("%b %Y"), assumed=True)

    if "last month" in s or "previous month" in s:
        start = today.replace(day=1) - relativedelta(months=1)
        end = start + relativedelta(months=1)
        return wrap(start, end, start.strftime("%b %Y"))

    if "ytd" in s or "year to date" in s:
        start = _fiscal_year_start(today, fiscal_start_month)
        end = pd.Timestamp(today) + relativedelta(days=1)
        return wrap(start, end.date(), "FYTD")

    # Bare calendar year: "2024"
    m = re.fullmatch(r"\s*(\d{4})\s*", s)
    if m:
        y = int(m.group(1))
        return wrap(date(y, 1, 1), date(y + 1, 1, 1), str(y))

    return None
