"""Deterministic BI tool layer (R7, R9, R10, R13).

Every public tool:
  * takes plain-JSON arguments the LLM can bind,
  * computes with pandas — **never** relies on the LLM for arithmetic,
  * returns ``{"result": ..., "caveats": {...}}`` so the agent can narrate honest
    numbers and state exactly what was excluded and why (R4, R6).

The LLM selects tools and reads back these numbers; it does not recompute them.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

from config import Settings, get_settings
from monday_client import Boards, MondayClient, MondayError, load_boards
from normalize import (
    NormalizedBoard,
    canonical_sector,
    normalize_deals,
    normalize_work_orders,
)
from periods import Period, resolve_period


# --------------------------------------------------------------------------- #
# Context: load + normalize both boards once per request
# --------------------------------------------------------------------------- #

@dataclass
class BIContext:
    deals: NormalizedBoard
    work_orders: NormalizedBoard
    settings: Settings
    warnings: list[str]

    @classmethod
    def build(cls, client: MondayClient | None = None, *, force: bool = False) -> "BIContext":
        settings = get_settings()
        boards: Boards = load_boards(client, force=force)
        deals = normalize_deals(boards.deals.df, settings.fiscal_year_start_month)
        work_orders = normalize_work_orders(boards.work_orders.df, settings.fiscal_year_start_month)
        return cls(deals, work_orders, settings, list(boards.warnings))


def _money(x) -> float:
    try:
        return round(float(x), 2)
    except (TypeError, ValueError):
        return 0.0


def _period_caveat(period: Period | None) -> dict[str, Any]:
    if period is None:
        return {"period": "all time", "period_assumed": False}
    return {
        "period": period.label,
        "period_start": period.start.date().isoformat(),
        "period_end": (period.end - pd.Timedelta(days=1)).date().isoformat(),
        "period_assumed": period.assumed,
    }


def _filter_period(df: pd.DataFrame, date_col: str, period: Period | None):
    """Return (in_period_df, n_missing_date, n_total_considered)."""
    total = len(df)
    if period is None:
        return df, int(df[date_col].isna().sum()) if date_col in df else 0, total
    has_date = df[df[date_col].notna()]
    missing = total - len(has_date)
    mask = has_date[date_col].apply(period.contains)
    return has_date[mask], missing, total


def _resolve_sector_filter(sector: str | None) -> tuple[str | None, bool]:
    if not sector:
        return None, True
    canon, matched = canonical_sector(sector)
    return canon, matched


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #

def get_pipeline_summary(
    ctx: BIContext,
    sector: str | None = None,
    period: str | None = None,
    stage: str | None = None,
) -> dict[str, Any]:
    df = ctx.deals.df.copy()
    fsm = ctx.settings.fiscal_year_start_month
    per = resolve_period(period, fiscal_start_month=fsm)

    sector_canon, sector_matched = _resolve_sector_filter(sector)
    caveats: dict[str, Any] = {"filters": {}, "excluded": {}}
    caveats.update(_period_caveat(per))

    if sector:
        caveats["filters"]["sector"] = sector_canon
        caveats["filters"]["sector_input_matched"] = sector_matched
        df = df[df["sector"] == sector_canon]
    if stage:
        from normalize import canonical_stage
        stage_canon, _ = canonical_stage(stage)
        caveats["filters"]["stage"] = stage_canon
        df = df[df["stage"] == stage_canon]

    considered = len(df)
    # Pipeline = open deals; period applies to expected close date.
    open_df = df[df["is_open"]]
    in_period, missing_date, _ = _filter_period(open_df, "close_date", per)

    by_stage = (
        in_period.groupby("stage")["amount"]
        .agg(value="sum", count="count")
        .reset_index()
        .sort_values("value", ascending=False)
    )
    stage_rows = [
        {"stage": r["stage"], "value": _money(r["value"]), "count": int(r["count"])}
        for _, r in by_stage.iterrows()
    ]

    # Weighted forecast: value * probability when probability present, else raw.
    wf = in_period.copy()
    prob = wf["probability"].fillna(0)
    # If probabilities look like percents (>1), scale to fraction.
    prob = prob.apply(lambda p: p / 100 if p > 1 else p)
    has_prob = wf["probability"].notna().sum()
    weighted = float((wf["amount"].fillna(0) * prob).sum()) if has_prob else None

    amount_nulls = int(in_period["amount"].isna().sum())
    caveats["excluded"] = {
        "open_deals_missing_close_date": int(missing_date),
        "deals_missing_amount": amount_nulls,
    }
    caveats["rows_considered"] = int(considered)
    caveats["open_deals_in_period"] = int(len(in_period))
    if len(in_period) == 0:
        caveats["weighted_forecast_basis"] = "no open deals fall in the selected period"
    elif has_prob:
        caveats["weighted_forecast_basis"] = (
            f"weighted = Σ(value × win-probability); {int(has_prob)} of {len(in_period)} "
            "open deals had a probability, the rest contributed 0 to the weighting"
        )
    else:
        caveats["weighted_forecast_basis"] = (
            "no win-probability values on the deals in scope; weighted forecast omitted"
        )

    return {
        "result": {
            "total_open_pipeline_value": _money(in_period["amount"].sum()),
            "open_deal_count": int(len(in_period)),
            "weighted_forecast": _money(weighted) if weighted is not None else None,
            "by_stage": stage_rows,
        },
        "caveats": caveats,
    }


def get_sector_performance(
    ctx: BIContext,
    period: str | None = None,
    metric: str = "pipeline_value",
    top_n: int | None = None,
) -> dict[str, Any]:
    df = ctx.deals.df.copy()
    fsm = ctx.settings.fiscal_year_start_month
    per = resolve_period(period, fiscal_start_month=fsm)
    caveats: dict[str, Any] = {"metric": metric, "excluded": {}}
    caveats.update(_period_caveat(per))

    in_period, missing_date, total = _filter_period(df, "close_date", per)
    if per is None:
        in_period = df  # no date filter for all-time
        missing_date = 0

    rows = []
    for sector, g in in_period.groupby("sector"):
        won = g[g["is_won"]]
        closed = g[g["is_won"] | g["is_lost"]]
        win_rate = round(100 * len(won) / len(closed), 1) if len(closed) else None
        open_g = g[g["is_open"]]
        rows.append({
            "sector": sector,
            "pipeline_value": _money(open_g["amount"].sum()),
            "closed_won_value": _money(won["amount"].sum()),
            "deal_count": int(len(g)),
            "won_count": int(len(won)),
            "win_rate_pct": win_rate,
            "avg_deal_size": _money(g["amount"].mean()) if g["amount"].notna().any() else None,
        })

    sort_key = metric if metric in {"pipeline_value", "closed_won_value", "deal_count", "avg_deal_size"} else "pipeline_value"
    rows.sort(key=lambda r: (r[sort_key] is None, r[sort_key] or 0), reverse=True)
    if top_n:
        rows = rows[:top_n]

    unclassified = int((in_period["sector"] == "Unclassified").sum())
    caveats["excluded"] = {
        "deals_missing_close_date": int(missing_date),
        "deals_in_unclassified_sector": unclassified,
    }
    caveats["rows_considered"] = int(len(in_period))
    return {"result": {"sectors": rows}, "caveats": caveats}


def get_revenue_summary(
    ctx: BIContext,
    period: str | None = None,
    granularity: str = "total",
    sector: str | None = None,
) -> dict[str, Any]:
    df = ctx.deals.df.copy()
    fsm = ctx.settings.fiscal_year_start_month
    per = resolve_period(period, fiscal_start_month=fsm)
    caveats: dict[str, Any] = {"revenue_definition": "closed-won deal value by close date", "excluded": {}}
    caveats.update(_period_caveat(per))

    sector_canon, sector_matched = _resolve_sector_filter(sector)
    if sector:
        caveats["filters"] = {"sector": sector_canon, "sector_input_matched": sector_matched}
        df = df[df["sector"] == sector_canon]

    won = df[df["is_won"]]
    in_period, missing_date, _ = _filter_period(won, "close_date", per)

    # Prior-period comparison (same length, immediately preceding).
    prior_value = None
    if per is not None:
        span = per.end - per.start
        prior = Period(per.start - span, per.start, "prior period")
        prior_df, _, _ = _filter_period(won, "close_date", prior)
        prior_value = _money(prior_df["amount"].sum())

    current_value = _money(in_period["amount"].sum())
    trend_pct = None
    if prior_value:
        trend_pct = round(100 * (current_value - prior_value) / prior_value, 1)

    breakdown = None
    if granularity in {"month", "monthly", "quarter", "quarterly"} and len(in_period):
        freq = "M" if granularity.startswith("month") else "Q"
        s = in_period.set_index("close_date")["amount"].resample(freq).sum()
        breakdown = [{"period": str(idx.date()), "value": _money(v)} for idx, v in s.items()]

    caveats["excluded"] = {
        "won_deals_missing_close_date": int(missing_date),
        "won_deals_missing_amount": int(in_period["amount"].isna().sum()),
    }
    caveats["rows_considered"] = int(len(in_period))
    return {
        "result": {
            "closed_won_value": current_value,
            "closed_won_count": int(len(in_period)),
            "prior_period_value": prior_value,
            "trend_vs_prior_pct": trend_pct,
            "breakdown": breakdown,
        },
        "caveats": caveats,
    }


def get_work_order_status(
    ctx: BIContext,
    sector: str | None = None,
    status: str | None = None,
    owner: str | None = None,
) -> dict[str, Any]:
    df = ctx.work_orders.df.copy()
    caveats: dict[str, Any] = {"filters": {}, "excluded": {}}

    if sector:
        sector_canon, matched = _resolve_sector_filter(sector)
        caveats["filters"]["sector"] = sector_canon
        caveats["filters"]["sector_input_matched"] = matched
        df = df[df["sector"] == sector_canon]
    if status:
        from normalize import canonical_wo_status
        status_canon, _ = canonical_wo_status(status)
        caveats["filters"]["status"] = status_canon
        df = df[df["status"] == status_canon]
    if owner:
        caveats["filters"]["owner"] = owner
        df = df[df["owner"].fillna("").str.contains(owner, case=False, na=False)]

    by_status = (
        df.groupby("status")["item_name"].count().reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    status_rows = [{"status": r["status"], "count": int(r["count"])} for _, r in by_status.iterrows()]

    overdue = df[df["is_overdue"]]
    overdue_rows = [
        {
            "work_order": r["item_name"],
            "account": r["account_raw"] if pd.notna(r["account_raw"]) else None,
            "status": r["status"],
            "due_date": r["due_date"].date().isoformat() if pd.notna(r["due_date"]) else None,
        }
        for _, r in overdue.sort_values("due_date").head(15).iterrows()
    ]

    caveats["excluded"] = {
        "work_orders_missing_due_date": int(df["due_date"].isna().sum()),
        "work_orders_unknown_status": int((df["status"] == "Unknown").sum()),
    }
    caveats["rows_considered"] = int(len(df))
    return {
        "result": {
            "total": int(len(df)),
            "by_status": status_rows,
            "overdue_count": int(len(overdue)),
            "overdue_items": overdue_rows,
        },
        "caveats": caveats,
    }


def cross_board_view(
    ctx: BIContext,
    account: str | None = None,
    sector: str | None = None,
) -> dict[str, Any]:
    """Join deals to work orders on the normalized account key (R10)."""
    deals = ctx.deals.df.copy()
    wos = ctx.work_orders.df.copy()
    caveats: dict[str, Any] = {"join_key": "normalized account", "excluded": {}}

    if sector:
        sector_canon, matched = _resolve_sector_filter(sector)
        caveats["filters"] = {"sector": sector_canon, "sector_input_matched": matched}
        deals = deals[deals["sector"] == sector_canon]
        wos = wos[wos["sector"] == sector_canon]
    if account:
        from normalize import normalize_account
        key = normalize_account(account)
        deals = deals[deals["account_key"] == key]
        wos = wos[wos["account_key"] == key]
        caveats.setdefault("filters", {})["account"] = account

    deals_no_key = int(deals["account_key"].isna().sum())
    wos_no_key = int(wos["account_key"].isna().sum())

    rows = []
    accounts = set(deals["account_key"].dropna()) | set(wos["account_key"].dropna())
    for key in accounts:
        d = deals[deals["account_key"] == key]
        w = wos[wos["account_key"] == key]
        name = None
        if len(d):
            name = d["account_raw"].dropna().iloc[0] if d["account_raw"].notna().any() else None
        if not name and len(w):
            name = w["account_raw"].dropna().iloc[0] if w["account_raw"].notna().any() else None
        rows.append({
            "account": name or key,
            "sector": (d["sector"].mode().iloc[0] if len(d) and d["sector"].notna().any()
                       else (w["sector"].mode().iloc[0] if len(w) else None)),
            "deal_count": int(len(d)),
            "won_value": _money(d[d["is_won"]]["amount"].sum()),
            "open_pipeline": _money(d[d["is_open"]]["amount"].sum()),
            "work_order_count": int(len(w)),
            "open_work_orders": int((~w["status"].isin(["Completed", "Cancelled"])).sum()) if len(w) else 0,
            "overdue_work_orders": int(w["is_overdue"].sum()) if len(w) else 0,
        })
    rows.sort(key=lambda r: r["won_value"] + r["open_pipeline"], reverse=True)

    deal_keys = set(deals["account_key"].dropna())
    wo_keys = set(wos["account_key"].dropna())
    both = deal_keys & wo_keys
    caveats["excluded"] = {
        "deals_without_account_key": deals_no_key,
        "work_orders_without_account_key": wos_no_key,
    }
    caveats["deal_accounts"] = len(deal_keys)
    caveats["work_order_accounts"] = len(wo_keys)
    caveats["accounts_in_both_boards"] = len(both)
    caveats["accounts_listed"] = int(len(rows))
    if not both:
        # Honest about the data: the two boards were anonymized independently
        # (Deals use COMPANY-NNN, Work Orders use WOCOMPANY_NNN with separate
        # numbering), so no account can be linked across them row-to-row.
        caveats["join_note"] = (
            "The Deals and Work Orders boards do not share a common account or deal "
            "key — they were anonymized independently, so a row-level join yields no "
            "matches. Figures below are per-board and are NOT linked across boards."
        )
    return {"result": {"accounts": rows}, "caveats": caveats}


def get_data_quality_report(ctx: BIContext, board: str | None = None) -> dict[str, Any]:
    boards = {"deals": ctx.deals, "work_orders": ctx.work_orders}
    if board in boards:
        selected = {board: boards[board]}
    else:
        selected = boards

    report = {}
    for name, nb in selected.items():
        report[name] = {
            "row_count": int(len(nb.df)),
            "resolved_columns": nb.resolved_columns,
            "unresolved_fields": nb.unresolved_fields,
            "field_quality": nb.quality_records(),
        }
    return {
        "result": report,
        "caveats": {
            "note": "coverage_pct = share of rows with a usable value for that field",
            "warnings": ctx.warnings,
        },
    }


def generate_leadership_brief(ctx: BIContext, period: str | None = None) -> dict[str, Any]:
    """Assemble a self-contained executive brief for a period (R13, §7)."""
    pipeline = get_pipeline_summary(ctx, period=period)
    sectors = get_sector_performance(ctx, period=period, metric="closed_won_value", top_n=5)
    revenue = get_revenue_summary(ctx, period=period)
    delivery = get_work_order_status(ctx)

    # Watch items: largest open deals in the period, sorted by amount.
    df = ctx.deals.df.copy()
    per = resolve_period(period, fiscal_start_month=ctx.settings.fiscal_year_start_month)
    open_df = df[df["is_open"]]
    in_period, _, _ = _filter_period(open_df, "close_date", per)
    watch = in_period.sort_values("amount", ascending=False).head(5)
    watch_items = [
        {
            "deal": r["item_name"],
            "sector": r["sector"],
            "stage": r["stage"],
            "amount": _money(r["amount"]),
        }
        for _, r in watch.iterrows() if pd.notna(r["amount"])
    ]

    return {
        "result": {
            "period": pipeline["caveats"].get("period"),
            "pipeline_snapshot": pipeline["result"],
            "revenue_movement": revenue["result"],
            "sector_performance": sectors["result"]["sectors"],
            "delivery_health": delivery["result"],
            "watch_items": watch_items,
        },
        "caveats": {
            "pipeline": pipeline["caveats"],
            "revenue": revenue["caveats"],
            "sectors": sectors["caveats"],
            "delivery": delivery["caveats"],
            "data_warnings": ctx.warnings,
        },
    }


# --------------------------------------------------------------------------- #
# Dispatch table used by the agent
# --------------------------------------------------------------------------- #

TOOL_FUNCTIONS = {
    "get_pipeline_summary": get_pipeline_summary,
    "get_sector_performance": get_sector_performance,
    "get_revenue_summary": get_revenue_summary,
    "get_work_order_status": get_work_order_status,
    "cross_board_view": cross_board_view,
    "get_data_quality_report": get_data_quality_report,
    "generate_leadership_brief": generate_leadership_brief,
}
