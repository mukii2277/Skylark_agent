"""Live smoke test — run after secrets are configured.

Verifies the monday.com connection, column resolution, and data-quality accounting
against the REAL boards, and prints the distinct stage/status/sector values so win
and loss detection can be tuned to the actual data.

    python smoke_test.py
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTHONUTF8", "1")

import pandas as pd

from bi_tools import (
    BIContext,
    cross_board_view,
    get_data_quality_report,
    get_pipeline_summary,
    get_sector_performance,
    get_work_order_status,
)

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)


def hr(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def main():
    hr("Loading + normalizing both boards")
    ctx = BIContext.build(force=True)
    for w in ctx.warnings:
        print("WARN:", w)

    for name, nb in {"Deals": ctx.deals, "Work Orders": ctx.work_orders}.items():
        hr(f"{name}: {len(nb.df)} rows")
        print("Resolved columns:")
        for k, v in nb.resolved_columns.items():
            print(f"  {k:12s} -> {v}")
        if nb.unresolved_fields:
            print("UNRESOLVED:", nb.unresolved_fields)
        print("\nField quality:")
        for q in nb.quality_records():
            print(f"  {q['field']:12s} coverage={q['coverage_pct']:5.1f}%  "
                  f"null={q['nulls']} unparsed={q['unparsed']} unclassified={q['unclassified']}")

    hr("Deals: distinct STAGE values (canonical <- how many)")
    print(ctx.deals.df["stage"].value_counts(dropna=False).to_string())
    hr("Deals: is_won / is_lost / is_open counts")
    print("won:", int(ctx.deals.df["is_won"].sum()),
          "lost:", int(ctx.deals.df["is_lost"].sum()),
          "open:", int(ctx.deals.df["is_open"].sum()))
    hr("Deals: distinct SECTOR values")
    print(ctx.deals.df["sector"].value_counts(dropna=False).to_string())

    hr("Work Orders: distinct STATUS values")
    print(ctx.work_orders.df["status"].value_counts(dropna=False).to_string())

    hr("TOOL: pipeline summary (energy, this quarter)")
    import json
    print(json.dumps(get_pipeline_summary(ctx, sector="energy", period="this quarter"), indent=2, default=str))

    hr("TOOL: sector performance (all time)")
    print(json.dumps(get_sector_performance(ctx, metric="pipeline_value"), indent=2, default=str))

    hr("TOOL: work order status")
    print(json.dumps(get_work_order_status(ctx), indent=2, default=str))

    hr("TOOL: cross board view (first accounts)")
    cbv = cross_board_view(ctx)
    print("accounts matched:", cbv["caveats"]["accounts_matched"])
    print(json.dumps(cbv["result"]["accounts"][:3], indent=2, default=str))
    print("caveats:", json.dumps(cbv["caveats"], indent=2, default=str))


if __name__ == "__main__":
    main()
