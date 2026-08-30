"""Unit checks for the normalization layer against realistic messy values.

Run directly (no pytest required):  python test_normalize.py
"""
from __future__ import annotations

import pandas as pd

from normalize import (
    canonical_sector,
    canonical_stage,
    canonical_wo_status,
    is_null,
    normalize_account,
    normalize_deals,
    parse_currency,
    parse_date,
    resolve_columns,
)
from periods import resolve_period

_failures: list[str] = []


def check(label, got, expected):
    ok = got == expected or (isinstance(expected, float) and pd.isna(expected) and pd.isna(got))
    print(("PASS" if ok else "FAIL"), f"{label}: got={got!r} expected={expected!r}")
    if not ok:
        _failures.append(label)


def approx(label, got, expected, tol=1e-6):
    ok = got is not None and abs(got - expected) <= tol
    print(("PASS" if ok else "FAIL"), f"{label}: got={got!r} expected≈{expected!r}")
    if not ok:
        _failures.append(label)


# ---- Nulls ----
for v in ["", "-", "N/A", "na", "TBD", None]:
    check(f"is_null({v!r})", is_null(v), True)
check("is_null('Energy')", is_null("Energy"), False)

# ---- Currency ----
approx("₹1,20,000", parse_currency("₹1,20,000"), 120000)
approx("$1.2M", parse_currency("$1.2M"), 1_200_000)
approx("12000 USD", parse_currency("12000 USD"), 12000)
approx("1,20,000/-", parse_currency("1,20,000/-"), 120000)
approx("2.5 Cr", parse_currency("2.5 Cr"), 25_000_000)
approx("5L", parse_currency("5L"), 500_000)
check("blank currency", pd.isna(parse_currency("")), True)

# ---- Dates ----
check("dd/mm/yyyy", parse_date("12/03/2024"), pd.Timestamp("2024-03-12"))
check("iso", parse_date("2024-03-12"), pd.Timestamp("2024-03-12"))
check("Mar 12 2024", parse_date("Mar 12 2024"), pd.Timestamp("2024-03-12"))
check("blank->NaT", pd.isna(parse_date("")), True)
check("garbage->NaT", pd.isna(parse_date("not a date")), True)
# Q1 with April fiscal start -> Apr 1
check("Q1 2024 (Apr FY)", parse_date("Q1 2024", fiscal_start_month=4), pd.Timestamp("2024-04-01"))

# ---- Sector ----
check("energy", canonical_sector("energy")[0], "Energy")
check("ENERGY", canonical_sector("ENERGY")[0], "Energy")
check("Power & Energy", canonical_sector("Power & Energy")[0], "Energy")
check("Renewables", canonical_sector("Renewables")[0], "Energy")
check("Gibberish->Unclassified", canonical_sector("Zxqw")[0], "Unclassified")

# ---- Stage ----
check("Closed Won", canonical_stage("Closed Won")[0], "Closed Won")
check("won emoji", canonical_stage("won ✅")[0], "Closed Won")
check("Negotiating", canonical_stage("Negotiating")[0], "Negotiation")

# ---- WO status ----
check("in progress", canonical_wo_status("In Progress")[0], "In Progress")
check("done", canonical_wo_status("done")[0], "Completed")

# ---- Account key ----
check("Pvt Ltd strip", normalize_account("Acme Pvt. Ltd."), "acme")
check("case+space", normalize_account("  ACME  "), "acme")
check("join equality", normalize_account("Acme Pvt Ltd") == normalize_account("acme"), True)

# ---- Column resolution ----
titles = ["Deal Name", "Client", "Industry", "Deal Value", "Expected Close", "Sales Owner", "Deal Stage"]
res = resolve_columns(titles, ["account", "sector", "amount", "close_date", "owner", "stage"])
check("resolve account", res["account"], "Client")
check("resolve sector", res["sector"], "Industry")
check("resolve amount", res["amount"], "Deal Value")
check("resolve stage", res["stage"], "Deal Stage")

# ---- Periods ----
p = resolve_period("Q1 FY25", fiscal_start_month=4)
check("Q1 FY25 start", p.start, pd.Timestamp("2025-04-01"))
check("Q1 FY25 end", p.end, pd.Timestamp("2025-07-01"))
p2 = resolve_period("2024", fiscal_start_month=4)
check("2024 cal start", p2.start, pd.Timestamp("2024-01-01"))

# ---- End-to-end normalize_deals on a messy frame ----
raw = pd.DataFrame([
    {"item_name": "D1", "Client": "Acme Pvt Ltd", "Industry": "energy",
     "Deal Value": "₹1,20,000", "Expected Close": "12/05/2024", "Deal Stage": "Closed Won", "Sales Owner": "Ravi"},
    {"item_name": "D2", "Client": "beta corp", "Industry": "Renewables",
     "Deal Value": "$1.2M", "Expected Close": "", "Deal Stage": "Negotiation", "Sales Owner": ""},
    {"item_name": "D3", "Client": "", "Industry": "Zxqw",
     "Deal Value": "N/A", "Expected Close": "not a date", "Deal Stage": "won ✅", "Sales Owner": "Sara"},
])
nb = normalize_deals(raw, fiscal_start_month=4)
check("deals rows", len(nb.df), 3)
check("D1 sector", nb.df.loc[0, "sector"], "Energy")
check("D3 sector unclassified", nb.df.loc[2, "sector"], "Unclassified")
approx("D1 amount", nb.df.loc[0, "amount"], 120000)
check("close_date coverage counts NaT", nb.quality["close_date"].nulls >= 1, True)
check("D1 is_won", bool(nb.df.loc[0, "is_won"]), True)

# ---- Coverage-aware resolution: real Deals schema with an empty duplicate date ----
from normalize import parse_probability
real = pd.DataFrame([
    {"Item": "Naruto", "Owner code": "OWNER_001", "Client Code": "COMPANY089",
     "Deal Status": "Open", "Close Date (A)": "", "Closure Probability": "High",
     "Estimated Deal value": "₹2,00,000", "Estimated Close Date": "Feb 26 2026",
     "Deal Stage": "B. Sales Qualified", "Sector/service": "Renewables"},
    {"Item": "Sasuke", "Owner code": "OWNER_002", "Client Code": "COMPANY124",
     "Deal Status": "Open", "Close Date (A)": "", "Closure Probability": "Medium",
     "Estimated Deal value": "₹5,00,000", "Estimated Close Date": "Mar 20 2026",
     "Deal Stage": "E. Proposal", "Sector/service": "Powerline"},
])
res2 = resolve_columns(list(real.columns),
                       ["account", "sector", "stage", "amount", "close_date", "owner", "probability"],
                       df=real)
check("coverage picks populated date", res2["close_date"], "Estimated Close Date")
check("resolve amount (estimated)", res2["amount"], "Estimated Deal value")
check("resolve sector/service", res2["sector"], "Sector/service")
check("resolve stage not status", res2["stage"], "Deal Stage")
check("resolve owner code", res2["owner"], "Owner code")
check("resolve client code as account", res2["account"], "Client Code")
check("Powerline->Energy", canonical_sector("Powerline")[0], "Energy")

# ---- Categorical probability ----
approx("prob High", parse_probability("High"), 0.8)
approx("prob Medium", parse_probability("Medium"), 0.5)
approx("prob 70", parse_probability(70), 0.7)
check("prob blank NaN", pd.isna(parse_probability("")), True)

print("\n" + ("ALL PASSED" if not _failures else f"{len(_failures)} FAILURES: {_failures}"))
raise SystemExit(1 if _failures else 0)
