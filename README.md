# Founder BI Agent — monday.com Business Intelligence

A conversational agent that answers founder-level business questions by querying two
live **monday.com** boards — **Deals** (sales pipeline) and **Work Orders** (project
execution) — and returns *insight with honest data-quality caveats*, not just numbers.

Reference question it handles end-to-end:

> *"How's our pipeline looking for the energy sector this quarter?"*

…which requires intent parsing, sector normalization, fiscal-quarter resolution,
null-safe aggregation, and a caveat about deals with missing close dates.

---

## Core design decision — the LLM never computes numbers

The model **selects a tool and binds arguments**; deterministic pandas code computes
the result; the model **narrates** the returned figures. This eliminates arithmetic
hallucination and keeps every number traceable to a code path. See `bi_tools.py`.

```
Streamlit chat UI  (app.py)
        │  user turn + history
Agent orchestrator (agent.py)  — LLM tool-calling: intent, clarify, route, narrate
        │  typed tool calls
BI tool layer (bi_tools.py)    — deterministic pandas, each returns result + caveats
        │
Normalization layer (normalize.py, periods.py)  — dates · currency · sector · status · account
        │
monday.com client (monday_client.py)  — GraphQL v2, read-only, cursor pagination, TTL cache, retry/backoff
```

## What it handles well

- **Messy data (`normalize.py`).** Currency `₹1,20,000` / `$1.2M` / `2.5 Cr` → float;
  dates `12/03/2024` / `2024-03-12` / `Mar 12 2024` / `Q1 2024` / blank → typed or
  counted-as-missing; sectors `Energy`/`ENERGY`/`Power & Energy`/`Renewables` →
  canonical, fuzzy-matched, with an explicit `Unclassified` bucket; account names
  (`Acme Pvt. Ltd.` vs `acme`) → a stable join key.
- **Honest caveats (R6).** Every tool returns a `caveats` payload — rows considered vs
  excluded and why, per-field coverage — which the agent surfaces in plain language.
- **Ambiguity (R8).** Relative periods, unmatched sectors, and ambiguous "revenue"
  trigger at most one clarifying question; declined clarifications become stated
  assumptions.
- **Resilience (R11).** 401 → explicit config message; 429/5xx → exponential backoff
  then stale-cache fallback with a staleness notice; renamed columns → fuzzy title
  resolution.

## Tools

| Tool | Purpose |
|---|---|
| `get_pipeline_summary` | Open pipeline value/count/weighted forecast, by stage |
| `get_sector_performance` | Per-sector value, win rate, avg deal size |
| `get_revenue_summary` | Closed-won value + trend vs prior period |
| `get_work_order_status` | Status counts, overdue/at-risk items |
| `cross_board_view` | Deals joined to work orders on the account key |
| `get_data_quality_report` | Coverage, unparsed values, unresolved columns |
| `generate_leadership_brief` | Full executive brief for a period |
| `request_clarification` | One clarifying question when genuinely ambiguous |

---

## Setup

### 1. monday.com configuration

1. Import the two spreadsheets as boards (Deals, Work Orders). *(Already done.)*
2. Note each **board ID** from its URL: `monday.com/boards/<BOARD_ID>`.
3. Generate a **read-only-capable API v2 token**: avatar → **Developers** →
   **My access tokens** (or **Admin → API**). The app issues **only read queries**.

### 2. Secrets

Copy the template and fill it in:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # local
# or paste the same keys in Streamlit Cloud → Manage app → Settings → Secrets
```

Required: `MONDAY_API_TOKEN`, `DEALS_BOARD_ID`, `WORK_ORDERS_BOARD_ID`,
`LLM_API_KEY`. Optional: `LLM_BASE_URL` / `LLM_MODEL` (default the NVIDIA NIM
endpoint and `nvidia/nemotron-3-ultra-550b-a55b`), `CACHE_TTL_SECONDS`,
`FISCAL_YEAR_START_MONTH` (default `4` = April/Indian FY).

**LLM provider:** the agent talks to any OpenAI-compatible chat endpoint. It defaults
to **NVIDIA NIM (Nemotron)**; point `LLM_BASE_URL`/`LLM_MODEL`/`LLM_API_KEY` at OpenAI
or another provider to switch — no code change.

### 3. Run

```bash
pip install -r requirements.txt
python test_normalize.py     # sanity-check the normalization layer (no API needed)
streamlit run app.py
```

### 4. Deploy (Streamlit Community Cloud)

Push to GitHub → share.streamlit.io → point at `app.py` → paste secrets → deploy.
Test the public URL in a private window to confirm no local setup is required.

## Assumptions

- **Fiscal year starts in April** (Indian convention); override via
  `FISCAL_YEAR_START_MONTH`. Relative periods trigger a clarification when the answer
  is convention-sensitive.
- **"Revenue" = closed-won deal value** by close date.
- **Account join key** = casefolded name with punctuation and company suffixes
  (`Pvt Ltd`, `Inc`, …) stripped.
- **Cache TTL = 5 min**: live enough for a founder conversation, light on the API
  complexity budget across multi-turn chats.

See `DECISION_LOG.md` for the full rationale and trade-offs.

## Project layout

```
app.py            Streamlit chat UI
agent.py          LLM tool-calling orchestrator (OpenAI-compatible / NVIDIA Nemotron)
bi_tools.py       Deterministic pandas BI tools (result + caveats)
normalize.py      Normalization: dates, currency, sector, status, account keys
periods.py        Fiscal-period resolution
monday_client.py  GraphQL v2 client: pagination, TTL cache, retry/backoff
config.py         Secret/config loading
test_normalize.py Standalone unit checks for the normalization layer
```
