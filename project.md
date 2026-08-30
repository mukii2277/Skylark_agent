# Monday.com Business Intelligence Agent — Project Plan

**Source:** Skylark Drones — Full-Stack Assignment (RVU)
**Time budget:** 6 hours
**Submission:** https://forms.gle/qGihfi4zCLBxKWK68

---

## 1. Objective

Build a conversational AI agent that answers founder-level business intelligence
questions by querying two monday.com boards live — **Work Orders** (project
execution) and **Deals** (sales pipeline) — and returning insight, not just
numbers, while degrading gracefully on messy and incomplete data.

Reference query the system must handle end-to-end:

> *"How's our pipeline looking for the energy sector this quarter?"*

This single question requires intent parsing, sector normalization, fiscal-period
resolution, cross-board joins, aggregation, and a caveat about records with
missing close dates.

---

## 2. Requirements Traceability

| # | Requirement (from brief) | Implementation approach | Priority |
|---|---|---|---|
| R1 | Connect to monday.com via MCP or API | GraphQL API v2 + token auth | Must |
| R2 | No hardcoded CSV data | Runtime fetch, TTL cache only | Must |
| R3 | Read-only access | Only `boards`/`items_page` queries; no mutations | Must |
| R4 | Handle missing/null values | Null-safe aggregations, coverage % reported | Must |
| R5 | Normalize dates, names, text | Dedicated normalization layer | Must |
| R6 | Communicate data quality caveats | Caveat block appended to every answer | Must |
| R7 | Interpret founder-level questions | LLM tool-calling over typed BI tools | Must |
| R8 | Ask clarifying questions | Ambiguity detector → clarify tool | Must |
| R9 | Revenue, pipeline, sector, ops metrics | Deterministic aggregation tools | Must |
| R10 | Cross-board queries | Shared normalized keys (account, sector) | Must |
| R11 | Graceful API failure handling | Retry + backoff + stale-cache fallback | Must |
| R12 | Hosted, testable without local setup | Streamlit Community Cloud | Must |
| R13 | Leadership update preparation | `generate_leadership_brief` tool | Optional |
| R14 | Decision Log (2 pages max) | Written last, from running notes | Must |
| R15 | Source ZIP + README | Repo export + architecture README | Must |

---

## 3. Architecture

```
┌──────────────────────────────────────────────────────────┐
│  Streamlit chat UI  (hosted, public link)                │
└───────────────────────────┬──────────────────────────────┘
                            │ user turn + history
┌───────────────────────────▼──────────────────────────────┐
│  Agent Orchestrator  (LLM with tool calling)             │
│  - intent + entity extraction                            │
│  - ambiguity check → clarifying question                 │
│  - tool selection & argument binding                     │
│  - narrative synthesis + caveats                         │
└───────────────────────────┬──────────────────────────────┘
                            │ typed tool calls
┌───────────────────────────▼──────────────────────────────┐
│  BI Tool Layer  (deterministic, pandas — no LLM math)    │
│  pipeline_summary | sector_performance | revenue_summary │
│  work_order_status | cross_board_view | data_quality     │
│  leadership_brief                                        │
└───────────────────────────┬──────────────────────────────┘
┌───────────────────────────▼──────────────────────────────┐
│  Normalization Layer                                     │
│  dates · currency · sector · status · owner · account    │
└───────────────────────────┬──────────────────────────────┘
┌───────────────────────────▼──────────────────────────────┐
│  monday.com Client  (GraphQL v2, read-only, TTL cache)   │
└──────────────────────────────────────────────────────────┘
```

### 3.1 Central design decision

**The LLM never computes numbers.** It selects a tool and binds arguments; pandas
computes the result; the LLM narrates the returned figures. This eliminates
arithmetic hallucination, which is the dominant failure mode when an LLM is handed
raw rows. It also keeps every number traceable to a deterministic code path —
essential when a founder acts on the answer.

---

## 4. Technology Stack

| Layer | Choice | Justification |
|---|---|---|
| UI | Streamlit | Chat primitives built in; deploys free in minutes; zero frontend time cost within a 6-hour budget |
| Language | Python 3.11 | pandas + strong LLM SDK support |
| Agent | Anthropic / OpenAI SDK with tool calling | Native structured tool invocation; no orchestration framework overhead |
| Data | pandas | Null-tolerant grouping and aggregation |
| Integration | monday.com GraphQL API v2 (`requests`) | Full control over pagination, column parsing and caching; MCP adds a dependency without adding capability here |
| Hosting | Streamlit Community Cloud | Public URL, secrets management, no infra work |
| Config | `st.secrets` / `.env` | API token and board IDs never committed |

**API vs MCP:** the brief permits either. Direct GraphQL is chosen because the
agent needs bulk board reads with column-type-aware parsing and a caching layer;
a generic MCP wrapper would return loosely-typed payloads that still require the
same normalization work. Record this reasoning in the Decision Log.

---

## 5. Data Handling Strategy

### 5.1 Ingestion

Fetch both boards via `items_page` with cursor pagination (500 items/page).
Flatten `column_values` into a wide DataFrame keyed by column title. Cache with a
**5-minute TTL** — satisfies R2 (data is queried dynamically, not hardcoded) while
staying inside monday.com's complexity budget across a multi-turn conversation.

### 5.2 Normalization rules

| Field type | Messiness expected | Rule |
|---|---|---|
| Dates | `12/03/2024`, `2024-03-12`, `Mar 12 2024`, `Q1 2024`, blank | Multi-format parser with day-first preference; unparseable → `NaT`, counted |
| Currency | `₹1,20,000`, `$1.2M`, `12000 USD`, `1,20,000/-` | Strip symbols/separators, expand `K`/`M`/`L`/`Cr` suffixes → float |
| Sector | `Energy`, `energy`, `ENERGY`, `Power & Energy`, `Renewables` | Canonical map + fuzzy match (threshold ≥ 85), else `Unclassified` |
| Status | `Won`, `Closed Won`, `won ✅` | Regex canonicalization into a fixed stage enum |
| Account/Owner | trailing spaces, case drift, `Pvt Ltd` vs `Pvt. Ltd.` | Trim, casefold, suffix stripping → join key |
| Nulls | blank, `-`, `N/A`, `TBD`, `NA` | Unified to `NaN` before aggregation |

### 5.3 Quality reporting

Every tool returns a `caveats` payload alongside its result:

- rows considered vs rows excluded, with the reason
- per-field null coverage for fields used in the answer
- records that fell into `Unclassified` or `NaT` buckets

The agent surfaces this in plain language, e.g. *"Based on 47 of 61 energy deals —
14 have no close date and are excluded from the quarterly split."* This directly
satisfies R6 and is a likely scoring differentiator.

---

## 6. Tool Specification

| Tool | Key arguments | Returns |
|---|---|---|
| `get_pipeline_summary` | sector, period, stage | value by stage, deal count, weighted forecast, caveats |
| `get_sector_performance` | period, metric, top_n | per-sector totals, win rate, avg deal size |
| `get_revenue_summary` | period, granularity | closed-won value, trend vs prior period |
| `get_work_order_status` | sector, status, owner | counts by status, overdue/at-risk items |
| `cross_board_view` | account or sector | deals joined to work orders on normalized account key |
| `get_data_quality_report` | board | null coverage, unparsed values, duplicates |
| `generate_leadership_brief` | period | structured executive summary (see §7) |
| `request_clarification` | question, options | returned to user when the query is ambiguous |

### 6.1 Clarification triggers

Ask, do not guess, when:

- the period is relative and fiscal-year convention is unstated (*"this quarter"*)
- a named sector matches no canonical value above threshold
- *"revenue"* could mean closed-won, recognized, or work-order value
- the question spans both boards without an explicit metric

Cap at **one clarifying question per turn**; if the user declines to specify, state
the assumption and proceed. The brief explicitly rewards handling ambiguity, so
these assumptions must also appear in the Decision Log.

---

## 7. Interpretation of "Leadership Updates" (Optional Requirement)

Interpreted as: *the agent can assemble, on request, a self-contained executive
brief for a stated period that a founder could paste into a board update without
further editing.*

The brief contains:

1. **Pipeline snapshot** — total value, stage distribution, weighted forecast
2. **Movement** — closed-won vs prior period, win rate, average cycle where derivable
3. **Sector performance** — top and bottom sectors by value and conversion
4. **Delivery health** — work orders by status, overdue or at-risk items
5. **Watch items** — largest deals stalled in a stage
6. **Data confidence** — coverage percentage and named exclusions

Delivered as formatted markdown in-chat with a copy/download action. This
interpretation must be documented in the Decision Log per the brief.

---

## 8. Implementation Schedule (6 Hours)

| Slot | Duration | Work | Exit criterion |
|---|---|---|---|
| H0 | 0:00–0:40 | Import both XLSX files as monday.com boards; set column types; capture board IDs; generate API token | Boards live, IDs recorded |
| H1 | 0:40–1:30 | GraphQL client: auth, pagination, column flattening, TTL cache, retry/backoff | Both boards load into DataFrames |
| H2 | 1:30–2:30 | Normalization layer + unit checks against real messy values | Clean typed DataFrames, quality report runs |
| H3 | 2:30–3:40 | BI tool layer (all seven tools) with caveat payloads | Tools callable and correct in isolation |
| H4 | 3:40–4:40 | Agent orchestration: tool schemas, system prompt, clarification logic, synthesis | Reference query answered end-to-end |
| H5 | 4:40–5:20 | Streamlit UI, secrets wiring, deploy to Community Cloud | Public URL testable without local setup |
| H6 | 5:20–6:00 | Decision Log, README, source ZIP, submission | All deliverables uploaded |

**Scope guard:** if H3 overruns, drop `cross_board_view` and the leadership brief
before cutting normalization or caveats — data resilience is a named core feature,
the brief is optional.

---

## 9. Error Handling

| Failure | Response |
|---|---|
| Auth failure (401) | Explicit configuration message in UI; no silent fallback |
| Rate limit / complexity budget (429) | Exponential backoff, 3 attempts, then serve stale cache with a staleness notice |
| Timeout or 5xx | Serve stale cache if available, otherwise state that live data is unreachable |
| Board or column renamed | Column resolution by fuzzy title match; log unresolved columns as a quality issue |
| Empty result set | State that no records matched, restate the filters applied, offer a broader query |
| Unparseable value | Exclude from computation, count it, surface in caveats |

**Principle:** never present a number the pipeline could not verify. Degrade to a
narrower but honest answer instead.

---

## 10. Deliverables Checklist

| # | Deliverable | Detail | Status |
|---|---|---|---|
| D1 | Hosted prototype | Public Streamlit URL, no local setup needed | ☐ |
| D2 | Decision Log | 2 pages max: assumptions, trade-offs, what you'd do differently, "leadership updates" interpretation | ☐ |
| D3 | Source code ZIP | Full repo, secrets excluded | ☐ |
| D4 | README | Architecture overview + monday.com configuration and setup steps | ☐ |
| D5 | Submission | Google Form completed | ☐ |

### 10.1 Decision Log outline

1. **Assumptions** — fiscal calendar convention, revenue definition, sector
   canonicalization map, account join key, cache TTL rationale
2. **Trade-offs** — GraphQL over MCP; deterministic tools over LLM computation;
   Streamlit over a custom frontend; caching against strict live-read
3. **With more time** — persistent store with incremental sync, evaluation set of
   graded BI questions, chart rendering, semantic column mapping, write-back of
   briefs to a monday.com board
4. **Leadership updates interpretation** — as specified in §7

---

## 11. Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| CSV import produces wrong column types in monday.com | Downstream parsing breaks | Verify types immediately at H0; parser tolerates text columns regardless |
| Sector values messier than anticipated | Wrong groupings, wrong answers | Fuzzy match with explicit `Unclassified` bucket; never silently merge |
| Board size exceeds a single API page | Silent data loss | Cursor pagination implemented at H1, not retrofitted |
| Deployment secrets misconfigured | Reviewer sees a broken link | Deploy at H5 with 40 minutes of buffer; test the public URL in a private window |
| Overrun on agent prompt tuning | Deliverables unfinished | Hard stop at 5:20; ship the working state |

---

## 12. Evaluation Signals to Optimize For

The brief states that handling ambiguity and messy data is itself part of the
assessment. Weight effort accordingly:

1. Correct, verifiable numbers on messy input — the primary bar
2. Visible, specific data-quality caveats rather than silent exclusion
3. Clarifying questions that show the ambiguity was understood, not deflected
4. A Decision Log that argues its trade-offs rather than listing them
5. A hosted link that works on first click, without setup
