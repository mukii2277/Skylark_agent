# Decision Log

Founder BI Agent over two monday.com boards (Deals, Work Orders). Kept to the
decisions that shaped the build and the ones a reviewer would question.

## 1. Assumptions

- **Fiscal calendar = April–March (Indian FY).** "This quarter"/"this year" resolve
  against an April start (`FISCAL_YEAR_START_MONTH=4`, overridable). Because this is a
  genuine convention choice, the agent asks a clarifying question when a *relative*
  period is used and the answer is sensitive to it, and states the assumption if the
  user declines to specify.
- **"Revenue" = closed-won deal value, dated by close date.** Recognized revenue and
  work-order billing are not in scope; the agent clarifies if a question implies a
  different meaning. This is stated in every revenue answer's data note.
- **Won/lost interpretation of the deal funnel.** The real Deals board runs a lettered
  A–O funnel. Stages at or beyond commitment — **Work Order Received, Invoice Sent,
  Amount Accrued, Project Completed** — are counted as **won** alongside literal
  *Closed Won*; the two **Not Relevant** states are counted as **lost**. Counting only
  literal "Closed Won" would understate secured business by ~4× (27 vs ~100 deals in
  the current data). Everything between prospecting and negotiation, plus *On Hold*,
  is treated as open pipeline. This mapping lives in `normalize.py::STAGE_PATTERNS`.
- **Sector canonicalization** uses an explicit map (e.g. `Power & Energy`,
  `Renewables`, `Solar`, `Utilities` → **Energy**) plus a fuzzy fallback at
  threshold 85. Anything below threshold goes to an explicit **Unclassified** bucket
  and is counted — never silently merged into a neighbour.
- **Account join key** = casefolded account name with punctuation and company
  suffixes (`Pvt Ltd`, `Inc`, `LLP`, …) stripped, so `Acme Pvt. Ltd.` and `acme`
  join. Rows with no account become no key and are reported as unjoinable.
- **Cross-board join is not possible in the provided data — and the agent says so.**
  Deals key accounts as `COMPANY-089`; Work Orders key them as `WOCOMPANY_042` with
  independent numbering, and the WO `Serial# SDPLDEAL-NNN` has no counterpart column
  on Deals. The boards were anonymized separately, so the key sets do not intersect
  (0 shared keys). `cross_board_view` reports this explicitly and returns per-board
  figures rather than fabricating links. With a shared client/deal ID the same tool
  would join row-to-row unchanged.
- **Coverage-aware column resolution.** Logical fields are matched to real columns by
  fuzzy title score *and* data coverage, so a populated `Tentative Close Date` (78%)
  wins over an empty duplicate `Close Date (A)` (7.5%), and the real owner column
  `BD/KAM Personnel code` (94%) beats a near-empty `AR Priority account` (6%).
- **Mixed sector column.** Deals store sector in a combined `Sector/service` field, so
  service-type values (Tender, Pure Service, Spectra) legitimately fall to
  `Unclassified` (~27% of rows) — surfaced as a caveat, never silently merged.
- **Cache TTL = 5 minutes.** Long enough to keep a multi-turn conversation off the
  API's complexity budget, short enough that data is effectively live. This satisfies
  "no hardcoded data": every figure is fetched at runtime, only briefly memoised.

## 2. Trade-offs

- **Direct GraphQL over an MCP wrapper.** The brief permits either. The agent needs
  bulk board reads with column-type-aware flattening, cursor pagination, and a caching
  + retry layer. A generic MCP wrapper returns loosely-typed payloads that would still
  need the same normalization, so it adds a dependency without removing work. Direct
  GraphQL gives full control over pagination and the complexity budget.
- **Deterministic tools over LLM computation.** The model routes and narrates; pandas
  computes. This is the central decision: it removes arithmetic hallucination (the
  dominant LLM failure on raw rows) and makes every number traceable to a code path —
  essential when a founder acts on the answer.
- **OpenAI-compatible LLM layer, NVIDIA Nemotron by default.** The agent targets any
  OpenAI-compatible chat endpoint; it runs on **NVIDIA NIM (`nvidia/nemotron-3-ultra-
  550b-a55b`)** here, which supports native function calling — verified before build.
  Swapping to OpenAI or another provider is three secrets (`LLM_BASE_URL`, `LLM_MODEL`,
  `LLM_API_KEY`) and no code change, so the deterministic tool layer is insulated from
  provider choice.
- **Streamlit over a custom frontend.** Chat primitives, secrets management, and a
  free public URL out of the box — the right call inside a 6-hour budget where
  frontend time is pure overhead against the scored criteria.
- **Caching vs strict live-read.** A strict per-call live read is cleaner in theory but
  burns the API complexity budget across a multi-turn chat and slows every turn. A
  short TTL with a manual "refresh now" button is the pragmatic middle, and the code
  falls back to stale cache (with a visible staleness notice) rather than failing.
- **Fuzzy column resolution over hardcoded titles.** Board columns can be renamed on
  import. Logical fields (amount, close_date, sector, …) are resolved to the best
  matching column title; unresolved fields are reported in the data-quality view
  instead of crashing.

## 3. Data-quality handling (why this is first-class, not a footnote)

Every tool returns a `caveats` block: rows considered vs excluded with the reason,
per-field null/​unparsed/​unclassified counts, and coverage percentages. The agent is
instructed to surface these in plain language — e.g. *"Based on 47 of 61 energy deals;
14 have no close date and are excluded from the quarterly split."* Silent exclusion is
treated as a correctness bug, not a convenience.

## 4. Error handling

- 401 → explicit "token not configured/rejected" message, no silent fallback.
- 429 / complexity / 5xx → exponential backoff (3 attempts) → serve stale cache with a
  staleness notice → otherwise state that live data is unreachable.
- Empty result set → state that nothing matched and restate the filters applied.
- Unparseable value → excluded, counted, surfaced in caveats. Never guessed.

**Principle:** never present a number the pipeline could not verify; degrade to a
narrower but honest answer.

## 5. Interpretation of "Leadership updates" (optional requirement)

Interpreted as: *on request, the agent assembles a self-contained executive brief for
a stated period that a founder could paste into a board update without further
editing.* `generate_leadership_brief` returns six sections — pipeline snapshot,
revenue movement, sector performance, delivery health, watch items (largest stalled
deals), and a data-confidence line with named exclusions — narrated as markdown with a
download action in the UI.

## 6. What I'd do with more time

- Persistent store with incremental sync instead of a TTL cache, so history and trend
  lines survive restarts and don't re-fetch whole boards.
- A graded evaluation set of BI questions with expected tool calls and numbers, run in
  CI, to catch regressions in routing and aggregation.
- Chart rendering in-chat (stage funnel, sector bars, revenue trend).
- Semantic column mapping (embeddings) beyond fuzzy title match, for boards whose
  column names don't lexically resemble the logical field.
- Write-back of generated briefs to a monday.com board (would require moving beyond the
  current strictly read-only scope, with explicit confirmation).
```
