"""Agent orchestration (R7, R8).

An OpenAI-compatible tool-calling loop (pointed at NVIDIA NIM / Nemotron by default,
but any OpenAI-compatible endpoint works) that:
  * interprets founder-level questions,
  * asks *one* clarifying question when the query is genuinely ambiguous
    (relative period + unstated fiscal convention, unmatched sector, ambiguous
    "revenue", cross-board query without a metric),
  * selects deterministic BI tools and binds their arguments,
  * narrates the returned figures with explicit data-quality caveats.

The model never computes numbers — it only routes to tools and explains results.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from openai import OpenAI

from bi_tools import BIContext, TOOL_FUNCTIONS
from config import get_settings

SYSTEM_PROMPT = """\
You are a business-intelligence analyst for the founders of a drone-services company.
You answer questions about the sales pipeline (Deals board) and project execution
(Work Orders board) by calling deterministic tools. Two absolute rules:

1. NEVER compute or estimate a number yourself. Call a tool and report the numbers it
   returns, exactly. If a figure is not in a tool result, say you don't have it.
2. ALWAYS surface the data-quality caveats returned by the tools in plain founder
   language — how many records were considered, and what was excluded and why (e.g.
   deals with no close date, amounts that could not be parsed, sectors that fell into
   "Unclassified"). This honesty is a feature, not a disclaimer to bury.

Clarify BEFORE calling a tool — using the request_clarification tool, at most once per
turn — only when it genuinely changes the answer:
  * A relative period is used ("this quarter", "this year") AND the answer is
    sensitive to the fiscal convention. Note: the fiscal year starts in April by
    default. If the user already accepted an assumption, proceed and state it.
  * A named sector does not map to any known sector (the tool will tell you it was
    unmatched) — ask which sector they mean.
  * "revenue" is ambiguous between closed-won value and other meanings AND context
    doesn't resolve it.
  * A question spans both boards but names no metric.
If the user waves off a clarification, state the assumption you're making and proceed.

Notes about this data:
  * "Won" business includes the post-commitment stages the pipeline uses (Work Order
    Received, Invoice Sent, Amount Accrued, Project Completed) as well as Closed Won —
    the tools already apply this; explain it if a user asks how "won" is defined.
  * The Deals and Work Orders boards were anonymized independently and share no common
    account/deal key, so they cannot be joined row-to-row. If a question needs that,
    say so plainly and answer per-board instead of implying a link.
  * The deals "sector" field is a mixed "Sector/service" column, so a meaningful share
    of rows are Unclassified (service types, not sectors) — always surface that count.

Style: lead with the answer, then the supporting numbers, then a short, specific
"Data note:" line with the caveats. Be concise and concrete. Use the actual figures.
Format money with thousands separators (Indian rupees unless stated). Never invent
deals, sectors, or trends. Do not show your private reasoning; give the final answer.
"""


def _tool_schemas() -> list[dict[str, Any]]:
    """Tool definitions in OpenAI function-calling format."""
    defs: list[tuple[str, str, dict]] = [
        ("get_pipeline_summary",
         "Open sales pipeline: total open value, deal count, weighted forecast, and a breakdown by stage. Filter by sector, period (applies to expected close date), and/or stage.",
         {"type": "object", "properties": {
             "sector": {"type": "string", "description": "Sector name, e.g. 'energy'. Optional."},
             "period": {"type": "string", "description": "Period phrase, e.g. 'this quarter', 'Q1 FY25', '2024'. Optional; omit for all-time."},
             "stage": {"type": "string", "description": "Deal stage filter, e.g. 'negotiation'. Optional."}}}),
        ("get_sector_performance",
         "Per-sector performance: open pipeline value, closed-won value, deal count, win rate, average deal size. Optionally rank by a metric and take top_n.",
         {"type": "object", "properties": {
             "period": {"type": "string", "description": "Period phrase. Optional; omit for all-time."},
             "metric": {"type": "string", "enum": ["pipeline_value", "closed_won_value", "deal_count", "avg_deal_size"], "description": "Ranking metric."},
             "top_n": {"type": "integer", "description": "Return only the top N sectors. Optional."}}}),
        ("get_revenue_summary",
         "Closed-won revenue for a period with comparison to the immediately preceding period, and optional monthly/quarterly breakdown. 'Revenue' means closed-won deal value by close date.",
         {"type": "object", "properties": {
             "period": {"type": "string", "description": "Period phrase. Optional; omit for all-time."},
             "granularity": {"type": "string", "enum": ["total", "month", "quarter"], "description": "Breakdown granularity."},
             "sector": {"type": "string", "description": "Optional sector filter."}}}),
        ("get_work_order_status",
         "Work-order (project execution) status counts, plus overdue/at-risk items. Filter by sector, status, or owner.",
         {"type": "object", "properties": {
             "sector": {"type": "string"},
             "status": {"type": "string", "description": "e.g. 'in progress', 'completed', 'on hold'."},
             "owner": {"type": "string"}}}),
        ("cross_board_view",
         "Attempt to relate deals and work orders. NOTE: in this dataset the two boards share no common key, so this returns per-board figures and says so. Use for account- or sector-level questions spanning sales and delivery.",
         {"type": "object", "properties": {
             "account": {"type": "string", "description": "Account/company name. Optional."},
             "sector": {"type": "string", "description": "Sector. Optional."}}}),
        ("get_data_quality_report",
         "Data-quality report for a board: resolved columns, unresolved fields, per-field null coverage and unparsed counts. Use when the user asks about data completeness or trustworthiness.",
         {"type": "object", "properties": {
             "board": {"type": "string", "enum": ["deals", "work_orders"], "description": "Optional; omit for both."}}}),
        ("generate_leadership_brief",
         "Assemble a complete executive brief for a period: pipeline snapshot, revenue movement, sector performance, delivery health, watch items, and data confidence.",
         {"type": "object", "properties": {
             "period": {"type": "string", "description": "Period phrase, e.g. 'this quarter'. Optional."}}}),
        ("request_clarification",
         "Ask the user ONE clarifying question when the request is genuinely ambiguous. Do not use this to avoid work — only when the answer materially depends on the missing detail.",
         {"type": "object", "properties": {
             "question": {"type": "string"},
             "options": {"type": "array", "items": {"type": "string"}, "description": "Optional suggested answers."}},
             "required": ["question"]}),
    ]
    return [
        {"type": "function", "function": {"name": n, "description": d, "parameters": p}}
        for n, d, p in defs
    ]


def run_agent(
    messages: list[dict[str, Any]],
    ctx: BIContext,
    *,
    on_tool: Callable[[str, dict], None] | None = None,
    max_iters: int = 6,
) -> dict[str, Any]:
    """Run the tool-calling loop over an existing message history.

    ``messages`` is a list of {"role","content"} dicts (user/assistant turns).
    Returns {"text": final_text, "clarification": {...}|None, "tool_trace": [...]}.
    """
    settings = get_settings()
    client = OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)
    tools = _tool_schemas()
    convo: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}, *messages]
    tool_trace: list[dict[str, Any]] = []

    for _ in range(max_iters):
        resp = client.chat.completions.create(
            model=settings.llm_model,
            messages=convo,
            tools=tools,
            tool_choice="auto",
            temperature=0.2,
            max_tokens=2000,
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            return {"text": _clean(msg.content or ""), "clarification": None, "tool_trace": tool_trace}

        # Append the assistant turn (with its tool_calls) before the tool results.
        convo.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ],
        })

        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            if on_tool:
                on_tool(name, args)

            if name == "request_clarification":
                return {
                    "text": "",
                    "clarification": {"question": args.get("question", ""), "options": args.get("options", [])},
                    "tool_trace": tool_trace,
                }

            func = TOOL_FUNCTIONS.get(name)
            if func is None:
                result = {"error": f"unknown tool {name}"}
            else:
                try:
                    result = func(ctx, **args)
                except Exception as exc:  # keep the loop alive; report the failure to the model
                    result = {"error": f"{type(exc).__name__}: {exc}"}
            tool_trace.append({"tool": name, "args": args, "result": result})
            convo.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, default=str),
            })

    return {
        "text": "I wasn't able to complete that within a reasonable number of steps. "
                "Try narrowing the question.",
        "clarification": None,
        "tool_trace": tool_trace,
    }


def _clean(text: str) -> str:
    """Strip any reasoning-model <think>…</think> block that leaks into content."""
    import re
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()
