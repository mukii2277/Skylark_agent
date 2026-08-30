"""Streamlit chat UI (R12) — the hosted, testable prototype.

Wires the browser chat to the agent, loads/normalizes both boards behind a short
TTL cache, and degrades gracefully when configuration or the API is unavailable.
"""
from __future__ import annotations

import json
import time

import streamlit as st

from agent import run_agent
from bi_tools import BIContext
from config import get_settings
from monday_client import MondayError

st.set_page_config(page_title="Skylark BI Agent", page_icon="🛰️", layout="centered")

st.title("🛰️ Founder BI Agent")
st.caption(
    "Ask about the sales pipeline and project delivery. Numbers come from monday.com "
    "via deterministic tools — the model routes and narrates, it never does the math."
)

settings = get_settings()

# --- Configuration gate ---------------------------------------------------- #
missing = settings.missing()
if missing:
    st.error(
        "This deployment is not fully configured yet. Missing secrets: "
        + ", ".join(f"`{m}`" for m in missing)
        + ".\n\nAdd them under **Manage app → Settings → Secrets** (or a local "
        "`.streamlit/secrets.toml`). See `secrets.toml.example`."
    )
    st.stop()


@st.cache_resource(show_spinner="Loading boards from monday.com…")
def _load_context(cache_bucket: int) -> BIContext:
    """Build the normalized context. cache_bucket rotates every TTL seconds."""
    return BIContext.build()


def get_context(force: bool = False) -> BIContext:
    ttl = max(settings.cache_ttl_seconds, 30)
    bucket = 0 if force else int(time.time() // ttl)
    if force:
        _load_context.clear()
    return _load_context(bucket)


# --- Sidebar: status + data quality ---------------------------------------- #
with st.sidebar:
    st.subheader("Data source")
    st.write(f"Fiscal year starts: **month {settings.fiscal_year_start_month}** (Apr = 4)")
    st.write(f"Cache TTL: **{settings.cache_ttl_seconds}s**")
    if st.button("🔄 Refresh data now"):
        get_context(force=True)
        st.success("Re-fetched from monday.com.")
    try:
        ctx = get_context()
        for w in ctx.warnings:
            st.warning(w)
        with st.expander("Data quality snapshot"):
            for name, nb in {"Deals": ctx.deals, "Work Orders": ctx.work_orders}.items():
                st.markdown(f"**{name}** — {len(nb.df)} rows")
                if nb.unresolved_fields:
                    st.caption("Unresolved fields: " + ", ".join(nb.unresolved_fields))
                for q in nb.quality_records():
                    st.caption(
                        f"{q['field']}: {q['coverage_pct']}% covered "
                        f"({q['nulls']} null, {q['unparsed']} unparsed, "
                        f"{q['unclassified']} unclassified)"
                    )
    except MondayError as exc:
        st.error(f"Could not load boards: {exc}")
        st.stop()

# --- Chat state ------------------------------------------------------------ #
if "history" not in st.session_state:
    st.session_state.history = []  # Anthropic-format messages
if "display" not in st.session_state:
    st.session_state.display = []  # [{role, content, trace}]

for msg in st.session_state.display:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("trace"):
            with st.expander("Tools called"):
                for t in msg["trace"]:
                    st.markdown(f"**{t['tool']}**  `{json.dumps(t['args'])}`")
                    st.json(t["result"].get("caveats", {}), expanded=False)

# Example prompts on first load
if not st.session_state.display:
    st.markdown("**Try:**")
    for ex in [
        "How's our pipeline looking for the energy sector this quarter?",
        "Which sectors are converting best?",
        "Any overdue work orders I should worry about?",
        "Prepare a leadership brief for this quarter.",
    ]:
        st.markdown(f"- {ex}")

prompt = st.chat_input("Ask a business question…")
if prompt:
    st.session_state.history.append({"role": "user", "content": prompt})
    st.session_state.display.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        tool_log: list[str] = []

        def _on_tool(name, args):
            tool_log.append(name)
            placeholder.markdown("⚙️ " + " → ".join(tool_log))

        try:
            ctx = get_context()
            result = run_agent(st.session_state.history, ctx, on_tool=_on_tool)
        except MondayError as exc:
            result = {"text": f"⚠️ Live data is unreachable right now: {exc}", "clarification": None, "tool_trace": []}
        except Exception as exc:  # noqa: BLE001 — surface, don't crash the app
            result = {"text": f"⚠️ Something went wrong: {type(exc).__name__}: {exc}", "clarification": None, "tool_trace": []}

        if result.get("clarification"):
            clar = result["clarification"]
            text = "❓ " + clar["question"]
            if clar.get("options"):
                text += "\n\n" + "\n".join(f"- {o}" for o in clar["options"])
            placeholder.markdown(text)
            # Record the clarification as the assistant turn so the follow-up has context.
            st.session_state.history.append({"role": "assistant", "content": clar["question"]})
            st.session_state.display.append({"role": "assistant", "content": text, "trace": []})
        else:
            text = result["text"] or "_(no answer)_"
            placeholder.markdown(text)
            st.session_state.history.append({"role": "assistant", "content": text})
            st.session_state.display.append(
                {"role": "assistant", "content": text, "trace": result.get("tool_trace", [])}
            )
            # Offer the leadership brief as a download when one was generated.
            for t in result.get("tool_trace", []):
                if t["tool"] == "generate_leadership_brief":
                    st.download_button(
                        "⬇️ Download brief (markdown)",
                        data=text,
                        file_name="leadership_brief.md",
                        mime="text/markdown",
                    )
