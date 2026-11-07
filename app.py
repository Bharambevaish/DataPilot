"""
DataPilot -- the Streamlit web app.

This file is only the user interface. It holds no SQL, no prompts and no agent
logic; it loads data, asks src/agent.py a question, and draws what comes back.
Keeping it that way is what makes the agent testable from the command line and
from eval/evaluate.py without Streamlit being involved at all.

One thing worth knowing if you have not used Streamlit: the whole script runs
again from the top on every click. So anything that must survive a click --
the loaded dataset, the chat history -- lives in st.session_state, and anything
expensive is only rebuilt when its inputs change.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import streamlit as st

from src import config, logger
from src.agent import Agent, AgentAnswer, AgentError
from src.charts import build_chart, suggest_chart
from src.data_loader import DataLoadError, DataStore, load_files, load_sample_dataset

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="DataPilot -- ask your data",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded",
)

SAMPLE_DATASETS = {
    "sales": {
        "label": "Sales (one table, 4,974 rows)",
        "description": "Order lines for 2024: region, city, product, category, "
                       "quantity, unit price and discount.",
        "questions": [
            "Which region's sales dropped the most last quarter?",
            "What are the top 5 products by revenue?",
            "How did monthly revenue change through 2024?",
            "Which category has the highest average discount?",
            "Compare revenue per region between the first and second half of the year",
        ],
    },
    "shop_orders": {
        "label": "Shop orders (two tables, needs JOINs)",
        "description": "A customers sheet and an orders sheet, so most questions "
                       "need to join the two.",
        "questions": [
            "Which customer segment has the highest average order value?",
            "How many customers have never placed an order?",
            "Who are the top 5 customers by total spend?",
            "What share of orders were cancelled or returned, by channel?",
            "Which city has the most Enterprise customers?",
        ],
    },
}


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def init_state() -> None:
    """Create the session keys once, so the rest of the file can assume them."""
    defaults: dict[str, Any] = {
        "store": None,             # the loaded DataStore
        "store_key": None,         # identifies what is loaded, to detect changes
        "store_label": "",         # shown in the sidebar
        "agent": None,             # the Agent built for that store
        "history": [],             # the chat, as a list of dicts
        "pending_question": None,  # set by the example-question buttons
        "questions_asked": 0,      # for the demo-mode cap
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def set_store(store: DataStore, key: str, label: str) -> None:
    """
    Swap in a new dataset.

    The old DataStore owns a temporary DuckDB file, so it is cleaned up before
    being replaced -- otherwise every upload would leave a file behind.
    """
    old = st.session_state.get("store")
    if old is not None and old is not store:
        old.cleanup()

    st.session_state.store = store
    st.session_state.store_key = key
    st.session_state.store_label = label
    st.session_state.agent = None      # rebuilt lazily against the new schema
    st.session_state.history = []      # old answers refer to data that is gone


def get_agent() -> Agent | None:
    """Build the agent for the current dataset, or show why it cannot be built."""
    if st.session_state.store is None:
        return None
    if st.session_state.agent is not None:
        return st.session_state.agent

    try:
        st.session_state.agent = Agent(st.session_state.store)
    except AgentError as error:
        st.error(str(error), icon=":material/key_off:")
        with st.expander("How to add your API key"):
            st.markdown(
                "**Running locally**\n"
                "1. Get a free key at [console.groq.com/keys](https://console.groq.com/keys)\n"
                "2. Copy `.env.example` to `.env`\n"
                "3. Put your key in it: `GROQ_API_KEY=gsk_...`\n"
                "4. Restart the app\n\n"
                "**Deployed on Streamlit Cloud**\n"
                "Open *Manage app -> Settings -> Secrets* and add:\n"
                "```toml\n"
                'GROQ_API_KEY = "gsk_..."\n'
                "```"
            )
        return None
    return st.session_state.agent


def dark_mode() -> bool:
    """Ask Streamlit whether the browser is showing the dark theme."""
    try:
        return str(st.context.theme.type).lower() == "dark"
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Sidebar: choosing the data
# ---------------------------------------------------------------------------

def sidebar() -> None:
    with st.sidebar:
        st.markdown("## Your data")

        demo = config.demo_mode()
        if demo:
            st.info(
                "**Public demo.** Uploads are turned off here, so please do not "
                "try to share private data. Run it locally to use your own files.",
                icon=":material/public:",
            )

        # --- sample datasets ------------------------------------------------
        st.markdown("**Use a sample dataset**")
        for name, info in SAMPLE_DATASETS.items():
            is_current = st.session_state.store_key == f"sample:{name}"
            if st.button(
                info["label"],
                key=f"load_{name}",
                width="stretch",
                type="primary" if is_current else "secondary",
                disabled=is_current,
            ):
                try:
                    set_store(load_sample_dataset(name), f"sample:{name}", info["label"])
                    st.rerun()
                except DataLoadError as error:
                    st.error(str(error), icon=":material/error:")
            st.caption(info["description"])

        # --- upload ---------------------------------------------------------
        if not demo:
            st.divider()
            st.markdown("**Or upload your own**")
            limit_mb = config.max_upload_mb()
            uploaded = st.file_uploader(
                f"CSV or Excel, up to {limit_mb} MB",
                type=["csv", "tsv", "xlsx", "xlsm", "xls"],
                accept_multiple_files=True,
                help="An Excel workbook becomes one table per sheet, so questions "
                     "can join across sheets.",
            )
            if uploaded:
                handle_upload(uploaded, limit_mb)

        # --- what is loaded -------------------------------------------------
        store: DataStore | None = st.session_state.store
        if store is not None:
            st.divider()
            st.markdown(f"**Loaded:** {st.session_state.store_label}")
            for table in store.tables.values():
                st.markdown(
                    f"- `{table.name}` -- {table.n_rows:,} rows x {table.n_cols} columns"
                )

            with st.expander("Preview data"):
                for table in store.tables.values():
                    st.caption(f"{table.name} ({table.source})")
                    st.dataframe(table.frame.head(50), width="stretch",
                                 hide_index=True)

        st.divider()
        st.caption(
            f"Model: `{config.model_name()}`  \n"
            f"Read-only SQL - row cap {config.max_rows():,} - "
            f"{config.query_timeout_seconds()}s query timeout"
        )


def handle_upload(uploaded_files: list[Any], limit_mb: int) -> None:
    """Save the uploaded files to a temporary folder and load them."""
    # A key built from names and sizes, so re-running does not reload the same files.
    key = "upload:" + "|".join(f"{f.name}:{f.size}" for f in uploaded_files)
    if st.session_state.store_key == key:
        return

    too_big = [f.name for f in uploaded_files if f.size > limit_mb * 1024 * 1024]
    if too_big:
        st.error(
            f"{', '.join(too_big)} is larger than {limit_mb} MB. "
            "Try a smaller file, or a sample of your data.",
            icon=":material/error:",
        )
        return

    temp_dir = Path(tempfile.mkdtemp(prefix="datapilot_upload_"))
    paths = []
    for item in uploaded_files:
        path = temp_dir / item.name
        path.write_bytes(item.getbuffer())
        paths.append(path)

    try:
        with st.spinner("Reading your file..."):
            store = load_files(paths)
    except DataLoadError as error:
        st.error(str(error), icon=":material/error:")
        return
    except Exception as error:   # noqa: BLE001 - never show a stack trace to a user
        st.error(f"Could not read that file: {error}", icon=":material/error:")
        return

    label = ", ".join(item.name for item in uploaded_files)
    set_store(store, key, label)
    st.rerun()


# ---------------------------------------------------------------------------
# Drawing one answer
# ---------------------------------------------------------------------------

def render_answer(entry: dict[str, Any]) -> None:
    """Draw one assistant turn: the words, the SQL, the table, the chart, the steps."""
    outcome: AgentAnswer = entry["outcome"]

    # 1. The answer in words.
    if outcome.refused:
        st.warning(outcome.answer, icon=":material/help:")
    elif not outcome.success:
        st.error(outcome.answer, icon=":material/error:")
    else:
        st.markdown(outcome.answer)

    # 2. The SQL it used.
    if outcome.sql:
        with st.expander("SQL used", expanded=False):
            st.code(outcome.sql, language="sql")

    # 3. The result table.
    if outcome.result is not None and not outcome.result.is_empty:
        st.caption(f"Result -- {outcome.result.row_count:,} row(s)")
        st.dataframe(outcome.result.dataframe, width="stretch", hide_index=True)
        for note in outcome.result.notes:
            st.caption(note)
    elif outcome.result is not None and outcome.result.is_empty:
        st.info(
            "The query ran but matched no rows. That usually means a filter value "
            "is not present in this data.",
            icon=":material/info:",
        )

    # 4. The chart.
    figure = entry.get("figure")
    if figure is not None:
        st.plotly_chart(figure, width="stretch",
                        key=f"chart_{entry['id']}", config={"displaylogo": False})
        if entry.get("chart_note"):
            st.caption(entry["chart_note"])

    # 5. How it got there.
    with st.expander(f"How I got this ({len(outcome.steps)} step(s))"):
        if not outcome.steps:
            st.caption("The agent answered without needing to query the data.")
        for number, call in enumerate(outcome.steps, start=1):
            icon = ":material/check_circle:" if call.ok else ":material/cancel:"
            st.markdown(f"**Step {number}** {icon} `{call.name}`")
            if call.name == "run_sql":
                st.code(str(call.arguments.get("query", "")), language="sql")
            elif call.arguments:
                st.caption(", ".join(f"{k}={v}" for k, v in call.arguments.items()))
            if not call.ok:
                st.caption(f"Error fed back to the agent: {call.error}")
        st.caption(
            f"{outcome.llm_turns} model turn(s) - "
            f"{outcome.self_corrections} self-correction(s) - "
            f"{outcome.duration_seconds:.1f}s - model `{outcome.model}`"
        )


def build_figure_for(outcome: AgentAnswer) -> tuple[Any, str]:
    """
    Decide whether to draw a chart, and draw it.

    If the agent asked for one, honour that and use the result it was bound to.
    If it did not ask but the result obviously suits a chart, draw one anyway --
    that is usually what the person wanted. Returns (figure or None, a caption).
    """
    is_dark = dark_mode()

    if outcome.chart_request is not None and outcome.chart_result is not None:
        figure, _ = build_chart(
            outcome.chart_result.dataframe,
            chart_type=outcome.chart_request.chart_type,
            x=outcome.chart_request.x,
            y=outcome.chart_request.y,
            title=outcome.chart_request.title,
            dark=is_dark,
        )
        note = ""
        if outcome.chart_is_from_another_query:
            note = ("This chart comes from an earlier query in the steps below, "
                    "not from the SQL shown above -- that is the query which "
                    "contains the charted columns.")
        return figure, note

    # No chart asked for: offer one only if the shape clearly suits it.
    if outcome.result is not None and not outcome.result.is_empty:
        spec = suggest_chart(outcome.result.dataframe)
        if spec.chart_type != "none":
            figure, used = build_chart(outcome.result.dataframe, dark=is_dark)
            return figure, used.reason
    return None, ""


# ---------------------------------------------------------------------------
# The "Ask" tab
# ---------------------------------------------------------------------------

def ask_tab() -> None:
    store: DataStore | None = st.session_state.store

    if store is None:
        st.info(
            "Pick a sample dataset in the sidebar, or upload a CSV or Excel file, "
            "and then ask a question about it in plain English.",
            icon=":material/arrow_back:",
        )
        return

    # Example questions for the sample datasets.
    key = str(st.session_state.store_key or "")
    if key.startswith("sample:"):
        name = key.split(":", 1)[1]
        questions = SAMPLE_DATASETS.get(name, {}).get("questions", [])
        if questions:
            st.caption("Try one of these:")
            columns = st.columns(len(questions))
            for column, question in zip(columns, questions):
                with column:
                    if st.button(question, key=f"example_{abs(hash(question))}",
                                 width="stretch"):
                        st.session_state.pending_question = question
                        st.rerun()

    # The conversation so far.
    for entry in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(entry["question"])
        with st.chat_message("assistant"):
            render_answer(entry)

    # Demo mode caps how many questions one visitor may ask.
    cap = config.max_questions_per_session()
    at_cap = config.demo_mode() and st.session_state.questions_asked >= cap
    if at_cap:
        st.warning(
            f"This demo allows {cap} questions per visit, to keep it free to run. "
            "Reload the page to start again, or run DataPilot locally for unlimited use.",
            icon=":material/hourglass_disabled:",
        )

    typed = st.chat_input(
        "Ask a question about this data...",
        disabled=at_cap,
    )
    question = st.session_state.pending_question or typed
    st.session_state.pending_question = None

    if not question:
        return

    agent = get_agent()
    if agent is None:
        return   # get_agent already explained what is missing

    # Show the question straight away, so the page does not look frozen while
    # the agent works. The real rendering happens from the history on the
    # rerun below, which keeps every answer in one place and in one order.
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Looking at your data, writing SQL, checking the result..."):
            outcome = agent.ask(question)
            figure, note = build_figure_for(outcome)

    st.session_state.history.append({
        "id": len(st.session_state.history),
        "question": question,
        "outcome": outcome,
        "figure": figure,
        "chart_note": note,
    })
    st.session_state.questions_asked += 1

    # Redraw, so the new exchange appears above the input box with the rest of
    # the conversation instead of below it.
    st.rerun()


# ---------------------------------------------------------------------------
# The "Query logs" tab
# ---------------------------------------------------------------------------

def logs_tab() -> None:
    st.markdown("### What the agent has been asked")
    st.caption(
        "Every question is written to `logs/queries.jsonl`. This is what turns "
        '"it worked when I tried it" into numbers I can actually quote.'
    )

    frame = logger.read_log()
    if frame.empty:
        st.info("No questions logged yet. Ask something on the Ask tab.",
                icon=":material/info:")
        return

    stats = logger.summarise(frame)
    columns = st.columns(5)
    columns[0].metric("Questions", f"{stats['total_questions']:,}")
    columns[1].metric("Answered", f"{stats['successful']:,}")
    columns[2].metric("Success rate", f"{stats['success_rate']}%")
    columns[3].metric("Average time", f"{stats['avg_duration_seconds']}s")
    columns[4].metric("Self-corrections", f"{stats['avg_self_corrections']} / question")

    st.divider()

    show_columns = [c for c in
                    ["timestamp", "question", "success", "refused", "self_corrections",
                     "duration_seconds", "sql", "error"]
                    if c in frame.columns]
    st.dataframe(
        frame[show_columns].sort_values("timestamp", ascending=False),
        width="stretch", hide_index=True,
        column_config={
            "timestamp": st.column_config.DatetimeColumn("When", format="D MMM, HH:mm"),
            "question": st.column_config.TextColumn("Question", width="large"),
            "success": st.column_config.CheckboxColumn("OK"),
            "refused": st.column_config.CheckboxColumn("Refused"),
            "self_corrections": st.column_config.NumberColumn("Fixes"),
            "duration_seconds": st.column_config.NumberColumn("Seconds", format="%.1f"),
            "sql": st.column_config.TextColumn("SQL", width="medium"),
        },
    )


# ---------------------------------------------------------------------------
# Putting it together
# ---------------------------------------------------------------------------

def main() -> None:
    init_state()

    st.title("DataPilot")
    st.caption(
        "Ask questions about a spreadsheet in plain English. The agent reads the "
        "table structure, writes SQL, runs it read-only, fixes its own mistakes, "
        "and shows you exactly how it got the answer."
    )

    sidebar()

    ask, logs = st.tabs(["Ask", "Query logs"])
    with ask:
        ask_tab()
    with logs:
        logs_tab()


if __name__ == "__main__":
    main()
