"""
Tests for the Streamlit app (Phase 4 and the demo-mode limits of Phase 6).

    python tests/test_app.py

These use Streamlit's own AppTest harness, which runs app.py headlessly --
no browser, no API key, and no LLM call, because none of the behaviour tested
here needs the agent to answer anything. That makes it fast and deterministic,
unlike driving a real browser against a rate-limited API.

What is checked:
  * demo mode really does hide uploads, show the banner, and cap questions
  * normal mode really does offer uploads
  * with no data loaded the app asks you to pick some, rather than erroring
  * the sidebar reports the tables that are loaded
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from streamlit.testing.v1 import AppTest   # noqa: E402

APP = str(PROJECT_ROOT / "app.py")

PASSED = 0
FAILED = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  PASS  {label}")
    else:
        FAILED += 1
        print(f"  FAIL  {label}  {detail}")


def all_text(app: AppTest) -> str:
    """Everything the page says, as one lower-case string."""
    pieces: list[str] = []
    for collection in (app.markdown, app.info, app.warning, app.error,
                       app.caption, app.success):
        pieces.extend(element.value for element in collection)
    return " ".join(str(p) for p in pieces).lower()


def set_environment(**values: str) -> None:
    """Set the environment variables app.py reads through src.config."""
    for key, value in values.items():
        os.environ[key] = value


# ---------------------------------------------------------------------------

def test_runs_without_an_api_key() -> None:
    print("\n1. The app starts even with no API key configured")
    set_environment(DEMO_MODE="false", GROQ_API_KEY="")
    app = AppTest.from_file(APP, default_timeout=60).run()

    check("the app runs without an exception", not app.exception,
          str(app.exception))
    check("no stack trace is shown to the user",
          "traceback" not in all_text(app))
    # The key is only needed to ANSWER a question, so the page must still
    # render and the data must still load without one.
    check("the data still loads", app.session_state["store"] is not None)


def test_missing_sample_files_explain_themselves() -> None:
    print("\n1a. If the sample files are missing, the app says so plainly")
    set_environment(DEMO_MODE="false")

    from src import config
    sales = config.SAMPLES_DIR / "sales.csv"
    hidden = sales.with_suffix(".csv.hidden-for-test")
    if not sales.exists():
        check("sales.csv is present to begin with", False)
        return

    sales.rename(hidden)
    try:
        app = AppTest.from_file(APP, default_timeout=60).run()
        text = all_text(app)
        check("it does not crash", not app.exception, str(app.exception))
        check("it says the sample data is missing",
              "sample data is missing" in text, text[:200])
        check("it tells you how to fix it",
              "make_samples.py" in text, text[:200])
    finally:
        hidden.rename(sales)


def test_sample_data_loads_on_startup() -> None:
    print("\n1b. A sample dataset is open as soon as the app starts")
    set_environment(DEMO_MODE="false")
    app = AppTest.from_file(APP, default_timeout=60).run()

    store = app.session_state["store"]
    check("a dataset is loaded without anyone clicking", store is not None)
    check("no autoload error", not app.session_state["autoload_error"],
          str(app.session_state["autoload_error"]))
    if store is not None:
        check("it is the sales sample", "sales" in store.table_names(),
              str(store.table_names()))
        check("the rows are really there", store.total_rows() > 4000,
              str(store.total_rows()))
    check("the chat input is ready immediately", bool(app.chat_input))
    check("it does not ask you to pick a dataset first",
          "pick a sample dataset" not in all_text(app))


def test_demo_hides_other_visitors_questions() -> None:
    print("\n1c. The public demo does not show visitors each other's questions")
    set_environment(DEMO_MODE="true", MAX_QUESTIONS_PER_SESSION="5")

    # Put a question in the log, as another visitor would have done.
    from src import logger
    logger.clear_log()
    logger.log_question(question="a previous visitor's private question",
                        answer="42", sql="SELECT 1", success=True)
    try:
        app = AppTest.from_file(APP, default_timeout=60).run()
        # Switch to the logs tab by rendering it: AppTest renders all tabs.
        text = all_text(app)
        check("the other visitor's question is not shown",
              "previous visitor's private question" not in text)
        check("it explains why", "not listed" in text, text[-200:])

        # ...but with demo mode off, the history is shown as normal.
        set_environment(DEMO_MODE="false")
        app = AppTest.from_file(APP, default_timeout=60).run()
        shown = any("previous visitor" in str(df.value.to_string())
                    for df in app.dataframe)
        check("running locally, your own history IS shown", shown)
    finally:
        logger.clear_log()


def test_normal_mode_allows_uploads() -> None:
    print("\n2. Normal mode offers file upload")
    set_environment(DEMO_MODE="false")
    app = AppTest.from_file(APP, default_timeout=60).run()

    text = all_text(app)
    check("the upload section is offered", "or upload your own" in text)
    check("no public-demo banner is shown", "public demo" not in text)


def test_demo_mode_restrictions() -> None:
    print("\n3. Demo mode hides uploads and shows the banner")
    set_environment(DEMO_MODE="true", MAX_QUESTIONS_PER_SESSION="3")
    app = AppTest.from_file(APP, default_timeout=60).run()

    text = all_text(app)
    check("the public-demo banner is shown", "public demo" in text, text[:160])
    check("it warns against uploading private data",
          "private data" in text, text[:160])
    check("the upload section is hidden", "or upload your own" not in text)


def test_demo_mode_question_cap() -> None:
    print("\n4. Demo mode stops accepting questions at the cap")
    set_environment(DEMO_MODE="true", MAX_QUESTIONS_PER_SESSION="3")

    app = AppTest.from_file(APP, default_timeout=60)
    app.run()

    # No clicking needed: a dataset is already open at startup. The button for
    # the dataset currently loaded is deliberately disabled, which is itself
    # worth asserting -- it is how a visitor sees which one they are on.
    check("the dataset is already loaded", app.session_state["store"] is not None)
    current = [b for b in app.button if "Sales" in str(b.label)]
    check("the button for the loaded dataset is disabled",
          bool(current) and current[0].disabled)
    check("the chat input is available before the cap",
          bool(app.chat_input) and not app.chat_input[0].disabled)
    check("no cap warning before the cap",
          "questions per visit" not in all_text(app))

    # Pretend the visitor has already asked three questions.
    app.session_state["questions_asked"] = 3
    app.run()

    text = all_text(app)
    check("the cap message appears at the limit",
          "allows 3 questions per visit" in text, text[-200:])
    check("the chat input is disabled at the cap",
          bool(app.chat_input) and app.chat_input[0].disabled)

    # One under the cap, it must still work -- an off-by-one here would lock
    # people out a question early.
    app.session_state["questions_asked"] = 2
    app.run()
    check("one question under the cap is still allowed",
          bool(app.chat_input) and not app.chat_input[0].disabled)


def test_sidebar_reports_loaded_tables() -> None:
    print("\n5. The sidebar reports what is loaded")
    set_environment(DEMO_MODE="false")
    app = AppTest.from_file(APP, default_timeout=60)
    app.run()

    # Switching away from the startup dataset: that button is enabled.
    shop = [b for b in app.button if "Shop orders" in str(b.label)]
    check("the other sample's button is offered and enabled",
          bool(shop) and not shop[0].disabled)
    if not shop:
        return
    shop[0].click().run()

    text = all_text(app)
    check("both sheets became tables",
          "customers" in text and "orders" in text, text[:200])
    check("row counts are shown", "300 rows" in text and "2,000 rows" in text,
          text[:300])


def main() -> int:
    print("=" * 74)
    print("DataPilot app tests (headless Streamlit, no API key needed)")
    print("=" * 74)

    try:
        test_runs_without_an_api_key()
        test_missing_sample_files_explain_themselves()
        test_sample_data_loads_on_startup()
        test_demo_hides_other_visitors_questions()
        test_normal_mode_allows_uploads()
        test_demo_mode_restrictions()
        test_demo_mode_question_cap()
        test_sidebar_reports_loaded_tables()
    finally:
        os.environ.pop("DEMO_MODE", None)
        os.environ.pop("MAX_QUESTIONS_PER_SESSION", None)

    print("\n" + "=" * 74)
    print(f"{PASSED} passed, {FAILED} failed")
    print("=" * 74)
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
