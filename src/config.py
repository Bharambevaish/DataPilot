"""
Central place for every setting in DataPilot.

Where a setting comes from, in order of priority:

1. Streamlit secrets  (st.secrets)  -> used when deployed on Streamlit Cloud
2. Real environment variables       -> used in CI / Docker / `export VAR=...`
3. A local .env file                -> used while developing on my laptop

Nothing secret is ever written in the code. The repository only contains
.env.example, which has placeholder values.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root = the folder that contains this "src" folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env into os.environ if the file exists. Real environment variables
# already set keep priority (override=False is the default).
load_dotenv(PROJECT_ROOT / ".env")

# Useful folder paths, created on demand so a fresh clone just works.
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"
LOGS_DIR = PROJECT_ROOT / "logs"
EVAL_DIR = PROJECT_ROOT / "eval"


def _from_streamlit_secrets(name: str) -> str | None:
    """
    Read a value from st.secrets, returning None if that is not possible.

    This is wrapped in try/except because:
      * streamlit may not be installed in a plain-python context,
      * st.secrets raises if there is no secrets.toml file at all.
    Either way we just fall through to environment variables.
    """
    try:
        import streamlit as st  # imported lazily: the CLI does not need it

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        return None
    return None


def get_setting(name: str, default: str | None = None) -> str | None:
    """Return a setting as text, following the priority order above."""
    value = _from_streamlit_secrets(name)
    if value is not None and value != "":
        return value

    value = os.environ.get(name)
    if value is not None and value != "":
        return value

    return default


def get_int(name: str, default: int) -> int:
    """Return a setting as a whole number, falling back if it is not a number."""
    raw = get_setting(name)
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return default


def get_bool(name: str, default: bool = False) -> bool:
    """Return a setting as True/False. Accepts true/yes/1/on in any case."""
    raw = get_setting(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


# ----- The settings the rest of the app reads -------------------------------

def groq_api_key() -> str | None:
    """The Groq API key, or None if the user has not configured one yet."""
    return get_setting("GROQ_API_KEY")


def model_name() -> str:
    """
    Which Groq model the agent talks to.

    Default is a current Groq production model that supports tool calling
    (checked on https://console.groq.com/docs/models). It is overridable so a
    model being retired never means editing the code.
    """
    return get_setting("MODEL_NAME", "openai/gpt-oss-120b")


def max_agent_steps() -> int:
    """How many tool-calling rounds the agent may take before giving up."""
    return get_int("MAX_AGENT_STEPS", 6)


def max_rows() -> int:
    """Row cap added to every generated query."""
    return get_int("MAX_ROWS", 1000)


def query_timeout_seconds() -> int:
    """How long a single SQL query may run before it is cancelled."""
    return get_int("QUERY_TIMEOUT_SECONDS", 10)


def demo_mode() -> bool:
    """True on the public demo: no uploads, sample data only, question cap."""
    return get_bool("DEMO_MODE", False)


def max_questions_per_session() -> int:
    """Question cap per browser session, used only in demo mode."""
    return get_int("MAX_QUESTIONS_PER_SESSION", 20)


def max_upload_mb() -> int:
    """Largest file a user may upload, in megabytes."""
    return get_int("MAX_UPLOAD_MB", 10)


def is_configured() -> bool:
    """True when there is an API key, so the UI can show a friendly message."""
    return bool(groq_api_key())
