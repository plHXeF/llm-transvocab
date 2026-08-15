"""Application defaults.

Secrets are deliberately read only from the environment or the local settings
store.  Never add an API key literal to this module.
"""

from __future__ import annotations

import os
from pathlib import Path


# --- OpenAI-compatible endpoint defaults ---------------------------------

# The application has one endpoint model. Legacy variables remain readable so
# existing local installations upgrade without losing their configuration.
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

DEFAULT_BASE_URL = os.getenv("VOCAB_BASE_URL", DEEPSEEK_BASE_URL).strip()
API_KEY = os.getenv("VOCAB_API_KEY", "")
if not API_KEY:
    normalized_default_url = DEFAULT_BASE_URL.rstrip("/")
    if normalized_default_url == OPENAI_BASE_URL.rstrip("/"):
        API_KEY = OPENAI_API_KEY
    elif normalized_default_url == DEEPSEEK_BASE_URL.rstrip("/"):
        API_KEY = DEEPSEEK_API_KEY

# Models are intentionally not hard-coded. The GUI loads /models from the
# configured endpoint; this value is only an optional startup/manual fallback.
DEFAULT_MODEL = os.getenv("VOCAB_MODEL", "").strip()

# Values are normalized and validated by ``llm_service.LLMSettings``.
DEFAULT_REASONING_EFFORT = os.getenv(
    "VOCAB_REASONING_EFFORT", "disabled"
).strip().lower()

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

APP_SETTINGS_FILE = Path(
    os.getenv(
        "VOCAB_SETTINGS_FILE",
        str(DATA_DIR / "app_settings.json"),
    )
).expanduser()
API_KEYS_FILE = Path(
    os.getenv(
        "VOCAB_API_KEYS_FILE",
        os.getenv(
            "VOCAB_PROVIDER_KEYS_FILE",
            str(DATA_DIR / "api_keys.json"),
        ),
    )
).expanduser()
LEGACY_PROVIDER_KEYS_FILE = DATA_DIR / "provider_keys.json"
# Deprecated alias for integrations that imported the old constant.
PROVIDER_KEYS_FILE = API_KEYS_FILE
LEARNING_DB_FILE = Path(
    os.getenv(
        "VOCAB_LEARNING_DB_FILE",
        str(DATA_DIR / "learning.db"),
    )
).expanduser()
MODEL_ERROR_LOG_FILE = Path(
    os.getenv(
        "VOCAB_MODEL_ERROR_LOG_FILE",
        str(DATA_DIR / "model_errors.jsonl"),
    )
).expanduser()


# --- Vocabulary data ------------------------------------------------------

VOCAB_FILE = os.getenv("VOCAB_FILE", str(PROJECT_ROOT / "vocabularies.csv"))


# --- Streamlit page -------------------------------------------------------

PAGE_TITLE = "英语词汇学习系统"
PAGE_ICON = "📚"
PAGE_LAYOUT = "wide"
