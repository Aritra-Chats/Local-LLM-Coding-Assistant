"""settings.py — Runtime settings and environment variable helpers."""
from __future__ import annotations
import os
from pathlib import Path


SENTINEL_HOME: Path = Path(os.environ.get("SENTINEL_HOME", Path.home() / ".sentinel"))
SESSIONS_DIR: Path  = Path(os.environ.get("SENTINEL_SESSIONS_DIR", SENTINEL_HOME / "sessions"))
METRICS_DIR: Path   = SENTINEL_HOME / "metrics"
INDEX_DIR: Path     = SENTINEL_HOME / "index"

# ---------------------------------------------------------------------------
# Ollama local model storage (Section 2 / Phase 1)
# All ollama models pulled by Sentinel are stored here so they are isolated
# from the system-default ~/.ollama location.  Override SENTINEL_HOME to
# change the parent; override OLLAMA_MODELS directly to share a model store
# with other tools.
# ---------------------------------------------------------------------------
SENTINEL_OLLAMA_HOME: Path   = SENTINEL_HOME / ".ollama"
SENTINEL_OLLAMA_MODELS: Path = SENTINEL_OLLAMA_HOME / "models"

# Ollama local daemon connection
OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# Ollama Cloud direct API (Section 6B)
# The :cloud suffix is NOT used with this URL; only with local-daemon usage.
OLLAMA_CLOUD_BASE_URL: str = "https://ollama.com"
OLLAMA_API_KEY: str        = os.environ.get("OLLAMA_API_KEY", "")

# ---------------------------------------------------------------------------
# Operating mode (Section 6A)
# Resolved at runtime by the connectivity check + user prompt in main.py.
# Pre-setting SENTINEL_MODE in the environment skips the interactive prompt.
# ---------------------------------------------------------------------------
SENTINEL_MODE: str = os.environ.get("SENTINEL_MODE", "offline").lower()

# ---------------------------------------------------------------------------
# External provider API keys (Section 6A / 6C)
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY: str    = os.environ.get("OPENAI_API_KEY", "")
GOOGLE_API_KEY: str    = os.environ.get("GOOGLE_API_KEY", "")

# External provider model overrides (optional)
SENTINEL_ANTHROPIC_MODEL: str = os.environ.get("SENTINEL_ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
SENTINEL_OPENAI_MODEL: str    = os.environ.get("SENTINEL_OPENAI_MODEL", "gpt-4o")
SENTINEL_GOOGLE_MODEL: str    = os.environ.get("SENTINEL_GOOGLE_MODEL", "gemini-2.0-flash")

# Embedding model
DEFAULT_EMBEDDING_MODEL: str = os.environ.get("SENTINEL_EMBEDDING_MODEL", "nomic-embed-text")

# Context
DEFAULT_TOKEN_BUDGET: int = int(os.environ.get("SENTINEL_TOKEN_BUDGET", "3000"))
