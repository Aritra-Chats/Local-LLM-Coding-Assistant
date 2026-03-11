"""settings.py — Runtime settings and environment variable helpers."""
from __future__ import annotations
import os
from pathlib import Path


SENTINEL_HOME: Path = Path(os.environ.get("SENTINEL_HOME", Path.home() / ".sentinel"))
SESSIONS_DIR: Path  = Path(os.environ.get("SENTINEL_SESSIONS_DIR", SENTINEL_HOME / "sessions"))
METRICS_DIR: Path   = SENTINEL_HOME / "metrics"
INDEX_DIR: Path     = SENTINEL_HOME / "index"

# Ollama connection
OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# Embedding model
DEFAULT_EMBEDDING_MODEL: str = os.environ.get("SENTINEL_EMBEDDING_MODEL", "nomic-embed-text")

# Context
DEFAULT_TOKEN_BUDGET: int = int(os.environ.get("SENTINEL_TOKEN_BUDGET", "3000"))
