"""retry_handler.py — Retry back-off logic for failed pipeline steps."""
from __future__ import annotations
import time
from typing import Any, Dict


_MAX_BACKOFF: float = 30.0


def backoff(attempt: int, base: float = 2.0) -> float:
    """Exponential back-off (capped at _MAX_BACKOFF seconds)."""
    return min(base ** attempt, _MAX_BACKOFF)


def should_retry(step: Dict[str, Any], retries_used: int) -> bool:
    """Return True if the step has remaining retry budget."""
    return retries_used < int(step.get("max_retries", 2))
