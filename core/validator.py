"""validator.py — Pipeline and step output validation utilities."""
from __future__ import annotations
from typing import Any, Dict, List


def validate_step_output(output: Dict[str, Any]) -> bool:
    """Basic structural check on a step result dict."""
    return isinstance(output, dict) and "status" in output


def validate_pipeline(steps: List[Dict[str, Any]]) -> bool:
    """Check pipeline list is non-empty and each step has required keys."""
    if not steps:
        return False
    required = {"name", "agent"}
    return all(required.issubset(s.keys()) for s in steps)
