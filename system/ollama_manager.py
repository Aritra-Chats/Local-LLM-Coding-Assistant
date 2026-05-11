"""ollama_manager.py — Ollama installation, startup, and model management.

Uses the system Ollama CLI and default Ollama storage directories.
"""
from __future__ import annotations
import shutil
import subprocess
from typing import List


def is_ollama_installed() -> bool:
    return shutil.which("ollama") is not None


def pull_model(model: str) -> bool:
    """Pull an Ollama model into the Sentinel model store.

    Returns True on success.
    """
    result = subprocess.run(
        ["ollama", "pull", model],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def list_local_models() -> List[str]:
    result = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []
    lines = result.stdout.strip().splitlines()[1:]  # skip header
    return [line.split()[0] for line in lines if line.strip()]


def is_model_available(model: str) -> bool:
    return model in list_local_models()
