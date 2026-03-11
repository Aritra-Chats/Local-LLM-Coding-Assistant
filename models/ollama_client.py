"""ollama_client.py — HTTP client wrapper for the Ollama local API."""
from __future__ import annotations
import json
import urllib.request
from typing import Any, Dict, Generator, Optional


class OllamaClient:
    """Minimal synchronous client for the Ollama REST API."""

    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        self.base_url = base_url.rstrip("/")

    def generate(self, model: str, prompt: str, stream: bool = False) -> Dict[str, Any]:
        payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())

    def list_models(self) -> list[str]:
        with urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=10) as resp:
            data = json.loads(resp.read())
            return [m["name"] for m in data.get("models", [])]

    def is_available(self) -> bool:
        try:
            self.list_models()
            return True
        except Exception:
            return False
