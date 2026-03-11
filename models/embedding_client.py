"""embedding_client.py — Embedding generation via Ollama."""
from __future__ import annotations
import json
import urllib.request
from typing import List


class EmbeddingClient:
    """Generate text embeddings using an Ollama embedding model."""

    def __init__(self, model: str = "nomic-embed-text",
                 base_url: str = "http://localhost:11434") -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    def embed(self, text: str) -> List[float]:
        payload = json.dumps({"model": self.model, "prompt": text}).encode()
        req = urllib.request.Request(
            f"{self.base_url}/api/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read()).get("embedding", [])

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]
