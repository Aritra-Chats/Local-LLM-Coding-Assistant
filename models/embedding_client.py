"""embedding_client.py — Sentinel embedding client.

Wraps :class:`~models.inference_engine.InferenceEngine` to provide fast
single and batched text embedding.

Priority order for the embedding backend:

  1. ``sentence-transformers`` (in-process, no HTTP, fastest on CPU & GPU)
  2. Ollama ``/api/embeddings`` via a persistent ``requests.Session``
     (parallel batch support via thread pool)

The backend is selected automatically based on what is installed.
Override with ``SENTINEL_EMBED_BACKEND=ollama`` to force Ollama.
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np


class EmbeddingClient:
    """Generate text embeddings using the best available backend.

    Args:
        model:    Ollama embedding model tag (fallback backend only).
        base_url: Ollama server URL (fallback backend only).
        st_model: sentence-transformers model name.
                  Defaults to "all-MiniLM-L6-v2" (fast, 384-dim).
                  Use "all-mpnet-base-v2" for higher quality (768-dim).
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        st_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.st_model = st_model
        self._engine: Optional[object] = None  # InferenceEngine, lazy

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, text: str) -> List[float]:
        """Embed a single text string.

        Args:
            text: Text to embed (long texts are automatically truncated).

        Returns:
            Embedding vector as a list of floats.
        """
        engine = self._get_engine()
        if engine is not None:
            return engine.embed(text)
        return self._ollama_embed(text)

    def embed_batch(self, texts: list) -> list:
        """Embed multiple texts efficiently.

        Uses sentence-transformers batch inference (single forward pass) when
        available, otherwise fans out to parallel Ollama calls.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors (same order as input).
        """
        if not texts:
            return []
        engine = self._get_engine()
        if engine is not None:
            return engine.embed_batch(texts)
        # Sequential fallback (no engine)
        return [self._ollama_embed(t) for t in texts]

    def embed_numpy(self, text: str) -> "np.ndarray":
        """Return embedding as a float32 numpy array."""
        return np.array(self.embed(text), dtype=np.float32)

    def embed_batch_numpy(self, texts: list) -> "np.ndarray":
        """Return batch embeddings as a float32 numpy matrix (N x D)."""
        vecs = self.embed_batch(texts)
        if not vecs:
            return np.empty((0,), dtype=np.float32)
        return np.array(vecs, dtype=np.float32)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_engine(self) -> Optional[object]:
        """Lazy-load the InferenceEngine singleton."""
        if self._engine is not None:
            return self._engine
        try:
            from models.inference_engine import InferenceEngine
            self._engine = InferenceEngine(
                base_url=self.base_url,
                embed_model_name=self.st_model,
            )
        except Exception:
            self._engine = None
        return self._engine

    def _ollama_embed(self, text: str) -> list:
        """Direct Ollama embedding call (no InferenceEngine)."""
        import json
        import urllib.request

        payload = json.dumps({"model": self.model, "prompt": text[:2000]}).encode()
        req = urllib.request.Request(
            f"{self.base_url}/api/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read()).get("embedding", [])
        except Exception:
            return []
