"""ollama_cloud_client.py — Direct HTTP client for the Ollama Cloud API.

Authentication
--------------
All requests carry an ``Authorization: Bearer <OLLAMA_API_KEY>`` header.
Set ``OLLAMA_API_KEY`` in the environment (or ``~/.sentinel/.env``) before
instantiating this client.

API reference: https://docs.ollama.com/cloud

Response normalisation
----------------------
Both :meth:`generate` and :meth:`chat` return a dict with at minimum::

    {"model": str, "response": str, "done": True}

This matches the shape returned by :class:`~models.ollama_client.OllamaClient`
so agents are provider-agnostic.

Important: cloud model tags do NOT use the ":cloud" suffix when calling
this direct API (https://ollama.com).  The ":cloud" suffix is only for
local-daemon usage (``ollama run gpt-oss:120b-cloud``).  Strip it from
any tag before passing it here.

Storage note
------------
Model storage is irrelevant for cloud calls — the inference runs remotely.
The ``OLLAMA_MODELS`` env var (pointing to ``~/.sentinel/.ollama/models/``)
applies only to locally-pulled models; it has no effect on this client.
See ``config.settings.SENTINEL_OLLAMA_MODELS`` for details.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_SENTINEL_HOME = Path(os.environ.get("SENTINEL_HOME", Path.home() / ".sentinel"))
_CLOUD_CACHE   = _SENTINEL_HOME / "cloud_models.json"


class OllamaCloudClient:
    """Direct HTTP client for the Ollama Cloud API (https://ollama.com).

    Parameters
    ----------
    api_key:
        Ollama Cloud API key.  Falls back to the ``OLLAMA_API_KEY``
        environment variable if not provided.
    max_retries:
        Number of retry attempts on transient errors (default 3).
    timeout:
        Per-request timeout in seconds (default 120).

    Raises
    ------
    ValueError:
        If no API key is available at construction time.
    """

    BASE_URL = "https://ollama.com"

    def __init__(
        self,
        api_key: str = "",
        max_retries: int = 3,
        timeout: int = 120,
    ) -> None:
        self.api_key     = api_key or os.environ.get("OLLAMA_API_KEY", "")
        self.max_retries = max_retries
        self.timeout     = timeout

        if not self.api_key:
            raise ValueError(
                "OLLAMA_API_KEY is required for Ollama Cloud access.\n"
                "Generate a key at: https://ollama.com/settings/keys"
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _auth_header(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
        }

    def _post(self, path: str, body: Dict[str, Any], timeout: Optional[int] = None) -> Dict:
        url = f"{self.BASE_URL}{path}"
        data = json.dumps(body).encode("utf-8")
        headers = self._auth_header()
        t = timeout or self.timeout

        last_exc: Optional[Exception] = None
        for _attempt in range(max(1, self.max_retries)):
            try:
                req = urllib.request.Request(url, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=t) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except (urllib.error.URLError, OSError) as exc:
                last_exc = exc
        raise RuntimeError(f"OllamaCloudClient POST {path} failed after retries: {last_exc}")

    def _get(self, path: str, timeout: Optional[int] = None) -> Dict:
        url = f"{self.BASE_URL}{path}"
        headers = self._auth_header()
        # Remove Content-Type for GET
        headers.pop("Content-Type", None)
        t = timeout or self.timeout

        last_exc: Optional[Exception] = None
        for _attempt in range(max(1, self.max_retries)):
            try:
                req = urllib.request.Request(url, headers=headers, method="GET")
                with urllib.request.urlopen(req, timeout=t) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except (urllib.error.URLError, OSError) as exc:
                last_exc = exc
        raise RuntimeError(f"OllamaCloudClient GET {path} failed after retries: {last_exc}")

    @staticmethod
    def _strip_cloud_suffix(tag: str) -> str:
        """Strip the ':cloud' suffix from a model tag if present.

        The ':cloud' suffix is only for local-daemon usage; the direct
        Ollama Cloud API uses the base tag (e.g. 'gpt-oss:120b').
        """
        if tag.endswith(":cloud"):
            return tag[: -len(":cloud")]
        return tag

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_models(self) -> List[Dict]:
        """Return cloud models accessible to this API key.

        Reads from the pre-built probe cache at ``~/.sentinel/cloud_models.json``
        which is populated by ``scripts/cloud_model_probe.py`` (runs daily via
        OS task scheduler, or on demand via ``sentinel refresh``).

        This approach avoids runtime probing delays — the scheduled task does
        the expensive work in the background, and Sentinel reads results
        instantly from the JSON file.

        Returns
        -------
        List[Dict]
            List of model dicts sorted by parameter count (largest first).
            Each dict has at minimum a ``"name"`` key (bare API tag, no
            ``:cloud`` suffix) and a ``"param_billions"`` float.
            Returns an empty list when the cache is absent or expired;
            run ``sentinel refresh`` to rebuild it.
        """
        try:
            raw  = _CLOUD_CACHE.read_text(encoding="utf-8")
            data = json.loads(raw)
            if datetime.now() < datetime.fromisoformat(data["expires_at"]):
                return data.get("models", [])
            # Cache is expired — warn but return stale data rather than nothing
            import warnings
            warnings.warn(
                "Ollama Cloud model cache has expired.  "
                "Run 'sentinel refresh' to update it.",
                stacklevel=2,
            )
            return data.get("models", [])
        except FileNotFoundError:
            # Normal on first run before probe has executed
            return []
        except Exception:
            return []


    def generate(
        self,
        model: str,
        prompt: str,
        stream: bool = False,
        timeout: Optional[int] = None,
        options: Optional[Dict] = None,
        system: Optional[str] = None,
    ) -> Dict:
        """Call ``POST https://ollama.com/api/generate``.

        Args:
            model:   Cloud model tag (no ':cloud' suffix required).
            prompt:  User prompt text.
            stream:  Set to False (streaming not supported in this client).
            timeout: Optional per-request timeout override.
            options: Optional model options dict.
            system:  Optional system prompt string.

        Returns:
            Normalised dict: ``{"model": ..., "response": ..., "done": True}``
        """
        model = self._strip_cloud_suffix(model)
        body: Dict[str, Any] = {"model": model, "prompt": prompt, "stream": False}
        if options:
            body["options"] = options
        if system:
            body["system"] = system

        raw = self._post("/api/generate", body, timeout=timeout)
        # Normalise to match OllamaClient response shape
        return {
            "model":    raw.get("model", model),
            "response": raw.get("response", ""),
            "done":     raw.get("done", True),
            **{k: v for k, v in raw.items() if k not in ("model", "response", "done")},
        }

    def chat(
        self,
        model: str,
        messages: List[Dict],
        stream: bool = False,
        timeout: Optional[int] = None,
    ) -> Dict:
        """Call ``POST https://ollama.com/api/chat``.

        Args:
            model:    Cloud model tag (no ':cloud' suffix required).
            messages: List of ``{"role": ..., "content": ...}`` dicts.
            stream:   Set to False (streaming not supported in this client).
            timeout:  Optional per-request timeout override.

        Returns:
            Normalised dict: ``{"model": ..., "response": ..., "done": True}``
        """
        model = self._strip_cloud_suffix(model)
        body: Dict[str, Any] = {"model": model, "messages": messages, "stream": False}

        raw = self._post("/api/chat", body, timeout=timeout)
        # Extract assistant content from the message field if present
        response_text = (
            raw.get("message", {}).get("content", "")
            or raw.get("response", "")
        )
        return {
            "model":    raw.get("model", model),
            "response": response_text,
            "done":     raw.get("done", True),
            **{k: v for k, v in raw.items() if k not in ("model", "response", "done", "message")},
        }

    def is_available(self) -> bool:
        """Ping ``https://ollama.com/api/tags``.

        Returns:
            ``True`` if reachable and the API key is valid (HTTP 200).
            ``False`` on any error.
        """
        try:
            models = self.list_models()
            return True  # If we got here without exception, it's available
        except Exception:
            return False

    def is_model_available(self, model_tag: str) -> bool:
        """Check that a specific model tag is accessible for inference.

        ``is_available()`` only confirms the API key is valid; it cannot
        detect tier-locked models that return 404 on ``/api/generate``.
        This method makes a minimal single-token probe generate call to
        verify the model is reachable on this plan.

        Returns:
            ``True`` if the model responds without a 404 / auth error.
            ``False`` if the model is tier-locked, unknown, or unavailable.
        """
        tag = self._strip_cloud_suffix(model_tag)
        body = {"model": tag, "prompt": "hi", "stream": False,
                "options": {"num_predict": 1}}
        try:
            self._post("/api/generate", body, timeout=10)
            return True
        except RuntimeError as exc:
            _msg = str(exc).lower()
            # 404 means tier-locked or unknown model — not a transient error
            if "404" in _msg or "not found" in _msg:
                return False
            # Other errors (timeout, 5xx) — treat as unavailable for safety
            return False
        except Exception:
            return False
