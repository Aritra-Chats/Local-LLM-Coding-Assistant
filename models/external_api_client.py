"""external_api_client.py — Unified client for Anthropic, OpenAI, and Google Gemini.

Normalises all provider responses to::

    {"model": str, "response": str, "done": True}

so downstream agents are provider-agnostic.

Provider configuration is driven by environment variables; see
``config/settings.py`` for the full list.

Storage note
------------
This client makes outbound HTTPS calls only; it does not interact with the
local Ollama daemon or ``~/.sentinel/.ollama/models/``.  See
``config.settings.SENTINEL_OLLAMA_MODELS`` for local model storage details.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Optional


class ExternalAPIClient:
    """Unified client for Anthropic, OpenAI, and Google Gemini APIs.

    Parameters
    ----------
    provider:
        One of ``"anthropic"``, ``"openai"``, or ``"google"``.

    Raises
    ------
    AssertionError:
        If *provider* is not in :attr:`PROVIDERS`.
    """

    PROVIDERS: Dict[str, Dict[str, str]] = {
        "anthropic": {
            "key_env":   "ANTHROPIC_API_KEY",
            "model_env": "SENTINEL_ANTHROPIC_MODEL",
            "default":   "claude-sonnet-4-20250514",
            "url":       "https://api.anthropic.com/v1/messages",
        },
        "openai": {
            "key_env":   "OPENAI_API_KEY",
            "model_env": "SENTINEL_OPENAI_MODEL",
            "default":   "gpt-4o",
            "url":       "https://api.openai.com/v1/chat/completions",
        },
        "google": {
            "key_env":   "GOOGLE_API_KEY",
            "model_env": "SENTINEL_GOOGLE_MODEL",
            "default":   "gemini-2.0-flash",
            # {model} is replaced at call time
            "url":       "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        },
    }

    def __init__(self, provider: str) -> None:
        assert provider in self.PROVIDERS, f"Unknown provider: {provider}"
        self.provider = provider
        self._meta = self.PROVIDERS[provider]

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return ``True`` if the provider's API key is set."""
        return bool(os.environ.get(self._meta["key_env"]))

    def _api_key(self) -> str:
        key = os.environ.get(self._meta["key_env"], "")
        if not key:
            raise RuntimeError(
                f"[ExternalAPIClient] {self._meta['key_env']} is not set. "
                f"Provider '{self.provider}' is unavailable."
            )
        return key

    def _model(self, model: str = "") -> str:
        if model:
            return model
        return (
            os.environ.get(self._meta["model_env"], "")
            or self._meta["default"]
        )

    # ------------------------------------------------------------------
    # Unified generate
    # ------------------------------------------------------------------

    def generate(
        self,
        model: str = "",
        prompt: str = "",
        system: str = "",
        stream: bool = False,
        timeout: int = 120,
    ) -> Dict:
        """Call the provider API and return a normalised response dict.

        Args:
            model:   Provider model ID (falls back to env var / default).
            prompt:  User prompt text.
            system:  Optional system prompt.
            stream:  Ignored (streaming not supported in this client).
            timeout: Per-request timeout in seconds.

        Returns:
            ``{"model": str, "response": str, "done": True}``
        """
        resolved_model = self._model(model)

        if self.provider == "anthropic":
            return self._anthropic(resolved_model, prompt, system, timeout)
        elif self.provider == "openai":
            return self._openai(resolved_model, prompt, system, timeout)
        elif self.provider == "google":
            return self._google(resolved_model, prompt, system, timeout)
        raise RuntimeError(f"Unknown provider: {self.provider}")

    # ------------------------------------------------------------------
    # Provider-specific implementations
    # ------------------------------------------------------------------

    def _post(self, url: str, headers: Dict[str, str], body: Dict, timeout: int) -> Dict:
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"[ExternalAPIClient/{self.provider}] HTTP {exc.code}: {body_text}"
            ) from exc

    def _anthropic(self, model: str, prompt: str, system: str, timeout: int) -> Dict:
        api_key = self._api_key()
        headers = {
            "x-api-key":         api_key,
            "anthropic-version":  "2023-06-01",
            "content-type":       "application/json",
        }
        body: Dict[str, Any] = {
            "model":      model,
            "max_tokens": 4096,
            "messages":   [{"role": "user", "content": prompt}],
        }
        if system:
            body["system"] = system

        raw = self._post(self._meta["url"], headers, body, timeout)
        # Anthropic returns content as a list of blocks
        content_blocks = raw.get("content", [])
        response_text = " ".join(
            b.get("text", "") for b in content_blocks if b.get("type") == "text"
        )
        return {"model": model, "response": response_text, "done": True}

    def _openai(self, model: str, prompt: str, system: str, timeout: int) -> Dict:
        api_key = self._api_key()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        }
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        body = {"model": model, "messages": messages}

        raw = self._post(self._meta["url"], headers, body, timeout)
        choices = raw.get("choices", [])
        response_text = choices[0]["message"]["content"] if choices else ""
        return {"model": model, "response": response_text, "done": True}

    def _google(self, model: str, prompt: str, system: str, timeout: int) -> Dict:
        api_key = self._api_key()
        url = self._meta["url"].format(model=model) + f"?key={api_key}"
        headers = {"Content-Type": "application/json"}
        contents: list = []
        if system:
            contents.append({
                "role": "user",
                "parts": [{"text": f"[System context]: {system}"}],
            })
        contents.append({"role": "user", "parts": [{"text": prompt}]})
        body = {"contents": contents}

        raw = self._post(url, headers, body, timeout)
        try:
            response_text = (
                raw["candidates"][0]["content"]["parts"][0]["text"]
            )
        except (KeyError, IndexError):
            response_text = ""
        return {"model": model, "response": response_text, "done": True}
