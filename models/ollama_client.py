"""ollama_client.py — HTTP client wrapper for the Ollama local API.

Storage note
------------
Sentinel directs all locally-pulled Ollama models to
``~/.sentinel/.ollama/models/`` by setting the ``OLLAMA_MODELS`` environment
variable before any subprocess or daemon call.  This path is defined in
``config.settings.SENTINEL_OLLAMA_MODELS`` and injected in
``core.bootstrap.Bootstrap.run()``.  This client communicates with the local
Ollama daemon at ``http://localhost:11434``; the daemon reads ``OLLAMA_MODELS``
at startup, so no changes to this HTTP client are required.

Improvements
------------
* Streaming support  — generate_stream() yields text chunks; generate() can
  also consume the stream internally and return an assembled dict.
* Automatic retries  — configurable max_retries with exponential back-off so
  transient model crashes / 503s are handled gracefully.
* Timeout recovery   — on TimeoutError / URLError the call is retried rather
  than immediately propagating to the caller.
* Response normalisation — "response" key always present in returned dicts.
* Async support      — generate_async() / chat_async() run the synchronous
  calls in a thread pool via asyncio so council mode can fire multiple model
  calls truly concurrently without blocking the event loop.
"""
from __future__ import annotations

import asyncio
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterator, List, Optional

# Shared executor for async wrappers (sized for RTX 3050 / 16 GB RAM —
# 3 parallel model calls is safe without GPU VRAM contention)
_ASYNC_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ollama_async")


def _backoff(attempt: int, base: float = 2.0, cap: float = 30.0) -> float:
    return min(base ** attempt, cap)


class OllamaClient:
    """Synchronous + streaming client for the Ollama REST API.

    Args:
        base_url:    Ollama server base URL.
        max_retries: Retry attempts before raising (default 3).
        timeout:     Default socket timeout in seconds (default 120).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        max_retries: int = 3,
        timeout: int = 120,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.timeout = timeout

    # ------------------------------------------------------------------
    # generate — primary call used by all agents
    # ------------------------------------------------------------------

    def generate(
        self,
        model: str,
        prompt: str,
        stream: bool = False,
        timeout: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
        system: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a completion from Ollama and return the full response dict.

        When stream=False (default) blocks until completion.
        When stream=True internally consumes the streaming endpoint and
        assembles the chunks so callers always get the same dict shape.

        Returns:
            Dict with at least "model", "response", "done" keys.
        """
        if stream:
            chunks: List[str] = []
            last: Dict[str, Any] = {}
            for chunk in self.generate_stream(
                model=model, prompt=prompt, timeout=timeout,
                options=options, system=system,
            ):
                text = chunk.get("response", "")
                if text:
                    chunks.append(text)
                last = chunk
            last["response"] = "".join(chunks)
            return last

        body: Dict[str, Any] = {"model": model, "prompt": prompt, "stream": False}
        if options:
            body["options"] = options
        if system:
            body["system"] = system
        return self._post_with_retry("/api/generate", body, timeout=timeout or self.timeout)

    # ------------------------------------------------------------------
    # generate_stream — yields incremental chunks
    # ------------------------------------------------------------------

    def generate_stream(
        self,
        model: str,
        prompt: str,
        timeout: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
        system: Optional[str] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Yield incremental generation chunks from Ollama.

        Each yielded dict has "response" (token text) and "done" keys.
        The final chunk has done=true.

        Raises:
            RuntimeError: When all retry attempts are exhausted.
        """
        body: Dict[str, Any] = {"model": model, "prompt": prompt, "stream": True}
        if options:
            body["options"] = options
        if system:
            body["system"] = system

        payload = json.dumps(body).encode()
        _timeout = timeout or self.timeout
        attempt = 0

        while True:
            req = urllib.request.Request(
                f"{self.base_url}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=_timeout) as resp:
                    for raw_line in resp:
                        line = raw_line.decode("utf-8").strip()
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                            yield chunk
                            if chunk.get("done"):
                                return
                        except json.JSONDecodeError:
                            continue
                return
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                if attempt >= self.max_retries:
                    raise RuntimeError(
                        f"OllamaClient.generate_stream: {self.max_retries + 1} "
                        f"attempts failed for model '{model}'. Last error: {exc}"
                    ) from exc
                time.sleep(_backoff(attempt))
                attempt += 1

    # ------------------------------------------------------------------
    # chat — multi-turn messages
    # ------------------------------------------------------------------

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool = False,
        timeout: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Send a chat-style messages list to Ollama /api/chat.

        Returns:
            Dict with "message" key containing the assistant reply dict.
        """
        body: Dict[str, Any] = {"model": model, "messages": messages, "stream": False}
        if options:
            body["options"] = options

        if stream:
            payload = json.dumps({**body, "stream": True}).encode()
            req = urllib.request.Request(
                f"{self.base_url}/api/chat",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            chunks: List[str] = []
            last: Dict[str, Any] = {}
            _timeout = timeout or self.timeout
            with urllib.request.urlopen(req, timeout=_timeout) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        content = chunk.get("message", {}).get("content", "")
                        if content:
                            chunks.append(content)
                        last = chunk
                    except json.JSONDecodeError:
                        continue
            if "message" not in last:
                last["message"] = {}
            last["message"]["content"] = "".join(chunks)
            return last

        return self._post_with_retry("/api/chat", body, timeout=timeout or self.timeout)

    # ------------------------------------------------------------------
    # list_models / is_available
    # ------------------------------------------------------------------

    def list_models(self) -> List[str]:
        """Return names of all locally available Ollama models."""
        with urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=10) as resp:
            data = json.loads(resp.read())
            return [m["name"] for m in data.get("models", [])]

    def is_available(self) -> bool:
        """Return True when the Ollama server is reachable."""
        try:
            self.list_models()
            return True
        except Exception:
            return False

    def best_available_model(
        self, preferred: List[str], fallback: str = "mistral:7b"
    ) -> str:
        """Return the first preferred model that is locally installed.

        Args:
            preferred: Ordered list of desired model tags (most preferred first).
            fallback:  Returned when none of the preferred models are installed.
        """
        try:
            available = set(self.list_models())
            for m in preferred:
                if m in available:
                    return m
        except Exception:
            pass
        return fallback

    # ------------------------------------------------------------------
    # Async wrappers — for parallel council mode
    # ------------------------------------------------------------------

    async def generate_async(
        self,
        model: str,
        prompt: str,
        timeout: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
        system: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Async wrapper around :meth:`generate`.

        Runs the synchronous HTTP call in a thread-pool executor so that
        multiple ``generate_async`` calls can be awaited concurrently via
        ``asyncio.gather`` without blocking the event loop.

        Example (parallel council calls)::

            results = await asyncio.gather(
                client.generate_async(model_a, prompt_a),
                client.generate_async(model_b, prompt_b),
                client.generate_async(model_c, prompt_c),
            )

        Returns:
            Same dict shape as :meth:`generate`.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _ASYNC_EXECUTOR,
            lambda: self.generate(
                model=model,
                prompt=prompt,
                timeout=timeout,
                options=options,
                system=system,
            ),
        )

    async def chat_async(
        self,
        model: str,
        messages: List[Dict[str, str]],
        timeout: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Async wrapper around :meth:`chat`.

        Runs the synchronous call in the shared thread pool so that multiple
        chat calls can proceed concurrently.

        Returns:
            Same dict shape as :meth:`chat`.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _ASYNC_EXECUTOR,
            lambda: self.chat(
                model=model,
                messages=messages,
                timeout=timeout,
                options=options,
            ),
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _post_with_retry(
        self,
        path: str,
        body: Dict[str, Any],
        timeout: int,
    ) -> Dict[str, Any]:
        """POST body to path with automatic retry / exponential back-off.

        *timeout* is used both as the per-socket-read deadline and as the
        overall wall-clock budget for a single attempt.  A background thread
        closes the connection if the wall-clock limit is exceeded so that
        very slow (or hung) models cannot block indefinitely.

        Raises:
            RuntimeError: When all retry attempts are exhausted.
        """
        import threading as _threading

        payload = json.dumps(body).encode()
        url = f"{self.base_url}{path}"
        attempt = 0
        last_exc: Exception = RuntimeError("unknown")

        while True:
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
            )

            # --- wall-clock watchdog ---
            # urllib's socket timeout resets on every successful recv(), so a
            # model that trickles data (or holds the connection open while
            # "thinking") can exceed the intended timeout by a large margin.
            # We use a threading.Timer to forcibly interrupt after `timeout`s.
            _resp_holder: Dict[str, Any] = {}
            _exc_holder:  Dict[str, Exception] = {}
            _conn_holder: list = []  # holds the urlopen response object

            def _do_request() -> None:
                try:
                    resp = urllib.request.urlopen(req, timeout=timeout)
                    _conn_holder.append(resp)
                    raw = resp.read()
                    result: Dict[str, Any] = json.loads(raw)
                    if "response" not in result and "message" in result:
                        result["response"] = result["message"].get("content", "")
                    elif "response" not in result:
                        result["response"] = ""
                    _resp_holder["result"] = result
                except Exception as exc:  # noqa: BLE001
                    _exc_holder["exc"] = exc

            worker = _threading.Thread(target=_do_request, daemon=True)
            worker.start()
            # Give the request slightly longer than the socket timeout so the
            # socket timeout fires first under normal slow-network conditions;
            # the hard wall-clock cut-off is 20 s more than the declared timeout.
            wall_limit = timeout + 20
            worker.join(wall_limit)

            if worker.is_alive():
                # Request exceeded wall-clock limit — close the response socket
                # to unblock the worker thread, then treat it as a timeout error.
                for _r in _conn_holder:
                    try:
                        _r.close()
                    except Exception:
                        pass
                worker.join(2)  # brief grace period
                last_exc = TimeoutError(
                    f"OllamaClient: wall-clock timeout ({wall_limit}s) exceeded "
                    f"for {path} (model={body.get('model', '?')})"
                )
            elif "result" in _resp_holder:
                return _resp_holder["result"]
            elif "exc" in _exc_holder:
                exc = _exc_holder["exc"]
                if isinstance(exc, urllib.error.HTTPError) and 400 <= exc.code < 500:
                    raise RuntimeError(
                        f"OllamaClient HTTP {exc.code} on {path}: {exc.reason}"
                    ) from exc
                last_exc = exc
            # else: thread finished without result or exc — treat as error
            # and fall through to retry logic

            if attempt >= self.max_retries:
                raise RuntimeError(
                    f"OllamaClient: all {self.max_retries + 1} attempts failed "
                    f"for {path} (model={body.get('model', '?')}). "
                    f"Last error: {last_exc}"
                ) from last_exc

            time.sleep(_backoff(attempt))
            attempt += 1
