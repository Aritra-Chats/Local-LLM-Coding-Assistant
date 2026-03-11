"""project_synopsis.py — Sentinel LLM-generated project summary.

Samples a representative set of files from the repository, sends them to
the Ollama LLM endpoint, and caches the generated prose summary under the
Sentinel workspace.  Re-generates the summary when source files change.

The synopsis is a ~300-word, human-readable description of what the project
does, its main components, and key entry points.  It is fed into the context
payload as a high-level orientation for every pipeline step.

No external dependencies — stdlib only, plus 'requests' (already required
by rag_engine.py).
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_SENTINEL_HOME = Path(os.environ.get("SENTINEL_HOME", Path.home() / ".sentinel"))
_SYNOPSIS_CACHE_FILE = _SENTINEL_HOME / "index" / "synopsis.json"

_OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
_GENERATE_ENDPOINT = f"{_OLLAMA_BASE_URL}/api/generate"

_DEFAULT_MODEL = "codellama:7b"
_REQUEST_TIMEOUT = 120  # seconds

# How many files to sample for synopsis generation
_MAX_SAMPLE_FILES = 10
_MAX_CHARS_PER_FILE = 1500  # truncate long files before sending

# Extensions to include in sampling
_SAMPLE_EXTENSIONS = {".py", ".md", ".txt", ".js", ".ts"}

# Directories to skip during sampling
_SKIP_DIRS = {"__pycache__", ".venv", "venv", ".git", "node_modules", ".sentinel"}

# System prompt injected before the file samples
_SYSTEM_PROMPT = (
    "You are an expert software architect. Read the following source files "
    "from a software project and write a concise technical summary "
    "(approximately 300 words) covering: (1) what the project does, "
    "(2) its main components and how they interact, (3) key entry points "
    "and important classes or modules. Be precise and avoid filler language."
)


# ---------------------------------------------------------------------------
# ProjectSynopsis
# ---------------------------------------------------------------------------


class ProjectSynopsis:
    """LLM-powered project synopsis generator with file-hash invalidation.

    Caches the synopsis at ``~/.sentinel/index/synopsis.json`` alongside a
    hash of the sampled files.  Regenerates automatically when source files
    change.

    Args:
        project_root: Absolute path to the project root directory.
        model: Ollama model tag used for generation.
    """

    def __init__(
        self,
        project_root: str,
        model: str = _DEFAULT_MODEL,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.model = model
        self._cache: Optional[Dict[str, str]] = None   # {"hash": ..., "text": ...}
        _SYNOPSIS_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, force: bool = False) -> str:
        """Return the project synopsis, regenerating it if necessary.

        Args:
            force: If True, regenerate even if the cache is still valid.

        Returns:
            Synopsis text string, or empty string on failure.
        """
        sample_files = self._select_sample_files()
        current_hash = self._hash_files(sample_files)
        cached = self._load_cache()

        if not force and cached and cached.get("hash") == current_hash:
            return cached.get("text", "")

        text = self._generate(sample_files)
        if text:
            self._save_cache(current_hash, text)

        return text

    def invalidate(self) -> None:
        """Remove the cached synopsis so it will regenerate on next call."""
        if _SYNOPSIS_CACHE_FILE.exists():
            _SYNOPSIS_CACHE_FILE.unlink()
        self._cache = None

    # ------------------------------------------------------------------
    # File sampling
    # ------------------------------------------------------------------

    def _select_sample_files(self) -> List[Path]:
        """Choose a representative set of files to include in the prompt.

        Selection strategy (in priority order):
          1. README.md / README.rst at the project root
          2. Entry-point candidates: main.py, __main__.py, cli*.py, app.py
          3. Largest .py files (by line count) up to the limit

        Returns:
            Ordered list of Path objects, at most ``_MAX_SAMPLE_FILES`` items.
        """
        seen: set = set()
        selected: List[Path] = []

        def _add(p: Path) -> None:
            if p.exists() and p not in seen:
                seen.add(p)
                selected.append(p)

        # 1. Root README
        for name in ("README.md", "README.rst", "README.txt"):
            _add(self.project_root / name)

        # 2. Entry points
        for pattern in ("main.py", "__main__.py", "app.py", "cli*.py"):
            for p in self.project_root.rglob(pattern):
                if not any(part in _SKIP_DIRS for part in p.parts):
                    _add(p)
                    if len(selected) >= 5:
                        break

        # 3. Fill remaining slots with the largest .py files
        if len(selected) < _MAX_SAMPLE_FILES:
            candidates: List[tuple] = []
            for p in self.project_root.rglob("*.py"):
                if any(part in _SKIP_DIRS for part in p.parts):
                    continue
                if p not in seen:
                    try:
                        size = p.stat().st_size
                        candidates.append((size, p))
                    except OSError:
                        pass
            candidates.sort(reverse=True)
            for _, p in candidates:
                if len(selected) >= _MAX_SAMPLE_FILES:
                    break
                _add(p)

        return selected[:_MAX_SAMPLE_FILES]

    # ------------------------------------------------------------------
    # LLM generation
    # ------------------------------------------------------------------

    def _generate(self, sample_files: List[Path]) -> str:
        """Construct the prompt and call the Ollama generation endpoint.

        Args:
            sample_files: Ordered list of files to include in the prompt.

        Returns:
            Generated synopsis string, or empty string on failure.
        """
        if not sample_files:
            return ""

        file_sections: List[str] = []
        for path in sample_files:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
                text = text[:_MAX_CHARS_PER_FILE]
                relative = path.relative_to(self.project_root)
                file_sections.append(f"### {relative}\n```\n{text}\n```")
            except OSError:
                continue

        if not file_sections:
            return ""

        prompt = _SYSTEM_PROMPT + "\n\n" + "\n\n".join(file_sections)

        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }

        try:
            response = requests.post(
                _GENERATE_ENDPOINT,
                json=payload,
                timeout=_REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()
        except (requests.RequestException, KeyError, ValueError):
            return ""

    # ------------------------------------------------------------------
    # Caching helpers
    # ------------------------------------------------------------------

    def _load_cache(self) -> Optional[Dict[str, str]]:
        """Load the synopsis cache from disk.

        Returns:
            Cache dict with keys ``hash`` and ``text``, or None.
        """
        if self._cache is not None:
            return self._cache
        if _SYNOPSIS_CACHE_FILE.exists():
            try:
                raw = _SYNOPSIS_CACHE_FILE.read_text(encoding="utf-8")
                self._cache = json.loads(raw)
                return self._cache
            except (OSError, json.JSONDecodeError):
                return None
        return None

    def _save_cache(self, file_hash: str, text: str) -> None:
        """Persist the synopsis to disk.

        Args:
            file_hash: Content hash of sampled files.
            text: Generated synopsis text.
        """
        self._cache = {"hash": file_hash, "text": text}
        try:
            _SYNOPSIS_CACHE_FILE.write_text(
                json.dumps(self._cache, indent=2),
                encoding="utf-8",
            )
        except OSError:
            pass

    @staticmethod
    def _hash_files(files: List[Path]) -> str:
        """Produce a combined content hash for a list of files.

        Args:
            files: Files to hash.

        Returns:
            Hex digest string (SHA-256).
        """
        h = hashlib.sha256()
        for path in sorted(files):
            try:
                h.update(str(path).encode())
                h.update(path.read_bytes())
            except OSError:
                pass
        return h.hexdigest()
