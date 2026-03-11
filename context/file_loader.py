"""file_loader.py — Sentinel @file: attachment loader.

Reads a file from the local filesystem and returns a structured
attachment dict ready for injection into the context payload.

Handles text and binary files gracefully:
- Text files are returned as UTF-8 (with lossy decoding on errors).
- Binary files have their raw bytes base64-encoded and are flagged
  as binary so downstream systems can handle them appropriately.

Supports glob patterns so ``@file:src/**/*.py`` expands to multiple
attachments.

No external dependencies — stdlib only.
"""

import base64
import glob
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List

# Maximum file size to load in full (10 MB).  Larger files are truncated.
_MAX_TEXT_BYTES = 10 * 1024 * 1024

# Extensions always treated as text regardless of MIME detection.
_TEXT_EXTENSIONS: frozenset = frozenset(
    {
        ".py", ".js", ".ts", ".json", ".yaml", ".yml", ".toml", ".ini",
        ".cfg", ".md", ".rst", ".txt", ".html", ".css", ".sh", ".bat",
        ".ps1", ".go", ".rs", ".java", ".c", ".cpp", ".h", ".hpp",
        ".xml", ".sql", ".graphql", ".env",
    }
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load(path: str, project_root: str = "") -> List[Dict[str, Any]]:
    """Load one or more files matching *path* (glob-expanded).

    Args:
        path: Absolute, project-relative, or glob file path.
        project_root: Optional project root for resolving relative paths.

    Returns:
        List of attachment dicts (one per matched file).  Each dict has:
            - ``type``        — always ``"file"``
            - ``path``        — resolved absolute file path
            - ``relative_path`` — path relative to project_root (if given)
            - ``language``    — language hint string (e.g. ``"python"``)
            - ``content``     — file text (str) or base64 bytes (str)
            - ``encoding``    — ``"text"`` or ``"base64"``
            - ``size_bytes``  — original file size
            - ``truncated``   — True if content was truncated at size limit
    """
    resolved_paths = _resolve_paths(path, project_root)
    return [_load_single(p, project_root) for p in resolved_paths]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_paths(path: str, project_root: str) -> List[Path]:
    """Expand a path or glob pattern to a list of existing file paths.

    Args:
        path: Raw path string, may include glob wildcards.
        project_root: Base directory for relative path resolution.

    Returns:
        Sorted list of resolved Path objects.
    """
    # Normalise separators
    path = path.replace("\\", "/")

    raw = Path(path)
    if raw.is_absolute():
        candidate = raw
    elif project_root:
        candidate = Path(project_root) / path
    else:
        candidate = Path.cwd() / path

    # Glob expansion
    if any(c in path for c in ("*", "?", "[")):
        base = Path(project_root) if project_root and not Path(path).is_absolute() else Path.cwd()
        matches = [Path(p) for p in glob.glob(str(base / path), recursive=True)]
        return sorted(f for f in matches if f.is_file())

    return [candidate] if candidate.is_file() else []


def _load_single(path: Path, project_root: str) -> Dict[str, Any]:
    """Load a single file and return its attachment dict.

    Args:
        path: Absolute Path object.
        project_root: Project root for computing relative path.

    Returns:
        Attachment dict.
    """
    size = path.stat().st_size
    language = _detect_language(path)
    rel = _relative(path, project_root)

    if _is_text(path):
        raw = path.read_bytes()[:_MAX_TEXT_BYTES]
        truncated = size > _MAX_TEXT_BYTES
        content = raw.decode("utf-8", errors="replace")
        return {
            "type": "file",
            "path": str(path),
            "relative_path": rel,
            "language": language,
            "content": content,
            "encoding": "text",
            "size_bytes": size,
            "truncated": truncated,
        }
    else:
        raw = path.read_bytes()[:_MAX_TEXT_BYTES]
        truncated = size > _MAX_TEXT_BYTES
        content = base64.b64encode(raw).decode("ascii")
        return {
            "type": "file",
            "path": str(path),
            "relative_path": rel,
            "language": language,
            "content": content,
            "encoding": "base64",
            "size_bytes": size,
            "truncated": truncated,
        }


def _is_text(path: Path) -> bool:
    """Return True if the file should be treated as text.

    Args:
        path: File path.

    Returns:
        True for text files.
    """
    if path.suffix.lower() in _TEXT_EXTENSIONS:
        return True
    mime, _ = mimetypes.guess_type(str(path))
    if mime and mime.startswith("text/"):
        return True
    return False


def _detect_language(path: Path) -> str:
    """Map a file extension to a language identifier string.

    Args:
        path: File path.

    Returns:
        Language string such as ``"python"`` or ``"text"``.
    """
    mapping = {
        ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".go": "go", ".rs": "rust", ".java": "java",
        ".c": "c", ".cpp": "cpp", ".h": "c", ".hpp": "cpp",
        ".md": "markdown", ".rst": "rst", ".txt": "text",
        ".json": "json", ".yaml": "yaml", ".yml": "yaml",
        ".toml": "toml", ".html": "html", ".css": "css",
        ".sql": "sql", ".sh": "bash", ".ps1": "powershell",
        ".xml": "xml", ".graphql": "graphql",
    }
    return mapping.get(path.suffix.lower(), "binary")


def _relative(path: Path, project_root: str) -> str:
    """Compute a project-relative path string.

    Args:
        path: Absolute file path.
        project_root: Project root directory.

    Returns:
        Relative path string, or the absolute path if not under root.
    """
    if not project_root:
        return str(path)
    try:
        return str(path.relative_to(Path(project_root)))
    except ValueError:
        return str(path)
