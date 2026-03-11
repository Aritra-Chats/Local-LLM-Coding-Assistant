"""project_index.py — Persistent project file index for fast lookups."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class ProjectIndex:
    """Maintains a simple JSON index of a project's file tree."""

    def __init__(self, project_root: str = ".") -> None:
        self.project_root = Path(project_root).resolve()
        self._index: Dict[str, Any] = {}

    def build(self) -> None:
        """Walk the project root and index all Python and text files."""
        self._index = {}
        for p in self.project_root.rglob("*"):
            if p.is_file() and not any(
                part.startswith(".") or part in {"__pycache__", ".venv", "node_modules"}
                for part in p.parts
            ):
                rel = str(p.relative_to(self.project_root))
                self._index[rel] = {"size": p.stat().st_size, "suffix": p.suffix}

    def search(self, query: str) -> List[str]:
        """Return paths whose names contain *query*."""
        q = query.lower()
        return [path for path in self._index if q in path.lower()]

    def persist(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._index, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, project_root: str = ".") -> "ProjectIndex":
        idx = cls(project_root)
        if path.exists():
            idx._index = json.loads(path.read_text(encoding="utf-8"))
        return idx
