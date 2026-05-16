"""file_change_map.py — Session-scoped registry of file changes.

Tracks every file created or modified during a pipeline run, providing:
  - A canonical absolute path for each logical (relative) path an agent used.
  - Chronological event log for audit / display.
  - JSON persistence so the CriticAgent and CLI /files command can query it.

Usage
-----
The FileChangeMap instance is owned by ConcreteExecutionEngine (one per
pipeline run) and accessible as ``engine.file_change_map``.  At pipeline
end it is saved to::

    ~/.sentinel/sessions/<session_id>_file_changes.json
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional


# ---------------------------------------------------------------------------
# Event type
# ---------------------------------------------------------------------------


class FileChangeEvent(NamedTuple):
    """A single file-change event recorded during a pipeline run.

    Attributes:
        logical_path:  Path as the agent originally specified it.
        absolute_path: Fully resolved global filesystem path.
        operation:     ``"create"``, ``"modify"``, or ``"delete"``.
        step_id:       Pipeline step that produced this change.
        agent:         Agent name that triggered the change.
        timestamp_ms:  Epoch milliseconds when the change was recorded.
    """

    logical_path: str
    absolute_path: str
    operation: str   # "create" | "modify" | "delete"
    step_id: str
    agent: str
    timestamp_ms: int


def _path_variant_matches(requested: str, candidate: str) -> bool:
    """Return True when two logical paths are close enough to resolve safely."""
    requested_path = requested.replace("\\", "/").strip()
    candidate_path = candidate.replace("\\", "/").strip()

    if requested_path.lower() == candidate_path.lower():
        return True

    requested_parent = os.path.dirname(requested_path)
    candidate_parent = os.path.dirname(candidate_path)
    if requested_parent and requested_parent.lower() != candidate_parent.lower():
        return False

    requested_base = os.path.basename(requested_path)
    candidate_base = os.path.basename(candidate_path)
    if requested_base.lower() == candidate_base.lower():
        return True

    requested_stem, requested_ext = os.path.splitext(requested_base)
    candidate_stem, candidate_ext = os.path.splitext(candidate_base)
    if requested_ext.lower() != candidate_ext.lower():
        return False

    requested_stem = requested_stem.lower()
    candidate_stem = candidate_stem.lower()
    if requested_stem == candidate_stem:
        return True

    if requested_stem + "s" == candidate_stem or candidate_stem + "s" == requested_stem:
        return True

    return False


# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------


class FileChangeMap:
    """In-memory (and optionally persisted) registry of file changes.

    All lookup keys are normalised to POSIX-style strings so cross-platform
    paths match correctly.
    """

    def __init__(self) -> None:
        self._events: List[FileChangeEvent] = []
        # logical_path → most-recent absolute_path
        self._index: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Mutating operations
    # ------------------------------------------------------------------

    def record(self, event: FileChangeEvent) -> None:
        """Append *event* and update the logical→absolute index.

        Args:
            event: The :class:`FileChangeEvent` to record.
        """
        self._events.append(event)
        self._index[event.logical_path] = event.absolute_path
        # Also index by absolute path so absolute lookups work too.
        self._index[event.absolute_path] = event.absolute_path

    # ------------------------------------------------------------------
    # Query operations
    # ------------------------------------------------------------------

    def resolve(self, logical_path: str) -> Optional[str]:
        """Return the absolute path for *logical_path*, or ``None``.

        Args:
            logical_path: The path as originally specified by an agent.

        Returns:
            The resolved absolute path string, or ``None`` if not found.
        """
        resolved = self._index.get(logical_path)
        if resolved is not None:
            return resolved

        normalized = logical_path.replace("\\", "/")
        resolved = self._index.get(normalized)
        if resolved is not None:
            return resolved

        candidates: List[str] = []
        for event in reversed(self._events):
            if _path_variant_matches(logical_path, event.logical_path):
                candidates.append(event.absolute_path)

        if len(candidates) == 1:
            return candidates[0]

        return None

    def all_events(self) -> List[FileChangeEvent]:
        """Return all recorded events in chronological order."""
        return list(self._events)

    def changed_paths(self) -> List[str]:
        """Return a de-duplicated list of unique absolute paths that changed."""
        seen: Dict[str, None] = {}
        for ev in self._events:
            seen[ev.absolute_path] = None
        return list(seen.keys())

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict:
        """Serialise the map to a plain dict (JSON-safe)."""
        return {
            "events": [ev._asdict() for ev in self._events],
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "FileChangeMap":
        """Reconstruct a :class:`FileChangeMap` from a serialised dict.

        Args:
            d: Dict produced by :meth:`to_dict`.

        Returns:
            A populated :class:`FileChangeMap`.
        """
        obj = cls()
        for raw in d.get("events", []):
            try:
                ev = FileChangeEvent(**raw)
                obj.record(ev)
            except (TypeError, KeyError):
                continue
        return obj

    def save(self, path: Path) -> None:
        """Persist the map to a JSON file.

        Args:
            path: Destination path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "FileChangeMap":
        """Load a previously persisted map.

        Args:
            path: Source JSON file.

        Returns:
            A :class:`FileChangeMap` instance, or an empty one if
            the file is missing or corrupt.
        """
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls.from_dict(data)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return cls()
