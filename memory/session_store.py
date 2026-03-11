"""session_manager.py — Sentinel session lifecycle manager.

Handles creation, persistence, resumption, and teardown of CLI sessions.
Each session stores conversation history, active task state, pipeline
state, and runtime metadata.
"""

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


_SESSIONS_DIR = Path(os.environ.get("SENTINEL_SESSIONS_DIR", Path.home() / ".sentinel" / "sessions"))


class SessionManager:
    """Manages the lifecycle of a single Sentinel CLI session.

    Attributes:
        session_id: Unique identifier for this session.
        created_at: UTC timestamp of session creation.
        history: Ordered list of conversation turns.
        pipeline_state: Snapshot of the currently active pipeline, if any.
        metadata: Arbitrary key-value metadata for the session.
    """

    def __init__(self, session_id: Optional[str] = None) -> None:
        """Initialise or resume a session.

        Args:
            session_id: If provided, attempt to load that session from disk.
                        If None, create a new session.
        """
        self.session_id: str = session_id or str(uuid.uuid4())
        self.created_at: str = datetime.now(timezone.utc).isoformat()
        self.history: List[Dict[str, Any]] = []
        self.pipeline_state: Optional[Dict[str, Any]] = None
        self.active_task: Optional[Dict[str, Any]] = None
        self.metadata: Dict[str, Any] = {}
        self._session_file: Path = _SESSIONS_DIR / f"{self.session_id}.json"

        if session_id is not None:
            self._load()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Mark the session as started and ensure the sessions directory exists."""
        _SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        self.metadata["started_at"] = datetime.now(timezone.utc).isoformat()

    def save(self) -> None:
        """Persist the current session state to disk as JSON."""
        _SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "history": self.history,
            "pipeline_state": self.pipeline_state,
            "active_task": self.active_task,
            "metadata": self.metadata,
        }
        self._session_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load(self) -> None:
        """Load session state from disk.

        Raises:
            FileNotFoundError: If the session file does not exist.
        """
        if not self._session_file.exists():
            raise FileNotFoundError(
                f"Session '{self.session_id}' not found at {self._session_file}"
            )
        payload = json.loads(self._session_file.read_text(encoding="utf-8"))
        self.created_at = payload.get("created_at", self.created_at)
        self.history = payload.get("history", [])
        self.pipeline_state = payload.get("pipeline_state")
        self.active_task = payload.get("active_task")
        self.metadata = payload.get("metadata", {})

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def add_turn(self, role: str, content: str) -> None:
        """Append a conversation turn to the session history.

        Args:
            role: Speaker role — 'user' or 'assistant'.
            content: The message content.
        """
        self.history.append(
            {
                "role": role,
                "content": content,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return conversation history, optionally limited to the last N turns.

        Args:
            limit: If provided, return only the most recent N turns.

        Returns:
            List of conversation turn dicts.
        """
        if limit is not None:
            return self.history[-limit:]
        return list(self.history)

    def clear_history(self) -> None:
        """Erase all conversation history for this session."""
        self.history.clear()

    # ------------------------------------------------------------------
    # Pipeline state
    # ------------------------------------------------------------------

    def set_pipeline_state(self, state: Dict[str, Any]) -> None:
        """Store a pipeline state snapshot.

        Args:
            state: The pipeline state dict from the ExecutionEngine.
        """
        self.pipeline_state = state

    def clear_pipeline_state(self) -> None:
        """Clear the stored pipeline state."""
        self.pipeline_state = None

    # ------------------------------------------------------------------
    # Task state
    # ------------------------------------------------------------------

    def set_active_task(self, task: Dict[str, Any]) -> None:
        """Set the currently active task.

        Args:
            task: The structured task dict.
        """
        self.active_task = task

    def clear_active_task(self) -> None:
        """Clear the active task reference."""
        self.active_task = None

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return a summary of the current session state.

        Returns:
            A dict with session ID, turn count, and active task info.
        """
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "turn_count": len(self.history),
            "has_active_task": self.active_task is not None,
            "has_pipeline": self.pipeline_state is not None,
            "metadata": self.metadata,
        }

    @staticmethod
    def list_sessions() -> List[str]:
        """Return a list of all saved session IDs.

        Returns:
            List of session ID strings found in the sessions directory.
        """
        if not _SESSIONS_DIR.exists():
            return []
        return [p.stem for p in _SESSIONS_DIR.glob("*.json")]
