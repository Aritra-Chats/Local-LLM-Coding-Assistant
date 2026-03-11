"""conversation_memory.py — Long-term conversation memory store."""
from __future__ import annotations
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ConversationTurn:
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)


class ConversationMemory:
    """Stores and retrieves conversation turns across sessions."""

    def __init__(self, max_turns: int = 200) -> None:
        self._turns: List[ConversationTurn] = []
        self._max_turns = max_turns

    def add(self, role: str, content: str) -> None:
        self._turns.append(ConversationTurn(role=role, content=content))
        if len(self._turns) > self._max_turns:
            self._turns = self._turns[-self._max_turns:]

    def recent(self, n: int = 10) -> List[Dict[str, Any]]:
        return [
            {"role": t.role, "content": t.content, "timestamp": t.timestamp}
            for t in self._turns[-n:]
        ]

    def clear(self) -> None:
        self._turns.clear()
