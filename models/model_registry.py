"""model_registry.py — Registry of known Ollama models and capabilities."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModelEntry:
    name: str
    size_gb: float
    capabilities: List[str] = field(default_factory=list)
    context_limit: int = 4096
    council_eligible: bool = False


class ModelRegistry:
    """Maintains a catalogue of model entries."""

    def __init__(self) -> None:
        self._entries: Dict[str, ModelEntry] = {}

    def register(self, entry: ModelEntry) -> None:
        self._entries[entry.name] = entry

    def get(self, name: str) -> Optional[ModelEntry]:
        return self._entries.get(name)

    def all(self) -> List[ModelEntry]:
        return list(self._entries.values())

    def for_capability(self, capability: str) -> List[ModelEntry]:
        return [e for e in self._entries.values() if capability in e.capabilities]
