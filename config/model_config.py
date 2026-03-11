"""model_config.py — Model capability and selection configuration."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModelConfig:
    """Configuration entry for a single Ollama model."""
    name: str
    size_gb: float
    capabilities: List[str] = field(default_factory=list)
    context_limit: int = 4096
    hardware_modes: List[str] = field(default_factory=lambda: ["minimal", "standard", "advanced"])
    council_eligible: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "size_gb": self.size_gb,
            "capabilities": self.capabilities,
            "context_limit": self.context_limit,
            "hardware_modes": self.hardware_modes,
            "council_eligible": self.council_eligible,
        }
