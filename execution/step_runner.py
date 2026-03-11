"""step_runner.py — Extracted step execution logic."""
from __future__ import annotations
from typing import Any, Dict


def run_step(step: Dict[str, Any], agent_registry: Dict[str, Any],
             tool_registry: Any, context: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a single pipeline step via its assigned agent."""
    agent_name = step.get("agent", "coding")
    agent = agent_registry.get(agent_name) or agent_registry.get("coding")
    return agent.run(step, context)
