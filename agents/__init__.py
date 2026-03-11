"""agents — Sentinel agent registry and all specialist agents."""

from agents.base_agent import BaseAgent
from agents.agent_action import AgentAction, ACTION_TYPES
from agents.supervisor import SupervisorAgent, ConcreteSupervisorAgent
from agents.planner import PlannerAgent, ConcretePlannerAgent
from agents.pipeline_generator import PipelineGenerator, ConcretePipelineGeneratorAgent
from agents.coding_agent import CodingAgent
from agents.debugging_agent import DebuggingAgent
from agents.reasoning_agent import ReasoningAgent
from agents.devops_agent import DevOpsAgent
from agents.research_agent import ResearchAgent
from agents.system_agent import SystemAgent
from typing import Dict, Any


def build_agent_registry() -> Dict[str, Any]:
    """Instantiate and return the default agent registry."""
    supervisor   = ConcreteSupervisorAgent()
    planner      = ConcretePlannerAgent()
    pipeline_gen = ConcretePipelineGeneratorAgent()
    return {
        "supervisor":         supervisor,
        "planner":            planner,
        "pipeline_generator": pipeline_gen,
        "coding":             CodingAgent(),
        "debugging":          DebuggingAgent(),
        "reasoning":          ReasoningAgent(),
        "devops":             DevOpsAgent(),
        "research":           ResearchAgent(),
        "system":             SystemAgent(),
    }


__all__ = [
    "BaseAgent", "AgentAction", "ACTION_TYPES",
    "SupervisorAgent", "ConcreteSupervisorAgent",
    "PlannerAgent", "ConcretePlannerAgent",
    "PipelineGenerator", "ConcretePipelineGeneratorAgent",
    "CodingAgent", "DebuggingAgent", "ReasoningAgent",
    "DevOpsAgent", "ResearchAgent", "SystemAgent",
    "build_agent_registry",
]
