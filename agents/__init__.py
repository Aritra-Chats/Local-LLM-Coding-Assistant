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
from agents.critic_agent import CriticAgent
from typing import Dict, Any, Optional


def build_agent_registry(model_router=None):
    """Instantiate and return the default agent registry.

    Uses InferenceEngine as the LLM client when available. It is a drop-in
    superset of OllamaClient that adds quantization hints, connection pooling,
    KV-cache prefix warmup, speculative decoding, and optional
    sentence-transformers / llama-cpp-python backends -- all transparent to agents.
    """
    ollama_client = None
    coding_model = ""
    reasoning_model = ""
    debugging_model = ""

    if model_router is not None:
        try:
            # Prefer InferenceEngine; fall back to OllamaClient if import fails
            try:
                from models.inference_engine import InferenceEngine
                from typing import Any
                hardware_mode = model_router.get_hardware_profile() if model_router else "standard"
                client: Any = InferenceEngine(hardware_mode=hardware_mode)
            except Exception:
                from models.ollama_client import OllamaClient
                client = OllamaClient()

            if client.is_available():
                ollama_client = client
                coding_model    = model_router.select_coding_model()
                reasoning_model = model_router.select_reasoning_model()
                debugging_model = (
                    model_router.select_debugging_model()
                    if hasattr(model_router, "select_debugging_model")
                    else coding_model
                )
        except Exception:
            pass

    supervisor   = ConcreteSupervisorAgent(ollama_client=ollama_client, model=reasoning_model)
    planner      = ConcretePlannerAgent(ollama_client=ollama_client, model=reasoning_model)
    pipeline_gen = ConcretePipelineGeneratorAgent()
    return {
        "supervisor":         supervisor,
        "planner":            planner,
        "pipeline_generator": pipeline_gen,
        "coding":    CodingAgent(ollama_client=ollama_client,   model=coding_model),
        "debugging": DebuggingAgent(ollama_client=ollama_client, model=debugging_model),
        "reasoning": ReasoningAgent(ollama_client=ollama_client, model=reasoning_model),
        "devops":    DevOpsAgent(ollama_client=ollama_client,   model=reasoning_model),
        "research":  ResearchAgent(ollama_client=ollama_client, model=reasoning_model),
        "system":    SystemAgent(ollama_client=ollama_client,   model=reasoning_model),
        # CriticAgent — reviews write_file actions before ToolRegistry dispatch
        "critic":    CriticAgent(ollama_client=ollama_client,   model=reasoning_model),
    }


__all__ = [
    "BaseAgent", "AgentAction", "ACTION_TYPES",
    "SupervisorAgent", "ConcreteSupervisorAgent",
    "PlannerAgent", "ConcretePlannerAgent",
    "PipelineGenerator", "ConcretePipelineGeneratorAgent",
    "CodingAgent", "DebuggingAgent", "ReasoningAgent",
    "DevOpsAgent", "ResearchAgent", "SystemAgent",
    "CriticAgent",
    "build_agent_registry",
]
