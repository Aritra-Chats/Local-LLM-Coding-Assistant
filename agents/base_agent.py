from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseAgent(ABC):
    """Abstract base class for all Sentinel agents.

    Every specialist and orchestrator agent must inherit from this class
    and implement all abstract methods.
    """

    def __init__(self) -> None:
        # Inference client injected at runtime by ConcreteExecutionEngine.
        # Agents that use an LLM directly should check self._inference_client
        # before falling back to their default local client.
        self._inference_client: Optional[Any] = None

    def use_client(self, client: Any) -> None:
        """Inject an inference client for this agent to use.

        Called by :class:`~core.execution_engine.ConcreteExecutionEngine`
        in online mode so agents transparently route to the selected
        provider (local Ollama, Ollama Cloud, Anthropic, OpenAI, Google).

        Args:
            client: An inference client that exposes at minimum a
                    ``generate(model, prompt, **kwargs)`` method.
        """
        self._inference_client = client

    @abstractmethod
    def run(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent's primary responsibility for a given task.

        Args:
            task: Structured task definition including goal, step, and metadata.
            context: Pre-built context payload from ContextBuilder.

        Returns:
            A result dict containing output, status, and any artefacts produced.
        """
        ...

    @abstractmethod
    def validate_output(self, output: Dict[str, Any]) -> bool:
        """Validate the output produced by this agent before it is accepted.

        Args:
            output: The raw output dict returned by run().

        Returns:
            True if the output passes validation, False otherwise.
        """
        ...

    @abstractmethod
    def handle_error(self, error: Exception, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an error that occurred during task execution.

        Args:
            error: The exception that was raised.
            task: The task that was being executed when the error occurred.

        Returns:
            A recovery result dict or a structured error payload.
        """
        ...

    @abstractmethod
    def describe(self) -> str:
        """Return a human-readable description of this agent's role and capabilities.

        Returns:
            A plain-text description string.
        """
        ...
