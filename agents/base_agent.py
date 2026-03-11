from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseAgent(ABC):
    """Abstract base class for all Sentinel agents.

    Every specialist and orchestrator agent must inherit from this class
    and implement all abstract methods.
    """

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
