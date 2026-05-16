"""supervisor_bus.py — Thread-safe event channel for the async supervisor.

The :class:`SupervisorBus` is the communication backbone between the
:class:`~core.execution_engine.ConcreteExecutionEngine` (producer) and the
:class:`~core.async_supervisor.AsyncSupervisorLoop` (consumer).

Both sides interact with the bus through typed :class:`BusEvent` and
:class:`FixProposal` objects so the interface is explicit and easy to
unit-test without spinning up real agents.
"""
from __future__ import annotations

import queue
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Event vocabulary
# ---------------------------------------------------------------------------

class BusEventType(str, Enum):
    TOOL_FAILED        = "TOOL_FAILED"
    STEP_ENTRY_FAILED  = "STEP_ENTRY_FAILED"
    STEP_EXIT_FAILED   = "STEP_EXIT_FAILED"
    STEP_COMPLETED     = "STEP_COMPLETED"
    PIPELINE_DONE      = "PIPELINE_DONE"
    ABORT_REQUESTED    = "ABORT_REQUESTED"


@dataclass
class BusEvent:
    """A single event emitted by the engine onto the :class:`SupervisorBus`.

    Attributes:
        type: One of the :class:`BusEventType` values.
        step_id: UUID string identifying the pipeline step.
        step_index: Integer position of the step in the pipeline.
        step_name: Human-readable step name.
        tool_name: Only populated for :attr:`BusEventType.TOOL_FAILED`.
        error: Error string / traceback.
        context: The step execution context dict at time of failure.
        attempt: Retry attempt number (0-based).
        extra: Arbitrary extra data (e.g. failed_items from contract checker).
    """

    type: BusEventType
    step_id: str
    step_index: int
    step_name: str
    tool_name: str = ""
    error: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    attempt: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Fix proposal — supervisor → engine
# ---------------------------------------------------------------------------

@dataclass
class FixProposal:
    """A repair plan returned by :class:`~core.async_supervisor.AsyncSupervisorLoop`.

    Attributes:
        fix_actions: Ordered list of :class:`~agents.agent_action.AgentAction`
            instances the engine should dispatch before retrying.
        rationale: One-sentence summary of what the fix does (shown in the
            progress tracker label).
        retry_original: When True the engine re-runs the original failed step
            after the fix actions complete.
        fix_possible: False means the supervisor gave up; engine should abort.
    """

    fix_actions: List[Any]       # List[AgentAction]
    rationale: str
    retry_original: bool = True
    fix_possible: bool = True


# ---------------------------------------------------------------------------
# Bus
# ---------------------------------------------------------------------------

class SupervisorBus:
    """Thin, thread-safe wrapper around :class:`queue.Queue`.

    The execution engine (main thread) calls :meth:`emit`; the supervisor
    loop (daemon thread) calls :meth:`get` in a tight loop.
    """

    def __init__(self) -> None:
        self._q: queue.Queue = queue.Queue()

    def emit(self, event: BusEvent) -> None:
        """Put *event* on the queue (non-blocking, unbounded)."""
        self._q.put(event)

    def get(self, timeout: float = 1.0) -> Optional[BusEvent]:
        """Block up to *timeout* seconds and return the next event, or None."""
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def empty(self) -> bool:
        """Return True if there are no events pending."""
        return self._q.empty()
