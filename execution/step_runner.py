"""step_runner.py — Centralised tool-level dispatcher.

:class:`StepRunner` is the single point through which every ``tool_call``
:class:`~agents.agent_action.AgentAction` is dispatched to the
:class:`~tools.tool_registry.ConcreteToolRegistry`.

Design contract
---------------
* :class:`StepRunner` does **not** retry.  Retry policy is the supervisor's
  responsibility once a TOOL_FAILED event lands on the bus.
* :class:`StepRunner` does **not** pause the engine.  It emits the event and
  returns immediately; the engine pauses itself after inspecting the result.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agents.agent_action import AgentAction
    from core.supervisor_bus import SupervisorBus


class StepRunner:
    """Dispatch a single ``tool_call`` action and report failures to the bus.

    Args:
        invoke_fn: Callable with signature
            ``(tool_name, params, action, context) -> dict``.
            This is ``ConcreteExecutionEngine._invoke_tool`` in practice.
        update_label_fn: Optional callable ``(step_index, label) -> None``.
            Used to push live action labels to the progress tracker.
    """

    def __init__(
        self,
        invoke_fn: Callable[..., Dict[str, Any]],
        update_label_fn: Optional[Callable[[int, str], None]] = None,
    ) -> None:
        self._invoke = invoke_fn
        self._update_label = update_label_fn

    def run_action(
        self,
        action: "AgentAction",
        supervisor_bus: Optional["SupervisorBus"],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Dispatch *action* and emit a TOOL_FAILED event on clean failure.

        "Clean failure" means the tool returned ``{"success": False, ...}``
        rather than raising an exception.

        Args:
            action: The ``tool_call`` AgentAction.
            supervisor_bus: Optional bus to emit TOOL_FAILED events on.
            context: The current step execution context.

        Returns:
            The tool result dict.
        """
        tool_name = action.payload.get("tool", "")
        params    = action.payload.get("params", {})

        # Resolve step_index for progress label
        step_idx: Optional[int] = None
        if context:
            raw = context.get("step_index")
            if raw is not None:
                try:
                    step_idx = int(raw)
                except (TypeError, ValueError):
                    pass

        if self._update_label and step_idx is not None:
            try:
                self._update_label(step_idx, f"→ {tool_name}")
            except Exception:
                pass

        # Invoke the tool
        result = self._invoke(tool_name, params, action, context)

        # Emit TOOL_FAILED for clean (non-exception) failures
        if not result.get("success", True) and supervisor_bus is not None:
            from core.supervisor_bus import BusEvent, BusEventType
            supervisor_bus.emit(BusEvent(
                type=BusEventType.TOOL_FAILED,
                step_id=str(action.step_id or ""),
                step_index=step_idx if step_idx is not None else -1,
                step_name=context.get("step_name", ""),
                tool_name=tool_name,
                error=result.get("error", ""),
                context=context,
                attempt=context.get("_attempt", 0),
                extra={"params": params, "result": result},
            ))

        return result
