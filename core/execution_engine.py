from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List


class ExecutionEngine(ABC):
    """Abstract base class for the Execution Engine.

    The Execution Engine is the runtime core that drives a pipeline to
    completion. It iterates over steps, builds context, selects models,
    dispatches agents, validates outputs, and handles failures.
    """

    @abstractmethod
    def run_pipeline(self, pipeline: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute all steps in a pipeline sequentially or in parallel.

        Args:
            pipeline: The fully generated pipeline list.

        Returns:
            A summary dict containing per-step results and final status.
        """
        ...

    @abstractmethod
    def run_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single pipeline step.

        Builds context, selects the appropriate model, invokes the assigned
        agent, and validates the result.

        Args:
            step: A single enriched pipeline step dict.

        Returns:
            The step result dict including output, status, and timing.
        """
        ...

    @abstractmethod
    def build_context(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Build the context payload required to execute a step.

        Args:
            step: The pipeline step for which context is being built.

        Returns:
            A context dict assembled from RAG, symbol graph, memory, and more.
        """
        ...

    @abstractmethod
    def select_model(self, step: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Select the most appropriate model for a given step.

        Args:
            step: The pipeline step requiring a model.
            context: The assembled context payload.

        Returns:
            The model identifier string to be used for this step.
        """
        ...

    @abstractmethod
    def handle_failure(self, step: Dict[str, Any], error: Exception) -> Dict[str, Any]:
        """Handle a step-level failure during pipeline execution.

        Determines whether to retry, modify the pipeline, or switch models.

        Args:
            step: The step that failed.
            error: The exception raised during execution.

        Returns:
            A recovery action dict.
        """
        ...


# ────────────────────────────────────────────────────────────────────────────

"""concrete_execution_engine.py — Sentinel concrete ExecutionEngine.

Responsibilities
----------------
1. **Execute pipeline steps** — iterate in dependency order, respecting
   ``can_parallelize`` and ``depends_on`` constraints.
2. **Construct context** — build per-step context dicts from system state
   and any context hints carried on the step.
3. **Call agents** — route each step to its assigned specialist (or council).
4. **Call tools** — dispatch ``tool_call`` :class:`~agents.agent_action.AgentAction`
   objects to :class:`~tools.ConcreteToolRegistry`.
5. **Validate outputs** — check every agent return value and tool result.
6. **Retry failed steps** — honour the ``max_retries`` budget per step;
   fall back gracefully on exhaustion.
7. **Stream progress** to the CLI via :class:`~cli.display.ProgressTracker`.

Architecture
------------
The engine is intentionally thin: it dispatches rather than reasons.
All intelligence lives in agents; the engine provides the dispatch loop,
retry logic, context assembly, and progress plumbing.

Execution flow per step
-----------------------
::

    build_context(step)
        ↓
    select_model(step, context)           ← uses step.model_hint
        ↓
    agent.run(step, context)              ← specialist or council
        ↓
    _dispatch_actions(actions, context)  ← tool_call / delegate / message
        ↓
    agent.validate_output(result)         ← structural check
        ↓
    emit STEP_COMPLETE progress event

Retry policy
------------
When ``agent.run`` or a tool invocation raises, or ``validate_output``
returns False, the engine retries up to ``step["max_retries"]`` times with
exponential back-off (base 2 s, capped at 30 s).  On exhaustion the step
is marked ``"failed"`` and the engine continues (unless ``abort_on_failure``
is set).

Council mode
------------
When a step carries a non-empty ``council_agents`` list, the engine runs
all listed agents in sequence and merges their ``actions`` lists before
dispatch.  The first agent in the list is the primary; subsequent agents
are reviewers whose ``message`` and ``decision`` actions are appended.

Streaming progress
------------------
The engine emits :class:`ProgressEvent` dicts at each lifecycle transition.
Callers can pass an ``on_progress`` callback or pull from the event list
after completion.  :class:`~cli.display.ProgressTracker`
is started and stopped automatically when ``show_progress=True``.
"""


import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional

from agents.agent_action import AgentAction


# ---------------------------------------------------------------------------
# Progress event type
# ---------------------------------------------------------------------------

# Valid event names emitted by the engine.
PROGRESS_EVENTS = (
    "pipeline_start",
    "step_start",
    "step_retry",
    "step_complete",
    "step_failed",
    "step_skipped",
    "action_dispatched",
    "pipeline_complete",
    "pipeline_failed",
)


@dataclass
class ProgressEvent:
    """A single streaming progress notification.

    Attributes:
        event: One of :data:`PROGRESS_EVENTS`.
        step_index: Zero-based step index (``-1`` for pipeline-level events).
        step_name: Display name of the step.
        message: Human-readable description of what happened.
        data: Arbitrary extra payload (step dict, result, error string, …).
        elapsed_ms: Wall-clock time since the engine started.
    """

    event: str
    step_index: int = -1
    step_name: str = ""
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    elapsed_ms: float = 0.0


# ---------------------------------------------------------------------------
# Step result
# ---------------------------------------------------------------------------


@dataclass
class StepResult:
    """Result of executing a single pipeline step.

    Attributes:
        step_id: ID of the step that was executed.
        step_name: Display name of the step.
        status: ``"completed"``, ``"failed"``, or ``"skipped"``.
        output: The agent's ``run()`` return value.
        actions: All :class:`~agents.agent_action.AgentAction` objects
            generated during the step (including tool results as messages).
        tool_results: Results from every ``tool_call`` action dispatched.
        retries_used: How many retry attempts were consumed.
        elapsed_ms: Wall-clock execution time in milliseconds.
        error: Error message if the step failed.
    """

    step_id: str
    step_name: str
    status: str = "pending"
    output: Dict[str, Any] = field(default_factory=dict)
    actions: List[AgentAction] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    retries_used: int = 0
    elapsed_ms: float = 0.0
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id":      self.step_id,
            "step_name":    self.step_name,
            "status":       self.status,
            "output":       self.output,
            "actions":      [a.to_dict() for a in self.actions],
            "tool_results": self.tool_results,
            "retries_used": self.retries_used,
            "elapsed_ms":   round(self.elapsed_ms, 2),
            "error":        self.error,
        }


# ---------------------------------------------------------------------------
# Pipeline run result
# ---------------------------------------------------------------------------


@dataclass
class PipelineRunResult:
    """Result of executing a complete pipeline.

    Attributes:
        run_id: Unique UUID for this execution run.
        pipeline_id: ID of the pipeline that was executed.
        goal: The pipeline's human-readable goal.
        status: ``"completed"``, ``"failed"``, or ``"partial"``.
        step_results: Per-step :class:`StepResult` objects in order.
        total_elapsed_ms: Total wall-clock time for the full pipeline.
        events: All :class:`ProgressEvent` objects emitted during execution.
    """

    run_id: str
    pipeline_id: str
    goal: str
    status: str = "completed"
    step_results: List[StepResult] = field(default_factory=list)
    total_elapsed_ms: float = 0.0
    events: List[ProgressEvent] = field(default_factory=list)

    # Convenience
    @property
    def completed_steps(self) -> int:
        return sum(1 for r in self.step_results if r.status == "completed")

    @property
    def failed_steps(self) -> int:
        return sum(1 for r in self.step_results if r.status == "failed")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id":           self.run_id,
            "pipeline_id":      self.pipeline_id,
            "goal":             self.goal,
            "status":           self.status,
            "completed_steps":  self.completed_steps,
            "failed_steps":     self.failed_steps,
            "step_results":     [r.to_dict() for r in self.step_results],
            "total_elapsed_ms": round(self.total_elapsed_ms, 2),
        }

    def summary(self) -> str:
        return (
            f"Run '{self.run_id[:8]}' | {self.status} | "
            f"steps={len(self.step_results)} "
            f"ok={self.completed_steps} fail={self.failed_steps} | "
            f"{self.total_elapsed_ms:.0f}ms"
        )


# ---------------------------------------------------------------------------
# Retry back-off helper
# ---------------------------------------------------------------------------

_MAX_BACKOFF_SECONDS = 30.0


def _backoff(attempt: int, base: float = 2.0) -> float:
    """Return exponential back-off delay (capped at :data:`_MAX_BACKOFF_SECONDS`)."""
    return min(base ** attempt, _MAX_BACKOFF_SECONDS)


# ---------------------------------------------------------------------------
# ConcreteExecutionEngine
# ---------------------------------------------------------------------------


class ConcreteExecutionEngine(ExecutionEngine):
    """Drives a :class:`~execution.pipeline.Pipeline` to completion.

    Parameters
    ----------
    agent_registry:
        Dict mapping agent name strings to :class:`~agents.base_agent.BaseAgent`
        instances.  Use :func:`~agents.build_agent_registry` to
        obtain the default registry.
    tool_registry:
        :class:`~tools.ConcreteToolRegistry` pre-loaded with tools.
        Use :func:`~tools.build_default_registry` to obtain the default.
    abort_on_failure:
        If ``True``, the pipeline halts as soon as one step is exhausted of
        retries.  If ``False`` (default), execution continues and the step
        is marked ``"failed"`` in the result.
    show_progress:
        If ``True``, start/stop :class:`~cli.display.ProgressTracker`
        automatically during ``run_pipeline``.
    on_progress:
        Optional callback ``(ProgressEvent) -> None`` called synchronously on
        every lifecycle event.  Useful for tests and non-Rich consumers.
    console:
        Optional Rich ``Console`` forwarded to the progress tracker.

    Example::

        from agents import build_agent_registry
        from tools import build_default_registry

        engine = ConcreteExecutionEngine(
            agent_registry=build_agent_registry(),
            tool_registry=build_default_registry(),
            show_progress=True,
        )
        result = engine.run_pipeline(pipeline)
        print(result.summary())
    """

    def __init__(
        self,
        agent_registry: Optional[Dict[str, Any]] = None,
        tool_registry: Optional[Any] = None,
        abort_on_failure: bool = False,
        show_progress: bool = True,
        on_progress: Optional[Callable[[ProgressEvent], None]] = None,
        console: Optional[Any] = None,
    ) -> None:
        self._agents: Dict[str, Any] = agent_registry or {}
        self._tools: Optional[Any] = tool_registry
        self.abort_on_failure = abort_on_failure
        self.show_progress = show_progress
        self.on_progress = on_progress
        self._console = console
        self._events: List[ProgressEvent] = []
        self._start_time: float = 0.0

    # ------------------------------------------------------------------
    # Primary public API — typed Pipeline input
    # ------------------------------------------------------------------

    def run_pipeline(self, pipeline: Any) -> PipelineRunResult:
        """Execute a :class:`~execution.pipeline.Pipeline`.

        Accepts either a typed
        :class:`~execution.pipeline.Pipeline` object or a
        plain list of step dicts (backwards-compat with the ABC signature).

        Args:
            pipeline: A :class:`~execution.pipeline.Pipeline`
                or a list of step dicts.

        Returns:
            :class:`PipelineRunResult` with per-step results and overall status.
        """
        # Normalise input: accept Pipeline objects or raw lists.
        if hasattr(pipeline, "ordered_steps"):
            steps = [s.to_dict() for s in pipeline.ordered_steps()]
            pipeline_id = pipeline.pipeline_id
            goal = pipeline.goal
        else:
            steps = list(pipeline)
            pipeline_id = str(uuid.uuid4())
            goal = next(
                (s.get("description", s.get("name", "")) for s in steps if s), ""
            )

        run_id = str(uuid.uuid4())
        self._events = []
        self._start_time = time.monotonic()

        result = PipelineRunResult(
            run_id=run_id,
            pipeline_id=pipeline_id,
            goal=goal,
        )

        # Progress tracker (Rich) — only start if Rich is available.
        tracker = self._make_tracker() if self.show_progress else None
        if tracker:
            try:
                tracker.start_pipeline(steps, task_name=f"Pipeline: {goal[:60]}")
            except Exception:
                tracker = None

        self._emit(ProgressEvent(
            event="pipeline_start",
            message=f"Starting pipeline '{pipeline_id[:8]}' with {len(steps)} steps.",
            data={"pipeline_id": pipeline_id, "goal": goal},
        ))

        try:
            for step in steps:
                if step.get("status") == "skipped":
                    self._emit(ProgressEvent(
                        event="step_skipped",
                        step_index=step.get("index", -1),
                        step_name=step.get("name", ""),
                        message=f"Skipping step '{step.get('name', '')}'.",
                    ))
                    if tracker:
                        tracker.skip_step(step.get("index", 0))
                    result.step_results.append(StepResult(
                        step_id=step.get("step_id", ""),
                        step_name=step.get("name", ""),
                        status="skipped",
                    ))
                    continue

                step_result = self._execute_step_with_retry(step, tracker)
                result.step_results.append(step_result)

                if step_result.status == "failed" and self.abort_on_failure:
                    self._emit(ProgressEvent(
                        event="pipeline_failed",
                        message=f"Pipeline aborted after step '{step.get('name', '')}' failed.",
                        data={"step_id": step.get("step_id", ""), "error": step_result.error},
                    ))
                    result.status = "failed"
                    break

            else:
                any_failed = any(r.status == "failed" for r in result.step_results)
                result.status = "partial" if any_failed else "completed"

        except Exception as exc:
            result.status = "failed"
            tb = traceback.format_exc()
            self._emit(ProgressEvent(
                event="pipeline_failed",
                message=f"Unhandled engine error: {exc}",
                data={"traceback": tb},
            ))
        finally:
            result.total_elapsed_ms = (time.monotonic() - self._start_time) * 1000
            result.events = list(self._events)
            if tracker:
                try:
                    tracker.stop_pipeline()
                    tracker.print_summary(steps)
                except Exception:
                    pass

        self._emit(ProgressEvent(
            event="pipeline_complete",
            message=result.summary(),
            data=result.to_dict(),
        ))
        return result

    # ------------------------------------------------------------------
    # ExecutionEngine ABC implementations
    # ------------------------------------------------------------------

    def run_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single step dict and return a result dict.

        This is the ABC-level interface.  Prefer
        :meth:`_execute_step_with_retry` internally.

        Args:
            step: A fully enriched pipeline step dict.

        Returns:
            ``{"status": "ok"|"error", "actions": [...], "tool_results": [...]}``
        """
        context = self.build_context(step)
        model = self.select_model(step, context)
        step = dict(step)
        step["_selected_model"] = model

        agent_name = step.get("agent", "")
        agent = self._agents.get(agent_name)
        if agent is None:
            return {
                "status": "error",
                "error": f"No agent registered for '{agent_name}'.",
                "actions": [],
                "tool_results": [],
            }

        output = agent.run(step, context)
        actions: List[AgentAction] = output.get("actions", [])
        tool_results = self._dispatch_actions(actions, context)

        return {
            "status": "ok",
            "output": output,
            "actions": actions,
            "tool_results": tool_results,
        }

    def build_context(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Assemble a context dict for the given step.

        Merges static system context with step-specific hints.  In a full
        deployment this would invoke the :class:`~context.ContextBuilder`;
        here it provides a well-structured envelope that agents can rely on.

        Args:
            step: The pipeline step dict.

        Returns:
            Context dict with ``"system"``, ``"step"``, ``"hints"``, and
            ``"model"`` keys.
        """
        hints = step.get("context_hints", [])
        return {
            "system": {
                "tools_available": (
                    self._tools.list_tools() if self._tools else []
                ),
                "agents_available": list(self._agents.keys()),
            },
            "step": {
                "step_id":     step.get("step_id", ""),
                "name":        step.get("name", ""),
                "description": step.get("description", ""),
                "agent":       step.get("agent", ""),
                "priority":    step.get("priority", "medium"),
                "tools":       step.get("tools", []),
            },
            "hints":   hints,
            "model":   step.get("model_hint", ""),
            "council": step.get("council_agents", []),
        }

    def select_model(self, step: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Return the model tag to use for this step.

        Prefers the step's ``model_hint`` field (set by the pipeline
        generator); falls back to a context-level model, then to an empty
        string (letting the model client decide).

        Args:
            step: The pipeline step dict.
            context: The assembled context payload.

        Returns:
            Ollama model identifier string (may be empty).
        """
        return (
            step.get("model_hint", "")
            or context.get("model", "")
            or ""
        )

    def handle_failure(self, step: Dict[str, Any], error: Exception) -> Dict[str, Any]:
        """Build a recovery action for a step-level failure.

        Args:
            step: The step that failed.
            error: The exception raised.

        Returns:
            ``{"action": "retry"|"abort", "reason": str}``
        """
        retries_left = step.get("max_retries", 0)
        if retries_left > 0:
            return {"action": "retry", "reason": str(error)}
        return {"action": "abort", "reason": str(error)}

    # ------------------------------------------------------------------
    # Internal: step runner with retry
    # ------------------------------------------------------------------

    def _execute_step_with_retry(
        self,
        step: Dict[str, Any],
        tracker: Optional[Any],
    ) -> StepResult:
        """Execute *step* with up to ``step["max_retries"]`` retry attempts.

        Emits :class:`ProgressEvent` objects at start, each retry, and
        completion/failure.  Updates the tracker if provided.

        Args:
            step: Fully enriched step dict.
            tracker: Optional :class:`~cli.display.ProgressTracker`.

        Returns:
            :class:`StepResult`.
        """
        idx = step.get("index", 0)
        name = step.get("name", "")
        step_id = step.get("step_id", str(uuid.uuid4()))
        max_retries = step.get("max_retries", 2)
        council = step.get("council_agents", [])

        self._emit(ProgressEvent(
            event="step_start",
            step_index=idx,
            step_name=name,
            message=f"Starting step [{idx}] '{name}' → agent={step.get('agent', '')}"
                    + (f", council={council}" if council else ""),
            data={"step": step},
        ))
        if tracker:
            try:
                tracker.start_step(idx, name)
            except Exception:
                pass

        step_start = time.monotonic()
        attempt = 0
        last_error = ""
        last_output: Dict[str, Any] = {}
        all_actions: List[AgentAction] = []
        all_tool_results: List[Dict[str, Any]] = []

        while attempt <= max_retries:
            try:
                context = self.build_context(step)
                model = self.select_model(step, context)
                step = dict(step)
                step["_selected_model"] = model
                step["_attempt"] = attempt

                # Council or solo dispatch.
                if council and len(council) > 1:
                    output, actions, tool_results = self._run_council(
                        step, context, council
                    )
                else:
                    output, actions, tool_results = self._run_solo(step, context)

                all_actions.extend(actions)
                all_tool_results.extend(tool_results)

                # Validate the primary agent's output.
                agent = self._agents.get(step.get("agent", ""))
                if agent and not agent.validate_output(output):
                    raise ValueError(
                        f"Agent '{step.get('agent', '')}' output failed validation."
                    )

                elapsed_ms = (time.monotonic() - step_start) * 1000
                self._emit(ProgressEvent(
                    event="step_complete",
                    step_index=idx,
                    step_name=name,
                    message=f"Step [{idx}] '{name}' completed in {elapsed_ms:.0f}ms.",
                    data={"output": output, "retries_used": attempt},
                    elapsed_ms=elapsed_ms,
                ))
                if tracker:
                    try:
                        tracker.complete_step(idx, success=True)
                    except Exception:
                        pass

                return StepResult(
                    step_id=step_id,
                    step_name=name,
                    status="completed",
                    output=output,
                    actions=all_actions,
                    tool_results=all_tool_results,
                    retries_used=attempt,
                    elapsed_ms=elapsed_ms,
                )

            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                if attempt < max_retries:
                    delay = _backoff(attempt)
                    self._emit(ProgressEvent(
                        event="step_retry",
                        step_index=idx,
                        step_name=name,
                        message=(
                            f"Step [{idx}] '{name}' failed (attempt {attempt + 1}/"
                            f"{max_retries + 1}). Retrying in {delay:.1f}s. Error: {exc}"
                        ),
                        data={"error": last_error, "attempt": attempt},
                    ))
                    time.sleep(delay)
                attempt += 1

        # All retries exhausted.
        elapsed_ms = (time.monotonic() - step_start) * 1000
        self._emit(ProgressEvent(
            event="step_failed",
            step_index=idx,
            step_name=name,
            message=f"Step [{idx}] '{name}' failed after {max_retries + 1} attempt(s).",
            data={"error": last_error},
            elapsed_ms=elapsed_ms,
        ))
        if tracker:
            try:
                tracker.complete_step(idx, success=False)
            except Exception:
                pass

        return StepResult(
            step_id=step_id,
            step_name=name,
            status="failed",
            output=last_output,
            actions=all_actions,
            tool_results=all_tool_results,
            retries_used=max_retries,
            elapsed_ms=elapsed_ms,
            error=last_error,
        )

    # ------------------------------------------------------------------
    # Solo and council dispatch
    # ------------------------------------------------------------------

    def _run_solo(
        self,
        step: Dict[str, Any],
        context: Dict[str, Any],
    ) -> tuple:
        """Run the step's assigned agent and dispatch its actions.

        Returns:
            ``(output, actions, tool_results)``
        """
        agent_name = step.get("agent", "")
        agent = self._agents.get(agent_name)
        if agent is None:
            raise RuntimeError(f"No agent registered for '{agent_name}'.")

        output = agent.run(step, context)
        actions: List[AgentAction] = output.get("actions", [])
        tool_results = self._dispatch_actions(actions, context)
        return output, actions, tool_results

    def _run_council(
        self,
        step: Dict[str, Any],
        context: Dict[str, Any],
        council: List[str],
    ) -> tuple:
        """Run all council agents and merge their actions.

        The first agent in *council* is the primary; its ``output`` is
        authoritative.  Subsequent agents' ``message`` and ``decision``
        actions are appended for traceability.

        Returns:
            ``(primary_output, merged_actions, tool_results)``
        """
        primary_output: Dict[str, Any] = {}
        all_actions: List[AgentAction] = []
        all_tool_results: List[Dict[str, Any]] = []

        for i, agent_name in enumerate(council):
            agent = self._agents.get(agent_name)
            if agent is None:
                # Emit a warning message and skip.
                all_actions.append(AgentAction.message(
                    f"[Council] Agent '{agent_name}' not found — skipping.",
                    agent="engine",
                    step_id=step.get("step_id", ""),
                ))
                continue

            output = agent.run(step, context)
            actions: List[AgentAction] = output.get("actions", [])

            if i == 0:
                # Primary — all actions, full output.
                primary_output = output
                all_actions.extend(actions)
                tool_results = self._dispatch_actions(actions, context)
                all_tool_results.extend(tool_results)
            else:
                # Reviewer — only informational actions (no tool_calls).
                review_actions = [
                    a for a in actions
                    if a.action_type in ("message", "decision")
                ]
                all_actions.extend(review_actions)

        return primary_output, all_actions, all_tool_results

    # ------------------------------------------------------------------
    # Action dispatcher
    # ------------------------------------------------------------------

    def _dispatch_actions(
        self,
        actions: List[AgentAction],
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Dispatch ``tool_call`` actions to the tool registry.

        Emits an ``action_dispatched`` event for every action processed.
        ``delegate``, ``message``, ``decision``, ``abort``, and ``noop``
        actions are recorded as events but not further executed (that
        responsibility belongs to the Supervisor / caller layer).

        Args:
            actions: Actions returned by an agent's ``run()`` call.
            context: Current step context (passed through to nested agents
                if a ``delegate`` action requires re-entry in the future).

        Returns:
            List of :class:`~tools.ToolResult` dicts from all
            ``tool_call`` actions that were dispatched.
        """
        tool_results: List[Dict[str, Any]] = []

        for action in actions:
            event_data: Dict[str, Any] = {"action": action.to_dict()}

            if action.action_type == "tool_call":
                tool_name = action.payload.get("tool", "")
                params = action.payload.get("params", {})
                result = self._invoke_tool(tool_name, params, action)
                tool_results.append(result)
                event_data["tool_result"] = result

            elif action.action_type == "abort":
                event_data["abort_reason"] = action.payload.get("reason", "")

            self._emit(ProgressEvent(
                event="action_dispatched",
                step_name=action.agent,
                message=(
                    f"Action [{action.action_type}] from {action.agent}"
                    + (f" → {action.payload.get('tool', '')}" if action.action_type == "tool_call" else "")
                ),
                data=event_data,
            ))

        return tool_results

    # ------------------------------------------------------------------
    # Tool invocation
    # ------------------------------------------------------------------

    def _invoke_tool(
        self,
        tool_name: str,
        params: Dict[str, Any],
        action: AgentAction,
    ) -> Dict[str, Any]:
        """Invoke *tool_name* via the tool registry.

        Returns a :class:`~tools.ToolResult` dict.  If no registry
        is configured, or the tool is missing, returns a structured error.

        Args:
            tool_name: Registered tool name.
            params: Parameter dict for the tool.
            action: The originating :class:`~agents.agent_action.AgentAction`.

        Returns:
            :class:`~tools.ToolResult` serialised as a dict.
        """
        if self._tools is None:
            return {
                "tool_name": tool_name,
                "success": False,
                "output": None,
                "error": "No tool registry configured.",
                "elapsed_ms": 0.0,
                "metadata": {},
            }

        try:
            raw = self._tools.invoke(tool_name, params)
            # ConcreteToolRegistry returns a ToolResult; normalise to dict.
            if hasattr(raw, "to_dict"):
                return raw.to_dict()
            if isinstance(raw, dict):
                return raw
            return {
                "tool_name": tool_name,
                "success": True,
                "output": raw,
                "error": None,
                "elapsed_ms": 0.0,
                "metadata": {},
            }
        except Exception as exc:
            return {
                "tool_name": tool_name,
                "success": False,
                "output": None,
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_ms": 0.0,
                "metadata": {"traceback": traceback.format_exc()},
            }

    # ------------------------------------------------------------------
    # Progress helpers
    # ------------------------------------------------------------------

    def _emit(self, event: ProgressEvent) -> None:
        """Record *event* and invoke the on_progress callback if set."""
        elapsed = (time.monotonic() - self._start_time) * 1000
        event.elapsed_ms = elapsed
        self._events.append(event)
        if self.on_progress:
            try:
                self.on_progress(event)
            except Exception:
                pass  # Never let callback errors crash the engine.

    def _make_tracker(self) -> Optional[Any]:
        """Attempt to instantiate a ProgressTracker; return None on failure."""
        try:
            from cli.display import ProgressTracker
            return ProgressTracker(console=self._console)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Streaming generator (pull-based alternative to on_progress)
    # ------------------------------------------------------------------

    def stream(
        self,
        pipeline: Any,
    ) -> Iterator[ProgressEvent]:
        """Execute *pipeline* and yield :class:`ProgressEvent` objects as they occur.

        This is a synchronous generator.  It uses ``on_progress`` internally
        to collect events and yields them in-order.

        Args:
            pipeline: A :class:`~execution.pipeline.Pipeline`
                or list of step dicts.

        Yields:
            :class:`ProgressEvent` in chronological order.

        Example::

            for event in engine.stream(pipeline):
                print(event.event, event.message)
        """
        collected: List[ProgressEvent] = []

        original_callback = self.on_progress

        def _capture(ev: ProgressEvent) -> None:
            collected.append(ev)
            if original_callback:
                original_callback(ev)

        self.on_progress = _capture
        try:
            self.run_pipeline(pipeline)
        finally:
            self.on_progress = original_callback

        yield from collected
