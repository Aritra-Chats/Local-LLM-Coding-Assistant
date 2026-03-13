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
        require_approval: bool = False,
        context_builder: Optional[Any] = None,
        performance_tracker: Optional[Any] = None,
        model_router: Optional[Any] = None,
    ) -> None:
        self._agents: Dict[str, Any] = agent_registry or {}
        self._tools: Optional[Any] = tool_registry
        self.abort_on_failure = abort_on_failure
        self.show_progress = show_progress
        self.on_progress = on_progress
        self._console = console
        self._require_approval = require_approval
        self._context_builder = context_builder
        # Learning loop: tracker collects metrics; router reads them for routing
        self._tracker: Optional[Any] = performance_tracker
        self._model_router: Optional[Any] = model_router
        # Wire tracker into router when both are provided
        if self._tracker is not None and self._model_router is not None:
            try:
                self._model_router.attach_tracker(self._tracker)
            except AttributeError:
                pass  # Router doesn't support attach_tracker yet
        self._events: List[ProgressEvent] = []
        self._start_time: float = 0.0
        self._progress_tracker: Optional[Any] = None  # Live UI progress tracker

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
        self._progress_tracker = tracker  # Store for access in approval prompts
        if tracker:
            try:
                tracker.start_pipeline(steps, task_name=f"Pipeline: {goal[:60]}")
            except Exception:
                tracker = None
                self._progress_tracker = None

        self._emit(ProgressEvent(
            event="pipeline_start",
            message=f"Starting pipeline '{pipeline_id[:8]}' with {len(steps)} steps.",
            data={"pipeline_id": pipeline_id, "goal": goal},
        ))

        try:
            # ----------------------------------------------------------------
            # Dependency-aware execution with parallel batching.
            # Steps that share the same dependency frontier and are all marked
            # can_parallelize=True are dispatched concurrently via a thread
            # pool.  Steps with unresolved dependencies or can_parallelize=False
            # are executed sequentially in their natural order.
            # ----------------------------------------------------------------
            import concurrent.futures as _cf

            completed_step_ids: set = set()
            step_result_map: Dict[str, StepResult] = {}

            # Build a quick lookup: step_id → step dict
            id_to_step = {s.get("step_id", ""): s for s in steps if s.get("step_id")}

            remaining = list(steps)

            while remaining:
                # Partition into steps that are ready (all depends_on satisfied)
                ready_batch: List[Dict[str, Any]] = []
                not_ready: List[Dict[str, Any]] = []

                for step in remaining:
                    if step.get("status") == "skipped":
                        ready_batch.append(step)
                        continue
                    deps = step.get("depends_on") or []
                    if all(d in completed_step_ids for d in deps):
                        ready_batch.append(step)
                    else:
                        not_ready.append(step)

                if not ready_batch:
                    # Circular dependency / unresolvable — run remaining sequentially
                    ready_batch = remaining
                    not_ready = []

                remaining = not_ready

                # Further partition ready_batch into parallel groups
                parallel_group: List[Dict[str, Any]] = []
                sequential_queue: List[Dict[str, Any]] = []

                for step in ready_batch:
                    if step.get("status") == "skipped":
                        sequential_queue.append(step)
                    elif step.get("can_parallelize", False):
                        parallel_group.append(step)
                    else:
                        sequential_queue.append(step)

                # ----------------------------------------------------------
                # Execute parallel group concurrently
                # ----------------------------------------------------------
                if parallel_group:
                    def _run_parallel_step(s):
                        return s, self._execute_step_with_retry(s, tracker)

                    max_workers = min(len(parallel_group), 4)
                    with _cf.ThreadPoolExecutor(
                        max_workers=max_workers,
                        thread_name_prefix="sentinel_step",
                    ) as pool:
                        futures = {pool.submit(_run_parallel_step, s): s for s in parallel_group}
                        for future in _cf.as_completed(futures):
                            try:
                                step, step_result = future.result()
                            except Exception as exc:
                                step = futures[future]
                                step_result = StepResult(
                                    step_id=step.get("step_id", ""),
                                    step_name=step.get("name", ""),
                                    status="failed",
                                    error=str(exc),
                                )
                            result.step_results.append(step_result)
                            step_result_map[step.get("step_id", "")] = step_result
                            step["status"] = step_result.status
                            step["elapsed_ms"] = step_result.elapsed_ms
                            completed_step_ids.add(step.get("step_id", ""))

                            if step_result.status == "failed" and self.abort_on_failure:
                                self._emit(ProgressEvent(
                                    event="pipeline_failed",
                                    message=f"Pipeline aborted after step '{step.get('name', '')}' failed.",
                                    data={"step_id": step.get("step_id", ""), "error": step_result.error},
                                ))
                                result.status = "failed"
                                remaining = []
                                break

                # ----------------------------------------------------------
                # Execute sequential queue one by one
                # ----------------------------------------------------------
                for step in sequential_queue:
                    if step.get("status") == "skipped":
                        self._emit(ProgressEvent(
                            event="step_skipped",
                            step_index=step.get("index", -1),
                            step_name=step.get("name", ""),
                            message=f"Skipping step '{step.get('name', '')}'.",
                        ))
                        if tracker:
                            try:
                                tracker.skip_step(step.get("index", 0))
                            except Exception:
                                pass
                        sr = StepResult(
                            step_id=step.get("step_id", ""),
                            step_name=step.get("name", ""),
                            status="skipped",
                        )
                        result.step_results.append(sr)
                        completed_step_ids.add(step.get("step_id", ""))
                        continue

                    step_result = self._execute_step_with_retry(step, tracker)
                    result.step_results.append(step_result)
                    step_result_map[step.get("step_id", "")] = step_result
                    step["status"] = step_result.status
                    step["elapsed_ms"] = step_result.elapsed_ms
                    completed_step_ids.add(step.get("step_id", ""))

                    if step_result.status == "failed" and self.abort_on_failure:
                        self._emit(ProgressEvent(
                            event="pipeline_failed",
                            message=f"Pipeline aborted after step '{step.get('name', '')}' failed.",
                            data={"step_id": step.get("step_id", ""), "error": step_result.error},
                        ))
                        result.status = "failed"
                        remaining = []
                        break

            if result.status != "failed":
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
            self._progress_tracker = None  # Clear reference

        self._emit(ProgressEvent(
            event="pipeline_complete",
            message=result.summary(),
            data=result.to_dict(),
        ))

        # Feed the learning tracker with pipeline-level metrics
        if self._tracker is not None:
            try:
                self._tracker.record_pipeline_result(result)
            except Exception:
                pass

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

        When a ConcreteContextBuilder is available, merges its rich output
        (RAG, symbol graph, dependency graph, synopsis) with the structural
        envelope.  Falls back to a minimal envelope otherwise.

        Args:
            step: The pipeline step dict.

        Returns:
            Context dict with ``"system"``, ``"step"``, ``"hints"``,
            ``"model"``, and optionally ``"rag"``, ``"synopsis"``, etc.
        """
        hints = step.get("context_hints", [])
        base = {
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
            "hints":        hints,
            "model":        step.get("model_hint", ""),
            "council":      step.get("council_agents", []),
            "project_root": step.get("project_root") or step.get("metadata", {}).get("project_root", ""),
        }

        # Enrich with ConcreteContextBuilder output if available
        if self._context_builder is not None:
            try:
                # Merge project_root into the step copy for the builder
                enriched_step = dict(step)
                enriched_step["project_root"] = base["project_root"]
                rich_ctx = self._context_builder.build(enriched_step)
                # Merge non-overlapping keys from rich context into base
                for key, value in rich_ctx.items():
                    if key not in base:
                        base[key] = value
                    elif key == "step" and isinstance(value, dict):
                        base["step"].update(value)
            except Exception:
                pass   # Never break execution due to context enrichment errors

        return base

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
                    # use_async_council=True enables asyncio.gather-based
                    # parallel dispatch (lower wall-clock latency on multi-model
                    # setups); falls back to ThreadPoolExecutor automatically.
                    if step.get("use_async_council") or step.get("metadata", {}).get("use_async_council"):
                        output, actions, tool_results = self.run_council_async(
                            step, context, council
                        )
                    else:
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

                # Record model success + tool outcomes in learning tracker
                if self._tracker is not None:
                    try:
                        _model = step.get("_selected_model", "")
                        _cat = step.get("task_category") or step.get("category") or step.get("agent", "")
                        self._tracker.record_model_call(
                            model=_model, category=str(_cat),
                            latency_ms=elapsed_ms, success=True,
                        )
                        self._tracker.record_tool_results(tool_results)
                    except Exception:
                        pass

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
                # Record model failure in learning tracker
                if self._tracker is not None:
                    try:
                        _model = step.get("_selected_model", "")
                        _cat = step.get("task_category") or step.get("category") or step.get("agent", "")
                        _elapsed = (time.monotonic() - step_start) * 1000
                        self._tracker.record_model_call(
                            model=_model, category=str(_cat),
                            latency_ms=_elapsed, success=False,
                        )
                    except Exception:
                        pass
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
        """Run the step's assigned agent, optionally applying CriticAgent review,
        and dispatch the resulting actions.

        Flow
        ----
        1. Primary agent generates actions (code, file writes, etc.)
        2. CriticAgent reviews write_file actions (if configured)
        3. Revised actions are dispatched to the ToolRegistry

        Returns:
            ``(output, actions, tool_results)``
        """
        agent_name = step.get("agent", "")
        agent = self._agents.get(agent_name)
        if agent is None:
            raise RuntimeError(f"No agent registered for '{agent_name}'.")

        output = agent.run(step, context)
        actions: List[AgentAction] = output.get("actions", [])

        # ------------------------------------------------------------------
        # Critic pass: review write_file actions produced by code-gen agents
        # before dispatching them. Critic is skipped for the critic agent
        # itself and for non-code agents.
        # ------------------------------------------------------------------
        critic = self._agents.get("critic")
        CODE_GEN_AGENTS = frozenset({"coding", "debugging", "devops"})
        if (
            critic is not None
            and agent_name in CODE_GEN_AGENTS
            and not step.get("skip_critic")
        ):
            try:
                revised_actions, critique = critic.review_actions(
                    actions=actions,
                    context=context,
                    step=step,
                )
                if critique is not None:
                    # Prepend a critique message so the progress log captures it
                    from agents.agent_action import AgentAction as _AA
                    critique_msg = _AA.message(
                        critic._format_critique_message(critique),
                        agent="critic",
                        step_id=step.get("step_id", ""),
                    )
                    actions = [critique_msg] + revised_actions
                    output = dict(output)
                    output["actions"] = actions
                    output["critique"] = critique
                else:
                    actions = revised_actions
            except Exception:
                pass  # Never let critic errors break the primary execution path

        tool_results = self._dispatch_actions(actions, context)
        return output, actions, tool_results

    def _run_council(
        self,
        step: Dict[str, Any],
        context: Dict[str, Any],
        council: List[str],
    ) -> tuple:
        """Run all council agents in PARALLEL and merge their outputs.

        All advisor agents execute concurrently via ThreadPoolExecutor.
        The first agent in *council* is the primary lead — its output is
        authoritative and its tool_call actions are dispatched.
        Subsequent advisor agents contribute message/decision actions only.

        After all advisors complete in parallel the lead model synthesises
        by receiving their perspectives appended to the context.

        Returns:
            ``(primary_output, merged_actions, tool_results)``
        """
        import concurrent.futures

        step_id = step.get("step_id", "")
        all_actions: List[AgentAction] = []
        all_tool_results: List[Dict[str, Any]] = []

        # Build per-agent tasks: (index, agent_name, agent)
        tasks = []
        for i, agent_name in enumerate(council):
            agent = self._agents.get(agent_name)
            if agent is None:
                all_actions.append(AgentAction.message(
                    f"[Council] Agent '{agent_name}' not found — skipping.",
                    agent="engine",
                    step_id=step_id,
                ))
            else:
                tasks.append((i, agent_name, agent))

        if not tasks:
            return {}, all_actions, all_tool_results

        # ------------------------------------------------------------------
        # Run all advisors (index > 0) in parallel; lead (index == 0) last
        # so it can incorporate advisor perspectives.
        # ------------------------------------------------------------------
        advisor_tasks = [(i, n, a) for i, n, a in tasks if i > 0]
        lead_task = next(((i, n, a) for i, n, a in tasks if i == 0), None)

        advisor_outputs: Dict[int, Dict[str, Any]] = {}

        def _run_agent(idx_name_agent):
            idx, name, agent = idx_name_agent
            try:
                return idx, agent.run(step, context)
            except Exception as exc:
                return idx, {
                    "status": "error",
                    "actions": [AgentAction.message(
                        f"[Council] Advisor '{name}' raised {type(exc).__name__}: {exc}",
                        agent=name,
                        step_id=step_id,
                    )],
                }

        # Execute advisors in parallel
        if advisor_tasks:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(advisor_tasks),
                thread_name_prefix="council_advisor",
            ) as pool:
                futures = {
                    pool.submit(_run_agent, t): t for t in advisor_tasks
                }
                for future in concurrent.futures.as_completed(futures):
                    try:
                        idx, output = future.result()
                        advisor_outputs[idx] = output
                    except Exception as exc:
                        # Collect any unexpected errors as messages
                        all_actions.append(AgentAction.message(
                            f"[Council] Advisor thread raised: {exc}",
                            agent="engine",
                            step_id=step_id,
                        ))

        # Collect advisor review actions (message + decision only, no tool_calls)
        advisor_perspectives: List[str] = []
        for idx in sorted(advisor_outputs.keys()):
            output = advisor_outputs[idx]
            actions = output.get("actions", [])
            review_actions = [
                a for a in actions
                if a.action_type in ("message", "decision")
            ]
            all_actions.extend(review_actions)
            # Collect text perspectives to pass to the lead
            for a in review_actions:
                if a.action_type == "message":
                    text = a.payload.get("text", "")
                    if text:
                        advisor_perspectives.append(text)

        # ------------------------------------------------------------------
        # Run the lead agent — enrich context with advisor perspectives
        # ------------------------------------------------------------------
        primary_output: Dict[str, Any] = {}
        if lead_task:
            _, lead_name, lead_agent = lead_task
            enriched_context = dict(context)
            if advisor_perspectives:
                enriched_context["council_perspectives"] = advisor_perspectives
                # Inject perspectives into system hint for LLM-driven agents
                existing_hints = list(enriched_context.get("hints", []))
                existing_hints.append(
                    "Council advisor perspectives:\n"
                    + "\n".join(f"  • {p}" for p in advisor_perspectives)
                )
                enriched_context["hints"] = existing_hints

            try:
                primary_output = lead_agent.run(step, enriched_context)
            except Exception as exc:
                primary_output = {
                    "status": "error",
                    "actions": [AgentAction.message(
                        f"[Council] Lead '{lead_name}' raised {type(exc).__name__}: {exc}",
                        agent=lead_name,
                        step_id=step_id,
                    )],
                }

            lead_actions: List[AgentAction] = primary_output.get("actions", [])
            all_actions.extend(lead_actions)
            lead_tool_results = self._dispatch_actions(lead_actions, context)
            all_tool_results.extend(lead_tool_results)

        return primary_output, all_actions, all_tool_results

    def run_council_async(
        self,
        step: Dict[str, Any],
        context: Dict[str, Any],
        council: List[str],
    ) -> tuple:
        """Run council agents using asyncio.gather for true async parallelism.

        This is an alternative to :meth:`_run_council` that drives the
        OllamaClient's ``generate_async`` coroutines via ``asyncio.gather``
        so that **all** advisor model calls fire concurrently at the HTTP
        level — not just at the thread level.

        Architecture
        ------------
        ::

            asyncio.gather(
                advisor_A.generate_async(model_a, prompt),
                advisor_B.generate_async(model_b, prompt),
                advisor_C.generate_async(model_c, prompt),
            )
                ↓  (all fire in parallel, lowest latency wins)
            lead_model.generate_async(lead_model, synthesised_prompt)

        Falls back to :meth:`_run_council` (ThreadPoolExecutor) automatically
        when:
          * No event loop is available / running.
          * Any agent's Ollama client lacks ``generate_async``.
          * asyncio is not importable (shouldn't happen on CPython 3.7+).

        Returns:
            Same ``(primary_output, all_actions, tool_results)`` tuple as
            :meth:`_run_council`.
        """
        import asyncio as _asyncio

        step_id = step.get("step_id", "")
        all_actions: List[AgentAction] = []
        all_tool_results: List[Dict[str, Any]] = []

        # Validate agents
        tasks = []
        for i, agent_name in enumerate(council):
            agent = self._agents.get(agent_name)
            if agent is None:
                all_actions.append(AgentAction.message(
                    f"[Council/async] Agent '{agent_name}' not found — skipping.",
                    agent="engine", step_id=step_id,
                ))
            else:
                tasks.append((i, agent_name, agent))

        if not tasks:
            return {}, all_actions, all_tool_results

        advisor_tasks = [(i, n, a) for i, n, a in tasks if i > 0]
        lead_task = next(((i, n, a) for i, n, a in tasks if i == 0), None)

        # ------------------------------------------------------------------
        # Build async coroutines for each advisor
        # ------------------------------------------------------------------
        async def _call_agent_async(idx: int, name: str, agent: Any) -> tuple:
            """Call one agent using its async client if available."""
            client = getattr(agent, "_ollama", None)
            model  = step.get("_selected_model") or getattr(agent, "_model", "")

            if client is not None and model and hasattr(client, "generate_async"):
                # Build the same prompt that run() would build
                # We call the agent's internal _llm_actions prompt-builder path
                # by running the whole agent.run() in the thread pool instead —
                # this keeps things consistent with the sync path.
                loop = _asyncio.get_event_loop()
                output = await loop.run_in_executor(
                    None,
                    lambda: agent.run(step, context),
                )
                return idx, output
            else:
                # Fall back to thread-pool run
                loop = _asyncio.get_event_loop()
                output = await loop.run_in_executor(
                    None,
                    lambda: agent.run(step, context),
                )
                return idx, output

        async def _gather_advisors() -> Dict[int, Any]:
            if not advisor_tasks:
                return {}
            coros = [
                _call_agent_async(i, n, a)
                for i, n, a in advisor_tasks
            ]
            results = await _asyncio.gather(*coros, return_exceptions=True)
            outputs: Dict[int, Any] = {}
            for res in results:
                if isinstance(res, Exception):
                    all_actions.append(AgentAction.message(
                        f"[Council/async] Advisor raised: {res}",
                        agent="engine", step_id=step_id,
                    ))
                else:
                    idx, output = res
                    outputs[idx] = output
            return outputs

        # ------------------------------------------------------------------
        # Run the gather — acquire or create event loop as needed
        # ------------------------------------------------------------------
        try:
            try:
                loop = _asyncio.get_event_loop()
                if loop.is_running():
                    # Already inside an async context — use nest_asyncio or
                    # fall back to the ThreadPoolExecutor council path.
                    raise RuntimeError("loop already running")
                advisor_outputs = loop.run_until_complete(_gather_advisors())
            except RuntimeError:
                # Create a fresh loop in the current thread
                loop = _asyncio.new_event_loop()
                try:
                    advisor_outputs = loop.run_until_complete(_gather_advisors())
                finally:
                    loop.close()
        except Exception:
            # Any async failure: gracefully fall back to thread-based council
            return self._run_council(step, context, council)

        # ------------------------------------------------------------------
        # Collect advisor perspectives (same logic as _run_council)
        # ------------------------------------------------------------------
        advisor_perspectives: List[str] = []
        for idx in sorted(advisor_outputs.keys()):
            output = advisor_outputs[idx]
            actions = output.get("actions", [])
            review_actions = [
                a for a in actions
                if a.action_type in ("message", "decision")
            ]
            all_actions.extend(review_actions)
            for a in review_actions:
                if a.action_type == "message":
                    text = a.payload.get("text", "")
                    if text:
                        advisor_perspectives.append(text)

        # ------------------------------------------------------------------
        # Lead agent synthesis
        # ------------------------------------------------------------------
        primary_output: Dict[str, Any] = {}
        if lead_task:
            _, lead_name, lead_agent = lead_task
            enriched_context = dict(context)
            if advisor_perspectives:
                enriched_context["council_perspectives"] = advisor_perspectives
                existing_hints = list(enriched_context.get("hints", []))
                existing_hints.append(
                    "Council advisor perspectives (async):\n"
                    + "\n".join(f"  • {p}" for p in advisor_perspectives)
                )
                enriched_context["hints"] = existing_hints
            try:
                primary_output = lead_agent.run(step, enriched_context)
            except Exception as exc:
                primary_output = {
                    "status": "error",
                    "actions": [AgentAction.message(
                        f"[Council/async] Lead '{lead_name}' raised "
                        f"{type(exc).__name__}: {exc}",
                        agent=lead_name, step_id=step_id,
                    )],
                }
            lead_actions: List[AgentAction] = primary_output.get("actions", [])
            all_actions.extend(lead_actions)
            lead_tool_results = self._dispatch_actions(lead_actions, context)
            all_tool_results.extend(lead_tool_results)

        return primary_output, all_actions, all_tool_results

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
                # ── Semantic validation ──────────────────────────────────
                try:
                    from core.validator import validate_tool_call
                    _proj = context.get("project_root", "") if context else ""
                    _vr = validate_tool_call(tool_name, params, _proj or None)
                    if not _vr.ok:
                        _blocked = {"tool_name": tool_name, "success": False,
                                    "output": None, "error": f"Validation: {_vr.reason}",
                                    "elapsed_ms": 0.0, "metadata": params}
                        tool_results.append(_blocked)
                        event_data["tool_result"] = _blocked
                        self._emit(ProgressEvent(
                            event="action_dispatched", step_name=action.agent,
                            message=f"[{tool_name}] blocked: {_vr.reason}",
                            data=event_data))
                        continue
                except Exception:
                    pass  # validator unavailable
                # ── Approve before apply ──────────────────────────────────
                if self._require_approval and tool_name in (
                    "write_file", "git_commit", "run_shell", "install_dependency", "project_initializer"
                ):
                    approved = self._request_approval(tool_name, params)
                    if not approved:
                        skipped = {
                            "tool_name": tool_name,
                            "success": False,
                            "output": None,
                            "error": "Declined by user.",
                            "elapsed_ms": 0.0,
                            "metadata": params,
                        }
                        tool_results.append(skipped)
                        event_data["tool_result"] = skipped
                        self._emit(ProgressEvent(
                            event="action_dispatched",
                            step_name=action.agent,
                            message=f"[{tool_name}] declined by user",
                            data=event_data,
                        ))
                        continue
                result = self._invoke_tool(tool_name, params, action, context)
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
    # Approval prompt
    # ------------------------------------------------------------------

    def _request_approval(self, tool_name: str, params: Dict[str, Any]) -> bool:
        """Ask the user to approve a destructive tool call.

        Returns True if approved, False if declined.
        """
        try:
            from rich.console import Console as _Console
            from rich.panel import Panel as _Panel
            
            # Pause the live progress display to allow clean user input
            if self._progress_tracker is not None:
                try:
                    self._progress_tracker.pause()
                except Exception:
                    pass
            
            con = self._console or _Console()
            if tool_name == "write_file":
                detail = f"Write → [bold]{params.get('path', '?')}[/bold]"
            elif tool_name == "run_shell":
                detail = f"Run shell → [bold]{params.get('command', '?')}[/bold]"
            elif tool_name == "git_commit":
                detail = f"Git commit: [bold]{params.get('message', '?')}[/bold]"
            elif tool_name == "install_dependency":
                detail = f"Install packages: [bold]{params.get('packages', [])}[/bold]"
            elif tool_name == "project_initializer":
                detail = f"Init project → [bold]{params.get('project_name', '?')}[/bold] ({params.get('project_type', 'auto-detect')})"
            else:
                detail = str(params)
            con.print(_Panel(
                f"[yellow]Sentinel wants to execute:[/yellow]\n{detail}",
                title="[bold yellow]⚠ Approval Required[/bold yellow]",
                border_style="yellow",
            ))
            
            # Use standard input with explicit prompt
            import sys
            sys.stdout.write("  Apply? [Y/n] › ")
            sys.stdout.flush()
            answer = sys.stdin.readline().strip().lower()
            
            approved = answer in ("", "y", "yes")
            
            # Resume the live progress display
            if self._progress_tracker is not None:
                try:
                    self._progress_tracker.resume()
                except Exception:
                    pass
            
            return approved
        except Exception as e:
            # Resume on error
            if self._progress_tracker is not None:
                try:
                    self._progress_tracker.resume()
                except Exception:
                    pass
            return True   # Non-interactive (CI/test) — default allow

    # ------------------------------------------------------------------
    # Tool invocation
    # ------------------------------------------------------------------

    def _invoke_tool(
        self,
        tool_name: str,
        params: Dict[str, Any],
        action: AgentAction,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Invoke *tool_name* via the tool registry.

        Returns a :class:`~tools.ToolResult` dict.  If no registry
        is configured, or the tool is missing, returns a structured error.

        Args:
            tool_name: Registered tool name.
            params: Parameter dict for the tool.
            action: The originating :class:`~agents.agent_action.AgentAction`.
            context: Optional context dict containing project_root.

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

        # Inject project_root into params for path-aware tools
        if context and context.get("project_root"):
            _proj = context["project_root"]
            if tool_name in ("write_file", "read_file"):
                params = {**params, "project_root": _proj}
            elif tool_name in ("search_code", "run_tests"):
                # Default path to the project root when the agent left it at "."
                if not params.get("path") or params.get("path") == ".":
                    params = {**params, "path": _proj}

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
