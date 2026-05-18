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
import threading
import traceback
import uuid
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from agents.agent_action import AgentAction


# ---------------------------------------------------------------------------
# Abort exception (raised when supervisor signals abort)
# ---------------------------------------------------------------------------

class AbortException(Exception):
    """Raised by the execution loop when supervisor requests an abort."""
    pass

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
        self._auto_approve_session: bool = False  # Auto-approve all requests if True
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

        # Architecture state — set when project_initializer runs; used to route
        # write_file calls into frontend/ or backend/ for fullstack projects.
        self._project_architecture: str = "single"   # "single" | "frontend" | "backend" | "fullstack"
        self._project_dirs: Dict[str, str] = {}       # keys: "frontend", "backend"

        # FileChangeMap — one per pipeline run, tracks all write_file outcomes
        from core.file_change_map import FileChangeMap
        self.file_change_map = FileChangeMap()
        self._session_id: str = ""  # set by run_pipeline from context if available

        # ── Supervisor integration ─────────────────────────────────────────────
        import queue as _queue
        self._supervisor_agent: Optional[Any] = None
        self._supervisor_bus: Optional[Any] = None        # SupervisorBus
        self._supervisor_loop: Optional[Any] = None       # AsyncSupervisorLoop
        self._pause_event: threading.Event = threading.Event()
        self._pause_event.set()    # initially un-paused (set = allowed to run)
        self._fix_queue: _queue.Queue = _queue.Queue()   # FixProposal objects
        self._abort_flag: bool = False
        self._abort_reason: str = ""
        self._shell_command_history: List[str] = []
        self._known_files: List[str] = []

    # ------------------------------------------------------------------
    # Supervisor integration API
    # ------------------------------------------------------------------

    def attach_supervisor(
        self,
        supervisor: Any,
        tracker: Optional[Any] = None,
    ) -> None:
        """Create the :class:`~core.supervisor_bus.SupervisorBus` and start the
        :class:`~core.async_supervisor.AsyncSupervisorLoop`.

        Call this after engine construction, before ``run_pipeline()``.

        Args:
            supervisor: A :class:`~agents.supervisor.ConcreteSupervisorAgent`.
            tracker: Optional :class:`~cli.progress_tracker.ConcreteProgressTracker`.
        """
        from core.supervisor_bus import SupervisorBus
        self._supervisor_agent = supervisor
        self._supervisor_bus = SupervisorBus()
        self._supervisor_loop = supervisor.start_async_monitoring(
            self._supervisor_bus, self, tracker
        )

    def pause(self, reason: str = "") -> None:
        """Block the execution loop until :meth:`resume` is called.

        Called by the supervisor thread when it needs to inject fix actions
        before the engine attempts the next retry.
        """
        self._pause_event.clear()

    def resume(self) -> None:
        """Unblock the execution loop after fix actions have been dispatched."""
        self._pause_event.set()

    def inject_fix_and_retry(self, step_id: str, fix_proposal: Any) -> None:
        """Dispatch supervisor fix actions and signal the engine to retry.

        This is called from the supervisor daemon thread.  It puts the
        proposal on :attr:`_fix_queue` (thread-safe) and resumes the engine.

        Args:
            step_id: UUID of the step that failed.
            fix_proposal: A :class:`~core.supervisor_bus.FixProposal`.
        """
        self._fix_queue.put(fix_proposal)
        self.resume()

    def abort(self, reason: str) -> None:
        """Signal the engine to abort the current pipeline run.

        Sets :attr:`_abort_flag` which the execution loop checks at the top
        of each iteration.  Also resumes a paused engine so it can see the flag.

        Args:
            reason: Human-readable abort message shown to the user.
        """
        self._abort_reason = reason
        self._abort_flag = True
        self.resume()  # unblock any waiting pause

    def _shell_history_context(self) -> List[str]:
        """Return the recent shell command history for supervisor review."""
        return list(self._shell_command_history[-25:])

    def review_shell_command(self, tool_name: str, params: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Ask the supervisor model whether a shell-like tool call should run.

        The supervisor can return a runtime policy dict with keys such as
        ``decision`` (``run``, ``skip``, ``bounded_run``), ``timeout_seconds``,
        ``display_label``, and ``reason``.
        """
        reviewer = self._supervisor_agent
        review_fn = getattr(reviewer, "review_shell_tool_call", None) if reviewer is not None else None
        if not callable(review_fn):
            return {}

        review_context = dict(context or {})
        review_context.setdefault("shell_command_history", self._shell_history_context())
        try:
            review = review_fn(tool_name=tool_name, params=dict(params), context=review_context)
            return review if isinstance(review, dict) else {}
        except Exception:
            return {}

    def _record_shell_history(self, tool_name: str, params: Dict[str, Any], result: Optional[Dict[str, Any]] = None) -> None:
        """Store shell commands so the supervisor can avoid rerunning them."""
        commands: List[str] = []

        def _should_record(command: str) -> bool:
            first = (command.strip().split(maxsplit=1)[0] if command.strip() else "").lower().strip("\"'")
            if first in {"mkdir", "md", "rmdir", "rd", "write", "read", "modify"}:
                return False
            return True

        if tool_name == "run_shell":
            command = str(params.get("command", "")).strip()
            if command and _should_record(command):
                commands.append(command)
        elif tool_name == "project_initializer":
            output = result.get("output") if isinstance(result, dict) else {}
            if isinstance(output, dict):
                steps = output.get("steps") or []
                for step in steps:
                    if isinstance(step, dict):
                        command = str(step.get("command", "")).strip()
                        if command and _should_record(command):
                            commands.append(command)

        for command in commands:
            if command not in self._shell_command_history:
                self._shell_command_history.append(command)

    def _refresh_known_files(self, project_root: str) -> None:
        """Refresh the in-memory known-files list from disk (bounded scan)."""
        if not project_root or not os.path.isdir(project_root):
            self._known_files = []
            return

        max_files = 800
        skip_dirs = {
            "node_modules", "venv", ".venv", ".git", "__pycache__",
            ".next", "dist", "build", ".pytest_cache", ".mypy_cache",
        }
        known: List[str] = []
        for root, dirs, files in os.walk(project_root):
            dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith(".")]
            for fname in files:
                if fname.startswith("."):
                    continue
                abs_path = os.path.join(root, fname)
                try:
                    rel_path = os.path.relpath(abs_path, project_root).replace("\\", "/")
                except ValueError:
                    rel_path = abs_path.replace("\\", "/")
                known.append(rel_path)
                if len(known) >= max_files:
                    self._known_files = sorted(set(known))
                    return
        self._known_files = sorted(set(known))

    def _detect_stack_from_known_files(self, project_root: str) -> Dict[str, str]:
        """Infer a lightweight stack snapshot from marker files."""
        stack: Dict[str, str] = {}

        def _exists(rel_path: str) -> bool:
            return os.path.exists(os.path.join(project_root, rel_path))

        if _exists("package.json"):
            stack["runtime"] = "Node.js"
        if _exists("requirements.txt") or _exists("pyproject.toml"):
            stack["runtime"] = "Python"
        if _exists("Dockerfile") or _exists("docker-compose.yml") or _exists("docker-compose.yaml"):
            stack["container"] = "Docker"
        if _exists("vite.config.js") or _exists("vite.config.ts"):
            stack["build"] = "Vite"
        if _exists("next.config.js") or _exists("next.config.ts"):
            stack["frontend"] = "Next.js"
        if _exists("angular.json"):
            stack["frontend"] = "Angular"

        package_json = os.path.join(project_root, "package.json")
        if os.path.isfile(package_json):
            try:
                with open(package_json, "r", encoding="utf-8", errors="ignore") as fh:
                    pkg = fh.read().lower()
                if '"react"' in pkg:
                    stack["frontend"] = "React"
                elif '"vue"' in pkg:
                    stack["frontend"] = "Vue"
                elif '"svelte"' in pkg:
                    stack["frontend"] = "Svelte"
                if '"express"' in pkg:
                    stack["backend"] = "Express"
                elif '"fastify"' in pkg:
                    stack["backend"] = "Fastify"
            except Exception:
                pass

        req_txt = os.path.join(project_root, "requirements.txt")
        if os.path.isfile(req_txt):
            try:
                with open(req_txt, "r", encoding="utf-8", errors="ignore") as fh:
                    req = fh.read().lower()
                if "fastapi" in req:
                    stack["backend"] = "FastAPI"
                elif "django" in req:
                    stack["backend"] = "Django"
                elif "flask" in req:
                    stack["backend"] = "Flask"
            except Exception:
                pass

        return stack

    def _workspace_snapshot(self, project_root: str) -> Dict[str, Any]:
        """Build a compact, per-step codebase snapshot for all subagents."""
        stack = self._detect_stack_from_known_files(project_root) if project_root else {}
        recent_changes = [
            {
                "path": ev.logical_path,
                "operation": ev.operation,
                "agent": ev.agent,
                "step_id": ev.step_id,
            }
            for ev in self.file_change_map.all_events()[-20:]
        ]
        return {
            "project_root": project_root,
            "known_files_count": len(self._known_files),
            "known_files_preview": self._known_files[:120],
            "stack": stack,
            "recent_changes": recent_changes,
            "summary": (
                f"Known files: {len(self._known_files)} | "
                + ("Stack: " + ", ".join(f"{k}={v}" for k, v in stack.items()) if stack else "Stack: unknown")
            ),
        }

    def _looks_like_long_running_shell(self, command: str) -> bool:
        """Return True for dev/watch/server shell commands that should be bounded."""
        cmd = (command or "").strip().lower()
        if not cmd:
            return False
        patterns = [
            r"\bnpm\s+run\s+(dev|start|watch)\b",
            r"\bpnpm\s+(dev|start|watch)\b",
            r"\byarn\s+(dev|start|watch)\b",
            r"\bnpx\s+vite\b",
            r"\bnext\s+dev\b",
            r"\bwebpack\b.*\b--watch\b",
            r"\bnodemon\b",
            r"\buvicorn\b.*\b--reload\b",
            r"\bflask\s+run\b",
            r"\bdjango-admin\s+runserver\b",
            r"\bpython\s+.*manage\.py\s+runserver\b",
        ]
        return any(re.search(p, cmd) for p in patterns)

    def _maybe_skip_project_initializer(self, params: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Tuple[bool, str]:
        """Detect already-initialized projects and skip duplicate scaffolding."""
        if bool(params.get("force", False)):
            return False, ""

        project_root = ""
        if context:
            project_root = str(context.get("project_root", "") or "")
        if not project_root or not os.path.isdir(project_root):
            return False, ""

        if not self._known_files:
            self._refresh_known_files(project_root)

        marker_files = {
            "package.json", "requirements.txt", "pyproject.toml",
            "manage.py", "Dockerfile", "vite.config.ts", "vite.config.js",
            "next.config.js", "next.config.ts",
        }
        known_set = {p.replace("\\", "/").strip() for p in self._known_files}
        marker_hit = any(m in known_set for m in marker_files)
        code_like_count = sum(
            1 for p in known_set
            if p.endswith((".py", ".js", ".ts", ".tsx", ".jsx", ".html", ".css"))
        )

        if marker_hit or code_like_count >= 3:
            stack = self._detect_stack_from_known_files(project_root)
            stack_hint = ", ".join(f"{k}={v}" for k, v in stack.items()) or "existing files"
            return True, f"Project already appears initialized ({stack_hint}); skipping duplicate project_initializer."

        return False, ""

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

        # Capture session ID for FileChangeMap persistence
        if hasattr(pipeline, "session_id"):
            self._session_id = pipeline.session_id or ""
        elif isinstance(pipeline, list) and pipeline:
            self._session_id = next(
                (s.get("session_id", "") for s in pipeline if isinstance(s, dict)), ""
            )

        # Reset FileChangeMap for this pipeline run
        from core.file_change_map import FileChangeMap
        self.file_change_map = FileChangeMap()
        self._shell_command_history = []
        self._known_files = []

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
                        thread_name_prefix=f"sentinel-{step.get('agent', 'step')}",
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

        except AbortException as _ae:
            result.status = "failed"
            _abort_msg = str(_ae)
            self._emit(ProgressEvent(
                event="pipeline_failed",
                message=f"Pipeline aborted by supervisor: {_abort_msg}",
                data={"abort_reason": _abort_msg},
            ))
            # Print structured abort message to console
            try:
                import sys
                print(f"\n[Sentinel] Pipeline aborted:\n{_abort_msg}", file=sys.stderr)
            except Exception:
                pass
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

        # ── Notify supervisor that the pipeline is done ────────────────────────
        if self._supervisor_bus is not None:
            try:
                from core.supervisor_bus import BusEvent, BusEventType
                self._supervisor_bus.emit(BusEvent(
                    type=BusEventType.PIPELINE_DONE,
                    step_id="", step_index=-1, step_name="pipeline",
                ))
            except Exception:
                pass
        if self._supervisor_loop is not None:
            try:
                self._supervisor_loop.stop()
            except Exception:
                pass

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

        # Persist FileChangeMap for this run
        try:
            from config.settings import SESSIONS_DIR
            sid = self._session_id or run_id[:8]
            fcm_path = SESSIONS_DIR / f"{sid}_file_changes.json"
            self.file_change_map.save(fcm_path)
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
            "shell_command_history": self._shell_history_context(),
            # Numeric pipeline index — used by _dispatch_actions to update the
            # live progress label via update_step_action.  Without this,
            # context.get("step_index") returns None, the int() cast on the
            # UUID-shaped step_id raises ValueError (swallowed), and the label
            # never updates from the static step name.
            "step_index": step.get("index", 0),
            # Architecture set by project_initializer — used by the coding agent
            # to generate correctly prefixed write_file paths.
            "project_architecture": self._project_architecture,
            "project_dirs":         dict(self._project_dirs),
            "known_files": list(self._known_files),
        }

        project_root = base.get("project_root", "")
        if project_root:
            self._refresh_known_files(project_root)
            base["known_files"] = list(self._known_files)
            base["workspace_snapshot"] = self._workspace_snapshot(project_root)

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

        # ── SDLC gate: entry contract check ───────────────────────────────────
        try:
            from core.step_contract import StepContract, ContractChecker
            _contract = StepContract.from_dict(step.get("contract"))
            if _contract:
                _ctx_for_entry = {"project_root": "", "output_dir": ""}
                try:
                    _ctx_for_entry = self.build_context(step)
                except Exception:
                    pass
                _entry_result = ContractChecker.check_entry(_contract, _ctx_for_entry)
                if not _entry_result.passed and self._supervisor_bus is not None:
                    from core.supervisor_bus import BusEvent, BusEventType
                    self._supervisor_bus.emit(BusEvent(
                        type=BusEventType.STEP_ENTRY_FAILED,
                        step_id=step_id,
                        step_index=idx,
                        step_name=name,
                        error=f"Entry check failed: {_entry_result.details}",
                        context=_ctx_for_entry,
                        extra={"failed_items": _entry_result.failed_items,
                               "agent": step.get("agent", "")},
                    ))
                    self.pause("Waiting for supervisor to fix entry requirements")
                    self._pause_event.wait()
                    # After supervisor fixes, check abort flag
                    if self._abort_flag:
                        raise AbortException(self._abort_reason)
        except AbortException:
            raise
        except Exception:
            pass  # Never let contract checks break execution

        while attempt <= max_retries:
            # ── Pause / abort check at the top of every iteration ─────────────
            self._pause_event.wait()  # blocks if supervisor called pause()
            if self._abort_flag:
                raise AbortException(self._abort_reason)

            # ── Drain any pending fix proposals from the supervisor ────────────
            _pending_fixes = []
            while not self._fix_queue.empty():
                try:
                    _pending_fixes.append(self._fix_queue.get_nowait())
                except Exception:
                    break
            for _fix in _pending_fixes:
                try:
                    _fix_actions = getattr(_fix, "fix_actions", []) or []
                    _fix_results = self._dispatch_actions(_fix_actions, {
                        "step_index": idx, "step_name": name,
                        "project_root": "",
                    })
                    all_tool_results.extend(_fix_results)
                except Exception:
                    pass

            try:
                context = self.build_context(step)
                model = self.select_model(step, context)
                step = dict(step)
                step["_selected_model"] = model
                step["_attempt"] = attempt

                # ── Online mode: override _selected_model with the cloud/external
                # model tag selected by OnlineModelDiscoveryEngine.  The local
                # ConcreteModelRouter only knows about offline Ollama models, so
                # select_model() above always returns a local tag (e.g. "codellama:13b")
                # even in online mode.  We must override it here so agents send the
                # correct model name to whichever provider client was injected.
                _online_sel = (
                    step.get("selected_model")
                    or step.get("metadata", {}).get("selected_model")
                )
                if isinstance(_online_sel, dict):
                    _online_provider = _online_sel.get("provider", "ollama_local")
                    _online_model_tag = _online_sel.get("model", "")
                    if _online_provider != "ollama_local" and _online_model_tag:
                        # Probe the full fallback chain before stamping
                        # _selected_model.  _resolve_with_fallback returns the
                        # effective model tag for whichever tier succeeded,
                        # preventing a cloud model name from being sent to a
                        # local endpoint (which always fails with HTTP 404).
                        _cloud_fallbacks = (
                            _online_sel.get("cloud_fallback_list", [])
                            if isinstance(_online_sel, dict) else []
                        )
                        _probe_client, _effective_tag = self._resolve_with_fallback(
                            _online_provider,
                            _online_model_tag,
                            cloud_fallback_list=_cloud_fallbacks,
                        )
                        if _probe_client is not None:
                            step["_selected_model"] = _effective_tag
                        # else: retain the local model from select_model()

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
                        _sel = step.get("selected_model") or step.get("metadata", {}).get("selected_model") or {}
                        _provider = _sel.get("provider", "ollama_local") if isinstance(_sel, dict) else "ollama_local"
                        self._tracker.record_model_call(
                            model=_model, category=str(_cat),
                            latency_ms=elapsed_ms, success=True,
                            provider=_provider,
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

                # ── SDLC gate: exit contract check ────────────────────────────
                try:
                    from core.step_contract import StepContract, ContractChecker
                    _exit_contract = StepContract.from_dict(step.get("contract"))
                    if _exit_contract and self._supervisor_bus is not None:
                        _proj_root = context.get("project_root", "") if context else ""
                        _known = context.get("known_files", []) if context else []
                        _exit_result = ContractChecker.check_exit(
                            _exit_contract, _proj_root, _known
                        )
                        if not _exit_result.passed:
                            from core.supervisor_bus import BusEvent, BusEventType
                            self._supervisor_bus.emit(BusEvent(
                                type=BusEventType.STEP_EXIT_FAILED,
                                step_id=step_id,
                                step_index=idx,
                                step_name=name,
                                error=f"Exit check failed: {_exit_result.details}",
                                context=context or {},
                                attempt=attempt,
                                extra={"failed_items": _exit_result.failed_items},
                            ))
                            self.pause("Waiting for supervisor to fix exit criteria")
                            self._pause_event.wait()
                            if self._abort_flag:
                                raise AbortException(self._abort_reason)
                            # Drain supervisor fix actions
                            while not self._fix_queue.empty():
                                try:
                                    _xfix = self._fix_queue.get_nowait()
                                    _xacts = getattr(_xfix, "fix_actions", []) or []
                                    self._dispatch_actions(_xacts, context or {})
                                except Exception:
                                    break
                            attempt += 1
                            continue
                except AbortException:
                    raise
                except Exception:
                    pass  # Never break execution for contract checks

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

            except AbortException:
                raise  # propagate abort without retrying
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                # Record model failure in learning tracker
                if self._tracker is not None:
                    try:
                        _model = step.get("_selected_model", "")
                        _cat = step.get("task_category") or step.get("category") or step.get("agent", "")
                        _elapsed = (time.monotonic() - step_start) * 1000
                        _sel = step.get("selected_model") or step.get("metadata", {}).get("selected_model") or {}
                        _provider = _sel.get("provider", "ollama_local") if isinstance(_sel, dict) else "ollama_local"
                        self._tracker.record_model_call(
                            model=_model, category=str(_cat),
                            latency_ms=_elapsed, success=False,
                            provider=_provider,
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
    # Inference client resolver (online mode)
    # ------------------------------------------------------------------

    def _resolve_inference_client(self, provider: str, model_tag: str) -> Optional[Any]:
        """Return the appropriate inference client for a provider.

        Returns ``None`` for ``"ollama_local"`` (agents use their default
        client in that case) or when the client cannot be constructed.

        Args:
            provider:  One of ``"ollama_local"``, ``"ollama_cloud"``,
                       ``"anthropic"``, ``"openai"``, ``"google"``.
            model_tag: Model identifier (used only for validation/logging).

        Returns:
            An inference client instance, or ``None``.
        """
        import os
        if provider == "ollama_local" or not provider:
            return None
        if provider == "ollama_cloud":
            try:
                from models.ollama_cloud_client import OllamaCloudClient
                return OllamaCloudClient(api_key=os.environ.get("OLLAMA_API_KEY", ""))
            except Exception:
                return None
        if provider in ("anthropic", "openai", "google"):
            try:
                from models.external_api_client import ExternalAPIClient
                return ExternalAPIClient(provider)
            except Exception:
                return None
        return None

    def _resolve_with_fallback(
        self,
        provider: str,
        model_tag: str,
        cloud_fallback_list: Optional[List[str]] = None,
    ) -> "tuple[Optional[Any], str]":
        """Resolve an inference client with automatic provider fallback.

        Implements the chain: **ollama_cloud → external provider → local**.

        If the requested provider cannot be constructed (missing API key,
        import failure, etc.) the method tries the next tier in the chain
        rather than immediately returning ``None``.  This ensures agents
        always get a working client when *any* online provider is configured,
        and only degrade to local Ollama when none is available.

        Args:
            provider:  The originally selected provider string.
            model_tag: The model tag associated with that provider.

        Returns:
            A ``(client, effective_model_tag)`` tuple.  ``client`` is ``None``
            when no online provider is available (caller should use local
            Ollama).  ``effective_model_tag`` may differ from *model_tag* when
            fallback to a different provider occurs — callers must use this
            value to stamp ``step["_selected_model"]`` so agents never send a
            cloud model name to a non-cloud endpoint.
        """
        import os

        # Local — no client needed; caller uses its default Ollama instance
        if provider == "ollama_local" or not provider:
            return None, model_tag

        # ── Tier 1: Ollama Cloud ─────────────────────────────────────────────
        if provider == "ollama_cloud":
            # is_available() only checks API-key validity against /api/tags.
            # is_model_available() makes a 1-token probe to /api/generate to
            # verify the specific model is accessible on this plan tier.
            # This catches 404 "tier-locked" responses before the real call.
            try:
                from models.ollama_cloud_client import OllamaCloudClient
                client = OllamaCloudClient(
                    api_key=os.environ.get("OLLAMA_API_KEY", "")
                )
                if client.is_model_available(model_tag):
                    return client, model_tag
            except Exception:
                pass

            # Primary tier-locked or unavailable — walk ranked fallback list.
            for fallback_tag in (cloud_fallback_list or []):
                try:
                    from models.ollama_cloud_client import OllamaCloudClient
                    fb_client = OllamaCloudClient(
                        api_key=os.environ.get("OLLAMA_API_KEY", "")
                    )
                    if fb_client.is_model_available(fallback_tag):
                        return fb_client, fallback_tag
                except Exception:
                    continue
            # All cloud models exhausted → fall through to Tier 2

        # ── Tier 2: External provider ────────────────────────────────────────
        # When the original provider was "ollama_cloud" (and failed), try all
        # external providers in affinity order.
        # When the original provider IS an external provider, try it first
        # then fall through to the others on failure.
        try:
            from models.external_api_client import ExternalAPIClient
            if provider in ("anthropic", "openai", "google"):
                ext_order = [provider] + [
                    p for p in ("anthropic", "google", "openai") if p != provider
                ]
            else:
                ext_order = ["anthropic", "google", "openai"]

            for ext_provider in ext_order:
                try:
                    ext_client = ExternalAPIClient(ext_provider)
                    if ext_client.is_available():
                        meta = ExternalAPIClient.PROVIDERS[ext_provider]
                        ext_model = (
                            os.environ.get(meta["model_env"], "")
                            or meta["default"]
                        )
                        return ext_client, ext_model
                except Exception:
                    continue
        except Exception:
            pass

        # ── Tier 3: Local Ollama (degraded) ─────────────────────────────────
        return None, model_tag

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

        # ── Online mode: inject provider-specific inference client ─────
        # Uses _resolve_with_fallback so the chain ollama_cloud → external
        # → local is respected at runtime, not just at discovery time.
        selected = step.get("selected_model") or step.get("metadata", {}).get("selected_model")
        if selected:
            try:
                provider  = selected.get("provider", "ollama_local")
                model_tag = selected.get("model", "")
                _cloud_fallbacks = (
                    selected.get("cloud_fallback_list", [])
                    if isinstance(selected, dict) else []
                )
                client_obj, effective_tag = self._resolve_with_fallback(
                    provider, model_tag, cloud_fallback_list=_cloud_fallbacks
                )
                if client_obj is not None and hasattr(agent, "use_client"):
                    agent.use_client(client_obj)
                    # Keep _selected_model in sync with the actual client used
                    # in case fallback chose a different provider/model.
                    if effective_tag and effective_tag != model_tag:
                        step["_selected_model"] = effective_tag
            except Exception:
                pass  # Never break execution for client injection errors

        # Update the live label to show the agent is reasoning before the LLM
        # call returns.  Without this the step sits on its static pipeline name
        # (e.g. "initialize project scaffold") for the entire inference period
        # — often 10–30 s — with no visible activity.
        # Label format:  "<Verb>: <step description>"
        # e.g.  "Writing code: initialize project scaffold"
        #        "Planning: analyse requirements"
        #        "Debugging: fix failing tests"
        _solo_step_idx: Optional[int] = None
        try:
            _solo_step_idx = int(step.get("index", -1))
        except (TypeError, ValueError):
            pass
        if self._progress_tracker is not None and _solo_step_idx is not None and _solo_step_idx >= 0:
            try:
                _AGENT_VERBS: Dict[str, str] = {
                    "coding":     "Writing code",
                    "coding_frontend": "Writing frontend code",
                    "coding_backend":  "Writing backend code",
                    "coding_general":  "Writing code",
                    "planner":    "Planning",
                    "planning":   "Planning",
                    "debugging":  "Debugging",
                    "devops":     "Configuring",
                    "research":   "Researching",
                    "reasoning":  "Analysing",
                    "system":     "Running system task",
                    "critic":     "Reviewing",
                    "supervisor": "Coordinating",
                }
                _agent_name = step.get("agent", "").lower().strip()
                _verb = _AGENT_VERBS.get(_agent_name, "Working on")
                # Use description preferentially; fall back to name
                _step_desc = (step.get("description") or step.get("name") or "").strip()
                _thinking_label = (
                    f"{_verb}: {_step_desc}" if _step_desc else f"{_verb}\u2026"
                )
                self._progress_tracker.update_step_action(
                    _solo_step_idx, _thinking_label
                )
            except Exception:
                pass

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
                thread_name_prefix="sentinel-council",
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

    # ------------------------------------------------------------------
    # Fullstack write-file routing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_file_path(path: str) -> str:
        """Classify *path* as ``'frontend'`` or ``'backend'``.

        Uses file extension, directory segments, and known config filenames
        as scoring signals.  Returns ``'frontend'`` as the safe default for
        ambiguous files (e.g. ``package.json``, ``README.md``).
        """
        from pathlib import Path as _Path
        p     = _Path(path.replace("\\", "/"))
        ext   = p.suffix.lower()
        parts = {part.lower() for part in p.parts}
        name  = p.name.lower()

        # ── Strong backend signals ────────────────────────────────────
        _be_exts = {".py", ".go", ".rs", ".rb", ".php", ".java", ".kt", ".cs"}
        _be_segs = {
            "routes", "controllers", "models", "middleware", "migrations",
            "api", "db", "database", "services", "repositories", "schemas",
            "daemon", "worker", "queue", "jobs", "tasks",
        }
        _be_names = {
            "manage.py", "requirements.txt", "pipfile", "pipfile.lock",
            "setup.py", "setup.cfg", "pyproject.toml",
            "server.js", "server.ts", "server.mjs",
            "app.py", "main.py", "wsgi.py", "asgi.py",
            "docker-compose.yml", "docker-compose.yaml",
            "dockerfile",
            ".env", ".env.example", ".env.production",
        }
        if ext in _be_exts or parts & _be_segs or name in _be_names:
            return "backend"

        # ── Strong frontend signals ───────────────────────────────────
        _fe_exts = {
            ".jsx", ".tsx", ".vue", ".svelte", ".css", ".scss", ".sass",
            ".less", ".html", ".svg", ".styl",
        }
        _fe_segs = {
            "components", "pages", "views", "styles", "assets", "public",
            "ui", "layouts", "hooks", "context", "store", "redux", "atoms",
            "molecules", "organisms", "templates",
        }
        _fe_names = {
            "index.html", "app.jsx", "app.tsx", "app.vue", "app.svelte",
            "main.jsx", "main.tsx", "main.vue",
            "vite.config.js", "vite.config.ts",
            "webpack.config.js", "webpack.config.ts",
            "angular.json", "tailwind.config.js", "tailwind.config.ts",
            "postcss.config.js", ".babelrc", ".eslintrc.js", ".eslintrc.json",
            "nuxt.config.js", "nuxt.config.ts",
            "next.config.js", "next.config.ts",
            "svelte.config.js",
        }
        if ext in _fe_exts or parts & _fe_segs or name in _fe_names:
            return "frontend"

        # ── Ambiguous: .js/.ts/.json/README/etc. → frontend by default ──
        return "frontend"

    def _route_write_path(
        self,
        raw_path: str,
        project_root: str,
    ) -> str:
        """Return the routed absolute path for a ``write_file`` action.

        For fullstack projects, files are written into ``frontend/`` or
        ``backend/`` subdirectories based on :meth:`_classify_file_path`.
        Paths that already contain ``/frontend/`` or ``/backend/`` are
        returned unchanged.  For non-fullstack projects the path is always
        returned unchanged.
        """
        if self._project_architecture != "fullstack" or not self._project_dirs:
            return raw_path

        norm = raw_path.replace("\\", "/")

        # Already explicitly routed by the LLM
        if (
            "/frontend/" in norm or norm.startswith("frontend/")
            or "/backend/"  in norm or norm.startswith("backend/")
        ):
            return raw_path

        bucket  = self._classify_file_path(raw_path)
        subdir  = self._project_dirs.get(bucket, "")
        if not subdir:
            return raw_path

        # Strip any project_root prefix so we don't double it
        rel = norm
        if project_root:
            norm_root = project_root.replace("\\", "/").rstrip("/") + "/"
            if rel.startswith(norm_root):
                rel = rel[len(norm_root):]

        import os as _os
        return _os.path.join(subdir, rel)

    # ------------------------------------------------------------------
    # Action label formatting (used for live status display)
    # ------------------------------------------------------------------

    @staticmethod
    def _format_action_label(tool_name: str, params: Dict[str, Any], rationale: str = "") -> str:
        """Return a one-line status label for the current tool call.

        Priority: LLM-generated rationale (if meaningful) → formatted from
        tool name + key params.  The result is what appears in the live
        progress bar while the action is running.

        Args:
            tool_name: Tool identifier string.
            params:    Tool parameter dict.
            rationale: LLM-generated rationale from the AgentAction.

        Returns:
            A short, human-readable action description.
        """
        p = params or {}

        # ── Build a rich param-based label first so we can compare ──────────
        if tool_name == "write_file":
            import os as _os
            path = p.get("path", "?")
            fname = _os.path.basename(path) if path != "?" else "?"
            param_label = f"Writing {fname}  ({path})" if fname != path else f"Writing {path}"
        elif tool_name == "read_file":
            import os as _os
            path = p.get("path", "?")
            param_label = f"Reading {_os.path.basename(path) or path}"
        elif tool_name == "find_files":
            param_label = f"Finding {p.get('pattern', '*')} in {p.get('path', '.')}"
        elif tool_name == "run_shell":
            cmd = str(
                p.get("command")
                or p.get("cmd")
                or p.get("shell_command")
                or ""
            ).strip()
            # Show the actual command (not just "Running:") so it's informative.
            # Avoid rendering a bare "$" when command text is missing.
            param_label = f"$ {cmd[:100]}" if cmd else "run_shell"
        elif tool_name == "run_tests":
            param_label = f"Running tests in {p.get('path', '.')}"
        elif tool_name == "search_code":
            param_label = f"Searching: {p.get('query', '?')}"
        elif tool_name == "project_initializer":
            pname = p.get("project_name", "")
            ptype = p.get("project_type", "auto-detect")
            param_label = f"Scaffolding {ptype} project: {pname}" if pname else f"Scaffolding {ptype} project"
        elif tool_name == "install_dependency":
            pkgs = p.get("packages", [])
            param_label = f"Installing: {', '.join(pkgs[:4])}" if pkgs else "Installing dependency"
        elif tool_name == "git_commit":
            param_label = f"git commit: {p.get('message', '?')[:60]}"
        else:
            param_label = f"{tool_name}: {str(p)[:60]}"

        # ── Prefer LLM rationale when it's genuinely informative ────────────
        # Reject generic filler strings that add no value over the param label.
        _filler = {
            f"LLM-requested: {tool_name}",
            f"LLM-requested: {tool_name.replace('_', '-')}",
            tool_name,
            tool_name.replace("_", " "),
        }
        if rationale and rationale.strip() not in _filler and len(rationale.strip()) > 8:
            short = rationale.split(".")[0].strip()
            # If the rationale is longer than the param label, it's likely more
            # informative — prefer it.  Otherwise keep the param-based label
            # since it always contains the concrete file/command name.
            if len(short) > len(param_label) or any(
                kw in short.lower() for kw in ("creat", "generat", "updat", "add", "implement", "fix", "remov")
            ):
                return short[:120]

        return param_label

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
                params: Dict[str, Any] = dict(action.payload.get("params", {}) or {})

                # ── Glob-pattern redirect: read_file with a glob → find_files ──
                # The LLM sometimes emits read_file actions with patterns such as
                # "**\*.html" instead of using find_files.  Detect this and
                # silently reroute so the user sees useful results rather than
                # "File not found" errors.
                import re as _re
                _GLOB_RE = _re.compile(r"[\*\?\[]")
                if tool_name == "read_file" and "path" in params and _GLOB_RE.search(params["path"]):
                    raw_path: str = params["path"]
                    _first = _GLOB_RE.search(raw_path).start()
                    _dir   = raw_path[:_first].rstrip("/\\") or "."
                    _pat   = raw_path[_first:].lstrip("/\\")
                    tool_name = "find_files"
                    params = {"pattern": _pat, "path": _dir}
                    # Update the action reference so downstream logging is accurate
                    action = AgentAction.tool_call(
                        tool="find_files", params=params,
                        agent=action.agent or "engine",
                        step_id=action.step_id or "",
                        rationale=f"[auto-redirect] read_file glob '{raw_path}' → find_files",
                    )

                # ── FileChangeMap: resolve logical→absolute before dispatch ──
                if "path" in params and tool_name in (
                    "read_file", "write_file", "search_code", "find_files", "run_tests"
                ):
                    resolved_abs = self.file_change_map.resolve(params["path"])
                    if resolved_abs:
                        params = {**params, "path": resolved_abs}

                # ── Pre-dispatch validation: ensure required params exist ───
                # install_dependency needs 'packages' (non-empty list/string)
                if tool_name == "install_dependency":
                    pkgs = params.get("packages") or []
                    if isinstance(pkgs, str):
                        pkgs = [pkgs]
                    pkgs = [str(p).strip() for p in pkgs if str(p).strip()]
                    if not pkgs:
                        _blocked = {
                            "tool_name": tool_name,
                            "success": False,
                            "output": None,
                            "error": "Missing or empty 'packages' parameter for install_dependency",
                            "elapsed_ms": 0.0,
                            "metadata": params,
                        }
                        tool_results.append(_blocked)
                        event_data["tool_result"] = _blocked
                        self._emit(ProgressEvent(
                            event="action_dispatched",
                            step_name=action.agent,
                            message=f"[{tool_name}] blocked: missing packages parameter",
                            data=event_data,
                        ))
                        continue
                    params["packages"] = pkgs

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
                # Update the live progress label so the user can see what
                # action the agent is about to execute before the approval
                # prompt appears (or while it runs in non-approval mode).
                _action_label = self._format_action_label(
                    tool_name, params, getattr(action, "rationale", "")
                )
                # Prefer context["step_index"] (always an int, injected by
                # build_context) over action.step_id which is a UUID string —
                # int(uuid) raises ValueError and was silently swallowed,
                # so update_step_action was never actually called.
                _step_idx: Optional[int] = None
                if context:
                    _raw_idx = context.get("step_index")
                    if _raw_idx is not None:
                        try:
                            _step_idx = int(_raw_idx)
                        except (TypeError, ValueError):
                            pass
                if _step_idx is None:
                    _sid = getattr(action, "step_id", None)
                    if _sid is not None:
                        try:
                            _step_idx = int(_sid)
                        except (TypeError, ValueError):
                            pass
                if self._progress_tracker is not None and _step_idx is not None:
                    try:
                        self._progress_tracker.update_step_action(
                            _step_idx, _action_label
                        )
                    except Exception:
                        pass

                if tool_name == "project_initializer":
                    should_skip, skip_reason = self._maybe_skip_project_initializer(params, context)
                    if should_skip:
                        skipped = {
                            "tool_name": tool_name,
                            "success": True,
                            "output": {"skipped": True, "reason": skip_reason},
                            "error": None,
                            "elapsed_ms": 0.0,
                            "metadata": {**params, "skipped": True},
                        }
                        tool_results.append(skipped)
                        event_data["tool_result"] = skipped
                        self._emit(ProgressEvent(
                            event="action_dispatched",
                            step_name=action.agent,
                            message=f"[{tool_name}] skipped (already initialized)",
                            data=event_data,
                        ))
                        continue

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

                if tool_name in ("run_shell", "project_initializer"):
                    review = self.review_shell_command(tool_name, params, context)
                    decision = str(review.get("decision", "")).strip().lower()
                    if decision == "skip":
                        skipped = {
                            "tool_name": tool_name,
                            "success": True,
                            "output": {"skipped": True, "reason": review.get("reason", "")},
                            "error": None,
                            "elapsed_ms": 0.0,
                            "metadata": {
                                **params,
                                "supervisor_review": review,
                                "skipped": True,
                            },
                        }
                        tool_results.append(skipped)
                        event_data["tool_result"] = skipped
                        self._emit(ProgressEvent(
                            event="action_dispatched",
                            step_name=action.agent,
                            message=f"[{tool_name}] skipped by supervisor",
                            data=event_data,
                        ))
                        continue

                    timeout_seconds = review.get("timeout_seconds")
                    if timeout_seconds is not None:
                        try:
                            params = dict(params)
                            params["timeout"] = max(1, int(timeout_seconds))
                        except (TypeError, ValueError):
                            pass

                    if decision == "bounded_run" and tool_name == "run_shell":
                        params = dict(params)
                        params["watch_mode"] = True

                    if tool_name == "run_shell" and decision != "skip":
                        _cmd = str(params.get("command", "") or "")
                        if self._looks_like_long_running_shell(_cmd):
                            params = dict(params)
                            params["watch_mode"] = True
                            try:
                                _cur_timeout = int(params.get("timeout", 120))
                            except (TypeError, ValueError):
                                _cur_timeout = 120
                            params["timeout"] = min(max(1, _cur_timeout), 120)
                            if self._progress_tracker is not None and _step_idx is not None:
                                try:
                                    self._progress_tracker.update_step_action(
                                        _step_idx,
                                        "Checking dev/watch command startup (bounded)",
                                    )
                                except Exception:
                                    pass

                    display_label = str(review.get("display_label", "")).strip()
                    if display_label and self._progress_tracker is not None and _step_idx is not None:
                        try:
                            self._progress_tracker.update_step_action(_step_idx, display_label)
                        except Exception:
                            pass
                # ── Inject progress_callback for project_initializer ──────────
                if tool_name == "project_initializer" and self._progress_tracker is not None:
                    _pt = self._progress_tracker
                    _pidx = _step_idx
                    def _progress_cb(line: str, _t=_pt, _i=_pidx) -> None:
                        if _t is not None and _i is not None:
                            _line = str(line or "").strip()
                            if not _line:
                                return
                            try:
                                _cmd_like = (
                                    _line.startswith("$ ")
                                    or _line.lower().startswith((
                                        "npm ", "npx ", "pnpm ", "yarn ", "python ",
                                        "pip ", "node ", "flutter ", "gradle ", "swift ",
                                        "django-admin ",
                                    ))
                                )
                                if _cmd_like:
                                    label = _line if _line.startswith("$ ") else f"$ {_line}"
                                else:
                                    label = _line
                                _t.update_step_action(_i, label[:80])
                            except Exception:
                                pass
                    params = dict(params)
                    params["progress_callback"] = _progress_cb

                result = self._invoke_tool(tool_name, params, action, context)
                self._record_shell_history(tool_name, params, result)

                # ── Supervisor bus: notify on clean tool failure ───────────────
                if not result.get("success", True) and self._supervisor_bus is not None:
                    try:
                        from core.supervisor_bus import BusEvent, BusEventType
                        self._supervisor_bus.emit(BusEvent(
                            type=BusEventType.TOOL_FAILED,
                            step_id=str(action.step_id or ""),
                            step_index=_step_idx if _step_idx is not None else -1,
                            step_name=context.get("step_name", "") if context else "",
                            tool_name=tool_name,
                            error=result.get("error", ""),
                            context=context or {},
                            attempt=context.get("_attempt", 0) if context else 0,
                            extra={"params": params, "result": result,
                                   "agent": action.agent or ""},
                        ))
                        # Pause and wait for supervisor to fix or abort
                        self.pause(f"Supervisor handling {tool_name} failure")
                        self._pause_event.wait()
                        if self._abort_flag:
                            raise AbortException(self._abort_reason)
                        # Drain any fix actions the supervisor injected
                        while not self._fix_queue.empty():
                            try:
                                _tfix = self._fix_queue.get_nowait()
                                _tfacts = getattr(_tfix, "fix_actions", []) or []
                                _tfresults = self._dispatch_actions(_tfacts, context or {})
                                tool_results.extend(_tfresults)
                            except AbortException:
                                raise
                            except Exception:
                                break
                    except AbortException:
                        raise
                    except Exception:
                        pass  # Never crash dispatch for supervisor integration

                tool_results.append(result)

                # ── Capture project architecture from project_initializer ──────
                # Store architecture and subdirectory paths so subsequent
                # write_file calls can be routed to frontend/ or backend/.
                if tool_name == "project_initializer" and result.get("success"):
                    _out = result.get("output") or result.get("metadata") or {}
                    _arch = (
                        _out.get("architecture")
                        or (_out.get("metadata") or {}).get("architecture", "")
                    )
                    if _arch == "fullstack":
                        self._project_architecture = "fullstack"
                        self._project_dirs = {
                            "frontend": _out.get("frontend_dir", ""),
                            "backend":  _out.get("backend_dir",  ""),
                        }
                    elif _arch in ("frontend", "backend"):
                        self._project_architecture = _arch
                        self._project_dirs = {}
                    # Propagate a renamed/sanitized directory into context so
                    # subsequent steps (write_file, run_shell, read_file) use
                    # the corrected path.  project_initializer now computes the
                    # safe path BEFORE creating the directory, so renamed_dir
                    # is always set when the basename needed sanitization.
                    _renamed = _out.get("renamed_dir") or _out.get("project_path")
                    if _renamed and context:
                        context["project_root"] = _renamed
                    # ── Inject scaffolded file listing into context ──────────
                    # Walk the initialised directory and record all files in
                    # file_change_map AND in context["known_files"] so that
                    # the coding agent can reference real paths (e.g. it knows
                    # index.html lives at public/index.html in a CRA scaffold,
                    # not at the project root).
                    _proj_path = _renamed or _out.get("output_dir", "")
                    if _proj_path and os.path.isdir(_proj_path):
                        try:
                            import time as _time2
                            from core.file_change_map import FileChangeEvent as _FCE
                            _known: list = []
                            for _root, _dirs, _fnames in os.walk(_proj_path):
                                # Skip hidden dirs and node_modules / venv
                                _dirs[:] = [
                                    d for d in _dirs
                                    if d not in ("node_modules", "venv", ".git",
                                                 "__pycache__", ".next", "dist", "build")
                                    and not d.startswith(".")
                                ]
                                for _fname in _fnames:
                                    _abs = os.path.join(_root, _fname)
                                    _rel = os.path.relpath(_abs, _proj_path)
                                    _known.append(_rel)
                                    # Register in file_change_map so read_file
                                    # can resolve relative paths to absolute ones.
                                    if not self.file_change_map.resolve(_rel):
                                        self.file_change_map.record(_FCE(
                                            logical_path=_rel,
                                            absolute_path=_abs,
                                            operation="create",
                                            step_id=action.step_id or "",
                                            agent="project_initializer",
                                            timestamp_ms=int(_time2.time() * 1000),
                                        ))
                            if context is not None:
                                context["known_files"] = _known
                            self._known_files = sorted(set(_known))
                        except Exception:
                            pass  # Non-critical — never break execution
                event_data["tool_result"] = result

                # ── FileChangeMap: record successful write_file ──────────
                if tool_name == "write_file" and result.get("success"):
                    try:
                        import time as _time
                        from core.file_change_map import FileChangeEvent
                        logical_path = action.payload.get("params", {}).get("path", "")
                        absolute_path = (
                            result.get("metadata", {}).get("path")
                            or logical_path
                        )
                        from pathlib import Path as _Path
                        # Record BEFORE the existence check so we capture the
                        # operation correctly (file was just written, so it now
                        # exists — check whether it existed BEFORE this write).
                        existed_before = (
                            self.file_change_map.resolve(logical_path) is not None
                        )
                        operation = "modify" if existed_before else "create"
                        self.file_change_map.record(FileChangeEvent(
                            logical_path=logical_path,
                            absolute_path=absolute_path,
                            operation=operation,
                            step_id=action.step_id or "",
                            agent=action.agent or "",
                            timestamp_ms=int(_time.time() * 1000),
                        ))
                        # Keep context["known_files"] in sync so the coding
                        # agent's next turn sees newly written files.
                        if context is not None:
                            _kf = context.setdefault("known_files", [])
                            _rel_or_abs = logical_path
                            if _rel_or_abs not in _kf:
                                _kf.append(_rel_or_abs)
                        if logical_path:
                            _norm = logical_path.replace("\\", "/")
                            if _norm not in self._known_files:
                                self._known_files.append(_norm)
                    except Exception:
                        pass  # Never break execution for telemetry

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

        The user can provide one of three responses:
        - Y (yes, empty string) — approve this call only
        - N (no)               — decline this call
        - A (all, auto-approve) — approve this and all future approval
                                  requests in the current session
        """
        if self._auto_approve_session:
            return True

        try:
            from rich.panel import Panel
            from cli.progress_tracker import NEED_INPUT

            tracker = self._progress_tracker
            con = (
                tracker.console
                if tracker is not None and getattr(tracker, "console", None) is not None
                else self._console
            )
            if con is None:
                from rich.console import Console as _FC
                con = _FC()

            # ── Build detail line ────────────────────────────────────────
            if tool_name == "write_file":
                detail = f"Write → [bold]{params.get('path', '?')}[/bold]"
            elif tool_name == "run_shell":
                detail = f"Run shell → [bold]{params.get('command', '?')}[/bold]"
            elif tool_name == "git_commit":
                detail = f"Git commit: [bold]{params.get('message', '?')}[/bold]"
            elif tool_name == "install_dependency":
                pkgs = params.get('packages', [])
                if not pkgs:
                    pkgs = "(no packages specified)"
                elif isinstance(pkgs, list):
                    pkgs = ", ".join(pkgs[:3])
                detail = f"Install packages: [bold]{pkgs}[/bold]"
            elif tool_name == "project_initializer":
                detail = (
                    f"Init project → [bold]{params.get('project_name', '?')}[/bold]"
                    f" ({params.get('project_type', 'auto-detect')})"
                )
            else:
                detail = str(params)

            approval_panel = Panel(
                f"[yellow]Sentinel wants to execute:[/yellow]\n{detail}",
                title="[bold yellow]⚠  Approval Required[/bold yellow]",
                border_style="yellow",
            )

            if tracker is not None and tracker._render_queue is not None:
                # ── Queue-based path: main thread handles input ───────────
                # paused_for_input() stops Lives under _console_lock before
                # we put NEED_INPUT in the queue, so the main thread finds
                # a clean terminal when it dequeues and prints the panel.
                with tracker.paused_for_input():
                    tracker._pending_approval_panel = approval_panel
                    tracker._render_queue.put(NEED_INPUT)
                    try:
                        answer = tracker._response_queue.get(timeout=300)
                    except Exception:
                        answer = "n"
            else:
                # ── Fallback: Rich unavailable, call input() directly ────
                con.print(approval_panel)
                try:
                    answer = input("  Apply? [Y/n/A] › ").strip().lower()
                except EOFError:
                    answer = "n"
                except KeyboardInterrupt:
                    con.print("\n[dim]Interrupted — treating as deny.[/dim]")
                    answer = "n"

            if answer in ("a", "accept all", "auto-approve"):
                self._auto_approve_session = True
                return True
            return answer in ("", "y", "yes")

        except Exception as _approval_err:
            import sys as _sys
            if _sys.stdin.isatty():
                try:
                    _sys.stdout.write(
                        f"\n[approval required — rich display failed: {_approval_err}]\n"
                        f"  Tool: {tool_name}  params: {str(params)[:120]}\n"
                        f"  Apply? [Y/n/A] › "
                    )
                    _sys.stdout.flush()
                    _raw = _sys.stdin.readline().strip().lower()
                    if _raw in ("a", "accept all", "auto-approve"):
                        self._auto_approve_session = True
                        return True
                    return _raw in ("", "y", "yes")
                except Exception:
                    return False
            return False

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
                if not params.get("path") or params.get("path") == ".":
                    params = {**params, "path": _proj}
            elif tool_name == "project_initializer":
                if not params.get("output_dir") or params.get("output_dir") == ".":
                    params = {**params, "output_dir": _proj}
            elif tool_name == "find_files":
                if not params.get("path") or params.get("path") == ".":
                    params = {**params, "path": _proj}

        # For fullstack projects: route write_file paths into the correct
        # frontend/ or backend/ subdirectory at the engine level so the agent
        # doesn't need to know the exact directory layout.
        if tool_name == "write_file":
            _proj_root = (context or {}).get("project_root", "")
            _routed = self._route_write_path(params.get("path", ""), _proj_root)
            if _routed != params.get("path", ""):
                params = {**params, "path": _routed}

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

        yield from collected# Changed core/execution_engine.py
