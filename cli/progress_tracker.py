"""progress_tracker.py — Sentinel live step progress tracker.

Provides a Rich-backed live progress display for pipeline execution.
Supports multi-step progress bars, per-step spinners, elapsed timing,
and a final execution summary table.
"""
from __future__ import annotations

import time
import queue
import threading
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional

# Sentinel objects for the render-queue state machine.
# Identity-compared with ``is`` — never equal to a callable.
NEED_INPUT   = object()   # backend → main: approval input needed
BACKEND_DONE = object()   # backend → main: pipeline finished

# BUG-4 fix: Rich is an optional dependency.  If it is not installed the
# entire module still imports and every method degrades gracefully to a
# no-op.  All failure paths ensure self._progress_tracker is explicitly
# set to None so that callers can safely test ``if tracker is not None``.
try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TaskID,
        TextColumn,
        TimeElapsedColumn,
    )
    from rich.table import Table
    from rich.text import Text
    from rich.tree import Tree as RichTree
    _RICH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _RICH_AVAILABLE = False
    # Stub out types so the rest of the module can reference them safely.
    Console = None          # type: ignore[assignment,misc]
    Live = None             # type: ignore[assignment,misc]
    Panel = None            # type: ignore[assignment,misc]
    Progress = None         # type: ignore[assignment,misc]
    SpinnerColumn = None    # type: ignore[assignment,misc]
    BarColumn = None        # type: ignore[assignment,misc]
    MofNCompleteColumn = None  # type: ignore[assignment,misc]
    TextColumn = None       # type: ignore[assignment,misc]
    TimeElapsedColumn = None  # type: ignore[assignment,misc]
    TaskID = None           # type: ignore[assignment,misc]
    Table = None            # type: ignore[assignment,misc]
    Text = None             # type: ignore[assignment,misc]
    RichTree = None         # type: ignore[assignment,misc]


class ProgressTracker:
    """Tracks and displays live pipeline execution progress using Rich.

    Attributes:
        console: The Rich Console to render output to.
    """

    def __init__(self, console: Optional[Console] = None) -> None:
        """Initialise the ProgressTracker.

        Args:
            console: Optional Rich Console. Creates a new one if not provided.
                     If Rich is unavailable, all methods degrade to no-ops and
                     ``self._progress_tracker`` is set to ``None``.
        """
        # BUG-4 fix: initialise _progress_tracker unconditionally so that
        # every code path (Rich present, Rich missing, constructor exception)
        # leaves the attribute in a known state and prevents AttributeError.
        self._progress_tracker: Optional[Any] = None

        if not _RICH_AVAILABLE:  # pragma: no cover
            self.console = None
            self._progress: Optional[Any] = None
            self._live: Optional[Any] = None
            self._task_ids: Dict[str, Any] = {}
            self._step_timings: Dict[str, float] = {}
            self._pipeline_task_id: Optional[Any] = None
            # _node_task_ids used by tree-display methods (Part 5)
            self._node_task_ids: Dict[str, Any] = {}
            self._decomp_rich_tree: Optional[Any] = None
            self._decomp_tree_live: Optional[Any] = None
            self._decomp_root_label: str = ""
            self._render_queue: Optional[Any] = None
            self._console_lock = None
            self._backend_active = None
            self._deferred_frontend = []
            self._response_queue = None
            self._pending_approval_panel = None
            return

        try:
            self.console = console or Console()
        except Exception:  # pragma: no cover
            self.console = None
            self._progress = None
            self._live = None
            self._task_ids = {}
            self._step_timings = {}
            self._pipeline_task_id = None
            self._node_task_ids = {}
            self._decomp_rich_tree = None
            self._decomp_tree_live = None
            self._decomp_root_label = ""
            self._render_queue = None
            self._console_lock = None
            self._backend_active = None
            self._deferred_frontend = []
            self._response_queue = None
            self._pending_approval_panel = None
            return

        self._progress: Optional[Progress] = None
        self._live: Optional[Live] = None
        self._task_ids: Dict[str, TaskID] = {}
        self._step_timings: Dict[str, float] = {}
        self._pipeline_task_id: Optional[TaskID] = None
        # Tracks Rich task IDs for tree node status updates (Part 5)
        self._node_task_ids: Dict[str, Any] = {}
        # Live decomposition-tree state
        self._decomp_rich_tree: Optional[Any] = None
        self._decomp_tree_live: Optional[Any] = None
        self._decomp_root_label: str = ""
        # Render queue — drained by the main thread in RENDER state.
        self._render_queue: "queue.SimpleQueue[Any]" = queue.SimpleQueue()
        # Lock serialising Live.stop()/start() against concurrent _enqueue() calls.
        self._console_lock: threading.Lock = threading.Lock()
        # While set, _drain_one() defers callables instead of dispatching them.
        self._backend_active: threading.Event = threading.Event()
        # Callables deferred during an approval window; replayed after Lives restart.
        self._deferred_frontend: List[Any] = []
        # Carries the user's approval answer from main thread back to backend thread.
        self._response_queue: "queue.SimpleQueue[str]" = queue.SimpleQueue()
        # Holds the approval Panel between the backend (builder) and main (printer).
        self._pending_approval_panel: Any = None

    # ------------------------------------------------------------------
    # Daemon render thread
    # ------------------------------------------------------------------

    def _drain_one(self) -> Any:
        """Dequeue and execute one render callable (called by main thread).

        Blocks on the render queue until an item is available. Returns the
        raw item so the caller can detect ``NEED_INPUT`` and ``BACKEND_DONE``
        sentinels. Regular callables are dispatched under ``_console_lock``
        unless deferred by ``_backend_active``.
        """
        item = self._render_queue.get()

        if item is NEED_INPUT or item is BACKEND_DONE:
            return item  # caller handles state transitions

        if self._backend_active.is_set():
            self._deferred_frontend.append(item)
        else:
            with self._console_lock:
                try:
                    item()
                except Exception:
                    pass

        return item

    def _enqueue(self, fn: Any) -> None:
        """Submit a rendering callable to the main thread's render queue.

        Thread-safe. No-ops silently when the render queue is unavailable.
        """
        if self._render_queue is not None:
            try:
                self._render_queue.put(fn)
            except Exception:
                pass

    def print(self, renderable: Any) -> None:
        """Enqueue a renderable for display via the main thread's render loop.

        External callers (e.g. ``tree_execution_engine``) must use this
        method instead of accessing ``self.console.print()`` directly so
        that all terminal output is serialised through the render queue.
        """
        if self.console is not None:
            self._enqueue(lambda r=renderable: self.console.print(r))

    # Pipeline-level progress
    # ------------------------------------------------------------------

    def start_pipeline(self, pipeline: List[Dict[str, Any]], task_name: str = "Running pipeline") -> None:
        """Begin live progress display for a full pipeline.

        Creates a Rich Progress bar for the overall pipeline and one
        spinner per step. Call stop_pipeline() when done.

        Args:
            pipeline: The list of pipeline step dicts.
            task_name: Label for the overall pipeline progress bar.
        """
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self.console,
            transient=False,
        )
        self._pipeline_task_id = self._progress.add_task(
            task_name, total=len(pipeline)
        )
        for step in pipeline:
            idx = step.get("index", "?")
            desc = step.get("description", f"Step {idx}")
            tid = self._progress.add_task(
                f"  [{idx}] {desc}", total=1, visible=False
            )
            self._task_ids[str(idx)] = tid

        self._live = Live(self._progress, console=self.console, refresh_per_second=10)
        self._live.start()

    def stop_pipeline(self) -> None:
        """Stop the live progress display."""
        if self._live is not None:
            self._live.stop()
            self._live = None
        self._progress = None
        self._task_ids.clear()
        self._pipeline_task_id = None

    def pause(self) -> None:
        """Temporarily pause the live display (for user input prompts)."""
        if self._live is not None and self._live.is_started:
            self._live.stop()

    def resume(self) -> None:
        """Resume the live display after pausing."""
        if self._live is not None and not self._live.is_started and self._progress is not None:
            # Need to recreate the Live instance since it can't be restarted
            self._live = Live(self._progress, console=self.console, refresh_per_second=10)
            self._live.start()

    @contextmanager
    def paused_for_input(self) -> Generator[None, None, None]:
        """Context manager that cleanly suspends ALL live displays for user input.

        Flushes the render queue, then stops both the pipeline-progress Live
        (``self._live``) and the decomposition-tree Live
        (``self._decomp_tree_live``) before yielding, so the terminal is fully
        free for printing and ``input()`` calls.  Both displays are restarted
        in the ``finally`` block — even if the body raises.

        This prevents the render thread from firing ``_decomp_tree_live.refresh()``
        while ``input()`` is blocking, which would corrupt the terminal and
        cause the approval panel to reappear multiple times.

        Usage::

            with tracker.paused_for_input():
                tracker.console.print(approval_panel)
                answer = input("Apply? [Y/n] › ")
        """
        self._backend_active.set()

        with self._console_lock:
            pipeline_was_running = (
                self._live is not None
                and getattr(self._live, "is_started", False)
            )
            if pipeline_was_running:
                self._live.stop()

            decomp_was_running = (
                self._decomp_tree_live is not None
                and getattr(self._decomp_tree_live, "is_started", False)
            )
            if decomp_was_running:
                try:
                    self._decomp_tree_live.stop()
                except Exception:
                    pass

        try:
            yield

        finally:
            if decomp_was_running and self._decomp_rich_tree is not None:
                try:
                    _panel = Panel(
                        self._decomp_rich_tree,
                        title="Decomposition Tree",
                        border_style="cyan",
                        expand=True,
                    )
                    self._decomp_tree_live = Live(
                        _panel,
                        console=self.console,
                        auto_refresh=False,
                        transient=False,
                    )
                    self._decomp_tree_live.start()
                except Exception:
                    pass

            if pipeline_was_running and self._progress is not None:
                try:
                    self._live = Live(
                        self._progress,
                        console=self.console,
                        refresh_per_second=10,
                    )
                    self._live.start()
                except Exception:
                    pass

            for fn in self._deferred_frontend:
                self._render_queue.put(fn)
            self._deferred_frontend.clear()

            self._backend_active.clear()

    # ------------------------------------------------------------------
    # Step-level updates
    # ------------------------------------------------------------------

    def update_step_action(self, step_index: int, action_message: str) -> None:
        """Update the live label for the currently running step.

        Called before each tool dispatch so the user can see *what the agent
        is doing right now* rather than just the static step description.

        The message is trimmed to 80 characters and prepended with a bullet
        so it fits inside the progress bar column without wrapping.

        Args:
            step_index:     The step's pipeline index (same value passed to
                            :meth:`start_step`).
            action_message: Short human-readable description of the current
                            action (e.g. ``"Writing src/App.jsx"`` or the
                            LLM-generated rationale for the action).
        """
        key = str(step_index)
        if not self._progress or key not in self._task_ids:
            return
        # Truncate long messages so they stay on one line
        short = action_message[:80].strip()
        if len(action_message) > 80:
            short += "…"
        label = f"  [cyan]◉[/cyan] [{step_index}] [dim]{short}[/dim]"
        try:
            self._progress.update(self._task_ids[key], description=label)
        except Exception:  # pragma: no cover
            pass

    def finalize_decomp_tree(self) -> None:
        """Stop the live decomposition-tree display.

        Flushes all pending render-queue operations first, then stops the
        Live context so the final tree state is flushed to the terminal.
        Must be called once after tree execution completes.
        """
        if self._decomp_tree_live is not None:
            _lock = self._console_lock
            try:
                if _lock is not None:
                    _lock.acquire()
                self._decomp_tree_live.stop()
            except Exception:  # pragma: no cover
                pass
            finally:
                if _lock is not None:
                    try:
                        _lock.release()
                    except RuntimeError:
                        pass
            self._decomp_tree_live = None
            self._decomp_rich_tree = None

    def start_step(
        self,
        step_index: int,
        description: str,
        provider: str = "",
        model: str = "",
    ) -> None:
        """Mark a step as active in the live display.

        Args:
            step_index:  The step's pipeline index.
            description: Human-readable step description.
            provider:    Optional provider name for online mode display.
            model:       Optional model tag for online mode display.
        """
        key = str(step_index)
        self._step_timings[key] = time.monotonic()

        # Build display label
        label = f"  ◉ [{step_index}] {description}"
        if provider and provider != "ollama_local":
            label += f" [dim]→ {provider}:{model}[/dim]"

        if self._progress and key in self._task_ids:
            self._progress.update(
                self._task_ids[key],
                description=label,
                visible=True,
            )

    def complete_step(self, step_index: int, success: bool = True) -> None:
        """Mark a step as completed or failed.

        Args:
            step_index: The step's pipeline index.
            success: True for success, False for failure.
        """
        key = str(step_index)
        elapsed = time.monotonic() - self._step_timings.get(key, time.monotonic())
        icon = "✔" if success else "✘"
        style = "green" if success else "red"

        if self._progress and key in self._task_ids:
            self._progress.update(
                self._task_ids[key],
                completed=1,
                description=f"  [{style}]{icon}[/{style}] [{step_index}] ({elapsed:.1f}s)",
            )

        if self._progress and self._pipeline_task_id is not None:
            self._progress.advance(self._pipeline_task_id)

    def skip_step(self, step_index: int) -> None:
        """Mark a step as skipped.

        Args:
            step_index: The step's pipeline index.
        """
        key = str(step_index)
        if self._progress and key in self._task_ids:
            self._progress.update(
                self._task_ids[key],
                completed=1,
                description=f"  [yellow]⊘[/yellow] [{step_index}] skipped",
                visible=True,
            )
        if self._progress and self._pipeline_task_id is not None:
            self._progress.advance(self._pipeline_task_id)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    @contextmanager
    def track_step(
        self, step_index: int, description: str
    ) -> Generator[None, None, None]:
        """Context manager that tracks a single step's start and completion.

        Usage::

            with tracker.track_step(1, "Analysing project"):
                do_work()

        Args:
            step_index: The step's pipeline index.
            description: Human-readable step label.
        """
        self.start_step(step_index, description)
        success = True
        try:
            yield
        except Exception:
            success = False
            raise
        finally:
            self.complete_step(step_index, success=success)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def print_summary(self, steps: List[Dict[str, Any]]) -> None:
        """Print a final execution summary table after pipeline completion.

        Args:
            steps: The pipeline steps list with final status values set.
        """
        table = Table(
            title="Execution Summary",
            border_style="cyan",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("#", width=4, style="dim")
        table.add_column("Step", style="white")
        table.add_column("Agent", style="dim")
        table.add_column("Provider : Model", style="dim")
        table.add_column("Status", width=12)
        table.add_column("Time", width=8, style="dim")

        total_success = 0
        total_failed = 0

        for step in steps:
            idx = step.get("index", "?")
            status = step.get("status", "pending")
            elapsed_ms = step.get("elapsed_ms")
            elapsed = self._step_timings.get(str(idx))
            if elapsed_ms is not None:
                time_str = f"{elapsed_ms / 1000:.1f}s"
            elif elapsed is not None:
                time_str = f"{elapsed:.1f}s"
            else:
                time_str = "—"

            # Provider / model from online mode selected_model metadata
            sel = (step.get("metadata") or {}).get("selected_model") or {}
            provider_str = sel.get("provider", "")
            model_str = sel.get("model", "")
            if provider_str and provider_str != "ollama_local":
                prov_display = f"{provider_str}:{model_str[:20]}" if model_str else provider_str
            elif provider_str == "ollama_local" and model_str:
                prov_display = f"local:{model_str[:20]}"
            else:
                prov_display = step.get("agent", "—")

            if status in ("success", "completed"):
                total_success += 1
                status_text = Text("✔ completed", style="bold green")
            elif status == "failed":
                total_failed += 1
                status_text = Text("✘ failed", style="bold red")
            elif status == "skipped":
                status_text = Text("⊘ skipped", style="bold yellow")
            else:
                status_text = Text(f"○ {status}", style="dim")

            table.add_row(
                str(idx),
                step.get("description", "—")[:60],
                step.get("agent", "—"),
                prov_display,
                status_text,
                time_str,
            )

        self._enqueue(lambda t=table: self.console.print(t))
        overall = (
            "[bold green]Pipeline complete[/bold green]"
            if total_failed == 0
            else f"[bold red]Pipeline finished with {total_failed} failure(s)[/bold red]"
        )
        _summary_panel = Panel(
            f"{overall}\n"
            f"[green]{total_success} succeeded[/green]  "
            f"[red]{total_failed} failed[/red]  "
            f"[yellow]{len(steps) - total_success - total_failed} other[/yellow]",
            border_style="cyan",
            expand=True,
        )
        self._enqueue(lambda p=_summary_panel: self.console.print(p))

    # ------------------------------------------------------------------
    # Simple one-shot spinner
    # ------------------------------------------------------------------

    @contextmanager
    def spinner(self, message: str) -> Generator[None, None, None]:
        """Display a simple spinner for a short blocking operation.

        Usage::

            with tracker.spinner("Indexing project…"):
                do_indexing()

        Args:
            message: The message to display beside the spinner.
        """
        progress = Progress(
            SpinnerColumn(),
            TextColumn(f"[cyan]{message}"),
            TimeElapsedColumn(),
            console=self.console,
            transient=True,
        )
        with progress:
            tid = progress.add_task(message, total=None)
            try:
                yield
            finally:
                progress.update(tid, completed=1)

    # ------------------------------------------------------------------
    # Tree decomposition display (Part 5)
    # ------------------------------------------------------------------

    def display_tree(self, tree: Any, parent_description: str = "") -> None:  # tree: TaskDecompositionTree
        """Render the decomposition tree as an indented Rich Tree widget.

        Called once before tree execution begins (or after lazy decomposition).
        Each node shows its description, complexity, and whether it is a leaf
        or has children.

        Args:
            tree: A :class:`~core.task_tree.TaskDecompositionTree` instance.
            parent_description: Optional description of the parent task being decomposed.
                               When provided, replaces the root label with a "Decomposing" heading.
        """
        if not _RICH_AVAILABLE or self.console is None:  # pragma: no cover
            return
        try:
            from rich.tree import Tree as RichTree
            from rich.panel import Panel as _Panel
            from rich.live import Live

            # ── inner helper: populate a RichTree node recursively ──────────
            def _add_node(rich_parent: Any, node: Any, _depth: int = 0) -> None:
                desc_raw = (
                    node.task_dict.get("refined_prompt")
                    or node.task_dict.get("raw_description", "")
                )
                # Compute available width dynamically from the live console so
                # the label uses as much space as the current terminal allows.
                # Budget = terminal_width
                #          - panel_borders (2)
                #          - tree_indent   (2 chars × depth)
                #          - fixed_suffix  (~28 chars: "  |  medium  →  N children")
                # Clamped to [30, 120] to stay readable on tiny/wide terminals.
                _term_w = getattr(self.console, "width", 80) or 80
                _desc_budget = max(30, min(120, _term_w - 2 - (2 * _depth) - 28))
                desc = (desc_raw[:_desc_budget] + "…") if len(desc_raw) > _desc_budget else desc_raw
                complexity = node.complexity()
                if node.is_leaf():
                    label = (
                        f"[dim]{desc}[/dim]  |  "
                        f"[yellow]{complexity}[/yellow]"
                    )
                else:
                    label = (
                        f"[dim]{desc}[/dim]  |  "
                        f"[yellow]{complexity}[/yellow]  →  "
                        f"{len(node.children)} children"
                    )
                branch = rich_parent.add(label)
                for child in node.children:
                    _add_node(branch, child, _depth + 1)

            # ── bootstrap: first call only ──────────────────────────────────
            if self._decomp_tree_live is None:
                self._decomp_root_label = (
                    "[bold green]Task Tree[/bold green]  [dim]root[/dim]"
                )
                self._decomp_rich_tree = RichTree(self._decomp_root_label)
                for child in tree.root.children:
                    _add_node(self._decomp_rich_tree, child)

                panel = _Panel(
                    self._decomp_rich_tree,
                    title="Decomposition Tree",
                    border_style="cyan",
                    expand=True,
                )
                self._decomp_tree_live = Live(
                    panel,
                    console=self.console,
                    auto_refresh=False,
                    transient=False,
                )
                self._decomp_tree_live.start()

            # ── subsequent calls: rebuild children, enqueue refresh ──────────
            else:
                self._decomp_rich_tree = RichTree(self._decomp_root_label)
                for child in tree.root.children:
                    _add_node(self._decomp_rich_tree, child)

                panel = _Panel(
                    self._decomp_rich_tree,
                    title="Decomposition Tree",
                    border_style="cyan",
                    expand=True,
                )
                _p = panel  # capture before enqueue to avoid late-binding
                def _do_refresh(_panel_ref: Any = _p) -> None:
                    # Called by _drain_one() under _console_lock.
                    # auto_refresh=False so this is the only refresh path.
                    if self._decomp_tree_live is not None:
                        self._decomp_tree_live.update(_panel_ref)
                        self._decomp_tree_live.refresh()
                self._enqueue(_do_refresh)

            # ── register nodes for live status tracking ───────────────────────
            if self._progress is not None:
                for node in tree.post_order():
                    if node.node_id not in self._node_task_ids:
                        tid = self._progress.add_task(
                            node.node_id[:8], total=1, visible=False
                        )
                        self._node_task_ids[node.node_id] = tid
            else:
                for node in tree.post_order():
                    if node.node_id not in self._node_task_ids:
                        desc = (
                            node.task_dict.get("refined_prompt")
                            or node.task_dict.get("raw_description", "?")
                        )[:50]
                        self._node_task_ids[node.node_id] = desc

        except Exception:  # pragma: no cover
            pass

    def update_node_status(
        self,
        node_id: str,
        status: str,
        message: str = "",
    ) -> None:
        """Update the live display of a tree node's status during execution.

        Status values map to Rich styles:

        =========== ========================
        Status       Style
        =========== ========================
        pending      dim white
        running      yellow (with spinner)
        unit_tested  green
        integrated   bold green
        failed       bold red
        =========== ========================

        Args:
            node_id: The :attr:`~core.task_tree.TaskNode.node_id` to update.
            status:  One of ``"pending"``, ``"running"``, ``"unit_tested"``,
                     ``"integrated"``, ``"failed"``.
            message: Optional human-readable detail appended to the label.
        """
        if not _RICH_AVAILABLE or self.console is None:  # pragma: no cover
            return

        _STATUS_STYLES: Dict[str, str] = {
            "pending":     "dim white",
            "running":     "yellow",
            "unit_tested": "green",
            "integrated":  "bold green",
            "failed":      "bold red",
        }
        _STATUS_ICONS: Dict[str, str] = {
            "pending":     "○",
            "running":     "◉",
            "unit_tested": "✔",
            "integrated":  "✔✔",
            "failed":      "✘",
        }
        style = _STATUS_STYLES.get(status, "white")
        icon = _STATUS_ICONS.get(status, "?")
        suffix = f"  [dim]{message}[/dim]" if message else ""
        label = f"[{style}]{icon} {status}{suffix}[/{style}]"

        if self._progress and node_id in self._node_task_ids:
            self._progress.update(
                self._node_task_ids[node_id],
                description=label,
                visible=True,
            )
        else:
            # Fallback: enqueue for render thread when no live progress is active
            self._enqueue(lambda l=label: self.console.print(l))