"""progress_tracker.py — Sentinel live step progress tracker.

Provides a Rich-backed live progress display for pipeline execution.
Supports multi-step progress bars, per-step spinners, elapsed timing,
and a final execution summary table.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional

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
            return

        self._progress: Optional[Progress] = None
        self._live: Optional[Live] = None
        self._task_ids: Dict[str, TaskID] = {}
        self._step_timings: Dict[str, float] = {}
        self._pipeline_task_id: Optional[TaskID] = None
        # Tracks Rich task IDs for tree node status updates (Part 5)
        self._node_task_ids: Dict[str, Any] = {}

    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Step-level updates
    # ------------------------------------------------------------------

    def start_step(self, step_index: int, description: str, provider: str = "", model: str = "") -> None:
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

        self.console.print(table)
        overall = (
            "[bold green]Pipeline complete[/bold green]"
            if total_failed == 0
            else f"[bold red]Pipeline finished with {total_failed} failure(s)[/bold red]"
        )
        self.console.print(
            Panel(
                f"{overall}\n"
                f"[green]{total_success} succeeded[/green]  "
                f"[red]{total_failed} failed[/red]  "
                f"[yellow]{len(steps) - total_success - total_failed} other[/yellow]",
                border_style="cyan",
                expand=False,
            )
        )

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

    def display_tree(self, tree: Any) -> None:  # tree: TaskDecompositionTree
        """Render the decomposition tree as an indented Rich Tree widget.

        Called once before tree execution begins.  Each node shows its
        ``node_id`` (first 8 chars), ``domain``, ``complexity``, and whether
        it is a leaf or has children.

        Args:
            tree: A :class:`~core.task_tree.TaskDecompositionTree` instance.
        """
        if not _RICH_AVAILABLE or self.console is None:  # pragma: no cover
            return
        try:
            from rich.tree import Tree as RichTree

            def _add_node(rich_parent: Any, node: Any) -> None:
                domain = node.task_dict.get("routing_domain",
                                            node.task_dict.get("domain", "?"))
                complexity = node.complexity()
                leaf_label = "leaf" if node.is_leaf() else f"{len(node.children)} children"
                label = (
                    f"[dim]{node.node_id[:8]}[/dim]  "
                    f"[cyan]{domain}[/cyan]  |  "
                    f"[yellow]{complexity}[/yellow]  →  {leaf_label}"
                )
                branch = rich_parent.add(label)
                for child in node.children:
                    _add_node(branch, child)

            root_domain = tree.root.task_dict.get(
                "routing_domain", tree.root.task_dict.get("domain", "root"))
            rich_tree = RichTree(
                f"[bold green]Task Tree[/bold green]  "
                f"[dim]{tree.root.node_id[:8]}[/dim]  "
                f"[cyan]{root_domain}[/cyan]"
            )
            for child in tree.root.children:
                _add_node(rich_tree, child)

            from rich.panel import Panel as _Panel
            self.console.print(_Panel(rich_tree, title="Decomposition Tree",
                                      border_style="cyan", expand=False))
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
        label = f"[{style}]{icon} {node_id[:8]} — {status}{suffix}[/{style}]"

        if self._progress and node_id in self._node_task_ids:
            self._progress.update(
                self._node_task_ids[node_id],
                description=label,
                visible=True,
            )
        else:
            # Fallback: print directly when no live progress is active
            try:
                self.console.print(label)
            except Exception:  # pragma: no cover
                pass
