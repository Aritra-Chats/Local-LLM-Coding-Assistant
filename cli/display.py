"""pipeline_viewer.py — Sentinel pipeline tree visualizer.

Renders the active pipeline as a Rich Tree, showing each step's index,
assigned agent, status, and tool bindings. Status is colour-coded:
pending (dim), running (cyan/bold), success (green), failed (red),
skipped (yellow).
"""

from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree


# ---------------------------------------------------------------------------
# Status styles
# ---------------------------------------------------------------------------

_STATUS_STYLES: Dict[str, str] = {
    "pending":  "dim white",
    "running":  "bold cyan",
    "success":  "bold green",
    "failed":   "bold red",
    "skipped":  "bold yellow",
    "retrying": "bold magenta",
}

_STATUS_ICONS: Dict[str, str] = {
    "pending":  "○",
    "running":  "◉",
    "success":  "✔",
    "failed":   "✘",
    "skipped":  "⊘",
    "retrying": "↺",
}


class PipelineViewer:
    """Renders a Sentinel pipeline as a Rich Tree or Table.

    Attributes:
        console: The Rich Console to print output to.
    """

    def __init__(self, console: Optional[Console] = None) -> None:
        """Initialise the PipelineViewer.

        Args:
            console: Optional Rich Console. Creates a new one if not provided.
        """
        self.console = console or Console()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(self, pipeline_state: Dict[str, Any]) -> None:
        """Render the full pipeline state as a Rich tree inside a panel.

        Args:
            pipeline_state: A pipeline state dict with a 'steps' list. Each
                step dict is expected to contain at minimum:
                    - 'index' (int)
                    - 'description' (str)
                    - 'agent' (str)
                    - 'status' (str)
                Optional fields:
                    - 'tools' (List[str])
                    - 'model' (str)
                    - 'error' (str)
        """
        steps: List[Dict[str, Any]] = pipeline_state.get("steps", [])
        task_name: str = pipeline_state.get("task", "Unnamed Task")
        current_step: int = pipeline_state.get("current_step", -1)

        tree = Tree(
            f"[bold cyan]Pipeline:[/bold cyan] [white]{task_name}[/white]",
            guide_style="dim cyan",
        )

        for step in steps:
            self._add_step_node(tree, step, is_current=(step.get("index") == current_step))

        summary = self._build_summary_text(steps)

        self.console.print(
            Panel(
                tree,
                title="[bold cyan]Pipeline View[/bold cyan]",
                subtitle=summary,
                border_style="cyan",
                expand=True,
            )
        )

    def render_table(self, pipeline_state: Dict[str, Any]) -> None:
        """Render the pipeline as a compact Rich Table.

        Args:
            pipeline_state: Same structure as accepted by render().
        """
        steps: List[Dict[str, Any]] = pipeline_state.get("steps", [])
        task_name: str = pipeline_state.get("task", "Unnamed Task")

        table = Table(
            title=f"Pipeline: {task_name}",
            border_style="cyan",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("#", width=4, style="dim")
        table.add_column("Status", width=10)
        table.add_column("Agent", style="bold white")
        table.add_column("Description", style="white")
        table.add_column("Tools", style="dim")

        for step in steps:
            status = step.get("status", "pending")
            icon = _STATUS_ICONS.get(status, "○")
            style = _STATUS_STYLES.get(status, "white")
            status_text = Text(f"{icon} {status}", style=style)
            tools = ", ".join(step.get("tools", [])) or "—"
            table.add_row(
                str(step.get("index", "")),
                status_text,
                step.get("agent", "—"),
                step.get("description", ""),
                tools,
            )

        self.console.print(table)

    def render_step_detail(self, step: Dict[str, Any]) -> None:
        """Render detailed information for a single pipeline step.

        Args:
            step: A single pipeline step dict.
        """
        status = step.get("status", "pending")
        style = _STATUS_STYLES.get(status, "white")
        icon = _STATUS_ICONS.get(status, "○")

        table = Table(show_header=False, border_style="cyan", expand=False)
        table.add_column("Field", style="bold cyan", width=16)
        table.add_column("Value", style="white")

        table.add_row("Step", str(step.get("index", "—")))
        table.add_row("Status", Text(f"{icon} {status}", style=style))
        table.add_row("Agent", step.get("agent", "—"))
        table.add_row("Description", step.get("description", "—"))
        table.add_row("Model", step.get("model", "—"))
        table.add_row("Tools", ", ".join(step.get("tools", [])) or "—")

        if step.get("error"):
            table.add_row("Error", Text(step["error"], style="bold red"))

        self.console.print(
            Panel(table, title="[bold cyan]Step Detail[/bold cyan]", border_style="cyan")
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _add_step_node(
        self,
        tree: Tree,
        step: Dict[str, Any],
        is_current: bool = False,
    ) -> None:
        """Add a step as a child node to the pipeline tree.

        Args:
            tree: The parent Rich Tree node.
            step: The step dict to render.
            is_current: True if this is the currently executing step.
        """
        status = step.get("status", "pending")
        icon = _STATUS_ICONS.get(status, "○")
        style = _STATUS_STYLES.get(status, "white")

        label = Text()
        label.append(f"{icon} ", style=style)
        label.append(f"[{step.get('index', '?')}] ", style="dim")
        label.append(step.get("description", "Unnamed step"), style=style)
        label.append(f"  ({step.get('agent', '?')})", style="dim italic")

        if is_current:
            label.append("  ← current", style="bold cyan")

        branch = tree.add(label)

        tools = step.get("tools", [])
        if tools:
            branch.add(Text("tools: " + ", ".join(tools), style="dim yellow"))

        model = step.get("model")
        if model:
            branch.add(Text(f"model: {model}", style="dim magenta"))

        error = step.get("error")
        if error:
            branch.add(Text(f"error: {error}", style="bold red"))

    def _build_summary_text(self, steps: List[Dict[str, Any]]) -> str:
        """Build a one-line summary string for the panel subtitle.

        Args:
            steps: List of step dicts.

        Returns:
            A summary string like '2/7 complete · 1 running · 4 pending'.
        """
        counts: Dict[str, int] = {}
        for step in steps:
            s = step.get("status", "pending")
            counts[s] = counts.get(s, 0) + 1

        total = len(steps)
        done = counts.get("success", 0)
        parts = [f"{done}/{total} complete"]
        for status in ("running", "failed", "retrying", "pending", "skipped"):
            if counts.get(status, 0):
                parts.append(f"{counts[status]} {status}")

        return " · ".join(parts)


# ────────────────────────────────────────────────────────────────────────────

"""progress_tracker.py — Sentinel live step progress tracker.

Provides a Rich-backed live progress display for pipeline execution.
Supports multi-step progress bars, per-step spinners, elapsed timing,
and a final execution summary table.
"""

import time
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional

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


class ProgressTracker:
    """Tracks and displays live pipeline execution progress using Rich.

    Attributes:
        console: The Rich Console to render output to.
    """

    def __init__(self, console: Optional[Console] = None) -> None:
        """Initialise the ProgressTracker.

        Args:
            console: Optional Rich Console. Creates a new one if not provided.
        """
        self.console = console or Console()
        self._progress: Optional[Progress] = None
        self._live: Optional[Live] = None
        self._task_ids: Dict[str, TaskID] = {}
        self._step_timings: Dict[str, float] = {}
        self._pipeline_task_id: Optional[TaskID] = None

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

    # ------------------------------------------------------------------
    # Step-level updates
    # ------------------------------------------------------------------

    def start_step(self, step_index: int, description: str) -> None:
        """Mark a step as active in the live display.

        Args:
            step_index: The step's pipeline index.
            description: Human-readable step description.
        """
        key = str(step_index)
        self._step_timings[key] = time.monotonic()

        if self._progress and key in self._task_ids:
            self._progress.update(
                self._task_ids[key],
                description=f"  ◉ [{step_index}] {description}",
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

        Usage:
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
        table.add_column("Status", width=12)
        table.add_column("Time", width=8, style="dim")

        total_success = 0
        total_failed = 0

        for step in steps:
            idx = step.get("index", "?")
            status = step.get("status", "pending")
            elapsed = self._step_timings.get(str(idx))
            time_str = f"{elapsed:.1f}s" if elapsed is not None else "—"

            if status == "success":
                total_success += 1
                status_text = Text("✔ success", style="bold green")
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

        Usage:
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
