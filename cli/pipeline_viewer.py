"""pipeline_viewer.py — Sentinel pipeline tree visualizer.

Renders the active pipeline as a Rich Tree, showing each step's index,
assigned agent, status, and tool bindings. Status is colour-coded:
pending (dim), running (cyan/bold), success (green), failed (red),
skipped (yellow).
"""
from __future__ import annotations

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
