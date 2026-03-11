"""diff_viewer.py — Sentinel colored diff renderer.

Renders unified diffs with full Rich syntax highlighting: additions in
green, deletions in red, hunks in yellow, and unchanged context in dim
white. Accepts raw unified-diff strings or two text blobs for comparison.
"""

import difflib
from typing import List, Optional, Sequence

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text


class DiffViewer:
    """Renders colored unified diffs using Rich.

    Attributes:
        console: The Rich Console to render output to.
    """

    def __init__(self, console: Optional[Console] = None) -> None:
        """Initialise the DiffViewer.

        Args:
            console: Optional Rich Console instance. Creates a new one if
                     not provided.
        """
        self.console = console or Console()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_diff(
        self,
        diff_text: str,
        title: str = "Diff",
        filename: Optional[str] = None,
    ) -> None:
        """Render a pre-computed unified diff string with colour.

        Args:
            diff_text: A unified diff string (output of `git diff` or
                       difflib.unified_diff).
            title: Panel title string.
            filename: Optional filename hint used in the panel subtitle.
        """
        subtitle = filename or ""
        lines = diff_text.splitlines()
        rendered = Text()

        for line in lines:
            rendered.append_text(self._style_line(line))
            rendered.append("\n")

        self.console.print(
            Panel(
                rendered,
                title=f"[bold cyan]{title}[/bold cyan]",
                subtitle=f"[dim]{subtitle}[/dim]" if subtitle else "",
                border_style="cyan",
                expand=True,
            )
        )

    def render_comparison(
        self,
        original: str,
        modified: str,
        fromfile: str = "original",
        tofile: str = "modified",
        context_lines: int = 3,
        title: str = "Diff",
    ) -> None:
        """Generate and render a diff between two text blobs.

        Args:
            original: The original text content.
            modified: The modified text content.
            fromfile: Label for the original file side.
            tofile: Label for the modified file side.
            context_lines: Number of unchanged lines to show around changes.
            title: Panel title string.
        """
        diff_lines = list(
            difflib.unified_diff(
                original.splitlines(keepends=True),
                modified.splitlines(keepends=True),
                fromfile=fromfile,
                tofile=tofile,
                n=context_lines,
            )
        )

        if not diff_lines:
            self.console.print(
                Panel(
                    "[dim]No differences found.[/dim]",
                    title=f"[bold cyan]{title}[/bold cyan]",
                    border_style="green",
                )
            )
            return

        diff_text = "".join(diff_lines)
        self.render_diff(diff_text, title=title, filename=f"{fromfile} → {tofile}")

    def render_file_diff(
        self,
        original_path: str,
        modified_path: str,
        context_lines: int = 3,
    ) -> None:
        """Load two files from disk and render their diff.

        Args:
            original_path: Path to the original file.
            modified_path: Path to the modified file.
            context_lines: Number of unchanged context lines to show.
        """
        with open(original_path, encoding="utf-8") as f:
            original = f.read()
        with open(modified_path, encoding="utf-8") as f:
            modified = f.read()

        self.render_comparison(
            original=original,
            modified=modified,
            fromfile=original_path,
            tofile=modified_path,
            context_lines=context_lines,
            title="File Diff",
        )

    def render_inline_patch(self, patch: str, language: str = "diff") -> None:
        """Render a patch using Rich Syntax highlighting.

        Useful for displaying patches as code blocks with line numbers.

        Args:
            patch: The diff/patch string to render.
            language: Syntax language identifier (default: 'diff').
        """
        syntax = Syntax(
            patch,
            language,
            theme="monokai",
            line_numbers=True,
            word_wrap=True,
        )
        self.console.print(syntax)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _style_line(self, line: str) -> Text:
        """Apply colour styling to a single diff line.

        Args:
            line: A single line from a unified diff.

        Returns:
            A Rich Text object with the appropriate style applied.
        """
        if line.startswith("+++") or line.startswith("---"):
            return Text(line, style="bold white")
        if line.startswith("+"):
            return Text(line, style="bold green")
        if line.startswith("-"):
            return Text(line, style="bold red")
        if line.startswith("@@"):
            return Text(line, style="bold yellow")
        if line.startswith("diff ") or line.startswith("index "):
            return Text(line, style="bold cyan")
        return Text(line, style="dim white")

    def summarise(self, diff_text: str) -> dict:
        """Parse a unified diff and return a count summary.

        Args:
            diff_text: A unified diff string.

        Returns:
            A dict with 'additions', 'deletions', and 'files_changed' counts.
        """
        additions = 0
        deletions = 0
        files: set = set()

        for line in diff_text.splitlines():
            if line.startswith("+") and not line.startswith("+++"):
                additions += 1
            elif line.startswith("-") and not line.startswith("---"):
                deletions += 1
            elif line.startswith("+++ "):
                files.add(line[4:].split("\t")[0])

        return {
            "additions": additions,
            "deletions": deletions,
            "files_changed": len(files),
        }
