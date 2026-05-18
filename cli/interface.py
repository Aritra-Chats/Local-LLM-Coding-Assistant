"""interactive_ui.py — Sentinel interactive terminal UI.

The main REPL loop. Uses Rich for output and prompt_toolkit for the
input prompt. Registers all supported slash commands and delegates
task input to the agent pipeline.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from cli.command_palette import Command, CommandParser, ParsedInput
from cli.diff_viewer import DiffViewer
from cli.display import PipelineViewer
from cli.display import ProgressTracker
import threading
from cli.progress_tracker import NEED_INPUT, BACKEND_DONE
from memory.session_store import SessionManager

console = Console()

_PROMPT_STYLE = Style.from_dict(
    {
        "prompt": "bold ansicyan",
    }
)


class InteractiveUI:
    """Main interactive REPL for Sentinel.

    Attributes:
        session: The active SessionManager instance.
        parser: The CommandParser with all commands registered.
        pipeline_viewer: PipelineViewer instance for /pipeline output.
        diff_viewer: DiffViewer instance for diff rendering.
        progress: ProgressTracker for live step progress.
        _running: Whether the REPL loop is active.
    """

    def __init__(self, session: SessionManager) -> None:
        """Initialise the UI with a live session.

        Args:
            session: An already-started SessionManager instance.
        """
        self.session = session
        self.parser = CommandParser()
        self.pipeline_viewer = PipelineViewer(console=console)
        self.diff_viewer = DiffViewer(console=console)
        self.progress = ProgressTracker(console=console)
        self._running = False
        self._runtime: Any = None   # Set from main.py for runtime-dependent commands
        self._last_context: Any = None   # Stored after each pipeline run
        self._prompt_session: PromptSession = PromptSession(
            history=InMemoryHistory(),
            auto_suggest=AutoSuggestFromHistory(),
            style=_PROMPT_STYLE,
        )
        self._backend_thread: Optional[threading.Thread] = None
        self._register_commands()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the interactive REPL loop.

        The main thread alternates between two states:

        REPL state
            prompt_session.prompt() blocks on stdin. No pipeline is running.

        RENDER state
            self.progress._drain_one() blocks on the render queue while the
            backend supervisor thread executes the pipeline. Worker threads
            enqueue display callables consumed here.

            NEED_INPUT sentinel: backend needs approval. Lives already stopped
            by paused_for_input() on the backend. Main thread prints panel,
            reads answer via prompt_session.prompt(), puts answer in
            progress._response_queue.

            BACKEND_DONE sentinel: pipeline finished. Exit render loop,
            return to REPL state.
        """
        self._running = True
        tracker = self.progress

        while self._running:

            # ── REPL state ────────────────────────────────────────────────
            try:
                raw = self._prompt_session.prompt(
                    [("class:prompt", "sentinel › ")]
                )
            except (EOFError, KeyboardInterrupt):
                self._running = False
                break

            if not raw.strip():
                continue

            parsed = self.parser.parse(raw)

            if parsed.is_command:
                known = self.parser.dispatch(parsed)
                if not known:
                    console.print(
                        f"[bold red]Unknown command:[/bold red] /{parsed.command}  "
                        "[dim](type /help for available commands)[/dim]"
                    )
                continue

            if not parsed.is_task:
                continue

            # ── Spawn backend supervisor thread ───────────────────────────
            prompt_text = parsed.raw.strip()

            def _backend_target(p: str = prompt_text) -> None:
                try:
                    self._handle_task(p)
                finally:
                    if tracker._render_queue is not None:
                        tracker._render_queue.put(BACKEND_DONE)

            self._backend_thread = threading.Thread(
                target=_backend_target,
                daemon=True,
                name="sentinel-backend",
            )
            self._backend_thread.start()

            # ── RENDER state ──────────────────────────────────────────────
            if tracker._render_queue is None:
                # Rich unavailable — block until backend finishes.
                self._backend_thread.join()
                self._backend_thread = None
                continue

            while True:
                item = tracker._drain_one()

                if item is BACKEND_DONE:
                    break

                if item is NEED_INPUT:
                    # Lives already stopped by paused_for_input() on backend.
                    # Main thread owns the terminal; print panel and read.
                    panel = tracker._pending_approval_panel
                    if panel is not None:
                        tracker.console.print(panel)
                        tracker._pending_approval_panel = None

                    try:
                        answer = self._prompt_session.prompt(
                            [("class:prompt", "  Apply? [Y/n/A] › ")]
                        ).strip().lower()
                    except (EOFError, KeyboardInterrupt):
                        answer = "n"

                    # paused_for_input() finally-block restarts Lives and
                    # clears _backend_active when response_queue.get() returns.
                    tracker._response_queue.put(answer)

            self._backend_thread = None

    # ------------------------------------------------------------------
    # Task handler (stub — wired to pipeline in full implementation)
    # ------------------------------------------------------------------

    def _handle_task(self, prompt: str) -> None:
        """Handle a user task prompt.

        Records the turn and emits a placeholder response until the
        agent pipeline is wired in.

        Args:
            prompt: The raw task string entered by the user.
        """
        self.session.add_turn("user", prompt)
        console.print(
            Panel(
                f"[dim]Task received:[/dim] {prompt}\n"
                "[dim italic]Pipeline execution will be wired here.[/dim italic]",
                title="[bold cyan]Sentinel[/bold cyan]",
                border_style="cyan",
            )
        )
        self.session.add_turn("assistant", "[pipeline stub]")

    # ------------------------------------------------------------------
    # Command registration
    # ------------------------------------------------------------------

    def _register_commands(self) -> None:
        """Register all supported slash commands with the parser."""
        commands: List[Command] = [
            Command(
                name="help",
                description="Show all available commands.",
                handler=self._cmd_help,
                aliases=["h", "?"],
            ),
            Command(
                name="status",
                description="Show current session and system status.",
                handler=self._cmd_status,
            ),
            Command(
                name="pipeline",
                description="Display the active pipeline and step states.",
                handler=self._cmd_pipeline,
            ),
            Command(
                name="models",
                description="List available local models and their status.",
                handler=self._cmd_models,
            ),
            Command(
                name="context",
                description="Show the context payload for the last step.",
                handler=self._cmd_context,
            ),
            Command(
                name="index",
                description="Rebuild the project index for the current workspace.",
                handler=self._cmd_index,
            ),
            Command(
                name="syscheck",
                description="Run a hardware and dependency check.",
                handler=self._cmd_syscheck,
            ),
            Command(
                name="tasks",
                description="List tasks in the current session.",
                handler=self._cmd_tasks,
            ),
            Command(
                name="resume",
                description="Resume a saved session by ID.",
                usage="/resume <session_id>",
                handler=self._cmd_resume,
            ),
            Command(
                name="clear",
                description="Clear the terminal screen.",
                handler=self._cmd_clear,
                aliases=["cls"],
            ),
            Command(
                name="exit",
                description="Save session and exit Sentinel.",
                handler=self._cmd_exit,
                aliases=["quit", "q"],
            ),
            Command(
                name="session",
                description="Show detailed session info (ID, project, turns, hardware mode).",
                handler=self._cmd_session,
            ),
            Command(
                name="diff",
                description="Show the diff produced by the last file edit.",
                handler=self._cmd_diff,
            ),
            Command(
                name="mode",
                description="Show or switch the hardware mode (minimal/standard/advanced) or online/offline mode.",
                usage="/mode [minimal|standard|advanced|online|offline]",
                handler=self._cmd_mode,
            ),
            Command(
                name="files",
                description="Show all files created or modified during the current session.",
                handler=self._cmd_files,
            ),
            Command(
                name="refresh",
                description="Refresh the Ollama Cloud model cache (online mode). Re-probes which models are accessible on your plan.",
                usage="/refresh",
                handler=self._cmd_refresh,
            ),
        ]
        self.parser.register_many(commands)

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------

    def _cmd_help(self, parsed: ParsedInput) -> None:
        table = Table(title="Sentinel Commands", border_style="cyan", show_header=True)
        table.add_column("Command", style="bold cyan", no_wrap=True)
        table.add_column("Description", style="white")
        for name, desc in self.parser.get_help_table():
            table.add_row(name, desc)
        console.print(table)

    def _cmd_status(self, parsed: ParsedInput) -> None:
        summary = self.session.summary()
        table = Table(title="Session Status", border_style="green", show_header=False)
        table.add_column("Key", style="bold green")
        table.add_column("Value", style="white")
        table.add_row("Session ID", summary["session_id"])
        table.add_row("Created", summary["created_at"])
        table.add_row("Turns", str(summary["turn_count"]))
        table.add_row("Active Task", "Yes" if summary["has_active_task"] else "No")
        table.add_row("Pipeline", "Active" if summary["has_pipeline"] else "None")
        console.print(table)

    def _cmd_pipeline(self, parsed: ParsedInput) -> None:
        state = self.session.pipeline_state
        if state is None:
            console.print("[dim]No active pipeline.[/dim]")
            return
        self.pipeline_viewer.render(state)

    def _cmd_models(self, parsed: ParsedInput) -> None:
        import json as _json
        import urllib.request as _req
        ollama_url = "http://localhost:11434/api/tags"
        try:
            with _req.urlopen(ollama_url, timeout=5) as resp:
                data = _json.loads(resp.read().decode())
            models = data.get("models", [])
            if not models:
                console.print("[dim]No models found — pull one with: ollama pull codellama:13b[/dim]")
                return
            table = Table(title="Available Ollama Models", border_style="cyan", show_header=True)
            table.add_column("Model", style="bold cyan")
            table.add_column("Size", style="dim")
            table.add_column("Modified", style="dim")
            for m in models:
                name = m.get("name", "?")
                size = f"{m.get('size', 0) // 1_000_000:.0f} MB"
                mod  = m.get("modified_at", "")[:19]
                table.add_row(name, size, mod)
            console.print(table)
        except Exception as exc:
            console.print(f"[red]Could not reach Ollama at {ollama_url}: {exc}[/red]")

    def _cmd_context(self, parsed: ParsedInput) -> None:
        import json as _json
        ctx = self._last_context or self.session.metadata.get("last_context")
        if ctx is None:
            console.print("[dim]No context recorded yet — run a task first.[/dim]")
            return
        try:
            text = _json.dumps(ctx, indent=2, default=str)
        except Exception:
            text = str(ctx)
        console.print(Panel(
            text[:4000] + ("\n[dim]…(truncated)[/dim]" if len(text) > 4000 else ""),
            title="[bold cyan]Last Step Context[/bold cyan]",
            border_style="cyan",
        ))

    def _cmd_index(self, parsed: ParsedInput) -> None:
        project_root = self.session.metadata.get("project_root", "")
        if not project_root:
            console.print("[dim]No project root known — start sentinel from a project directory.[/dim]")
            return
        console.print(f"[bold yellow]Indexing project at {project_root}…[/bold yellow]")
        try:
            from context.rag_search import RagEngine
            engine = RagEngine(project_root=project_root)
            count = engine.index_project()
            console.print(f"[green]✔[/green] Indexed {count} chunks.")
        except Exception as exc:
            console.print(f"[yellow]Could not index: {exc}[/yellow]")

    def _cmd_syscheck(self, parsed: ParsedInput) -> None:
        import platform
        import shutil

        table = Table(title="System Check", border_style="yellow", show_header=False)
        table.add_column("Check", style="bold yellow")
        table.add_column("Result", style="white")
        table.add_row("OS", platform.system() + " " + platform.release())
        table.add_row("Python", platform.python_version())
        table.add_row("Ollama", "Found" if shutil.which("ollama") else "[red]Not found[/red]")
        table.add_row("Git", "Found" if shutil.which("git") else "[red]Not found[/red]")

        import importlib.util
        for pkg in ["rich", "prompt_toolkit", "pdfplumber"]:
            found = importlib.util.find_spec(pkg) is not None
            table.add_row(pkg, "[green]OK[/green]" if found else "[red]Missing[/red]")

        console.print(table)

    def _cmd_tasks(self, parsed: ParsedInput) -> None:
        history = self.session.get_history()
        task_turns = [t for t in history if t["role"] == "user"]
        if not task_turns:
            console.print("[dim]No tasks in this session.[/dim]")
            return
        table = Table(title="Session Tasks", border_style="cyan", show_header=True)
        table.add_column("#", style="dim", width=4)
        table.add_column("Timestamp", style="dim")
        table.add_column("Task", style="white")
        for i, turn in enumerate(task_turns, start=1):
            table.add_row(str(i), turn.get("timestamp", ""), turn["content"][:80])
        console.print(table)

    def _cmd_resume(self, parsed: ParsedInput) -> None:
        if not parsed.args:
            sessions = SessionManager.list_sessions()
            if not sessions:
                console.print("[dim]No saved sessions found.[/dim]")
            else:
                table = Table(title="Saved Sessions", border_style="cyan")
                table.add_column("Session ID", style="cyan")
                for sid in sessions:
                    table.add_row(sid)
                console.print(table)
                console.print("[dim]Use /resume <session_id> to resume.[/dim]")
            return

        target_id = parsed.args[0]
        console.print(f"[dim]Resuming session [bold]{target_id}[/bold]…[/dim]")
        try:
            new_session = SessionManager(session_id=target_id)
            self.session = new_session
            console.print(
                f"[bold green]Session resumed.[/bold green] "
                f"{new_session.summary()['turn_count']} turns loaded."
            )
        except FileNotFoundError:
            console.print(f"[bold red]Session not found:[/bold red] {target_id}")

    def _cmd_session(self, parsed: ParsedInput) -> None:
        summary = self.session.summary()
        meta = self.session.metadata or {}
        table = Table(title="Session Details", border_style="green", show_header=False)
        table.add_column("Key", style="bold green")
        table.add_column("Value", style="white")
        table.add_row("Session ID",   summary["session_id"])
        table.add_row("Created",      summary["created_at"])
        table.add_row("Turns",        str(summary["turn_count"]))
        table.add_row("Project Root", meta.get("project_root", "[not set]"))
        table.add_row("Hardware Mode",meta.get("hardware_mode", "[not set]"))
        table.add_row("Started At",   meta.get("started_at", "[unknown]"))
        table.add_row("Pipeline",     "Active" if summary["has_pipeline"] else "None")
        console.print(table)

    def _cmd_diff(self, parsed: ParsedInput) -> None:
        last_diff = self.session.metadata.get("last_diff")
        if last_diff is None:
            console.print("[dim]No diff recorded yet — run a task that writes files first.[/dim]")
            return
        self.diff_viewer.render_comparison(
            last_diff.get("old", ""),
            last_diff.get("new", ""),
            fromfile=last_diff.get("fromfile", "before"),
            tofile=last_diff.get("tofile", "after"),
            title=last_diff.get("title", "Last diff"),
        )

    def _cmd_mode(self, parsed: ParsedInput) -> None:
        hw_valid = ("minimal", "standard", "advanced")
        io_valid = ("online", "offline")
        all_valid = hw_valid + io_valid
        if not parsed.args:
            current_hw = self.session.metadata.get("hardware_mode", "[unknown]")
            import os
            current_io = os.environ.get("SENTINEL_MODE", "offline")
            console.print(
                f"  Hardware mode : [bold cyan]{current_hw}[/bold cyan]\n"
                f"  Network mode  : [bold cyan]{current_io}[/bold cyan]\n"
                f"  To switch hw  : /mode <{'|'.join(hw_valid)}>\n"
                f"  To switch net : /mode <{'|'.join(io_valid)}>\n"
                "  Note: hardware mode change takes effect on next restart (--hw-mode flag)."
            )
            return
        mode = parsed.args[0].lower()
        if mode not in all_valid:
            console.print(f"[red]Unknown mode '{mode}'. Choose: {', '.join(all_valid)}[/red]")
            return
        if mode in hw_valid:
            self.session.metadata["hardware_mode"] = mode
            console.print(
                f"[yellow]Hardware mode set to [bold]{mode}[/bold] in session metadata.\n"
                "Restart Sentinel with [bold]--hw-mode " + mode + "[/bold] to apply."
            )
        else:
            import os
            os.environ["SENTINEL_MODE"] = mode
            self.session.metadata["sentinel_mode"] = mode
            console.print(
                f"[green]✔[/green] Network mode switched to [bold]{mode.upper()}[/bold] for this session."
            )

    def _cmd_files(self, parsed: ParsedInput) -> None:
        """Show all files changed during the active pipeline runs."""
        from datetime import datetime
        runtime = getattr(self, "_runtime", None)
        engine = getattr(runtime, "_engine", None) if runtime else None
        fcm = getattr(engine, "file_change_map", None) if engine else None

        if fcm is None:
            console.print("[dim]No file change map available for this session.[/dim]")
            return

        events = fcm.all_events()
        if not events:
            console.print("[dim]No files have been changed in this session yet.[/dim]")
            return

        from rich.table import Table as _Table
        table = _Table(title="File Changes — Current Session", border_style="cyan")
        table.add_column("Operation", style="bold", no_wrap=True)
        table.add_column("Logical Path")
        table.add_column("Absolute Path")
        table.add_column("Step")
        table.add_column("Agent")
        table.add_column("Time")

        op_colours = {"create": "green", "modify": "yellow", "delete": "red"}
        for ev in events:
            colour = op_colours.get(ev.operation, "white")
            ts = datetime.fromtimestamp(ev.timestamp_ms / 1000).strftime("%H:%M:%S")
            table.add_row(
                f"[{colour}]{ev.operation}[/{colour}]",
                ev.logical_path,
                ev.absolute_path,
                ev.step_id[:8] if ev.step_id else "-",
                ev.agent or "-",
                ts,
            )
        console.print(table)

    def _cmd_refresh(self, parsed: ParsedInput) -> None:
        """Re-probe Ollama Cloud and rebuild the accessible-model cache."""
        import os as _os
        mode = _os.environ.get("SENTINEL_MODE", "offline")
        if mode != "online":
            console.print(
                "[yellow]! /refresh is only useful in online mode. "
                "Switch with [bold]/mode online[/bold] first.[/yellow]"
            )
            return

        console.print("[cyan]Refreshing Ollama Cloud model cache…[/cyan]")
        try:
            import importlib.util
            _probe = Path(__file__).parent.parent / "scripts" / "cloud_model_probe.py"
            spec   = importlib.util.spec_from_file_location("cloud_model_probe", _probe)
            mod    = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
            spec.loader.exec_module(mod)                    # type: ignore[union-attr]
            ok = mod.run_probe(force=True, verbose=True)
            if ok:
                console.print("[bold green]✓ Cloud model cache refreshed.[/bold green]")
            else:
                console.print(
                    "[red]✗ Refresh failed — check OLLAMA_API_KEY and connectivity.[/red]"
                )
        except Exception as _exc:
            console.print(f"[red]Refresh error: {_exc}[/red]")

    def _cmd_clear(self, parsed: ParsedInput) -> None:
        console.clear()

    def _cmd_exit(self, parsed: ParsedInput) -> None:
        console.print("[dim]Saving session…[/dim]")
        self.session.save()
        console.print("[bold cyan]Goodbye.[/bold cyan]")
        self._running = False


# ────────────────────────────────────────────────────────────────────────────
# launch / main  (from cli_entry)
# ────────────────────────────────────────────────────────────────────────────





BANNER = """
 ███████╗███████╗███╗   ██╗████████╗██╗███╗   ██╗███████╗██╗
 ██╔════╝██╔════╝████╗  ██║╚══██╔══╝██║████╗  ██║██╔════╝██║
 ███████╗█████╗  ██╔██╗ ██║   ██║   ██║██╔██╗ ██║█████╗  ██║
 ╚════██║██╔══╝  ██║╚██╗██║   ██║   ██║██║╚██╗██║██╔══╝  ██║
 ███████║███████╗██║ ╚████║   ██║   ██║██║ ╚████║███████╗███████╗
 ╚══════╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝╚═╝  ╚═══╝╚══════╝╚══════╝
"""


def _print_banner() -> None:
    """Render the Sentinel ASCII banner and version header."""
    console.print(Text(BANNER, style="bold cyan"))
    console.print(
        Panel(
            "[bold white]Sentinel[/bold white] · Local Autonomous Development Assistant\n"
            "[dim]Type a task or [bold]/help[/bold] to see available commands.[/dim]",
            border_style="cyan",
            expand=True,
        )
    )


def _check_environment() -> bool:
    """Run a basic pre-flight environment check.

    Returns:
        True if the environment is ready, False if a critical dependency
        is missing.
    """
    import importlib.util

    required = ["rich", "prompt_toolkit"]
    missing = [pkg for pkg in required if importlib.util.find_spec(pkg) is None]

    if missing:
        console.print(
            f"[bold red]Missing required packages:[/bold red] {', '.join(missing)}\n"
            "Run: [bold]pip install " + " ".join(missing) + "[/bold]"
        )
        return False
    return True


def launch(session_id: Optional[str] = None) -> None:
    """Launch the Sentinel CLI.

    Args:
        session_id: Optional session ID to resume an existing session.
                    If None a new session is created.
    """
    if not _check_environment():
        sys.exit(1)

    _print_banner()

    session = SessionManager(session_id=session_id)
    session.start()

    ui = InteractiveUI(session=session)

    try:
        ui.run()
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted. Saving session…[/dim]")
        session.save()
        console.print("[bold cyan]Goodbye.[/bold cyan]")
        sys.exit(0)
    finally:
        session.save()


def main() -> None:
    """CLI entry point invoked by the pyproject.toml console script."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="sentinel",
        description="Sentinel — Local Autonomous Development Assistant",
    )
    parser.add_argument(
        "--resume",
        metavar="SESSION_ID",
        default=None,
        help="Resume a previously saved session by ID.",
    )
    args = parser.parse_args()
    launch(session_id=args.resume)


