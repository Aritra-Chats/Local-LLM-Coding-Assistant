"""main.py — Sentinel Local Autonomous Development Assistant
Entry Point & Runtime Orchestrator

Usage
-----
    python main.py
    python main.py --resume <session_id>
    python main.py --project /path/to/project
    python main.py --mode   minimal|standard|advanced
    python main.py --no-bootstrap
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()

_ENTRY_DIR = Path(__file__).resolve().parent


# ── Deferred imports ─────────────────────────────────────────────────────
def _import_modules() -> Dict[str, Any]:
    from core.bootstrap import Bootstrap
    from core.execution_engine import ConcreteExecutionEngine
    from core.model_router import ConcreteModelRouter
    from execution.pipeline import DynamicPipelineGenerator
    from tasks.task_manager import TaskPlanner
    from agents import build_agent_registry, ConcreteSupervisorAgent
    from cli.interface import InteractiveUI, launch
    from memory.session_store import SessionManager
    from context.context_builder import ConcreteContextBuilder
    from learning.metrics_tracker import PerformanceTracker
    from learning.feedback_loop import LearningPipelineOptimizer
    from tools import build_default_registry
    return {
        "Bootstrap":                Bootstrap,
        "ConcreteExecutionEngine":  ConcreteExecutionEngine,
        "ConcreteModelRouter":      ConcreteModelRouter,
        "DynamicPipelineGenerator": DynamicPipelineGenerator,
        "TaskPlanner":              TaskPlanner,
        "build_agent_registry":     build_agent_registry,
        "ConcreteSupervisorAgent":  ConcreteSupervisorAgent,
        "launch":                   launch,
        "InteractiveUI":            InteractiveUI,
        "SessionManager":           SessionManager,
        "ConcreteContextBuilder":   ConcreteContextBuilder,
        "PerformanceTracker":       PerformanceTracker,
        "LearningPipelineOptimizer": LearningPipelineOptimizer,
        "build_default_registry":   build_default_registry,
    }


# ── SentinelRuntime ──────────────────────────────────────────────────────

class SentinelRuntime:
    def __init__(self, project_root: str = "", force_mode: Optional[str] = None,
                 skip_bootstrap: bool = False) -> None:
        self.project_root    = Path(project_root or os.getcwd()).resolve()
        self._force_mode     = force_mode
        self._skip_bootstrap = skip_bootstrap
        self.profile = self._supervisor = self._engine = None
        self._agent_registry = self._tool_registry = {}
        self._context_builder = self._task_planner = self._pipeline_gen = None
        self._model_router = self._perf_tracker = self._pipeline_opt = None
        self._mods: Dict[str, Any] = {}

    def initialise(self, session_id: str = "") -> None:
        self._mods = _import_modules()
        console.print("[dim]Step 1/6 — Loading hardware profile…[/dim]")
        self.profile = self._bootstrap()

        console.print("[dim]Step 2/6 — Initialising model router…[/dim]")
        self._model_router = self._mods["ConcreteModelRouter"](
            hardware_profile=self.profile, force_mode=self._force_mode)
        hw_mode = self._model_router.get_hardware_profile()
        console.print(
            f"  [green]✔[/green] Model router  [bold]{hw_mode.upper()}[/bold]  "
            f"coding={self._model_router.select_coding_model()}  "
            f"reasoning={self._model_router.select_reasoning_model()}"
        )

        console.print("[dim]Step 3/6 — Building agent registry…[/dim]")
        self._agent_registry = self._mods["build_agent_registry"]()
        self._supervisor = self._agent_registry["supervisor"]
        console.print(f"  [green]✔[/green] Agents  ({len(self._agent_registry)} registered)")

        console.print("[dim]Step 4/6 — Building tool registry…[/dim]")
        self._tool_registry = self._mods["build_default_registry"]()
        console.print(
            f"  [green]✔[/green] Tools  ({len(self._tool_registry.list_tools())} tools)")

        console.print("[dim]Step 5/6 — Initialising context engine…[/dim]")
        self._context_builder = self._mods["ConcreteContextBuilder"](
            project_root=str(self.project_root))
        self._task_planner = self._mods["TaskPlanner"]()
        self._pipeline_gen = self._mods["DynamicPipelineGenerator"](
            system_mode=hw_mode, mode="solo")
        console.print(
            f"  [green]✔[/green] Context engine  (project: {self.project_root.name})")

        console.print("[dim]Step 6/6 — Initialising learning system…[/dim]")
        self._perf_tracker = self._mods["PerformanceTracker"](
            session_id=session_id or "default")
        self._pipeline_opt = self._mods["LearningPipelineOptimizer"](
            tracker=self._perf_tracker, hardware_mode=hw_mode)
        self._engine = self._mods["ConcreteExecutionEngine"](
            agent_registry=self._agent_registry,
            tool_registry=self._tool_registry,
            show_progress=True,
            on_progress=lambda e: None,
        )
        console.print("[bold green]✔ Sentinel runtime initialised.[/bold green]")

    def _bootstrap(self):
        Bootstrap = self._mods["Bootstrap"]
        try:
            if self._skip_bootstrap:
                from system.hardware_detector import SystemCheck
                from config.hardware_profile import HardwareProfiler
                return HardwareProfiler().classify(SystemCheck().run())
            return Bootstrap().run()
        except Exception as exc:
            console.print(f"  [yellow]⚠[/yellow] Bootstrap error ({exc}); using standard defaults.")
            from config.hardware_profile import HardwareMode, HardwareProfile
            return HardwareProfile(
                mode=HardwareMode.STANDARD,
                recommended_model="codellama:13b",
                context_limit=8192,
                max_pipeline_concurrency=2,
                embedding_model="nomic-embed-text",
                reasoning_model="mistral:13b",
                notes="Bootstrap failed; using standard defaults.",
            )

    def process_prompt(self, prompt: str, session_id: str = "") -> Dict[str, Any]:
        t0   = time.monotonic()
        task = self._supervisor.parse_prompt(prompt)
        task.update({"session_id": session_id, "project_root": str(self.project_root)})
        console.print(
            f"  [cyan]→[/cyan] Goal: [bold]{task['goal']}[/bold]  complexity={task['complexity']}")
        self._supervisor.run(task, {"session_id": session_id})
        plan     = self._task_planner.plan(task)
        pipeline = self._pipeline_gen.from_execution_plan(plan)
        if self._pipeline_opt:
            try:
                pipeline, _ = self._pipeline_opt.optimize(pipeline)
            except Exception:
                pass
        result     = self._engine.run_pipeline(pipeline)
        elapsed_ms = (time.monotonic() - t0) * 1000
        self._record_metrics(pipeline, result)
        return {
            "status": result.status, "summary": result.summary(),
            "result": result, "pipeline": pipeline,
            "elapsed_ms": round(elapsed_ms, 2),
        }

    def _record_metrics(self, pipeline: Any, result: Any) -> None:
        if not self._perf_tracker:
            return
        try:
            category = (pipeline.classification.get("category", "coding")
                        if hasattr(pipeline, "classification")
                        and isinstance(pipeline.classification, dict) else "coding")
            self._perf_tracker.record_pipeline_run(
                category=category, mode=getattr(pipeline, "mode", "solo"),
                success=result.status != "failed",
                elapsed_ms=result.total_elapsed_ms,
                total_steps=len(result.step_results),
                failed_steps=result.failed_steps,
            )
            for sr in result.step_results:
                self._perf_tracker.record_tool_results(sr.tool_results)
        except Exception:
            pass

    def make_task_handler(self, session: Any) -> Callable[[str], None]:
        def _handle(prompt: str) -> None:
            session.add_turn("user", prompt)
            try:
                out = self.process_prompt(prompt, session_id=session.session_id)
                console.print(Panel(
                    out["summary"],
                    title="[bold cyan]Pipeline Complete[/bold cyan]",
                    border_style="green" if out["status"] == "completed" else "yellow",
                ))
                session.pipeline_state = {
                    "status": out["status"], "summary": out["summary"],
                    "elapsed_ms": out["elapsed_ms"],
                }
                session.add_turn("assistant", out["summary"])
            except Exception:
                err = traceback.format_exc()
                console.print(Panel(
                    f"[red]{err}[/red]",
                    title="[bold red]Runtime Error[/bold red]",
                    border_style="red",
                ))
                session.add_turn("assistant", "[error — see above]")
        return _handle


# ── Banner ───────────────────────────────────────────────────────────────

_BANNER = r"""
 ███████╗███████╗███╗   ██╗████████╗██╗███╗   ██╗███████╗██╗
 ██╔════╝██╔════╝████╗  ██║╚══██╔══╝██║████╗  ██║██╔════╝██║
 ███████╗█████╗  ██╔██╗ ██║   ██║   ██║██╔██╗ ██║█████╗  ██║
 ╚════██║██╔══╝  ██║╚██╗██║   ██║   ██║██║╚██╗██║██╔══╝  ██║
 ███████║███████╗██║ ╚████║   ██║   ██║██║ ╚████║███████╗███████╗
 ╚══════╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝╚═╝  ╚═══╝╚══════╝╚══════╝
"""


def _print_banner() -> None:
    console.print(Text(_BANNER, style="bold cyan"))
    console.print(Panel(
        "[bold white]Sentinel[/bold white] · Local Autonomous Development Assistant\n"
        "[dim]Type a task or [bold]/help[/bold] to see available commands.[/dim]",
        border_style="cyan", expand=False,
    ))


# ── Entry point ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="sentinel",
        description="Sentinel — Local Autonomous Development Assistant",
    )
    parser.add_argument("--resume",   metavar="SESSION_ID", default=None)
    parser.add_argument("--project",  metavar="PATH",        default=None)
    parser.add_argument("--mode",     choices=["minimal", "standard", "advanced"], default=None)
    parser.add_argument("--no-bootstrap", action="store_true")
    args = parser.parse_args()

    _print_banner()

    runtime = SentinelRuntime(
        project_root=args.project or os.environ.get("SENTINEL_PROJECT_DIR") or os.getcwd(),
        force_mode=args.mode,
        skip_bootstrap=args.no_bootstrap,
    )

    mods    = _import_modules()
    session = mods["SessionManager"](session_id=args.resume)
    session.start()

    try:
        runtime.initialise(session_id=session.session_id)
    except Exception:
        console.print(Panel(
            "[red]" + traceback.format_exc() + "[/red]",
            title="[bold red]Initialisation Error[/bold red]",
            border_style="red",
        ))
        sys.exit(1)

    ui = mods["InteractiveUI"](session=session)
    ui._handle_task = runtime.make_task_handler(session)

    try:
        ui.run()
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted. Saving session…[/dim]")
    finally:
        session.save()
        if runtime._perf_tracker:
            from config.settings import SENTINEL_HOME
            metrics_path = SENTINEL_HOME / "metrics" / f"{session.session_id}.json"
            try:
                metrics_path.parent.mkdir(parents=True, exist_ok=True)
                runtime._perf_tracker.persist(metrics_path)
                console.print(f"[dim]Metrics saved → {metrics_path}[/dim]")
            except Exception:
                pass
        console.print("[bold cyan]Goodbye.[/bold cyan]")


if __name__ == "__main__":
    main()
