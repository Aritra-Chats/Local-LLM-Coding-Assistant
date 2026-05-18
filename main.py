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
import re
import shlex
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()

_ENTRY_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# BUG-1 fix helper: domain-aware sub_task → step matching
# ---------------------------------------------------------------------------

# Maps routing_domain / domain values → canonical agent names used in steps.
_DOMAIN_TO_AGENT: Dict[str, str] = {
    "coding_frontend":  "coding",
    "coding_backend":   "coding",
    "coding_general":   "coding",
    "data_science":     "coding",
    "debugging":        "debugging",
    "research":         "research",
    "devops":           "devops",
    "security":         "debugging",
    "system":           "system",
    "math":             "reasoning",
    "creative":         "coding",
    "other":            "coding",
    "root":             "coding",
}


def _pick_best_supervisor_cloud_model(
    cloud_models: list,
    discovery_engine: Optional[Any] = None,
) -> str:
    """Return the best cloud model tag for the supervisor (uncapped).

    Uses the same two-layer scoring pipeline as OnlineModelDiscoveryEngine:
      Layer 1 (metadata): parameter size × complexity fit + "reasoning" keyword score
      Layer 2 (web):      web-search benchmark mention score (via discovery_engine)

    No parameter cap is applied — the supervisor benefits from the largest,
    most capable model available.

    Args:
        cloud_models:     List of model dicts from OllamaCloudClient.list_models().
                          Dicts may use ``"name"`` (api/tags) or ``"id"``
                          (/v1/models OpenAI format) — both are handled.
        discovery_engine: Optional :class:`~core.online_model_discovery.OnlineModelDiscoveryEngine`
                          instance.  When provided, web-search scores are included.
                          When ``None`` (e.g. at very early startup), only metadata
                          scoring is used.

    Returns:
        The base model tag (no ``:cloud`` suffix) of the selected model,
        or an empty string if the list is empty.
    """
    from core.online_model_discovery import (
        _complexity_size_score,
        _extract_param_count_from_model,
        _name_keyword_score,
    )

    if not cloud_models:
        return ""

    # Fetch web-search scores once if a discovery engine is available
    web_scores: dict = {}
    if discovery_engine is not None:
        try:
            web_scores = discovery_engine._web_search_scores("reasoning", "high")
        except Exception:
            web_scores = {}

    best_tag   = ""
    best_score = -1.0
    for m in cloud_models:
        tag = m.get("name") or m.get("id") or m.get("model", "")
        if not tag:
            continue

        param_b = _extract_param_count_from_model(m, tag)

        # Layer 1: metadata score
        score  = _complexity_size_score(param_b, "high")   # supervisor handles complex tasks
        score += _name_keyword_score(tag, "reasoning")      # supervision is reasoning

        # Layer 2: web-search score (cached per discovery_engine instance)
        base_tag = tag.replace(":cloud", "").replace("-cloud", "")
        score += web_scores.get(base_tag, 0.0)

        if score > best_score:
            best_score = score
            best_tag   = tag

    # Strip :cloud and -cloud suffixes — the direct API uses bare tags
    return re.sub(r"[:-]cloud$", "", best_tag) if best_tag else ""

def _match_subtask_for_step(
    step: Any,
    sub_tasks: List[Dict],
    used_indices: Set[int],
) -> Optional[Dict]:
    """Return the best-matching sub_task for *step*, avoiding re-use.

    Matching priority:
    1. A not-yet-used sub_task whose ``routing_domain`` or ``domain``
       maps to the step's ``agent``.
    2. The first not-yet-used sub_task (positional fallback).
    3. The last sub_task in the list (ultimate fallback — mirrors original
       behaviour without silent clipping).

    Args:
        step:        A :class:`~execution.pipeline.PipelineStep`.
        sub_tasks:   Flat list of sub_task dicts from TaskSegregator.
        used_indices: Set of indices already consumed; mutated in-place
                      when a match is found.

    Returns:
        The matched sub_task dict, or ``None`` when *sub_tasks* is empty.
    """
    if not sub_tasks:
        return None
    step_agent = getattr(step, "agent", "")
    # ── Pass 1: domain-matched, not yet consumed ──────────────────────
    for idx, st in enumerate(sub_tasks):
        if idx in used_indices:
            continue
        domain = st.get("routing_domain") or st.get("domain", "")
        mapped_agent = _DOMAIN_TO_AGENT.get(domain, domain)
        if mapped_agent == step_agent or domain == step_agent:
            used_indices.add(idx)
            return st
    # ── Pass 2: first unconsumed ──────────────────────────────────────
    for idx, st in enumerate(sub_tasks):
        if idx not in used_indices:
            used_indices.add(idx)
            return st
    # ── Pass 3: last entry (never re-added to used_indices) ───────────
    return sub_tasks[-1]


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
                 force_io_mode: Optional[str] = None,
                 skip_bootstrap: bool = False) -> None:
        self.project_root    = Path(project_root or os.getcwd()).resolve()
        self._force_mode     = force_mode      # hardware profile (minimal/standard/advanced)
        self._force_io_mode  = force_io_mode   # "online" | "offline" | None (prompt)
        self._mode: str      = "offline"       # resolved after connectivity check
        self._skip_bootstrap = skip_bootstrap
        self.profile = self._supervisor = self._engine = None
        self._exploration_report = None
        self._agent_registry = self._tool_registry = {}
        self._context_builder = self._task_planner = self._pipeline_gen = None
        self._model_router = self._perf_tracker = self._pipeline_opt = None
        self._mods: Dict[str, Any] = {}
        self._tree_engine: Any = None  # Part 4: ConcreteTreeExecutionEngine

    def initialise(self, session_id: str = "") -> None:
        self._mods = _import_modules()

        # ── Step 0: Connectivity check + mode selection ──────────────────
        from core.connectivity import ConnectivityChecker

        internet_available = ConnectivityChecker.check()

        if not internet_available:
            console.print(Panel(
                "[yellow]⚠  No internet connection detected.[/yellow]\n"
                "Sentinel will run in [bold]OFFLINE[/bold] mode.\n"
                "All inference will use your local Ollama models.",
                title="[bold red]Connectivity[/bold red]",
                border_style="red",
            ))
            resolved_mode = "offline"

        elif self._force_io_mode is not None:
            resolved_mode = self._force_io_mode
            console.print(
                f"[dim]Mode forced via CLI flag:[/dim] [bold]{resolved_mode.upper()}[/bold]"
            )

        else:
            console.print(Panel(
                "[green]●  Internet connection detected.[/green]\n\n"
                "Use the arrow keys to choose a launch mode, then press Enter.\n\n"
                "  [bold cyan]ONLINE mode[/bold cyan]  — Tasks are routed to the best\n"
                "       available cloud model (Ollama Cloud, Claude, Gemini, ChatGPT).\n"
                "       Requires API keys for external providers.\n\n"
                "  [bold white]OFFLINE mode[/bold white] — All inference runs locally via\n"
                "       your Ollama installation. No API keys needed.",
                title="[bold green]Select Operating Mode[/bold green]",
                border_style="green",
            ))
            from cli.selection_menu import select_option

            resolved_mode = select_option(
                title="Select Operating Mode",
                prompt="Use the arrow keys to move, Enter to confirm, and Esc to keep the default.",
                options=[
                    ("online", "ONLINE mode"),
                    ("offline", "OFFLINE mode"),
                ],
                default_index=1,
            )

        os.environ["SENTINEL_MODE"] = resolved_mode
        self._mode = resolved_mode
        console.print(f"  [green]✔[/green] Mode set to [bold]{resolved_mode.upper()}[/bold]")

        # ── API key collection (online mode only) ────────────────────────
        if self._mode == "online":
            from core.api_key_manager import APIKeyManager
            from config.settings import SENTINEL_HOME
            mgr = APIKeyManager(env_file=SENTINEL_HOME / ".env")
            safe_to_proceed = mgr.check_and_collect()
            if not safe_to_proceed:
                self._mode = "offline"
                os.environ["SENTINEL_MODE"] = "offline"
                console.print("[yellow]Continuing in OFFLINE mode.[/yellow]")

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
        self._agent_registry = self._mods["build_agent_registry"](model_router=self._model_router)
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
            require_approval=True,
            context_builder=self._context_builder,
            console=console,
        )

        # Attach async supervisor watchdog (creates SupervisorBus + daemon thread).
        # The tracker is not available yet at construction time — it is created
        # per-run inside run_pipeline().  We pass None here; the supervisor thread
        # will pick up the tracker reference via engine._progress_tracker at
        # runtime since that field is set by run_pipeline() before execution.
        try:
            self._engine.attach_supervisor(self._supervisor)
        except Exception as _sup_err:
            import sys as _sys
            print(f"[Sentinel] Warning: async supervisor could not start: {_sup_err}",
                  file=_sys.stderr)

        # Part 4: Construct TreeExecutionEngine once, reusing existing components.
        from core.tree_execution_engine import ConcreteTreeExecutionEngine
        self._tree_engine = ConcreteTreeExecutionEngine(
            concrete_engine=self._engine,
            task_planner=self._task_planner,
            pipeline_generator=self._pipeline_gen,
            tool_registry=self._tool_registry,
            supervisor_agent=self._supervisor,
        )
        console.print("[bold green]✔ Sentinel runtime initialised.[/bold green]")

    def _bootstrap(self):
        Bootstrap = self._mods["Bootstrap"]
        try:
            if self._skip_bootstrap:
                from system.hardware_detector import SystemCheck
                from config.hardware_profile import HardwareProfiler
                return HardwareProfiler().classify(SystemCheck().run())
            return Bootstrap().run(launch_mode=self._mode)
        except Exception as exc:
            console.print(f"  [yellow]⚠[/yellow] Bootstrap error ({exc}); using standard defaults.")
            from config.hardware_profile import HardwareMode, HardwareProfile
            return HardwareProfile(
                mode=HardwareMode.STANDARD,
                recommended_model="codellama:13b",
                context_limit=8192,
                max_pipeline_concurrency=2,
                embedding_model="nomic-embed-text",
                reasoning_model="mixtral:8x7b",
                notes="Bootstrap failed; using standard defaults.",
            )

    def process_prompt(self, prompt: str, session_id: str = "") -> Dict[str, Any]:
        t0 = time.monotonic()

        # ── Phase 0: Repository Exploration ────────────────────────────────────────────────
        # Explore the project before planning so every downstream agent
        # operates with accurate architectural knowledge.
        self._run_repo_exploration()

        # ── Phase 0.5: Resolve supervisor client for online mode ───────────
        # In online mode we want parse_prompt() to run on the best available
        # cloud reasoning model, not the local one.  We resolve this BEFORE
        # calling parse_prompt so the supervisor already has the right client
        # injected when it does its LLM call.  Falls back to local silently.
        import os as _os_sup
        if _os_sup.environ.get("SENTINEL_MODE", "offline") == "online":
            try:
                _sup_local_client = None
                _coding_agent = self._agent_registry.get("coding")
                if _coding_agent is not None:
                    _sup_local_client = getattr(_coding_agent, "_ollama", None)

                _sup_cloud_client = None
                _sup_cloud_model  = None
                try:
                    from models.ollama_cloud_client import OllamaCloudClient as _OCC_sup
                    _cloud_sup   = _OCC_sup()
                    _cloud_mdls  = _cloud_sup.list_models()
                    if _cloud_mdls:
                        _best_sup = _pick_best_supervisor_cloud_model(
                            _cloud_mdls, discovery_engine=None
                        )
                        if _best_sup:
                            _sup_cloud_client = _cloud_sup
                            _sup_cloud_model  = _best_sup
                except Exception:
                    pass  # cloud unavailable — stay on local

                if _sup_cloud_client and _sup_cloud_model:
                    # Inject cloud client + model into the registered supervisor so
                    # parse_prompt() and run() both use it for this request.
                    self._supervisor.use_client(_sup_cloud_client)
                    self._supervisor._model = _sup_cloud_model
                    console.print(
                        f"  [cyan]→[/cyan] Supervisor using cloud model "
                        f"[bold]{_sup_cloud_model}[/bold]"
                    )
            except Exception:
                pass  # never fail the whole pipeline for supervisor upgrade

        task = self._supervisor.parse_prompt(prompt)
        task.update({"session_id": session_id, "project_root": str(self.project_root)})
        console.print(
            f"  [cyan]→[/cyan] Goal: [bold]{task['goal']}[/bold]  complexity={task['complexity']}")

        # Enrich supervisor context with exploration results
        supervisor_ctx: Dict[str, Any] = {"session_id": session_id}
        if self._exploration_report is not None:
            supervisor_ctx["exploration"] = self._exploration_report.to_dict()
            supervisor_ctx["synopsis"]    = self._exploration_report.synopsis
            supervisor_ctx["stack"]       = self._exploration_report.stack
        self._supervisor.run(task, supervisor_ctx)

        # ── Online mode: task segregation + model discovery ───────────────
        import os
        if os.environ.get("SENTINEL_MODE", "offline") == "online":
            try:
                from core.task_segregator import TaskSegregator
                from core.online_model_discovery import OnlineModelDiscoveryEngine

                # ── Supervisor client / model for online mode ─────────────
                # Re-use the same cloud client already resolved above (if any).
                # Falls back to the local reasoning model if cloud is unavailable.
                ollama_client = None
                coding_agent = self._agent_registry.get("coding")
                if coding_agent is not None:
                    ollama_client = getattr(coding_agent, "_ollama", None)

                sup_client = ollama_client  # default: local
                sup_model  = (
                    self._model_router.select_reasoning_model()
                    if self._model_router else "mistral:7b"
                )

                try:
                    from models.ollama_cloud_client import OllamaCloudClient
                    cloud_sup = OllamaCloudClient()
                    cloud_models = cloud_sup.list_models()
                    if cloud_models:
                        # Pick the best reasoning model available on the cloud:
                        # prefer the largest model (most capable for structured
                        # JSON decomposition) that the API currently serves.
                        best_cloud_model = _pick_best_supervisor_cloud_model(
                            cloud_models, discovery_engine=None
                        )
                        if best_cloud_model:
                            sup_client = cloud_sup
                            sup_model  = best_cloud_model
                            console.print(
                                f"  [cyan]→[/cyan] Supervisor: cloud model "
                                f"[bold]{sup_model}[/bold]"
                            )
                except Exception as _sup_exc:
                    console.print(
                        f"  [dim]Cloud supervisor unavailable ({_sup_exc}); "
                        "using local reasoning model.[/dim]"
                    )

                segregator = TaskSegregator(
                    ollama_client=sup_client,
                    supervisor_model=sup_model,
                )
                sub_tasks = segregator.segregate(prompt)
                for st in sub_tasks:
                    segregator.refine(st)
                    segregator.classify(st)

                # Stamp selected_model onto the task for downstream pipeline steps
                if sub_tasks:
                    task["sub_tasks"] = sub_tasks
            except Exception as exc:
                console.print(f"  [yellow]⚠[/yellow] Online segregation failed ({exc}); using offline path.")

        # ── Part 4: Tree execution path for non-trivial tasks ────────────────
        # When complexity is not "low" (and the tree engine is available), we
        # iteratively decompose the prompt and execute via TreeExecutionEngine.
        # The existing flat-pipeline path runs unchanged for low-complexity
        # tasks or when SENTINEL_MODE=="offline".
        import os as _os
        _mode_is_online = _os.environ.get("SENTINEL_MODE", "offline") == "online"
        _complexity = task.get("complexity", "medium")
        if (
            _complexity != "low"
            and self._tree_engine is not None
            and _mode_is_online
        ):
            try:
                from core.task_segregator import TaskSegregator as _TS
                from core.online_model_discovery import (
                    OnlineModelDiscoveryEngine as _OMD,
                    pick_decomposition_model as _pdm,
                )

                _ollama_client = None
                _coding_agent = self._agent_registry.get("coding")
                if _coding_agent is not None:
                    _ollama_client = getattr(_coding_agent, "_ollama", None)

                # ── 6b: Resolve discovery engine ONCE per process_prompt() ───
                try:
                    from models.ollama_cloud_client import OllamaCloudClient as _OCCI
                    _cloud_for_discovery = _OCCI()
                except Exception:
                    _cloud_for_discovery = None

                _discovery = _OMD(
                    ollama_cloud_client=_cloud_for_discovery,
                    local_router=self._model_router,
                    tool_registry=self._tool_registry,
                )

                # ── Resolve supervisor model with two-layer scoring ───────────
                _sup_client = _ollama_client
                _sup_model  = (
                    self._model_router.select_reasoning_model()
                    if self._model_router else "mistral:7b"
                )
                try:
                    from models.ollama_cloud_client import OllamaCloudClient as _OCC
                    _cloud_sup  = _OCC()
                    _cloud_mdls = _cloud_sup.list_models()
                    if _cloud_mdls:
                        _best = _pick_best_supervisor_cloud_model(
                            _cloud_mdls, discovery_engine=_discovery
                        )
                        if _best:
                            _sup_client = _cloud_sup
                            _sup_model  = _best
                except Exception:
                    pass

                # ── 6c: Resolve decomposition model (≤250B cap) ──────────────
                _decomp_model = _sup_model  # safe fallback
                try:
                    _cloud_models_for_decomp = (
                        _cloud_for_discovery.list_models()
                        if _cloud_for_discovery else []
                    )
                    _decomp_candidate = _pdm(
                        cloud_models=_cloud_models_for_decomp,
                        domain="reasoning",
                        complexity=task.get("complexity", "medium"),
                        tool_registry=self._tool_registry,
                    )
                    if _decomp_candidate:
                        _decomp_model = _decomp_candidate
                except Exception:
                    pass

                console.print(
                    f"  [cyan]→[/cyan] Decomposition model (≤250B): "
                    f"[bold]{_decomp_model}[/bold]"
                )

                # ── 6d: Construct TaskSegregator with BOTH models ─────────────
                _segregator = _TS(
                    ollama_client=_sup_client,
                    supervisor_model=_sup_model,
                    decomposition_model=_decomp_model,
                )

                # ── 6e: Build only root + one layer (lazy) ────────────────────
                # Reuse the sub_tasks already computed by the flat segregation
                # block above (refine/classify already applied).  This avoids
                # calling segregate() a second time with the decomp model, which
                # may be a vision model incapable of JSON generation.
                _pre_seg = task.get("sub_tasks") or []
                console.print("  [cyan]→[/cyan] Building initial task layer…")
                _tree = _segregator.build_tree_lazy(
                    task.get("goal", ""),
                    pre_segregated=_pre_seg,
                )

                # Display tree structure in CLI before execution.
                # Re-use the shared tracker from the UI so the decomp tree
                # Live and the pipeline Live share one Console and one
                # render queue — eliminating the multiple-Console race.
                try:
                    _shared_tracker = getattr(self, "_shared_tracker", None)
                    if _shared_tracker is not None:
                        _shared_tracker.display_tree(_tree)
                        self._tree_engine._tracker = _shared_tracker
                except Exception:
                    pass

                # ── 6f: Inject segregator, agent_registry, discovery engine ───
                self._tree_engine._segregator       = _segregator
                self._tree_engine._agent_registry   = self._agent_registry
                self._tree_engine._discovery_engine = _discovery

                _session_ctx = {
                    "session_id":   task.get("session_id", ""),
                    "project_root": task.get("project_root", str(self.project_root)),
                }
                _tree_result = self._tree_engine.execute_tree(_tree, _session_ctx)
                elapsed_ms = (time.monotonic() - t0) * 1000
                return {
                    "status":      _tree_result.status,
                    "summary":     _tree_result.summary(),
                    "result":      _tree_result,
                    "tree":        _tree,
                    "elapsed_ms":  round(elapsed_ms, 2),
                    "exploration": (
                        self._exploration_report.to_dict()
                        if self._exploration_report else None
                    ),
                }
            except Exception as _exc:
                console.print(
                    f"  [yellow]⚠[/yellow] Tree execution failed ({_exc}); "
                    "falling back to flat pipeline."
                )

        plan     = self._task_planner.plan(task)
        pipeline = self._pipeline_gen.from_execution_plan(plan)

        # Stamp project_root, raw_prompt, and exploration onto every step.
        project_root_str = str(self.project_root)
        raw_prompt = task.get("raw_prompt", "")
        sub_tasks = task.get("sub_tasks", [])
        # BUG-1 fix: map sub_tasks → steps by domain/agent affinity instead
        # of relying on positional index, which silently clips when counts
        # differ.  A consumed-set prevents the same sub_task from being
        # stamped onto two steps.
        _used_st_indices: set = set()
        for i, step in enumerate(pipeline.steps):
            step.metadata["project_root"] = project_root_str
            step.metadata["raw_prompt"]   = raw_prompt
            if self._exploration_report is not None:
                step.metadata["synopsis"]      = self._exploration_report.synopsis
                step.metadata["stack"]         = self._exploration_report.stack
                step.metadata["entry_points"]  = self._exploration_report.entry_points
            # Online mode: stamp per-step selected_model from sub_task decomposition
            if sub_tasks:
                st = _match_subtask_for_step(step, sub_tasks, _used_st_indices)
                if st and "selected_model" in st:
                    step.metadata["selected_model"] = st["selected_model"]
                    try:
                        step.selected_model = st["selected_model"]
                    except AttributeError:
                        pass

        if self._pipeline_opt:
            try:
                pipeline, _ = self._pipeline_opt.optimize(pipeline)
            except Exception:
                pass
        result     = self._engine.run_pipeline(pipeline)
        elapsed_ms = (time.monotonic() - t0) * 1000
        self._record_metrics(pipeline, result)
        return {
            "status":      result.status,
            "summary":     result.summary(),
            "result":      result,
            "pipeline":    pipeline,
            "elapsed_ms":  round(elapsed_ms, 2),
            "exploration": (self._exploration_report.to_dict()
                            if self._exploration_report else None),
        }

    def _run_repo_exploration(self) -> None:
        """Run RepoExplorer and cache the result on self._exploration_report.

        Re-uses the explorer's hash-based on-disk cache so repeated calls
        within the same session are instant.  Never raises -- errors are
        printed as a warning and exploration is simply skipped.

        In online mode the synopsis model is upgraded to the best available
        cloud reasoning model so architectural summaries are higher quality.
        """
        try:
            from context.repo_explorer import RepoExplorer
            import os as _os

            ollama_client = None
            coding_agent = self._agent_registry.get("coding")
            if coding_agent is not None:
                ollama_client = getattr(coding_agent, "_ollama", None)

            synopsis_client = ollama_client
            synopsis_model  = (
                self._model_router.select_reasoning_model()
                if self._model_router else "mistral:7b"
            )

            # In online mode, use the best cloud model for richer repo synopsis
            if _os.environ.get("SENTINEL_MODE", "offline") == "online":
                try:
                    from models.ollama_cloud_client import OllamaCloudClient
                    _cloud = OllamaCloudClient()
                    _mdls  = _cloud.list_models()
                    if _mdls:
                        _best = _pick_best_supervisor_cloud_model(_mdls)
                        if _best:
                            synopsis_client = _cloud
                            synopsis_model  = _best
                except Exception:
                    pass  # fall through to local model
            explorer = RepoExplorer(
                project_root=str(self.project_root),
                ollama_client=synopsis_client,
                synopsis_model=synopsis_model,
                use_cache=True,
            )
            report = explorer.explore()
            self._exploration_report = report
            cache_note = " [dim](cached)[/dim]" if report.from_cache else ""
            stack_str = ", ".join(
                f"{k}: {v}" for k, v in list(report.stack.items())[:4]
            ) or "stack unknown"
            console.print(
                f"  [green]✔[/green] Repo explored{cache_note}  "
                f"[dim]{report.total_files} files | "
                f"{', '.join(report.languages[:3]) or 'unknown'} | "
                f"{stack_str}[/dim]"
            )
        except Exception as exc:
            console.print(f"  [yellow]⚠[/yellow] Repo exploration skipped: {exc}")
            self._exploration_report = None

    def _record_metrics(self, pipeline: Any, result: Any) -> None:
        if not self._perf_tracker:
            return
        try:
            category = (pipeline.classification.get("category", "coding")
                        if hasattr(pipeline, "classification")
                        and isinstance(pipeline.classification, dict) else "coding")
            # TreeRunResult has node_results; flat PipelineRunResult has step_results
            if not hasattr(result, "step_results"):
                return
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

    def _expand_attachments(self, prompt: str) -> str:
        """Replace @file:, @pdf:, @url: tokens with their content inline."""
        import re
        project_root = str(self.project_root)

        # @file:PATH
        def _replace_file(m: "re.Match[str]") -> str:
            path = m.group(1).strip()
            try:
                from context.file_loader import load as file_load
                items = file_load(path, project_root=project_root)
                if not items:
                    return m.group(0)
                parts = []
                for it in items:
                    lang = it.get("language", "")
                    content = it.get("content", "")
                    rpath = it.get("relative_path") or it.get("path", path)
                    fence = f"```{lang}" if lang else "```"
                    parts.append(f"[File: {rpath}]\n{fence}\n{content}\n```")
                return "\n\n".join(parts)
            except Exception as exc:
                return f"[Could not load @file:{path} — {exc}]"

        prompt = re.sub(r"@file:([^\s]+)", _replace_file, prompt)

        # @pdf:PATH
        def _replace_pdf(m: "re.Match[str]") -> str:
            path = m.group(1).strip()
            try:
                from context.pdf_parser import load as pdf_load
                it = pdf_load(path, project_root=project_root)
                content = it.get("content", "")
                rpath = it.get("path", path)
                return f"[PDF: {rpath}]\n{content}"
            except Exception as exc:
                return f"[Could not load @pdf:{path} — {exc}]"

        prompt = re.sub(r"@pdf:([^\s]+)", _replace_pdf, prompt)

        # @url:URL (basic — fetch page text)
        def _replace_url(m: "re.Match[str]") -> str:
            url = m.group(1).strip()
            try:
                from context.url_fetcher import fetch as url_fetch
                it = url_fetch(url)
                content = it.get("content", "") or it.get("text", "")
                return f"[URL: {url}]\n{content[:8000]}"
            except Exception as exc:
                return f"[Could not fetch @url:{url} — {exc}]"

        prompt = re.sub(r"@url:([^\s]+)", _replace_url, prompt)

        return prompt

    def _looks_like_shell_command(self, prompt: str) -> bool:
        """Heuristically decide whether *prompt* is a shell command."""
        text = (prompt or "").strip()
        if not text:
            return False

        # Slash commands are handled by the UI parser, not here.
        if text.startswith("/"):
            return False

        # Multi-line natural language prompts should go to the pipeline.
        if "\n" in text and not any(op in text for op in ("&&", "||", "|", ";")):
            return False

        # Strong shell-command signals.
        if re.search(r"[|><]|&&|\|\|", text):
            return True

        try:
            parts = shlex.split(text, posix=(os.name != "nt"))
        except ValueError:
            parts = text.split()

        if not parts:
            return False

        first = parts[0].strip('"\'').lower()
        first_base = Path(first).name.lower()

        shell_builtins = {
            "cd", "dir", "ls", "pwd", "echo", "type", "cat", "cls", "clear",
            "mkdir", "rmdir", "copy", "move", "ren", "del", "set", "export",
            "where", "which",
        }
        if first_base in shell_builtins:
            return True

        if first_base.endswith((".ps1", ".bat", ".cmd", ".exe", ".py", ".sh")):
            return True

        if shutil.which(first_base) or shutil.which(first_base + ".cmd"):
            return True

        return False

    def _parse_prefixed_command(self, prompt: str) -> Dict[str, str]:
        """Parse explicit @shell / @open prefixes from user prompt."""
        text = (prompt or "").strip()
        if not text:
            return {"mode": "", "payload": ""}
        m = re.match(r"^@(shell|open)\s+(.+)$", text, re.IGNORECASE)
        if not m:
            return {"mode": "", "payload": ""}
        return {"mode": m.group(1).lower(), "payload": m.group(2).strip()}

    def _candidate_executable_names(self, app_name: str) -> List[str]:
        """Generate likely executable names for an app query."""
        raw = (app_name or "").strip()
        if not raw:
            return []
        slug = re.sub(r"[^a-z0-9]+", "", raw.lower())
        tokenized = re.sub(r"[^a-z0-9]+", " ", raw.lower()).strip()
        compact = tokenized.replace(" ", "")

        aliases = {
            "vscode": ["code.exe", "code-insiders.exe", "Code.exe"],
            "visualstudiocode": ["code.exe", "code-insiders.exe"],
            "chrome": ["chrome.exe"],
            "googlechrome": ["chrome.exe"],
            "edge": ["msedge.exe"],
            "microsoftedge": ["msedge.exe"],
            "notepadplusplus": ["notepad++.exe"],
            "terminal": ["wt.exe", "WindowsTerminal.exe"],
            "powershell": ["powershell.exe", "pwsh.exe"],
        }

        names: List[str] = []
        names.extend(aliases.get(slug, []))
        for base in {raw, tokenized, compact, slug}:
            if not base:
                continue
            b = base.strip().strip("\"'")
            if not b:
                continue
            if b.lower().endswith(".exe"):
                names.append(b)
            else:
                names.append(f"{b}.exe")
                names.append(b)

        # De-duplicate while preserving order
        seen: Set[str] = set()
        out: List[str] = []
        for n in names:
            k = n.lower()
            if k in seen:
                continue
            seen.add(k)
            out.append(n)
        return out

    def _windows_search_roots(self) -> List[Path]:
        """Return likely roots to search for installed Windows applications."""
        roots: List[Path] = []
        env_candidates = [
            os.environ.get("ProgramFiles"),
            os.environ.get("ProgramFiles(x86)"),
            os.environ.get("LOCALAPPDATA"),
            os.environ.get("APPDATA"),
        ]
        for p in env_candidates:
            if p:
                pp = Path(p)
                if pp.exists():
                    roots.append(pp)

        # Include project root as a quick check for local binaries.
        if self.project_root.exists():
            roots.append(self.project_root)

        # Add mounted drive roots as deep fallback.
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            drive = Path(f"{letter}:\\")
            if drive.exists():
                roots.append(drive)

        # De-duplicate
        deduped: List[Path] = []
        seen: Set[str] = set()
        for r in roots:
            key = str(r).lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(r)
        return deduped

    def _find_executable_in_roots(
        self,
        names: Iterable[str],
        roots: Iterable[Path],
        app_query: str,
        time_budget_s: float = 25.0,
    ) -> Optional[Path]:
        """Search filesystem roots for a matching executable within a time budget."""
        name_set = {n.lower() for n in names}
        query_tokens = [t for t in re.sub(r"[^a-z0-9]+", " ", app_query.lower()).split() if t]
        t_deadline = time.monotonic() + max(2.0, time_budget_s)

        skip_dirs = {
            "$recycle.bin", "system volume information", "windows", "winsxs",
            "recovery", "programdata\\microsoft\\windows\\wer",
            "node_modules", ".git", "__pycache__",
        }

        for root in roots:
            if time.monotonic() > t_deadline:
                break
            if not root.exists():
                continue

            for dirpath, dirnames, filenames in os.walk(root, topdown=True):
                if time.monotonic() > t_deadline:
                    break

                dlow = dirpath.lower()
                dirnames[:] = [
                    d for d in dirnames
                    if d.lower() not in skip_dirs
                    and not (Path(dlow) / d).as_posix().lower().endswith("/windows")
                ]

                # Exact filename match first.
                for fn in filenames:
                    fl = fn.lower()
                    if fl in name_set:
                        return Path(dirpath) / fn

                # Fuzzy match by app tokens in executable stem.
                for fn in filenames:
                    fl = fn.lower()
                    if not fl.endswith(".exe"):
                        continue
                    stem = Path(fl).stem
                    if query_tokens and all(tok in stem for tok in query_tokens):
                        return Path(dirpath) / fn
        return None

    def _extract_exe_names_from_web_results(self, output: Any) -> List[str]:
        """Extract candidate '*.exe' names from web_search output entries."""
        if not isinstance(output, list):
            return []
        exe_re = re.compile(r"\b([a-zA-Z0-9_.+\-]+\.exe)\b")
        names: List[str] = []
        for row in output:
            if not isinstance(row, dict):
                continue
            blob = " ".join(str(row.get(k, "")) for k in ("title", "snippet", "url"))
            for m in exe_re.findall(blob):
                names.append(m)
        seen: Set[str] = set()
        deduped: List[str] = []
        for n in names:
            k = n.lower()
            if k in seen:
                continue
            seen.add(k)
            deduped.append(n)
        return deduped

    def _resolve_installed_application(self, app_name: str) -> Dict[str, Any]:
        """Resolve an application query to an installed executable path.

        Strategy:
          1) Local search by candidate executable names.
          2) Web search for likely executable filename, then local search again.
          3) Return not-found.
        """
        candidates = self._candidate_executable_names(app_name)
        roots = self._windows_search_roots() if os.name == "nt" else [self.project_root]

        # Fast PATH check first
        for c in candidates:
            hit = shutil.which(c) or (shutil.which(c + ".cmd") if not c.lower().endswith(".cmd") else None)
            if hit:
                return {"success": True, "path": str(Path(hit).resolve()), "source": "path"}

        # Full device/local roots search
        found = self._find_executable_in_roots(candidates, roots, app_name, time_budget_s=30.0)
        if found:
            return {"success": True, "path": str(found.resolve()), "source": "filesystem"}

        # Web-assist: find likely filename, then search again
        ws = self._tool_registry.invoke("web_search", {
            "query": f"Windows executable filename for {app_name}",
            "max_results": 5,
        })
        web_candidates = self._extract_exe_names_from_web_results(ws.get("output"))
        if web_candidates:
            found2 = self._find_executable_in_roots(web_candidates, roots, app_name, time_budget_s=30.0)
            if found2:
                return {
                    "success": True,
                    "path": str(found2.resolve()),
                    "source": "web+filesystem",
                    "web_candidates": web_candidates,
                }

        return {
            "success": False,
            "error": f"Application not found: {app_name}",
            "web_candidates": web_candidates,
        }

    def _launch_executable(self, exe_path: str) -> Dict[str, Any]:
        """Launch executable path and return structured result."""
        p = Path(exe_path)
        if not p.exists():
            return {"success": False, "error": f"Executable path does not exist: {exe_path}"}
        try:
            if os.name == "nt":
                os.startfile(str(p))  # type: ignore[attr-defined]
                return {"success": True, "output": f"Opened: {p}"}
            proc = subprocess.Popen([str(p)])
            return {"success": True, "output": f"Opened: {p}", "pid": proc.pid}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def _run_shell_realtime(self, command: str, timeout: int = 120, watch_mode: bool = False) -> Dict[str, Any]:
        """Run shell command and stream output live to the terminal.

        This is used by explicit ``@shell`` prompts so users see native,
        real-time terminal output before any fallback repair flow runs.
        """
        try:
            if watch_mode:
                proc = subprocess.Popen(
                    command,
                    shell=True,
                    cwd=str(self.project_root),
                    stdin=None,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            else:
                proc = subprocess.Popen(
                    command,
                    shell=True,
                    cwd=str(self.project_root),
                    # Passthrough stdio for true terminal behavior (interactive prompts,
                    # colors, progress bars, and native command output formatting).
                    stdin=None,
                    stdout=None,
                    stderr=None,
                )
        except Exception as exc:
            return {
                "success": False,
                "returncode": -1,
                "stdout": "",
                "stderr": str(exc),
                "error": f"Failed to start command: {exc}",
            }

        try:
            if watch_mode:
                try:
                    stdout_data, stderr_data = proc.communicate(timeout=max(1, timeout))
                    stderr_text = (stderr_data or b"").decode("utf-8", errors="replace") if isinstance(stderr_data, (bytes, bytearray)) else str(stderr_data or "")
                    stdout_text = (stdout_data or b"").decode("utf-8", errors="replace") if isinstance(stdout_data, (bytes, bytearray)) else str(stdout_data or "")
                    return {
                        "success": proc.returncode == 0,
                        "returncode": int(proc.returncode or 0),
                        "stdout": stdout_text,
                        "stderr": stderr_text,
                        "error": None if proc.returncode == 0 else f"Process exited with code {proc.returncode}.",
                        "timed_out": False,
                        "watch_mode": True,
                    }
                except subprocess.TimeoutExpired as exc:
                    proc.kill()
                    stdout_data, stderr_data = proc.communicate()
                    def _decode_chunk(chunk: Any) -> str:
                        if isinstance(chunk, (bytes, bytearray)):
                            return chunk.decode("utf-8", errors="replace")
                        return str(chunk or "")

                    stdout_text = _decode_chunk(getattr(exc, "output", None))
                    stderr_text = _decode_chunk(getattr(exc, "stderr", None))
                    if stdout_data:
                        stdout_text = stdout_text or _decode_chunk(stdout_data)
                    if stderr_data:
                        stderr_text = stderr_text or _decode_chunk(stderr_data)
                    return {
                        "success": True,
                        "returncode": 0,
                        "stdout": stdout_text,
                        "stderr": stderr_text,
                        "error": None,
                        "timed_out": True,
                        "watch_mode": True,
                    }
            proc.wait(timeout=max(1, timeout))
        except subprocess.TimeoutExpired:
            proc.kill()
            return {
                "success": False,
                "returncode": -1,
                "stdout": "",
                "stderr": f"Command timed out after {timeout}s.",
                "error": f"Command timed out after {timeout}s.",
            }
        except Exception as exc:
            try:
                proc.kill()
            except Exception:
                pass
            return {
                "success": False,
                "returncode": -1,
                "stdout": "",
                "stderr": str(exc),
                "error": str(exc),
            }

        rc = int(proc.returncode or 0)
        return {
            "success": rc == 0,
            "returncode": rc,
            "stdout": "",
            "stderr": "",
            "error": None if rc == 0 else f"Process exited with code {rc}.",
        }

    def _run_shell_command_with_repair(self, command: str) -> Dict[str, Any]:
        """Run shell command directly, then auto-repair once on failure."""
        params = {
            "command": command,
            "cwd": str(self.project_root),
            "timeout": 120,
            "shell": True,
        }
        first = self._tool_registry.invoke("run_shell", params)
        if first.get("success"):
            return {"is_shell": True, "success": True, "result": first, "attempted": [command]}

        output = first.get("output") if isinstance(first.get("output"), dict) else {}
        repaired = self._repair_shell_command(
            command,
            error=str(first.get("error") or ""),
            stderr=str(output.get("stderr") if isinstance(output, dict) else ""),
        )
        if not repaired:
            return {"is_shell": True, "success": False, "result": first, "attempted": [command]}

        second_params = {
            "command": repaired,
            "cwd": str(self.project_root),
            "timeout": 120,
            "shell": True,
        }
        second = self._tool_registry.invoke("run_shell", second_params)
        return {
            "is_shell": True,
            "success": bool(second.get("success")),
            "result": second,
            "first_error": first.get("error"),
            "attempted": [command, repaired],
        }

    def _repair_shell_command(self, command: str, error: str = "", stderr: str = "") -> str:
        """Return a best-effort repaired shell command for failed invocations."""
        cmd = (command or "").strip()
        err_blob = f"{error}\n{stderr}".lower()

        # Common React bootstrap issue: npm package names must be lowercase.
        m = re.match(r"^(?:npx|npm\s+create)\s+create-react-app\s+(.+)$", cmd, re.IGNORECASE)
        if m:
            raw_name = m.group(1).strip().strip('"')
            lowered = raw_name.lower()
            lowered = re.sub(r"[^a-z0-9-_]", "-", lowered)
            lowered = re.sub(r"-+", "-", lowered).strip("-")
            if lowered and lowered != raw_name:
                return cmd.replace(raw_name, lowered, 1)

        # Fast heuristic repairs first.
        if "no module named pytest" in err_blob and "pytest" in cmd and "-m pytest" not in cmd:
            if cmd.startswith("pytest"):
                return cmd.replace("pytest", "python -m pytest", 1)
            return "python -m pytest"

        if "not recognized" in err_blob and cmd.startswith("npm "):
            return "npm.cmd " + cmd[4:]

        # LLM fallback: ask for a corrected Windows command.
        try:
            coding_agent = self._agent_registry.get("coding") if isinstance(self._agent_registry, dict) else None
            client = getattr(coding_agent, "_ollama", None) if coding_agent is not None else None
            if client is None:
                return ""

            model = self._model_router.select_reasoning_model() if self._model_router else ""
            if not model:
                return ""

            fix_prompt = (
                "You fix shell commands for Windows PowerShell. "
                "Return ONLY a corrected single-line command, no markdown, no explanation.\n\n"
                f"Original command:\n{cmd}\n\n"
                f"Error:\n{error}\n\n"
                f"stderr:\n{stderr}\n"
            )
            response = client.generate(
                model=model,
                prompt=fix_prompt,
                timeout=120,
                options={"temperature": 0.1, "num_predict": 120},
            )
            fixed = (response.get("response", "") or "").strip()
            if fixed.startswith("```"):
                lines = [ln for ln in fixed.splitlines() if not ln.strip().startswith("```")]
                fixed = "\n".join(lines).strip()
            fixed = fixed.splitlines()[0].strip() if fixed else ""
            if fixed and fixed.lower() != cmd.lower():
                return fixed
        except Exception:
            return ""

        return ""

    def _translate_shell_intent(self, intent: str) -> str:
        """Translate natural-language shell intent into a concrete command.

        Returns an empty string when translation is unavailable.
        """
        text = (intent or "").strip()
        if not text:
            return ""

        try:
            coding_agent = self._agent_registry.get("coding") if isinstance(self._agent_registry, dict) else None
            client = getattr(coding_agent, "_ollama", None) if coding_agent is not None else None
            if client is None:
                return ""

            model = self._model_router.select_reasoning_model() if self._model_router else ""
            if not model:
                return ""

            prompt = (
                "Convert the following user intent into ONE valid Windows PowerShell command. "
                "Return ONLY the command on a single line. No markdown, no explanation, no bullets.\n\n"
                f"Intent: {text}\n"
            )
            response = client.generate(
                model=model,
                prompt=prompt,
                timeout=120,
                options={"temperature": 0.1, "num_predict": 120},
            )
            cmd = (response.get("response", "") or "").strip()
            if cmd.startswith("```"):
                lines = [ln for ln in cmd.splitlines() if not ln.strip().startswith("```")]
                cmd = "\n".join(lines).strip()
            cmd = cmd.splitlines()[0].strip() if cmd else ""
            if not cmd:
                return ""
            return cmd
        except Exception:
            return ""

    def _try_run_user_shell_command(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Run prompt as shell command when it looks like one.

        Returns:
            None if prompt is not a shell command, otherwise a dict with
            execution details including success/failure and attempted commands.
        """
        if not self._looks_like_shell_command(prompt):
            return None
        return self._run_shell_command_with_repair(prompt)

    # ------------------------------------------------------------------
    # Post-pipeline helpers
    # ------------------------------------------------------------------

    def _verify_changed_files(
        self, events: List[Any], project_root: str
    ) -> Dict[str, str]:
        """Recursively search the project for every file the pipeline touched.

        For each FileChangeEvent the engine recorded we first check whether the
        absolute_path still exists on disk.  If it doesn't we fall back to a
        recursive walk of *project_root* with fuzzy filename matching (same
        strategy used by FileChangeMap.resolve).

        Returns:
            mapping of logical_path → resolved_absolute_path (or "" when the
            file truly cannot be located).
        """
        import fnmatch

        resolved: Dict[str, str] = {}
        if not events:
            return resolved

        # Build a flat list of all files in the project tree (excluding noise).
        all_disk_files: List[str] = []
        root = Path(project_root)
        if root.is_dir():
            for dirpath, dirnames, filenames in os.walk(root):
                # Skip heavy/irrelevant directories
                dirnames[:] = [
                    d for d in dirnames
                    if d not in {"node_modules", ".git", "__pycache__", ".next",
                                 "dist", "build", ".venv", "venv", ".cache"}
                ]
                for fname in filenames:
                    all_disk_files.append(os.path.join(dirpath, fname))

        for ev in events:
            logical = ev.logical_path
            absolute = ev.absolute_path

            # Fast path: the recorded absolute path is still valid.
            if absolute and os.path.isfile(absolute):
                resolved[logical] = absolute
                continue

            # Fuzzy search: match by basename (case-insensitive).
            target_name = os.path.basename(logical).lower()
            candidates = [
                p for p in all_disk_files
                if os.path.basename(p).lower() == target_name
            ]

            if len(candidates) == 1:
                resolved[logical] = candidates[0]
            elif len(candidates) > 1:
                # Prefer the one whose directory component best overlaps with
                # the logical path's parent directory name.
                logical_parent = os.path.basename(
                    os.path.dirname(logical)
                ).lower()
                ranked = sorted(
                    candidates,
                    key=lambda p: (
                        os.path.basename(os.path.dirname(p)).lower()
                        == logical_parent
                    ),
                    reverse=True,
                )
                resolved[logical] = ranked[0]
            else:
                # Partial-name fallback (e.g. "styles" matching "styles.css")
                stem = os.path.splitext(target_name)[0]
                partial = [
                    p for p in all_disk_files
                    if stem in os.path.basename(p).lower()
                ]
                resolved[logical] = partial[0] if partial else ""

        return resolved

    def _run_final_debug_check(
        self,
        verified_files: Dict[str, str],
        project_root: str,
        task_goal: str,
    ) -> str:
        """Ask the debugging model to verify cross-file references in changed files.

        Reads all verified HTML/JSX/TSX files and checks that every <script>,
        <link>, and import reference points to a file that actually exists.
        Returns a human-readable report string.
        """
        import re as _re

        if not verified_files:
            return "No changed files to verify."

        # Collect file contents for the files we can actually find.
        file_snapshots: List[str] = []
        abs_paths_set: Set[str] = set()
        for logical, abs_path in verified_files.items():
            if not abs_path or not os.path.isfile(abs_path):
                continue
            abs_paths_set.add(abs_path)
            try:
                content = Path(abs_path).read_text(encoding="utf-8", errors="replace")
                # Truncate very large files
                if len(content) > 4000:
                    content = content[:4000] + "\n... (truncated)"
                file_snapshots.append(
                    f"=== {abs_path} ===\n{content}"
                )
            except Exception:
                pass

        if not file_snapshots:
            return "Changed files could not be read for verification."

        # Inline reference check (fast, no LLM needed for simple cases).
        broken_refs: List[str] = []
        project_path = Path(project_root)
        for logical, abs_path in verified_files.items():
            if not abs_path or not os.path.isfile(abs_path):
                continue
            ext = os.path.splitext(abs_path)[1].lower()
            if ext not in {".html", ".htm"}:
                continue
            try:
                html = Path(abs_path).read_text(encoding="utf-8", errors="replace")
                file_dir = Path(abs_path).parent
                # Find all src= and href= attribute values
                refs = _re.findall(r'(?:src|href)=["\']([^"\'#?]+)["\']', html)
                for ref in refs:
                    if ref.startswith(("http://", "https://", "//", "data:")):
                        continue
                    # Resolve relative to the HTML file's directory
                    candidate = (file_dir / ref).resolve()
                    if not candidate.exists():
                        # Also try resolving from project root
                        candidate2 = (project_path / ref.lstrip("/")).resolve()
                        if not candidate2.exists():
                            broken_refs.append(
                                f"  ✘  {os.path.basename(abs_path)}: '{ref}' → not found"
                            )
            except Exception:
                pass

        # Build the LLM-based check only when we have a client.
        client = None
        model = ""
        try:
            if self._model_router:
                model = self._model_router.select_debugging_model()
            debug_agent = self._agent_registry.get("debugging")
            if debug_agent:
                client = getattr(debug_agent, "_inference_client", None) \
                         or getattr(debug_agent, "_ollama", None)
        except Exception:
            pass

        llm_verdict = ""
        if client and model and file_snapshots:
            try:
                combined = "\n\n".join(file_snapshots[:6])  # cap at 6 files
                debug_prompt = (
                    "You are a senior web developer doing a final code review.\n\n"
                    f"Task that was completed: {task_goal}\n\n"
                    "Below are the files that were created or modified. "
                    "Check that:\n"
                    "1. All HTML <script src=...> and <link href=...> references "
                    "match actual filenames in the project.\n"
                    "2. JavaScript and CSS are syntactically valid (no obvious errors).\n"
                    "3. Any dynamically-referenced function names (onclick=, "
                    "addEventListener) are defined in the included scripts.\n\n"
                    f"Files:\n{combined}\n\n"
                    "Respond with a concise verdict:\n"
                    "- PASS if everything looks correct.\n"
                    "- ISSUES FOUND followed by a short bullet list if you spot problems.\n"
                    "Keep the response under 200 words."
                )
                resp = client.generate(
                    model=model,
                    prompt=debug_prompt,
                    timeout=120,
                    options={"num_predict": 512, "temperature": 0.1},
                )
                llm_verdict = resp.get("response", "").strip()
            except Exception as e:
                llm_verdict = f"(Debugging model check skipped: {e})"

        parts: List[str] = []
        if broken_refs:
            parts.append("Reference check found issues:\n" + "\n".join(broken_refs))
        else:
            parts.append("Reference check: all src/href links resolved correctly.")
        if llm_verdict:
            parts.append(f"Debugging model verdict:\n{llm_verdict}")
        return "\n\n".join(parts)

    def _generate_conclusion(
        self,
        task_goal: str,
        verified_files: Dict[str, str],
        project_root: str,
        debug_report: str,
        pipeline_status: str,
    ) -> str:
        """Generate a Copilot-style conclusion message via the LLM.

        Falls back to a structured plain-text summary when no LLM is available.
        """
        changed_list = [
            abs_p for abs_p in verified_files.values() if abs_p
        ]
        changed_display = "\n".join(
            f"  • {p}" for p in changed_list[:20]
        ) or "  (no files recorded)"

        # Try LLM-generated conclusion first.
        client = None
        model = ""
        try:
            if self._model_router:
                model = (
                    self._model_router.select_reasoning_model()
                    or self._model_router.select_coding_model()
                )
            sup = self._agent_registry.get("supervisor") if self._agent_registry else None
            if sup:
                client = getattr(sup, "_inference_client", None) \
                         or getattr(sup, "_ollama", None)
        except Exception:
            pass

        if client and model:
            try:
                prompt_text = (
                    "You are an AI coding assistant. A pipeline has just finished. "
                    "Write a clear, friendly conclusion message (like GitHub Copilot "
                    "Workspace) in plain prose — no JSON, no bullet points inside "
                    "paragraphs.\n\n"
                    f"Task goal: {task_goal}\n"
                    f"Pipeline status: {pipeline_status}\n"
                    f"Files created or modified:\n{changed_display}\n"
                    f"Debug report:\n{debug_report}\n\n"
                    "Your conclusion must cover (in order, using short paragraphs):\n"
                    "1. What was done and which files were changed.\n"
                    "2. How to run or preview the project (give the exact commands).\n"
                    "3. Recommended next steps for the developer.\n\n"
                    "Keep the total length under 300 words. "
                    "Start directly with the summary — no greeting."
                )
                resp = client.generate(
                    model=model,
                    prompt=prompt_text,
                    timeout=120,
                    options={"num_predict": 768, "temperature": 0.4},
                )
                conclusion = resp.get("response", "").strip()
                if conclusion:
                    return conclusion
            except Exception:
                pass

        # Fallback: structured plain-text summary.
        lines = [
            f"Task completed ({pipeline_status}): {task_goal}",
            "",
            "Files changed:",
            changed_display,
            "",
            "To run the project, open the project directory and follow your "
            "framework's standard start command (e.g. `npm start`, `python app.py`, "
            "or open index.html in a browser).",
            "",
            "Next steps: review the generated files, run the test suite, and "
            "extend the implementation as needed.",
        ]
        if "ISSUES FOUND" in debug_report:
            lines += ["", "⚠ Note — the debug check found potential issues:", debug_report]
        return "\n".join(lines)

    def make_task_handler(self, session: Any) -> Callable[[str], None]:
        from cli.diff_viewer import DiffViewer
        diff_viewer = DiffViewer(console=console)

        def _handle(prompt: str) -> None:
            # ── Expand @file: / @pdf: / @url: attachment tokens ─────────
            raw_prompt = prompt
            prefixed = self._parse_prefixed_command(raw_prompt)
            if prefixed["mode"] not in ("shell", "open"):
                prompt = self._expand_attachments(prompt)
            else:
                prompt = raw_prompt

            session.add_turn("user", prompt)
            try:
                if prefixed["mode"] == "shell":
                    cmd = prefixed["payload"].strip()
                    if not cmd:
                        console.print("[red]@shell requires a command.[/red]")
                        session.add_turn("assistant", "@shell requires a command.")
                        return

                    review: Dict[str, Any] = {}
                    if self._engine is not None and hasattr(self._engine, "review_shell_command"):
                        try:
                            review = self._engine.review_shell_command(
                                "run_shell",
                                {
                                    "command": cmd,
                                    "cwd": str(self.project_root),
                                    "timeout": 120,
                                    "shell": True,
                                },
                                {
                                    "step_name": "@shell",
                                    "project_root": str(self.project_root),
                                },
                            ) or {}
                        except Exception:
                            review = {}

                    if str(review.get("decision", "")).strip().lower() == "skip":
                        msg = review.get("reason") or "Supervisor skipped the command."
                        console.print(Panel(
                            f"[yellow]{msg}[/yellow]",
                            title="[bold yellow]@shell Skipped[/bold yellow]",
                            border_style="yellow",
                            expand=True,
                        ))
                        session.pipeline_state = {
                            "status": "completed",
                            "summary": msg,
                            "elapsed_ms": 0,
                        }
                        session.add_turn("assistant", msg)
                        return

                    if str(review.get("display_label", "")).strip():
                        console.print(Panel(
                            f"[bold]{review.get('display_label')}[/bold]",
                            title="[bold cyan]@shell Review[/bold cyan]",
                            border_style="cyan",
                            expand=True,
                        ))

                    watch_mode = str(review.get("decision", "")).strip().lower() == "bounded_run"
                    if watch_mode:
                        timeout_value = review.get("timeout_seconds", 120)
                        try:
                            timeout_value = max(1, int(timeout_value))
                        except (TypeError, ValueError):
                            timeout_value = 120
                    else:
                        timeout_value = 120

                    # Warp-like UX: allow natural-language intent and translate
                    # to a concrete shell command before execution.
                    if not self._looks_like_shell_command(cmd):
                        translated = self._translate_shell_intent(cmd)
                        if not translated:
                            msg = "Could not translate @shell intent into a command."
                            console.print(Panel(
                                f"[red]{msg}[/red]",
                                title="[bold red]@shell Translation Failed[/bold red]",
                                border_style="red",
                                expand=True,
                            ))
                            session.pipeline_state = {
                                "status": "failed",
                                "summary": msg,
                                "elapsed_ms": 0,
                            }
                            session.add_turn("assistant", msg)
                            return

                        if self._engine is not None and getattr(self._engine, "_auto_approve_session", False):
                            cmd = translated
                        else:
                            console.print(Panel(
                                (
                                    f"[bold]Intent:[/bold] {cmd}\n"
                                    f"[bold]Proposed command:[/bold] {translated}\n\n"
                                    "Run this command? [Y/n/A]"
                                ),
                                title="[bold cyan]@shell Proposal[/bold cyan]",
                                border_style="cyan",
                                expand=True,
                            ))
                            sys.stdout.write("  Apply? [Y/n/A] › ")
                            sys.stdout.flush()
                            answer = (sys.stdin.readline() or "").strip().lower()
                            
                            # Handle auto-approve option
                            if answer in ("a", "A", "Accept All", "auto-approve") and self._engine is not None:
                                self._engine._auto_approve_session = True
                                answer = "y"  # Convert to yes to proceed
                            
                            if answer not in ("", "y", "yes"):
                                msg = "@shell command cancelled by user."
                                session.pipeline_state = {
                                    "status": "failed",
                                    "summary": msg,
                                    "elapsed_ms": 0,
                                }
                                session.add_turn("assistant", msg)
                                return
                            cmd = translated

                    # First run: show live terminal output exactly as command emits it.
                    realtime = self._run_shell_realtime(cmd, timeout=timeout_value, watch_mode=watch_mode)
                    if realtime.get("success"):
                        summary = (
                            review.get("reason")
                            if watch_mode and review.get("reason")
                            else "@shell command executed successfully."
                        )
                        session.pipeline_state = {
                            "status": "completed",
                            "summary": summary,
                            "elapsed_ms": 0,
                        }
                        session.add_turn("assistant", summary)
                        return

                    # Failure path: start repair logic and show run_shell output panel.
                    console.print(Panel(
                        "[yellow]Command failed. Attempting to auto-fix and re-run...[/yellow]",
                        title="[bold yellow]@shell Repair[/bold yellow]",
                        border_style="yellow",
                        expand=True,
                    ))
                    repaired = self._repair_shell_command(
                        cmd,
                        error=str(realtime.get("error") or ""),
                        stderr=str(realtime.get("stderr") or ""),
                    )
                    if not repaired:
                        summary = f"@shell execution failed: {realtime.get('error', 'Unknown error')}"
                        session.pipeline_state = {
                            "status": "failed",
                            "summary": summary,
                            "elapsed_ms": 0,
                        }
                        session.add_turn("assistant", summary)
                        return

                    tr = self._tool_registry.invoke("run_shell", {
                        "command": repaired,
                        "cwd": str(self.project_root),
                        "timeout": 120,
                        "shell": True,
                    })
                    success = bool(tr.get("success"))
                    status_style = "green" if success else "red"
                    output_blob = tr.get("output") if isinstance(tr.get("output"), dict) else {}
                    stdout = str(output_blob.get("stdout", ""))[:3000]
                    stderr = str(output_blob.get("stderr", ""))[:3000]
                    ret = output_blob.get("returncode", "?")
                    panel_text = (
                        f"[bold]Mode:[/bold] @shell repair execution\n"
                        f"[bold]Original command:[/bold] {cmd}\n"
                        f"[bold]Repaired command:[/bold] {repaired}\n\n"
                        f"[bold]Return code:[/bold] {ret}\n"
                        f"[bold]stdout:[/bold]\n{stdout or '[none]'}\n\n"
                        f"[bold]stderr:[/bold]\n{stderr or '[none]'}"
                    )
                    console.print(Panel(
                        f"[{status_style}]{panel_text}[/{status_style}]",
                        title="[bold]run_shell[/bold] output",
                        border_style=status_style,
                        expand=True,
                    ))
                    summary = (
                        "@shell repaired command executed successfully."
                        if success else
                        f"@shell repair failed: {tr.get('error', 'Unknown error')}"
                    )
                    session.pipeline_state = {
                        "status": "completed" if success else "failed",
                        "summary": summary,
                        "elapsed_ms": 0,
                    }
                    session.add_turn("assistant", summary)
                    return

                if prefixed["mode"] == "open":
                    app_name = prefixed["payload"].strip()
                    if not app_name:
                        console.print("[red]@Open requires an application name.[/red]")
                        session.add_turn("assistant", "@Open requires an application name.")
                        return

                    console.print(f"[dim]Resolving application:[/dim] {app_name}")
                    resolved = self._resolve_installed_application(app_name)
                    if not resolved.get("success"):
                        web_candidates = resolved.get("web_candidates", [])
                        hint = (
                            f"\n[dim]Web filename hints tried: {', '.join(web_candidates[:5])}[/dim]"
                            if web_candidates else ""
                        )
                        msg = f"Application not found: {app_name}{hint}"
                        console.print(Panel(
                            f"[red]{msg}[/red]",
                            title="[bold red]@Open Failed[/bold red]",
                            border_style="red",
                            expand=True,
                        ))
                        session.pipeline_state = {
                            "status": "failed",
                            "summary": msg,
                            "elapsed_ms": 0,
                        }
                        session.add_turn("assistant", msg)
                        return

                    exe_path = str(resolved.get("path", ""))
                    launched = self._launch_executable(exe_path)
                    if launched.get("success"):
                        src = resolved.get("source", "filesystem")
                        msg = f"Opened application: {exe_path} (source: {src})"
                        console.print(Panel(
                            f"[green]{msg}[/green]",
                            title="[bold green]@Open Success[/bold green]",
                            border_style="green",
                            expand=True,
                        ))
                        session.pipeline_state = {
                            "status": "completed",
                            "summary": msg,
                            "elapsed_ms": 0,
                        }
                        session.add_turn("assistant", msg)
                        return

                    msg = f"Found executable but failed to open: {launched.get('error', 'Unknown error')}"
                    console.print(Panel(
                        f"[red]{msg}[/red]",
                        title="[bold red]@Open Failed[/bold red]",
                        border_style="red",
                        expand=True,
                    ))
                    session.pipeline_state = {
                        "status": "failed",
                        "summary": msg,
                        "elapsed_ms": 0,
                    }
                    session.add_turn("assistant", msg)
                    return

                out = self.process_prompt(prompt, session_id=session.session_id)
                result = out["result"]

                # ── Per-step output ──────────────────────────────────────────
                # Only show actionable tool results (diffs and shell output).
                # Agent reasoning/analysis messages are intentionally suppressed
                # here; a single Copilot-style conclusion is generated below.
                # TreeRunResult has node_results, not step_results — skip loop.
                _step_results = getattr(result, "step_results", None) or []
                for sr in _step_results:
                    if sr.status == "skipped":
                        continue

                    # Tool results — file writes → diff, run_shell/run_tests → output
                    for tr in sr.tool_results:
                        tool = tr.get("tool_name", "")
                        success = tr.get("success", False)
                        meta = tr.get("metadata", {})

                        if tool == "write_file" and success:
                            path = meta.get("path", "")
                            content_written = tr.get("output", "")
                            try:
                                from pathlib import Path as _Path
                                existing = _Path(path).read_text(encoding="utf-8", errors="replace")
                                new_content = ""
                                for action in sr.actions:
                                    if (action.action_type == "tool_call"
                                            and action.payload.get("tool") == "write_file"
                                            and action.payload.get("params", {}).get("path") == path):
                                        new_content = action.payload["params"].get("content", "")
                                        break
                                if new_content and existing != new_content:
                                    diff_viewer.render_comparison(
                                        existing, new_content,
                                        fromfile=f"a/{path}",
                                        tofile=f"b/{path}",
                                        title=f"Changes · {_Path(path).name}",
                                    )
                                    session.metadata["last_diff"] = {
                                        "old": existing, "new": new_content,
                                        "fromfile": f"a/{path}", "tofile": f"b/{path}",
                                        "title": f"Changes · {_Path(path).name}",
                                    }
                                elif new_content:
                                    console.print(
                                        f"  [green]✔[/green] Written (no diff): [dim]{path}[/dim]"
                                    )
                            except FileNotFoundError:
                                new_content = ""
                                for action in sr.actions:
                                    if (action.action_type == "tool_call"
                                            and action.payload.get("tool") == "write_file"):
                                        new_content = action.payload.get("params", {}).get("content", "")
                                        break
                                diff_viewer.render_comparison(
                                    "", new_content,
                                    fromfile="/dev/null",
                                    tofile=path,
                                    title=f"New file · {path}",
                                )
                            except Exception:
                                console.print(f"  [green]✔[/green] Written: [dim]{path}[/dim]")

                        elif tool in ("run_tests", "run_shell") and tr.get("output"):
                            output_text = str(tr["output"])[:3000]
                            status_style = "green" if success else "red"
                            console.print(Panel(
                                f"[{status_style}]{output_text}[/{status_style}]",
                                title=f"[bold]{tool}[/bold] output · {sr.step_name}",
                                border_style=status_style,
                                expand=True,
                            ))

                        elif not success and tr.get("error"):
                            console.print(
                                f"  [red]✘[/red] {tool} failed: [dim]{tr['error']}[/dim]"
                            )

                # ── Summary panel ────────────────────────────────────────────
                border = "green" if out["status"] == "completed" else "yellow"
                console.print(Panel(
                    out["summary"],
                    title="[bold cyan]Pipeline Complete[/bold cyan]",
                    border_style=border,
                ))

                # ── Post-pipeline: locate changed files ───────────────────────
                task_goal = ""
                try:
                    task_goal = out.get("pipeline", [{}])[0].get("goal", "") \
                                or session.metadata.get("last_goal", prompt)
                except Exception:
                    task_goal = prompt

                fcm_events: List[Any] = []
                try:
                    if self._engine is not None and hasattr(self._engine, "file_change_map"):
                        fcm_events = self._engine.file_change_map.all_events()
                except Exception:
                    pass

                verified_files: Dict[str, str] = {}
                if fcm_events:
                    console.print(
                        f"  [dim]→ Locating {len(fcm_events)} changed file(s)…[/dim]"
                    )
                    try:
                        verified_files = self._verify_changed_files(
                            fcm_events, str(self.project_root)
                        )
                        missing = [lp for lp, ap in verified_files.items() if not ap]
                        if missing:
                            console.print(
                                f"  [yellow]⚠  Could not locate: "
                                f"{', '.join(os.path.basename(m) for m in missing[:5])}[/yellow]"
                            )
                    except Exception:
                        pass

                # ── Post-pipeline: debugging model verification ───────────────
                debug_report = ""
                if verified_files:
                    console.print("  [dim]→ Running final code check…[/dim]")
                    try:
                        debug_report = self._run_final_debug_check(
                            verified_files, str(self.project_root), task_goal
                        )
                        if "ISSUES FOUND" in debug_report:
                            console.print(Panel(
                                debug_report,
                                title="[bold yellow]⚠ Debug Check — Issues Found[/bold yellow]",
                                border_style="yellow",
                                expand=True,
                            ))
                        else:
                            console.print(
                                f"  [green]✔[/green] Debug check: "
                                f"[dim]{debug_report.splitlines()[0]}[/dim]"
                            )
                    except Exception:
                        debug_report = ""

                # ── Post-pipeline: Copilot-style conclusion ───────────────────
                console.print("  [dim]→ Generating conclusion…[/dim]")
                conclusion = ""
                try:
                    conclusion = self._generate_conclusion(
                        task_goal=task_goal,
                        verified_files=verified_files,
                        project_root=str(self.project_root),
                        debug_report=debug_report,
                        pipeline_status=out["status"],
                    )
                except Exception:
                    conclusion = out["summary"]

                console.print(Panel(
                    conclusion,
                    title="[bold green]✓ What was done / How to run / Next steps[/bold green]",
                    border_style="green",
                    expand=True,
                ))

                session.pipeline_state = {
                    "status": out["status"], "summary": conclusion,
                    "elapsed_ms": out["elapsed_ms"],
                }
                # Store last context for /context command
                try:
                    _sr_list = getattr(result, "step_results", None) or []
                    last_sr = _sr_list[-1] if _sr_list else None
                    if last_sr and last_sr.output:
                        session.metadata["last_context"] = last_sr.output.get("context") or {}
                except Exception:
                    pass
                session.add_turn("assistant", conclusion)
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
        border_style="cyan", expand=True,
    ))


# ── Entry point ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="sentinel",
        description="Sentinel — Local Autonomous Development Assistant",
    )
    parser.add_argument("--resume",   metavar="SESSION_ID", default=None)
    parser.add_argument("--project",  metavar="PATH",        default=None)
    # Hardware profile selection (renamed from --mode to free it for online/offline)
    parser.add_argument("--hw-mode",  dest="hw_mode",
                        choices=["minimal", "standard", "advanced"], default=None)
    # Keep --mode for backward compatibility (maps to hw_mode)
    parser.add_argument("--mode",     choices=["minimal", "standard", "advanced"], default=None)
    # Explicit online / offline flags (skip the interactive connectivity prompt)
    parser.add_argument("--online",   action="store_true",
                        help="Force online mode (skips connectivity prompt).")
    parser.add_argument("--offline",  action="store_true",
                        help="Force offline mode (skips connectivity prompt).")
    parser.add_argument("--no-bootstrap", action="store_true")
    # --refresh: re-probe Ollama Cloud model cache and exit.
    # Called by the OS task scheduler and by `sentinel refresh`.
    parser.add_argument(
        "--refresh", action="store_true",
        help="Refresh the Ollama Cloud model cache and exit.",
    )
    args = parser.parse_args()

    # ── --refresh: run probe and exit immediately, no UI ─────────────────────
    if args.refresh:
        import importlib.util as _ilu
        _probe_path = Path(__file__).parent / "scripts" / "cloud_model_probe.py"
        _spec = _ilu.spec_from_file_location("cloud_model_probe", _probe_path)
        _mod  = _ilu.module_from_spec(_spec)          # type: ignore[arg-type]
        _spec.loader.exec_module(_mod)                # type: ignore[union-attr]
        ok = _mod.run_probe(force=True, verbose=True)
        sys.exit(0 if ok else 1)

    # Resolve force_mode for online/offline from explicit flags
    if args.online:
        _force_io_mode = "online"
    elif args.offline:
        _force_io_mode = "offline"
    else:
        _force_io_mode = None

    # Hardware mode: --hw-mode takes priority over legacy --mode
    hw_mode_arg = args.hw_mode or args.mode

    _print_banner()

    runtime = SentinelRuntime(
        project_root=args.project or os.environ.get("SENTINEL_PROJECT_DIR") or os.getcwd(),
        force_mode=hw_mode_arg,
        force_io_mode=_force_io_mode,
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
    ui._runtime = runtime  # allow commands like /models to query runtime state
    # Share the UI's ProgressTracker with the runtime so the decomp tree
    # Live and pipeline Live always use the same Console and render queue.
    runtime._shared_tracker = ui.progress

    # Store hardware mode in session metadata for /session and /mode commands.
    if runtime._model_router:
        session.metadata["hardware_mode"] = runtime._model_router.get_hardware_profile()
    session.metadata["project_root"] = str(runtime.project_root)
    session.metadata["sentinel_mode"] = runtime._mode

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
