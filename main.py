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
        self._exploration_report = None
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
                reasoning_model="mixtral:8x7b",
                notes="Bootstrap failed; using standard defaults.",
            )

    def process_prompt(self, prompt: str, session_id: str = "") -> Dict[str, Any]:
        t0 = time.monotonic()

        # ── Phase 0: Repository Exploration ────────────────────────────────────────────────
        # Explore the project before planning so every downstream agent
        # operates with accurate architectural knowledge.
        self._run_repo_exploration()

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

        plan     = self._task_planner.plan(task)
        pipeline = self._pipeline_gen.from_execution_plan(plan)

        # Stamp project_root, raw_prompt, and exploration onto every step.
        project_root_str = str(self.project_root)
        raw_prompt = task.get("raw_prompt", "")
        for step in pipeline.steps:
            step.metadata["project_root"] = project_root_str
            step.metadata["raw_prompt"]   = raw_prompt
            if self._exploration_report is not None:
                step.metadata["synopsis"]      = self._exploration_report.synopsis
                step.metadata["stack"]         = self._exploration_report.stack
                step.metadata["entry_points"]  = self._exploration_report.entry_points

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
        """
        try:
            from context.repo_explorer import RepoExplorer
            ollama_client = None
            coding_agent = self._agent_registry.get("coding")
            if coding_agent is not None:
                ollama_client = getattr(coding_agent, "_ollama", None)
            synopsis_model = (
                self._model_router.select_reasoning_model()
                if self._model_router else "mistral:7b"
            )
            explorer = RepoExplorer(
                project_root=str(self.project_root),
                ollama_client=ollama_client,
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

    def _run_shell_realtime(self, command: str, timeout: int = 120) -> Dict[str, Any]:
        """Run shell command and stream output live to the terminal.

        This is used by explicit ``@shell`` prompts so users see native,
        real-time terminal output before any fallback repair flow runs.
        """
        try:
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
                                expand=False,
                            ))
                            session.pipeline_state = {
                                "status": "failed",
                                "summary": msg,
                                "elapsed_ms": 0,
                            }
                            session.add_turn("assistant", msg)
                            return

                        console.print(Panel(
                            (
                                f"[bold]Intent:[/bold] {cmd}\n"
                                f"[bold]Proposed command:[/bold] {translated}\n\n"
                                "Run this command? [Y/n]"
                            ),
                            title="[bold cyan]@shell Proposal[/bold cyan]",
                            border_style="cyan",
                            expand=False,
                        ))
                        sys.stdout.write("  Apply? [Y/n] › ")
                        sys.stdout.flush()
                        answer = (sys.stdin.readline() or "").strip().lower()
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
                    realtime = self._run_shell_realtime(cmd, timeout=120)
                    if realtime.get("success"):
                        summary = "@shell command executed successfully."
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
                        expand=False,
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
                        expand=False,
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
                            expand=False,
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
                            expand=False,
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
                        expand=False,
                    ))
                    session.pipeline_state = {
                        "status": "failed",
                        "summary": msg,
                        "elapsed_ms": 0,
                    }
                    session.add_turn("assistant", msg)
                    return

                if self._looks_like_shell_command(prompt):
                    realtime = self._run_shell_realtime(prompt, timeout=120)
                    if realtime.get("success"):
                        summary = "Direct command executed successfully."
                        session.pipeline_state = {
                            "status": "completed",
                            "summary": summary,
                            "elapsed_ms": 0,
                        }
                        session.add_turn("assistant", summary)
                        return

                    console.print(Panel(
                        "[yellow]Command failed. Attempting to auto-fix and re-run...[/yellow]",
                        title="[bold yellow]Shell Repair[/bold yellow]",
                        border_style="yellow",
                        expand=False,
                    ))
                    repaired = self._repair_shell_command(
                        prompt,
                        error=str(realtime.get("error") or ""),
                        stderr=str(realtime.get("stderr") or ""),
                    )
                    if not repaired:
                        summary = f"Direct command execution failed: {realtime.get('error', 'Unknown error')}"
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
                        f"[bold]Mode:[/bold] Direct shell repair execution\n"
                        f"[bold]Original command:[/bold] {prompt}\n"
                        f"[bold]Repaired command:[/bold] {repaired}\n\n"
                        f"[bold]Return code:[/bold] {ret}\n"
                        f"[bold]stdout:[/bold]\n{stdout or '[none]'}\n\n"
                        f"[bold]stderr:[/bold]\n{stderr or '[none]'}"
                    )
                    console.print(Panel(
                        f"[{status_style}]{panel_text}[/{status_style}]",
                        title="[bold]run_shell[/bold] output",
                        border_style=status_style,
                        expand=False,
                    ))
                    summary = (
                        "Direct repaired command executed successfully."
                        if success else
                        f"Direct command repair failed: {tr.get('error', 'Unknown error')}"
                    )
                    session.pipeline_state = {
                        "status": "completed" if success else "failed",
                        "summary": summary,
                        "elapsed_ms": 0,
                    }
                    session.add_turn("assistant", summary)
                    return

                out = self.process_prompt(prompt, session_id=session.session_id)
                result = out["result"]

                # ── Per-step output ──────────────────────────────────────────
                for sr in result.step_results:
                    if sr.status == "skipped":
                        continue

                    # Agent messages (LLM reasoning/analysis text)
                    for action in sr.actions:
                        if action.action_type == "message":
                            text = action.payload.get("content", "")
                            if text and not text.startswith("[") :
                                console.print(Panel(
                                    text,
                                    title=f"[bold cyan]{sr.step_name}[/bold cyan]",
                                    border_style="cyan",
                                    expand=False,
                                ))

                    # Tool results — file writes → diff, others → output text
                    for tr in sr.tool_results:
                        tool = tr.get("tool_name", "")
                        success = tr.get("success", False)
                        meta = tr.get("metadata", {})

                        if tool == "write_file" and success:
                            path = meta.get("path", "")
                            content_written = tr.get("output", "")
                            # Try to render a diff against the file on disk
                            try:
                                from pathlib import Path as _Path
                                existing = _Path(path).read_text(encoding="utf-8", errors="replace")
                                # content_written is summary text, not the file content.
                                # The actual new content is in the action payload.
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
                                # New file — show it as pure addition
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
                                expand=False,
                            ))

                        elif tool == "search_code" and success and tr.get("output"):
                            results_text = str(tr["output"])[:2000]
                            console.print(Panel(
                                f"[dim]{results_text}[/dim]",
                                title=f"[bold]search_code[/bold] · {sr.step_name}",
                                border_style="dim",
                                expand=False,
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
                session.pipeline_state = {
                    "status": out["status"], "summary": out["summary"],
                    "elapsed_ms": out["elapsed_ms"],
                }
                # Store last context for /context command
                try:
                    last_sr = result.step_results[-1] if result.step_results else None
                    if last_sr and last_sr.output:
                        session.metadata["last_context"] = last_sr.output.get("context") or {}
                except Exception:
                    pass
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
    ui._runtime = runtime  # allow commands like /models to query runtime state

    # Store hardware mode in session metadata for /session and /mode commands.
    if runtime._model_router:
        session.metadata["hardware_mode"] = runtime._model_router.get_hardware_profile()
    session.metadata["project_root"] = str(runtime.project_root)

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
