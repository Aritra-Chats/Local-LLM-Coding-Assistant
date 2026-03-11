from __future__ import annotations
"""bootstrap.py — Sentinel first-launch bootstrap routine.

Executes the full seven-step initialisation sequence defined in the
Sentinel blueprint:

    1. Detect system capabilities
    2. Install required Python dependencies
    3. Install Ollama if missing
    4. Pull appropriate language models
    5. Pull embedding models
    6. Create project workspace directories
    7. Build initial project index

Cross-platform: Windows · Linux · macOS
"""

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from config.hardware_profile import HardwareMode, HardwareProfile, HardwareProfiler
from system.hardware_detector import SystemCheck, SystemInfo

console = Console()

# ---------------------------------------------------------------------------
# Workspace layout
# ---------------------------------------------------------------------------

_SENTINEL_HOME = Path(os.environ.get("SENTINEL_HOME", Path.home() / ".sentinel"))

WORKSPACE_DIRS: List[Path] = [
    _SENTINEL_HOME,
    _SENTINEL_HOME / "sessions",
    _SENTINEL_HOME / "projects",
    _SENTINEL_HOME / "index",
    _SENTINEL_HOME / "memory",
    _SENTINEL_HOME / "logs",
    _SENTINEL_HOME / "models",
    _SENTINEL_HOME / "cache",
]

_CONFIG_FILE = _SENTINEL_HOME / "config.json"
_PROFILE_FILE = _SENTINEL_HOME / "hardware_profile.json"
_BOOTSTRAP_STAMP = _SENTINEL_HOME / ".bootstrapped"

# ---------------------------------------------------------------------------
# Required Python packages (beyond direct dependencies)
# ---------------------------------------------------------------------------

REQUIRED_PACKAGES: List[str] = [
    "rich",
    "prompt_toolkit",
    "psutil",
    "pdfplumber",
    "pypdf2",
    "requests",
]

# ---------------------------------------------------------------------------
# Ollama install commands per platform
# ---------------------------------------------------------------------------

_OLLAMA_INSTALL: Dict[str, List[str]] = {
    "Linux": ["sh", "-c", "curl -fsSL https://ollama.com/install.sh | sh"],
    "Darwin": ["sh", "-c", "curl -fsSL https://ollama.com/install.sh | sh"],
    "Windows": [],  # Windows uses a GUI installer; we guide the user instead
}


# ---------------------------------------------------------------------------
# Bootstrap class
# ---------------------------------------------------------------------------


class Bootstrap:
    """Sentinel first-launch bootstrap controller.

    Call run() to execute the full initialisation sequence. Subsequent
    launches detect the .bootstrapped stamp and skip to a fast-path check.

    Attributes:
        force: If True, re-run the full bootstrap even if already stamped.
        info: SystemInfo populated after step 1.
        profile: HardwareProfile populated after step 1.
    """

    def __init__(self, force: bool = False) -> None:
        """Initialise the Bootstrap controller.

        Args:
            force: Force a full re-bootstrap even if already completed.
        """
        self.force = force
        self.info: Optional[SystemInfo] = None
        self.profile: Optional[HardwareProfile] = None
        self._errors: List[str] = []

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> HardwareProfile:
        """Execute the full bootstrap sequence.

        Skips the full sequence on subsequent launches unless force=True.

        Returns:
            The resolved HardwareProfile for this machine.

        Raises:
            RuntimeError: If a critical bootstrap step fails.
        """
        if _BOOTSTRAP_STAMP.exists() and not self.force:
            return self._fast_path()

        console.print(
            Panel(
                "[bold cyan]Sentinel[/bold cyan] — Running first-launch bootstrap…",
                border_style="cyan",
            )
        )

        self._step1_detect_hardware()
        self._step2_install_python_deps()
        self._step3_install_ollama()
        self._step4_pull_language_models()
        self._step5_pull_embedding_models()
        self._step6_create_workspace()
        self._step7_build_project_index()

        self._write_stamp()
        self._print_summary()

        assert self.profile is not None
        return self.profile

    # ------------------------------------------------------------------
    # Fast path (already bootstrapped)
    # ------------------------------------------------------------------

    def _fast_path(self) -> HardwareProfile:
        """Return the cached profile on subsequent launches.

        Reads the persisted hardware_profile.json. Falls back to a live
        detection if the file is missing or corrupt.

        Returns:
            The cached or freshly detected HardwareProfile.
        """
        if _PROFILE_FILE.exists():
            try:
                data = json.loads(_PROFILE_FILE.read_text(encoding="utf-8"))
                mode = HardwareMode(data["mode"])
                profiler = HardwareProfiler()
                self.info = SystemCheck().run()
                return profiler._build_profile(mode, data.get("notes", ""))
            except (KeyError, ValueError, json.JSONDecodeError):
                pass

        # Fallback: re-detect
        self.info = SystemCheck().run()
        self.profile = HardwareProfiler().classify(self.info)
        return self.profile

    # ------------------------------------------------------------------
    # Step 1 — Hardware detection
    # ------------------------------------------------------------------

    def _step1_detect_hardware(self) -> None:
        """Step 1: Detect system hardware and build the hardware profile."""
        with self._spinner("Detecting system hardware…"):
            checker = SystemCheck()
            self.info = checker.run()
            profiler = HardwareProfiler()
            self.profile = profiler.classify(self.info)

        console.print(
            f"  [green]✔[/green] Hardware detected — "
            f"[bold]{self.profile.mode.value.upper()}[/bold] mode  "
            f"({self.info.ram_gb:.1f} GB RAM"
            + (f", {self.info.gpus[0].name}" if self.info.gpus else "")
            + ")"
        )

    # ------------------------------------------------------------------
    # Step 2 — Python dependencies
    # ------------------------------------------------------------------

    def _step2_install_python_deps(self) -> None:
        """Step 2: Ensure required Python packages are installed."""
        import importlib.util

        missing = [
            pkg for pkg in REQUIRED_PACKAGES
            if importlib.util.find_spec(pkg.replace("-", "_").split("[")[0]) is None
        ]

        if not missing:
            console.print("  [green]✔[/green] Python dependencies already satisfied.")
            return

        console.print(f"  [yellow]→[/yellow] Installing: {', '.join(missing)}")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--quiet"] + missing,
                check=True,
                timeout=300,
            )
            console.print("  [green]✔[/green] Python dependencies installed.")
        except subprocess.CalledProcessError as exc:
            msg = f"pip install failed: {exc}"
            self._errors.append(msg)
            console.print(f"  [red]✘[/red] {msg}")

    # ------------------------------------------------------------------
    # Step 3 — Ollama
    # ------------------------------------------------------------------

    def _step3_install_ollama(self) -> None:
        """Step 3: Install Ollama if not already present on PATH."""
        if shutil.which("ollama"):
            console.print("  [green]✔[/green] Ollama already installed.")
            return

        os_name = platform.system()
        install_cmd = _OLLAMA_INSTALL.get(os_name, [])

        if os_name == "Windows":
            console.print(
                "  [yellow]![/yellow] Ollama not found.\n"
                "  Please download and install from: [bold]https://ollama.com/download[/bold]\n"
                "  Then re-run Sentinel."
            )
            self._errors.append("Ollama not installed (Windows requires manual install).")
            return

        if not install_cmd:
            console.print(f"  [red]✘[/red] No auto-install method for OS: {os_name}")
            self._errors.append(f"No Ollama install method for {os_name}.")
            return

        console.print("  [yellow]→[/yellow] Installing Ollama…")
        try:
            subprocess.run(install_cmd, check=True, timeout=300)
            if shutil.which("ollama"):
                console.print("  [green]✔[/green] Ollama installed successfully.")
            else:
                raise RuntimeError("ollama binary not found after install script.")
        except (subprocess.CalledProcessError, RuntimeError) as exc:
            msg = f"Ollama install failed: {exc}"
            self._errors.append(msg)
            console.print(f"  [red]✘[/red] {msg}")

    # ------------------------------------------------------------------
    # Step 4 — Language models
    # ------------------------------------------------------------------

    def _step4_pull_language_models(self) -> None:
        """Step 4: Pull language models appropriate for the hardware profile."""
        assert self.profile is not None

        models_to_pull = [
            self.profile.recommended_model,
            self.profile.reasoning_model,
        ]
        # Deduplicate
        models_to_pull = list(dict.fromkeys(models_to_pull))
        self._pull_models(models_to_pull, label="language model")

    # ------------------------------------------------------------------
    # Step 5 — Embedding models
    # ------------------------------------------------------------------

    def _step5_pull_embedding_models(self) -> None:
        """Step 5: Pull the embedding model required for RAG search."""
        assert self.profile is not None
        self._pull_models([self.profile.embedding_model], label="embedding model")

    # ------------------------------------------------------------------
    # Step 6 — Workspace directories
    # ------------------------------------------------------------------

    def _step6_create_workspace(self) -> None:
        """Step 6: Create the .sentinel workspace directory structure."""
        assert self.profile is not None

        created: List[Path] = []
        for directory in WORKSPACE_DIRS:
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
                created.append(directory)

        # Write hardware profile cache
        profile_data = {
            "mode": self.profile.mode.value,
            "recommended_model": self.profile.recommended_model,
            "reasoning_model": self.profile.reasoning_model,
            "embedding_model": self.profile.embedding_model,
            "context_limit": self.profile.context_limit,
            "max_pipeline_concurrency": self.profile.max_pipeline_concurrency,
            "notes": self.profile.notes,
        }
        _PROFILE_FILE.write_text(json.dumps(profile_data, indent=2), encoding="utf-8")

        # Write default config if absent
        if not _CONFIG_FILE.exists():
            default_config = {
                "version": "0.1.0",
                "workspace_home": str(_SENTINEL_HOME),
                "profile": self.profile.mode.value,
                "telemetry": False,
            }
            _CONFIG_FILE.write_text(json.dumps(default_config, indent=2), encoding="utf-8")

        if created:
            console.print(
                f"  [green]✔[/green] Workspace created at "
                f"[bold]{_SENTINEL_HOME}[/bold] ({len(created)} directories)"
            )
        else:
            console.print(
                f"  [green]✔[/green] Workspace already exists at [bold]{_SENTINEL_HOME}[/bold]"
            )

    # ------------------------------------------------------------------
    # Step 7 — Project index
    # ------------------------------------------------------------------

    def _step7_build_project_index(self) -> None:
        """Step 7: Build the initial project index stub.

        A full index build requires an active project to be loaded.
        This step creates the index directory and writes a placeholder
        manifest so the code understanding engine can populate it later.
        """
        index_dir = _SENTINEL_HOME / "index"
        index_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "status": "empty",
            "projects": [],
            "last_indexed": None,
        }
        manifest_file = index_dir / "manifest.json"
        if not manifest_file.exists():
            manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        console.print(
            "  [green]✔[/green] Project index directory initialised. "
            "Load a project to begin indexing."
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pull_models(self, models: List[str], label: str) -> None:
        """Pull a list of Ollama models, skipping ones already present.

        Args:
            models: List of Ollama model tag strings to pull.
            label: Human-readable label for log output.
        """
        if not shutil.which("ollama"):
            console.print(
                f"  [yellow]![/yellow] Skipping {label} pull — Ollama not on PATH."
            )
            return

        already = self._list_ollama_models()

        for model in models:
            if model in already:
                console.print(f"  [green]✔[/green] {label} [bold]{model}[/bold] already present.")
                continue

            console.print(f"  [yellow]→[/yellow] Pulling {label}: [bold]{model}[/bold]")
            try:
                subprocess.run(
                    ["ollama", "pull", model],
                    check=True,
                    timeout=1800,  # 30 min — large models take time
                )
                console.print(f"  [green]✔[/green] {label} [bold]{model}[/bold] pulled.")
            except subprocess.CalledProcessError as exc:
                msg = f"Failed to pull {model}: {exc}"
                self._errors.append(msg)
                console.print(f"  [red]✘[/red] {msg}")

    @staticmethod
    def _list_ollama_models() -> List[str]:
        """Return the list of model tags already pulled in Ollama.

        Returns:
            List of model tag strings (e.g. ['codellama:7b', 'mistral:7b']).
        """
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            lines = result.stdout.strip().splitlines()
            models: List[str] = []
            for line in lines[1:]:  # skip header
                parts = line.split()
                if parts:
                    models.append(parts[0])
            return models
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return []

    def _write_stamp(self) -> None:
        """Write the bootstrap completion stamp file."""
        _SENTINEL_HOME.mkdir(parents=True, exist_ok=True)
        _BOOTSTRAP_STAMP.write_text("bootstrapped", encoding="utf-8")

    def _print_summary(self) -> None:
        """Print a post-bootstrap summary to the console."""
        assert self.profile is not None
        assert self.info is not None

        status = "[bold green]Bootstrap complete[/bold green]" if not self._errors else (
            f"[bold yellow]Bootstrap complete with {len(self._errors)} warning(s)[/bold yellow]"
        )

        table = Table(show_header=False, border_style="cyan", expand=False)
        table.add_column("", style="bold cyan", width=22)
        table.add_column("", style="white")
        table.add_row("Mode", self.profile.mode.value.upper())
        table.add_row("Primary model", self.profile.recommended_model)
        table.add_row("Reasoning model", self.profile.reasoning_model)
        table.add_row("Embedding model", self.profile.embedding_model)
        table.add_row("Context limit", f"{self.profile.context_limit:,} tokens")
        table.add_row("Workspace", str(_SENTINEL_HOME))

        if self._errors:
            table.add_row("[red]Warnings[/red]", "\n".join(self._errors))

        console.print(Panel(table, title=status, border_style="cyan"))

    @staticmethod
    def _spinner(message: str):
        """Return a Rich Progress context manager acting as a spinner.

        Args:
            message: Message displayed beside the spinner.

        Returns:
            Context manager that shows a spinner while active.
        """
        return Progress(
            SpinnerColumn(),
            TextColumn(f"[cyan]{message}"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def run_bootstrap(force: bool = False) -> HardwareProfile:
    """Run the Sentinel bootstrap sequence.

    Args:
        force: If True, re-run even if already bootstrapped.

    Returns:
        The resolved HardwareProfile.
    """
    return Bootstrap(force=force).run()
