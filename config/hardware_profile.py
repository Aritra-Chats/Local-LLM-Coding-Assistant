"""hardware_profile.py — Sentinel hardware profile classifier.

Evaluates a SystemInfo snapshot and categorises the machine into one of
three runtime modes defined by the Sentinel blueprint:

    Minimal   — 8–12 GB RAM, CPU-only
    Standard  — 12–20 GB RAM, larger models, no required GPU
    Advanced  — GPU with VRAM, or > 20 GB RAM

Each profile carries a recommended primary model, context token limit,
and pipeline concurrency setting.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from system.hardware_detector import SystemInfo


# ---------------------------------------------------------------------------
# Mode enum
# ---------------------------------------------------------------------------


class HardwareMode(str, Enum):
    """Sentinel runtime hardware mode."""

    MINIMAL = "minimal"
    STANDARD = "standard"
    ADVANCED = "advanced"


# ---------------------------------------------------------------------------
# Profile dataclass
# ---------------------------------------------------------------------------


@dataclass
class HardwareProfile:
    """Resolved hardware profile for the current machine.

    Attributes:
        mode: The classified HardwareMode.
        recommended_model: Default Ollama model tag for this profile.
        context_limit: Maximum context window in tokens.
        max_pipeline_concurrency: Number of pipeline steps that may run
            concurrently (1 = sequential).
        embedding_model: Recommended embedding model tag.
        reasoning_model: Recommended model for complex reasoning steps.
        notes: Human-readable explanation of the classification decision.
    """

    mode: HardwareMode
    recommended_model: str
    context_limit: int
    max_pipeline_concurrency: int
    embedding_model: str
    reasoning_model: str
    notes: str

    def is_gpu_capable(self) -> bool:
        """Return True if this profile assumes GPU acceleration."""
        return self.mode == HardwareMode.ADVANCED

    def summary(self) -> str:
        """Return a single-line summary of the profile.

        Returns:
            Formatted summary string.
        """
        return (
            f"[{self.mode.value.upper()}] "
            f"model={self.recommended_model}  "
            f"ctx={self.context_limit}  "
            f"concurrency={self.max_pipeline_concurrency}"
        )


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class HardwareProfiler:
    """Classifies a SystemInfo into a HardwareProfile.

    Thresholds (from the Sentinel blueprint):
        Minimal   — RAM < 12 GB  AND  no GPU with usable VRAM
        Standard  — 12 GB ≤ RAM < 20 GB  OR  GPU with < 6 GB VRAM
        Advanced  — RAM ≥ 20 GB  OR  GPU with ≥ 6 GB VRAM  OR  Apple Metal
    """

    # Minimum VRAM (GB) required to unlock Advanced mode via GPU
    _ADVANCED_VRAM_THRESHOLD: float = 6.0

    # RAM thresholds (GB)
    _MINIMAL_RAM_MAX: float = 12.0
    _STANDARD_RAM_MAX: float = 20.0

    def classify(self, info: SystemInfo) -> HardwareProfile:
        """Classify system hardware into a HardwareProfile.

        Args:
            info: A fully populated SystemInfo from SystemCheck.run().

        Returns:
            The appropriate HardwareProfile for the detected hardware.
        """
        mode, notes = self._determine_mode(info)
        return self._build_profile(mode, notes)

    # ------------------------------------------------------------------
    # Mode determination
    # ------------------------------------------------------------------

    def _determine_mode(self, info: SystemInfo) -> tuple:
        """Determine the runtime mode from system info.

        Args:
            info: Populated SystemInfo.

        Returns:
            Tuple of (HardwareMode, notes_string).
        """
        ram = info.ram_gb
        vram = info.total_vram_gb
        has_capable_gpu = (
            (info.has_cuda or info.has_rocm) and vram >= self._ADVANCED_VRAM_THRESHOLD
        )
        has_metal = info.has_metal

        # Advanced: meaningful GPU or large RAM
        if has_capable_gpu:
            notes = (
                f"GPU detected ({info.gpus[0].name}, {vram:.1f} GB VRAM). "
                f"System RAM: {ram:.1f} GB. Running in Advanced mode."
            )
            return HardwareMode.ADVANCED, notes

        if has_metal:
            notes = (
                f"Apple Metal GPU detected ({info.gpus[0].name if info.gpus else 'integrated'}). "
                f"System RAM: {ram:.1f} GB. Running in Advanced mode."
            )
            return HardwareMode.ADVANCED, notes

        if ram >= self._STANDARD_RAM_MAX:
            notes = (
                f"No GPU with sufficient VRAM, but {ram:.1f} GB RAM available. "
                "Running in Advanced mode (CPU large-memory path)."
            )
            return HardwareMode.ADVANCED, notes

        # Standard: medium RAM, no powerful GPU
        if ram >= self._MINIMAL_RAM_MAX:
            gpu_note = (
                f" GPU present ({info.gpus[0].name}, {vram:.1f} GB VRAM) but below "
                f"{self._ADVANCED_VRAM_THRESHOLD} GB threshold."
                if info.has_gpu else " No GPU detected."
            )
            notes = (
                f"{ram:.1f} GB RAM.{gpu_note} Running in Standard mode."
            )
            return HardwareMode.STANDARD, notes

        # Minimal: low RAM, no GPU
        notes = (
            f"Only {ram:.1f} GB RAM detected and no qualifying GPU. "
            "Running in Minimal mode — smaller models and reduced context."
        )
        return HardwareMode.MINIMAL, notes

    # ------------------------------------------------------------------
    # Profile building
    # ------------------------------------------------------------------

    @staticmethod
    def _build_profile(mode: HardwareMode, notes: str) -> HardwareProfile:
        """Construct the HardwareProfile for a given mode.

        Args:
            mode: The classified HardwareMode.
            notes: Human-readable classification explanation.

        Returns:
            A fully populated HardwareProfile.
        """
        if mode == HardwareMode.MINIMAL:
            return HardwareProfile(
                mode=mode,
                recommended_model="codellama:7b",
                context_limit=4096,
                max_pipeline_concurrency=1,
                embedding_model="nomic-embed-text",
                reasoning_model="mistral:7b",
                notes=notes,
            )

        if mode == HardwareMode.STANDARD:
            return HardwareProfile(
                mode=mode,
                recommended_model="codellama:13b",
                context_limit=8192,
                max_pipeline_concurrency=2,
                embedding_model="nomic-embed-text",
                reasoning_model="mixtral:8x7b",
                notes=notes,
            )

        # Advanced
        return HardwareProfile(
            mode=mode,
            recommended_model="codellama:34b",
            context_limit=16384,
            max_pipeline_concurrency=4,
            embedding_model="nomic-embed-text",
            reasoning_model="mixtral:8x7b",
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def classify_and_print(self, info: SystemInfo) -> HardwareProfile:
        """Classify the system and print a Rich-formatted summary.

        Args:
            info: A fully populated SystemInfo.

        Returns:
            The classified HardwareProfile.
        """
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        profile = self.classify(info)
        console = Console()

        mode_colours = {
            HardwareMode.MINIMAL: "yellow",
            HardwareMode.STANDARD: "cyan",
            HardwareMode.ADVANCED: "bold green",
        }
        colour = mode_colours[profile.mode]

        table = Table(show_header=False, border_style=colour, expand=False)
        table.add_column("Field", style=f"bold {colour}", width=24)
        table.add_column("Value", style="white")

        table.add_row("Mode", f"[{colour}]{profile.mode.value.upper()}[/{colour}]")
        table.add_row("OS", f"{info.os_name} {info.os_version[:40]}")
        table.add_row("CPU", f"{info.cpu_name} ({info.cpu_cores}C / {info.cpu_threads}T)")
        table.add_row("RAM", f"{info.ram_gb:.1f} GB total  ({info.ram_available_gb:.1f} GB available)")

        if info.gpus:
            for gpu in info.gpus:
                vram_str = f"{gpu.vram_gb:.1f} GB" if gpu.vram_gb else "unknown VRAM"
                table.add_row("GPU", f"{gpu.name}  [{vram_str}]  via {gpu.backend}")
        else:
            table.add_row("GPU", "None detected")

        table.add_row("Primary Model", profile.recommended_model)
        table.add_row("Reasoning Model", profile.reasoning_model)
        table.add_row("Embedding Model", profile.embedding_model)
        table.add_row("Context Limit", f"{profile.context_limit:,} tokens")
        table.add_row("Concurrency", str(profile.max_pipeline_concurrency))
        table.add_row("Notes", profile.notes)

        console.print(
            Panel(
                table,
                title=f"[bold {colour}]Hardware Profile[/bold {colour}]",
                border_style=colour,
            )
        )
        return profile
