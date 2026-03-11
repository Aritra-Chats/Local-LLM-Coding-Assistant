from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class ModelRouter(ABC):
    """Abstract base class for the Model Router.

    Selects the most appropriate local language model for a given step
    based on the active hardware profile, task type, context size, and
    observed model performance metrics.
    """

    @abstractmethod
    def select(self, step: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Select the best model for a pipeline step.

        Args:
            step: The pipeline step requiring a model, including task type
                  and complexity hints.
            context: The assembled context payload, used to estimate token load.

        Returns:
            The model identifier string (e.g. 'codellama:13b').
        """
        ...

    @abstractmethod
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Return metadata for all models currently available via Ollama.

        Returns:
            A list of model metadata dicts, each containing name, size,
            capabilities, and hardware requirements.
        """
        ...

    @abstractmethod
    def get_hardware_profile(self) -> str:
        """Return the currently active hardware profile.

        Returns:
            One of: 'minimal', 'standard', or 'advanced'.
        """
        ...

    @abstractmethod
    def fallback(self, failed_model: str, step: Dict[str, Any]) -> str:
        """Select a fallback model when the primary model fails or is unavailable.

        Args:
            failed_model: The identifier of the model that failed.
            step: The pipeline step that needs to be retried.

        Returns:
            The fallback model identifier string.
        """
        ...

    @abstractmethod
    def record_performance(self, model: str, step: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Record performance metrics for a model after a step completes.

        Used by the learning subsystem to improve future routing decisions.

        Args:
            model: The model identifier that was used.
            step: The step that was executed.
            result: The result dict including success status and timing.
        """
        ...


# ────────────────────────────────────────────────────────────────────────────

"""concrete_model_router.py — Sentinel concrete model router.

Responsibilities
----------------
1. Select the best Ollama model for a given pipeline step, tuned by:
   - Hardware profile (minimal / standard / advanced)
   - Task category   (reasoning / coding / debugging / research / devops / system)
   - Context size    (estimated token load from step context)
2. Enable council mode automatically when hardware profile is advanced.
3. Track per-model performance metrics (latency, success rate, token throughput).
4. Provide deterministic fallback chains so execution never stalls.

Model catalogue
---------------
Minimal   (≤12 GB RAM, CPU-only)
  reasoning / research / devops / system  → mistral:7b
  coding  / debugging                     → codellama:7b

Standard  (12–20 GB RAM)
  reasoning / research                    → mistral:13b  (with mixtral:8x7b as upgrade)
  coding  / debugging / devops            → codellama:13b
  system                                  → mistral:7b

Advanced  (≥20 GB RAM  OR  GPU ≥ 6 GB VRAM)
  reasoning / research                    → mixtral:8x7b
  debugging                               → deepseek-coder:33b
  coding  / devops                        → codellama:34b
  system                                  → mistral:13b

Council mode
------------
Enabled automatically when profile.mode == "advanced" and the step's
task category is "reasoning" or "research".  The engine's CouncilPlanner
can promote any step; ``council_eligible`` returns True for those steps.

Performance tracking
--------------------
``record_performance(model, step, result)`` stores rolling stats per
(model, category) pair, computing an exponential moving average for
latency and success rate.  ``select()`` uses these stats to downgrade a
model to its fallback when its rolling success rate drops below
``PERFORMANCE_DEGRADED_THRESHOLD``.
"""


import statistics
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from config.hardware_profile import HardwareMode, HardwareProfile, HardwareProfiler
from system.hardware_detector import SystemCheck

# ---------------------------------------------------------------------------
# Model catalogue
# ---------------------------------------------------------------------------

# (hardware_mode, task_category) → preferred model tag
_MODEL_CATALOGUE: Dict[Tuple[str, str], str] = {
    # ── Minimal ─────────────────────────────────────────────────────────────
    ("minimal", "reasoning"):  "mistral:7b",
    ("minimal", "research"):   "mistral:7b",
    ("minimal", "devops"):     "mistral:7b",
    ("minimal", "system"):     "mistral:7b",
    ("minimal", "coding"):     "codellama:7b",
    ("minimal", "debugging"):  "codellama:7b",
    # ── Standard ────────────────────────────────────────────────────────────
    ("standard", "reasoning"): "mistral:13b",
    ("standard", "research"):  "mistral:13b",
    ("standard", "coding"):    "codellama:13b",
    ("standard", "debugging"): "codellama:13b",
    ("standard", "devops"):    "codellama:13b",
    ("standard", "system"):    "mistral:7b",
    # ── Advanced ────────────────────────────────────────────────────────────
    ("advanced", "reasoning"): "mixtral:8x7b",
    ("advanced", "research"):  "mixtral:8x7b",
    ("advanced", "coding"):    "codellama:34b",
    ("advanced", "debugging"): "deepseek-coder:33b",
    ("advanced", "devops"):    "codellama:34b",
    ("advanced", "system"):    "mistral:13b",
}

# Default fallback when (mode, category) is not in catalogue
_DEFAULT_MODEL: Dict[str, str] = {
    "minimal":  "mistral:7b",
    "standard": "mistral:13b",
    "advanced": "mixtral:8x7b",
}

# ---------------------------------------------------------------------------
# Fallback chains — ordered list of models to try after the primary fails
# ---------------------------------------------------------------------------

_FALLBACK_CHAINS: Dict[str, List[str]] = {
    # Coding specialisms
    "deepseek-coder:33b": ["codellama:34b",   "codellama:13b",   "codellama:7b"],
    "codellama:34b":      ["codellama:13b",   "codellama:7b",    "mistral:7b"],
    "codellama:13b":      ["codellama:7b",    "mistral:7b"],
    "codellama:7b":       ["mistral:7b"],
    # General reasoning
    "mixtral:8x7b":       ["mistral:13b",     "mistral:7b"],
    "mistral:13b":        ["mistral:7b"],
    "mistral:7b":         [],
}

# ---------------------------------------------------------------------------
# Metadata catalogue: model capabilities and minimum hardware requirements
# ---------------------------------------------------------------------------

_MODEL_METADATA: List[Dict[str, Any]] = [
    {
        "name":           "codellama:7b",
        "size_gb":        4.1,
        "capabilities":   ["coding", "debugging"],
        "hardware_modes": ["minimal", "standard", "advanced"],
        "context_limit":  4096,
        "council_eligible": False,
    },
    {
        "name":           "codellama:13b",
        "size_gb":        7.8,
        "capabilities":   ["coding", "debugging", "devops"],
        "hardware_modes": ["standard", "advanced"],
        "context_limit":  8192,
        "council_eligible": False,
    },
    {
        "name":           "codellama:34b",
        "size_gb":        19.0,
        "capabilities":   ["coding", "debugging", "devops"],
        "hardware_modes": ["advanced"],
        "context_limit":  16384,
        "council_eligible": True,
    },
    {
        "name":           "deepseek-coder:33b",
        "size_gb":        18.5,
        "capabilities":   ["debugging", "coding"],
        "hardware_modes": ["advanced"],
        "context_limit":  16384,
        "council_eligible": True,
    },
    {
        "name":           "mistral:7b",
        "size_gb":        4.1,
        "capabilities":   ["reasoning", "research", "devops", "system"],
        "hardware_modes": ["minimal", "standard", "advanced"],
        "context_limit":  4096,
        "council_eligible": False,
    },
    {
        "name":           "mistral:13b",
        "size_gb":        7.8,
        "capabilities":   ["reasoning", "research", "system"],
        "hardware_modes": ["standard", "advanced"],
        "context_limit":  8192,
        "council_eligible": False,
    },
    {
        "name":           "mixtral:8x7b",
        "size_gb":        26.0,
        "capabilities":   ["reasoning", "research", "devops", "system"],
        "hardware_modes": ["advanced"],
        "context_limit":  32768,
        "council_eligible": True,
    },
]

# ---------------------------------------------------------------------------
# Task types that unlock council in advanced mode
# ---------------------------------------------------------------------------

_COUNCIL_ELIGIBLE_CATEGORIES = frozenset({"reasoning", "research"})

# Rolling EMA performance threshold — below this success rate, fall back
PERFORMANCE_DEGRADED_THRESHOLD: float = 0.60

# EMA alpha for updating rolling statistics
_EMA_ALPHA: float = 0.25


# ---------------------------------------------------------------------------
# Performance record dataclass
# ---------------------------------------------------------------------------


@dataclass
class ModelPerformanceRecord:
    """Rolling performance statistics for a (model, category) pair.

    Attributes:
        model:          Ollama model identifier.
        category:       Task category (coding, debugging, …).
        total_calls:    Total times the model was invoked.
        success_count:  Times the step completed with status != "failed".
        total_latency_ms: Cumulative latency in milliseconds.
        rolling_success_rate: EMA of per-call success (1.0 or 0.0).
        rolling_latency_ms:   EMA of per-call latency.
        last_updated:   Unix timestamp of last record_performance call.
    """

    model: str
    category: str
    total_calls: int = 0
    success_count: int = 0
    total_latency_ms: float = 0.0
    rolling_success_rate: float = 1.0   # optimistic seed
    rolling_latency_ms: float = 0.0
    last_updated: float = field(default_factory=time.time)

    # ----------------------------------------------------------------
    def update(self, success: bool, latency_ms: float) -> None:
        """Update rolling stats with a new observation.

        Args:
            success:    Whether the step was completed successfully.
            latency_ms: Wall-clock time taken in milliseconds.
        """
        self.total_calls += 1
        if success:
            self.success_count += 1
        self.total_latency_ms += latency_ms

        s = 1.0 if success else 0.0
        if self.total_calls == 1:
            self.rolling_success_rate = s
            self.rolling_latency_ms = latency_ms
        else:
            self.rolling_success_rate = (
                _EMA_ALPHA * s + (1 - _EMA_ALPHA) * self.rolling_success_rate
            )
            self.rolling_latency_ms = (
                _EMA_ALPHA * latency_ms + (1 - _EMA_ALPHA) * self.rolling_latency_ms
            )
        self.last_updated = time.time()

    # ----------------------------------------------------------------
    @property
    def average_latency_ms(self) -> float:
        """Mean latency across all recorded calls."""
        if self.total_calls == 0:
            return 0.0
        return self.total_latency_ms / self.total_calls

    @property
    def is_degraded(self) -> bool:
        """True when rolling success rate is below the degradation threshold."""
        return (
            self.total_calls >= 3
            and self.rolling_success_rate < PERFORMANCE_DEGRADED_THRESHOLD
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict for logging / inspection."""
        return {
            "model":                self.model,
            "category":             self.category,
            "total_calls":          self.total_calls,
            "success_count":        self.success_count,
            "total_latency_ms":     round(self.total_latency_ms, 2),
            "rolling_success_rate": round(self.rolling_success_rate, 4),
            "rolling_latency_ms":   round(self.rolling_latency_ms, 2),
            "average_latency_ms":   round(self.average_latency_ms, 2),
            "is_degraded":          self.is_degraded,
            "last_updated":         self.last_updated,
        }


# ---------------------------------------------------------------------------
# Concrete model router
# ---------------------------------------------------------------------------


class ConcreteModelRouter(ModelRouter):
    """Hardware-aware, performance-tracking model router.

    Args:
        hardware_profile: Pre-built HardwareProfile.  When omitted the
            router auto-detects the current machine on first call.
        force_mode: Override the detected hardware mode
            (``"minimal"`` | ``"standard"`` | ``"advanced"``).  Useful
            for testing or locked deployments.
    """

    def __init__(
        self,
        hardware_profile: Optional[HardwareProfile] = None,
        force_mode: Optional[str] = None,
    ) -> None:
        self._profile: Optional[HardwareProfile] = hardware_profile
        self._force_mode: Optional[str] = force_mode
        # (model, category) → ModelPerformanceRecord
        self._perf: Dict[Tuple[str, str], ModelPerformanceRecord] = {}

    # ------------------------------------------------------------------
    # ABC: select
    # ------------------------------------------------------------------

    def select(self, step: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Select the best available model for a pipeline step.

        Selection logic (in priority order):
        1. Explicit ``model_hint`` on the step, if present and non-empty.
        2. Catalogue look-up by (hardware_mode, category).
        3. Performance override — if the selected model is degraded, choose
           its first non-degraded fallback.
        4. Default model for the active hardware mode.

        Args:
            step:    Pipeline step dict; uses ``task_category``, ``category``,
                     ``agent``, and ``model_hint`` keys.
            context: Assembled context payload; currently unused for routing
                     but inspected for estimated token load in future work.

        Returns:
            Ollama model identifier string.
        """
        # 1. Honour an explicit hint
        hint = (step.get("model_hint") or "").strip()
        if hint:
            return hint

        profile = self._get_profile()
        mode = self._active_mode(profile)
        category = self._step_category(step)

        # 2. Catalogue look-up
        model = _MODEL_CATALOGUE.get((mode, category), _DEFAULT_MODEL.get(mode, "mistral:7b"))

        # 3. Performance degradation override
        key = (model, category)
        rec = self._perf.get(key)
        if rec and rec.is_degraded:
            model = self._first_healthy_fallback(model, category) or model

        return model

    # ------------------------------------------------------------------
    # ABC: get_available_models
    # ------------------------------------------------------------------

    def get_available_models(self) -> List[Dict[str, Any]]:
        """Return metadata for all models in the catalogue.

        The list is filtered to those compatible with the active hardware
        mode so the caller only sees models the current machine can run.

        Returns:
            List of model metadata dicts.
        """
        mode = self._active_mode(self._get_profile())
        return [
            m for m in _MODEL_METADATA
            if mode in m["hardware_modes"]
        ]

    # ------------------------------------------------------------------
    # ABC: get_hardware_profile
    # ------------------------------------------------------------------

    def get_hardware_profile(self) -> str:
        """Return the active hardware profile name.

        Returns:
            One of ``"minimal"``, ``"standard"``, ``"advanced"``.
        """
        return self._active_mode(self._get_profile())

    # ------------------------------------------------------------------
    # ABC: fallback
    # ------------------------------------------------------------------

    def fallback(self, failed_model: str, step: Dict[str, Any]) -> str:
        """Return the next model in the fallback chain after a failure.

        Falls back through the chain defined in ``_FALLBACK_CHAINS``.
        If the chain is exhausted, returns the hardware profile default.

        Args:
            failed_model: Identifier of the model that failed.
            step:         The pipeline step being retried.

        Returns:
            Fallback model identifier string.
        """
        chain = _FALLBACK_CHAINS.get(failed_model, [])
        category = self._step_category(step)
        for candidate in chain:
            rec = self._perf.get((candidate, category))
            if rec is None or not rec.is_degraded:
                return candidate

        # Chain exhausted — use profile default
        mode = self._active_mode(self._get_profile())
        return _DEFAULT_MODEL.get(mode, "mistral:7b")

    # ------------------------------------------------------------------
    # ABC: record_performance
    # ------------------------------------------------------------------

    def record_performance(
        self,
        model: str,
        step: Dict[str, Any],
        result: Dict[str, Any],
    ) -> None:
        """Record performance stats for a completed step.

        Args:
            model:  Ollama model identifier that was used.
            step:   The pipeline step that was executed.
            result: Result dict; inspected for ``status`` and
                    ``elapsed_ms`` keys.
        """
        category = self._step_category(step)
        success = result.get("status", "failed") != "failed"
        latency_ms = float(result.get("elapsed_ms", 0.0))

        key = (model, category)
        if key not in self._perf:
            self._perf[key] = ModelPerformanceRecord(model=model, category=category)
        self._perf[key].update(success, latency_ms)

    # ------------------------------------------------------------------
    # Council mode helpers
    # ------------------------------------------------------------------

    def council_eligible(self, step: Dict[str, Any]) -> bool:
        """Return True when this step should run in council mode.

        Council is enabled when ALL of the following hold:
        * Hardware profile is ``"advanced"``.
        * Step task category is in the council-eligible set
          (currently ``"reasoning"`` and ``"research"``).

        This supplements the pipeline-level CouncilPlanner — the router
        provides a hardware gate so council is never triggered on a
        resource-constrained machine.

        Args:
            step: Pipeline step dict.

        Returns:
            ``True`` if council mode should be applied.
        """
        if self._active_mode(self._get_profile()) != "advanced":
            return False
        return self._step_category(step) in _COUNCIL_ELIGIBLE_CATEGORIES

    # ------------------------------------------------------------------
    # Specialist selectors (also part of public API)
    # ------------------------------------------------------------------

    def select_reasoning_model(self, hardware_mode: Optional[str] = None) -> str:
        """Return the recommended reasoning model for the given hardware mode.

        Args:
            hardware_mode: Override mode; defaults to detected profile.

        Returns:
            Ollama model identifier.
        """
        mode = hardware_mode or self._active_mode(self._get_profile())
        return _MODEL_CATALOGUE.get((mode, "reasoning"), _DEFAULT_MODEL.get(mode, "mistral:7b"))

    def select_coding_model(self, hardware_mode: Optional[str] = None) -> str:
        """Return the recommended coding model for the given hardware mode.

        Args:
            hardware_mode: Override mode; defaults to detected profile.

        Returns:
            Ollama model identifier.
        """
        mode = hardware_mode or self._active_mode(self._get_profile())
        return _MODEL_CATALOGUE.get((mode, "coding"), _DEFAULT_MODEL.get(mode, "mistral:7b"))

    def select_debugging_model(self, hardware_mode: Optional[str] = None) -> str:
        """Return the recommended debugging model for the given hardware mode.

        Args:
            hardware_mode: Override mode; defaults to detected profile.

        Returns:
            Ollama model identifier.
        """
        mode = hardware_mode or self._active_mode(self._get_profile())
        return _MODEL_CATALOGUE.get((mode, "debugging"), _DEFAULT_MODEL.get(mode, "mistral:7b"))

    # ------------------------------------------------------------------
    # Performance inspection
    # ------------------------------------------------------------------

    def get_performance_stats(self) -> List[Dict[str, Any]]:
        """Return all recorded performance stats sorted by model name.

        Returns:
            List of :class:`ModelPerformanceRecord` dicts.
        """
        return [rec.to_dict() for rec in sorted(self._perf.values(), key=lambda r: r.model)]

    def get_model_stats(self, model: str) -> List[Dict[str, Any]]:
        """Return performance stats for a specific model across all categories.

        Args:
            model: Ollama model identifier.

        Returns:
            List of performance record dicts for the model.
        """
        return [
            rec.to_dict()
            for rec in self._perf.values()
            if rec.model == model
        ]

    def reset_performance(self) -> None:
        """Clear all recorded performance data.

        Useful for testing or after switching models/hardware.
        """
        self._perf.clear()

    def summary(self) -> str:
        """Return a one-line human-readable summary of the active routing config.

        Returns:
            Formatted summary string suitable for CLI display.
        """
        profile = self._get_profile()
        mode = self._active_mode(profile)
        coding = self.select_coding_model(mode)
        reasoning = self.select_reasoning_model(mode)
        debugging = self.select_debugging_model(mode)
        return (
            f"ModelRouter [{mode.upper()}] | "
            f"coding={coding}  reasoning={reasoning}  debugging={debugging} | "
            f"council={'ON' if mode == 'advanced' else 'OFF (not advanced)'} | "
            f"tracked={len(self._perf)} model-category pairs"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_profile(self) -> HardwareProfile:
        """Lazy-load and cache the hardware profile."""
        if self._profile is None:
            try:
                sc = SystemCheck()
                info = sc.run()
                profiler = HardwareProfiler()
                self._profile = profiler.classify(info)
            except Exception:
                # If hardware detection fails, fall back to standard profile
                from config.hardware_profile import HardwareMode
                from dataclasses import dataclass as _dc
                self._profile = HardwareProfile(
                    mode=HardwareMode.STANDARD,
                    recommended_model="codellama:13b",
                    context_limit=8192,
                    max_pipeline_concurrency=2,
                    embedding_model="nomic-embed-text",
                    reasoning_model="mistral:13b",
                    notes="Hardware detection failed; defaulting to Standard profile.",
                )
        return self._profile

    def _active_mode(self, profile: HardwareProfile) -> str:
        """Return the effective hardware mode string, applying any force override."""
        if self._force_mode:
            return self._force_mode
        return profile.mode.value  # HardwareMode is a str-Enum

    @staticmethod
    def _step_category(step: Dict[str, Any]) -> str:
        """Extract the task category from a step dict.

        Prefers the ``task_category`` key; falls back to ``category``,
        then the first word of the assigned ``agent`` name.

        Args:
            step: Pipeline step dict.

        Returns:
            Normalised lowercase category string.
        """
        raw = (
            step.get("task_category")
            or step.get("category")
            or step.get("agent", "")
        )
        # agent names are like "coding_agent", "reasoning_agent" — strip suffix
        cat = str(raw).lower().replace("_agent", "").replace("-", "_").strip()
        # Map known aliases
        _ALIAS = {
            "code":   "coding",
            "debug":  "debugging",
            "plan":   "reasoning",
            "search": "research",
            "ops":    "devops",
        }
        return _ALIAS.get(cat, cat) if cat else "reasoning"

    def _first_healthy_fallback(self, model: str, category: str) -> Optional[str]:
        """Walk the fallback chain and return the first non-degraded model.

        Args:
            model:    The primary (degraded) model.
            category: Task category for performance lookup.

        Returns:
            First healthy fallback identifier, or ``None`` if all are degraded.
        """
        for candidate in _FALLBACK_CHAINS.get(model, []):
            rec = self._perf.get((candidate, category))
            if rec is None or not rec.is_degraded:
                return candidate
        return None
