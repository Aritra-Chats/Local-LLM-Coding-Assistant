"""pipeline_optimizer.py — Sentinel learning system: pipeline optimizations.

Consumes :class:`~learning.metrics_tracker.PerformanceTracker`
data to derive concrete, annotated changes to a pipeline before (or between)
execution runs.  The optimizer never mutates a pipeline in-place; it always
returns a *copy* with the applied patches and an attached
:class:`PipelineOptimizationReport`.

Optimization strategies
-----------------------
retry_budget
    When a ``(category, mode)`` pipeline bucket has a low rolling success
    rate, raise ``max_retries`` for steps whose task category matches.

model_downgrade
    When a model's rolling latency exceeds the ``latency_threshold_ms``
    (default 8 000 ms), replace its ``model_hint`` with the next lighter
    model from the fallback chain.

model_upgrade
    When advanced hardware is detected, upgrade ``model_hint`` for steps
    whose current hint is a sub-optimal small model.

drop_council
    When council steps consistently fail (pipeline rolling success rate
    below ``council_drop_threshold``), convert the step to solo mode by
    clearing ``council_agents``.

add_council
    When a step category is consistently successful in solo mode but the
    hardware profile allows council, optionally promote high-priority steps
    to council for better quality.

timeout_relaxation
    When a step category has a rolling latency significantly above its
    current ``timeout_seconds``, increase the timeout.

skip_unreliable_tools
    When a tool is flagged unreliable by :class:`PerformanceTracker`,
    remove it from the ``tools`` list of affected steps and record a
    suggestion to the report.

Each strategy emits one or more :class:`OptimizationSuggestion` objects
that are collected into a :class:`PipelineOptimizationReport`.
"""

from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from learning.metrics_tracker import PerformanceTracker

# ---------------------------------------------------------------------------
# Thresholds (all tunable)
# ---------------------------------------------------------------------------

# Rolling pipeline success rate below which we raise max_retries
RETRY_BOOST_THRESHOLD: float = 0.70

# Rolling model latency (ms) above which we suggest a model downgrade
LATENCY_THRESHOLD_MS: float = 8_000.0

# Rolling pipeline success rate for council steps below which we drop council
COUNCIL_DROP_THRESHOLD: float = 0.60

# How much headroom to add when relaxing a timeout (multiplier)
TIMEOUT_RELAXATION_FACTOR: float = 1.5

# Maximum retry budget we'll ever raise a step to
MAX_ALLOWED_RETRIES: int = 5

# Model downgrade chain (lighter ← heavier)
_DOWNGRADE_CHAIN: Dict[str, str] = {
    "codellama:34b":      "codellama:13b",
    "deepseek-coder:33b": "codellama:13b",
    "codellama:13b":      "codellama:7b",
    "mixtral:8x7b":       "mistral:13b",
    "mistral:13b":        "mistral:7b",
}

# Model upgrade chain (heavier ← lighter), used only in advanced mode
_UPGRADE_CHAIN: Dict[str, str] = {
    "codellama:7b":  "codellama:13b",
    "codellama:13b": "codellama:34b",
    "mistral:7b":    "mistral:13b",
    "mistral:13b":   "mixtral:8x7b",
}

# Categories that produce code edits → get upgrade candidates first in advanced
_CODE_CATEGORIES = frozenset({"coding", "debugging", "devops"})
_REASONING_CATEGORIES = frozenset({"reasoning", "research"})


# ---------------------------------------------------------------------------
# Suggestion dataclass
# ---------------------------------------------------------------------------


@dataclass
class OptimizationSuggestion:
    """A single, actionable recommended change to a pipeline step.

    Attributes:
        suggestion_id:  Unique identifier for this suggestion.
        strategy:       Name of the optimization strategy that produced it.
        step_id:        ID of the affected step (empty = pipeline-level).
        step_name:      Human-readable step name.
        field:          Which step field was changed (e.g. ``"max_retries"``).
        old_value:      Value before the optimization.
        new_value:      Value after the optimization.
        reason:         Human-readable explanation.
        confidence:     0–1 confidence score for the suggestion.
    """

    suggestion_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    strategy: str = ""
    step_id: str = ""
    step_name: str = ""
    field: str = ""
    old_value: Any = None
    new_value: Any = None
    reason: str = ""
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suggestion_id": self.suggestion_id,
            "strategy":      self.strategy,
            "step_id":       self.step_id,
            "step_name":     self.step_name,
            "field":         self.field,
            "old_value":     self.old_value,
            "new_value":     self.new_value,
            "reason":        self.reason,
            "confidence":    round(self.confidence, 4),
        }


# ---------------------------------------------------------------------------
# Optimization report
# ---------------------------------------------------------------------------


@dataclass
class PipelineOptimizationReport:
    """Summary of all optimizations applied (or suggested) for a pipeline.

    Attributes:
        report_id:     Unique identifier for this report.
        pipeline_id:   ID of the optimized pipeline.
        suggestions:   List of :class:`OptimizationSuggestion` objects.
        applied:       Whether suggestions were applied to the pipeline copy.
    """

    report_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    pipeline_id: str = ""
    suggestions: List[OptimizationSuggestion] = field(default_factory=list)
    applied: bool = False

    @property
    def total_suggestions(self) -> int:
        return len(self.suggestions)

    @property
    def strategies_used(self) -> List[str]:
        return sorted({s.strategy for s in self.suggestions})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id":         self.report_id,
            "pipeline_id":       self.pipeline_id,
            "total_suggestions": self.total_suggestions,
            "strategies_used":   self.strategies_used,
            "applied":           self.applied,
            "suggestions":       [s.to_dict() for s in self.suggestions],
        }

    def summary(self) -> str:
        if not self.suggestions:
            return f"PipelineOptimizationReport [{self.report_id}] — no changes needed"
        lines = [
            f"PipelineOptimizationReport [{self.report_id}] pipeline={self.pipeline_id}"
            f"  applied={self.applied}  suggestions={self.total_suggestions}",
        ]
        for s in self.suggestions:
            lines.append(
                f"  [{s.strategy}] step={s.step_name!r}  "
                f"{s.field}: {s.old_value!r} → {s.new_value!r}  ({s.reason})"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pipeline optimizer
# ---------------------------------------------------------------------------


class LearningPipelineOptimizer:
    """Derives and applies data-driven optimizations to pipelines.

    Args:
        tracker:              :class:`PerformanceTracker` providing the metrics.
        hardware_mode:        Active hardware profile string
                              (``"minimal"`` / ``"standard"`` / ``"advanced"``).
        latency_threshold_ms: Model latency (ms) above which a downgrade is
                              suggested.
        retry_threshold:      Pipeline rolling success rate below which retry
                              budgets are raised.
    """

    def __init__(
        self,
        tracker: PerformanceTracker,
        hardware_mode: str = "standard",
        latency_threshold_ms: float = LATENCY_THRESHOLD_MS,
        retry_threshold: float = RETRY_BOOST_THRESHOLD,
    ) -> None:
        self._tracker = tracker
        self._hw = hardware_mode
        self._latency_threshold = latency_threshold_ms
        self._retry_threshold = retry_threshold

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def optimize(self, pipeline: Any) -> tuple[Any, PipelineOptimizationReport]:
        """Apply optimizations to a deep copy of *pipeline*.

        Accepts either a :class:`~execution.pipeline.Pipeline` object (with a
        ``.steps`` list of :class:`~execution.pipeline.PipelineStep` objects and a
        ``.pipeline_id`` attribute) or a plain ``dict`` with the same schema.
        Returns (optimized_pipeline, report).

        Args:
            pipeline: The pipeline to optimize.

        Returns:
            Tuple of (optimized pipeline copy, :class:`PipelineOptimizationReport`).
        """
        pipeline_copy = copy.deepcopy(pipeline)
        report = PipelineOptimizationReport(
            pipeline_id=self._get_pipeline_id(pipeline_copy)
        )

        steps = self._get_steps(pipeline_copy)
        for step in steps:
            self._apply_retry_budget(step, report)
            self._apply_model_adjustment(step, report)
            self._apply_timeout_relaxation(step, report)
            self._apply_council_adjustment(step, report)
            self._apply_tool_prune(step, report)

        report.applied = True
        return pipeline_copy, report

    def suggest(self, pipeline: Any) -> PipelineOptimizationReport:
        """Generate a report of suggested optimizations WITHOUT mutating *pipeline*.

        Args:
            pipeline: The pipeline to analyse.

        Returns:
            :class:`PipelineOptimizationReport` with unapplied suggestions.
        """
        report = PipelineOptimizationReport(
            pipeline_id=self._get_pipeline_id(pipeline)
        )
        for step in self._get_steps(pipeline):
            # Work on a temp dict copy so we don't lose typed attributes
            sd = step if isinstance(step, dict) else step.to_dict()
            self._apply_retry_budget(sd, report, dry_run=True)
            self._apply_model_adjustment(sd, report, dry_run=True)
            self._apply_timeout_relaxation(sd, report, dry_run=True)
            self._apply_council_adjustment(sd, report, dry_run=True)
            self._apply_tool_prune(sd, report, dry_run=True)
        return report

    # ------------------------------------------------------------------
    # Strategy: retry_budget
    # ------------------------------------------------------------------

    def _apply_retry_budget(
        self, step: Any, report: PipelineOptimizationReport, dry_run: bool = False
    ) -> None:
        """Raise max_retries when the pipeline category has a low success rate."""
        category = self._step_category(step)
        mode = self._step_mode(step)
        pm = self._pipeline_metric(category, mode)
        if pm is None:
            return
        if pm["rolling_success_rate"] < self._retry_threshold:
            old = self._get(step, "max_retries", 2)
            new = min(old + 1, MAX_ALLOWED_RETRIES)
            if new != old:
                confidence = 1.0 - pm["rolling_success_rate"]
                report.suggestions.append(OptimizationSuggestion(
                    strategy="retry_budget",
                    step_id=self._step_id(step),
                    step_name=self._step_name(step),
                    field="max_retries",
                    old_value=old,
                    new_value=new,
                    reason=(
                        f"Pipeline rolling success rate for category '{category}' "
                        f"is {pm['rolling_success_rate']:.0%} (threshold {self._retry_threshold:.0%})"
                    ),
                    confidence=round(confidence, 4),
                ))
                if not dry_run:
                    self._set(step, "max_retries", new)

    # ------------------------------------------------------------------
    # Strategy: model_downgrade / model_upgrade
    # ------------------------------------------------------------------

    def _apply_model_adjustment(
        self, step: Any, report: PipelineOptimizationReport, dry_run: bool = False
    ) -> None:
        category  = self._step_category(step)
        model     = self._get(step, "model_hint", "")
        if not model:
            return

        mm = self._model_metric(model, category)

        # Downgrade: model is too slow
        if mm and mm["rolling_latency_ms"] > self._latency_threshold:
            downgraded = _DOWNGRADE_CHAIN.get(model)
            if downgraded:
                report.suggestions.append(OptimizationSuggestion(
                    strategy="model_downgrade",
                    step_id=self._step_id(step),
                    step_name=self._step_name(step),
                    field="model_hint",
                    old_value=model,
                    new_value=downgraded,
                    reason=(
                        f"Model '{model}' rolling latency "
                        f"{mm['rolling_latency_ms']:.0f} ms exceeds "
                        f"threshold {self._latency_threshold:.0f} ms"
                    ),
                    confidence=0.8,
                ))
                if not dry_run:
                    self._set(step, "model_hint", downgraded)
                return  # don't also try upgrade

        # Upgrade: we're on advanced hardware with a small model
        if self._hw == "advanced" and model in _UPGRADE_CHAIN:
            upgraded = _UPGRADE_CHAIN[model]
            report.suggestions.append(OptimizationSuggestion(
                strategy="model_upgrade",
                step_id=self._step_id(step),
                step_name=self._step_name(step),
                field="model_hint",
                old_value=model,
                new_value=upgraded,
                reason=(
                    f"Hardware profile is 'advanced'; upgrading '{model}' "
                    f"to '{upgraded}' for better quality"
                ),
                confidence=0.75,
            ))
            if not dry_run:
                self._set(step, "model_hint", upgraded)

    # ------------------------------------------------------------------
    # Strategy: timeout_relaxation
    # ------------------------------------------------------------------

    def _apply_timeout_relaxation(
        self, step: Any, report: PipelineOptimizationReport, dry_run: bool = False
    ) -> None:
        category = self._step_category(step)
        model    = self._get(step, "model_hint", "")
        timeout  = self._get(step, "timeout_seconds", 120)
        if not model:
            return
        mm = self._model_metric(model, category)
        if mm is None:
            return
        # Current timeout in ms for comparison
        latency_ms = mm["rolling_latency_ms"]
        timeout_ms = timeout * 1000
        if latency_ms > timeout_ms * 0.85:   # within 85% of current timeout
            new_timeout = int(timeout * TIMEOUT_RELAXATION_FACTOR)
            report.suggestions.append(OptimizationSuggestion(
                strategy="timeout_relaxation",
                step_id=self._step_id(step),
                step_name=self._step_name(step),
                field="timeout_seconds",
                old_value=timeout,
                new_value=new_timeout,
                reason=(
                    f"Rolling latency {latency_ms:.0f} ms is within 85% of "
                    f"timeout {timeout_ms:.0f} ms; relaxing to {new_timeout}s"
                ),
                confidence=0.85,
            ))
            if not dry_run:
                self._set(step, "timeout_seconds", new_timeout)

    # ------------------------------------------------------------------
    # Strategy: council adjustment
    # ------------------------------------------------------------------

    def _apply_council_adjustment(
        self, step: Any, report: PipelineOptimizationReport, dry_run: bool = False
    ) -> None:
        council = self._get(step, "council_agents", [])
        category = self._step_category(step)
        pm = self._pipeline_metric(category, "council")

        # Drop council if consistently failing
        if council and pm and pm["rolling_success_rate"] < COUNCIL_DROP_THRESHOLD:
            report.suggestions.append(OptimizationSuggestion(
                strategy="drop_council",
                step_id=self._step_id(step),
                step_name=self._step_name(step),
                field="council_agents",
                old_value=list(council),
                new_value=[],
                reason=(
                    f"Council mode rolling success rate for '{category}' is "
                    f"{pm['rolling_success_rate']:.0%} (threshold {COUNCIL_DROP_THRESHOLD:.0%})"
                ),
                confidence=0.9,
            ))
            if not dry_run:
                self._set(step, "council_agents", [])

        # Add council for high-priority reasoning/research steps on advanced hw
        elif (
            not council
            and self._hw == "advanced"
            and category in _REASONING_CATEGORIES
            and self._get(step, "priority", "medium") == "high"
        ):
            solo_pm = self._pipeline_metric(category, "solo")
            if solo_pm and solo_pm["rolling_success_rate"] > 0.75:
                candidates = ["reasoning_agent", "research_agent"]
                report.suggestions.append(OptimizationSuggestion(
                    strategy="add_council",
                    step_id=self._step_id(step),
                    step_name=self._step_name(step),
                    field="council_agents",
                    old_value=[],
                    new_value=candidates,
                    reason=(
                        f"Advanced hardware + high-priority '{category}' step with "
                        f"solo success rate {solo_pm['rolling_success_rate']:.0%}; "
                        "council will improve quality"
                    ),
                    confidence=0.65,
                ))
                if not dry_run:
                    self._set(step, "council_agents", candidates)

    # ------------------------------------------------------------------
    # Strategy: tool pruning
    # ------------------------------------------------------------------

    def _apply_tool_prune(
        self, step: Any, report: PipelineOptimizationReport, dry_run: bool = False
    ) -> None:
        tools = self._get(step, "tools", [])
        unreliable = set(self._tracker.get_unreliable_tools())
        bad = [t for t in tools if t in unreliable]
        if not bad:
            return
        new_tools = [t for t in tools if t not in unreliable]
        report.suggestions.append(OptimizationSuggestion(
            strategy="skip_unreliable_tools",
            step_id=self._step_id(step),
            step_name=self._step_name(step),
            field="tools",
            old_value=list(tools),
            new_value=new_tools,
            reason=f"Tools {bad} are flagged as unreliable by PerformanceTracker",
            confidence=0.95,
        ))
        if not dry_run:
            self._set(step, "tools", new_tools)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pipeline_metric(self, category: str, mode: str) -> Optional[Dict[str, Any]]:
        for m in self._tracker.get_pipeline_metrics():
            if m["category"] == category and m["mode"] == mode:
                return m
        return None

    def _model_metric(self, model: str, category: str) -> Optional[Dict[str, Any]]:
        for m in self._tracker.get_model_metrics():
            if m["model"] == model and m["category"] == category:
                return m
        return None

    @staticmethod
    def _get_pipeline_id(pipeline: Any) -> str:
        if isinstance(pipeline, dict):
            return pipeline.get("pipeline_id", "")
        return str(getattr(pipeline, "pipeline_id", ""))

    @staticmethod
    def _get_steps(pipeline: Any) -> List[Any]:
        if isinstance(pipeline, dict):
            return pipeline.get("steps", [])
        steps = getattr(pipeline, "steps", None)
        if steps is not None:
            return list(steps)
        # Pipeline objects may expose ordered_steps
        ordered = getattr(pipeline, "ordered_steps", None)
        return list(ordered) if ordered is not None else []

    @staticmethod
    def _step_category(step: Any) -> str:
        if isinstance(step, dict):
            raw = step.get("task_category") or step.get("category") or step.get("agent", "")
        else:
            raw = (
                getattr(step, "task_category", None)
                or getattr(step, "category", None)
                or getattr(step, "agent", "")
            )
        return str(raw).lower().replace("_agent", "").strip() or "coding"

    @staticmethod
    def _step_mode(step: Any) -> str:
        council = LearningPipelineOptimizer._get(step, "council_agents", [])
        return "council" if council else "solo"

    @staticmethod
    def _step_id(step: Any) -> str:
        return LearningPipelineOptimizer._get(step, "step_id", "")

    @staticmethod
    def _step_name(step: Any) -> str:
        return LearningPipelineOptimizer._get(step, "name", "")

    @staticmethod
    def _get(obj: Any, key: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    @staticmethod
    def _set(obj: Any, key: str, value: Any) -> None:
        if isinstance(obj, dict):
            obj[key] = value
        else:
            try:
                setattr(obj, key, value)
            except AttributeError:
                pass
