"""performance_tracker.py — Sentinel learning system: performance metrics.

Tracks four first-class metric families across the lifetime of a Sentinel
session.  All metrics are stored in-process in rolling data structures; a
``persist()`` / ``load()`` pair serialises them to JSON so progress survives
restarts.

Metric families
---------------
pipeline_success_rate
    Per-(goal_category, pipeline_mode) success / failure counts and a
    rolling EMA of the pass rate.  Fed by :meth:`record_pipeline_run`.

model_latency
    Per-(model, task_category) latency in milliseconds and a rolling EMA.
    Also tracks first-token latency separately when provided.
    Fed by :meth:`record_model_call`.

edit_acceptance_rate
    Per-agent edit acceptance observations (user accepted / rejected a
    code edit produced by the agent).  Tracks overall rate and per-agent
    rate.  Fed by :meth:`record_edit`.

tool_reliability
    Per-tool success / failure counts, latency, and consecutive-failure
    streak (used to surface unreliable tools quickly).
    Fed by :meth:`record_tool_call`.

EMA
---
All four families use an exponential moving average with ``alpha = 0.2``
(conservative smoothing) so recent observations matter more than old ones
without over-reacting to isolated outliers.

Persistence
-----------
Call ``tracker.persist(path)`` to write a JSON snapshot.
Call ``PerformanceTracker.load(path)`` to restore state.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# EMA helper
# ---------------------------------------------------------------------------

_EMA_ALPHA = 0.20   # 20 % weight on each new observation


def _ema(current: float, new_value: float) -> float:
    """Update an exponential moving average.

    Args:
        current:   Previous EMA value.
        new_value: New observation to fold in.

    Returns:
        Updated EMA value.
    """
    return _EMA_ALPHA * new_value + (1.0 - _EMA_ALPHA) * current


# ---------------------------------------------------------------------------
# Pipeline success-rate metric
# ---------------------------------------------------------------------------


@dataclass
class PipelineMetric:
    """Rolling statistics for a (category, mode) pipeline bucket.

    Attributes:
        category:        Task category string (e.g. ``"coding"``).
        mode:            Pipeline execution mode (``"solo"`` / ``"council"``).
        total_runs:      Total pipeline executions recorded.
        successful_runs: Runs whose status was *not* ``"failed"``.
        total_steps:     Cumulative step count across all runs.
        failed_steps:    Cumulative failed-step count.
        total_elapsed_ms: Cumulative wall-clock time.
        rolling_success_rate: EMA of per-run success (seed = 1.0).
        rolling_elapsed_ms:   EMA of per-run wall-clock time.
        last_updated:    Unix timestamp of last update.
    """

    category: str
    mode: str
    total_runs: int = 0
    successful_runs: int = 0
    total_steps: int = 0
    failed_steps: int = 0
    total_elapsed_ms: float = 0.0
    rolling_success_rate: float = 1.0
    rolling_elapsed_ms: float = 0.0
    last_updated: float = field(default_factory=time.time)

    # ----------------------------------------------------------------
    def record(self, success: bool, elapsed_ms: float,
               steps: int = 0, failed_step_count: int = 0) -> None:
        """Fold in a single pipeline run observation.

        Args:
            success:          Whether the pipeline completed without failure.
            elapsed_ms:       Total run time in milliseconds.
            steps:            Total step count for the run.
            failed_step_count: How many steps failed within the run.
        """
        self.total_runs += 1
        if success:
            self.successful_runs += 1
        self.total_steps += steps
        self.failed_steps += failed_step_count
        self.total_elapsed_ms += elapsed_ms

        s = 1.0 if success else 0.0
        if self.total_runs == 1:
            self.rolling_success_rate = s
            self.rolling_elapsed_ms = elapsed_ms
        else:
            self.rolling_success_rate = _ema(self.rolling_success_rate, s)
            self.rolling_elapsed_ms = _ema(self.rolling_elapsed_ms, elapsed_ms)
        self.last_updated = time.time()

    # ----------------------------------------------------------------
    @property
    def success_rate(self) -> float:
        """Cumulative (non-EMA) success rate across all recorded runs."""
        if self.total_runs == 0:
            return 1.0
        return self.successful_runs / self.total_runs

    @property
    def average_elapsed_ms(self) -> float:
        """Mean wall-clock time per run."""
        if self.total_runs == 0:
            return 0.0
        return self.total_elapsed_ms / self.total_runs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category":            self.category,
            "mode":                self.mode,
            "total_runs":          self.total_runs,
            "successful_runs":     self.successful_runs,
            "total_steps":         self.total_steps,
            "failed_steps":        self.failed_steps,
            "total_elapsed_ms":    round(self.total_elapsed_ms, 2),
            "rolling_success_rate": round(self.rolling_success_rate, 4),
            "success_rate":        round(self.success_rate, 4),
            "rolling_elapsed_ms":  round(self.rolling_elapsed_ms, 2),
            "average_elapsed_ms":  round(self.average_elapsed_ms, 2),
            "last_updated":        self.last_updated,
        }


# ---------------------------------------------------------------------------
# Model latency metric
# ---------------------------------------------------------------------------


@dataclass
class ModelMetric:
    """Rolling latency and reliability stats for a (model, category) pair.

    Attributes:
        model:                 Ollama model identifier.
        category:              Task category the model was used for.
        total_calls:           Total invocations recorded.
        successful_calls:      Calls where ``success=True``.
        total_latency_ms:      Cumulative latency.
        total_first_token_ms:  Cumulative first-token latency (0 if unknown).
        first_token_samples:   Count of calls where first-token latency was provided.
        rolling_latency_ms:    EMA of per-call latency.
        rolling_success_rate:  EMA of per-call success.
        rolling_first_token_ms: EMA of first-token latency.
        last_updated:          Unix timestamp of last update.
    """

    model: str
    category: str
    total_calls: int = 0
    successful_calls: int = 0
    total_latency_ms: float = 0.0
    total_first_token_ms: float = 0.0
    first_token_samples: int = 0
    rolling_latency_ms: float = 0.0
    rolling_success_rate: float = 1.0
    rolling_first_token_ms: float = 0.0
    last_updated: float = field(default_factory=time.time)

    # ----------------------------------------------------------------
    def record(self, success: bool, latency_ms: float,
               first_token_ms: Optional[float] = None) -> None:
        """Fold in a single model-call observation.

        Args:
            success:        Whether the call produced a usable result.
            latency_ms:     Full response latency in milliseconds.
            first_token_ms: Time-to-first-token in ms (``None`` if unavailable).
        """
        self.total_calls += 1
        if success:
            self.successful_calls += 1
        self.total_latency_ms += latency_ms

        s = 1.0 if success else 0.0
        if self.total_calls == 1:
            self.rolling_latency_ms = latency_ms
            self.rolling_success_rate = s
        else:
            self.rolling_latency_ms = _ema(self.rolling_latency_ms, latency_ms)
            self.rolling_success_rate = _ema(self.rolling_success_rate, s)

        if first_token_ms is not None:
            self.total_first_token_ms += first_token_ms
            self.first_token_samples += 1
            if self.first_token_samples == 1:
                self.rolling_first_token_ms = first_token_ms
            else:
                self.rolling_first_token_ms = _ema(
                    self.rolling_first_token_ms, first_token_ms
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
    def average_first_token_ms(self) -> float:
        """Mean first-token latency over sampled calls."""
        if self.first_token_samples == 0:
            return 0.0
        return self.total_first_token_ms / self.first_token_samples

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model":                  self.model,
            "category":               self.category,
            "total_calls":            self.total_calls,
            "successful_calls":       self.successful_calls,
            "total_latency_ms":       round(self.total_latency_ms, 2),
            "rolling_latency_ms":     round(self.rolling_latency_ms, 2),
            "average_latency_ms":     round(self.average_latency_ms, 2),
            "rolling_success_rate":   round(self.rolling_success_rate, 4),
            "first_token_samples":    self.first_token_samples,
            "rolling_first_token_ms": round(self.rolling_first_token_ms, 2),
            "average_first_token_ms": round(self.average_first_token_ms, 2),
            "last_updated":           self.last_updated,
        }


# ---------------------------------------------------------------------------
# Edit acceptance metric
# ---------------------------------------------------------------------------


@dataclass
class EditMetric:
    """Acceptance-rate statistics for code edits produced by an agent.

    An "edit" is any code change proposed to the user (via the CLI diff
    view or inline suggestion).  The user accepts or rejects it; this
    metric captures the aggregate signal.

    Attributes:
        agent:           Name of the agent that produced the edits.
        total_edits:     Total edits presented to the user.
        accepted_edits:  Edits the user explicitly accepted.
        rejected_edits:  Edits the user explicitly rejected.
        rolling_acceptance_rate: EMA of per-edit acceptance.
        last_updated:    Unix timestamp of last update.
    """

    agent: str
    total_edits: int = 0
    accepted_edits: int = 0
    rejected_edits: int = 0
    rolling_acceptance_rate: float = 1.0
    last_updated: float = field(default_factory=time.time)

    # ----------------------------------------------------------------
    def record(self, accepted: bool) -> None:
        """Fold in a single edit observation.

        Args:
            accepted: ``True`` if the user accepted the edit.
        """
        self.total_edits += 1
        if accepted:
            self.accepted_edits += 1
        else:
            self.rejected_edits += 1

        s = 1.0 if accepted else 0.0
        if self.total_edits == 1:
            self.rolling_acceptance_rate = s
        else:
            self.rolling_acceptance_rate = _ema(self.rolling_acceptance_rate, s)
        self.last_updated = time.time()

    # ----------------------------------------------------------------
    @property
    def acceptance_rate(self) -> float:
        """Cumulative (non-EMA) acceptance rate."""
        if self.total_edits == 0:
            return 1.0
        return self.accepted_edits / self.total_edits

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent":                   self.agent,
            "total_edits":             self.total_edits,
            "accepted_edits":          self.accepted_edits,
            "rejected_edits":          self.rejected_edits,
            "acceptance_rate":         round(self.acceptance_rate, 4),
            "rolling_acceptance_rate": round(self.rolling_acceptance_rate, 4),
            "last_updated":            self.last_updated,
        }


# ---------------------------------------------------------------------------
# Tool reliability metric
# ---------------------------------------------------------------------------


@dataclass
class ToolMetric:
    """Reliability statistics for a single registered tool.

    Attributes:
        tool_name:              Registered tool identifier.
        total_calls:            Total invocations recorded.
        successful_calls:       Calls that returned without error.
        total_latency_ms:       Cumulative latency.
        consecutive_failures:   Streak of consecutive failures (reset on success).
        rolling_success_rate:   EMA of per-call success.
        rolling_latency_ms:     EMA of per-call latency.
        last_updated:           Unix timestamp of last update.
    """

    tool_name: str
    total_calls: int = 0
    successful_calls: int = 0
    total_latency_ms: float = 0.0
    consecutive_failures: int = 0
    rolling_success_rate: float = 1.0
    rolling_latency_ms: float = 0.0
    last_updated: float = field(default_factory=time.time)

    # ----------------------------------------------------------------
    def record(self, success: bool, latency_ms: float = 0.0) -> None:
        """Fold in a single tool-call observation.

        Args:
            success:    Whether the tool invocation succeeded.
            latency_ms: Tool execution time in milliseconds.
        """
        self.total_calls += 1
        if success:
            self.successful_calls += 1
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1
        self.total_latency_ms += latency_ms

        s = 1.0 if success else 0.0
        if self.total_calls == 1:
            self.rolling_success_rate = s
            self.rolling_latency_ms = latency_ms
        else:
            self.rolling_success_rate = _ema(self.rolling_success_rate, s)
            self.rolling_latency_ms = _ema(self.rolling_latency_ms, latency_ms)
        self.last_updated = time.time()

    # ----------------------------------------------------------------
    @property
    def reliability_rate(self) -> float:
        """Cumulative (non-EMA) success rate."""
        if self.total_calls == 0:
            return 1.0
        return self.successful_calls / self.total_calls

    @property
    def average_latency_ms(self) -> float:
        """Mean latency across all recorded calls."""
        if self.total_calls == 0:
            return 0.0
        return self.total_latency_ms / self.total_calls

    @property
    def is_unreliable(self) -> bool:
        """True when consecutive failures exceed 3 OR rolling rate drops below 0.5."""
        return self.consecutive_failures >= 3 or (
            self.total_calls >= 5 and self.rolling_success_rate < 0.50
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name":             self.tool_name,
            "total_calls":           self.total_calls,
            "successful_calls":      self.successful_calls,
            "total_latency_ms":      round(self.total_latency_ms, 2),
            "consecutive_failures":  self.consecutive_failures,
            "rolling_success_rate":  round(self.rolling_success_rate, 4),
            "reliability_rate":      round(self.reliability_rate, 4),
            "rolling_latency_ms":    round(self.rolling_latency_ms, 2),
            "average_latency_ms":    round(self.average_latency_ms, 2),
            "is_unreliable":         self.is_unreliable,
            "last_updated":          self.last_updated,
        }


# ---------------------------------------------------------------------------
# Aggregated snapshot
# ---------------------------------------------------------------------------


@dataclass
class PerformanceSnapshot:
    """A point-in-time view of all performance metrics.

    Attributes:
        captured_at:    Unix timestamp when the snapshot was taken.
        pipelines:      All :class:`PipelineMetric` records.
        models:         All :class:`ModelMetric` records.
        edits:          All :class:`EditMetric` records.
        tools:          All :class:`ToolMetric` records.
    """

    captured_at: float = field(default_factory=time.time)
    pipelines: List[Dict[str, Any]] = field(default_factory=list)
    models: List[Dict[str, Any]] = field(default_factory=list)
    edits: List[Dict[str, Any]] = field(default_factory=list)
    tools: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "captured_at": self.captured_at,
            "pipelines":   self.pipelines,
            "models":      self.models,
            "edits":       self.edits,
            "tools":       self.tools,
        }


# ---------------------------------------------------------------------------
# PerformanceTracker — main public class
# ---------------------------------------------------------------------------


class PerformanceTracker:
    """Central metrics store for the Sentinel learning system.

    All ``record_*`` methods are safe to call from any part of the
    codebase; they never raise — errors are silently swallowed to avoid
    impacting the execution engine.

    Args:
        session_id: Optional human-readable label for this session.
    """

    def __init__(self, session_id: str = "") -> None:
        self.session_id = session_id or f"session_{int(time.time())}"
        # Internal stores keyed by opaque string keys
        self._pipelines: Dict[str, PipelineMetric] = {}
        self._models:    Dict[str, ModelMetric] = {}
        self._edits:     Dict[str, EditMetric] = {}
        self._tools:     Dict[str, ToolMetric] = {}

    # ------------------------------------------------------------------
    # Pipeline success rate
    # ------------------------------------------------------------------

    def record_pipeline_run(
        self,
        category: str,
        mode: str,
        success: bool,
        elapsed_ms: float,
        total_steps: int = 0,
        failed_steps: int = 0,
    ) -> None:
        """Record the outcome of a completed pipeline run.

        Args:
            category:    Task category (e.g. ``"coding"``).
            mode:        Pipeline mode (``"solo"`` / ``"council"``).
            success:     Whether the pipeline completed without failure status.
            elapsed_ms:  Total wall-clock time for the run.
            total_steps: Number of steps in the pipeline.
            failed_steps: How many steps failed.
        """
        try:
            key = f"{category}:{mode}"
            if key not in self._pipelines:
                self._pipelines[key] = PipelineMetric(category=category, mode=mode)
            self._pipelines[key].record(success, elapsed_ms, total_steps, failed_steps)
        except Exception:
            pass

    def record_pipeline_result(self, result: Any) -> None:
        """Convenience method: ingest a :class:`~core.execution_engine.PipelineRunResult`.

        Extracts category from the pipeline goal and delegates to
        :meth:`record_pipeline_run`.

        Args:
            result: A ``PipelineRunResult`` (or any object with compatible
                    attributes: ``.status``, ``.total_elapsed_ms``,
                    ``.completed_steps``, ``.failed_steps``).
        """
        try:
            success = getattr(result, "status", "failed") != "failed"
            elapsed = float(getattr(result, "total_elapsed_ms", 0))
            total = int(getattr(result, "completed_steps", 0)) + int(
                getattr(result, "failed_steps", 0)
            )
            failed = int(getattr(result, "failed_steps", 0))
            # Derive category from goal text (first word) and pipeline mode
            goal = str(getattr(result, "goal", ""))
            category = goal.split()[0].lower() if goal else "unknown"
            mode = "solo"  # PipelineRunResult doesn't carry mode; default solo
            self.record_pipeline_run(category, mode, success, elapsed, total, failed)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Model latency
    # ------------------------------------------------------------------

    def record_model_call(
        self,
        model: str,
        category: str,
        latency_ms: float,
        success: bool = True,
        first_token_ms: Optional[float] = None,
    ) -> None:
        """Record a single model invocation.

        Args:
            model:          Ollama model identifier (e.g. ``"codellama:13b"``).
            category:       Task category the model was invoked for.
            latency_ms:     Full response latency in milliseconds.
            success:        Whether the invocation produced a usable result.
            first_token_ms: Optional time-to-first-token in milliseconds.
        """
        try:
            key = f"{model}:{category}"
            if key not in self._models:
                self._models[key] = ModelMetric(model=model, category=category)
            self._models[key].record(success, latency_ms, first_token_ms)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Edit acceptance rate
    # ------------------------------------------------------------------

    def record_edit(self, agent: str, accepted: bool) -> None:
        """Record whether a user accepted an agent-generated code edit.

        Args:
            agent:    Name of the agent that produced the edit.
            accepted: ``True`` if the user accepted; ``False`` if rejected.
        """
        try:
            if agent not in self._edits:
                self._edits[agent] = EditMetric(agent=agent)
            self._edits[agent].record(accepted)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Tool reliability
    # ------------------------------------------------------------------

    def record_tool_call(
        self,
        tool_name: str,
        success: bool,
        latency_ms: float = 0.0,
    ) -> None:
        """Record a single tool invocation outcome.

        Args:
            tool_name:  Registered tool identifier.
            success:    Whether the tool call succeeded.
            latency_ms: Execution time in milliseconds.
        """
        try:
            if tool_name not in self._tools:
                self._tools[tool_name] = ToolMetric(tool_name=tool_name)
            self._tools[tool_name].record(success, latency_ms)
        except Exception:
            pass

    def record_tool_results(self, tool_results: List[Dict[str, Any]]) -> None:
        """Bulk-record tool results from an execution step.

        Expects each dict to carry ``"tool"``, ``"success"`` (or ``"status"``),
        and optionally ``"elapsed_ms"`` keys.

        Args:
            tool_results: List of tool-result dicts from the execution engine.
        """
        for r in tool_results:
            try:
                name = r.get("tool") or r.get("tool_name", "unknown")
                success = r.get("success", r.get("status", "ok") != "error")
                latency = float(r.get("elapsed_ms", 0.0))
                self.record_tool_call(name, bool(success), latency)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_pipeline_metrics(self) -> List[Dict[str, Any]]:
        """Return all pipeline metrics sorted by category."""
        return [m.to_dict() for m in sorted(
            self._pipelines.values(), key=lambda m: (m.category, m.mode)
        )]

    def get_model_metrics(self) -> List[Dict[str, Any]]:
        """Return all model metrics sorted by model name."""
        return [m.to_dict() for m in sorted(
            self._models.values(), key=lambda m: (m.model, m.category)
        )]

    def get_edit_metrics(self) -> List[Dict[str, Any]]:
        """Return all edit metrics sorted by agent name."""
        return [m.to_dict() for m in sorted(
            self._edits.values(), key=lambda m: m.agent
        )]

    def get_tool_metrics(self) -> List[Dict[str, Any]]:
        """Return all tool metrics sorted by tool name."""
        return [m.to_dict() for m in sorted(
            self._tools.values(), key=lambda m: m.tool_name
        )]

    def get_unreliable_tools(self) -> List[str]:
        """Return names of tools currently flagged as unreliable.

        Returns:
            List of tool_name strings.
        """
        return [m.tool_name for m in self._tools.values() if m.is_unreliable]

    def get_slowest_models(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """Return the ``top_n`` slowest models by rolling latency.

        Args:
            top_n: How many entries to return.

        Returns:
            List of model metric dicts, slowest first.
        """
        return sorted(
            self.get_model_metrics(),
            key=lambda m: m["rolling_latency_ms"],
            reverse=True,
        )[:top_n]

    def snapshot(self) -> PerformanceSnapshot:
        """Capture a point-in-time copy of all metrics.

        Returns:
            A :class:`PerformanceSnapshot` with all four metric lists
            populated as plain dicts.
        """
        return PerformanceSnapshot(
            pipelines=self.get_pipeline_metrics(),
            models=self.get_model_metrics(),
            edits=self.get_edit_metrics(),
            tools=self.get_tool_metrics(),
        )

    def summary(self) -> str:
        """Return a short human-readable summary of current metrics.

        Returns:
            Multi-line string suitable for CLI display.
        """
        p_total = sum(m.total_runs for m in self._pipelines.values())
        p_ok    = sum(m.successful_runs for m in self._pipelines.values())
        t_total = sum(m.total_calls for m in self._tools.values())
        t_ok    = sum(m.successful_calls for m in self._tools.values())
        e_total = sum(m.total_edits for m in self._edits.values())
        e_ok    = sum(m.accepted_edits for m in self._edits.values())
        unreliable = self.get_unreliable_tools()

        lines = [
            f"PerformanceTracker [{self.session_id}]",
            f"  pipelines : {p_ok}/{p_total} successful"
            + (f"  (rate={p_ok/p_total:.0%})" if p_total else ""),
            f"  tools     : {t_ok}/{t_total} successful"
            + (f"  (rate={t_ok/t_total:.0%})" if t_total else ""),
            f"  edits     : {e_ok}/{e_total} accepted"
            + (f"  (rate={e_ok/e_total:.0%})" if e_total else ""),
            f"  models    : {len(self._models)} tracked",
        ]
        if unreliable:
            lines.append(f"  UNRELIABLE TOOLS: {', '.join(unreliable)}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def persist(self, path: str | Path) -> None:
        """Write all metrics to a JSON file.

        Creates parent directories if needed.  Safe to call repeatedly;
        overwrites the existing file each time.

        Args:
            path: Filesystem path for the JSON snapshot.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "session_id": self.session_id,
            "saved_at":   time.time(),
            "pipelines":  {k: v.to_dict() for k, v in self._pipelines.items()},
            "models":     {k: v.to_dict() for k, v in self._models.items()},
            "edits":      {k: v.to_dict() for k, v in self._edits.items()},
            "tools":      {k: v.to_dict() for k, v in self._tools.items()},
        }
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "PerformanceTracker":
        """Restore a :class:`PerformanceTracker` from a JSON snapshot.

        Args:
            path: Path to the JSON file written by :meth:`persist`.

        Returns:
            A new :class:`PerformanceTracker` pre-populated with saved metrics.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tracker = cls(session_id=data.get("session_id", ""))

        for key, d in data.get("pipelines", {}).items():
            m = PipelineMetric(category=d["category"], mode=d["mode"])
            m.__dict__.update(d)
            tracker._pipelines[key] = m

        for key, d in data.get("models", {}).items():
            m = ModelMetric(model=d["model"], category=d["category"])
            m.__dict__.update(d)
            tracker._models[key] = m

        for key, d in data.get("edits", {}).items():
            m = EditMetric(agent=d["agent"])
            m.__dict__.update(d)
            tracker._edits[key] = m

        for key, d in data.get("tools", {}).items():
            m = ToolMetric(tool_name=d["tool_name"])
            m.__dict__.update(d)
            tracker._tools[key] = m

        return tracker
