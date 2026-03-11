"""prompt_optimizer.py — Sentinel learning system: prompt optimization.

Learns which prompt patterns produce the best outcomes and surfaces
concrete improvements the system can apply to future agent invocations.

Core concepts
-------------
PromptTemplate
    A named, parameterisable prompt fragment keyed by
    ``(agent_name, task_category)``.  Each template has a ``body`` (the
    actual text / jinja-style snippet) and metadata.

PromptObservation
    A single recorded invocation outcome: which template was used, whether
    the resulting output was accepted by the user (edit acceptance) or
    completed successfully (pipeline step status), and optional quality
    signals (latency, retry count).

PromptVariant
    A candidate variant of a template with its own rolling acceptance and
    success stats.  The optimizer compares variants to pick the best one.

PromptOptimizer
    Main class.  Exposes:
    - ``register_template``   — add a named template the system can use.
    - ``record_observation``  — log a usage outcome.
    - ``best_template``       — retrieve the current best-performing variant.
    - ``suggest_improvement`` — return a dict of actionable prompt patches.
    - ``all_stats``           — full stats breakdown for inspection.

Scoring
-------
Each variant is scored by a weighted combination of:
    acceptance_rate  × 0.50   (user accepted the produced edit)
    success_rate     × 0.30   (pipeline step completed without failure)
    speed_score      × 0.20   (inverse-normalised rolling latency)

Weights are exposed as module-level constants so they can be tuned without
changing the class implementation.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Scoring weights
# ---------------------------------------------------------------------------

WEIGHT_ACCEPTANCE: float = 0.50
WEIGHT_SUCCESS:    float = 0.30
WEIGHT_SPEED:      float = 0.20

# EMA alpha for individual variant stats
_EMA_ALPHA: float = 0.20

# Minimum observations before a variant is considered "reliable"
MIN_OBSERVATIONS: int = 3

# Reference latency (ms) used to normalise speed score — calls faster than
# this score 1.0, calls slower approach 0.0 asymptotically.
REFERENCE_LATENCY_MS: float = 3_000.0


def _ema(current: float, new_value: float) -> float:
    return _EMA_ALPHA * new_value + (1.0 - _EMA_ALPHA) * current


# ---------------------------------------------------------------------------
# PromptObservation
# ---------------------------------------------------------------------------


@dataclass
class PromptObservation:
    """A single recorded outcome for a prompt template invocation.

    Attributes:
        obs_id:         Unique identifier.
        template_key:   ``(agent, category)`` key of the template used.
        variant_id:     Which variant was used (maps to :class:`PromptVariant.variant_id`).
        accepted:       Whether the user accepted the generated edit (``None`` = unknown).
        success:        Whether the pipeline step completed successfully.
        latency_ms:     Full response latency in milliseconds.
        retries:        Number of retries consumed.
        recorded_at:    Unix timestamp.
    """

    template_key: str
    variant_id: str
    accepted:   Optional[bool] = None
    success:    bool = True
    latency_ms: float = 0.0
    retries:    int = 0
    obs_id:     str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    recorded_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "obs_id":       self.obs_id,
            "template_key": self.template_key,
            "variant_id":   self.variant_id,
            "accepted":     self.accepted,
            "success":      self.success,
            "latency_ms":   round(self.latency_ms, 2),
            "retries":      self.retries,
            "recorded_at":  self.recorded_at,
        }


# ---------------------------------------------------------------------------
# PromptVariant
# ---------------------------------------------------------------------------


@dataclass
class PromptVariant:
    """A single prompt variant with rolling performance statistics.

    Attributes:
        variant_id:              Unique identifier.
        template_key:            Parent template key.
        body:                    The prompt text / fragment.
        description:             Human-readable description of what makes this
                                 variant distinct.
        total_uses:              Times this variant was used.
        accepted_count:          Times user accepted the resulting edit.
        rejection_count:         Times user rejected the resulting edit.
        success_count:           Times the pipeline step succeeded.
        failure_count:           Times the step failed.
        total_latency_ms:        Cumulative latency.
        rolling_acceptance_rate: EMA of acceptance (1.0 or 0.0 per obs).
        rolling_success_rate:    EMA of step success.
        rolling_latency_ms:      EMA of latency.
        is_default:              True if this is the original baseline variant.
    """

    variant_id:   str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    template_key: str = ""
    body:         str = ""
    description:  str = ""
    total_uses:   int = 0
    accepted_count:  int = 0
    rejection_count: int = 0
    success_count:   int = 0
    failure_count:   int = 0
    total_latency_ms:        float = 0.0
    rolling_acceptance_rate: float = 1.0
    rolling_success_rate:    float = 1.0
    rolling_latency_ms:      float = 0.0
    is_default: bool = False

    # ----------------------------------------------------------------
    def record(
        self,
        accepted: Optional[bool],
        success: bool,
        latency_ms: float,
    ) -> None:
        """Fold in a single observation.

        Args:
            accepted:   Whether the user accepted the generated edit.
                        ``None`` means the edit acceptance is unknown.
            success:    Whether the pipeline step completed successfully.
            latency_ms: Full response latency.
        """
        self.total_uses += 1
        self.total_latency_ms += latency_ms

        s_step = 1.0 if success else 0.0
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1

        if accepted is not None:
            s_edit = 1.0 if accepted else 0.0
            if accepted:
                self.accepted_count += 1
            else:
                self.rejection_count += 1
        else:
            # Unknown acceptance → neither count incremented; use success as proxy
            s_edit = s_step

        if self.total_uses == 1:
            self.rolling_acceptance_rate = s_edit
            self.rolling_success_rate = s_step
            self.rolling_latency_ms = latency_ms
        else:
            self.rolling_acceptance_rate = _ema(self.rolling_acceptance_rate, s_edit)
            self.rolling_success_rate    = _ema(self.rolling_success_rate, s_step)
            self.rolling_latency_ms      = _ema(self.rolling_latency_ms, latency_ms)

    # ----------------------------------------------------------------
    @property
    def score(self) -> float:
        """Composite quality score (0–1, higher is better).

        Weighted average of acceptance rate, success rate, and speed score.
        Returns 0.0 for variants with fewer than :data:`MIN_OBSERVATIONS`.
        """
        if self.total_uses < MIN_OBSERVATIONS:
            return 0.0
        speed = REFERENCE_LATENCY_MS / max(self.rolling_latency_ms, 1.0)
        speed = min(speed, 1.0)   # cap at 1.0
        return (
            WEIGHT_ACCEPTANCE * self.rolling_acceptance_rate
            + WEIGHT_SUCCESS   * self.rolling_success_rate
            + WEIGHT_SPEED     * speed
        )

    @property
    def acceptance_rate(self) -> float:
        """Cumulative edit acceptance rate."""
        denominator = self.accepted_count + self.rejection_count
        return (self.accepted_count / denominator) if denominator else 1.0

    @property
    def success_rate(self) -> float:
        """Cumulative step success rate."""
        total = self.success_count + self.failure_count
        return (self.success_count / total) if total else 1.0

    @property
    def is_reliable(self) -> bool:
        """True when the variant has enough observations to be statistically useful."""
        return self.total_uses >= MIN_OBSERVATIONS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variant_id":             self.variant_id,
            "template_key":           self.template_key,
            "body":                   self.body,
            "description":            self.description,
            "total_uses":             self.total_uses,
            "accepted_count":         self.accepted_count,
            "rejection_count":        self.rejection_count,
            "success_count":          self.success_count,
            "failure_count":          self.failure_count,
            "total_latency_ms":       round(self.total_latency_ms, 2),
            "rolling_acceptance_rate": round(self.rolling_acceptance_rate, 4),
            "rolling_success_rate":   round(self.rolling_success_rate, 4),
            "rolling_latency_ms":     round(self.rolling_latency_ms, 2),
            "acceptance_rate":        round(self.acceptance_rate, 4),
            "success_rate":           round(self.success_rate, 4),
            "score":                  round(self.score, 4),
            "is_reliable":            self.is_reliable,
            "is_default":             self.is_default,
        }


# ---------------------------------------------------------------------------
# PromptTemplate
# ---------------------------------------------------------------------------


@dataclass
class PromptTemplate:
    """A named prompt template for a specific (agent, category) context.

    Contains one or more :class:`PromptVariant` objects.  The template
    tracks which variant is currently winning based on composite score.

    Attributes:
        template_key: Unique key, typically ``"{agent}:{category}"``.
        agent:        Agent name this template is designed for.
        category:     Task category (e.g. ``"coding"``).
        variants:     Ordered list of variants (index 0 = baseline).
    """

    template_key: str
    agent:    str = ""
    category: str = ""
    variants: List[PromptVariant] = field(default_factory=list)

    # ----------------------------------------------------------------
    def add_variant(self, body: str, description: str = "", is_default: bool = False) -> PromptVariant:
        """Create and register a new variant.

        Args:
            body:        Prompt text for this variant.
            description: Human-readable description.
            is_default:  Mark as the baseline variant.

        Returns:
            The newly created :class:`PromptVariant`.
        """
        v = PromptVariant(
            template_key=self.template_key,
            body=body,
            description=description,
            is_default=is_default,
        )
        self.variants.append(v)
        return v

    def get_variant(self, variant_id: str) -> Optional[PromptVariant]:
        """Look up a variant by its ID."""
        for v in self.variants:
            if v.variant_id == variant_id:
                return v
        return None

    def best_variant(self) -> Optional[PromptVariant]:
        """Return the highest-scoring reliable variant.

        Falls back to the default (baseline) variant when no reliable
        variant exists.

        Returns:
            The best :class:`PromptVariant`, or ``None`` if no variants exist.
        """
        if not self.variants:
            return None
        reliable = [v for v in self.variants if v.is_reliable]
        if reliable:
            return max(reliable, key=lambda v: v.score)
        # No reliable variant yet — return the default
        defaults = [v for v in self.variants if v.is_default]
        return defaults[0] if defaults else self.variants[0]

    def to_dict(self) -> Dict[str, Any]:
        best = self.best_variant()
        return {
            "template_key": self.template_key,
            "agent":        self.agent,
            "category":     self.category,
            "total_variants": len(self.variants),
            "best_variant_id": best.variant_id if best else None,
            "variants":     [v.to_dict() for v in self.variants],
        }


# ---------------------------------------------------------------------------
# PromptOptimizer — main public class
# ---------------------------------------------------------------------------


class PromptOptimizer:
    """Tracks prompt variant performance and suggests improvements.

    Args:
        session_id: Optional human-readable label for this session.
    """

    def __init__(self, session_id: str = "") -> None:
        self.session_id = session_id or f"prompt_opt_{int(time.time())}"
        self._templates: Dict[str, PromptTemplate] = {}
        self._observations: List[PromptObservation] = []

    # ------------------------------------------------------------------
    # Template management
    # ------------------------------------------------------------------

    def register_template(
        self,
        agent: str,
        category: str,
        body: str,
        description: str = "default",
    ) -> PromptTemplate:
        """Register the baseline prompt template for an (agent, category) pair.

        If a template for this key already exists, this call is a no-op and
        the existing template is returned.

        Args:
            agent:       Agent name (e.g. ``"coding_agent"``).
            category:    Task category (e.g. ``"coding"``).
            body:        Baseline prompt text.
            description: Human-readable description.

        Returns:
            The :class:`PromptTemplate` (new or existing).
        """
        key = _make_key(agent, category)
        if key not in self._templates:
            tmpl = PromptTemplate(template_key=key, agent=agent, category=category)
            tmpl.add_variant(body=body, description=description, is_default=True)
            self._templates[key] = tmpl
        return self._templates[key]

    def add_variant(
        self,
        agent: str,
        category: str,
        body: str,
        description: str = "",
    ) -> Optional[PromptVariant]:
        """Add a challenger variant to an existing template.

        The template must already be registered.  Returns ``None`` if not.

        Args:
            agent:       Agent name.
            category:    Task category.
            body:        Variant prompt text.
            description: Human-readable description.

        Returns:
            The new :class:`PromptVariant`, or ``None`` if template not found.
        """
        key = _make_key(agent, category)
        tmpl = self._templates.get(key)
        if tmpl is None:
            return None
        return tmpl.add_variant(body=body, description=description)

    # ------------------------------------------------------------------
    # Observation recording
    # ------------------------------------------------------------------

    def record_observation(
        self,
        agent: str,
        category: str,
        variant_id: str,
        *,
        accepted: Optional[bool] = None,
        success: bool = True,
        latency_ms: float = 0.0,
        retries: int = 0,
    ) -> None:
        """Record a single prompt invocation outcome.

        If the template or variant is not found the call is silently ignored.

        Args:
            agent:      Agent name.
            category:   Task category.
            variant_id: Which variant was used.
            accepted:   Whether the user accepted the generated edit.
            success:    Whether the pipeline step completed successfully.
            latency_ms: Full response latency.
            retries:    Retry count consumed.
        """
        try:
            key = _make_key(agent, category)
            tmpl = self._templates.get(key)
            if tmpl is None:
                return
            variant = tmpl.get_variant(variant_id)
            if variant is None:
                return
            variant.record(accepted=accepted, success=success, latency_ms=latency_ms)
            obs = PromptObservation(
                template_key=key,
                variant_id=variant_id,
                accepted=accepted,
                success=success,
                latency_ms=latency_ms,
                retries=retries,
            )
            self._observations.append(obs)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Best template retrieval
    # ------------------------------------------------------------------

    def best_template(self, agent: str, category: str) -> Optional[Dict[str, Any]]:
        """Return the best-performing variant body for an (agent, category) pair.

        Args:
            agent:    Agent name.
            category: Task category.

        Returns:
            Dict with keys ``"body"``, ``"variant_id"``, ``"score"``, and
            ``"is_default"``; or ``None`` if no template is registered.
        """
        key = _make_key(agent, category)
        tmpl = self._templates.get(key)
        if tmpl is None:
            return None
        v = tmpl.best_variant()
        if v is None:
            return None
        return {
            "body":       v.body,
            "variant_id": v.variant_id,
            "score":      round(v.score, 4),
            "is_default": v.is_default,
        }

    # ------------------------------------------------------------------
    # Improvement suggestions
    # ------------------------------------------------------------------

    def suggest_improvement(
        self, agent: str, category: str
    ) -> Dict[str, Any]:
        """Analyse variants and return an improvement recommendation.

        Compares all variants for the template and identifies:
        * The current champion (best variant by score).
        * Whether a non-default challenger has overtaken the default.
        * Variants that are clearly underperforming (score < 0.4 with ≥ 5 uses).

        Args:
            agent:    Agent name.
            category: Task category.

        Returns:
            A dict with keys:
            ``"template_key"``, ``"status"``, ``"champion_id"``,
            ``"champion_score"``, ``"champion_body"``,
            ``"recommendations"`` (list of human-readable strings),
            ``"underperforming_variants"`` (list of variant IDs to retire).
        """
        key = _make_key(agent, category)
        tmpl = self._templates.get(key)
        if tmpl is None:
            return {"template_key": key, "status": "no_template", "recommendations": []}

        best = tmpl.best_variant()
        recommendations: List[str] = []
        underperforming: List[str] = []

        if best is None:
            return {"template_key": key, "status": "no_variants", "recommendations": []}

        # Has a challenger beaten the default?
        defaults = [v for v in tmpl.variants if v.is_default]
        non_defaults = [v for v in tmpl.variants if not v.is_default and v.is_reliable]

        if defaults and non_defaults:
            default_score = defaults[0].score
            best_challenger = max(non_defaults, key=lambda v: v.score)
            if best_challenger.score > default_score + 0.05:
                recommendations.append(
                    f"Variant '{best_challenger.variant_id}' "
                    f"(score={best_challenger.score:.2f}) outperforms the default "
                    f"(score={default_score:.2f}). Consider promoting it."
                )

        # Identify underperforming variants
        for v in tmpl.variants:
            if v.is_reliable and v.score < 0.40:
                underperforming.append(v.variant_id)
                recommendations.append(
                    f"Variant '{v.variant_id}' has low score ({v.score:.2f}) "
                    f"after {v.total_uses} uses — consider retiring it."
                )

        # Low acceptance rate overall?
        all_reliable = [v for v in tmpl.variants if v.is_reliable]
        if all_reliable:
            avg_acceptance = sum(v.rolling_acceptance_rate for v in all_reliable) / len(all_reliable)
            if avg_acceptance < 0.50:
                recommendations.append(
                    f"Average acceptance rate is {avg_acceptance:.0%} across all variants. "
                    "Consider revising the prompt strategy for this (agent, category) pair."
                )

        status = "needs_attention" if recommendations else (
            "champion_identified" if best.is_reliable else "collecting_data"
        )
        return {
            "template_key":            key,
            "status":                  status,
            "champion_id":             best.variant_id,
            "champion_score":          round(best.score, 4),
            "champion_body":           best.body,
            "recommendations":         recommendations,
            "underperforming_variants": underperforming,
        }

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def all_stats(self) -> List[Dict[str, Any]]:
        """Return full stats for every registered template.

        Returns:
            List of template dicts (each containing variant breakdowns).
        """
        return [t.to_dict() for t in sorted(
            self._templates.values(), key=lambda t: t.template_key
        )]

    def observation_count(self) -> int:
        """Total number of observations recorded."""
        return len(self._observations)

    def summary(self) -> str:
        """Return a short human-readable summary.

        Returns:
            Multi-line string suitable for CLI display.
        """
        total_variants = sum(len(t.variants) for t in self._templates.values())
        reliable = sum(
            1 for t in self._templates.values()
            for v in t.variants if v.is_reliable
        )
        lines = [
            f"PromptOptimizer [{self.session_id}]",
            f"  templates  : {len(self._templates)}",
            f"  variants   : {total_variants}  ({reliable} reliable)",
            f"  observations: {self.observation_count()}",
        ]
        for tmpl in self._templates.values():
            best = tmpl.best_variant()
            if best:
                lines.append(
                    f"  {tmpl.template_key:<40} "
                    f"best={best.variant_id}  score={best.score:.3f}"
                    f"  uses={best.total_uses}"
                )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------


def _make_key(agent: str, category: str) -> str:
    """Build the canonical template key string."""
    return f"{agent.strip()}:{category.strip().lower()}"
