"""pipeline_generator.py — Sentinel dynamic pipeline generator.

Responsibilities
----------------
1. Convert an :class:`~tasks.task_manager.ExecutionPlan` (or a raw
   list of step dicts) into an executable :class:`Pipeline`.
2. Assign agents to each step (inherited from the plan; enforced here).
3. Support **council mode** — a panel of specialist agents reviews
   high-stakes or low-confidence steps.
4. Optimise step concurrency, model hints, retry budgets, and timeouts
   based on the active system mode (``"minimal"``, ``"standard"``,
   ``"advanced"``).

Pipeline JSON schema
--------------------
Each pipeline serialises to::

    {
      "pipeline_id":           "<uuid>",
      "task_id":               "<uuid|empty>",
      "goal":                  "<string>",
      "mode":                  "solo" | "council",
      "system_mode":           "minimal" | "standard" | "advanced",
      "steps": [
        {
          "step_id":           "<uuid>",
          "name":              "<string>",
          "description":       "<string>",
          "index":             0,
          "agent":             "<agent_name>",
          "tools":             ["<tool_name>", ...],
          "depends_on":        ["<step_id>", ...],
          "priority":          "high" | "medium" | "low",
          "context_hints":     ["<hint_name>", ...],
          "model_hint":        "<ollama_model_tag>",
          "max_retries":       2,
          "timeout_seconds":   120,
          "can_parallelize":   false,
          "council_agents":    ["<agent_name>", ...],
          "status":            "pending",
          "metadata":          {}
        }
      ],
      "complexity":            "low" | "medium" | "high",
      "estimated_steps":       5,
      "context_hints":         ["<hint_name>", ...],
      "created_at":            "<ISO-8601 UTC>",
      "optimizations_applied": ["<tag>", ...],
      "classification": {
        "category":   "<category>",
        "confidence": 0.95,
        "secondary":  null,
        "signals":    [],
        "scores":     {}
      },
      "plan_id": "<uuid>"
    }

Execution modes
---------------
``solo``
    A single specialist agent handles each step.  Default mode.

``council``
    High-priority or low-confidence steps gain a ``council_agents`` list.
    The ExecutionEngine invokes all listed agents and synthesises their
    outputs before advancing.  Disabled entirely in ``"minimal"`` mode.

System-mode optimisations
--------------------------
``minimal``
    * All steps execute sequentially (``can_parallelize = False``).
    * ``max_retries = 1``, ``timeout_seconds = 60``.
    * Smallest (7B) model hints.
    * Council mode forced off regardless of the requested mode.

``standard``
    * Steps may run concurrently (``can_parallelize = True``).
    * ``max_retries = 2``, ``timeout_seconds = 120``.
    * Medium (13B) model hints.
    * Council mode active when requested.

``advanced``
    * Maximum parallelism (``can_parallelize = True``).
    * ``max_retries = 3``, ``timeout_seconds = 300``.
    * Largest (34B / 8x7B) model hints.
    * Council mode auto-applied to all ``"high"`` priority steps.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

PIPELINE_MODES = ("solo", "council")
SYSTEM_MODES = ("minimal", "standard", "advanced")

# ---------------------------------------------------------------------------
# System-mode configuration
# ---------------------------------------------------------------------------

_SYSTEM_MODE_CONFIG: Dict[str, Dict[str, Any]] = {
    "minimal": {
        "max_retries": 1,
        "timeout_seconds": 60,
        "allow_parallelism": False,
        "council_allowed": False,
        "optimization_tags": ["sequential_forced", "retry_budget_minimal", "model_downscaled"],
    },
    "standard": {
        "max_retries": 2,
        "timeout_seconds": 120,
        "allow_parallelism": True,
        "council_allowed": True,
        "optimization_tags": ["parallelism_enabled", "retry_budget_standard", "model_standard"],
    },
    "advanced": {
        "max_retries": 3,
        "timeout_seconds": 300,
        "allow_parallelism": True,
        "council_allowed": True,
        "optimization_tags": ["parallelism_enabled", "retry_budget_extended", "model_upscaled"],
    },
}

# ---------------------------------------------------------------------------
# Model hints: (system_mode, agent_category) → Ollama model tag
# ---------------------------------------------------------------------------

_MODEL_HINTS: Dict[Tuple[str, str], str] = {
    ("minimal",  "coding"):    "codellama:7b",
    ("minimal",  "debugging"): "codellama:7b",
    ("minimal",  "research"):  "mistral:7b",
    ("minimal",  "reasoning"): "mistral:7b",
    ("minimal",  "devops"):    "mistral:7b",
    ("minimal",  "system"):    "mistral:7b",
    ("standard", "coding"):    "codellama:13b",
    ("standard", "debugging"): "codellama:13b",
    ("standard", "research"):  "mistral-nemo:12b",
    ("standard", "reasoning"): "mistral-nemo:12b",
    ("standard", "devops"):    "codellama:13b",
    ("standard", "system"):    "mistral:7b",
    ("advanced", "coding"):    "codellama:34b",
    ("advanced", "debugging"): "deepseek-coder:33b",
    ("advanced", "research"):  "mistral-nemo:12b",
    ("advanced", "reasoning"): "mistral-nemo:12b",
    ("advanced", "devops"):    "codellama:34b",
    ("advanced", "system"):    "mistral:13b",
}

# Default model when agent category is unknown
_DEFAULT_MODELS: Dict[str, str] = {
    "minimal":  "mistral:7b",
    "standard": "mistral-nemo:12b",
    "advanced": "mistral-nemo:12b",
}

# ---------------------------------------------------------------------------
# Council configuration
# ---------------------------------------------------------------------------

# Minimum classification confidence below which council is applied in
# non-minimal modes even if pipeline mode is "solo".
_COUNCIL_CONFIDENCE_THRESHOLD: float = 0.5

# Council agents by primary-agent category.  The first entry is always the
# primary agent; the rest are reviewers.
_COUNCIL_ELIGIBLE: Dict[str, List[str]] = {
    "coding":    ["coding", "reasoning"],
    "debugging": ["debugging", "reasoning"],
    "research":  ["research", "reasoning"],
    "devops":    ["devops", "reasoning"],
    "reasoning": ["reasoning", "research"],  # research provides supporting context
    "system":    ["system", "devops"],
}

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PipelineStep:
    """A single executable step within a :class:`Pipeline`.

    Attributes:
        step_id: Unique UUID string.
        name: Short display name.
        description: Full description of what this step does.
        index: Zero-based position in the ordered pipeline.
        agent: Specialist agent responsible for executing this step.
        tools: Tool names the agent is permitted to use.
        depends_on: ``step_id`` values that must complete before this step.
        priority: ``"high"``, ``"medium"``, or ``"low"``.
        context_hints: Context slice names to pre-load.
        model_hint: Recommended Ollama model tag for this step.
        max_retries: Maximum retry attempts on transient failure.
        timeout_seconds: Wall-clock timeout for this step's execution.
        can_parallelize: Whether this step may run concurrently with
            independent peers (determined by system mode).
        council_agents: Agents forming the review council.  Empty list
            means solo execution.
        status: Lifecycle state — ``"pending"`` initially.
        metadata: Arbitrary extra data.
    """

    step_id: str
    name: str
    description: str
    index: int
    agent: str
    tools: List[str] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)
    priority: str = "medium"
    context_hints: List[str] = field(default_factory=list)
    model_hint: str = ""
    max_retries: int = 2
    timeout_seconds: int = 120
    can_parallelize: bool = False
    council_agents: List[str] = field(default_factory=list)
    status: str = "pending"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict (JSON-compatible)."""
        return {
            "step_id":         self.step_id,
            "name":            self.name,
            "description":     self.description,
            "index":           self.index,
            "agent":           self.agent,
            "tools":           self.tools,
            "depends_on":      self.depends_on,
            "priority":        self.priority,
            "context_hints":   self.context_hints,
            "model_hint":      self.model_hint,
            "max_retries":     self.max_retries,
            "timeout_seconds": self.timeout_seconds,
            "can_parallelize": self.can_parallelize,
            "council_agents":  self.council_agents,
            "status":          self.status,
            "metadata":        self.metadata,
        }


@dataclass
class Pipeline:
    """A fully resolved, executable pipeline.

    Attributes:
        pipeline_id: Unique UUID string.
        task_id: Optional task identifier from the originating task dict.
        goal: The normalised goal string.
        mode: Execution mode — ``"solo"`` or ``"council"``.
        system_mode: Hardware mode — ``"minimal"``, ``"standard"``, or
            ``"advanced"``.
        steps: Ordered list of :class:`PipelineStep` objects.
        complexity: Overall plan complexity — ``"low"``, ``"medium"``, or
            ``"high"``.
        estimated_steps: Total step count (convenience field).
        context_hints: Aggregated unique context hint names.
        created_at: ISO-8601 UTC creation timestamp.
        optimizations_applied: Tags identifying which optimisations were
            applied during generation (e.g. ``"parallelism_enabled"``).
        classification: Serialised :class:`~tasks.task_manager.TaskClassification`
            dict, or empty dict if the pipeline was built from raw steps.
        plan_id: Originating plan UUID, or empty string.
    """

    pipeline_id: str
    task_id: str
    goal: str
    mode: str
    system_mode: str
    steps: List[PipelineStep] = field(default_factory=list)
    complexity: str = "medium"
    estimated_steps: int = 0
    context_hints: List[str] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    optimizations_applied: List[str] = field(default_factory=list)
    classification: Dict[str, Any] = field(default_factory=dict)
    plan_id: str = ""

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def ordered_steps(self) -> List[PipelineStep]:
        """Return steps sorted by :attr:`PipelineStep.index`."""
        return sorted(self.steps, key=lambda s: s.index)

    def steps_by_agent(self) -> Dict[str, List[PipelineStep]]:
        """Return a ``{agent_name: [steps]}`` grouping."""
        result: Dict[str, List[PipelineStep]] = {}
        for step in self.ordered_steps():
            result.setdefault(step.agent, []).append(step)
        return result

    def council_steps(self) -> List[PipelineStep]:
        """Return only steps that have a non-empty ``council_agents`` list."""
        return [s for s in self.ordered_steps() if s.council_agents]

    def step_by_id(self, step_id: str) -> Optional[PipelineStep]:
        """Look up a step by its ``step_id``."""
        return next((s for s in self.steps if s.step_id == step_id), None)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the full pipeline to a JSON-compatible dict."""
        return {
            "pipeline_id":           self.pipeline_id,
            "task_id":               self.task_id,
            "goal":                  self.goal,
            "mode":                  self.mode,
            "system_mode":           self.system_mode,
            "steps":                 [s.to_dict() for s in self.ordered_steps()],
            "complexity":            self.complexity,
            "estimated_steps":       self.estimated_steps,
            "context_hints":         self.context_hints,
            "created_at":            self.created_at,
            "optimizations_applied": self.optimizations_applied,
            "classification":        self.classification,
            "plan_id":               self.plan_id,
        }

    def summary(self) -> str:
        """Return a one-line human-readable summary."""
        council_count = len(self.council_steps())
        return (
            f"Pipeline '{self.pipeline_id[:8]}' | mode={self.mode} "
            f"system={self.system_mode} | steps={len(self.steps)} "
            f"council={council_count} | complexity={self.complexity}"
        )

    def __repr__(self) -> str:  # pragma: no cover
        return self.summary()


# ---------------------------------------------------------------------------
# StepEnricher
# ---------------------------------------------------------------------------


class StepEnricher:
    """Enrich a step dict or :class:`PipelineStep` with execution metadata.

    Adds:
    * ``model_hint`` — recommended Ollama model for the assigned agent.
    * ``max_retries`` — retry budget from system-mode configuration.
    * ``timeout_seconds`` — wall-clock timeout from system-mode configuration.
    * ``can_parallelize`` — whether concurrency is permitted.

    Example::

        enricher = StepEnricher("standard")
        step_dict = enricher.enrich(raw_step_dict)
    """

    def __init__(self, system_mode: str = "standard") -> None:
        if system_mode not in SYSTEM_MODES:
            raise ValueError(
                f"Invalid system_mode '{system_mode}'. Must be one of {SYSTEM_MODES}."
            )
        self.system_mode = system_mode
        self._cfg = _SYSTEM_MODE_CONFIG[system_mode]

    def enrich(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Return an enriched copy of *step* with execution metadata.

        Args:
            step: A step dict.  Must contain at least ``"agent"``.

        Returns:
            Enriched step dict.
        """
        step = dict(step)
        agent = step.get("agent", "")
        category = step.get("category", agent)

        step.setdefault("model_hint",       self._model_hint(agent, category))
        step.setdefault("max_retries",      self._cfg["max_retries"])
        step.setdefault("timeout_seconds",  self._cfg["timeout_seconds"])
        step.setdefault("can_parallelize",  self._cfg["allow_parallelism"])
        step.setdefault("council_agents",   [])
        step.setdefault("status",           "pending")
        step.setdefault("metadata",         {})

        return step

    def _model_hint(self, agent: str, category: str) -> str:
        """Resolve the recommended model tag for an agent/category."""
        key = (self.system_mode, agent)
        if key in _MODEL_HINTS:
            return _MODEL_HINTS[key]
        key2 = (self.system_mode, category)
        if key2 in _MODEL_HINTS:
            return _MODEL_HINTS[key2]
        return _DEFAULT_MODELS.get(self.system_mode, "mistral:7b")


# ---------------------------------------------------------------------------
# CouncilPlanner
# ---------------------------------------------------------------------------


class CouncilPlanner:
    """Determine council agents for each pipeline step.

    Council assignment rules
    ------------------------
    A step receives a non-empty ``council_agents`` list when **all** of the
    following hold:

    1. System mode is not ``"minimal"``.
    2. *Either* the pipeline ``mode`` is ``"council"`` *or* system mode is
       ``"advanced"`` and the step has ``priority == "high"``.
    3. The step's primary agent has at least two entries in
       :data:`_COUNCIL_ELIGIBLE`.

    Additionally, when classification confidence is below
    :data:`_COUNCIL_CONFIDENCE_THRESHOLD`, council is applied to all steps
    in ``"standard"`` and ``"advanced"`` modes regardless of priority.

    Example::

        planner = CouncilPlanner("standard", "council")
        agents = planner.council_for_step(step_dict, confidence=0.4)
        # → ["research", "reasoning"]
    """

    def __init__(
        self,
        system_mode: str = "standard",
        pipeline_mode: str = "solo",
    ) -> None:
        cfg = _SYSTEM_MODE_CONFIG.get(system_mode, _SYSTEM_MODE_CONFIG["standard"])
        self.system_mode = system_mode
        self.pipeline_mode = pipeline_mode
        self._council_allowed: bool = cfg["council_allowed"]
        self._is_advanced: bool = system_mode == "advanced"

    def council_for_step(
        self,
        step: Dict[str, Any],
        confidence: float = 1.0,
    ) -> List[str]:
        """Return the council agent list for *step*.

        Returns an empty list if council is disabled or not applicable.

        Args:
            step: A step dict with at least ``"agent"`` and ``"priority"``.
            confidence: Classification confidence for the task (0.0–1.0).

        Returns:
            List of agent names forming the council (includes the primary
            agent as the first entry).  Empty list → solo execution.
        """
        if not self._council_allowed:
            return []

        agent = step.get("agent", "")
        priority = step.get("priority", "medium")
        candidates = _COUNCIL_ELIGIBLE.get(agent, [agent])

        # Solo council lists (len=1) provide no benefit — skip.
        if len(candidates) <= 1:
            return []

        apply_council = (
            self.pipeline_mode == "council"
            or (self._is_advanced and priority == "high")
            or confidence < _COUNCIL_CONFIDENCE_THRESHOLD
        )
        return list(candidates) if apply_council else []


# ---------------------------------------------------------------------------
# PipelineOptimizer
# ---------------------------------------------------------------------------


class PipelineOptimizer:
    """Apply system-mode optimisations to a list of step dicts.

    Orchestrates :class:`StepEnricher` and :class:`CouncilPlanner` to
    produce a fully enriched step list, and collects the optimisation
    tags applied.

    Example::

        opt = PipelineOptimizer("advanced", "council")
        enriched_steps, tags = opt.optimise(raw_steps, confidence=0.45)
    """

    def __init__(
        self,
        system_mode: str = "standard",
        pipeline_mode: str = "solo",
    ) -> None:
        self.system_mode = system_mode
        self.pipeline_mode = pipeline_mode
        self._enricher = StepEnricher(system_mode)
        self._council = CouncilPlanner(system_mode, pipeline_mode)
        self._base_tags: List[str] = list(
            _SYSTEM_MODE_CONFIG[system_mode]["optimization_tags"]
        )

    def optimise(
        self,
        steps: List[Dict[str, Any]],
        confidence: float = 1.0,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Enrich *steps* and return ``(enriched_steps, optimization_tags)``.

        Args:
            steps: Raw step dicts (from planner or explicit list).
            confidence: Classification confidence score.

        Returns:
            Tuple of (enriched step dicts, list of optimisation tag strings).
        """
        tags = list(self._base_tags)
        enriched: List[Dict[str, Any]] = []

        council_used = False
        for step in steps:
            step = self._enricher.enrich(step)
            agents = self._council.council_for_step(step, confidence)
            if agents:
                step["council_agents"] = agents
                council_used = True
            enriched.append(step)

        if council_used:
            tags.append("council_enabled")
            if self.system_mode == "advanced":
                tags.append("council_auto_advanced")

        return enriched, tags


# ---------------------------------------------------------------------------
# Pipeline validator
# ---------------------------------------------------------------------------


class PipelineValidator:
    """Validate the structural integrity of a :class:`Pipeline`.

    Checks performed
    ----------------
    * Every step has a non-empty ``step_id``, ``name``, and ``agent``.
    * No duplicate ``step_id`` values.
    * All ``depends_on`` references resolve to known ``step_id`` values.
    * No circular dependency chains (DFS).
    * ``mode`` and ``system_mode`` are valid enum values.

    Example::

        validator = PipelineValidator()
        ok, errors = validator.validate(pipeline)
        if not ok:
            print(errors)
    """

    def validate(self, pipeline: "Pipeline") -> Tuple[bool, List[str]]:
        """Validate *pipeline* and return ``(is_valid, error_messages)``.

        Args:
            pipeline: The :class:`Pipeline` to validate.

        Returns:
            ``(True, [])`` if valid; ``(False, [error, ...])`` otherwise.
        """
        errors: List[str] = []

        if pipeline.mode not in PIPELINE_MODES:
            errors.append(f"Invalid mode '{pipeline.mode}'.")
        if pipeline.system_mode not in SYSTEM_MODES:
            errors.append(f"Invalid system_mode '{pipeline.system_mode}'.")

        step_ids: set = set()
        for step in pipeline.steps:
            if not step.step_id:
                errors.append(f"Step '{step.name}' missing step_id.")
            if not step.name:
                errors.append(f"Step index {step.index} missing name.")
            if not step.agent:
                errors.append(f"Step '{step.name}' missing agent.")
            if step.step_id in step_ids:
                errors.append(f"Duplicate step_id '{step.step_id}'.")
            step_ids.add(step.step_id)

        # Check depends_on references
        for step in pipeline.steps:
            for dep_id in step.depends_on:
                if dep_id not in step_ids:
                    errors.append(
                        f"Step '{step.name}' depends_on unknown step_id '{dep_id}'."
                    )

        # Check for cycles using DFS
        if not errors:
            id_to_step = {s.step_id: s for s in pipeline.steps}
            if self._has_cycle(pipeline.steps, id_to_step):
                errors.append("Pipeline contains a circular dependency.")

        return (len(errors) == 0), errors

    # ------------------------------------------------------------------
    # Cycle detection
    # ------------------------------------------------------------------

    def _has_cycle(
        self,
        steps: List["PipelineStep"],
        id_to_step: Dict[str, "PipelineStep"],
    ) -> bool:
        """Return True if the dependency graph has a cycle (DFS)."""
        WHITE, GREY, BLACK = 0, 1, 2
        colour: Dict[str, int] = {s.step_id: WHITE for s in steps}

        def dfs(sid: str) -> bool:
            if colour[sid] == GREY:
                return True   # back-edge → cycle
            if colour[sid] == BLACK:
                return False  # already fully explored
            colour[sid] = GREY
            for dep in id_to_step[sid].depends_on:
                if dep in colour and dfs(dep):
                    return True
            colour[sid] = BLACK
            return False

        return any(dfs(s.step_id) for s in steps if colour[s.step_id] == WHITE)


# ---------------------------------------------------------------------------
# DynamicPipelineGenerator — top-level public API
# ---------------------------------------------------------------------------


class DynamicPipelineGenerator:
    """Convert plans into optimised, executable pipelines.

    This is the single public entry point for the pipeline generation
    subsystem.  :class:`ConcretePipelineGeneratorAgent` delegates to this
    class for all generation, enrichment, and validation work.

    Attributes:
        system_mode: Active hardware profile mode.
        mode: Default pipeline execution mode (``"solo"`` or ``"council"``).
        optimizer: The :class:`PipelineOptimizer` instance.
        validator: The :class:`PipelineValidator` instance.

    Example::

        generator = DynamicPipelineGenerator(system_mode="advanced", mode="council")

        # From a TaskPlanner ExecutionPlan:
        pipeline = generator.from_execution_plan(execution_plan)

        # From a raw list of step dicts:
        pipeline = generator.from_steps(steps, goal="Deploy the service")

        print(pipeline.summary())
        for step in pipeline.ordered_steps():
            print(step.name, "→", step.agent, step.model_hint, step.council_agents)
    """

    def __init__(
        self,
        system_mode: str = "standard",
        mode: str = "solo",
    ) -> None:
        if system_mode not in SYSTEM_MODES:
            raise ValueError(
                f"Invalid system_mode '{system_mode}'. Must be one of {SYSTEM_MODES}."
            )
        if mode not in PIPELINE_MODES:
            raise ValueError(
                f"Invalid mode '{mode}'. Must be one of {PIPELINE_MODES}."
            )
        self.system_mode = system_mode
        self.mode = mode
        self.optimizer = PipelineOptimizer(system_mode, mode)
        self.validator = PipelineValidator()

    # ------------------------------------------------------------------
    # Primary entry points
    # ------------------------------------------------------------------

    def from_execution_plan(
        self,
        plan: Any,  # tasks.task_manager.ExecutionPlan
        task_id: str = "",
        mode: Optional[str] = None,
    ) -> "Pipeline":
        """Build a :class:`Pipeline` from a :class:`~tasks.task_manager.ExecutionPlan`.

        Args:
            plan: An :class:`~tasks.task_manager.ExecutionPlan` instance.
            task_id: Optional originating task identifier.
            mode: Override the instance-level execution mode for this pipeline.

        Returns:
            A fully resolved :class:`Pipeline`.
        """
        raw_steps = [s.to_dict() for s in plan.ordered_subtasks()]
        classification = plan.classification.to_dict()
        confidence = plan.classification.confidence
        return self._build(
            steps=raw_steps,
            goal=plan.goal,
            plan_id=plan.plan_id,
            task_id=task_id,
            complexity=plan.complexity,
            classification=classification,
            confidence=confidence,
            mode=mode or self.mode,
        )

    def from_steps(
        self,
        steps: List[Dict[str, Any]],
        goal: str = "",
        classification: Optional[Dict[str, Any]] = None,
        plan_id: str = "",
        task_id: str = "",
        complexity: str = "medium",
        mode: Optional[str] = None,
    ) -> "Pipeline":
        """Build a :class:`Pipeline` from a raw list of step dicts.

        This entry point is useful when the caller holds step dicts rather
        than a full :class:`~tasks.task_manager.ExecutionPlan`.

        Args:
            steps: List of step dicts.  Each must contain at least ``"name"``
                and ``"agent"``.
            goal: Human-readable goal string.
            classification: Optional serialised classification dict.
            plan_id: Optional originating plan UUID.
            task_id: Optional originating task identifier.
            complexity: Overall task complexity.
            mode: Override the instance-level execution mode for this pipeline.

        Returns:
            A fully resolved :class:`Pipeline`.
        """
        confidence = (classification or {}).get("confidence", 1.0)
        return self._build(
            steps=steps,
            goal=goal,
            plan_id=plan_id,
            task_id=task_id,
            complexity=complexity,
            classification=classification or {},
            confidence=float(confidence),
            mode=mode or self.mode,
        )

    # ------------------------------------------------------------------
    # Lower-level API (used by the abstract agent wrapper)
    # ------------------------------------------------------------------

    def enrich_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich a single step dict with execution metadata.

        Adds ``model_hint``, ``max_retries``, ``timeout_seconds``,
        ``can_parallelize``, and ``council_agents`` (always empty — for
        council assignment use :meth:`from_steps` / :meth:`from_execution_plan`).

        Args:
            step: A step dict.

        Returns:
            Enriched step dict.
        """
        return self.optimizer._enricher.enrich(step)

    def validate(self, pipeline: "Pipeline") -> Tuple[bool, List[str]]:
        """Validate *pipeline* and return ``(is_valid, errors)``."""
        return self.validator.validate(pipeline)

    def modify_for_failure(
        self,
        pipeline: "Pipeline",
        failure: Dict[str, Any],
    ) -> "Pipeline":
        """Revise a pipeline in response to a step failure.

        Strategy
        --------
        * Mark the failed step ``"failed"`` in metadata.
        * If ``failure["retry"]`` is True, reset the step status to
          ``"pending"`` and decrement its ``max_retries`` counter.
        * If ``failure["reclassify"]`` is set to a new category, inject a
          replacement step using the new agent category.
        * If ``failure["skip"]`` is True, mark the step ``"skipped"`` and
          relink its dependents to its own dependencies.

        Args:
            pipeline: The current :class:`Pipeline`.
            failure: Failure report dict.  Recognised keys:

                * ``"step_id"`` — ID of the failed step.
                * ``"retry"`` — bool; retry with decremented budget.
                * ``"reclassify"`` — str; new agent category for the step.
                * ``"skip"`` — bool; skip the step entirely.
                * ``"reason"`` — human-readable failure description.

        Returns:
            A revised :class:`Pipeline`.
        """
        failed_id = failure.get("step_id", "")
        step = pipeline.step_by_id(failed_id)

        if step is None:
            # Unknown step — return unchanged.
            return pipeline

        if failure.get("retry") and step.max_retries > 0:
            step.max_retries -= 1
            step.status = "pending"
            step.metadata["last_failure"] = failure.get("reason", "unknown")

        elif failure.get("reclassify"):
            new_category = str(failure["reclassify"])
            step.agent = new_category
            step.model_hint = self.optimizer._enricher._model_hint(
                new_category, new_category
            )
            step.council_agents = (
                self.optimizer._council.council_for_step(step.to_dict())
            )
            step.status = "pending"
            step.metadata["reclassified_from"] = step.agent
            step.metadata["last_failure"] = failure.get("reason", "unknown")

        elif failure.get("skip"):
            # Relink: remove failed step from all dependents' depends_on;
            # replace with the failed step's own dependencies.
            step.status = "skipped"
            for other in pipeline.steps:
                if failed_id in other.depends_on:
                    other.depends_on = [
                        d for d in other.depends_on if d != failed_id
                    ] + [d for d in step.depends_on if d not in other.depends_on]

        else:
            step.status = "failed"
            step.metadata["last_failure"] = failure.get("reason", "unknown")

        pipeline.optimizations_applied = list(
            dict.fromkeys(
                pipeline.optimizations_applied + ["failure_recovery_applied"]
            )
        )
        return pipeline

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _build(
        self,
        steps: List[Dict[str, Any]],
        goal: str,
        plan_id: str,
        task_id: str,
        complexity: str,
        classification: Dict[str, Any],
        confidence: float,
        mode: str,
    ) -> "Pipeline":
        """Core factory — enriches steps and assembles a Pipeline."""
        enriched_dicts, opt_tags = self.optimizer.optimise(steps, confidence)
        pipeline_steps = [self._dict_to_pipeline_step(d) for d in enriched_dicts]

        # Aggregate unique context hints from all steps.
        seen: set = set()
        ctx_hints: List[str] = []
        for st in pipeline_steps:
            for h in st.context_hints:
                if h and h not in seen:
                    seen.add(h)
                    ctx_hints.append(h)

        return Pipeline(
            pipeline_id=str(uuid.uuid4()),
            task_id=task_id,
            goal=goal,
            mode=mode,
            system_mode=self.system_mode,
            steps=pipeline_steps,
            complexity=complexity,
            estimated_steps=len(pipeline_steps),
            context_hints=ctx_hints,
            optimizations_applied=opt_tags,
            classification=classification,
            plan_id=plan_id,
        )

    @staticmethod
    def _dict_to_pipeline_step(d: Dict[str, Any]) -> "PipelineStep":
        """Convert a step dict into a :class:`PipelineStep`."""
        # Normalise step_id — subtasks use "subtask_id", raw dicts use "step_id".
        step_id = d.get("step_id") or d.get("subtask_id") or str(uuid.uuid4())
        return PipelineStep(
            step_id=step_id,
            name=d.get("name", ""),
            description=d.get("description", ""),
            index=d.get("index", 0),
            agent=d.get("agent", ""),
            tools=list(d.get("tools", [])),
            depends_on=list(d.get("depends_on", [])),
            priority=d.get("priority", "medium"),
            context_hints=list(d.get("context_hints", [])),
            model_hint=d.get("model_hint", ""),
            max_retries=d.get("max_retries", 2),
            timeout_seconds=d.get("timeout_seconds", 120),
            can_parallelize=bool(d.get("can_parallelize", False)),
            council_agents=list(d.get("council_agents", [])),
            status=d.get("status", "pending"),
            metadata=dict(d.get("metadata", {})),
        )
