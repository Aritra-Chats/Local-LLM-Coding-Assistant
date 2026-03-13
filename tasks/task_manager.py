"""task_planner.py — Sentinel TaskPlanner.

Responsibilities
----------------
1. **Classify** a task prompt into one of six categories.
2. **Decompose** the task into atomic, ordered subtasks.
3. **Generate** a complete :class:`ExecutionPlan` with per-subtask agent
   assignments, tool bindings, dependency annotations, and context hints.

Task categories
---------------
reasoning  — analysis, explanation, comparison, decision-making
coding     — code generation, editing, file manipulation
debugging  — error diagnosis, test failure resolution, bug fixing
research   — web search, documentation lookup, knowledge gathering
devops     — CI/CD, git workflows, shell execution, dependency management
system     — OS operations, application launching, environment management

Design
------
* ``TaskClassifier``         — scores a goal against weighted keyword sets,
                               returns :class:`TaskClassification`.
* ``SubtaskDecomposer``      — applies per-category templates to a goal,
                               producing a list of :class:`Subtask` objects.
* ``ExecutionPlanGenerator`` — assembles subtasks into an :class:`ExecutionPlan`
                               with dependency graph and context hints.
* ``TaskPlanner``            — top-level orchestrator; the public API.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TASK_CATEGORIES = ("reasoning", "coding", "debugging", "research", "devops", "system")

# Minimum confidence score for the primary classification to be trusted.
_CONFIDENCE_THRESHOLD = 0.15

# ---------------------------------------------------------------------------
# Keyword scoring matrix
# Each entry: (category, weight, keyword_set)
# A keyword match for a category adds `weight` to that category's total score.
# ---------------------------------------------------------------------------

_SCORING_MATRIX: List[Tuple[str, float, frozenset]] = [
    # ── coding ──────────────────────────────────────────────────────────────
    ("coding", 2.0, frozenset({"write", "implement", "create", "generate", "build"})),
    ("coding", 1.5, frozenset({"function", "class", "method", "module", "script", "code"})),
    ("coding", 1.0, frozenset({"refactor", "edit", "update", "modify", "add", "change"})),
    ("coding", 0.5, frozenset({"file", "source", "program", "interface"})),
    # ── debugging ───────────────────────────────────────────────────────────
    ("debugging", 2.0, frozenset({"debug", "fix", "bug", "broken", "crash", "failure"})),
    ("debugging", 2.0, frozenset({"error", "exception", "traceback", "stacktrace"})),
    ("debugging", 1.5, frozenset({"test", "unittest", "pytest", "failing test", "failing"})),
    ("debugging", 1.0, frozenset({"why", "issue", "problem", "wrong", "incorrect"})),
    # ── research ────────────────────────────────────────────────────────────
    ("research", 2.0, frozenset({"research", "find out", "look up", "search for", "investigate"})),
    ("research", 1.5, frozenset({"documentation", "docs", "reference", "example", "how to"})),
    ("research", 1.0, frozenset({"web", "online", "internet", "library", "package", "api"})),
    ("research", 0.5, frozenset({"learn", "understand", "read about", "explore"})),
    # ── devops ──────────────────────────────────────────────────────────────
    ("devops", 2.0, frozenset({"deploy", "ci", "cd", "pipeline", "docker", "container"})),
    ("devops", 2.0, frozenset({"git", "commit", "push", "pull", "merge", "branch", "diff"})),
    ("devops", 1.5, frozenset({"install", "dependency", "package", "requirements", "pip"})),
    ("devops", 1.0, frozenset({"shell", "command", "script", "automate", "workflow"})),
    ("devops", 0.5, frozenset({"build", "release", "version", "tag"})),
    # ── reasoning ───────────────────────────────────────────────────────────
    ("reasoning", 2.0, frozenset({"reason", "explain", "analyse", "analyze", "evaluate"})),
    ("reasoning", 2.0, frozenset({"compare", "decide", "choose", "recommend", "assess"})),
    ("reasoning", 1.5, frozenset({"summarise", "summarize", "review", "critique", "justify"})),
    ("reasoning", 1.0, frozenset({"think", "consider", "reflect", "plan", "strategy"})),
    ("reasoning", 0.5, frozenset({"conclusion", "insight", "outcome", "result", "impact"})),
    # ── system ──────────────────────────────────────────────────────────────
    ("system", 2.0, frozenset({"launch", "open", "start", "run application", "execute"})),
    ("system", 1.5, frozenset({"os", "operating system", "platform", "environment", "process"})),
    ("system", 1.0, frozenset({"file manager", "browser", "editor", "terminal", "task"})),
    ("system", 0.5, frozenset({"path", "directory", "folder", "permission", "setting"})),
]


# ---------------------------------------------------------------------------
# Per-category subtask templates
# ---------------------------------------------------------------------------
# Each template is a list of step dicts defining the canonical workflow for
# that category.  Keys:
#   name        — display name
#   description — description template (may contain {goal})
#   agent       — specialist agent responsible
#   tools       — tool names likely needed
#   priority    — "high" | "medium" | "low"
#   context_hint— what context slice to pre-load
# ---------------------------------------------------------------------------

_TEMPLATES: Dict[str, List[Dict[str, Any]]] = {
    "coding": [
        {
            "name": "gather codebase context",
            "description": "Search and read relevant source files to understand the existing code before writing: {goal}",
            "agent": "coding",
            "tools": ["search_code", "read_file"],
            "priority": "high",
            "context_hint": "source_files",
        },
        {
            "name": "plan implementation",
            "description": "Reason about the design approach and structure for: {goal}",
            "agent": "reasoning",
            "tools": [],
            "priority": "high",
            "context_hint": "project_synopsis",
        },
        {
            "name": "implement code",
            "description": "Write the source code for: {goal}",
            "agent": "coding",
            "tools": ["write_file", "read_file"],
            "priority": "high",
            "context_hint": "source_files",
        },
        {
            "name": "write tests",
            "description": "Write unit tests covering the implementation for: {goal}",
            "agent": "coding",
            "tools": ["write_file"],
            "priority": "medium",
            "context_hint": "test_files",
        },
        {
            "name": "run tests",
            "description": "Execute the test suite to verify the implementation for: {goal}",
            "agent": "debugging",
            "tools": ["run_tests"],
            "priority": "high",
            "context_hint": "test_results",
        },
        {
            "name": "validate output",
            "description": "Confirm that all implementation goals are met for: {goal}",
            "agent": "reasoning",
            "tools": [],
            "priority": "medium",
            "context_hint": "test_results",
        },
    ],
    "debugging": [
        {
            "name": "reproduce the error",
            "description": "Run the failing test or command to reproduce: {goal}",
            "agent": "debugging",
            "tools": ["run_tests", "run_shell"],
            "priority": "high",
            "context_hint": "test_results",
        },
        {
            "name": "inspect error source",
            "description": "Read the file(s) that raised the error for: {goal}",
            "agent": "debugging",
            "tools": ["read_file", "search_code"],
            "priority": "high",
            "context_hint": "source_files",
        },
        {
            "name": "trace root cause",
            "description": "Search the codebase to trace the root cause of: {goal}",
            "agent": "debugging",
            "tools": ["search_code"],
            "priority": "high",
            "context_hint": "symbol_graph",
        },
        {
            "name": "reason about fix",
            "description": "Analyse the root cause and decide on the correct fix for: {goal}",
            "agent": "reasoning",
            "tools": [],
            "priority": "high",
            "context_hint": "source_files",
        },
        {
            "name": "apply fix",
            "description": "Edit the source file(s) to apply the fix for: {goal}",
            "agent": "coding",
            "tools": ["write_file", "read_file"],
            "priority": "high",
            "context_hint": "source_files",
        },
        {
            "name": "verify fix",
            "description": "Re-run the test suite to confirm the fix for: {goal}",
            "agent": "debugging",
            "tools": ["run_tests"],
            "priority": "high",
            "context_hint": "test_results",
        },
    ],
    "research": [
        {
            "name": "formulate search queries",
            "description": "Identify the best search terms and questions to answer: {goal}",
            "agent": "reasoning",
            "tools": [],
            "priority": "medium",
            "context_hint": "project_synopsis",
        },
        {
            "name": "web search",
            "description": "Search the web for resources relevant to: {goal}",
            "agent": "research",
            "tools": ["web_search"],
            "priority": "high",
            "context_hint": "web_results",
        },
        {
            "name": "read local documentation",
            "description": "Read any local documentation or spec files related to: {goal}",
            "agent": "research",
            "tools": ["read_file", "search_code"],
            "priority": "medium",
            "context_hint": "source_files",
        },
        {
            "name": "synthesize findings",
            "description": "Combine and analyse all gathered information about: {goal}",
            "agent": "reasoning",
            "tools": [],
            "priority": "high",
            "context_hint": "web_results",
        },
        {
            "name": "produce summary",
            "description": "Write a concise summary of findings for: {goal}",
            "agent": "reasoning",
            "tools": [],
            "priority": "medium",
            "context_hint": "web_results",
        },
    ],
    "devops": [
        {
            "name": "inspect repository state",
            "description": "Check current git diff and staged changes before: {goal}",
            "agent": "devops",
            "tools": ["git_diff"],
            "priority": "medium",
            "context_hint": "git_state",
        },
        {
            "name": "install dependencies",
            "description": "Install any required packages or dependencies for: {goal}",
            "agent": "devops",
            "tools": ["install_dependency"],
            "priority": "high",
            "context_hint": "environment",
        },
        {
            "name": "execute CI operations",
            "description": "Run shell commands and build steps required for: {goal}",
            "agent": "devops",
            "tools": ["run_shell"],
            "priority": "high",
            "context_hint": "environment",
        },
        {
            "name": "run CI test gate",
            "description": "Execute the test suite as a quality gate for: {goal}",
            "agent": "devops",
            "tools": ["run_tests"],
            "priority": "high",
            "context_hint": "test_results",
        },
        {
            "name": "commit changes",
            "description": "Stage and commit all changes related to: {goal}",
            "agent": "devops",
            "tools": ["git_commit"],
            "priority": "medium",
            "context_hint": "git_state",
        },
    ],
    "reasoning": [
        {
            "name": "gather relevant context",
            "description": "Read or retrieve information needed to reason about: {goal}",
            "agent": "reasoning",
            "tools": ["read_file", "search_code"],
            "priority": "high",
            "context_hint": "source_files",
        },
        {
            "name": "analyse the problem",
            "description": "Perform structured analysis of: {goal}",
            "agent": "reasoning",
            "tools": [],
            "priority": "high",
            "context_hint": "project_synopsis",
        },
        {
            "name": "compare options",
            "description": "Evaluate and compare candidate approaches for: {goal}",
            "agent": "reasoning",
            "tools": [],
            "priority": "medium",
            "context_hint": "project_synopsis",
        },
        {
            "name": "form decision",
            "description": "Select the best option with explicit rationale for: {goal}",
            "agent": "reasoning",
            "tools": [],
            "priority": "high",
            "context_hint": "project_synopsis",
        },
        {
            "name": "produce explanation",
            "description": "Write a clear explanation of the conclusion for: {goal}",
            "agent": "reasoning",
            "tools": [],
            "priority": "medium",
            "context_hint": "project_synopsis",
        },
    ],
    "system": [
        {
            "name": "inspect environment",
            "description": "Query the current system state relevant to: {goal}",
            "agent": "system",
            "tools": ["run_shell"],
            "priority": "medium",
            "context_hint": "environment",
        },
        {
            "name": "install required software",
            "description": "Install any software or packages needed for: {goal}",
            "agent": "system",
            "tools": ["install_dependency"],
            "priority": "high",
            "context_hint": "environment",
        },
        {
            "name": "execute system operation",
            "description": "Carry out the primary OS-level operation for: {goal}",
            "agent": "system",
            "tools": ["open_application", "run_shell"],
            "priority": "high",
            "context_hint": "environment",
        },
        {
            "name": "verify outcome",
            "description": "Confirm the system operation completed successfully for: {goal}",
            "agent": "system",
            "tools": ["run_shell"],
            "priority": "medium",
            "context_hint": "environment",
        },
    ],
}

# How many explicit phrases (split on conjunctions) trigger extra subtasks
_GOAL_SPLIT_RE = re.compile(
    r"\b(?:then|and then|after that|also|additionally|furthermore|finally|;)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TaskClassification:
    """Result of classifying a task prompt.

    Attributes:
        category: Primary task category — one of :data:`TASK_CATEGORIES`.
        confidence: Normalised confidence score in ``[0.0, 1.0]``.
        secondary: Second-best category, or ``None`` if not applicable.
        signals: Keywords that triggered this classification.
        scores: Raw per-category score dict (useful for debugging).
    """

    category: str
    confidence: float
    secondary: Optional[str] = None
    signals: List[str] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "confidence": round(self.confidence, 4),
            "secondary": self.secondary,
            "signals": self.signals,
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
        }


@dataclass
class Subtask:
    """A single atomic unit of work within an execution plan.

    Attributes:
        subtask_id: Unique UUID for this subtask.
        name: Short display name.
        description: Detailed description (may include the original goal text).
        category: Task category this subtask belongs to.
        agent: Name of the specialist agent responsible for this subtask.
        tools: Tool names likely needed by the agent.
        depends_on: List of ``subtask_id`` values this subtask must wait for.
        index: Position in the ordered plan (0-based).
        priority: ``"high"``, ``"medium"``, or ``"low"``.
        context_hints: Context slice names to pre-load before execution.
        status: Current lifecycle state — ``"pending"`` initially.
        metadata: Arbitrary extra data (prompt fragments, tags, etc.).
    """

    subtask_id: str
    name: str
    description: str
    category: str
    agent: str
    tools: List[str] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)
    index: int = 0
    priority: str = "medium"
    context_hints: List[str] = field(default_factory=list)
    status: str = "pending"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subtask_id": self.subtask_id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "agent": self.agent,
            "tools": self.tools,
            "depends_on": self.depends_on,
            "index": self.index,
            "priority": self.priority,
            "context_hints": self.context_hints,
            "status": self.status,
            "metadata": self.metadata,
        }


@dataclass
class ExecutionPlan:
    """A complete, ordered execution plan for a task.

    Attributes:
        plan_id: Unique UUID for this plan.
        goal: The normalised goal string extracted from the original prompt.
        raw_prompt: The original user prompt.
        classification: How the task was classified.
        subtasks: Ordered list of :class:`Subtask` objects.
        complexity: ``"low"``, ``"medium"``, or ``"high"``.
        context_hints: Aggregated unique context slice names needed by the plan.
        created_at: ISO-8601 UTC timestamp of plan creation.
    """

    plan_id: str
    goal: str
    raw_prompt: str
    classification: TaskClassification
    subtasks: List[Subtask] = field(default_factory=list)
    complexity: str = "medium"
    context_hints: List[str] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def ordered_subtasks(self) -> List[Subtask]:
        """Return subtasks sorted by their ``index`` field."""
        return sorted(self.subtasks, key=lambda s: s.index)

    def high_priority_subtasks(self) -> List[Subtask]:
        """Return only ``"high"`` priority subtasks in order."""
        return [s for s in self.ordered_subtasks() if s.priority == "high"]

    def subtask_by_id(self, subtask_id: str) -> Optional[Subtask]:
        """Look up a subtask by its ID."""
        return next((s for s in self.subtasks if s.subtask_id == subtask_id), None)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the plan to a plain dict suitable for JSON."""
        return {
            "plan_id": self.plan_id,
            "goal": self.goal,
            "raw_prompt": self.raw_prompt,
            "classification": self.classification.to_dict(),
            "subtasks": [s.to_dict() for s in self.ordered_subtasks()],
            "complexity": self.complexity,
            "context_hints": self.context_hints,
            "created_at": self.created_at,
        }

    def summary(self) -> str:
        """Return a one-line human-readable summary."""
        return (
            f"Plan '{self.plan_id[:8]}' | category={self.classification.category} "
            f"({self.classification.confidence:.0%}) | steps={len(self.subtasks)} | "
            f"complexity={self.complexity}"
        )

    def __repr__(self) -> str:  # pragma: no cover
        return self.summary()


# ---------------------------------------------------------------------------
# TaskClassifier
# ---------------------------------------------------------------------------


class TaskClassifier:
    """Classify a task goal into one of the six Sentinel task categories.

    Uses a weighted keyword scoring approach: each entry in
    :data:`_SCORING_MATRIX` contributes its weight to a category's total when
    any of its keywords appear in the (lowercased) goal.

    Confidence is computed by normalising the top score against the sum of
    all scores, giving a value in ``[0.0, 1.0]``.

    Example::

        clf = TaskClassifier()
        result = clf.classify("Fix the failing pytest test in auth.py")
        # result.category == "debugging", result.confidence > 0.5
    """

    def classify(self, goal: str) -> TaskClassification:
        """Classify *goal* and return a :class:`TaskClassification`.

        Args:
            goal: The task goal string (raw or normalised).

        Returns:
            :class:`TaskClassification` with category, confidence, secondary,
            signals, and raw scores.
        """
        lower = goal.lower()
        scores: Dict[str, float] = {cat: 0.0 for cat in TASK_CATEGORIES}
        signals: List[str] = []

        for category, weight, keywords in _SCORING_MATRIX:
            for kw in keywords:
                if kw in lower:
                    scores[category] += weight
                    signals.append(kw)

        total = sum(scores.values())
        if total == 0:
            # No signals — fall back to "coding" as the safest default
            return TaskClassification(
                category="coding",
                confidence=0.0,
                secondary=None,
                signals=[],
                scores=scores,
            )

        # Rank categories by score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        primary_cat, primary_score = ranked[0]
        secondary_cat, secondary_score = ranked[1]

        confidence = primary_score / total
        secondary = secondary_cat if secondary_score > 0 else None

        # Deduplicate signals while preserving order
        seen: set = set()
        unique_signals: List[str] = []
        for s in signals:
            if s not in seen:
                seen.add(s)
                unique_signals.append(s)

        return TaskClassification(
            category=primary_cat,
            confidence=confidence,
            secondary=secondary,
            signals=unique_signals,
            scores=scores,
        )

    def classify_multi(self, goals: List[str]) -> List[TaskClassification]:
        """Classify multiple goal strings at once."""
        return [self.classify(g) for g in goals]


# ---------------------------------------------------------------------------
# SubtaskDecomposer
# ---------------------------------------------------------------------------


class SubtaskDecomposer:
    """Decompose a task into an ordered list of :class:`Subtask` objects.

    Strategy
    --------
    1. Select the template for the primary category.
    2. Parse *goal_phrases* from the raw goal by splitting on conjunctions.
    3. For each goal phrase that doesn't obviously map to an existing template
       step, insert a custom subtask derived from the phrase.
    4. If a secondary category exists and its template adds non-redundant
       steps (e.g. debugging steps appended to coding), include those.
    5. Re-index and resolve sequential dependencies.

    Example::

        dec = SubtaskDecomposer()
        subtasks = dec.decompose("Fix the auth bug and then commit", classification)
    """

    def decompose(
        self,
        goal: str,
        classification: TaskClassification,
        task: Optional[Dict[str, Any]] = None,
    ) -> List[Subtask]:
        """Decompose *goal* into an ordered list of :class:`Subtask` objects.

        Args:
            goal: The normalised goal string.
            classification: Result of :class:`TaskClassifier`.
            task: Optional task dict; if it contains a ``"steps"`` key those
                  are used directly instead of template expansion.

        Returns:
            Ordered list of :class:`Subtask` instances.
        """
        # If the caller already provided explicit steps, honour them.
        if task and task.get("steps"):
            return self._from_explicit_steps(task["steps"], classification)

        primary = classification.category
        template = _TEMPLATES.get(primary, _TEMPLATES["coding"])
        goal_phrases = self._split_goal(goal)

        subtasks = self._expand_template(template, goal, primary)
        subtasks = self._inject_goal_phrases(subtasks, goal_phrases, primary)

        # Optionally blend secondary category steps
        if (
            classification.secondary
            and classification.secondary != primary
            and classification.confidence < 0.6
        ):
            secondary_template = _TEMPLATES.get(classification.secondary, [])
            secondary_steps = self._expand_template(
                secondary_template, goal, classification.secondary
            )
            subtasks = self._merge_secondary(subtasks, secondary_steps)

        subtasks = self._reindex_and_link(subtasks)
        return subtasks

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _split_goal(self, goal: str) -> List[str]:
        """Split the goal on common conjunctions into individual phrases."""
        parts = _GOAL_SPLIT_RE.split(goal)
        return [p.strip() for p in parts if p.strip()]

    def _expand_template(
        self,
        template: List[Dict[str, Any]],
        goal: str,
        category: str,
    ) -> List[Subtask]:
        """Instantiate template step dicts into Subtask objects."""
        result = []
        for i, tmpl in enumerate(template):
            desc = tmpl["description"].format(goal=goal)
            result.append(
                Subtask(
                    subtask_id=str(uuid.uuid4()),
                    name=tmpl["name"],
                    description=desc,
                    category=category,
                    agent=tmpl["agent"],
                    tools=list(tmpl["tools"]),
                    priority=tmpl["priority"],
                    context_hints=[tmpl["context_hint"]] if tmpl.get("context_hint") else [],
                    index=i,
                )
            )
        return result

    def _inject_goal_phrases(
        self,
        subtasks: List[Subtask],
        phrases: List[str],
        category: str,
    ) -> List[Subtask]:
        """Inject explicit goal phrases as custom subtasks where appropriate.

        For multi-phrase goals (e.g. "implement X then commit"), inserts a
        custom subtask for each phrase that isn't already covered by the
        template's ``name`` field.

        Only injects if there are more than one phrase (single-phrase goals
        are fully covered by the template).
        """
        if len(phrases) <= 1:
            return subtasks

        template_names = {s.name.lower() for s in subtasks}
        extra: List[Subtask] = []

        for phrase in phrases:
            # Check if this phrase already maps to a template step.
            phrase_lower = phrase.lower()
            already_covered = any(
                kw in phrase_lower
                for kw in {s.name.split()[0].lower() for s in subtasks}
            )
            if not already_covered:
                inferred_agent = _route_by_keywords(phrase)
                extra.append(
                    Subtask(
                        subtask_id=str(uuid.uuid4()),
                        name=phrase[:60],
                        description=phrase,
                        category=category,
                        agent=inferred_agent,
                        tools=_default_tools_for_agent(inferred_agent),
                        priority="high",
                        context_hints=[],
                        index=0,  # re-indexed later
                        metadata={"source": "goal_phrase"},
                    )
                )

        if extra:
            # Insert custom steps just before the final "validate" step if present.
            insert_pos = len(subtasks)
            for i, s in enumerate(subtasks):
                if "validate" in s.name.lower():
                    insert_pos = i
                    break
            subtasks = subtasks[:insert_pos] + extra + subtasks[insert_pos:]

        return subtasks

    def _merge_secondary(
        self,
        primary_steps: List[Subtask],
        secondary_steps: List[Subtask],
    ) -> List[Subtask]:
        """Append non-redundant secondary-category steps to the primary list."""
        primary_names = {s.name.lower() for s in primary_steps}
        additions = [s for s in secondary_steps if s.name.lower() not in primary_names]
        # Insert secondary steps before the final validation step.
        insert_pos = len(primary_steps)
        for i, s in enumerate(primary_steps):
            if "validate" in s.name.lower():
                insert_pos = i
                break
        return primary_steps[:insert_pos] + additions + primary_steps[insert_pos:]

    def _reindex_and_link(self, subtasks: List[Subtask]) -> List[Subtask]:
        """Assign sequential indices and resolve linear depends_on chains."""
        for i, st in enumerate(subtasks):
            st.index = i
            st.depends_on = [subtasks[j].subtask_id for j in range(i)]
        return subtasks

    def _from_explicit_steps(
        self,
        steps: List[Any],
        classification: TaskClassification,
    ) -> List[Subtask]:
        """Convert caller-provided step list into Subtask objects."""
        result = []
        for i, raw in enumerate(steps):
            if isinstance(raw, str):
                raw = {"name": raw, "description": raw}
            agent = raw.get("agent") or _route_by_keywords(raw.get("description", raw.get("name", "")))
            result.append(
                Subtask(
                    subtask_id=str(uuid.uuid4()),
                    name=raw.get("name", f"step_{i + 1}"),
                    description=raw.get("description", raw.get("name", "")),
                    category=classification.category,
                    agent=agent,
                    tools=raw.get("tools", _default_tools_for_agent(agent)),
                    priority=raw.get("priority", "medium"),
                    context_hints=raw.get("context_hints", []),
                    index=i,
                    metadata=raw.get("metadata", {}),
                )
            )
        result = self._reindex_and_link(result)
        return result


# ---------------------------------------------------------------------------
# ExecutionPlanGenerator
# ---------------------------------------------------------------------------


class ExecutionPlanGenerator:
    """Assemble a :class:`ExecutionPlan` from classification and subtasks.

    Computes:
    * **Complexity** from subtask count and priority distribution.
    * **Context hints** by aggregating unique hints from all subtasks.
    * Plan-level metadata (ID, timestamps, goal/prompt fields).

    Example::

        gen = ExecutionPlanGenerator()
        plan = gen.generate(goal, raw_prompt, classification, subtasks)
    """

    def generate(
        self,
        goal: str,
        raw_prompt: str,
        classification: TaskClassification,
        subtasks: List[Subtask],
    ) -> ExecutionPlan:
        """Build and return an :class:`ExecutionPlan`.

        Args:
            goal: Normalised goal string.
            raw_prompt: Original user input.
            classification: Result of :class:`TaskClassifier`.
            subtasks: Ordered list from :class:`SubtaskDecomposer`.

        Returns:
            A fully populated :class:`ExecutionPlan`.
        """
        complexity = self._compute_complexity(subtasks)
        context_hints = self._aggregate_context_hints(subtasks)

        return ExecutionPlan(
            plan_id=str(uuid.uuid4()),
            goal=goal,
            raw_prompt=raw_prompt,
            classification=classification,
            subtasks=subtasks,
            complexity=complexity,
            context_hints=context_hints,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_complexity(self, subtasks: List[Subtask]) -> str:
        n = len(subtasks)
        high_count = sum(1 for s in subtasks if s.priority == "high")
        if n <= 2 or high_count <= 1:
            return "low"
        if n <= 5 or high_count <= 3:
            return "medium"
        return "high"

    def _aggregate_context_hints(self, subtasks: List[Subtask]) -> List[str]:
        seen: set = set()
        hints: List[str] = []
        for st in subtasks:
            for h in st.context_hints:
                if h and h not in seen:
                    seen.add(h)
                    hints.append(h)
        return hints


# ---------------------------------------------------------------------------
# Helper functions (package-internal)
# ---------------------------------------------------------------------------

# Routing and alias data now live in core.categories — single source of truth.
# These thin wrappers preserve the existing internal call sites.
from core.categories import (  # noqa: E402
    KEYWORD_AGENT_MAP   as _KEYWORD_AGENT_MAP,
    AGENT_DEFAULT_TOOLS as _AGENT_DEFAULT_TOOLS,
    route_by_keywords        as _route_by_keywords,
    default_tools_for_agent  as _default_tools_for_agent,
)


# ---------------------------------------------------------------------------
# TaskPlanner — top-level public API
# ---------------------------------------------------------------------------


class TaskPlanner:
    """Orchestrate classification, decomposition, and plan generation.

    This is the single entry point for the planning subsystem.  The
    :class:`ConcretePlannerAgent` delegates to this class rather than
    maintaining its own heuristics.

    Attributes:
        classifier: :class:`TaskClassifier` instance.
        decomposer: :class:`SubtaskDecomposer` instance.
        generator: :class:`ExecutionPlanGenerator` instance.

    Example::

        planner = TaskPlanner()
        plan = planner.plan({
            "goal": "Fix the failing auth test and then commit the fix",
            "raw_prompt": "...",
        })
        print(plan.summary())
        for step in plan.ordered_subtasks():
            print(step.name, "→", step.agent, step.tools)
    """

    def __init__(self) -> None:
        self.classifier = TaskClassifier()
        self.decomposer = SubtaskDecomposer()
        self.generator = ExecutionPlanGenerator()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(
        self,
        task: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> ExecutionPlan:
        """Produce a complete :class:`ExecutionPlan` for *task*.

        Processing pipeline:
        1. Extract and normalise ``goal`` from *task*.
        2. Classify the goal → :class:`TaskClassification`.
        3. Decompose into subtasks → ``List[Subtask]``.
        4. Generate the :class:`ExecutionPlan`.

        Args:
            task: Dict with at least a ``"goal"`` key.  May optionally
                  contain ``"raw_prompt"`` and ``"steps"``.
            context: Optional context payload (reserved for future use).

        Returns:
            A fully populated :class:`ExecutionPlan`.
        """
        goal = task.get("goal") or task.get("raw_prompt", "")
        raw_prompt = task.get("raw_prompt", goal)
        goal = self._normalise_goal(goal)

        classification = self.classifier.classify(goal)
        subtasks = self.decomposer.decompose(goal, classification, task)
        plan = self.generator.generate(goal, raw_prompt, classification, subtasks)
        return plan

    def classify(self, goal: str) -> TaskClassification:
        """Classify *goal* without generating a full plan.

        Convenience wrapper around :class:`TaskClassifier`.

        Args:
            goal: The task goal string.

        Returns:
            :class:`TaskClassification`.
        """
        return self.classifier.classify(self._normalise_goal(goal))

    def reclassify_and_replan(
        self,
        plan: ExecutionPlan,
        hint: str,
    ) -> ExecutionPlan:
        """Re-run the planning pipeline after injecting a category hint.

        Useful when the supervisor detects that initial classification was
        wrong mid-execution.

        Args:
            plan: The original plan.
            hint: A category string to force (overrides classification).

        Returns:
            A new :class:`ExecutionPlan` with the updated category.
        """
        if hint not in TASK_CATEGORIES:
            raise ValueError(f"Invalid category hint '{hint}'.  Must be one of {TASK_CATEGORIES}.")
        forced_clf = TaskClassification(
            category=hint,
            confidence=1.0,
            secondary=plan.classification.category,
            signals=["forced_reclassify"],
            scores={c: (1.0 if c == hint else 0.0) for c in TASK_CATEGORIES},
        )
        subtasks = self.decomposer.decompose(plan.goal, forced_clf)
        return self.generator.generate(plan.goal, plan.raw_prompt, forced_clf, subtasks)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_goal(goal: str) -> str:
        """Strip whitespace and truncate to a reasonable length."""
        goal = goal.strip()
        if len(goal) > 500:
            goal = goal[:500]
        return goal
