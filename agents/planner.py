from __future__ import annotations
from abc import abstractmethod
from typing import Any, Dict, List

from agents.base_agent import BaseAgent


class PlannerAgent(BaseAgent):
    """Abstract base class for the Planner Agent.

    The Planner receives a structured task from the Supervisor and is
    responsible for decomposing it into an ordered list of discrete,
    assignable steps that the Pipeline Generator can transform into an
    executable pipeline.
    """

    @abstractmethod
    def decompose(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Decompose a task into an ordered list of plan steps.

        Args:
            task: The structured task dict from the Supervisor.

        Returns:
            A list of step dicts, each containing a description, assigned
            agent type, required tools, and dependencies.
        """
        ...

    @abstractmethod
    def assign_agents(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Assign the appropriate specialist agent type to each plan step.

        Args:
            steps: The list of decomposed steps produced by decompose().

        Returns:
            The same steps list with an 'agent' key added to each step.
        """
        ...

    @abstractmethod
    def resolve_dependencies(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify and annotate ordering dependencies between plan steps.

        Args:
            steps: The list of agent-assigned steps.

        Returns:
            Steps list with dependency references added to each step.
        """
        ...

    @abstractmethod
    def estimate_complexity(self, task: Dict[str, Any]) -> str:
        """Estimate the complexity level of a task.

        Args:
            task: The structured task dict.

        Returns:
            A complexity label: 'low', 'medium', or 'high'.
        """
        ...


# ────────────────────────────────────────────────────────────────────────────

"""concrete_planner.py — Concrete PlannerAgent implementation.

The PlannerAgent receives a structured task from the SupervisorAgent and
decomposes it into an ordered list of plan steps, assigns the most suitable
specialist agent to each step, resolves inter-step dependencies, and
estimates overall task complexity.

Design contract
---------------
* The Planner **never** calls tools directly.
* All decomposition and classification is delegated to
  :class:`~tasks.task_manager.TaskPlanner`.
* Output is a list of step dicts plus supporting :class:`AgentAction` instances
  returned inside the ``run()`` result.
"""


import traceback
import uuid
from typing import Any, Dict, List

from agents.agent_action import AgentAction
from tasks.task_manager import (
    ExecutionPlan,
    Subtask,
    TaskPlanner,
    TASK_CATEGORIES,
)

_COMPLEXITY_MAP = {"low": 1, "medium": 2, "high": 3}


def _complexity_from_step_count(n: int) -> str:
    if n <= 2:
        return "low"
    if n <= 5:
        return "medium"
    return "high"


# ---------------------------------------------------------------------------
# Concrete implementation
# ---------------------------------------------------------------------------


class ConcretePlannerAgent(PlannerAgent):
    """Concrete task decomposition and agent-assignment agent.

    Delegates all classification, decomposition, and plan generation to
    :class:`~tasks.task_manager.TaskPlanner`.

    Attributes:
        name: Registry identifier for this agent.
        task_planner: The :class:`~tasks.task_manager.TaskPlanner`
            instance used internally.
    """

    name = "planner"

    def __init__(self) -> None:
        self.task_planner = TaskPlanner()

    # ------------------------------------------------------------------
    # BaseAgent — required overrides
    # ------------------------------------------------------------------

    def run(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Decompose *task* into an ordered plan and generate delegate actions.

        Delegates to :class:`~tasks.task_manager.TaskPlanner` for the
        full classify → decompose → plan pipeline, then wraps the resulting
        :class:`~tasks.task_manager.ExecutionPlan` as ``AgentAction``
        instances.

        Args:
            task: Structured task dict (must contain ``"goal"``).
            context: Context payload from ContextBuilder.

        Returns:
            ``{"status": "ok", "plan": [step_dict, ...], "actions": [AgentAction, ...],
               "execution_plan": ExecutionPlan}``
        """
        step_id = task.get("step_id", str(uuid.uuid4()))
        task.setdefault("step_id", step_id)

        execution_plan: ExecutionPlan = self.task_planner.plan(task, context)

        # Convert Subtask objects → plain step dicts for callers that expect dicts.
        steps = [s.to_dict() for s in execution_plan.ordered_subtasks()]

        classification = execution_plan.classification
        actions: List[AgentAction] = [
            AgentAction.message(
                f"[Planner] Task classified as '{classification.category}' "
                f"(confidence {classification.confidence:.0%}).  "
                f"Decomposed into {len(steps)} step(s).",
                agent=self.name,
                step_id=step_id,
            )
        ]

        for step in steps:
            actions.append(
                AgentAction.delegate(
                    target_agent=step["agent"],
                    task=step,
                    agent=self.name,
                    step_id=step["subtask_id"],
                    rationale=(
                        f"Assign step '{step['name']}' "
                        f"(priority={step['priority']}) to {step['agent']} agent."
                    ),
                )
            )

        return {
            "status": "ok",
            "plan": steps,
            "actions": actions,
            "task": task,
            "execution_plan": execution_plan,
            "classification": classification.to_dict(),
        }

    def validate_output(self, output: Dict[str, Any]) -> bool:
        return (
            isinstance(output, dict)
            and output.get("status") == "ok"
            and isinstance(output.get("plan"), list)
        )

    def handle_error(self, error: Exception, task: Dict[str, Any]) -> Dict[str, Any]:
        step_id = task.get("step_id", "unknown")
        tb = traceback.format_exc()
        actions = [
            AgentAction.abort(
                reason=f"PlannerAgent error: {error}\n{tb}",
                agent=self.name,
                step_id=step_id,
            )
        ]
        return {"status": "error", "actions": actions, "error": str(error), "task": task}

    def describe(self) -> str:
        return (
            "PlannerAgent: classifies tasks, decomposes goals into ordered subtasks, "
            "assigns the most appropriate specialist agent to each subtask, and "
            "resolves inter-step dependencies via TaskPlanner."
        )

    # ------------------------------------------------------------------
    # PlannerAgent — abstract method implementations
    # ------------------------------------------------------------------

    def decompose(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Decompose *task* into subtask dicts via :class:`TaskPlanner`.

        Args:
            task: Structured task dict.

        Returns:
            List of subtask dicts (serialised :class:`~tasks.task_manager.Subtask`).
        """
        clf = self.task_planner.classify(task.get("goal", ""))
        subtasks = self.task_planner.decomposer.decompose(
            task.get("goal", ""), clf, task
        )
        return [s.to_dict() for s in subtasks]

    def assign_agents(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ensure every step dict has an ``"agent"`` key.

        Steps produced by :meth:`decompose` already carry agent assignments
        from :class:`~tasks.task_manager.SubtaskDecomposer`; this
        method performs a fallback assignment for any step missing one.

        Args:
            steps: List of step dicts.

        Returns:
            Same list with ``"agent"`` populated on every step.
        """
        from tasks.task_manager import _route_by_keywords
        for step in steps:
            if not step.get("agent"):
                step["agent"] = _route_by_keywords(
                    step.get("description", step.get("name", ""))
                )
        return steps

    def resolve_dependencies(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ensure sequential ``depends_on`` chains are populated.

        Steps from :meth:`decompose` already have dependencies resolved;
        this method is a safety net for manually constructed step lists.

        Args:
            steps: List of step dicts.

        Returns:
            Steps with ``"depends_on"`` lists populated.
        """
        id_key = "subtask_id" if steps and "subtask_id" in steps[0] else "step_id"
        for i, step in enumerate(steps):
            if not step.get("depends_on"):
                step["depends_on"] = [steps[j][id_key] for j in range(i)]
        return steps

    def estimate_complexity(self, task: Dict[str, Any]) -> str:
        """Estimate task complexity via :class:`TaskPlanner`.

        Args:
            task: Structured task dict.

        Returns:
            ``"low"``, ``"medium"``, or ``"high"``.
        """
        declared = task.get("complexity", "")
        if declared in _COMPLEXITY_MAP:
            return declared
        plan = self.task_planner.plan(task)
        return plan.complexity
