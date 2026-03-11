from __future__ import annotations
from abc import abstractmethod
from typing import Any, Dict, List, Optional

from agents.base_agent import BaseAgent


class SupervisorAgent(BaseAgent):
    """Abstract base class for the Supervisor Agent.

    The Supervisor is the top-level orchestrator. It is responsible for
    understanding the user's intent, delegating to the Planner, monitoring
    pipeline progress, and triggering recovery strategies on failure.
    """

    @abstractmethod
    def delegate(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Delegate a parsed task to the Planner Agent for decomposition.

        Args:
            task: The structured task parsed from the user prompt.

        Returns:
            The structured plan produced by the Planner.
        """
        ...

    @abstractmethod
    def monitor(self, pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor the execution state of an active pipeline.

        Args:
            pipeline_state: Current state snapshot of the running pipeline.

        Returns:
            Updated state dict with monitoring annotations.
        """
        ...

    @abstractmethod
    def recover(self, failure: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Trigger a recovery strategy when a pipeline step fails.

        Args:
            failure: Structured failure report including error details and step info.
            pipeline_state: Current state of the pipeline at the time of failure.

        Returns:
            A recovery action dict (retry, modify pipeline, switch model, or abort).
        """
        ...

    @abstractmethod
    def parse_prompt(self, prompt: str) -> Dict[str, Any]:
        """Parse a raw user prompt into a structured task definition.

        Args:
            prompt: The raw user input string.

        Returns:
            A structured task dict with goal, constraints, and metadata.
        """
        ...


# ────────────────────────────────────────────────────────────────────────────

"""concrete_supervisor.py — Concrete SupervisorAgent implementation.

The SupervisorAgent is the top-level orchestrator of the Sentinel hierarchy.
It is responsible for:

1. Parsing a raw user prompt into a structured task.
2. Delegating the task to the PlannerAgent via a ``delegate`` action.
3. Monitoring pipeline progress and injecting ``monitor`` checkpoints.
4. Triggering recovery strategies when a pipeline step reports failure.

Design contract
---------------
* The Supervisor **never** calls tools directly.
* All side effects are expressed as :class:`~agents.agent_action.AgentAction`
  instances returned from :py:meth:`run`.
* The ExecutionEngine (caller) is solely responsible for dispatching actions.
"""


import re
import traceback
import uuid
from typing import Any, Dict, List, Optional

from agents.agent_action import AgentAction

# ---------------------------------------------------------------------------
# Complexity heuristics
# ---------------------------------------------------------------------------

_HIGH_COMPLEXITY_KEYWORDS = frozenset(
    {
        "refactor",
        "architecture",
        "migrate",
        "optimise",
        "optimize",
        "benchmark",
        "security audit",
        "upgrade",
        "pipeline",
    }
)
_LOW_COMPLEXITY_KEYWORDS = frozenset(
    {"explain", "summarise", "summarize", "describe", "what is", "show", "list"}
)


def _estimate_complexity(goal: str) -> str:
    lower = goal.lower()
    if any(k in lower for k in _HIGH_COMPLEXITY_KEYWORDS):
        return "high"
    if any(k in lower for k in _LOW_COMPLEXITY_KEYWORDS):
        return "low"
    return "medium"


# ---------------------------------------------------------------------------
# Concrete implementation
# ---------------------------------------------------------------------------


class ConcreteSupervisorAgent(SupervisorAgent):
    """Concrete top-level orchestrator agent.

    Attributes:
        name: Registry identifier for this agent.
        max_retries: Maximum number of recovery attempts before aborting.
    """

    name = "supervisor"

    def __init__(self, max_retries: int = 2) -> None:
        self.max_retries = max_retries

    # ------------------------------------------------------------------
    # BaseAgent — required overrides
    # ------------------------------------------------------------------

    def run(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate the task and return generated actions.

        The Supervisor:
        1. Validates the incoming task structure.
        2. Generates a ``delegate`` action aimed at the PlannerAgent.
        3. Generates a ``message`` action summarising its intent.

        Args:
            task: Structured task dict (must contain at least ``"goal"``).
            context: Context payload from ContextBuilder.

        Returns:
            ``{"status": "ok", "actions": [AgentAction, ...], "task": task}``
        """
        step_id = task.get("step_id") or str(uuid.uuid4())
        task.setdefault("step_id", step_id)
        task.setdefault("complexity", _estimate_complexity(task.get("goal", "")))

        actions: List[AgentAction] = [
            AgentAction.message(
                f"[Supervisor] Received task: {task.get('goal', '(no goal)')}",
                agent=self.name,
                step_id=step_id,
            ),
            AgentAction.delegate(
                target_agent="planner",
                task=task,
                agent=self.name,
                step_id=step_id,
                rationale="Delegating to PlannerAgent for decomposition.",
            ),
        ]
        return {"status": "ok", "actions": actions, "task": task}

    def validate_output(self, output: Dict[str, Any]) -> bool:
        """Verify that the output contains a non-empty actions list."""
        return (
            isinstance(output, dict)
            and output.get("status") == "ok"
            and isinstance(output.get("actions"), list)
            and len(output["actions"]) > 0
        )

    def handle_error(self, error: Exception, task: Dict[str, Any]) -> Dict[str, Any]:
        """Emit an abort action carrying the traceback."""
        step_id = task.get("step_id", "unknown")
        tb = traceback.format_exc()
        actions = [
            AgentAction.abort(
                reason=f"SupervisorAgent error: {error}\n{tb}",
                agent=self.name,
                step_id=step_id,
            )
        ]
        return {"status": "error", "actions": actions, "error": str(error), "task": task}

    def describe(self) -> str:
        return (
            "SupervisorAgent: top-level orchestrator.  Parses user prompts, "
            "delegates to the PlannerAgent, monitors pipeline progress, and "
            "initiates recovery strategies on failure."
        )

    # ------------------------------------------------------------------
    # SupervisorAgent — abstract method implementations
    # ------------------------------------------------------------------

    def parse_prompt(self, prompt: str) -> Dict[str, Any]:
        """Parse a raw user prompt into a structured task dict.

        Extracts the first sentence as the goal, detects an optional
        ``[context: ...]`` annotation, and estimates complexity.

        Args:
            prompt: Raw user input string.

        Returns:
            ``{"goal": str, "raw_prompt": str, "complexity": str, "constraints": list,
               "step_id": str}``
        """
        prompt = prompt.strip()

        # First sentence → goal
        goal_match = re.match(r"([^.!?\n]+)[.!?\n]?", prompt)
        goal = goal_match.group(1).strip() if goal_match else prompt[:200]

        # Optional inline constraints: [constraint: ...]
        constraints = re.findall(r"\[constraint:\s*([^\]]+)\]", prompt, re.IGNORECASE)

        return {
            "goal": goal,
            "raw_prompt": prompt,
            "complexity": _estimate_complexity(goal),
            "constraints": constraints,
            "step_id": str(uuid.uuid4()),
        }

    def delegate(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a ``delegate`` action targeting the PlannerAgent.

        Args:
            task: Structured task dict.

        Returns:
            ``{"actions": [AgentAction], "target": "planner"}``
        """
        action = AgentAction.delegate(
            target_agent="planner",
            task=task,
            agent=self.name,
            step_id=task.get("step_id", ""),
            rationale="Supervisor delegating task to PlannerAgent.",
        )
        return {"actions": [action], "target": "planner"}

    def monitor(self, pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Inspect a pipeline snapshot and emit monitoring annotations.

        Generates a ``message`` action summarising current pipeline progress.
        If any step has ``"status": "failed"``, generates an additional
        ``decision`` action flagging the failure for recovery consideration.

        Args:
            pipeline_state: Dict with ``"steps"`` list and ``"current_step"`` index.

        Returns:
            Pipeline state dict extended with ``"monitor_actions"`` key.
        """
        step_id = pipeline_state.get("step_id", "")
        steps: List[Dict] = pipeline_state.get("steps", [])
        current = pipeline_state.get("current_step", 0)
        total = len(steps)

        monitor_actions: List[AgentAction] = [
            AgentAction.message(
                f"[Monitor] Step {current}/{total} in progress.",
                agent=self.name,
                step_id=step_id,
            )
        ]

        failed = [s for s in steps if s.get("status") == "failed"]
        if failed:
            failed_names = [s.get("name", "?") for s in failed]
            monitor_actions.append(
                AgentAction.decision(
                    choice="recover",
                    options=["recover", "abort", "skip"],
                    rationale=f"Failed steps detected: {failed_names}",
                    agent=self.name,
                    step_id=step_id,
                )
            )

        pipeline_state["monitor_actions"] = [a.to_dict() for a in monitor_actions]
        return pipeline_state

    def recover(
        self, failure: Dict[str, Any], pipeline_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate recovery actions for a failed pipeline step.

        Recovery strategy:
        * Attempt retry up to ``self.max_retries`` times.
        * If retries are exhausted, emit an ``abort`` action.

        Args:
            failure: ``{"step": step_dict, "error": str, "attempt": int}``
            pipeline_state: Current pipeline state.

        Returns:
            ``{"actions": [AgentAction], "strategy": str}``
        """
        attempt = failure.get("attempt", 1)
        step = failure.get("step", {})
        step_id = step.get("step_id", pipeline_state.get("step_id", ""))
        error_msg = failure.get("error", "unknown error")

        if attempt <= self.max_retries:
            action = AgentAction.delegate(
                target_agent=step.get("agent", "planner"),
                task={**step, "attempt": attempt + 1, "step_id": step_id},
                agent=self.name,
                step_id=step_id,
                rationale=f"Recovery attempt {attempt}/{self.max_retries} for: {error_msg}",
            )
            return {"actions": [action], "strategy": "retry"}

        action = AgentAction.abort(
            reason=f"Max retries ({self.max_retries}) exhausted for step '{step.get('name', '?')}': {error_msg}",
            agent=self.name,
            step_id=step_id,
        )
        return {"actions": [action], "strategy": "abort"}
