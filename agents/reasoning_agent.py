from __future__ import annotations
"""reasoning_agent.py — Sentinel ReasoningAgent.

Responsible for structured reasoning, analysis, explanation, and decision-
making steps.  This agent works primarily with ``message`` and ``decision``
actions; it may also emit ``tool_call`` actions for ``read_file`` or
``search_code`` when it needs artefacts to reason over.

Registered name: ``"reasoning"``
"""


import traceback
import uuid
from typing import Any, Dict, List, Optional

from agents.agent_action import AgentAction
from agents.base_agent import BaseAgent


class ReasoningAgent(BaseAgent):
    """Specialist agent for analysis, explanation, and decision-making.

    This agent does not modify files or run external processes.  Its primary
    outputs are structured ``message`` and ``decision`` actions that record
    its reasoning chain for downstream agents or the user interface.

    Supported task actions
    ----------------------
    ``"analyse"`` / ``"analyze"``
        Emit a reasoning message and an optional decision.
    ``"explain"``
        Emit an explanation message.
    ``"compare"``
        Emit a decision action choosing between provided options.
    ``"summarise"`` / ``"summarize"``
        Emit a summary message from provided content.
    ``"validate"``
        Verify that a prior step's output meets stated criteria.
    ``"read_for_context"``
        Generate a tool_call to read a file before reasoning.
    """

    name = "reasoning"

    # ------------------------------------------------------------------
    # BaseAgent — required overrides
    # ------------------------------------------------------------------

    def run(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Produce reasoning actions for the given task.

        Args:
            task: Step dict from PlannerAgent.
            context: Context payload (may include prior results for analysis).

        Returns:
            ``{"status": "ok", "actions": [AgentAction, ...], "task": task}``
        """
        step_id = task.get("step_id", str(uuid.uuid4()))
        actions = self._generate_actions(task, context, step_id)
        return {"status": "ok", "actions": actions, "task": task}

    def validate_output(self, output: Dict[str, Any]) -> bool:
        return (
            isinstance(output, dict)
            and output.get("status") == "ok"
            and isinstance(output.get("actions"), list)
        )

    def handle_error(self, error: Exception, task: Dict[str, Any]) -> Dict[str, Any]:
        step_id = task.get("step_id", "unknown")
        actions = [
            AgentAction.abort(
                reason=f"ReasoningAgent error: {error}\n{traceback.format_exc()}",
                agent=self.name,
                step_id=step_id,
            )
        ]
        return {"status": "error", "actions": actions, "error": str(error), "task": task}

    def describe(self) -> str:
        return (
            "ReasoningAgent: performs structured analysis, explanation, and "
            "decision-making.  Emits message and decision actions; may also "
            "request read_file or search_code to gather context before reasoning."
        )

    # ------------------------------------------------------------------
    # Action generation
    # ------------------------------------------------------------------

    def _generate_actions(
        self,
        task: Dict[str, Any],
        context: Dict[str, Any],
        step_id: str,
    ) -> List[AgentAction]:
        raw_action = task.get("action", "analyse").lower().replace("analyze", "analyse").replace("summarize", "summarise")
        actions: List[AgentAction] = []

        if raw_action == "analyse":
            subject = task.get("subject", task.get("description", "the task"))
            criteria = task.get("criteria", [])
            reasoning = (
                f"[ReasoningAgent] Analysing: {subject}."
                + (f"  Criteria: {', '.join(criteria)}." if criteria else "")
            )
            actions.append(AgentAction.message(reasoning, agent=self.name, step_id=step_id))

            options = task.get("options", [])
            if options:
                chosen = options[0]  # Placeholder: first option; real LLM call replaces this.
                actions.append(
                    AgentAction.decision(
                        choice=chosen,
                        options=options,
                        rationale=f"Selected '{chosen}' as the most appropriate option for: {subject}",
                        agent=self.name,
                        step_id=step_id,
                    )
                )

        elif raw_action == "explain":
            subject = task.get("subject", task.get("description", "this topic"))
            actions.append(
                AgentAction.message(
                    f"[ReasoningAgent] Explanation requested for: {subject}",
                    agent=self.name,
                    step_id=step_id,
                )
            )

        elif raw_action == "compare":
            options = task.get("options", [])
            if len(options) < 2:
                actions.append(
                    AgentAction.message(
                        "[ReasoningAgent] compare action requires at least two options.",
                        agent=self.name,
                        step_id=step_id,
                    )
                )
            else:
                chosen = options[0]  # Placeholder.
                actions.append(
                    AgentAction.decision(
                        choice=chosen,
                        options=options,
                        rationale=f"Compared options and selected '{chosen}'.",
                        agent=self.name,
                        step_id=step_id,
                    )
                )

        elif raw_action == "summarise":
            content = task.get("content", context.get("text", ""))
            summary = content[:500] + ("…" if len(content) > 500 else "")
            actions.append(
                AgentAction.message(
                    f"[ReasoningAgent] Summary: {summary}",
                    agent=self.name,
                    step_id=step_id,
                )
            )

        elif raw_action == "validate":
            criteria = task.get("criteria", [])
            prior_output = task.get("prior_output", {})
            passed = bool(prior_output) and all(
                str(c).lower() in str(prior_output).lower() for c in criteria
            ) if criteria else bool(prior_output)
            verdict = "PASSED" if passed else "FAILED"
            actions.append(
                AgentAction.message(
                    f"[ReasoningAgent] Validation {verdict} for criteria: {criteria}",
                    agent=self.name,
                    step_id=step_id,
                )
            )
            if not passed:
                actions.append(
                    AgentAction.decision(
                        choice="retry",
                        options=["retry", "abort", "accept"],
                        rationale=f"Validation failed — criteria not met: {criteria}",
                        agent=self.name,
                        step_id=step_id,
                    )
                )

        elif raw_action == "read_for_context":
            path = task.get("path", "")
            if path:
                actions.append(
                    AgentAction.tool_call(
                        tool="read_file",
                        params={"path": path},
                        agent=self.name,
                        step_id=step_id,
                        rationale=f"Read '{path}' to gather context before reasoning.",
                    )
                )
            else:
                actions.append(
                    AgentAction.message(
                        "[ReasoningAgent] 'read_for_context' requires a 'path' field.",
                        agent=self.name,
                        step_id=step_id,
                    )
                )

        else:
            actions.append(
                AgentAction.message(
                    f"[ReasoningAgent] Unknown action '{raw_action}'.",
                    agent=self.name,
                    step_id=step_id,
                )
            )

        return actions
