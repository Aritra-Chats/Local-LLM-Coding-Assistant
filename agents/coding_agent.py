from __future__ import annotations
"""coding_agent.py — Sentinel CodingAgent.

Responsible for all source-code generation, editing, and search tasks.
Generates ``tool_call`` actions for ``read_file``, ``write_file``, and
``search_code`` — it never invokes tools directly.

Registered name: ``"coding"``
"""


import traceback
import uuid
from typing import Any, Dict, List, Optional

from agents.agent_action import AgentAction
from agents.base_agent import BaseAgent

# Tools this agent is permitted to request.
_ALLOWED_TOOLS = frozenset({"read_file", "write_file", "search_code"})


class CodingAgent(BaseAgent):
    """Specialist agent for source-code generation and file manipulation.

    Generates ``tool_call`` actions that the ExecutionEngine dispatches to
    the ToolRegistry.  Does **not** execute tools itself.

    Supported task keys
    -------------------
    ``"action"``
        One of ``"read"``, ``"write"``, ``"search"``.
    ``"path"``
        Target file path (for ``read``/``write``).
    ``"content"``
        Text to write (for ``write``).
    ``"query"``
        Search term (for ``search``).
    ``"glob"``
        File glob filter (for ``search``).
    ``"start_line"`` / ``"end_line"``
        Optional line range (for ``read``).
    """

    name = "coding"

    # ------------------------------------------------------------------
    # BaseAgent — required overrides
    # ------------------------------------------------------------------

    def run(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate code-related actions for the given task.

        Args:
            task: Step dict from PlannerAgent.  Must contain ``"action"``.
            context: Context payload from ContextBuilder.

        Returns:
            ``{"status": "ok", "actions": [AgentAction, ...], "task": task}``
        """
        step_id = task.get("step_id", str(uuid.uuid4()))
        actions = self._generate_actions(task, step_id)
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
                reason=f"CodingAgent error: {error}\n{traceback.format_exc()}",
                agent=self.name,
                step_id=step_id,
            )
        ]
        return {"status": "error", "actions": actions, "error": str(error), "task": task}

    def describe(self) -> str:
        return (
            "CodingAgent: generates tool_call actions for reading files, "
            "writing / editing source code, and searching the codebase.  "
            "Tools: read_file, write_file, search_code."
        )

    # ------------------------------------------------------------------
    # Action generation
    # ------------------------------------------------------------------

    def _generate_actions(self, task: Dict[str, Any], step_id: str) -> List[AgentAction]:
        """Map task fields to concrete tool_call AgentActions."""
        action_type = task.get("action", "read").lower()
        actions: List[AgentAction] = []

        if action_type == "read":
            params: Dict[str, Any] = {"path": task.get("path", "")}
            if task.get("start_line"):
                params["start_line"] = int(task["start_line"])
            if task.get("end_line"):
                params["end_line"] = int(task["end_line"])
            if task.get("encoding"):
                params["encoding"] = task["encoding"]

            actions.append(
                AgentAction.tool_call(
                    tool="read_file",
                    params=params,
                    agent=self.name,
                    step_id=step_id,
                    rationale=f"Read source file: {params['path']}",
                )
            )

        elif action_type == "write":
            params = {
                "path": task.get("path", ""),
                "content": task.get("content", ""),
                "mode": task.get("mode", "overwrite"),
            }
            actions.append(
                AgentAction.tool_call(
                    tool="write_file",
                    params=params,
                    agent=self.name,
                    step_id=step_id,
                    rationale=f"Write source file: {params['path']}",
                )
            )

        elif action_type == "search":
            params = {
                "query": task.get("query", ""),
                "path": task.get("path", "."),
                "glob": task.get("glob", "**/*.py"),
                "is_regex": task.get("is_regex", False),
            }
            if task.get("max_results"):
                params["max_results"] = int(task["max_results"])
            actions.append(
                AgentAction.tool_call(
                    tool="search_code",
                    params=params,
                    agent=self.name,
                    step_id=step_id,
                    rationale=f"Search codebase for: {params['query']}",
                )
            )

        else:
            actions.append(
                AgentAction.message(
                    f"[CodingAgent] Unknown action '{action_type}' — no tool_call generated.",
                    agent=self.name,
                    step_id=step_id,
                )
            )

        return actions
