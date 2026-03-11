from __future__ import annotations
"""debugging_agent.py — Sentinel DebuggingAgent.

Responsible for diagnosing and resolving runtime errors, test failures,
and logic bugs.  Generates ``tool_call`` actions for ``run_tests``,
``run_shell``, and ``read_file`` — it never invokes tools directly.

Registered name: ``"debugging"``
"""


import traceback
import uuid
from typing import Any, Dict, List

from agents.agent_action import AgentAction
from agents.base_agent import BaseAgent

_ALLOWED_TOOLS = frozenset({"run_tests", "run_shell", "read_file", "search_code"})


class DebuggingAgent(BaseAgent):
    """Specialist agent for diagnosing bugs and test failures.

    Supported task actions
    ----------------------
    ``"run_tests"``
        Execute the test suite and capture results.
    ``"inspect_error"``
        Read the erroring file and search for the relevant symbol.
    ``"run_shell"``
        Execute a diagnostic shell command (e.g. ``python -c "import ..."``)
    ``"search_traceback"``
        Search the codebase for an identifier appearing in a traceback.
    """

    name = "debugging"

    # ------------------------------------------------------------------
    # BaseAgent — required overrides
    # ------------------------------------------------------------------

    def run(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate debugging actions for the given task.

        Args:
            task: Step dict from PlannerAgent.
            context: Context payload.

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
                reason=f"DebuggingAgent error: {error}\n{traceback.format_exc()}",
                agent=self.name,
                step_id=step_id,
            )
        ]
        return {"status": "error", "actions": actions, "error": str(error), "task": task}

    def describe(self) -> str:
        return (
            "DebuggingAgent: diagnoses runtime errors and test failures.  "
            "Generates tool_call actions for run_tests, run_shell, read_file, "
            "and search_code to locate and contextualise bugs."
        )

    # ------------------------------------------------------------------
    # Action generation
    # ------------------------------------------------------------------

    def _generate_actions(self, task: Dict[str, Any], step_id: str) -> List[AgentAction]:
        action_type = task.get("action", "run_tests").lower()
        actions: List[AgentAction] = []

        if action_type == "run_tests":
            params: Dict[str, Any] = {"path": task.get("path", ".")}
            if task.get("args"):
                params["args"] = task["args"]
            if task.get("runner"):
                params["runner"] = task["runner"]
            actions.append(
                AgentAction.tool_call(
                    tool="run_tests",
                    params=params,
                    agent=self.name,
                    step_id=step_id,
                    rationale="Run test suite to surface failing tests.",
                )
            )

        elif action_type == "inspect_error":
            path = task.get("path", "")
            if path:
                actions.append(
                    AgentAction.tool_call(
                        tool="read_file",
                        params={"path": path},
                        agent=self.name,
                        step_id=step_id,
                        rationale=f"Read erroring file for inspection: {path}",
                    )
                )
            symbol = task.get("symbol", "")
            if symbol:
                actions.append(
                    AgentAction.tool_call(
                        tool="search_code",
                        params={"query": symbol, "path": task.get("search_root", "."), "glob": "**/*.py"},
                        agent=self.name,
                        step_id=step_id,
                        rationale=f"Locate symbol '{symbol}' in codebase.",
                    )
                )

        elif action_type == "run_shell":
            cmd = task.get("command", "")
            if not cmd:
                actions.append(
                    AgentAction.message(
                        "[DebuggingAgent] No command provided for run_shell action.",
                        agent=self.name,
                        step_id=step_id,
                    )
                )
            else:
                actions.append(
                    AgentAction.tool_call(
                        tool="run_shell",
                        params={"command": cmd, "timeout": task.get("timeout", 30)},
                        agent=self.name,
                        step_id=step_id,
                        rationale=f"Run diagnostic command: {cmd}",
                    )
                )

        elif action_type == "search_traceback":
            identifier = task.get("identifier", task.get("query", ""))
            actions.append(
                AgentAction.tool_call(
                    tool="search_code",
                    params={
                        "query": identifier,
                        "path": task.get("path", "."),
                        "glob": "**/*.py",
                        "is_regex": task.get("is_regex", False),
                    },
                    agent=self.name,
                    step_id=step_id,
                    rationale=f"Search for traceback identifier: {identifier}",
                )
            )

        else:
            actions.append(
                AgentAction.message(
                    f"[DebuggingAgent] Unknown action '{action_type}'.",
                    agent=self.name,
                    step_id=step_id,
                )
            )

        return actions
