from __future__ import annotations
"""devops_agent.py — Sentinel DevOpsAgent.

Responsible for CI/CD operations, dependency management, shell execution,
and version-control workflows.  Generates ``tool_call`` actions for
``run_shell``, ``git_diff``, ``git_commit``, and ``install_dependency``
— it never invokes tools directly.

Registered name: ``"devops"``
"""


import traceback
import uuid
from typing import Any, Dict, List, Optional

from agents.agent_action import AgentAction
from agents.base_agent import BaseAgent

_ALLOWED_TOOLS = frozenset({"run_shell", "git_diff", "git_commit", "install_dependency", "run_tests"})


class DevOpsAgent(BaseAgent):
    """Specialist agent for DevOps and infrastructure tasks.

    Supported task actions
    ----------------------
    ``"git_diff"``
        Show the current diff (staged, unstaged, or range).
    ``"git_commit"``
        Stage and commit changes with a provided message.
    ``"install"``
        Install one or more Python packages via pip.
    ``"run_shell"``
        Execute a safe shell command.
    ``"run_tests"``
        Run the test suite (CI gate).
    ``"pipeline"``
        Convenience: install → test → commit in one step sequence.
    """

    name = "devops"

    # ------------------------------------------------------------------
    # BaseAgent — required overrides
    # ------------------------------------------------------------------

    def run(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate DevOps actions for the given task.

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
                reason=f"DevOpsAgent error: {error}\n{traceback.format_exc()}",
                agent=self.name,
                step_id=step_id,
            )
        ]
        return {"status": "error", "actions": actions, "error": str(error), "task": task}

    def describe(self) -> str:
        return (
            "DevOpsAgent: handles CI/CD operations, git workflows, shell commands, "
            "and dependency management.  Tools: run_shell, git_diff, git_commit, "
            "install_dependency, run_tests."
        )

    # ------------------------------------------------------------------
    # Action generation
    # ------------------------------------------------------------------

    def _generate_actions(self, task: Dict[str, Any], step_id: str) -> List[AgentAction]:
        action_type = task.get("action", "run_shell").lower()
        actions: List[AgentAction] = []

        if action_type == "git_diff":
            params: Dict[str, Any] = {}
            for key in ("path", "cwd", "staged", "base", "compare"):
                if task.get(key) is not None:
                    params[key] = task[key]
            actions.append(
                AgentAction.tool_call(
                    tool="git_diff",
                    params=params,
                    agent=self.name,
                    step_id=step_id,
                    rationale="Inspect current repository diff.",
                )
            )

        elif action_type == "git_commit":
            message = task.get("message", task.get("commit_message", "automated commit"))
            params = {"message": message}
            if task.get("paths"):
                params["paths"] = task["paths"]
            if task.get("add_all"):
                params["add_all"] = True
            actions.append(
                AgentAction.tool_call(
                    tool="git_commit",
                    params=params,
                    agent=self.name,
                    step_id=step_id,
                    rationale=f"Commit changes: {message}",
                )
            )

        elif action_type == "install":
            packages = task.get("packages", [])
            if isinstance(packages, str):
                packages = [packages]
            params = {"packages": packages}
            if task.get("upgrade"):
                params["upgrade"] = True
            actions.append(
                AgentAction.tool_call(
                    tool="install_dependency",
                    params=params,
                    agent=self.name,
                    step_id=step_id,
                    rationale=f"Install packages: {packages}",
                )
            )

        elif action_type == "run_shell":
            cmd = task.get("command", "")
            if not cmd:
                actions.append(
                    AgentAction.message(
                        "[DevOpsAgent] No command provided for run_shell.",
                        agent=self.name,
                        step_id=step_id,
                    )
                )
            else:
                params = {"command": cmd}
                if task.get("cwd"):
                    params["cwd"] = task["cwd"]
                if task.get("timeout"):
                    params["timeout"] = int(task["timeout"])
                actions.append(
                    AgentAction.tool_call(
                        tool="run_shell",
                        params=params,
                        agent=self.name,
                        step_id=step_id,
                        rationale=f"Execute: {cmd}",
                    )
                )

        elif action_type == "run_tests":
            params = {"path": task.get("path", ".")}
            if task.get("args"):
                params["args"] = task["args"]
            actions.append(
                AgentAction.tool_call(
                    tool="run_tests",
                    params=params,
                    agent=self.name,
                    step_id=step_id,
                    rationale="Run CI test gate.",
                )
            )

        elif action_type == "pipeline":
            # Convenience: install → run_tests → git_commit
            packages = task.get("packages", [])
            if packages:
                actions.append(
                    AgentAction.tool_call(
                        tool="install_dependency",
                        params={"packages": packages if isinstance(packages, list) else [packages]},
                        agent=self.name,
                        step_id=step_id,
                        rationale="Install dependencies before pipeline run.",
                    )
                )
            actions.append(
                AgentAction.tool_call(
                    tool="run_tests",
                    params={"path": task.get("test_path", ".")},
                    agent=self.name,
                    step_id=step_id,
                    rationale="Run tests as part of pipeline.",
                )
            )
            commit_msg = task.get("message", "ci: automated pipeline commit")
            actions.append(
                AgentAction.tool_call(
                    tool="git_commit",
                    params={"message": commit_msg, "add_all": task.get("add_all", False)},
                    agent=self.name,
                    step_id=step_id,
                    rationale=f"Pipeline commit: {commit_msg}",
                )
            )

        else:
            actions.append(
                AgentAction.message(
                    f"[DevOpsAgent] Unknown action '{action_type}'.",
                    agent=self.name,
                    step_id=step_id,
                )
            )

        return actions
