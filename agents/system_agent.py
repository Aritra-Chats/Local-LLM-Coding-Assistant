from __future__ import annotations
"""system_agent.py — Sentinel SystemAgent.

Responsible for OS-level operations: launching applications, executing
system-level shell commands, and managing installed software.  Generates
``tool_call`` actions for ``open_application``, ``run_shell``, and
``install_dependency`` — it never invokes tools directly.

Registered name: ``"system"``
"""


import traceback
import uuid
from typing import Any, Dict, List, Optional

from agents.agent_action import AgentAction
from agents.base_agent import BaseAgent
from agents.coding_agent import _parse_llm_actions

_ALLOWED_TOOLS = frozenset({"open_application", "run_shell", "install_dependency"})

_SYSTEM_PROMPT = """You are a system operations assistant. Given a task description, \
decide which OS-level operations are needed and respond with a JSON array of actions. \
Each action must be one of:
  {"tool": "open_application", "params": {"target": "<path_or_url>", "application": "<optional>"}}
  {"tool": "run_shell",        "params": {"command": "<command>", "cwd": "<optional_path>"}}
  {"tool": "install_dependency","params": {"packages": ["<pkg1>"]}}
  {"tool": "message",          "params": {"text": "<explanation>"}}
Respond ONLY with a valid JSON array. No prose before or after it."""


class SystemAgent(BaseAgent):
    """Specialist agent for operating-system level tasks.

    Supported task actions
    ----------------------
    ``"open"``
        Open a file or URL with the system default application.
    ``"run_shell"``
        Execute a safe OS-level shell command.
    ``"install"``
        Install a system-level Python package (delegates to pip via
        install_dependency tool).
    ``"launch"``
        Launch a named application with an optional target.
    """

    name = "system"

    def __init__(self, ollama_client: Optional[Any] = None, model: str = "") -> None:
        self._ollama = ollama_client
        self._model = model

    # ------------------------------------------------------------------
    # BaseAgent — required overrides
    # ------------------------------------------------------------------

    def run(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate system-level actions for the given task.

        Args:
            task: Step dict from PlannerAgent.
            context: Context payload.

        Returns:
            ``{"status": "ok", "actions": [AgentAction, ...], "task": task}``
        """
        step_id = task.get("step_id", str(uuid.uuid4()))
        model = task.get("_selected_model") or self._model
        client = self._ollama

        if client is not None and model:
            # Primary path: LLM-driven actions -- exceptions propagate so the
            # execution engine can retry with back-off / model fallback.
            actions = self._llm_actions(task, context, step_id, client, model)
            return {"status": "ok", "actions": actions, "task": task}

        # Fallback only when Ollama is not configured at all.
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
                reason=f"SystemAgent error: {error}\n{traceback.format_exc()}",
                agent=self.name,
                step_id=step_id,
            )
        ]
        return {"status": "error", "actions": actions, "error": str(error), "task": task}

    def describe(self) -> str:
        return (
            "SystemAgent: handles OS-level operations including launching "
            "applications, running shell commands, and installing packages.  "
            "Tools: open_application, run_shell, install_dependency."
        )

    # ------------------------------------------------------------------
    # LLM-driven action generation
    # ------------------------------------------------------------------

    def _llm_actions(
        self,
        task: Dict[str, Any],
        context: Dict[str, Any],
        step_id: str,
        client: Any,
        model: str,
    ) -> List[AgentAction]:
        description = task.get("description") or task.get("name", "")
        project_root = context.get("project_root", "")
        prompt = (
            f"{_SYSTEM_PROMPT}\n\n"
            f"Project root: {project_root}\n"
            f"Task: {description}\n"
        )
        response = client.generate(model=model, prompt=prompt)
        raw = response.get("response", "")
        return _parse_llm_actions(raw, self.name, step_id)

    # ------------------------------------------------------------------
    # Action generation (keyword fallback)
    # ------------------------------------------------------------------

    def _generate_actions(self, task: Dict[str, Any], step_id: str) -> List[AgentAction]:
        action_type = task.get("action", "run_shell").lower()
        actions: List[AgentAction] = []

        if action_type == "open":
            target = task.get("target", task.get("path", task.get("url", "")))
            if not target:
                actions.append(
                    AgentAction.message(
                        "[SystemAgent] 'open' action requires a 'target' (path or URL).",
                        agent=self.name,
                        step_id=step_id,
                    )
                )
            else:
                params: Dict[str, Any] = {"target": target}
                if task.get("application"):
                    params["application"] = task["application"]
                if task.get("wait") is not None:
                    params["wait"] = bool(task["wait"])
                actions.append(
                    AgentAction.tool_call(
                        tool="open_application",
                        params=params,
                        agent=self.name,
                        step_id=step_id,
                        rationale=f"Open: {target}",
                    )
                )

        elif action_type in ("launch", "open_application"):
            target = task.get("target", task.get("path", ""))
            application = task.get("application", "")
            params = {"target": target}
            if application:
                params["application"] = application
            actions.append(
                AgentAction.tool_call(
                    tool="open_application",
                    params=params,
                    agent=self.name,
                    step_id=step_id,
                    rationale=f"Launch application '{application or target}'.",
                )
            )

        elif action_type == "run_shell":
            cmd = task.get("command", "")
            if not cmd:
                actions.append(
                    AgentAction.message(
                        "[SystemAgent] No command provided for run_shell.",
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
                if task.get("env_extra"):
                    params["env_extra"] = task["env_extra"]
                actions.append(
                    AgentAction.tool_call(
                        tool="run_shell",
                        params=params,
                        agent=self.name,
                        step_id=step_id,
                        rationale=f"System shell command: {cmd}",
                    )
                )

        elif action_type == "install":
            packages = task.get("packages", [])
            if isinstance(packages, str):
                packages = [packages]
            if not packages:
                actions.append(
                    AgentAction.message(
                        "[SystemAgent] 'install' action requires a non-empty 'packages' field.",
                        agent=self.name,
                        step_id=step_id,
                    )
                )
            else:
                params = {"packages": packages}
                if task.get("upgrade"):
                    params["upgrade"] = True
                actions.append(
                    AgentAction.tool_call(
                        tool="install_dependency",
                        params=params,
                        agent=self.name,
                        step_id=step_id,
                        rationale=f"Install system packages: {packages}",
                    )
                )

        else:
            actions.append(
                AgentAction.message(
                    f"[SystemAgent] Unknown action '{action_type}'.",
                    agent=self.name,
                    step_id=step_id,
                )
            )

        return actions
