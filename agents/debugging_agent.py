from __future__ import annotations
"""debugging_agent.py — Sentinel DebuggingAgent."""


import json
import traceback
import uuid
from typing import Any, Dict, List, Optional

from agents.agent_action import AgentAction
from agents.base_agent import BaseAgent
from agents.coding_agent import _parse_llm_actions

_ALLOWED_TOOLS = frozenset({"run_tests", "run_shell", "read_file", "search_code"})

_SYSTEM_PROMPT = """You are a debugging assistant. Given a task description and project context, \
decide which diagnostic operations are needed and respond with a JSON array of actions. \
Each action must be one of:
  {"tool": "run_tests",   "params": {"path": "<test_path_or_."}}
  {"tool": "run_shell",   "params": {"command": "<shell_command>", "timeout": 30}}
  {"tool": "read_file",   "params": {"path": "<relative_path>"}}
  {"tool": "search_code", "params": {"query": "<search_term>", "path": "."}}
  {"tool": "message",     "params": {"text": "<explanation or finding>"}}
Respond ONLY with a valid JSON array. No prose before or after it."""


class DebuggingAgent(BaseAgent):
    """Specialist agent for diagnosing bugs and test failures."""

    name = "debugging"

    def __init__(self, ollama_client: Optional[Any] = None, model: str = "") -> None:
        self._ollama = ollama_client
        self._model = model

    # ------------------------------------------------------------------
    # BaseAgent — required overrides
    # ------------------------------------------------------------------

    def run(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        step_id = task.get("step_id", str(uuid.uuid4()))
        model = task.get("_selected_model") or self._model
        client = self._ollama

        if client is not None and model:
            # Primary path: LLM-driven actions — exceptions propagate so the
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
        synopsis = context.get("synopsis", "")
        rag_hits = context.get("rag", [])
        rag_text = ""
        if rag_hits:
            rag_text = "\n".join(
                f"  [{h.get('file_path','')}:{h.get('start_line','')}]\n{h.get('content','')}"
                for h in rag_hits[:3]
            )

        prompt = (
            f"{_SYSTEM_PROMPT}\n\n"
            f"Project root: {project_root}\n"
            f"Task: {description}\n"
            + (f"Project synopsis:\n{synopsis}\n" if synopsis else "")
            + (f"Relevant code:\n{rag_text}\n" if rag_text else "")
        )

        response = client.generate(model=model, prompt=prompt)
        raw = response.get("response", "")
        return _parse_llm_actions(raw, self.name, step_id)

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
