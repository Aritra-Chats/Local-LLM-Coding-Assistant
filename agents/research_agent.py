from __future__ import annotations
"""research_agent.py — Sentinel ResearchAgent.

Responsible for information gathering from the web and the local filesystem.
Generates ``tool_call`` actions for ``web_search`` and ``read_file`` — it
never invokes tools directly.

Registered name: ``"research"``
"""


import traceback
import uuid
from typing import Any, Dict, List, Optional

from agents.agent_action import AgentAction
from agents.base_agent import BaseAgent
from agents.coding_agent import _parse_llm_actions

_ALLOWED_TOOLS = frozenset({"web_search", "read_file", "search_code"})

_SYSTEM_PROMPT = """You are a research assistant. Given a task description, \
decide which information-gathering operations are needed and respond with a JSON array of actions. \
Each action must be one of:
  {"tool": "web_search",  "params": {"query": "<search_query>", "max_results": 5}}
  {"tool": "read_file",   "params": {"path": "<relative_path>"}}
  {"tool": "search_code", "params": {"query": "<pattern>", "path": ".", "is_regex": false}}
  {"tool": "message",     "params": {"text": "<explanation or findings>"}}
Respond ONLY with a valid JSON array. No prose before or after it."""


class ResearchAgent(BaseAgent):
    """Specialist agent for web research and local documentation lookup.

    Supported task actions
    ----------------------
    ``"web_search"``
        Issue a web search and return result snippets.
    ``"read_docs"``
        Read a local documentation file.
    ``"search_code"``
        Search the codebase for a pattern (used during research).
    ``"multi_search"``
        Issue multiple web searches sequentially (one per query in the
        ``"queries"`` list).
    """

    name = "research"

    def __init__(self, ollama_client: Optional[Any] = None, model: str = "") -> None:
        self._ollama = ollama_client
        self._model = model

    # ------------------------------------------------------------------
    # BaseAgent — required overrides
    # ------------------------------------------------------------------

    def run(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate research actions for the given task.

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
                reason=f"ResearchAgent error: {error}\n{traceback.format_exc()}",
                agent=self.name,
                step_id=step_id,
            )
        ]
        return {"status": "error", "actions": actions, "error": str(error), "task": task}

    def describe(self) -> str:
        return (
            "ResearchAgent: gathers information from the web and local files.  "
            "Generates tool_call actions for web_search, read_file, and "
            "search_code."
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
        action_type = task.get("action", "web_search").lower()
        actions: List[AgentAction] = []

        if action_type == "web_search":
            query = task.get("query", task.get("description", ""))
            if not query:
                actions.append(
                    AgentAction.message(
                        "[ResearchAgent] No 'query' provided for web_search.",
                        agent=self.name,
                        step_id=step_id,
                    )
                )
            else:
                params: Dict[str, Any] = {"query": query}
                if task.get("max_results"):
                    params["max_results"] = int(task["max_results"])
                actions.append(
                    AgentAction.tool_call(
                        tool="web_search",
                        params=params,
                        agent=self.name,
                        step_id=step_id,
                        rationale=f"Search web for: {query}",
                    )
                )

        elif action_type == "read_docs":
            path = task.get("path", "")
            if not path:
                actions.append(
                    AgentAction.message(
                        "[ResearchAgent] No 'path' provided for read_docs.",
                        agent=self.name,
                        step_id=step_id,
                    )
                )
            else:
                params = {"path": path}
                if task.get("start_line"):
                    params["start_line"] = int(task["start_line"])
                if task.get("end_line"):
                    params["end_line"] = int(task["end_line"])
                actions.append(
                    AgentAction.tool_call(
                        tool="read_file",
                        params=params,
                        agent=self.name,
                        step_id=step_id,
                        rationale=f"Read documentation: {path}",
                    )
                )

        elif action_type == "search_code":
            query = task.get("query", task.get("description", ""))
            params = {
                "query": query,
                "path": task.get("path", "."),
                "glob": task.get("glob", "**/*"),
                "is_regex": task.get("is_regex", False),
            }
            actions.append(
                AgentAction.tool_call(
                    tool="search_code",
                    params=params,
                    agent=self.name,
                    step_id=step_id,
                    rationale=f"Search codebase for: {query}",
                )
            )

        elif action_type == "multi_search":
            queries = task.get("queries", [])
            if not queries:
                actions.append(
                    AgentAction.message(
                        "[ResearchAgent] 'multi_search' requires a non-empty 'queries' list.",
                        agent=self.name,
                        step_id=step_id,
                    )
                )
            else:
                for q in queries:
                    actions.append(
                        AgentAction.tool_call(
                            tool="web_search",
                            params={"query": str(q), "max_results": task.get("max_results", 5)},
                            agent=self.name,
                            step_id=step_id,
                            rationale=f"Multi-search query: {q}",
                        )
                    )

        else:
            actions.append(
                AgentAction.message(
                    f"[ResearchAgent] Unknown action '{action_type}'.",
                    agent=self.name,
                    step_id=step_id,
                )
            )

        return actions
