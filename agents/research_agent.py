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

_ALLOWED_TOOLS = frozenset({"web_search", "read_file", "search_code"})


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
    # Action generation
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
