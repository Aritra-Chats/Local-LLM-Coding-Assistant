from __future__ import annotations
"""reasoning_agent.py — Sentinel ReasoningAgent."""


import traceback
import uuid
from typing import Any, Dict, List, Optional

from agents.agent_action import AgentAction
from agents.base_agent import BaseAgent
from agents.coding_agent import _parse_llm_actions


_SYSTEM_PROMPT = """You are a reasoning assistant. Given a task description and project context, \
analyse the situation and respond with a JSON array of actions. \
Each action must be one of:
  {"tool": "message",     "params": {"text": "<your analysis, findings, or recommendations>"}}
  {"tool": "read_file",   "params": {"path": "<relative_path>"}}
  {"tool": "search_code", "params": {"query": "<search_term>", "path": "."}}
Provide your full reasoning inside "message" actions. \
Respond ONLY with a valid JSON array. No prose before or after it."""


class ReasoningAgent(BaseAgent):
    """Specialist agent for analysis, explanation, and decision-making."""

    name = "reasoning"

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
