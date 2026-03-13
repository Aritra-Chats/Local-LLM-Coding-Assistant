"""critic_agent.py — Sentinel CriticAgent.

Implements a senior-reviewer pass that intercepts ``write_file`` actions
produced by any code-generation agent (CodingAgent, DebuggingAgent, etc.),
submits the code to the LLM for structured critique, and emits a revision
request when the model surfaces real problems.

Architecture
------------
The CriticAgent is *not* a pipeline step agent in the normal sense —
it is called by the ExecutionEngine's ``_run_solo`` path immediately after
the primary coding agent runs and before the write_file actions are
dispatched to the ToolRegistry.

Flow
----
::

    CodingAgent.run()
        → returns write_file actions
        ↓
    CriticAgent.review(actions, context)
        → LLM reviews the code
        ↓
    If issues found → CodingAgent.revise(original_actions, critique)
        → revised write_file actions
        ↓
    ExecutionEngine dispatches to ToolRegistry

The review is skipped when:
  * No Ollama client or model is configured.
  * No ``write_file`` actions exist in the action list (nothing to review).
  * The step carries ``"skip_critic": True`` metadata (e.g. test stubs).
  * The code passes the critic with no issues.

Critique dimensions
-------------------
The critic evaluates code against five dimensions:

1. **Logic errors** — incorrect control flow, off-by-one errors, wrong
   variable usage, unreachable code.
2. **Architecture violations** — coupling, missing abstractions, wrong
   layer placement, circular imports.
3. **Performance issues** — O(n²) loops, unnecessary re-computation,
   missing caching opportunities, large memory allocations.
4. **Security issues** — SQL injection, path traversal, hard-coded
   secrets, unsafe deserialization.
5. **Style / correctness** — syntax errors, type mismatches, missing
   docstrings, inconsistent naming.

Output format
-------------
The critic returns a structured JSON verdict::

    {
        "verdict": "pass" | "revise",
        "issues": [
            {
                "dimension": "logic" | "architecture" | "performance" | "security" | "style",
                "severity": "critical" | "major" | "minor",
                "line_hint": 12,
                "description": "Brief description of the problem.",
                "suggestion": "How to fix it."
            }
        ],
        "overall_comment": "One-paragraph summary."
    }

When ``verdict == "pass"`` the original actions are returned unchanged.
When ``verdict == "revise"`` the agent re-prompts the coding LLM with the
issues list and returns the revised write_file actions.

Registered name: ``"critic"``
"""
from __future__ import annotations

import json
import re
import traceback
import uuid
from typing import Any, Dict, List, Optional, Tuple

from agents.agent_action import AgentAction
from agents.base_agent import BaseAgent

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_CRITIC_SYSTEM_PROMPT = """\
You are a senior software engineer performing a rigorous code review.

Analyse the code below and identify issues across five dimensions:
  1. logic        — incorrect control-flow, wrong variables, unreachable code
  2. architecture — bad coupling, wrong abstractions, circular imports
  3. performance  — O(n²) loops, unnecessary re-computation, memory waste
  4. security     — injection, path traversal, hard-coded secrets, unsafe deserialization
  5. style        — syntax errors, type mismatches, missing docstrings, naming

Respond ONLY with a single valid JSON object, no prose before or after it:

{
  "verdict": "pass" | "revise",
  "issues": [
    {
      "dimension": "logic" | "architecture" | "performance" | "security" | "style",
      "severity": "critical" | "major" | "minor",
      "line_hint": <integer or null>,
      "description": "<concise description>",
      "suggestion": "<concrete fix>"
    }
  ],
  "overall_comment": "<one paragraph summary>"
}

Rules:
- If no issues are found, set verdict to "pass" and issues to [].
- Only set verdict to "revise" when there is at least one critical or major issue.
- Be precise — do not flag minor style nits as reasons to revise.
"""

_REVISION_SYSTEM_PROMPT = """\
You are a senior software engineer revising code based on a code review.

Apply the following critique to the original code and produce a corrected version.
Output ONLY the corrected file content — no explanations, no markdown code fences,
no comments like "// Fixed version". Start with the very first character of the file.
"""


class CriticAgent(BaseAgent):
    """LLM-powered code reviewer that enforces quality before execution.

    Args:
        ollama_client: An :class:`~models.ollama_client.OllamaClient` instance.
        model:         Ollama model tag used for reviews (defaults to the
                       step's ``_selected_model`` when not set here).
        max_revisions: Maximum revision attempts per review cycle (default 1).
                       Set to 0 to make the agent advisory-only (it emits
                       critique messages but never blocks actions).
    """

    name = "critic"

    def __init__(
        self,
        ollama_client: Optional[Any] = None,
        model: str = "",
        max_revisions: int = 1,
    ) -> None:
        self._ollama = ollama_client
        self._model = model
        self.max_revisions = max_revisions

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def run(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Run critic review when invoked as a standalone pipeline step.

        In normal operation the critic is called via :meth:`review_actions`
        rather than as a pipeline step.  This ``run()`` implementation
        exists to satisfy the :class:`~agents.base_agent.BaseAgent` ABC
        and to support pipelines that include a dedicated critic step.

        The task must carry an ``"actions"`` key with the list of
        :class:`~agents.agent_action.AgentAction` objects to review.
        """
        step_id = task.get("step_id", str(uuid.uuid4()))
        raw_actions = task.get("actions", [])
        if not raw_actions:
            return {
                "status": "ok",
                "actions": [AgentAction.message(
                    "[CriticAgent] No actions provided to review.",
                    agent=self.name, step_id=step_id,
                )],
                "critique": None,
                "task": task,
            }

        revised_actions, critique = self.review_actions(
            actions=raw_actions,
            context=context,
            step=task,
        )
        feedback_action = AgentAction.message(
            self._format_critique_message(critique),
            agent=self.name,
            step_id=step_id,
        )
        return {
            "status": "ok",
            "actions": [feedback_action] + revised_actions,
            "critique": critique,
            "task": task,
        }

    def validate_output(self, output: Dict[str, Any]) -> bool:
        return (
            isinstance(output, dict)
            and output.get("status") == "ok"
            and isinstance(output.get("actions"), list)
        )

    def handle_error(self, error: Exception, task: Dict[str, Any]) -> Dict[str, Any]:
        step_id = task.get("step_id", "unknown")
        return {
            "status": "error",
            "actions": [AgentAction.abort(
                reason=f"CriticAgent error: {error}\n{traceback.format_exc()}",
                agent=self.name, step_id=step_id,
            )],
            "error": str(error),
            "task": task,
        }

    def describe(self) -> str:
        return (
            "CriticAgent: performs LLM-powered code review on write_file actions "
            "before they reach the ToolRegistry.  Emits critique messages and "
            "optionally triggers a revision pass to fix critical/major issues."
        )

    # ------------------------------------------------------------------
    # Primary public API: review_actions
    # ------------------------------------------------------------------

    def review_actions(
        self,
        actions: List[AgentAction],
        context: Dict[str, Any],
        step: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[AgentAction], Optional[Dict[str, Any]]]:
        """Review a list of actions and return (possibly revised actions, critique).

        This is the main entry point used by
        :class:`~core.execution_engine.ConcreteExecutionEngine`.

        Args:
            actions: Actions produced by the primary agent.
            context: Current step context dict.
            step:    The pipeline step dict (used to check ``skip_critic``
                     and to select the model).

        Returns:
            ``(actions, critique_dict)`` — if no review was performed or the
            code passed, the original actions are returned unchanged and
            ``critique_dict`` is ``None``.  If issues were found and revised,
            the revised actions replace the originals.
        """
        step = step or {}

        # Skip when explicitly disabled on a step
        if step.get("skip_critic") or step.get("metadata", {}).get("skip_critic"):
            return actions, None

        client = self._ollama
        model = step.get("_selected_model") or self._model
        if client is None or not model:
            return actions, None

        # Collect write_file actions — those are the code to review
        write_actions = [
            a for a in actions
            if a.action_type == "tool_call"
            and a.payload.get("tool") == "write_file"
        ]
        if not write_actions:
            return actions, None

        all_revised = list(actions)  # mutable copy
        last_critique: Optional[Dict[str, Any]] = None

        for wa in write_actions:
            params = wa.payload.get("params", {})
            path = params.get("path", "unknown")
            code = params.get("content", "")
            if not code.strip():
                continue

            critique = self._llm_critique(code, path, client, model, context)
            last_critique = critique

            if critique is None:
                continue

            verdict = critique.get("verdict", "pass")
            issues = critique.get("issues", [])
            critical_or_major = [
                i for i in issues
                if i.get("severity") in ("critical", "major")
            ]

            if verdict == "pass" or not critical_or_major:
                continue  # Code is good, keep original action

            # Attempt revision
            if self.max_revisions <= 0:
                continue  # Advisory-only mode — don't revise

            for _rev_attempt in range(self.max_revisions):
                revised_code = self._llm_revise(code, path, critique, client, model)
                if not revised_code or revised_code.strip() == code.strip():
                    break  # Revision produced no change
                code = revised_code
                critique = self._llm_critique(code, path, client, model, context)
                if critique is None:
                    break
                last_critique = critique
                remaining = [
                    i for i in critique.get("issues", [])
                    if i.get("severity") in ("critical", "major")
                ]
                if not remaining:
                    break  # Issues resolved

            # Replace the original write_file action with revised content
            revised_wa = AgentAction.tool_call(
                tool="write_file",
                params={**params, "content": code},
                agent=wa.agent,
                step_id=wa.step_id,
                rationale=f"[CriticAgent] Revised content for {path}",
            )
            idx = all_revised.index(wa)
            all_revised[idx] = revised_wa

        return all_revised, last_critique

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    def _llm_critique(
        self,
        code: str,
        path: str,
        client: Any,
        model: str,
        context: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Ask the LLM to critique *code* and return a parsed verdict dict.

        Returns ``None`` on any error so callers degrade gracefully.
        """
        synopsis = context.get("synopsis", "")
        prompt = (
            f"{_CRITIC_SYSTEM_PROMPT}\n\n"
            f"File path: {path}\n"
            + (f"Project context:\n{synopsis[:600]}\n\n" if synopsis else "")
            + f"Code to review:\n```\n{code[:6000]}\n```"
        )

        try:
            response = client.generate(
                model=model,
                prompt=prompt,
                options={"temperature": 0.1, "num_predict": 1024},
            )
            raw = response.get("response", "").strip()
            return self._parse_critique(raw)
        except Exception:
            return None

    def _llm_revise(
        self,
        code: str,
        path: str,
        critique: Dict[str, Any],
        client: Any,
        model: str,
    ) -> str:
        """Ask the LLM to fix *code* given a critique dict.

        Returns the revised code string, or the original code on failure.
        """
        issues_text = "\n".join(
            f"  [{i.get('severity','?')}] {i.get('dimension','?')}: "
            f"{i.get('description','')}  →  {i.get('suggestion','')}"
            for i in critique.get("issues", [])
            if i.get("severity") in ("critical", "major")
        )
        overall = critique.get("overall_comment", "")

        prompt = (
            f"{_REVISION_SYSTEM_PROMPT}\n\n"
            f"File: {path}\n\n"
            f"Issues to fix:\n{issues_text}\n\n"
            f"Reviewer summary: {overall}\n\n"
            f"Original code:\n```\n{code[:6000]}\n```\n\n"
            "Corrected file content:"
        )

        try:
            response = client.generate(
                model=model,
                prompt=prompt,
                options={"temperature": 0.2, "num_predict": 4096},
            )
            revised = response.get("response", "").strip()
            # Strip any accidental markdown fences
            if revised.startswith("```"):
                lines = revised.splitlines()
                revised = "\n".join(
                    ln for ln in lines if not ln.startswith("```")
                ).strip()
            return revised if revised else code
        except Exception:
            return code

    @staticmethod
    def _parse_critique(raw: str) -> Optional[Dict[str, Any]]:
        """Parse the LLM's JSON critique response.

        Tolerates markdown code fences and minor JSON slop.

        Returns:
            Parsed critique dict, or ``None`` on failure.
        """
        text = raw.strip()
        # Strip markdown fences
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(
                ln for ln in lines if not ln.startswith("```")
            ).strip()
        # Extract the first {...} block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
            # Normalise required keys
            if "verdict" not in data:
                data["verdict"] = "pass"
            if "issues" not in data:
                data["issues"] = []
            if "overall_comment" not in data:
                data["overall_comment"] = ""
            return data
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _format_critique_message(critique: Optional[Dict[str, Any]]) -> str:
        """Convert a critique dict to a human-readable message string."""
        if critique is None:
            return "[CriticAgent] Review skipped (no LLM or no write_file actions)."

        verdict = critique.get("verdict", "pass")
        issues = critique.get("issues", [])
        overall = critique.get("overall_comment", "")

        if verdict == "pass" and not issues:
            return f"[CriticAgent] ✓ Code review passed. {overall}"

        lines = [f"[CriticAgent] Code review verdict: {verdict.upper()}"]
        if overall:
            lines.append(f"  Summary: {overall}")
        for iss in issues:
            sev = iss.get("severity", "?")
            dim = iss.get("dimension", "?")
            desc = iss.get("description", "")
            sug = iss.get("suggestion", "")
            hint = iss.get("line_hint")
            loc = f" (line {hint})" if hint else ""
            lines.append(f"  [{sev.upper()} / {dim}]{loc}: {desc}")
            if sug:
                lines.append(f"    → Fix: {sug}")
        return "\n".join(lines)
