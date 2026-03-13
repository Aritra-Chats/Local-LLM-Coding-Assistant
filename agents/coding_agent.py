from __future__ import annotations
"""coding_agent.py — Sentinel CodingAgent.

Responsible for all source-code generation, editing, and search tasks.
Generates ``tool_call`` actions for ``read_file``, ``write_file``, and
``search_code`` — it never invokes tools directly.

Registered name: ``"coding"``
"""


import json
import re
import traceback
import uuid
from typing import Any, Dict, List, Optional

from agents.agent_action import AgentAction
from agents.base_agent import BaseAgent

# Tools this agent is permitted to request.
_ALLOWED_TOOLS = frozenset({"read_file", "write_file", "search_code", "find_files", "run_shell"})

_SYSTEM_PROMPT = """You are a coding assistant. Given a task description and project context, \
decide which file operations are needed and respond with a JSON array of actions. \
Each action must be one of:
  {"tool": "read_file",   "params": {"path": "<relative_path>"}}
  {"tool": "search_code", "params": {"query": "<search_term>", "path": "."}}
  {"tool": "write_file",  "params": {"path": "<relative_path>", "content": "<new_content>"}}
  {"tool": "run_shell",   "params": {"command": "<shell_command>"}}
  {"tool": "find_files",  "params": {"pattern": "<glob_pattern>"}}
  {"tool": "message",     "params": {"text": "<explanation>"}}

IMPORTANT FOR NEW PROJECTS: When the task is to "write the source code" or "implement code" \
for a NEW feature or project (not editing existing files), you MUST:
1. Generate write_file actions to CREATE all necessary files (package.json, source code, config, etc.)
2. For Node.js/npm projects: After writing package.json, generate run_shell actions to:
   - Run 'npm install' in the directory containing package.json
   - Run any other necessary setup commands
3. Do NOT just return search_code or read_file actions.

Respond ONLY with a valid JSON array. No prose before or after it."""


class CodingAgent(BaseAgent):
    """Specialist agent for source-code generation and file manipulation."""

    name = "coding"

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
            # Primary path: LLM-driven actions — exceptions propagate to the
            # execution engine which will retry with back-off.
            actions = self._llm_actions(task, context, step_id, client, model)
            return {"status": "ok", "actions": actions, "task": task}

        # Fallback only when Ollama is not configured at all (e.g. integration
        # tests without a running Ollama instance).
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
        """Ask the LLM to decide which file operations are needed."""
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

        # ------------------------------------------------------------------
        # Two-stage path: when task is "write a new file" with a concrete
        # path, ask the LLM to generate raw file content instead of a JSON
        # action list.  This avoids the model reflexively returning
        # read_file "." before it ever produces a write_file action.
        # ------------------------------------------------------------------
        # NOTE: json|yaml|yml must come before js|ts to avoid partial matches.
        # \b word-boundary prevents '.js' from matching inside '.json'.
        # We also search raw_prompt in case the path lives there rather than in description.
        raw_prompt = task.get("metadata", {}).get("raw_prompt", "") or description
        _path_search_text = description + " " + raw_prompt
        path_match = re.search(
            r'([A-Za-z]:[/\\][^"\n\r]+\.(?:html?|css|json|yaml|yml|jsx|tsx|js|ts|py|md|txt|sh))\b',
            _path_search_text,
            re.IGNORECASE,
        )
        is_write_step = bool(
            re.match(r"write(?:\s+the)?\s+source\s+code", description, re.IGNORECASE)
        )
        if path_match and is_write_step:
            target_path = path_match.group(1).strip()
            try:
                # Use the full raw_prompt so the LLM has all requirements.
                # raw_prompt already assigned above.
                # Extract just the requirements section (skip the meta-instruction first line).
                req_lines = raw_prompt.split("\n")
                req_body = "\n".join(req_lines[2:]).strip() if len(req_lines) > 2 else raw_prompt
                content_prompt = (
                    f"Write the complete source code for the file: {target_path}\n"
                    "Output ONLY the raw file content. Start with the very first character "
                    "of the file. No explanations, no markdown code fences, no placeholder "
                    "comments like '// New file content'.\n\n"
                    f"Requirements:\n{req_body}\n\n"
                    "File content:"
                )
                response = client.generate(
                    model=model,
                    prompt=content_prompt,
                    timeout=600,
                    options={"num_ctx": 8192, "num_predict": 4096, "temperature": 0.3},
                )
                content = response.get("response", "").strip()
                if content:
                    # Strip any accidental markdown code fences
                    if content.startswith("```"):
                        lines = content.splitlines()
                        content = "\n".join(
                            ln for ln in lines if not ln.startswith("```")
                        ).strip()
                    return [
                        AgentAction.tool_call(
                            tool="write_file",
                            params={"path": target_path, "content": content},
                            agent=self.name,
                            step_id=step_id,
                            rationale=f"Write LLM-generated content to {target_path}",
                        )
                    ]
            except Exception:
                raise

        # ------------------------------------------------------------------
        # Standard JSON-action path for all other (non-write) tasks.
        # ------------------------------------------------------------------
        
        # Detect task type
        import os
        is_test_task = bool(
            re.search(r"write\s+unit\s+tests|write\s+tests|unit\s+test|test\s+suite|covering\s+the\s+impl", description, re.IGNORECASE)
        )
        is_implementation_task = (not is_test_task) and bool(
            re.search(r"write.*(?:source\s+)?code|implement", description, re.IGNORECASE)
        )

        # Check if project directory is empty or has very few files
        project_is_empty = False
        if project_root and os.path.isdir(project_root):
            try:
                file_count = sum(1 for _ in os.walk(project_root) for f in _[2]
                                 if not f.startswith('.') and 'node_modules' not in _[0])
                project_is_empty = file_count < 5
            except Exception:
                pass

        # Build step-specific guidance
        implementation_guidance = ""
        if is_test_task:
            implementation_guidance = (
                "\n\nThis is a TEST WRITING task. You must ONLY write test files. "
                "Do NOT re-write any existing implementation files (package.json, App.js, index.js, server files, etc.). "
                "Only create test files such as: src/__tests__/App.test.js, src/App.test.js, tests/test_*.py, etc.\n"
                "For React projects write Jest tests. For Node/Express write Mocha/Jest tests.\n"
                "Example:\n"
                '[\n  {"tool": "write_file", "params": {"path": "client/src/__tests__/App.test.js", "content": "..."}}\n]\n'
            )
        elif is_implementation_task and (not rag_text or project_is_empty):
            implementation_guidance = (
                "\n\nThis is a NEW implementation task. You must generate write_file actions "
                "to CREATE all necessary files, then run setup commands. Think about the complete project structure:\n"
                "1. Identify ALL files needed (e.g., package.json, source files, config files, README)\n"
                "2. For EACH file, create a write_file action with the full file path and complete content\n"
                "3. Use proper paths relative to project root\n"
                "4. Generate actual working code, not placeholders\n"
                "5. For Node.js/React projects: After writing package.json, generate run_shell actions to:\n"
                "   - 'npm install' in the directory containing package.json (e.g., 'npm install' if cwd=client/)\n"
                "   - Any other setup commands needed\n"
                "Example structure:\n"
                '[\n  {"tool": "write_file", "params": {"path": "client/package.json", "content": "..."}},\n'
                '  {"tool": "write_file", "params": {"path": "client/src/App.js", "content": "..."}},\n'
                '  {"tool": "write_file", "params": {"path": "server/index.js", "content": "..."}},\n'
                '  {"tool": "write_file", "params": {"path": "public/index.html", "content": "..."}},\n'
                '  {"tool": "run_shell", "params": {"command": "cd client && npm install"}},\n'
                '  {"tool": "run_shell", "params": {"command": "cd server && npm install"}}\n]\n'
            )
        
        prompt = (
            f"{_SYSTEM_PROMPT}\n\n"
            f"Project root: {project_root}\n"
            f"Task: {description}\n"
            + (f"Project synopsis:\n{synopsis}\n" if synopsis else "")
            + (f"Relevant code:\n{rag_text}\n" if rag_text else "")
            + implementation_guidance
        )

        response = client.generate(model=model, prompt=prompt, timeout=600,
                                   options={"num_ctx": 8192, "num_predict": 4096, "temperature": 0.3})
        raw = response.get("response", "")
        return _parse_llm_actions(raw, self.name, step_id)

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


# ---------------------------------------------------------------------------
# Shared LLM response parser (used by all agents in this package)
# ---------------------------------------------------------------------------

_TOOL_ACTIONS = frozenset({"read_file", "write_file", "search_code", "run_tests",
                            "run_shell", "web_search", "git_diff", "git_commit"})


def _parse_llm_actions(raw: str, agent_name: str, step_id: str) -> List[AgentAction]:
    """Parse a raw LLM response into AgentAction objects.

    Expects a JSON array of ``{"tool": ..., "params": {...}}`` objects.
    Falls back gracefully to a message action on any parse error.
    """
    # Strip markdown code fences if present
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            l for l in lines if not l.startswith("```")
        ).strip()

    actions: List[AgentAction] = []
    try:
        items = json.loads(text)
        if not isinstance(items, list):
            items = [items]
        for item in items:
            if not isinstance(item, dict):
                continue
            tool = item.get("tool", "")
            params = item.get("params", {})
            rationale = item.get("rationale", "")
            if tool == "message":
                actions.append(AgentAction.message(
                    params.get("text", str(item)),
                    agent=agent_name, step_id=step_id,
                ))
            elif tool in _TOOL_ACTIONS:
                actions.append(AgentAction.tool_call(
                    tool=tool, params=params,
                    agent=agent_name, step_id=step_id,
                    rationale=rationale or f"LLM-requested: {tool}",
                ))
            else:
                actions.append(AgentAction.message(
                    f"[{agent_name}] {tool}: {params}",
                    agent=agent_name, step_id=step_id,
                ))
    except Exception:
        # If the model returned prose instead of JSON, surface it as a message
        actions.append(AgentAction.message(
            raw[:2000],
            agent=agent_name, step_id=step_id,
        ))
    return actions or [AgentAction.message(
        f"[{agent_name}] No actions produced.",
        agent=agent_name, step_id=step_id,
    )]
