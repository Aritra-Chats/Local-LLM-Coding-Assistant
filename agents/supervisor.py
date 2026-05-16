from __future__ import annotations
import json
import re
import traceback
import uuid
from abc import abstractmethod
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from agents.agent_action import AgentAction
from agents.base_agent import BaseAgent

if TYPE_CHECKING:
    from core.supervisor_bus import SupervisorBus, BusEvent, FixProposal
    from core.async_supervisor import AsyncSupervisorLoop
    from core.execution_engine import ConcreteExecutionEngine
    from cli.progress_tracker import ConcreteProgressTracker


class SupervisorAgent(BaseAgent):
    """Abstract base class for the Supervisor Agent.

    The Supervisor is the top-level orchestrator. It is responsible for
    understanding the user's intent, delegating to the Planner, monitoring
    pipeline progress, and triggering recovery strategies on failure.
    """

    @abstractmethod
    def delegate(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Delegate a parsed task to the Planner Agent for decomposition.

        Args:
            task: The structured task parsed from the user prompt.

        Returns:
            The structured plan produced by the Planner.
        """
        ...

    @abstractmethod
    def monitor(self, pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor the execution state of an active pipeline.

        Args:
            pipeline_state: Current state snapshot of the running pipeline.

        Returns:
            Updated state dict with monitoring annotations.
        """
        ...

    @abstractmethod
    def recover(self, failure: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Trigger a recovery strategy when a pipeline step fails.

        Args:
            failure: Structured failure report including error details and step info.
            pipeline_state: Current state of the pipeline at the time of failure.

        Returns:
            A recovery action dict (retry, modify pipeline, switch model, or abort).
        """
        ...

    @abstractmethod
    def parse_prompt(self, prompt: str) -> Dict[str, Any]:
        """Parse a raw user prompt into a structured task definition.

        Args:
            prompt: The raw user input string.

        Returns:
            A structured task dict with goal, constraints, and metadata.
        """
        ...


# ────────────────────────────────────────────────────────────────────────────

"""concrete_supervisor.py — Concrete SupervisorAgent implementation.

The SupervisorAgent is the top-level orchestrator of the Sentinel hierarchy.
It is responsible for:

1. Parsing a raw user prompt into a structured task.
2. Delegating the task to the PlannerAgent via a ``delegate`` action.
3. Monitoring pipeline progress and injecting ``monitor`` checkpoints.
4. Triggering recovery strategies when a pipeline step reports failure.

Design contract
---------------
* The Supervisor **never** calls tools directly.
* All side effects are expressed as :class:`~agents.agent_action.AgentAction`
  instances returned from :py:meth:`run`.
* The ExecutionEngine (caller) is solely responsible for dispatching actions.
"""


# ---------------------------------------------------------------------------
# Complexity heuristics
# ---------------------------------------------------------------------------

_HIGH_COMPLEXITY_KEYWORDS = frozenset(
    {
        "refactor",
        "architecture",
        "migrate",
        "optimise",
        "optimize",
        "benchmark",
        "security audit",
        "upgrade",
        "pipeline",
    }
)
_LOW_COMPLEXITY_KEYWORDS = frozenset(
    {"explain", "summarise", "summarize", "describe", "what is", "show", "list"}
)

_HIGH_LENGTH_THRESHOLD = 300
_COMPLEX_LENGTH_THRESHOLD = 600


def _estimate_complexity(goal: str) -> str:
    lower = goal.lower()
    plen = len(goal)

    # 1. High-keyword + long goal -> complex
    if any(k in lower for k in _HIGH_COMPLEXITY_KEYWORDS) and plen >= _HIGH_LENGTH_THRESHOLD:
        return "complex"

    # 2. Extremely long goals -> complex
    if plen >= _COMPLEX_LENGTH_THRESHOLD:
        return "complex"

    # 3. High-keyword present -> high
    if any(k in lower for k in _HIGH_COMPLEXITY_KEYWORDS):
        return "high"

    # 4. Moderately long goals -> high
    if plen >= _HIGH_LENGTH_THRESHOLD:
        return "high"

    # 5. Low-keyword present -> low
    if any(k in lower for k in _LOW_COMPLEXITY_KEYWORDS):
        return "low"

    # Default
    return "medium"


# ---------------------------------------------------------------------------
# Concrete implementation
# ---------------------------------------------------------------------------


_SUPERVISOR_PARSE_PROMPT = """\
You are a senior software engineering assistant. Parse the user's request and extract a structured task.

User request: {prompt}

Respond ONLY with a valid JSON object with these exact keys:
{{
  "goal": "<concise one-line summary of what needs to be done>",
    "complexity": "<one of: low, medium, high, complex>",
  "constraints": ["<any stated constraints or requirements>"],
  "task_category": "<one of: coding, debugging, reasoning, devops, research, system>",
  "affected_files": ["<list any specific files mentioned, or empty list>"],
  "language": "<primary programming language if relevant, else empty string>"
}}

Rules:
- goal must be action-oriented and specific (not just a restatement)
 - complexity: one of low, medium, high, complex — assessed holistically from the goal and context, not by keyword matching
- No prose before or after the JSON object."""


class ConcreteSupervisorAgent(SupervisorAgent):
    """Concrete top-level orchestrator agent.

    Attributes:
        name: Registry identifier for this agent.
        max_retries: Maximum number of recovery attempts before aborting.
        ollama_client: Optional OllamaClient for LLM-driven prompt parsing.
        model: Ollama model tag used by this agent.
    """

    name = "supervisor"

    def __init__(
        self,
        max_retries: int = 2,
        ollama_client: Optional[Any] = None,
        model: str = "",
    ) -> None:
        super().__init__()
        self.max_retries = max_retries
        self._ollama = ollama_client
        self._model = model

    # ------------------------------------------------------------------
    # BaseAgent — required overrides
    # ------------------------------------------------------------------

    def run(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate the task and return generated actions.

        The Supervisor:
        1. Validates the incoming task structure.
        2. Generates a ``delegate`` action aimed at the PlannerAgent.
        3. Generates a ``message`` action summarising its intent.

        Args:
            task: Structured task dict (must contain at least ``"goal"``).
            context: Context payload from ContextBuilder.

        Returns:
            ``{"status": "ok", "actions": [AgentAction, ...], "task": task}``
        """
        step_id = task.get("step_id") or str(uuid.uuid4())
        task.setdefault("step_id", step_id)
        task.setdefault("complexity", _estimate_complexity(task.get("goal", "")))

        actions: List[AgentAction] = [
            AgentAction.message(
                f"[Supervisor] Received task: {task.get('goal', '(no goal)')}",
                agent=self.name,
                step_id=step_id,
            ),
            AgentAction.delegate(
                target_agent="planner",
                task=task,
                agent=self.name,
                step_id=step_id,
                rationale="Delegating to PlannerAgent for decomposition.",
            ),
        ]
        return {"status": "ok", "actions": actions, "task": task}

    def validate_output(self, output: Dict[str, Any]) -> bool:
        """Verify that the output contains a non-empty actions list."""
        return (
            isinstance(output, dict)
            and output.get("status") == "ok"
            and isinstance(output.get("actions"), list)
            and len(output["actions"]) > 0
        )

    def handle_error(self, error: Exception, task: Dict[str, Any]) -> Dict[str, Any]:
        """Emit an abort action carrying the traceback."""
        step_id = task.get("step_id", "unknown")
        tb = traceback.format_exc()
        actions = [
            AgentAction.abort(
                reason=f"SupervisorAgent error: {error}\n{tb}",
                agent=self.name,
                step_id=step_id,
            )
        ]
        return {"status": "error", "actions": actions, "error": str(error), "task": task}

    def describe(self) -> str:
        return (
            "SupervisorAgent: top-level orchestrator.  Parses user prompts, "
            "delegates to the PlannerAgent, monitors pipeline progress, and "
            "initiates recovery strategies on failure."
        )

    # ------------------------------------------------------------------
    # SupervisorAgent — abstract method implementations
    # ------------------------------------------------------------------

    def parse_prompt(self, prompt: str) -> Dict[str, Any]:
        """Parse a raw user prompt into a structured task dict.

        When an OllamaClient is available, uses the LLM to extract a rich
        structured task dict (goal, complexity, task_category, constraints,
        affected_files, language).  Falls back to regex-based extraction
        when Ollama is unavailable or the model returns an unusable response.

        Model self-healing
        ------------------
        If the first attempt returns HTTP 404 (model not installed) the
        method re-queries ``/api/tags``, picks the best available installed
        model, and retries exactly once.  This prevents the misleading
        "LLM task parsing failed" message when the catalogue model hasn't
        been pulled yet.

        Args:
            prompt: Raw user input string.

        Returns:
            ``{"goal": str, "raw_prompt": str, "complexity": str,
               "constraints": list, "task_category": str,
               "affected_files": list, "language": str, "step_id": str}``
        """
        import sys
        prompt = prompt.strip()
        step_id = str(uuid.uuid4())

        # ── LLM-driven extraction ────────────────────────────────────────────────
        # In online mode the execution engine may have injected a cloud client
        # via use_client().  Prefer it over the local Ollama client so the
        # supervisor itself benefits from the cloud model for task parsing.
        _active_client = self._inference_client or self._ollama
        _active_model  = self._model
        if _active_client and _active_model:
            # Try up to 2 times: first with _active_model, then with the
            # best actually-installed model if the first call 404s.
            models_to_try = [_active_model]
            tried: set = set()

            for attempt_model in models_to_try:
                if attempt_model in tried:
                    continue
                tried.add(attempt_model)
                try:
                    llm_prompt = _SUPERVISOR_PARSE_PROMPT.format(prompt=prompt)
                    response = _active_client.generate(
                        model=attempt_model,
                        prompt=llm_prompt,
                        # 60 s is enough for a 7B model on modest hardware;
                        # prevents the process from hanging on first use.
                        timeout=60,
                        options={"num_predict": 512, "temperature": 0.1},
                    )
                    raw = response.get("response", "").strip()
                    if raw.startswith("```"):
                        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw).rstrip("` \n")
                    parsed = json.loads(raw)
                    self._model = attempt_model  # remember working tag
                    return {
                        "goal":           parsed.get("goal", prompt[:200]),
                        "raw_prompt":     prompt,
                        "complexity":     parsed.get("complexity", "medium"),
                        "constraints":    parsed.get("constraints", []),
                        "task_category":  parsed.get("task_category", "coding"),
                        "affected_files": parsed.get("affected_files", []),
                        "language":       parsed.get("language", ""),
                        "step_id":        step_id,
                        "_parsed_by":     "llm",
                        "_model_used":    attempt_model,
                    }

                except Exception as _llm_err:
                    err_str = str(_llm_err)
                    is_404 = (
                        "404" in err_str
                        or "Not Found" in err_str
                        or "model not found" in err_str.lower()
                    )

                    if is_404 and len(tried) == 1:
                        # Self-heal: discover installed local models and retry once.
                        # Only applicable for local Ollama clients; cloud clients
                        # serve model names from their own catalogue and don't 404.
                        try:
                            from core.model_router import ConcreteModelRouter
                            ConcreteModelRouter.invalidate_model_cache()
                            installed = self._ollama.list_models() if self._ollama else []
                        except Exception:
                            installed = []

                        if installed:
                            _pref = [
                                "codellama", "mistral", "llama", "phi", "gemma", "qwen"
                            ]
                            best = installed[0]
                            for pref in _pref:
                                hits = [t for t in installed if pref in t.lower()]
                                if hits:
                                    best = hits[0]
                                    break
                            if best not in tried:
                                print(
                                    f"[SupervisorAgent] Model '{attempt_model}'"
                                    f" not installed — retrying with '{best}'.",
                                    file=sys.stderr,
                                )
                                models_to_try.append(best)
                                continue

                    # Non-404 errors (JSON parse fail, timeout, etc.) fall through
                    print(
                        f"[SupervisorAgent] LLM task parsing failed "
                        f"({type(_llm_err).__name__}: {_llm_err}) "
                        f"— falling back to rule-based parsing.",
                        file=sys.stderr,
                    )
                    break

        # ── Rule-based fallback ──────────────────────────────────────────────────
        goal_match = re.match(r"([^!?\n]+)[!?\n]?", prompt)
        goal = goal_match.group(1).strip() if goal_match else prompt[:200]
        constraints = re.findall(r"\[constraint:\s*([^\]]+)\]", prompt, re.IGNORECASE)

        return {
            "goal":           goal,
            "raw_prompt":     prompt,
            "complexity":     _estimate_complexity(goal),
            "constraints":    constraints,
            "task_category":  "coding",
            "affected_files": [],
            "language":       "",
            "step_id":        step_id,
            "_parsed_by":     "regex",
        }

    def delegate(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a ``delegate`` action targeting the PlannerAgent.

        Args:
            task: Structured task dict.

        Returns:
            ``{"actions": [AgentAction], "target": "planner"}``
        """
        action = AgentAction.delegate(
            target_agent="planner",
            task=task,
            agent=self.name,
            step_id=task.get("step_id", ""),
            rationale="Supervisor delegating task to PlannerAgent.",
        )
        return {"actions": [action], "target": "planner"}

    def monitor(self, pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Inspect a pipeline snapshot and emit monitoring annotations.

        Generates a ``message`` action summarising current pipeline progress.
        If any step has ``"status": "failed"``, generates an additional
        ``decision`` action flagging the failure for recovery consideration.

        Args:
            pipeline_state: Dict with ``"steps"`` list and ``"current_step"`` index.

        Returns:
            Pipeline state dict extended with ``"monitor_actions"`` key.
        """
        step_id = pipeline_state.get("step_id", "")
        steps: List[Dict] = pipeline_state.get("steps", [])
        current = pipeline_state.get("current_step", 0)
        total = len(steps)

        monitor_actions: List[AgentAction] = [
            AgentAction.message(
                f"[Monitor] Step {current}/{total} in progress.",
                agent=self.name,
                step_id=step_id,
            )
        ]

        failed = [s for s in steps if s.get("status") == "failed"]
        if failed:
            failed_names = [s.get("name", "?") for s in failed]
            monitor_actions.append(
                AgentAction.decision(
                    choice="recover",
                    options=["recover", "abort", "skip"],
                    rationale=f"Failed steps detected: {failed_names}",
                    agent=self.name,
                    step_id=step_id,
                )
            )

        pipeline_state["monitor_actions"] = [a.to_dict() for a in monitor_actions]
        return pipeline_state

    def recover(
        self, failure: Dict[str, Any], pipeline_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate recovery actions for a failed pipeline step.

        Recovery strategy:
        * Attempt retry up to ``self.max_retries`` times.
        * If retries are exhausted, emit an ``abort`` action.

        Args:
            failure: ``{"step": step_dict, "error": str, "attempt": int}``
            pipeline_state: Current pipeline state.

        Returns:
            ``{"actions": [AgentAction], "strategy": str}``
        """
        attempt = failure.get("attempt", 1)
        step = failure.get("step", {})
        step_id = step.get("step_id", pipeline_state.get("step_id", ""))
        error_msg = failure.get("error", "unknown error")

        if attempt <= self.max_retries:
            action = AgentAction.delegate(
                target_agent=step.get("agent", "planner"),
                task={**step, "attempt": attempt + 1, "step_id": step_id},
                agent=self.name,
                step_id=step_id,
                rationale=f"Recovery attempt {attempt}/{self.max_retries} for: {error_msg}",
            )
            return {"actions": [action], "strategy": "retry"}

        action = AgentAction.abort(
            reason=f"Max retries ({self.max_retries}) exhausted for step '{step.get('name', '?')}': {error_msg}",
            agent=self.name,
            step_id=step_id,
        )
        return {"actions": [action], "strategy": "abort"}

    # ------------------------------------------------------------------
    # Async supervisor integration
    # ------------------------------------------------------------------

    _DIAGNOSIS_PROMPT = """\
You are a senior software engineer diagnosing a build failure in an automated
coding pipeline.

## Failed step
Name: {step_name}
Agent: {agent}

## Tool that failed
Tool: {tool_name}
Parameters: {params_json}

## Error output
{error}

## System context
Available tools in this pipeline: run_shell, write_file, read_file, find_files,
install_dependency, project_initializer, run_tests, git_commit

## Recent shell command history
{shell_history}

## Instructions
Analyse the error and propose the minimum sequence of tool calls needed to fix
the root cause so the original step can succeed on retry.

Respond ONLY with a valid JSON object:
{{
  "root_cause": "<one sentence>",
  "fix_possible": true | false,
  "fix_rationale": "<why this fix will work>",
  "fix_actions": [
    {{
      "tool": "<tool_name>",
      "params": {{ }},
      "rationale": "<why this specific call>"
    }}
  ]
}}

Rules:
- fix_actions must be ordered (each action may depend on prior ones)
- If fix_possible is false, fix_actions must be empty
- Never suggest re-running the original failed tool as a fix action
- If the required recovery would repeat a shell command already present in the
    history, skip it instead of proposing it again.
- Prefer install_dependency over run_shell for package installation
- Maximum 4 fix actions
"""

    _SHELL_REVIEW_PROMPT = """\
You are a senior software engineer reviewing a proposed shell-like tool call
inside an automated coding pipeline.

## Tool call
Tool: {tool_name}
Parameters: {params_json}

## Recent shell command history
{shell_history}

## System context
{context_json}

## Instructions
Decide whether the command should run, be skipped because it is already in
history, or be treated as a bounded watch/check command.

Rules:
- If the command is already present in the shell command history, return
    decision="skip" and do not repeat it.
- If Tool is "project_initializer" and the context indicates an existing
    stack/known files, return decision="skip".
- If the command is a long-lived dev/watch process such as a local app server,
    return decision="bounded_run" with timeout_seconds between 60 and 120 and
    a short display label such as "Checking compilation".
- Otherwise return decision="run".
- Do not invent new shell commands unless you return decision="run".

Respond ONLY with valid JSON:
{{
    "decision": "run" | "skip" | "bounded_run",
    "timeout_seconds": 120,
    "display_label": "<short UI label>",
    "reason": "<one sentence>",
    "rewrite_command": "<optional command text>"
}}
"""

    def diagnose_failure(self, event: "BusEvent") -> "FixProposal":
        """Call the LLM with a structured diagnosis prompt and return a FixProposal.

        On LLM failure or JSON parse error returns a FixProposal with
        ``fix_possible=False`` so the caller can abort cleanly.

        Args:
            event: The :class:`~core.supervisor_bus.BusEvent` that triggered
                the diagnosis.

        Returns:
            :class:`~core.supervisor_bus.FixProposal` with AgentAction instances
            ready to inject into the engine.
        """
        import sys
        from core.supervisor_bus import FixProposal

        step_name = event.step_name
        tool_name = event.tool_name
        error     = event.error
        params    = event.extra.get("params", {})
        step_id   = event.step_id

        # Active client: prefer the injected inference client, fall back to Ollama.
        _client = self._inference_client or self._ollama
        _model  = self._model

        if not _client or not _model:
            return FixProposal(
                fix_actions=[],
                rationale="No LLM client available for diagnosis.",
                fix_possible=False,
            )

        try:
            shell_history = json.dumps(
                event.context.get("shell_command_history", []) or [],
                indent=2,
                default=str,
            )[:1200]
            prompt_text = self._DIAGNOSIS_PROMPT.format(
                step_name=step_name,
                agent=event.extra.get("agent", "unknown"),
                tool_name=tool_name,
                params_json=json.dumps(params, indent=2, default=str)[:800],
                error=error[:1200],
                shell_history=shell_history,
            )
            response = _client.generate(
                model=_model,
                prompt=prompt_text,
                timeout=60,
                options={"num_predict": 1024, "temperature": 0.1},
            )
            raw = response.get("response", "").strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw).rstrip("` \n")

            parsed = json.loads(raw)

            if not parsed.get("fix_possible", False):
                return FixProposal(
                    fix_actions=[],
                    rationale=parsed.get("root_cause", "No fix found by supervisor."),
                    fix_possible=False,
                )

            # Convert raw action dicts → AgentAction instances
            fix_actions = []
            for raw_action in parsed.get("fix_actions", [])[:4]:
                act = AgentAction.tool_call(
                    tool=raw_action["tool"],
                    params=raw_action.get("params", {}),
                    agent=self.name,
                    step_id=step_id,
                    rationale=raw_action.get("rationale", ""),
                )
                fix_actions.append(act)

            return FixProposal(
                fix_actions=fix_actions,
                rationale=parsed.get("fix_rationale", parsed.get("root_cause", "")),
                retry_original=True,
                fix_possible=True,
            )

        except Exception as exc:
            print(
                f"[SupervisorAgent] diagnose_failure failed "
                f"({type(exc).__name__}: {exc}) — no fix available.",
                file=sys.stderr,
            )
            return FixProposal(
                fix_actions=[],
                rationale=f"Diagnosis error: {exc}",
                fix_possible=False,
            )

    def review_shell_tool_call(self, tool_name: str, params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Ask the model whether a shell-like tool call should run now.

        The response is a small runtime policy dict that the execution engine
        can use to skip duplicates or bound watch processes.
        """
        import sys

        _client = self._inference_client or self._ollama
        _model = self._model
        if not _client or not _model:
            return {}

        try:
            shell_history = context.get("shell_command_history", []) or []
            prompt_text = self._SHELL_REVIEW_PROMPT.format(
                tool_name=tool_name,
                params_json=json.dumps(params, indent=2, default=str)[:1200],
                shell_history=json.dumps(shell_history, indent=2, default=str)[:1200],
                context_json=json.dumps(
                    {
                        "step_name": context.get("step_name", ""),
                        "step_index": context.get("step_index", -1),
                        "project_root": context.get("project_root", ""),
                        "workspace_snapshot": context.get("workspace_snapshot", {}),
                        "known_files_count": len(context.get("known_files", []) or []),
                    },
                    indent=2,
                    default=str,
                )[:1200],
            )
            response = _client.generate(
                model=_model,
                prompt=prompt_text,
                timeout=45,
                options={"num_predict": 512, "temperature": 0.1},
            )
            raw = response.get("response", "").strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw).rstrip("` \n")
            parsed = json.loads(raw)
            decision = str(parsed.get("decision", "")).strip().lower()
            timeout_seconds = parsed.get("timeout_seconds", 120)
            try:
                timeout_seconds = max(1, int(timeout_seconds))
            except (TypeError, ValueError):
                timeout_seconds = 120
            display_label = str(parsed.get("display_label", "")).strip()
            reason = str(parsed.get("reason", "")).strip()
            rewrite_command = str(parsed.get("rewrite_command", "")).strip()
            if decision not in {"run", "skip", "bounded_run"}:
                decision = "run"
            return {
                "decision": decision,
                "timeout_seconds": timeout_seconds,
                "display_label": display_label,
                "reason": reason,
                "rewrite_command": rewrite_command,
            }
        except Exception as exc:
            print(
                f"[SupervisorAgent] review_shell_tool_call failed "
                f"({type(exc).__name__}: {exc}) — defaulting to run.",
                file=sys.stderr,
            )
            return {}

    def start_async_monitoring(
        self,
        bus: "SupervisorBus",
        engine: "ConcreteExecutionEngine",
        tracker: Optional["ConcreteProgressTracker"] = None,
    ) -> "AsyncSupervisorLoop":
        """Create, start, and return the :class:`~core.async_supervisor.AsyncSupervisorLoop`.

        Args:
            bus: The shared :class:`~core.supervisor_bus.SupervisorBus`.
            engine: The running :class:`~core.execution_engine.ConcreteExecutionEngine`.
            tracker: Optional progress tracker for live UI label updates.

        Returns:
            The started (daemon thread running) :class:`~core.async_supervisor.AsyncSupervisorLoop`.
        """
        from core.async_supervisor import AsyncSupervisorLoop
        loop = AsyncSupervisorLoop(supervisor=self, bus=bus, engine=engine, tracker=tracker)
        loop.start()
        return loop
# Changed agents/supervisor.py
