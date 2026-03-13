from __future__ import annotations
import json
import re
import traceback
import uuid
from abc import abstractmethod
from typing import Any, Dict, List, Optional

from agents.agent_action import AgentAction
from agents.base_agent import BaseAgent


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


def _estimate_complexity(goal: str) -> str:
    lower = goal.lower()
    if any(k in lower for k in _HIGH_COMPLEXITY_KEYWORDS):
        return "high"
    if any(k in lower for k in _LOW_COMPLEXITY_KEYWORDS):
        return "low"
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
  "complexity": "<one of: low, medium, high>",
  "constraints": ["<any stated constraints or requirements>"],
  "task_category": "<one of: coding, debugging, reasoning, devops, research, system>",
  "affected_files": ["<list any specific files mentioned, or empty list>"],
  "language": "<primary programming language if relevant, else empty string>"
}}

Rules:
- goal must be action-oriented and specific (not just a restatement)
- complexity: low=simple change, medium=multi-file/multi-step, high=architectural/large-scale
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
        if self._ollama and self._model:
            # Try up to 2 times: first with self._model, then with the
            # best actually-installed model if the first call 404s.
            models_to_try = [self._model]
            tried: set = set()

            for attempt_model in models_to_try:
                if attempt_model in tried:
                    continue
                tried.add(attempt_model)
                try:
                    llm_prompt = _SUPERVISOR_PARSE_PROMPT.format(prompt=prompt)
                    response = self._ollama.generate(
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
                        # Self-heal: discover installed models and retry once
                        try:
                            from core.model_router import ConcreteModelRouter
                            ConcreteModelRouter.invalidate_model_cache()
                            installed = self._ollama.list_models()
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
