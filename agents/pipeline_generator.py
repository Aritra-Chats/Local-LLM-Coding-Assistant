from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List


class PipelineGenerator(ABC):
    """Abstract base class for the Dynamic Pipeline Generator.

    Transforms a structured plan produced by the Planner Agent into a
    fully executable pipeline — a sequence of steps with resolved agents,
    tools, context requirements, and execution order.
    """

    @abstractmethod
    def generate(self, plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate an executable pipeline from a structured plan.

        Args:
            plan: Ordered list of plan steps from the PlannerAgent.

        Returns:
            An executable pipeline — a list of enriched step dicts ready
            for the ExecutionEngine to process.
        """
        ...

    @abstractmethod
    def enrich_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich a single plan step with execution metadata.

        Adds tool bindings, context hints, model preferences, and
        validation criteria to a step.

        Args:
            step: A single plan step dict.

        Returns:
            The enriched step dict.
        """
        ...

    @abstractmethod
    def validate_pipeline(self, pipeline: List[Dict[str, Any]]) -> bool:
        """Validate the structural integrity of a generated pipeline.

        Checks for missing fields, circular dependencies, and unresolvable
        agent assignments.

        Args:
            pipeline: The generated pipeline list.

        Returns:
            True if the pipeline is valid, False otherwise.
        """
        ...

    @abstractmethod
    def modify(self, pipeline: List[Dict[str, Any]], failure: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Modify an existing pipeline in response to a step failure.

        Args:
            pipeline: The current pipeline list.
            failure: Structured failure report indicating which step failed and why.

        Returns:
            A revised pipeline with the failure addressed.
        """
        ...


# ────────────────────────────────────────────────────────────────────────────

"""concrete_pipeline_generator.py — Concrete PipelineGeneratorAgent.

Implements the :class:`~agents.pipeline_generator.PipelineGenerator`
ABC.  All generation logic is delegated to
:class:`~execution.pipeline.DynamicPipelineGenerator`.

Design contract
---------------
* This agent **never** calls tools directly.
* Its sole responsibility is to receive a plan (list of step dicts) or an
  :class:`~tasks.task_manager.ExecutionPlan` and return an enriched
  pipeline as a list of step dicts **plus** a ``Pipeline`` object and
  supporting :class:`~agents.agent_action.AgentAction` instances.
"""


import traceback
import uuid
from typing import Any, Dict, List, Optional, Tuple

from agents.agent_action import AgentAction
from execution.pipeline import (
    DynamicPipelineGenerator,
    Pipeline,
    PipelineStep,
    PIPELINE_MODES,
    SYSTEM_MODES,
)


class ConcretePipelineGeneratorAgent(PipelineGenerator):
    """Concrete dynamic pipeline generator agent.

    Converts a structured plan into an optimised, executable
    :class:`~execution.pipeline.Pipeline` and exposes the
    result both as a list of enriched step dicts (the ABC contract) and as
    a typed :class:`~execution.pipeline.Pipeline` object for
    callers that need richer access.

    Attributes:
        name: Registry identifier for this agent.
        system_mode: Active hardware profile mode
            (``"minimal"``, ``"standard"``, ``"advanced"``).
        mode: Default pipeline execution mode (``"solo"`` or ``"council"``).
        generator: The :class:`~execution.pipeline.DynamicPipelineGenerator`
            instance used internally.
    """

    name = "pipeline_generator"

    def __init__(
        self,
        system_mode: str = "standard",
        mode: str = "solo",
    ) -> None:
        if system_mode not in SYSTEM_MODES:
            raise ValueError(
                f"Invalid system_mode '{system_mode}'. Must be one of {SYSTEM_MODES}."
            )
        if mode not in PIPELINE_MODES:
            raise ValueError(
                f"Invalid mode '{mode}'. Must be one of {PIPELINE_MODES}."
            )
        self.system_mode = system_mode
        self.mode = mode
        self.generator = DynamicPipelineGenerator(system_mode=system_mode, mode=mode)

    # ------------------------------------------------------------------
    # PipelineGenerator — abstract method implementations
    # ------------------------------------------------------------------

    def generate(self, plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate an executable pipeline from *plan* (list of step dicts).

        Delegates to :class:`~execution.pipeline.DynamicPipelineGenerator`
        and returns the steps list as a list of enriched dicts.

        Args:
            plan: Ordered list of step dicts from the PlannerAgent.
                  Each dict must contain at least ``"name"`` and ``"agent"``.

        Returns:
            List of enriched step dicts ready for the ExecutionEngine.
        """
        pipeline = self.generator.from_steps(plan)
        return [s.to_dict() for s in pipeline.ordered_steps()]

    def enrich_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich a single plan step with execution metadata.

        Adds ``model_hint``, ``max_retries``, ``timeout_seconds``,
        ``can_parallelize``, ``council_agents``, and ``status``.

        Args:
            step: A single step dict.  Must contain at least ``"agent"``.

        Returns:
            Enriched step dict.
        """
        return self.generator.enrich_step(step)

    def validate_pipeline(self, pipeline: List[Dict[str, Any]]) -> bool:
        """Validate the structural integrity of a pipeline (list of step dicts).

        Constructs a temporary :class:`~execution.pipeline.Pipeline`
        and runs :class:`~execution.pipeline.PipelineValidator`.

        Args:
            pipeline: The pipeline as a list of step dicts.

        Returns:
            True if the pipeline is structurally valid, False otherwise.
        """
        temp = self.generator.from_steps(pipeline, goal="(validation)")
        valid, _ = self.generator.validate(temp)
        return valid

    def modify(
        self,
        pipeline: List[Dict[str, Any]],
        failure: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Modify *pipeline* in response to *failure*.

        Builds a temporary :class:`~execution.pipeline.Pipeline`,
        applies :meth:`~execution.pipeline.DynamicPipelineGenerator.modify_for_failure`,
        then returns the revised steps as a list of dicts.

        Args:
            pipeline: Current pipeline as a list of step dicts.
            failure: Failure report dict (see
                :meth:`~execution.pipeline.DynamicPipelineGenerator.modify_for_failure`).

        Returns:
            Revised pipeline as a list of step dicts.
        """
        temp = self.generator.from_steps(pipeline, goal="(recovery)")
        revised = self.generator.modify_for_failure(temp, failure)
        return [s.to_dict() for s in revised.ordered_steps()]

    # ------------------------------------------------------------------
    # Extended API — returns typed Pipeline objects
    # ------------------------------------------------------------------

    def generate_from_plan(
        self,
        execution_plan: Any,  # tasks.task_manager.ExecutionPlan
        task_id: str = "",
        mode: Optional[str] = None,
    ) -> Tuple[Pipeline, List[AgentAction]]:
        """Generate a :class:`~execution.pipeline.Pipeline` from
        a :class:`~tasks.task_manager.ExecutionPlan`.

        This is the preferred entry point when a full
        :class:`~tasks.task_manager.ExecutionPlan` is available.

        Args:
            execution_plan: Fully populated
                :class:`~tasks.task_manager.ExecutionPlan`.
            task_id: Optional originating task identifier.
            mode: Override the instance-level pipeline mode for this call.

        Returns:
            ``(Pipeline, [AgentAction, ...])``
        """
        pipeline = self.generator.from_execution_plan(
            execution_plan, task_id=task_id, mode=mode
        )
        actions = self._build_actions(pipeline)
        return pipeline, actions

    def generate_from_steps(
        self,
        steps: List[Dict[str, Any]],
        goal: str = "",
        classification: Optional[Dict[str, Any]] = None,
        plan_id: str = "",
        task_id: str = "",
        complexity: str = "medium",
        mode: Optional[str] = None,
    ) -> Tuple[Pipeline, List[AgentAction]]:
        """Generate a :class:`~execution.pipeline.Pipeline` from
        a raw list of step dicts.

        Args:
            steps: List of step dicts.
            goal: Human-readable goal string.
            classification: Optional serialised classification dict.
            plan_id: Optional originating plan UUID.
            task_id: Optional originating task identifier.
            complexity: Overall task complexity.
            mode: Override the instance-level pipeline mode for this call.

        Returns:
            ``(Pipeline, [AgentAction, ...])``
        """
        pipeline = self.generator.from_steps(
            steps,
            goal=goal,
            classification=classification,
            plan_id=plan_id,
            task_id=task_id,
            complexity=complexity,
            mode=mode,
        )
        actions = self._build_actions(pipeline)
        return pipeline, actions

    # ------------------------------------------------------------------
    # BaseAgent-style helpers
    # ------------------------------------------------------------------

    def run(
        self,
        task: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate a pipeline from *task* and return a result dict.

        Accepts either an ``"execution_plan"`` key (typed
        :class:`~tasks.task_manager.ExecutionPlan`) or a ``"plan"``
        key (list of step dicts from the PlannerAgent's output).

        Args:
            task: Dict containing one of:

                * ``"execution_plan"`` — an
                  :class:`~tasks.task_manager.ExecutionPlan`.
                * ``"plan"`` — a list of step dicts.
                * ``"goal"`` — used as the pipeline goal string.
                * ``"mode"`` — override pipeline mode for this call.
                * ``"task_id"`` — optional task identifier.

            context: Context payload (currently unused; reserved for future
                context-aware optimisations).

        Returns:
            ``{"status": "ok", "pipeline": Pipeline, "steps": [step_dict, ...],
               "actions": [AgentAction, ...]}``
        """
        step_id = task.get("step_id", str(uuid.uuid4()))
        mode_override = task.get("mode") or None
        task_id = task.get("task_id", "")
        goal = task.get("goal", "")

        if "execution_plan" in task:
            pipeline, actions = self.generate_from_plan(
                task["execution_plan"],
                task_id=task_id,
                mode=mode_override,
            )
        else:
            plan_steps = task.get("plan", [])
            pipeline, actions = self.generate_from_steps(
                plan_steps,
                goal=goal,
                classification=task.get("classification"),
                plan_id=task.get("plan_id", ""),
                task_id=task_id,
                complexity=task.get("complexity", "medium"),
                mode=mode_override,
            )

        return {
            "status": "ok",
            "pipeline": pipeline,
            "steps": [s.to_dict() for s in pipeline.ordered_steps()],
            "actions": actions,
            "task": task,
        }

    def validate_output(self, output: Dict[str, Any]) -> bool:
        return (
            isinstance(output, dict)
            and output.get("status") == "ok"
            and isinstance(output.get("pipeline"), Pipeline)
            and isinstance(output.get("steps"), list)
        )

    def handle_error(
        self,
        error: Exception,
        task: Dict[str, Any],
    ) -> Dict[str, Any]:
        step_id = task.get("step_id", "unknown")
        tb = traceback.format_exc()
        actions = [
            AgentAction.abort(
                reason=f"PipelineGeneratorAgent error: {error}\n{tb}",
                agent=self.name,
                step_id=step_id,
            )
        ]
        return {"status": "error", "actions": actions, "error": str(error), "task": task}

    def describe(self) -> str:
        return (
            f"PipelineGeneratorAgent: converts execution plans into optimised "
            f"pipelines (mode={self.mode}, system={self.system_mode}). "
            f"Assigns agents, model hints, retry budgets, timeouts, parallelism "
            f"flags, and council panels to every step."
        )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _build_actions(self, pipeline: Pipeline) -> List[AgentAction]:
        """Build AgentAction messages summarising the generated pipeline."""
        council_steps = pipeline.council_steps()
        actions: List[AgentAction] = [
            AgentAction.message(
                f"[PipelineGenerator] Generated {pipeline.mode} pipeline "
                f"'{pipeline.pipeline_id[:8]}' "
                f"({pipeline.system_mode}, {len(pipeline.steps)} steps, "
                f"complexity={pipeline.complexity}, "
                f"council_steps={len(council_steps)}).",
                agent=self.name,
            )
        ]

        # Emit a decision action documenting the execution mode chosen.
        actions.append(
            AgentAction.decision(
                choice=pipeline.mode,
                options=list(PIPELINE_MODES),
                rationale=(
                    f"System mode '{pipeline.system_mode}' "
                    + (
                        "disables council."
                        if pipeline.system_mode == "minimal"
                        else f"allows '{pipeline.mode}' execution."
                    )
                ),
                agent=self.name,
            )
        )

        # Emit one delegate action per step pointing at the responsible agent.
        for step in pipeline.ordered_steps():
            actions.append(
                AgentAction.delegate(
                    target_agent=step.agent,
                    task=step.to_dict(),
                    agent=self.name,
                    step_id=step.step_id,
                    rationale=(
                        f"Step '{step.name}' → {step.agent} "
                        f"(model={step.model_hint}, "
                        f"retries={step.max_retries}, "
                        f"timeout={step.timeout_seconds}s, "
                        f"council={step.council_agents or 'none'})."
                    ),
                )
            )

        return actions
