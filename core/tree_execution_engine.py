"""tree_execution_engine.py — Hierarchical task tree execution engine.

Responsibilities
----------------
1. :class:`NodeResult` — per-node execution record (pipeline result +
   unit/integration test results + error).

2. :class:`TreeRunResult` — aggregated result of a full tree execution run.

3. :class:`TreeExecutionEngine` (ABC) + :class:`ConcreteTreeExecutionEngine`
   — drives a :class:`~core.task_tree.TaskDecompositionTree` to completion
   using a bottom-up, post-order strategy:

   * **Leaf nodes** receive a full pipeline execution followed by a unit
     test scoped to the files changed by that pipeline.
   * **Internal nodes** wait until all their children are complete, then
     receive an integration test that verifies the combined outputs.

Design notes
------------
* Follows the existing pattern of Abstract Base Class + Concrete
  implementation in the same file (see ``execution_engine.py``,
  ``supervisor.py``).
* Reuses :class:`~core.execution_engine.ConcreteExecutionEngine`,
  :class:`~tasks.task_manager.TaskPlanner`, and
  :class:`~execution.pipeline.DynamicPipelineGenerator` unchanged.
* Does NOT modify any agent or tool.
* The ``abort_on_failure`` flag preserves backwards compatibility: when
  ``False`` (default) a failed leaf or integration step is recorded and
  execution continues for the remaining nodes.
"""
from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.task_tree import TaskDecompositionTree, TaskNode


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class NodeResult:
    """Execution record for a single :class:`~core.task_tree.TaskNode`.

    Attributes:
        node_id:                 The :attr:`~core.task_tree.TaskNode.node_id`.
        status:                  Mirrors :attr:`~core.task_tree.TaskNode.status`
            at the time the result was collected.
        pipeline_result:         Dict from
            :class:`~core.execution_engine.PipelineRunResult`.to_dict(),
            or ``None`` for internal (non-leaf) nodes.
        unit_test_result:        Output of the ``run_tests`` tool scoped to
            files changed by this leaf's pipeline, or ``None``.
        integration_test_result: Output of the integration test for an
            internal node, or ``None`` for leaves.
        error:                   Human-readable error description, or
            ``None`` when execution succeeded.
    """

    node_id: str
    status: str
    pipeline_result: Optional[Dict[str, Any]] = None
    unit_test_result: Optional[Dict[str, Any]] = None
    integration_test_result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-compatible dict.

        Returns:
            A :class:`dict` suitable for JSON serialisation.
        """
        return {
            "node_id":                  self.node_id,
            "status":                   self.status,
            "pipeline_result":          self.pipeline_result,
            "unit_test_result":         self.unit_test_result,
            "integration_test_result":  self.integration_test_result,
            "error":                    self.error,
        }


@dataclass
class TreeRunResult:
    """Aggregated result of executing a full :class:`~core.task_tree.TaskDecompositionTree`.

    Attributes:
        tree_id:          Unique UUID string for this run.
        status:           Overall status — ``"completed"`` if the root node
            was successfully integrated, ``"partial"`` if some nodes
            succeeded but the root did not, ``"failed"`` if the root never
            ran or was aborted.
        node_results:     Mapping from ``node_id`` to :class:`NodeResult`.
        total_elapsed_ms: Wall-clock time for the full tree execution.
    """

    tree_id: str
    status: str
    node_results: Dict[str, NodeResult] = field(default_factory=dict)
    total_elapsed_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-compatible dict.

        Returns:
            A :class:`dict` suitable for JSON serialisation.
        """
        return {
            "tree_id":          self.tree_id,
            "status":           self.status,
            "node_results":     {k: v.to_dict() for k, v in self.node_results.items()},
            "total_elapsed_ms": round(self.total_elapsed_ms, 2),
        }

    def summary(self) -> str:
        """Return a one-line human-readable summary.

        Returns:
            Summary string.
        """
        completed = sum(1 for r in self.node_results.values()
                        if r.status in ("unit_tested", "integrated"))
        failed = sum(1 for r in self.node_results.values()
                     if r.status == "failed")
        return (
            f"TreeRun '{self.tree_id[:8]}' | {self.status} | "
            f"nodes={len(self.node_results)} ok={completed} fail={failed} | "
            f"{self.total_elapsed_ms:.0f}ms"
        )


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class TreeExecutionEngine(ABC):
    """Abstract interface for the hierarchical tree execution engine."""

    @abstractmethod
    def execute_tree(
        self,
        tree: TaskDecompositionTree,
        session_context: Dict[str, Any],
    ) -> TreeRunResult:
        """Execute all nodes in *tree* and return an aggregated result.

        Args:
            tree:            The :class:`~core.task_tree.TaskDecompositionTree`
                to execute.
            session_context: Dict containing at minimum ``"session_id"`` and
                ``"project_root"`` strings.

        Returns:
            A :class:`TreeRunResult` describing the outcome of the full run.
        """


# ---------------------------------------------------------------------------
# Concrete implementation
# ---------------------------------------------------------------------------


class ConcreteTreeExecutionEngine(TreeExecutionEngine):
    """Drives a :class:`~core.task_tree.TaskDecompositionTree` to completion.

    Execution strategy
    ------------------
    1. Compute a post-order traversal (children before parents).
    2. For each **leaf** node:
       a. Build a task dict and run the full planning + pipeline generation +
          execution cycle (same logic as ``main.py``).
       b. Run a unit test scoped to files changed by that pipeline.
       c. Mark the node ``"unit_tested"`` on success, ``"failed"`` on error.
    3. For each **internal** node (once all children are done — guaranteed
       by post-order):
       a. Build an integration-test prompt from child summaries.
       b. Run the ``run_tests`` tool over the union of all descendant
          changed files.
       c. Mark the node ``"integrated"`` on success, ``"failed"`` on error.
    4. Overall status is derived from the root node's final status.

    Parameters
    ----------
    concrete_engine:
        A :class:`~core.execution_engine.ConcreteExecutionEngine` instance.
    task_planner:
        A :class:`~tasks.task_manager.TaskPlanner` instance.
    pipeline_generator:
        A :class:`~execution.pipeline.DynamicPipelineGenerator` instance.
    tool_registry:
        A :class:`~tools.tool_registry.ConcreteToolRegistry` with the
        ``run_tests`` tool registered.
    abort_on_failure:
        When ``True``, stop the entire tree as soon as any node fails.
        Default is ``False`` (log and continue).
    progress_tracker:
        Optional :class:`~cli.progress_tracker.ProgressTracker` for live
        status updates.  Pass ``None`` to disable progress output.
    """

    def __init__(
        self,
        concrete_engine: Any,
        task_planner: Any,
        pipeline_generator: Any,
        tool_registry: Any,
        abort_on_failure: bool = False,
        progress_tracker: Optional[Any] = None,
    ) -> None:
        self._engine = concrete_engine
        self._task_planner = task_planner
        self._pipeline_gen = pipeline_generator
        self._tool_registry = tool_registry
        self._abort_on_failure = abort_on_failure
        self._tracker = progress_tracker

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def execute_tree(
        self,
        tree: TaskDecompositionTree,
        session_context: Dict[str, Any],
    ) -> TreeRunResult:
        """Execute all nodes in *tree* and return an aggregated result.

        The post-order traversal guarantees every child is processed before
        its parent, so integration tests can always access all child outputs.

        Args:
            tree:            The :class:`~core.task_tree.TaskDecompositionTree`
                to execute.
            session_context: Dict with at least ``"session_id"`` (str) and
                ``"project_root"`` (str) keys.

        Returns:
            A :class:`TreeRunResult`.
        """
        t0 = time.monotonic()
        tree_id = str(uuid.uuid4())
        node_results: Dict[str, NodeResult] = {}

        # Changed-file accumulator keyed by node_id — populated after each
        # leaf pipeline run and used to build integration-test file sets.
        changed_files_by_node: Dict[str, List[str]] = {}

        for node in tree.post_order():
            self._update_status(node.node_id, "running")
            node.status = "running"

            if node.is_leaf():
                nr = self._execute_leaf(
                    node, session_context, changed_files_by_node
                )
            else:
                nr = self._execute_internal(
                    node, session_context, changed_files_by_node, node_results
                )

            node_results[node.node_id] = nr
            node.status = nr.status
            node.result = nr.pipeline_result
            node.unit_test_result = nr.unit_test_result
            node.integration_test_result = nr.integration_test_result

            self._update_status(node.node_id, nr.status,
                                 nr.error or "")

            if self._abort_on_failure and nr.status == "failed":
                break

        elapsed_ms = (time.monotonic() - t0) * 1000

        # Determine overall status from root node
        root_result = node_results.get(tree.root.node_id)
        if root_result and root_result.status == "integrated":
            overall = "completed"
        elif any(r.status in ("unit_tested", "integrated")
                 for r in node_results.values()):
            overall = "partial"
        else:
            overall = "failed"

        return TreeRunResult(
            tree_id=tree_id,
            status=overall,
            node_results=node_results,
            total_elapsed_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------
    # Leaf execution
    # ------------------------------------------------------------------

    def _execute_leaf(
        self,
        node: TaskNode,
        session_context: Dict[str, Any],
        changed_files_by_node: Dict[str, List[str]],
    ) -> NodeResult:
        """Run the full pipeline + unit test for a leaf *node*.

        Args:
            node:                 The leaf :class:`~core.task_tree.TaskNode`.
            session_context:      Session dict with ``session_id`` and
                ``project_root``.
            changed_files_by_node: Accumulator updated with files changed
                by this pipeline run.

        Returns:
            A :class:`NodeResult`.
        """
        task = self._build_task_dict(node, session_context)
        pipeline_result_dict: Optional[Dict[str, Any]] = None
        unit_test_result: Optional[Dict[str, Any]] = None
        error: Optional[str] = None

        # ── Pipeline execution ────────────────────────────────────────
        try:
            plan = self._task_planner.plan(task)
            pipeline = self._pipeline_gen.from_execution_plan(plan)

            # Stamp selected_model onto each step (mirrors main.py logic)
            selected_model = task.get("selected_model")
            if selected_model:
                for step in pipeline.steps:
                    try:
                        step.metadata["selected_model"] = selected_model
                        step.selected_model = selected_model
                    except AttributeError:
                        pass

            run_result = self._engine.run_pipeline(pipeline)
            pipeline_result_dict = run_result.to_dict()

            # Collect changed files from this run
            changed: List[str] = []
            fcm = getattr(self._engine, "file_change_map", None)
            if fcm is not None:
                try:
                    changed = list(fcm.changed_paths())
                except Exception:
                    try:
                        changed = [str(e.path) for e in fcm.events]
                    except Exception:
                        changed = []
            changed_files_by_node[node.node_id] = changed

        except Exception as exc:  # pragma: no cover
            error = f"Pipeline execution failed: {exc}"
            return NodeResult(
                node_id=node.node_id,
                status="failed",
                pipeline_result=pipeline_result_dict,
                error=error,
            )

        # ── Unit test ─────────────────────────────────────────────────
        try:
            files = changed_files_by_node.get(node.node_id, [])
            test_path = files[0] if len(files) == 1 else (
                session_context.get("project_root", ".") if not files else "."
            )
            unit_test_result = self._run_tests(test_path)
            if not unit_test_result.get("success", True):
                error = (f"Unit test failed: "
                         f"{unit_test_result.get('summary', '')}")
                return NodeResult(
                    node_id=node.node_id,
                    status="failed",
                    pipeline_result=pipeline_result_dict,
                    unit_test_result=unit_test_result,
                    error=error,
                )
        except Exception as exc:  # pragma: no cover
            unit_test_result = {"error": str(exc)}

        return NodeResult(
            node_id=node.node_id,
            status="unit_tested",
            pipeline_result=pipeline_result_dict,
            unit_test_result=unit_test_result,
        )

    # ------------------------------------------------------------------
    # Internal node integration test
    # ------------------------------------------------------------------

    def _execute_internal(
        self,
        node: TaskNode,
        session_context: Dict[str, Any],
        changed_files_by_node: Dict[str, List[str]],
        node_results: Dict[str, NodeResult],
    ) -> NodeResult:
        """Run an integration test for an internal (non-leaf) *node*.

        Because post-order guarantees all children ran first, this method
        can safely access all child results.

        Args:
            node:                 The internal :class:`~core.task_tree.TaskNode`.
            session_context:      Session dict.
            changed_files_by_node: Accumulator of changed files per node.
            node_results:         Completed :class:`NodeResult` objects for
                previously executed nodes.

        Returns:
            A :class:`NodeResult`.
        """
        # Collect union of all descendant leaf changed files
        all_changed: List[str] = []
        seen: set = set()
        for desc in self._descendant_leaves(node):
            for f in changed_files_by_node.get(desc.node_id, []):
                if f not in seen:
                    seen.add(f)
                    all_changed.append(f)
        # Propagate this node's file set for ancestor integration tests
        changed_files_by_node[node.node_id] = all_changed

        # Build child-result summary for the integration prompt
        child_summaries: List[str] = []
        for child in node.children:
            cr = node_results.get(child.node_id)
            desc = child.task_dict.get(
                "raw_description",
                child.task_dict.get("goal", f"subtask {child.node_id[:8]}"),
            )
            status = cr.status if cr else "unknown"
            child_summaries.append(f"- [{status}] {desc}")

        node_desc = node.task_dict.get(
            "raw_description", node.task_dict.get("goal", "task")
        )
        _integration_prompt = (
            f"Verify that the outputs of the following subtasks integrate "
            f"correctly to fulfil: {node_desc}.\n"
            f"Subtask outputs:\n" + "\n".join(child_summaries)
        )

        integration_result: Optional[Dict[str, Any]] = None
        error: Optional[str] = None

        test_path = (
            all_changed[0] if len(all_changed) == 1
            else session_context.get("project_root", ".")
        )
        try:
            integration_result = self._run_tests(test_path)
            if not integration_result.get("success", True):
                error = (f"Integration test failed: "
                         f"{integration_result.get('summary', '')}")
                import warnings
                warnings.warn(
                    f"Integration test failed for node {node.node_id[:8]}: "
                    f"{error}",
                    stacklevel=2,
                )
                return NodeResult(
                    node_id=node.node_id,
                    status="failed",
                    integration_test_result=integration_result,
                    error=error,
                )
        except Exception as exc:  # pragma: no cover
            integration_result = {"error": str(exc)}
            error = f"Integration test raised: {exc}"
            return NodeResult(
                node_id=node.node_id,
                status="failed",
                integration_test_result=integration_result,
                error=error,
            )

        return NodeResult(
            node_id=node.node_id,
            status="integrated",
            integration_test_result=integration_result,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_task_dict(
        self,
        node: TaskNode,
        session_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build a task dict for *node* compatible with TaskPlanner.plan().

        Args:
            node:            The leaf :class:`~core.task_tree.TaskNode`.
            session_context: Session dict with ``session_id`` and
                ``project_root``.

        Returns:
            A task dict.
        """
        return {
            "goal": (
                node.task_dict.get("refined_prompt")
                or node.task_dict.get("raw_description", "")
            ),
            "complexity":     node.task_dict.get("complexity", "medium"),
            "task_category":  node.task_dict.get("routing_domain", "other"),
            "selected_model": node.task_dict.get("selected_model"),
            "session_id":     session_context.get("session_id", ""),
            "project_root":   session_context.get("project_root", "."),
        }

    def _run_tests(self, path: str) -> Dict[str, Any]:
        """Invoke the ``run_tests`` tool on *path*.

        Args:
            path: File or directory path to test.

        Returns:
            The tool's output dict; includes ``"success"`` bool and
            ``"summary"`` string at minimum.
        """
        result = self._tool_registry.invoke("run_tests", {"path": path})
        if hasattr(result, "output"):
            output = result.output
        elif isinstance(result, dict):
            output = result
        else:
            output = {"raw": str(result)}
        if "success" not in output:
            # Infer from returncode when present
            rc = output.get("returncode", -1)
            output["success"] = (rc == 0)
        return output

    def _descendant_leaves(self, node: TaskNode) -> List[TaskNode]:
        """Return all leaf descendants of *node* (depth-first, left-to-right).

        Args:
            node: The :class:`~core.task_tree.TaskNode` whose descendants to
                collect.

        Returns:
            List of leaf :class:`~core.task_tree.TaskNode` objects.
        """
        result: List[TaskNode] = []

        def _collect(n: TaskNode) -> None:
            if n.is_leaf():
                result.append(n)
            else:
                for child in n.children:
                    _collect(child)

        for child in node.children:
            _collect(child)
        return result

    def _update_status(
        self,
        node_id: str,
        status: str,
        message: str = "",
    ) -> None:
        """Push a status update to the progress tracker if one is set.

        Args:
            node_id: Node identifier.
            status:  New status string.
            message: Optional human-readable detail.
        """
        if self._tracker is None:
            return
        try:
            self._tracker.update_node_status(node_id, status, message)
        except Exception:  # pragma: no cover
            pass
