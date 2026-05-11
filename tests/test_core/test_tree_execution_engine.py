"""test_tree_execution_engine.py — Unit tests for core.tree_execution_engine.

All heavy dependencies (ConcreteExecutionEngine, run_tests tool) are stubbed
with unittest.mock so tests run without a local Ollama daemon or filesystem.

Tests cover:
- Leaf nodes run pipeline then unit test.
- Internal nodes run integration test after all children complete.
- A failed unit test marks the node "failed" without aborting the whole tree.
- Overall TreeRunResult.status == "completed" when root is "integrated".
"""
from __future__ import annotations

import uuid
import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch, PropertyMock

from core.task_tree import TaskDecompositionTree, TaskNode
from core.tree_execution_engine import (
    ConcreteTreeExecutionEngine,
    NodeResult,
    TreeRunResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _node(
    depth: int = 0,
    complexity: str = "medium",
    parent_id: Optional[str] = None,
    description: str = "do something",
) -> TaskNode:
    return TaskNode(
        node_id=str(uuid.uuid4()),
        task_dict={
            "sub_task_id":     str(uuid.uuid4())[:8],
            "raw_description": description,
            "domain":          "coding_general",
            "complexity":      complexity,
            "dependencies":    [],
            "refined_prompt":  description,
            "routing_domain":  "coding_general",
        },
        depth=depth,
        parent_id=parent_id,
        children=[],
        status="pending",
    )


def _leaf_tree() -> tuple[TaskDecompositionTree, TaskNode]:
    """Single-node (root-is-leaf) tree."""
    root = _node(depth=0, complexity="low")
    tree = TaskDecompositionTree(root=root)
    return tree, root


def _parent_child_tree() -> tuple[TaskDecompositionTree, TaskNode, TaskNode, TaskNode]:
    """Three-node tree: root → [child_a (leaf), child_b (leaf)]."""
    root = _node(depth=0, complexity="high")
    child_a = _node(depth=1, complexity="low", parent_id=root.node_id)
    child_b = _node(depth=1, complexity="low", parent_id=root.node_id)
    tree = TaskDecompositionTree(root=root)
    tree.add_child(root.node_id, child_a)
    tree.add_child(root.node_id, child_b)
    return tree, root, child_a, child_b


def _make_engine(
    run_tests_success: bool = True,
    pipeline_run_raises: Optional[Exception] = None,
) -> ConcreteTreeExecutionEngine:
    """Build a ConcreteTreeExecutionEngine with all deps stubbed."""

    # Stub PipelineRunResult
    mock_run_result = MagicMock()
    mock_run_result.to_dict.return_value = {"status": "completed", "step_results": []}
    mock_run_result.status = "completed"

    # Stub FileChangeMap
    mock_fcm = MagicMock()
    mock_fcm.changed_paths.return_value = ["/fake/file.py"]

    # Stub ConcreteExecutionEngine
    mock_concrete = MagicMock()
    if pipeline_run_raises:
        mock_concrete.run_pipeline.side_effect = pipeline_run_raises
    else:
        mock_concrete.run_pipeline.return_value = mock_run_result
    mock_concrete.file_change_map = mock_fcm

    # Stub TaskPlanner
    mock_planner = MagicMock()
    mock_plan = MagicMock()
    mock_planner.plan.return_value = mock_plan

    # Stub DynamicPipelineGenerator
    mock_pipeline = MagicMock()
    mock_pipeline.steps = []
    mock_gen = MagicMock()
    mock_gen.from_execution_plan.return_value = mock_pipeline

    # Stub ToolRegistry with run_tests
    test_output: Dict[str, Any] = {
        "returncode": 0 if run_tests_success else 1,
        "success":    run_tests_success,
        "summary":    "passed" if run_tests_success else "FAILED",
        "stdout":     "",
        "stderr":     "",
    }
    mock_tool_result = MagicMock()
    mock_tool_result.output = test_output
    mock_registry = MagicMock()
    mock_registry.invoke.return_value = mock_tool_result

    return ConcreteTreeExecutionEngine(
        concrete_engine=mock_concrete,
        task_planner=mock_planner,
        pipeline_generator=mock_gen,
        tool_registry=mock_registry,
        abort_on_failure=False,
    )


_SESSION_CTX: Dict[str, Any] = {
    "session_id":   "test-session",
    "project_root": "/tmp/test-project",
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLeafRunsPipelineThenUnitTest(unittest.TestCase):
    """test_leaf_runs_pipeline_then_unit_test."""

    def test_pipeline_called_for_leaf(self):
        tree, root = _leaf_tree()
        engine = _make_engine()
        engine.execute_tree(tree, _SESSION_CTX)
        engine._engine.run_pipeline.assert_called_once()

    def test_unit_test_invoked_for_leaf(self):
        tree, root = _leaf_tree()
        engine = _make_engine()
        engine.execute_tree(tree, _SESSION_CTX)
        # run_tests must have been invoked (unit test phase)
        engine._tool_registry.invoke.assert_called()
        call_args_list = engine._tool_registry.invoke.call_args_list
        tool_names = [c.args[0] if c.args else c.kwargs.get("tool") for c in call_args_list]
        self.assertIn("run_tests", tool_names)

    def test_leaf_node_status_unit_tested_on_success(self):
        tree, root = _leaf_tree()
        engine = _make_engine(run_tests_success=True)
        result = engine.execute_tree(tree, _SESSION_CTX)
        self.assertEqual(result.node_results[root.node_id].status, "unit_tested")

    def test_planner_and_pipeline_gen_called(self):
        tree, root = _leaf_tree()
        engine = _make_engine()
        engine.execute_tree(tree, _SESSION_CTX)
        engine._task_planner.plan.assert_called_once()
        engine._pipeline_gen.from_execution_plan.assert_called_once()


class TestInternalNodeIntegrationAfterChildren(unittest.TestCase):
    """test_internal_node_runs_integration_after_children."""

    def test_integration_called_after_leaves(self):
        tree, root, child_a, child_b = _parent_child_tree()
        engine = _make_engine(run_tests_success=True)
        result = engine.execute_tree(tree, _SESSION_CTX)

        # run_tests should have been called once per leaf (unit) + once for root (integration)
        call_count = engine._tool_registry.invoke.call_count
        # 2 unit tests + 1 integration = 3 minimum
        self.assertGreaterEqual(call_count, 3)

    def test_internal_node_status_integrated_when_tests_pass(self):
        tree, root, child_a, child_b = _parent_child_tree()
        engine = _make_engine(run_tests_success=True)
        result = engine.execute_tree(tree, _SESSION_CTX)
        root_nr = result.node_results[root.node_id]
        self.assertEqual(root_nr.status, "integrated")

    def test_integration_result_stored_on_internal_node(self):
        tree, root, child_a, child_b = _parent_child_tree()
        engine = _make_engine(run_tests_success=True)
        result = engine.execute_tree(tree, _SESSION_CTX)
        root_nr = result.node_results[root.node_id]
        self.assertIsNotNone(root_nr.integration_test_result)

    def test_pipeline_not_called_for_internal_node(self):
        """Internal (non-leaf) nodes must NOT receive a pipeline run."""
        tree, root, child_a, child_b = _parent_child_tree()
        engine = _make_engine(run_tests_success=True)
        engine.execute_tree(tree, _SESSION_CTX)
        # pipeline was run exactly once per leaf (2 leaves → 2 calls)
        self.assertEqual(engine._engine.run_pipeline.call_count, 2)


class TestFailedUnitTestMarksNodeFailedNotAborts(unittest.TestCase):
    """test_failed_unit_test_marks_node_failed_not_aborts."""

    def test_failed_leaf_does_not_abort_sibling(self):
        tree, root, child_a, child_b = _parent_child_tree()

        # Make unit tests fail
        engine = _make_engine(run_tests_success=False)
        result = engine.execute_tree(tree, _SESSION_CTX)

        # Both leaf nodes should appear in node_results (neither was skipped)
        self.assertIn(child_a.node_id, result.node_results)
        self.assertIn(child_b.node_id, result.node_results)

    def test_failed_leaf_status_is_failed(self):
        tree, root = _leaf_tree()
        engine = _make_engine(run_tests_success=False)
        result = engine.execute_tree(tree, _SESSION_CTX)
        self.assertEqual(result.node_results[root.node_id].status, "failed")

    def test_failed_leaf_error_field_populated(self):
        tree, root = _leaf_tree()
        engine = _make_engine(run_tests_success=False)
        result = engine.execute_tree(tree, _SESSION_CTX)
        nr = result.node_results[root.node_id]
        self.assertIsNotNone(nr.error)
        self.assertIn("failed", nr.error.lower())

    def test_abort_on_failure_flag_stops_at_first_failure(self):
        tree, root, child_a, child_b = _parent_child_tree()
        engine = _make_engine(run_tests_success=False)
        engine._abort_on_failure = True
        result = engine.execute_tree(tree, _SESSION_CTX)
        # With abort_on_failure=True, execution stops after first failed leaf.
        # root should not appear as integrated.
        root_nr = result.node_results.get(root.node_id)
        if root_nr:
            self.assertNotEqual(root_nr.status, "integrated")


class TestTreeResultStatusWhenRootIntegrated(unittest.TestCase):
    """test_tree_result_status_completed_when_root_integrated."""

    def test_completed_when_root_integrated(self):
        tree, root, child_a, child_b = _parent_child_tree()
        engine = _make_engine(run_tests_success=True)
        result = engine.execute_tree(tree, _SESSION_CTX)
        self.assertEqual(result.status, "completed")

    def test_partial_when_some_nodes_succeed_root_fails(self):
        """If the root integration test fails but leaves succeeded → partial."""
        tree, root, child_a, child_b = _parent_child_tree()
        engine = _make_engine(run_tests_success=True)

        original_run_tests = engine._run_tests
        call_count = {"n": 0}

        def _selective_run_tests(path: str) -> Dict[str, Any]:
            call_count["n"] += 1
            # Leaf unit tests (calls 1 and 2) pass; root integration (call 3) fails
            if call_count["n"] <= 2:
                return {"success": True, "summary": "passed", "returncode": 0}
            return {"success": False, "summary": "FAILED", "returncode": 1}

        engine._run_tests = _selective_run_tests
        result = engine.execute_tree(tree, _SESSION_CTX)
        self.assertEqual(result.status, "partial")

    def test_failed_when_root_never_ran(self):
        """If abort_on_failure halts before root → overall status is failed or partial."""
        tree, root, child_a, child_b = _parent_child_tree()
        engine = _make_engine(run_tests_success=False)
        engine._abort_on_failure = True
        result = engine.execute_tree(tree, _SESSION_CTX)
        self.assertIn(result.status, ("failed", "partial"))

    def test_tree_run_result_to_dict_serialisable(self):
        import json
        tree, root, child_a, child_b = _parent_child_tree()
        engine = _make_engine(run_tests_success=True)
        result = engine.execute_tree(tree, _SESSION_CTX)
        d = result.to_dict()
        json_str = json.dumps(d)
        self.assertIsInstance(json_str, str)
        self.assertIn("tree_id", d)
        self.assertIn("status", d)
        self.assertIn("node_results", d)

    def test_node_result_to_dict_has_required_fields(self):
        nr = NodeResult(
            node_id="abc",
            status="unit_tested",
            pipeline_result={"status": "completed"},
            unit_test_result={"success": True},
        )
        d = nr.to_dict()
        for key in ("node_id", "status", "pipeline_result",
                    "unit_test_result", "integration_test_result", "error"):
            self.assertIn(key, d, f"Missing key: {key}")

    def test_summary_string_format(self):
        tree, root, child_a, child_b = _parent_child_tree()
        engine = _make_engine(run_tests_success=True)
        result = engine.execute_tree(tree, _SESSION_CTX)
        summary = result.summary()
        self.assertIn(result.tree_id[:8], summary)
        self.assertIn(result.status, summary)


if __name__ == "__main__":
    unittest.main()
