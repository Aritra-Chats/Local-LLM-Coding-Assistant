"""test_task_tree.py — Unit tests for core.task_tree.

Tests cover:
- Leaf detection when a node has no children.
- Post-order traversal returns children before parents.
- all_children_complete gate honours the correct status set.
- Infinite-loop guard in TaskSegregator._decompose_node.
- max_depth forces nodes to be leaves.
- to_dict() returns a JSON-serialisable representation.
"""
from __future__ import annotations

import json
import uuid
import unittest
from unittest.mock import MagicMock, patch

from core.task_tree import TaskDecompositionTree, TaskNode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_node(
    depth: int = 0,
    complexity: str = "medium",
    status: str = "pending",
    parent_id: str | None = None,
) -> TaskNode:
    """Return a minimal TaskNode with sensible defaults."""
    return TaskNode(
        node_id=str(uuid.uuid4()),
        task_dict={
            "sub_task_id":     str(uuid.uuid4())[:8],
            "raw_description": "do something",
            "domain":          "coding_general",
            "complexity":      complexity,
            "dependencies":    [],
        },
        depth=depth,
        parent_id=parent_id,
        children=[],
        status=status,
    )


def _make_tree_with_children() -> tuple[TaskDecompositionTree, TaskNode, TaskNode, TaskNode]:
    """Return a tree:  root → [child_a, child_b].

    Returns:
        (tree, root, child_a, child_b)
    """
    root = _make_node(depth=0)
    child_a = _make_node(depth=1, parent_id=root.node_id)
    child_b = _make_node(depth=1, parent_id=root.node_id)
    tree = TaskDecompositionTree(root=root)
    tree.add_child(root.node_id, child_a)
    tree.add_child(root.node_id, child_b)
    return tree, root, child_a, child_b


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLeafNode(unittest.TestCase):
    """test_leaf_node_no_children — leaf detection."""

    def test_leaf_when_no_children(self):
        node = _make_node()
        self.assertTrue(node.is_leaf())

    def test_not_leaf_when_children_present(self):
        root = _make_node()
        tree = TaskDecompositionTree(root=root)
        child = _make_node(depth=1, parent_id=root.node_id)
        tree.add_child(root.node_id, child)
        self.assertFalse(root.is_leaf())

    def test_leaves_method_returns_only_leaves(self):
        tree, root, child_a, child_b = _make_tree_with_children()
        leaves = tree.leaves()
        self.assertIn(child_a, leaves)
        self.assertIn(child_b, leaves)
        self.assertNotIn(root, leaves)


class TestPostOrder(unittest.TestCase):
    """test_post_order_children_before_parents."""

    def test_children_before_parent_two_levels(self):
        tree, root, child_a, child_b = _make_tree_with_children()
        order = tree.post_order()
        # Both children must appear before root
        self.assertLess(order.index(child_a), order.index(root))
        self.assertLess(order.index(child_b), order.index(root))

    def test_left_to_right_sibling_order(self):
        tree, root, child_a, child_b = _make_tree_with_children()
        order = tree.post_order()
        self.assertLess(order.index(child_a), order.index(child_b))

    def test_three_level_ordering(self):
        root = _make_node(depth=0)
        mid = _make_node(depth=1, parent_id=root.node_id)
        leaf = _make_node(depth=2, parent_id=mid.node_id)
        tree = TaskDecompositionTree(root=root)
        tree.add_child(root.node_id, mid)
        tree.add_child(mid.node_id, leaf)
        order = tree.post_order()
        self.assertLess(order.index(leaf), order.index(mid))
        self.assertLess(order.index(mid), order.index(root))

    def test_single_root_post_order(self):
        root = _make_node()
        tree = TaskDecompositionTree(root=root)
        self.assertEqual(tree.post_order(), [root])


class TestAllChildrenCompleteGate(unittest.TestCase):
    """test_all_children_complete_gate."""

    def test_false_when_children_pending(self):
        tree, root, child_a, child_b = _make_tree_with_children()
        self.assertFalse(tree.all_children_complete(root.node_id))

    def test_true_when_all_unit_tested(self):
        tree, root, child_a, child_b = _make_tree_with_children()
        child_a.status = "unit_tested"
        child_b.status = "unit_tested"
        self.assertTrue(tree.all_children_complete(root.node_id))

    def test_true_when_all_integrated(self):
        tree, root, child_a, child_b = _make_tree_with_children()
        child_a.status = "integrated"
        child_b.status = "integrated"
        self.assertTrue(tree.all_children_complete(root.node_id))

    def test_false_when_mixed_statuses(self):
        tree, root, child_a, child_b = _make_tree_with_children()
        child_a.status = "unit_tested"
        child_b.status = "running"
        self.assertFalse(tree.all_children_complete(root.node_id))

    def test_false_when_one_failed(self):
        tree, root, child_a, child_b = _make_tree_with_children()
        child_a.status = "unit_tested"
        child_b.status = "failed"
        self.assertFalse(tree.all_children_complete(root.node_id))

    def test_false_when_no_children(self):
        root = _make_node()
        tree = TaskDecompositionTree(root=root)
        self.assertFalse(tree.all_children_complete(root.node_id))


class TestInfiniteLoopGuard(unittest.TestCase):
    """test_infinite_loop_guard_in_decompose_node."""

    def test_single_identical_subtask_is_leaf(self):
        """_decompose_node must not recurse when segregate() echoes the parent."""
        from core.task_segregator import TaskSegregator

        prompt = "do something interesting"
        identical_subtask = {
            "sub_task_id":     "st-1",
            "raw_description": prompt,  # same as parent — triggers guard
            "domain":          "other",
            "complexity":      "high",
            "dependencies":    [],
        }

        mock_client = MagicMock()
        segregator = TaskSegregator(
            ollama_client=mock_client,
            supervisor_model="test-model",
        )

        with patch.object(segregator, "segregate", return_value=[identical_subtask]), \
             patch.object(segregator, "refine", side_effect=lambda x: x), \
             patch.object(segregator, "classify", side_effect=lambda x: x):
            tree = segregator.build_tree(prompt, max_depth=4)

        # Root should have no children because the guard fired
        self.assertEqual(len(tree.root.children), 0)
        self.assertTrue(tree.root.is_leaf())

    def test_distinct_subtasks_do_recurse(self):
        """When subtasks differ from the parent, recursion proceeds normally."""
        from core.task_segregator import TaskSegregator

        prompt = "big complex task"
        sub1 = {
            "sub_task_id": "st-1", "raw_description": "part A",
            "domain": "coding_general", "complexity": "low", "dependencies": [],
        }
        sub2 = {
            "sub_task_id": "st-2", "raw_description": "part B",
            "domain": "coding_general", "complexity": "low", "dependencies": [],
        }

        mock_client = MagicMock()
        segregator = TaskSegregator(
            ollama_client=mock_client,
            supervisor_model="test-model",
        )

        # First call (root) returns two distinct subtasks; subsequent calls
        # return a single identical subtask (triggers guard at depth 1).
        call_count = {"n": 0}

        def _seg(p):
            if call_count["n"] == 0:
                call_count["n"] += 1
                return [sub1, sub2]
            return [{"sub_task_id": "echo", "raw_description": p,
                     "domain": "other", "complexity": "high",
                     "dependencies": []}]

        with patch.object(segregator, "segregate", side_effect=_seg), \
             patch.object(segregator, "refine", side_effect=lambda x: x), \
             patch.object(segregator, "classify", side_effect=lambda x: x):
            tree = segregator.build_tree(prompt, max_depth=4)

        self.assertEqual(len(tree.root.children), 2)


class TestMaxDepthCreatesLeaf(unittest.TestCase):
    """test_max_depth_creates_leaf — depth cap prevents infinite decomposition."""

    def test_depth_cap_prevents_recursion(self):
        from core.task_segregator import TaskSegregator

        high_complexity_subtask = {
            "sub_task_id": "st-x", "raw_description": "different sub-task",
            "domain": "coding_general", "complexity": "high", "dependencies": [],
        }

        mock_client = MagicMock()
        segregator = TaskSegregator(
            ollama_client=mock_client,
            supervisor_model="test-model",
        )

        with patch.object(segregator, "segregate",
                          return_value=[high_complexity_subtask]), \
             patch.object(segregator, "refine", side_effect=lambda x: x), \
             patch.object(segregator, "classify", side_effect=lambda x: x):
            tree = segregator.build_tree("some prompt", max_depth=1)

        # Root's children should exist (depth 1 == max_depth) but be leaves
        self.assertGreater(len(tree.root.children), 0)
        for child in tree.root.children:
            self.assertTrue(child.is_leaf(),
                            f"child at depth {child.depth} should be a leaf")

    def test_nodes_at_max_depth_have_no_grandchildren(self):
        from core.task_segregator import TaskSegregator

        sub = {
            "sub_task_id": "st-1", "raw_description": "sub",
            "domain": "coding_general", "complexity": "high", "dependencies": [],
        }
        mock_client = MagicMock()
        segregator = TaskSegregator(mock_client, "test-model")
        with patch.object(segregator, "segregate", return_value=[sub]), \
             patch.object(segregator, "refine", side_effect=lambda x: x), \
             patch.object(segregator, "classify", side_effect=lambda x: x):
            tree = segregator.build_tree("prompt", max_depth=2)

        for node in tree.post_order():
            if node.depth >= 2:
                self.assertEqual(node.children, [],
                                 f"node at depth {node.depth} must be leaf")


class TestTreeToDictSerializable(unittest.TestCase):
    """test_tree_to_dict_is_serialisable."""

    def test_root_only_tree_serialises(self):
        root = _make_node()
        tree = TaskDecompositionTree(root=root)
        d = tree.to_dict()
        # Must not raise
        json_str = json.dumps(d)
        self.assertIsInstance(json_str, str)
        self.assertIn("root", d)

    def test_two_level_tree_serialises(self):
        tree, root, child_a, child_b = _make_tree_with_children()
        d = tree.to_dict()
        json_str = json.dumps(d)
        self.assertIsInstance(json_str, str)
        # Children appear nested inside root
        root_d = d["root"]
        self.assertEqual(len(root_d["children"]), 2)

    def test_to_dict_includes_required_fields(self):
        node = _make_node(status="unit_tested", complexity="low")
        d = node.to_dict()
        for key in ("node_id", "task_dict", "depth", "parent_id",
                    "status", "is_leaf", "complexity", "children"):
            self.assertIn(key, d, f"Missing key: {key}")

    def test_complexity_proxy(self):
        node = _make_node(complexity="high")
        self.assertEqual(node.complexity(), "high")

    def test_siblings_of(self):
        tree, root, child_a, child_b = _make_tree_with_children()
        siblings = tree.siblings_of(child_a.node_id)
        self.assertIn(child_a, siblings)
        self.assertIn(child_b, siblings)
        self.assertEqual(len(siblings), 2)

    def test_siblings_of_root_returns_empty(self):
        root = _make_node()
        tree = TaskDecompositionTree(root=root)
        self.assertEqual(tree.siblings_of(root.node_id), [])


if __name__ == "__main__":
    unittest.main()
