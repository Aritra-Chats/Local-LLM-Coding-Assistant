"""task_tree.py — Hierarchical task decomposition data structures.

Responsibilities
----------------
1. :class:`TaskNode` — a single node in the decomposition tree.  Each node
   wraps a sub-task dict produced by :class:`~core.task_segregator.TaskSegregator`
   and tracks its execution status and test results.

2. :class:`TaskDecompositionTree` — container for the full tree with O(1)
   node lookup, post-order traversal, sibling queries, and completion gating.

Design notes
------------
* The tree is built top-down by ``TaskSegregator.build_tree()`` and consumed
  bottom-up by ``TreeExecutionEngine.execute_tree()``.
* Leaf nodes (``is_leaf() == True``) are the only nodes that receive a
  pipeline execution.  Internal nodes only receive integration tests.
* No third-party dependencies — only stdlib ``dataclasses``, ``uuid``,
  and ``typing``.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# TaskNode
# ---------------------------------------------------------------------------


@dataclass
class TaskNode:
    """A single node in a :class:`TaskDecompositionTree`.

    Attributes:
        node_id:                Unique UUID string identifying this node.
        task_dict:              The sub-task dict from
            :class:`~core.task_segregator.TaskSegregator` containing at least
            ``sub_task_id``, ``raw_description``, ``domain``, ``complexity``,
            ``dependencies``, ``refined_prompt``, ``routing_domain``, and
            ``selected_model`` keys.
        depth:                  Tree depth; 0 = root (original user prompt).
        parent_id:              ``node_id`` of the parent node, or ``None``
            for the root.
        children:               Ordered list of child :class:`TaskNode` objects
            (left-to-right insertion order).
        status:                 Lifecycle state — one of ``"pending"``,
            ``"running"``, ``"unit_tested"``, ``"integrated"``,
            ``"failed"``.
        result:                 Dict representation of the
            :class:`~core.execution_engine.PipelineRunResult` after execution,
            or ``None`` before the node has run.
        unit_test_result:       Output dict of the ``run_tests`` tool for this
            node's leaf execution, or ``None``.
        integration_test_result: Output dict of the integration-test run for
            an internal (non-leaf) node, or ``None``.
    """

    node_id: str
    task_dict: Dict[str, Any]
    depth: int
    parent_id: Optional[str]
    children: List["TaskNode"] = field(default_factory=list)
    status: str = "pending"
    result: Optional[Dict[str, Any]] = None
    unit_test_result: Optional[Dict[str, Any]] = None
    integration_test_result: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def is_leaf(self) -> bool:
        """Return ``True`` when this node has no children.

        A node is a leaf if it has ``complexity == "low"`` OR if the maximum
        decomposition depth was reached.  Both cases are expressed by the
        absence of child nodes at construction time.

        Returns:
            ``True`` when ``self.children`` is empty.
        """
        return len(self.children) == 0

    def complexity(self) -> str:
        """Proxy to ``self.task_dict["complexity"]``.

        Returns:
            One of ``"low"``, ``"medium"``, or ``"high"``.  Falls back to
            ``"medium"`` if the key is absent.
        """
        return self.task_dict.get("complexity", "medium")

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return a fully serialisable dict representation of this node.

        Children are recursively serialised so the entire subtree is
        captured in a single call on the root node.

        Returns:
            A JSON-compatible :class:`dict`.
        """
        return {
            "node_id":                  self.node_id,
            "task_dict":                self.task_dict,
            "depth":                    self.depth,
            "parent_id":                self.parent_id,
            "status":                   self.status,
            "is_leaf":                  self.is_leaf(),
            "complexity":               self.complexity(),
            "result":                   self.result,
            "unit_test_result":         self.unit_test_result,
            "integration_test_result":  self.integration_test_result,
            "children":                 [c.to_dict() for c in self.children],
        }


# ---------------------------------------------------------------------------
# TaskDecompositionTree
# ---------------------------------------------------------------------------


class TaskDecompositionTree:
    """Container for the full hierarchical task decomposition.

    Provides O(1) node lookup via ``_node_index``, post-order traversal,
    sibling queries, and completion gating used by the execution engine.

    Attributes:
        root:         The root :class:`TaskNode` at depth 0.
        _node_index:  Internal ``{node_id: TaskNode}`` mapping kept in sync
            with every :meth:`add_child` call.
    """

    def __init__(self, root: TaskNode) -> None:
        """Initialise the tree with a single root node.

        Args:
            root: The root :class:`TaskNode`.  Its ``node_id`` is added to
                  the internal index immediately.
        """
        self.root: TaskNode = root
        self._node_index: Dict[str, TaskNode] = {root.node_id: root}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_child(self, parent_id: str, child_node: TaskNode) -> None:
        """Attach *child_node* to the node identified by *parent_id*.

        Also registers *child_node* in the internal index for O(1) lookup.

        Args:
            parent_id:  ``node_id`` of the existing parent node.
            child_node: The new :class:`TaskNode` to attach as the
                        rightmost child.

        Raises:
            KeyError: If *parent_id* is not found in the tree.
        """
        parent = self.get_node(parent_id)
        parent.children.append(child_node)
        self._node_index[child_node.node_id] = child_node

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_node(self, node_id: str) -> TaskNode:
        """Look up a node by its ``node_id`` in O(1).

        Args:
            node_id: The UUID string of the target node.

        Returns:
            The matching :class:`TaskNode`.

        Raises:
            KeyError: If *node_id* is not registered in this tree.
        """
        return self._node_index[node_id]

    def leaves(self) -> List[TaskNode]:
        """Return all leaf nodes in left-to-right post-order.

        Leaf nodes are those where :meth:`~TaskNode.is_leaf` returns
        ``True``.  They are collected during a full post-order traversal so
        the return order respects the dependency structure.

        Returns:
            Ordered list of leaf :class:`TaskNode` objects.
        """
        return [n for n in self.post_order() if n.is_leaf()]

    def post_order(self) -> List[TaskNode]:
        """Return all nodes in left-to-right post-order (children before parents).

        This ordering guarantees that when the execution engine processes
        nodes sequentially, every child is executed before its parent.

        Returns:
            Full list of :class:`TaskNode` objects in post-order.
        """
        result: List[TaskNode] = []

        def _visit(node: TaskNode) -> None:
            for child in node.children:
                _visit(child)
            result.append(node)

        _visit(self.root)
        return result

    def siblings_of(self, node_id: str) -> List[TaskNode]:
        """Return all children of the same parent, ordered left-to-right.

        Args:
            node_id: ``node_id`` of the reference node.

        Returns:
            List of sibling :class:`TaskNode` objects (including *node_id*
            itself), or an empty list if the node has no parent (i.e. it is
            the root).
        """
        node = self.get_node(node_id)
        if node.parent_id is None:
            return []
        parent = self.get_node(node.parent_id)
        return list(parent.children)

    def all_children_complete(self, parent_id: str) -> bool:
        """Return ``True`` when every direct child of *parent_id* is done.

        "Done" means the child's :attr:`~TaskNode.status` is in
        ``{"unit_tested", "integrated"}``.

        Args:
            parent_id: ``node_id`` of the parent to check.

        Returns:
            ``True`` when all children are complete; ``False`` otherwise,
            including when the parent has no children.
        """
        _COMPLETE = {"unit_tested", "integrated"}
        parent = self.get_node(parent_id)
        if not parent.children:
            return False
        return all(c.status in _COMPLETE for c in parent.children)

    # ------------------------------------------------------------------
    # Lazy execution ordering (iterative deepening)
    # ------------------------------------------------------------------

    def _iter_bfs(self) -> List[TaskNode]:
        """Return all nodes in BFS order as a fresh snapshot (re-reads tree).

        Because execution_order_generator may have new children added between
        yields, this always re-traverses the tree from the root.

        Returns:
            List of :class:`TaskNode` objects in breadth-first order.
        """
        from collections import deque
        result: List[TaskNode] = []
        q: deque = deque([self.root])
        while q:
            n = q.popleft()
            result.append(n)
            q.extend(n.children)
        return result

    def _iter_dfs(self) -> List[TaskNode]:
        """Return all nodes in pre-order DFS order as a fresh snapshot.

        Pre-order DFS ensures that after a node is lazily decomposed, its
        children appear before its siblings in the scan, giving the expected
        depth-first execution order.
        """
        result: List[TaskNode] = []

        def _visit(n: TaskNode) -> None:
            result.append(n)
            for child in n.children:
                _visit(child)

        _visit(self.root)
        return result

    def execution_order_generator(self):
        """Yield next-ready node one at a time. Caller loops until exhausted.

        Ordering (post-order compatible with lazy decomposition):
          1. Yield a pending LEAF node (deepest available work unit).
          2. If no pending leaves exist, yield an INTERNAL node whose
             ALL children have already completed (not pending/running).
          3. Return when nothing is left.

        Because the execution engine may add new children between yields
        (iterative deepening), this generator re-scans the tree on every
        iteration via _iter_bfs() rather than pre-computing the order.

        Yields:
            :class:`TaskNode` objects in execution order.
        """
        while True:
            # Find next pending leaf (DFS snapshot every scan)
            for node in self._iter_dfs():
                if node.is_leaf() and node.status == "pending":
                    yield node
                    break
            else:
                # No pending leaves — find internal with all children done
                for node in self._iter_dfs():
                    if (
                        not node.is_leaf()
                        and node.status == "pending"
                        and all(
                            c.status not in ("pending", "running")
                            for c in node.children
                        )
                    ):
                        yield node
                        break
                else:
                    return  # nothing left

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return the full tree as a nested, JSON-compatible dict.

        Delegates to :meth:`TaskNode.to_dict` on the root, which recursively
        serialises all descendants.

        Returns:
            A JSON-compatible :class:`dict`.
        """
        return {
            "root": self.root.to_dict(),
            "node_count": len(self._node_index),
        }
