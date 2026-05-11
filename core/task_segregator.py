"""task_segregator.py — Online-mode task decomposition, refinement, and routing.

Used in online mode only, before TaskPlanner.plan().  Intercepts the raw
user prompt and performs three sequential steps:

1. **Task decomposition** — uses the local supervisor model to split the
   prompt into atomic sub-tasks with a domain and complexity estimate.
2. **Prompt refinement** — rewrites each sub-task into a precise,
   model-ready prompt.
3. **Routing classification** — assigns a ``routing_domain`` using keyword
   detection (fast) with NLP zero-shot classification as a fallback.

The local supervisor model is always used here regardless of the current
``SENTINEL_MODE``, keeping the decomposition cost within the local machine.
"""
from __future__ import annotations

import json
import re
import uuid
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Complexity estimation (mirrors _estimate_complexity in agents/supervisor.py)
# ---------------------------------------------------------------------------

_HIGH_COMPLEXITY_KEYWORDS = frozenset(
    {"refactor", "architecture", "migrate", "optimise", "optimize",
     "benchmark", "security audit", "upgrade", "pipeline"}
)
_LOW_COMPLEXITY_KEYWORDS = frozenset(
    {"explain", "summarise", "summarize", "describe", "what is", "show", "list"}
)
_HIGH_LENGTH_THRESHOLD = 300  # chars — long prompts are rarely trivial


def _estimate_complexity(prompt: str) -> str:
    """Estimate task complexity from *prompt* text and length.

    Mirrors the keyword logic in ``agents.supervisor._estimate_complexity``
    and adds a length heuristic so that the TaskSegregator fallback path
    produces accurate complexity values without calling the LLM.

    Args:
        prompt: The raw user prompt or sub-task description.

    Returns:
        ``"low"``, ``"medium"``, or ``"high"``.
    """
    lower = prompt.lower()
    if any(k in lower for k in _HIGH_COMPLEXITY_KEYWORDS):
        return "high"
    if len(prompt) >= _HIGH_LENGTH_THRESHOLD:
        return "high"
    if any(k in lower for k in _LOW_COMPLEXITY_KEYWORDS):
        return "low"
    return "medium"


# ---------------------------------------------------------------------------
# Keyword routing seeds (illustrative; not exhaustive)
# ---------------------------------------------------------------------------

_KEYWORD_DOMAINS: Dict[str, List[str]] = {
    "math":             ["math", "equation", "calculus", "matrix", "statistics", "integral",
                         "derivative", "algebra", "probability", "arithmetic"],
    "coding_frontend":  ["react", "vue", "angular", "css", "tailwind", "html", "svelte",
                         "typescript", "javascript", "jsx", "tsx", "ui", "component",
                         "frontend", "browser"],
    "coding_backend":   ["api", "rest", "graphql", "database", "backend", "server",
                         "django", "flask", "fastapi", "postgresql", "mysql", "redis",
                         "endpoint", "microservice"],
    "coding_general":   ["python", "java", "golang", "rust", "c++", "c#", "function",
                         "algorithm", "data structure", "code", "implement", "write a"],
    "debugging":        ["debug", "traceback", "error", "exception", "fix", "bug",
                         "crash", "fail", "stacktrace", "issue", "broken"],
    "research":         ["research", "survey", "explain", "summarise", "summarize",
                         "literature", "overview", "comparison", "analyse", "analyze"],
    "devops":           ["dockerfile", "ci", "deploy", "kubernetes", "nginx", "terraform",
                         "ansible", "pipeline", "docker", "helm", "yaml", "github actions"],
    "data_science":     ["pandas", "numpy", "sklearn", "scikit", "pytorch", "tensorflow",
                         "dataset", "model training", "accuracy", "f1", "regression",
                         "classification", "clustering"],
    "security":         ["security", "vulnerability", "exploit", "pentest", "cve",
                         "authentication", "authorization", "jwt", "oauth", "csrf", "xss"],
    "creative":         ["write a story", "write a poem", "creative", "fiction", "narrative"],
    "system":           ["system", "os", "operating system", "file system", "process",
                         "memory", "cpu", "disk", "kernel"],
}


# ---------------------------------------------------------------------------
# TaskSegregator
# ---------------------------------------------------------------------------


class TaskSegregator:
    """Decomposes and classifies a user prompt into routable sub-tasks.

    Parameters
    ----------
    ollama_client:
        An :class:`~models.ollama_client.OllamaClient` instance pointing
        at the *local* Ollama daemon.  Always local, regardless of mode.
    supervisor_model:
        Ollama model tag to use for decomposition / refinement / NLP
        classification.
    """

    def __init__(
        self,
        ollama_client: Any,
        supervisor_model: str,
    ) -> None:
        self._client = ollama_client
        self._model = supervisor_model

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def segregate(self, prompt: str) -> List[Dict[str, Any]]:
        """Decompose *prompt* into a list of sub-task dicts.

        Each sub-task has the keys defined in the schema below.  Subsequent
        calls to :meth:`refine` and :meth:`classify` augment the dicts.

        Sub-task schema::

            {
                "sub_task_id":     "<uuid>",
                "raw_description": "<original wording>",
                "domain":          "<inferred domain string>",
                "complexity":      "low|medium|high",
                "dependencies":    ["<sub_task_id>", ...],
            }

        Args:
            prompt: The raw user prompt.

        Returns:
            List of sub-task dicts.  Falls back to a single sub-task wrapping
            the full prompt on any LLM or parse error.
        """
        system = (
            "You are a task decomposition assistant. "
            "Given a user request, decompose it into a list of atomic sub-tasks. "
            "Respond ONLY with a JSON array. Each element must have exactly these keys: "
            '"sub_task_id" (a short unique identifier like "st-1"), '
            '"raw_description" (verbatim sub-task text), '
            '"domain" (e.g. coding_frontend, coding_backend, debugging, math, research, '
            "devops, data_science, security, creative, system, other), "
            '"complexity" ("low", "medium", or "high"), '
            '"dependencies" (array of sub_task_id strings of tasks this one depends on). '
            "Output ONLY valid JSON, no markdown, no commentary."
        )
        try:
            resp = self._client.generate(
                model=self._model,
                prompt=prompt,
                system=system,
                options={"temperature": 0.1, "num_predict": 1024},
                timeout=120,
            )
            raw_text = resp.get("response", "").strip()
            # Strip markdown fences if present
            raw_text = re.sub(r"^```[a-z]*\n?", "", raw_text)
            raw_text = re.sub(r"\n?```$", "", raw_text)
            sub_tasks: List[Dict] = json.loads(raw_text)
            if not isinstance(sub_tasks, list):
                raise ValueError("Not a list")
            # Assign proper UUIDs and normalise
            for st in sub_tasks:
                if not st.get("sub_task_id"):
                    st["sub_task_id"] = str(uuid.uuid4())[:8]
                st.setdefault("dependencies", [])
                st.setdefault("complexity", "medium")
                st.setdefault("domain", "other")
            return sub_tasks
        except Exception:
            # Fallback: single sub-task = entire prompt.
            # Re-estimate complexity from prompt content/length instead of
            # blindly assigning "medium" (BUG-2 fix).
            return [{
                "sub_task_id":     str(uuid.uuid4())[:8],
                "raw_description": prompt,
                "domain":          "other",
                "complexity":      _estimate_complexity(prompt),
                "dependencies":    [],
            }]

    def refine(self, sub_task: Dict[str, Any]) -> Dict[str, Any]:
        """Rewrite ``sub_task["raw_description"]`` into a precise prompt.

        Rules applied by the LLM:
          - Remove filler words ("maybe", "sort of", "something like")
          - Expand implicit references
          - Add output format requirements if missing
          - Preserve all technical constraints verbatim

        Stores result in ``sub_task["refined_prompt"]`` and returns the
        mutated dict.

        Args:
            sub_task: A sub-task dict from :meth:`segregate`.

        Returns:
            The same dict with ``"refined_prompt"`` added.
        """
        system = (
            "You are a prompt-refinement assistant. "
            "Given a raw sub-task description, rewrite it into a precise, "
            "unambiguous, model-ready prompt. "
            "Rules: remove filler ('maybe', 'sort of', 'something like'); "
            "expand implicit references; add output format requirements if absent; "
            "preserve all technical constraints verbatim. "
            "Return ONLY the refined prompt text, no commentary."
        )
        try:
            resp = self._client.generate(
                model=self._model,
                prompt=sub_task.get("raw_description", ""),
                system=system,
                options={"temperature": 0.1, "num_predict": 512},
                timeout=60,
            )
            refined = resp.get("response", "").strip()
            sub_task["refined_prompt"] = refined or sub_task.get("raw_description", "")
        except Exception:
            sub_task["refined_prompt"] = sub_task.get("raw_description", "")
        return sub_task

    def classify(self, sub_task: Dict[str, Any]) -> Dict[str, Any]:
        """Assign a ``routing_domain`` via keyword detection + NLP fallback.

        Signal A — keyword detection (fast path).
        Signal B — NLP zero-shot classification via local supervisor model.

        Stores result in ``sub_task["routing_domain"]`` and returns the
        mutated dict.

        Args:
            sub_task: A sub-task dict (optionally already refined).

        Returns:
            The same dict with ``"routing_domain"`` added.
        """
        text = (
            sub_task.get("refined_prompt")
            or sub_task.get("raw_description", "")
        ).lower()

        # ── Signal A: keyword match ────────────────────────────────────
        best_domain: Optional[str] = None
        best_count = 0
        for domain, keywords in _KEYWORD_DOMAINS.items():
            count = sum(1 for kw in keywords if kw in text)
            if count > best_count:
                best_count = count
                best_domain = domain

        if best_domain and best_count >= 1:
            sub_task["routing_domain"] = best_domain
            return sub_task

        # ── Signal B: NLP zero-shot classification ────────────────────
        domains_list = ", ".join(_KEYWORD_DOMAINS.keys()) + ", other"
        system = (
            "You are a task classifier. "
            "Given a task description, output exactly ONE domain label from this list: "
            f"{domains_list}. "
            "Output ONLY the label, nothing else."
        )
        try:
            resp = self._client.generate(
                model=self._model,
                prompt=text,
                system=system,
                options={"temperature": 0.0, "num_predict": 20},
                timeout=30,
            )
            domain = resp.get("response", "").strip().split()[0].lower()
            sub_task["routing_domain"] = domain if domain in _KEYWORD_DOMAINS else "other"
        except Exception:
            sub_task["routing_domain"] = sub_task.get("domain", "other")

        return sub_task

    # ------------------------------------------------------------------
    # Recursive tree decomposition (Part 2)
    # ------------------------------------------------------------------

    def build_tree(
        self,
        prompt: str,
        max_depth: int = 4,
    ) -> "TaskDecompositionTree":  # noqa: F821  (forward ref; import below)
        """Decompose *prompt* recursively into a :class:`~core.task_tree.TaskDecompositionTree`.

        The root node represents the original user prompt (depth 0).  Each
        non-leaf child is further decomposed until either ``max_depth`` is
        reached or :meth:`segregate` returns a single sub-task whose
        description matches the parent (infinite-loop guard).

        Args:
            prompt:    The raw user prompt.
            max_depth: Maximum recursion depth (inclusive).  Nodes at this
                       depth are always treated as leaves regardless of
                       complexity.

        Returns:
            A fully-constructed :class:`~core.task_tree.TaskDecompositionTree`.
        """
        from core.task_tree import TaskNode, TaskDecompositionTree

        root_task_dict: Dict[str, Any] = {
            "sub_task_id":     str(uuid.uuid4()),
            "raw_description": prompt,
            "domain":          "root",
            "complexity":      "high",
            "dependencies":    [],
        }
        root = TaskNode(
            node_id=str(uuid.uuid4()),
            task_dict=root_task_dict,
            depth=0,
            parent_id=None,
            children=[],
            status="pending",
            result=None,
            unit_test_result=None,
            integration_test_result=None,
        )
        tree = TaskDecompositionTree(root=root)
        self._decompose_node(root, tree, max_depth)
        return tree

    def _decompose_node(
        self,
        node: "TaskNode",  # noqa: F821
        tree: "TaskDecompositionTree",  # noqa: F821
        max_depth: int,
    ) -> None:
        """Recursively decompose *node* and attach children to *tree*.

        Stops recursing when *node.depth* >= *max_depth*, when complexity
        is ``"low"``, or when the LLM returns only one sub-task that
        duplicates the parent description (infinite-loop guard).

        Args:
            node:      The :class:`~core.task_tree.TaskNode` to decompose.
            tree:      The :class:`~core.task_tree.TaskDecompositionTree`
                       being built (used for ``add_child``).
            max_depth: Maximum allowed depth.
        """
        from core.task_tree import TaskNode

        if node.depth >= max_depth:
            return  # depth cap — this node is already a leaf

        parent_desc = node.task_dict.get("raw_description", "").strip()
        sub_tasks = self.segregate(parent_desc)

        # Infinite-loop guard: single sub-task identical to parent
        if len(sub_tasks) == 1:
            child_desc = sub_tasks[0].get("raw_description", "").strip()
            if child_desc == parent_desc:
                return  # treat node as leaf — no children added

        for st in sub_tasks:
            self.refine(st)
            self.classify(st)

            child = TaskNode(
                node_id=str(uuid.uuid4()),
                task_dict=st,
                depth=node.depth + 1,
                parent_id=node.node_id,
                children=[],
                status="pending",
                result=None,
                unit_test_result=None,
                integration_test_result=None,
            )
            tree.add_child(node.node_id, child)

            # Recurse only for non-leaf children within depth budget
            if child.complexity() != "low" and child.depth < max_depth:
                self._decompose_node(child, tree, max_depth)
