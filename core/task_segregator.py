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

import re as _re
import difflib
import os as _os

# ---------------------------------------------------------------------------
# Fuzzy loop-guard helpers
# ---------------------------------------------------------------------------

# Similarity threshold above which two task descriptions are considered
# semantically equivalent (child is just a restatement of the parent).
# Tunable via environment variable for easy adjustment without code changes.
_SIMILARITY_THRESHOLD = float(_os.environ.get("SENTINEL_LOOP_THRESHOLD", "0.85"))


def _normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = _re.sub(r"[^\w\s]", " ", text)   # punctuation → space
    text = _re.sub(r"\s+", " ", text).strip()
    return text


def _token_jaccard(a: str, b: str) -> float:
    """Jaccard similarity over word-token sets.

    Catches paraphrases that reorder or partially reword sentences while
    keeping the same core vocabulary.
    Returns a float in [0.0, 1.0].
    """
    na, nb = _normalise(a), _normalise(b)
    tokens_a = set(na.split())
    tokens_b = set(nb.split())
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union        = len(tokens_a | tokens_b)
    return intersection / union


def _sequence_ratio(a: str, b: str) -> float:
    """difflib.SequenceMatcher character-level similarity ratio.

    Catches near-duplicates with minor insertions, deletions, or local
    rewording. Always available (stdlib).
    Returns a float in [0.0, 1.0].
    """
    na, nb = _normalise(a), _normalise(b)
    return difflib.SequenceMatcher(None, na, nb).ratio()


def _semantic_cosine(a: str, b: str) -> Optional[float]:
    """Cosine similarity between sentence embeddings.

    Uses the _STEmbedder singleton from models/inference_engine.py (backed
    by sentence-transformers all-MiniLM-L6-v2). Returns None when
    sentence-transformers is not installed, so callers can fall back.
    Returns a float in [0.0, 1.0], or None if unavailable.
    """
    try:
        from models.inference_engine import _STEmbedder
        import numpy as _np
        embedder = _STEmbedder.get()   # returns None if ST not installed
        if embedder is None:
            return None
        vec_a = embedder.encode(a)   # already L2-normalised
        vec_b = embedder.encode(b)
        # Both are unit vectors → dot product == cosine similarity
        return float(_np.dot(vec_a, vec_b))
    except Exception:
        return None


def _is_semantically_duplicate(text_a: str, text_b: str) -> bool:
    """Return True when text_a and text_b describe the same task.

    Uses a three-tier approach with graceful degradation:

    Tier 1 — Semantic embeddings (best quality, optional):
      If sentence-transformers is installed, compute cosine similarity of
      sentence embeddings. Fire if cosine >= _SIMILARITY_THRESHOLD.
      This is authoritative when available; tiers 2 and 3 are skipped.

    Tier 2 — Combined token + sequence signal (stdlib only):
      Compute Jaccard over word-token sets AND difflib SequenceMatcher ratio.
      Fire if BOTH signals independently exceed _SIMILARITY_THRESHOLD.
      Requiring BOTH reduces false positives.

    Tier 3 — Exact normalised match (always):
      Normalised strings are identical after lowercasing + stripping punctuation.

    Threshold: _SIMILARITY_THRESHOLD = 0.85 (env-tunable)
    """
    if not text_a.strip() or not text_b.strip():
        return text_a.strip() == text_b.strip()

    # ── Tier 1: semantic cosine (best, optional) ──────────────────────────
    cosine = _semantic_cosine(text_a, text_b)
    if cosine is not None:
        return cosine >= _SIMILARITY_THRESHOLD

    # ── Tier 2: token Jaccard AND sequence ratio (stdlib) ─────────────────
    jaccard  = _token_jaccard(text_a, text_b)
    seqratio = _sequence_ratio(text_a, text_b)
    if jaccard >= _SIMILARITY_THRESHOLD and seqratio >= _SIMILARITY_THRESHOLD:
        return True

    # ── Tier 3: exact normalised match (backstop) ─────────────────────────
    return _normalise(text_a) == _normalise(text_b)

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
_COMPLEX_LENGTH_THRESHOLD = 600  # chars — extremely long prompts likely complex


def _estimate_complexity(prompt: str) -> str:
    """Estimate task complexity from *prompt* text and length.

    Returns one of: "low", "medium", "high", or "complex".
    """
    lower = prompt.lower()
    plen = len(prompt)

    # 1. High-keyword + long prompt -> complex
    if any(k in lower for k in _HIGH_COMPLEXITY_KEYWORDS) and plen >= _HIGH_LENGTH_THRESHOLD:
        return "complex"

    # 2. Extremely long prompts -> complex
    if plen >= _COMPLEX_LENGTH_THRESHOLD:
        return "complex"

    # 3. High-keyword present -> high
    if any(k in lower for k in _HIGH_COMPLEXITY_KEYWORDS):
        return "high"

    # 4. Moderately long prompts -> high
    if plen >= _HIGH_LENGTH_THRESHOLD:
        return "high"

    # 5. Low-keyword present -> low
    if any(k in lower for k in _LOW_COMPLEXITY_KEYWORDS):
        return "low"

    # Default
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
        decomposition_model: str = "",   # NEW: ≤250B model for decompose calls
    ) -> None:
        self._client = ollama_client
        self._model = supervisor_model          # used for refine() and classify()
        self._decomp_model = decomposition_model or supervisor_model  # for segregate()

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
            "You are a task decomposition assistant. Given a task description, break it into DIRECT sub-components only — one level down. Do NOT recursively decompose into atomic leaves. Return 3 to 7 direct child sub-tasks that together fully cover the parent task. Respond ONLY with a JSON array. Each element must have exactly two keys: \"sub_task_id\" (a short unique identifier like \"st-1\") and \"raw_description\" (the verbatim sub-task text). Output ONLY valid JSON, no markdown, no commentary."
        )
        try:
            resp = self._client.generate(
                model=self._decomp_model,   # ≤250B model for structured decomposition
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
            # Only set sub_task_id and raw_description; others added later.
            return [{
                "sub_task_id":     str(uuid.uuid4())[:8],
                "raw_description": prompt,
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

    def _annotate_complexity(self, sub_task: Dict[str, Any]) -> Dict[str, Any]:
        """Assign complexity to a sub-task using the supervisor model.

        Calls the supervisor model to assign a complexity level (low, medium, or high)
        based on the refined prompt. Falls back to _estimate_complexity() on error.

        Args:
            sub_task: A sub-task dict (with refined_prompt already set).

        Returns:
            The same dict with ``"complexity"`` added or updated.
        """
        refined_text = (
            sub_task.get("refined_prompt")
            or sub_task.get("raw_description", "")
        )

        # Build contextual signals
        routing_domain = sub_task.get("routing_domain", "")
        affected_files = sub_task.get("affected_files", []) or []
        dependencies = sub_task.get("dependencies", []) or []
        desc_len = len(refined_text)

        context_parts = [
            f"Description: {refined_text}",
            f"Domain: {routing_domain}",
            f"Affected files: {', '.join(affected_files) if affected_files else 'none'}",
            f"Dependency count: {len(dependencies)}",
            f"Description length: {desc_len}",
            f"Estimated token count (approx): {desc_len // 4}"
        ]
        system = (
            "You are assessing the implementation complexity of a software engineering sub-task at runtime. "
            "Use ALL context provided — task description, domain, files affected, and any other signals — to holistically assess how complex this task will be to implement. "
            "Consider: scope of change, number of systems or layers involved, degree of coordination required, risk of unintended side effects, and whether the task requires designing new architecture or only making contained changes.\n\n"
            "Additionally, consider whether this task can be OPTIMALLY handled end-to-end by a single small language model "
            "with more than 20 billion but fewer than 50 billion parameters, without requiring multi-step reasoning, "
            "tool use, or external retrieval. Tasks that are self-contained, well-scoped, stateless, and do not require "
            "broad architectural knowledge qualify. If such a small model can handle the task optimally, that is a strong "
            "signal to rate the complexity as 'low', even if the description is moderately long.\n\n"
            "Respond with ONLY one word from this exact set: low, medium, high, complex"
        )

        try:
            resp = self._client.generate(
                model=self._model,
                prompt="\n".join(context_parts),
                system=system,
                options={"temperature": 0.0, "num_predict": 20},
                timeout=30,
            )
            complexity = resp.get("response", "").strip().lower().split()[0]
            if complexity not in ("low", "medium", "high", "complex"):
                complexity = _estimate_complexity(refined_text)
            sub_task["complexity"] = complexity
        except Exception:
            sub_task["complexity"] = _estimate_complexity(refined_text)

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

        # Fuzzy infinite-loop guard: single sub-task semantically identical to parent
        if len(sub_tasks) == 1:
            child_desc = sub_tasks[0].get("raw_description", "").strip()
            if _is_semantically_duplicate(parent_desc, child_desc):
                return  # treat node as leaf — no children added

        for st in sub_tasks:
            self.refine(st)
            self.classify(st)
            self._annotate_complexity(st)

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

    # ------------------------------------------------------------------
    # Lazy / iterative-deepening decomposition (Change 3)
    # ------------------------------------------------------------------

    def build_tree_lazy(
        self,
        prompt: str,
        pre_segregated: Optional[List[Dict[str, Any]]] = None,
    ) -> "TaskDecompositionTree":  # noqa: F821
        """Build ONLY the root + one layer of children (depth-1 subtasks).

        This is NOT the full tree. The tree execution engine will call
        decompose_node_with_context() just before executing each medium/high
        complexity node, iteratively deepening the tree at execution time.

        Args:
            prompt:          The raw user prompt.
            pre_segregated:  Optional list of already-computed sub-task dicts
                             (from the flat segregate() + refine() + classify()
                             run in main.py).  When supplied these are used
                             directly as the first layer of children, avoiding
                             a redundant second call to segregate() with the
                             decomposition model.  When None, _decompose_node_lazy
                             calls segregate() internally as before.

        Returns:
            A :class:`~core.task_tree.TaskDecompositionTree` with root + depth-1 children.
        """
        from core.task_tree import TaskNode, TaskDecompositionTree

        root_dict: Dict[str, Any] = {
            "sub_task_id":     str(uuid.uuid4()),
            "raw_description": prompt,
            "domain":          "root",
            "complexity":      "high",
            "dependencies":    [],
        }
        root = TaskNode(
            node_id=str(uuid.uuid4()),
            task_dict=root_dict,
            depth=0,
            parent_id=None,
        )
        tree = TaskDecompositionTree(root=root)

        if pre_segregated:
            # Fast path: plant the already-refined first layer directly.
            # refine() and classify() were already called by the caller.
            for st in pre_segregated:
                child = TaskNode(
                    node_id=str(uuid.uuid4()),
                    task_dict=st,
                    depth=1,
                    parent_id=root.node_id,
                )
                tree.add_child(root.node_id, child)
        else:
            # Slow path: call segregate() → _decompose_node_lazy.
            self._decompose_node_lazy(root, tree)

        return tree

    def _decompose_node_lazy(
        self,
        node: "TaskNode",   # noqa: F821
        tree: "TaskDecompositionTree",  # noqa: F821
    ) -> bool:
        """Decompose node ONE layer deeper. Returns True if children were added.

        Termination conditions (node treated as leaf, returns False):
          i.   node.complexity() == "low"
          ii.  segregate() returns a single subtask whose raw_description
               is semantically duplicate of node's raw_description (fuzzy loop guard)
          iii. segregate() raises an exception or returns empty

        Does NOT recurse. The execution engine calls this iteratively.

        Args:
            node: The :class:`~core.task_tree.TaskNode` to decompose.
            tree: The :class:`~core.task_tree.TaskDecompositionTree` being built.

        Returns:
            True if children were added to the tree; False otherwise.
        """
        from core.task_tree import TaskNode

        if node.complexity() == "low":
            return False

        parent_desc = node.task_dict.get("raw_description", "").strip()
        try:
            sub_tasks = self.segregate(parent_desc)
        except Exception:
            return False

        if not sub_tasks:
            return False

        # Fuzzy infinite-loop guard
        if len(sub_tasks) == 1:
            child_desc = sub_tasks[0].get("raw_description", "").strip()
            if _is_semantically_duplicate(parent_desc, child_desc):
                return False   # treat node as a leaf — no further decomposition

        for st in sub_tasks:
            self.refine(st)
            self.classify(st)
            self._annotate_complexity(st)

            child = TaskNode(
                node_id=str(uuid.uuid4()),
                task_dict=st,
                depth=node.depth + 1,
                parent_id=node.node_id,
            )
            tree.add_child(node.node_id, child)

        return True

    def decompose_node_with_context(
        self,
        node: "TaskNode",   # noqa: F821
        tree: "TaskDecompositionTree",  # noqa: F821
        sibling_context: Dict[str, Any],
    ) -> bool:
        """Like _decompose_node_lazy but enriches the prompt with sibling context.

        Called by the execution engine after siblings have completed, so the
        decomposition of later nodes is informed by what earlier ones produced.

        sibling_context schema::

            {
                "completed_siblings": [
                    {
                        "description": str,
                        "status": str,
                        "changed_files": [str, ...],
                        "result_summary": str,   # human-readable 1-2 sentence summary
                    },
                    ...
                ],
                "project_root": str,
            }

        If completed_siblings is empty, falls back to _decompose_node_lazy().

        Args:
            node:            The node to decompose.
            tree:            The tree being built.
            sibling_context: Context from already-completed siblings.

        Returns:
            True if children were added; False otherwise.
        """
        if not sibling_context.get("completed_siblings"):
            return self._decompose_node_lazy(node, tree)

        base_desc = (
            node.task_dict.get("refined_prompt")
            or node.task_dict.get("raw_description", "")
        )
        sibling_lines = []
        for s in sibling_context["completed_siblings"]:
            line = f"  - [{s['status']}] {s['description']}: {s['result_summary']}"
            if s.get("changed_files"):
                line += f" (changed: {', '.join(s['changed_files'][:3])})"
            sibling_lines.append(line)

        enriched_prompt = (
            f"{base_desc}\n\n"
            f"Context from already-completed sibling tasks at the same level:\n"
            + "\n".join(sibling_lines)
            + f"\n\nProject root: {sibling_context.get('project_root', '.')}"
        )

        # Temporarily override the description so segregate() uses the enriched prompt
        original_desc = node.task_dict.get("raw_description", "")
        node.task_dict["raw_description"] = enriched_prompt
        result = self._decompose_node_lazy(node, tree)
        node.task_dict["raw_description"] = original_desc  # restore
        return result
