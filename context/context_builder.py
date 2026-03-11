"""context_builder.py — Sentinel context assembly.

Contains:
  ContextBuilder  — Abstract base class (ABC) defining the interface.
  ConcreteContextBuilder — Concrete implementation wiring RAGEngine,
                           SymbolGraph, DependencyGraph, ProjectSynopsis,
                           and conversation memory into a single payload.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional


class ContextBuilder(ABC):
    """Abstract base class for the Context Builder.

    Assembles a rich, token-efficient context payload for each pipeline step
    by combining multiple sources: RAG search results, symbol graph data,
    dependency graph data, conversation memory, project synopsis, and
    recently edited files.
    """

    @abstractmethod
    def build(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Assemble the full context payload for a pipeline step.

        Args:
            step: The pipeline step for which context is being built.

        Returns:
            A context dict containing all relevant sources, prioritised by
            relevance and token budget.
        """
        ...

    @abstractmethod
    def fetch_rag(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Retrieve the most relevant code chunks via RAG search.

        Args:
            query: The search query derived from the step's goal.
            top_k: Maximum number of results to retrieve.

        Returns:
            A list of ranked code chunk dicts with content and metadata.
        """
        ...

    @abstractmethod
    def fetch_symbol_graph(self, symbols: List[str]) -> Dict[str, Any]:
        """Retrieve symbol-level relationships for the given identifiers.

        Args:
            symbols: A list of class, method, or variable names to resolve.

        Returns:
            A dict mapping each symbol to its extracted graph data.
        """
        ...

    @abstractmethod
    def fetch_dependency_graph(self, modules: List[str]) -> Dict[str, Any]:
        """Retrieve module-level dependency data for the given modules.

        Args:
            modules: A list of module names to look up.

        Returns:
            A dependency graph dict for the requested modules.
        """
        ...

    @abstractmethod
    def fetch_conversation_memory(self, session_id: str, limit: int) -> List[Dict[str, Any]]:
        """Retrieve recent conversation turns from memory.

        Args:
            session_id: The active session identifier.
            limit: Maximum number of turns to retrieve.

        Returns:
            A list of conversation turn dicts in chronological order.
        """
        ...

    @abstractmethod
    def prioritise(self, sources: Dict[str, Any], token_budget: int) -> Dict[str, Any]:
        """Trim and rank context sources to fit within the token budget.

        Args:
            sources: The raw assembled sources dict.
            token_budget: Maximum token count allowed for the context payload.

        Returns:
            A trimmed context dict that fits within the token budget.
        """
        ...


# ---------------------------------------------------------------------------
# Concrete implementation
# ---------------------------------------------------------------------------


class ConcreteContextBuilder(ContextBuilder):
    """Full context builder that wires all Sentinel context sources.

    Instantiates each subsystem lazily on first use so that the class can be
    constructed even when Ollama is not yet running.

    Args:
        project_root: Absolute path to the project being assisted.
        session_store: Optional mapping of session_id → list of turn dicts,
            used for conversation memory retrieval.  If not provided, memory
            retrieval returns an empty list.
        embedding_model: Ollama model tag used for RAG embeddings.
        token_budget: Default token budget applied when ``build()`` is called
            without an explicit budget. Defaults to 3000 tokens (≈12 000
            characters at 4 chars/token).
    """

    _CHARS_PER_TOKEN = 4  # approximate

    def __init__(
        self,
        project_root: str,
        session_store: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        embedding_model: str = "nomic-embed-text",
        token_budget: int = 3000,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self._session_store = session_store or {}
        self._embedding_model = embedding_model
        self._default_token_budget = token_budget

        # Lazy-initialised subsystems
        self._rag: Optional[Any] = None          # RAGEngine
        self._symbols: Optional[Any] = None      # SymbolGraph
        self._deps: Optional[Any] = None         # DependencyGraph
        self._synopsis: Optional[Any] = None     # ProjectSynopsis

    # ------------------------------------------------------------------
    # ContextBuilder interface
    # ------------------------------------------------------------------

    def build(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Assemble a full, token-budgeted context payload for a step.

        Uses the step's ``goal``, ``task``, and ``involves_symbols`` keys
        to drive retrieval.  Falls back gracefully when any subsystem is
        unavailable.

        Args:
            step: Pipeline step dict.  Recognised keys:
                - ``goal`` (str): Human-readable step objective.
                - ``task`` (str): Short task label used as RAG query fallback.
                - ``involves_symbols`` (List[str]): Symbol names to look up.
                - ``involves_modules`` (List[str]): Module names for dep graph.
                - ``session_id`` (str): Active session for memory retrieval.
                - ``token_budget`` (int): Override token budget (optional).

        Returns:
            Dict with keys: ``rag``, ``symbols``, ``deps``, ``memory``,
            ``synopsis``, ``step``.  Each value may be empty if the source
            was unavailable.
        """
        query = step.get("goal") or step.get("task", "")
        symbol_names: List[str] = step.get("involves_symbols", [])
        module_names: List[str] = step.get("involves_modules", [])
        session_id: str = step.get("session_id", "")
        budget: int = step.get("token_budget", self._default_token_budget)

        raw_sources: Dict[str, Any] = {
            "step": step,
            "rag": self.fetch_rag(query, top_k=5) if query else [],
            "symbols": self.fetch_symbol_graph(symbol_names) if symbol_names else {},
            "deps": self.fetch_dependency_graph(module_names) if module_names else {},
            "memory": self.fetch_conversation_memory(session_id, limit=10) if session_id else [],
            "synopsis": self._get_synopsis(),
        }

        return self.prioritise(raw_sources, budget)

    def fetch_rag(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top-k RAG results for a query string.

        Args:
            query: Search query.
            top_k: Maximum number of results.

        Returns:
            List of dicts, each with ``chunk_id``, ``file_path``,
            ``start_line``, ``end_line``, ``content``, ``language``,
            and ``score``.
        """
        rag = self._get_rag()
        if rag is None:
            return []
        try:
            results = rag.search(query, top_k=top_k)
            return [
                {
                    "chunk_id": r.chunk.chunk_id,
                    "file_path": r.chunk.file_path,
                    "start_line": r.chunk.start_line,
                    "end_line": r.chunk.end_line,
                    "content": r.chunk.content,
                    "language": r.chunk.language,
                    "score": r.score,
                }
                for r in results
            ]
        except Exception:
            return []

    def fetch_symbol_graph(self, symbols: List[str]) -> Dict[str, Any]:
        """Retrieve symbol definitions and relationships.

        Args:
            symbols: Symbol names to look up.

        Returns:
            Dict mapping each symbol name to a list of definition dicts,
            plus ``edges`` with relevant relationship edges.
        """
        sg = self._get_symbol_graph()
        if sg is None:
            return {}

        import dataclasses
        result: Dict[str, Any] = {}
        for name in symbols:
            defs = sg.find_definitions(name)
            result[name] = [dataclasses.asdict(s) for s in defs]

        # Include edges involving these symbols
        related_edges = [
            dataclasses.asdict(e)
            for e in sg.edges
            if e.source in symbols or e.target in symbols
        ]
        result["edges"] = related_edges
        return result

    def fetch_dependency_graph(self, modules: List[str]) -> Dict[str, Any]:
        """Retrieve direct dependencies and reverse dependencies for modules.

        Args:
            modules: Module or file path names to query.

        Returns:
            Dict mapping each module to ``dependencies`` and
            ``reverse_dependencies`` lists.
        """
        dg = self._get_dependency_graph()
        if dg is None:
            return {}

        result: Dict[str, Any] = {}
        for module in modules:
            result[module] = {
                "dependencies": dg.get_dependencies(module),
                "reverse_dependencies": dg.get_reverse_dependencies(module),
            }
        return result

    def fetch_conversation_memory(
        self, session_id: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Retrieve recent conversation turns for a session.

        Args:
            session_id: Session identifier.
            limit: Maximum number of turns to return.

        Returns:
            List of turn dicts in chronological order.
        """
        turns = self._session_store.get(session_id, [])
        return turns[-limit:] if limit else turns

    def prioritise(
        self, sources: Dict[str, Any], token_budget: int
    ) -> Dict[str, Any]:
        """Trim sources to fit within an approximate token budget.

        Allocation strategy (decreasing priority):
            1. step        — always kept (control data)
            2. synopsis    — kept if ≤ 20 % of budget
            3. memory      — most recent turns, up to 30 % of budget
            4. rag         — highest-scoring chunks first, up to 35 % budget
            5. symbols     — trimmed per entry, up to 10 % budget
            6. deps        — remaining budget

        Args:
            sources: Raw assembled sources dict.
            token_budget: Maximum number of approximate tokens.

        Returns:
            Trimmed sources dict.
        """
        char_budget = token_budget * self._CHARS_PER_TOKEN
        used = 0

        out: Dict[str, Any] = {"step": sources.get("step", {})}

        # 1. Synopsis (20%)
        synopsis = sources.get("synopsis", "")
        alloc_synopsis = int(char_budget * 0.20)
        if synopsis and len(synopsis) <= alloc_synopsis:
            out["synopsis"] = synopsis
            used += len(synopsis)
        elif synopsis:
            out["synopsis"] = synopsis[:alloc_synopsis]
            used += alloc_synopsis
        else:
            out["synopsis"] = ""

        # 2. Conversation memory (30%)
        alloc_memory = int(char_budget * 0.30)
        memory_turns = sources.get("memory", [])
        trimmed_memory: List[Dict[str, Any]] = []
        mem_used = 0
        for turn in reversed(memory_turns):
            text = str(turn.get("content", ""))
            if mem_used + len(text) > alloc_memory:
                break
            trimmed_memory.insert(0, turn)
            mem_used += len(text)
        out["memory"] = trimmed_memory
        used += mem_used

        # 3. RAG (35%)
        alloc_rag = int(char_budget * 0.35)
        rag_chunks = sources.get("rag", [])
        trimmed_rag: List[Dict[str, Any]] = []
        rag_used = 0
        for chunk in rag_chunks:
            content = chunk.get("content", "")
            if rag_used + len(content) > alloc_rag:
                remaining = alloc_rag - rag_used
                if remaining > 100:
                    chunk = {**chunk, "content": content[:remaining]}
                    trimmed_rag.append(chunk)
                break
            trimmed_rag.append(chunk)
            rag_used += len(content)
        out["rag"] = trimmed_rag
        used += rag_used

        # 4. Symbols (10%)
        alloc_sym = int(char_budget * 0.10)
        sym_data = sources.get("symbols", {})
        sym_text = str(sym_data)
        out["symbols"] = sym_data if len(sym_text) <= alloc_sym else {}

        # 5. Deps (remainder)
        remaining = max(0, char_budget - used - alloc_sym)
        dep_data = sources.get("deps", {})
        dep_text = str(dep_data)
        out["deps"] = dep_data if len(dep_text) <= remaining else {}

        return out

    # ------------------------------------------------------------------
    # Lazy subsystem accessors
    # ------------------------------------------------------------------

    def _get_rag(self) -> Optional[Any]:
        """Lazily initialise and return the RAGEngine instance."""
        if self._rag is None:
            try:
                from context.rag_search import RAGEngine  # type: ignore
                self._rag = RAGEngine(
                    str(self.project_root),
                    embedding_model=self._embedding_model,
                )
            except Exception:
                return None
        return self._rag

    def _get_symbol_graph(self) -> Optional[Any]:
        """Lazily initialise and return a fully-indexed SymbolGraph."""
        if self._symbols is None:
            try:
                from context.symbol_graph import SymbolGraph  # type: ignore
                sg = SymbolGraph()
                sg.add_directory(str(self.project_root))
                self._symbols = sg
            except Exception:
                return None
        return self._symbols

    def _get_dependency_graph(self) -> Optional[Any]:
        """Lazily initialise and return a fully-indexed DependencyGraph."""
        if self._deps is None:
            try:
                from context.dependency_graph import DependencyGraph  # type: ignore
                dg = DependencyGraph()
                dg.add_directory(
                    str(self.project_root),
                    first_party_prefix=self.project_root.name,
                )
                self._deps = dg
            except Exception:
                return None
        return self._deps

    def _get_synopsis(self) -> str:
        """Lazily initialise ProjectSynopsis and return its text."""
        if self._synopsis is None:
            try:
                from context.project_synopsis import ProjectSynopsis  # type: ignore
                self._synopsis = ProjectSynopsis(str(self.project_root))
            except Exception:
                return ""
        try:
            return self._synopsis.get()
        except Exception:
            return ""
