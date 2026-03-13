"""context — Context assembly subsystem.

Public exports:
    ContextBuilder          Abstract base class for context assembly.
    ConcreteContextBuilder  Concrete implementation (RAG + symbol/dep graph + memory).
    RAGEngine               File-based vector search over repository chunks.
    SymbolGraph             AST-based cross-file symbol and relationship graph.
    DependencyGraph         Module import/dependency directed graph.
    ProjectSynopsis         LLM-generated codebase summary with cache.

Attachment system:
    ContextLoader           Parses @file/@image/@url/@pdf/@snippet tokens.
    LoadResult              Result dataclass with cleaned prompt + attachments.
    load_attachments()      Module-level convenience wrapper.
"""

from context.context_builder import ContextBuilder, ConcreteContextBuilder
from context.rag_search import RAGEngine, Chunk, SearchResult
from context.symbol_graph import SymbolGraph, Symbol, SymbolEdge
from context.dependency_graph import DependencyGraph, ImportEdge
from context.project_synopsis import ProjectSynopsis
from context.context_loader import ContextLoader, LoadResult, load_attachments
from context.snippet_loader import SnippetLoader


from context.context_cache import ContextCache, cache as context_cache
from context.repo_explorer import RepoExplorer, ExplorationReport
__all__ = [
    # Core context assembly
    "ContextBuilder",
    "ConcreteContextBuilder",
    "RAGEngine",
    "Chunk",
    "SearchResult",
    "SymbolGraph",
    "Symbol",
    "SymbolEdge",
    "DependencyGraph",
    "ImportEdge",
    "ProjectSynopsis",
    # Attachment system
    "ContextLoader",
    "LoadResult",
    "load_attachments",
    "SnippetLoader",
]
