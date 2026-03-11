"""symbol_graph.py — Sentinel AST-based symbol extraction and relationship graph.

Walks one or more Python source files using the built-in 'ast' module,
extracts class, function, method, and import declarations, and builds a
cross-file symbol graph that records:

  - Classes and their base classes (inheritance edges)
  - Functions and their containing class (membership edges)
  - Import relationships between files (import edges)
  - Global constants / variable assignments

No external dependencies required — only Python stdlib.
"""

import ast
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class Symbol:
    """A named symbol extracted from a Python source file.

    Attributes:
        name: Unqualified symbol name (e.g. ``"MyClass"``).
        qualified_name: Dot-qualified name including containing scope.
        symbol_type: One of ``"class"``, ``"function"``, ``"method"``,
            ``"variable"``, ``"import"``.
        file_path: Repository-relative path of the source file.
        start_line: 1-based line number of the symbol definition.
        end_line: 1-based last line of the symbol definition.
        parent: Qualified name of the enclosing scope (if any).
        bases: For classes, list of base class names.
        docstring: First docstring of the symbol, if present.
        signature: Parameter signature string for callable symbols.
    """

    name: str
    qualified_name: str
    symbol_type: str        # 'class' | 'function' | 'method' | 'variable' | 'import'
    file_path: str
    start_line: int
    end_line: int
    parent: Optional[str] = None
    bases: List[str] = field(default_factory=list)
    docstring: Optional[str] = None
    signature: Optional[str] = None


@dataclass
class SymbolEdge:
    """A directed relationship between two symbols.

    Attributes:
        source: Qualified name of the source symbol.
        target: Qualified name (or module path) of the target.
        edge_type: One of ``"inherits"``, ``"contains"``, ``"imports"``,
            ``"calls"`` (reserved for future use).
        file_path: File where this relationship is defined.
    """

    source: str
    target: str
    edge_type: str      # 'inherits' | 'contains' | 'imports'
    file_path: str


# ---------------------------------------------------------------------------
# Symbol Graph
# ---------------------------------------------------------------------------


class SymbolGraph:
    """Multi-file Python symbol and relationship graph.

    Usage::

        sg = SymbolGraph()
        sg.add_file("sentinel/agents/base_agent.py", source_text)
        sg.add_file("sentinel/agents/supervisor.py", supervisor_source)

        results = sg.find_definitions("BaseAgent")
        children = sg.get_subclasses("BaseAgent")

    Attributes:
        symbols: List of all extracted Symbol objects.
        edges: List of all SymbolEdge relationship objects.
    """

    def __init__(self) -> None:
        self.symbols: List[Symbol] = []
        self.edges: List[SymbolEdge] = []
        self._symbol_index: Dict[str, List[Symbol]] = {}     # name → symbols
        self._file_index: Dict[str, List[Symbol]] = {}       # file → symbols

    # ------------------------------------------------------------------
    # Building the graph
    # ------------------------------------------------------------------

    def add_file(self, file_path: str, source: str) -> int:
        """Parse a Python source file and add all discovered symbols/edges.

        Args:
            file_path: Repository-relative path (used as identifier).
            source: Full source text of the file.

        Returns:
            Number of new symbols added.
        """
        try:
            tree = ast.parse(source, filename=file_path)
        except SyntaxError:
            return 0

        extractor = _FileExtractor(file_path)
        new_symbols, new_edges = extractor.extract(tree)

        for sym in new_symbols:
            self.symbols.append(sym)
            self._symbol_index.setdefault(sym.name, []).append(sym)
            self._file_index.setdefault(sym.file_path, []).append(sym)

        self.edges.extend(new_edges)
        return len(new_symbols)

    def add_directory(self, root: str, base: str = "") -> int:
        """Recursively index all Python files under a directory.

        Args:
            root: Absolute path to the directory to index.
            base: Optional prefix to strip from file paths (defaults to
                  the absolute root, producing relative paths).

        Returns:
            Total number of symbols added.
        """
        root_path = Path(root).resolve()
        base_path = Path(base).resolve() if base else root_path
        total = 0
        skip_dirs = {"__pycache__", ".venv", "venv", ".git", "node_modules"}

        for py_file in root_path.rglob("*.py"):
            if any(part in skip_dirs for part in py_file.parts):
                continue
            try:
                source = py_file.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            try:
                rel = str(py_file.relative_to(base_path))
            except ValueError:
                rel = str(py_file)

            total += self.add_file(rel, source)

        return total

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def find_definitions(self, name: str) -> List[Symbol]:
        """Return all symbols with the given name.

        Args:
            name: Unqualified or qualified symbol name.

        Returns:
            List of matching Symbol objects.
        """
        # Try exact match first
        results = self._symbol_index.get(name, [])
        if results:
            return results
        # Fall back to qualified-name suffix match
        return [s for s in self.symbols if s.qualified_name.endswith("." + name) or s.qualified_name == name]

    def get_symbols_in_file(self, file_path: str) -> List[Symbol]:
        """Return all symbols defined in a given file.

        Args:
            file_path: Repository-relative file path.

        Returns:
            List of Symbol objects for that file.
        """
        return self._file_index.get(file_path, [])

    def get_subclasses(self, class_name: str) -> List[Symbol]:
        """Find all classes that directly inherit from the named class.

        Args:
            class_name: Name of the base class.

        Returns:
            List of class Symbol objects that extend the given class.
        """
        return [
            s for s in self.symbols
            if s.symbol_type == "class" and class_name in s.bases
        ]

    def get_methods(self, class_name: str) -> List[Symbol]:
        """Return all methods defined inside a class.

        Args:
            class_name: Unqualified class name.

        Returns:
            List of method Symbol objects.
        """
        return [
            s for s in self.symbols
            if s.symbol_type == "method" and s.parent is not None and s.parent.endswith(class_name)
        ]

    def get_imports(self, file_path: str) -> List[Symbol]:
        """Return all import symbols declared in a file.

        Args:
            file_path: Repository-relative file path.

        Returns:
            List of import Symbol objects.
        """
        return [
            s for s in self.get_symbols_in_file(file_path)
            if s.symbol_type == "import"
        ]

    def summary(self) -> Dict[str, Any]:
        """Return aggregate statistics about the graph.

        Returns:
            Dict with symbol counts by type, file count, edge counts.
        """
        from collections import Counter
        type_counts = Counter(s.symbol_type for s in self.symbols)
        return {
            "total_symbols": len(self.symbols),
            "by_type": dict(type_counts),
            "total_edges": len(self.edges),
            "files_indexed": len(self._file_index),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the full graph to a JSON-compatible dictionary.

        Returns:
            Dict with 'symbols' and 'edges' lists.
        """
        import dataclasses
        return {
            "symbols": [dataclasses.asdict(s) for s in self.symbols],
            "edges": [dataclasses.asdict(e) for e in self.edges],
        }


# ---------------------------------------------------------------------------
# Internal AST extractor
# ---------------------------------------------------------------------------


class _FileExtractor(ast.NodeVisitor):
    """AST visitor that extracts symbols and edges from a single file.

    Not part of the public API. Used internally by SymbolGraph.
    """

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.symbols: List[Symbol] = []
        self.edges: List[SymbolEdge] = []
        self._scope_stack: List[str] = []   # qualified name stack

    def extract(
        self, tree: ast.AST
    ) -> Tuple[List[Symbol], List[SymbolEdge]]:
        """Walk the AST and collect all symbols and edges.

        Args:
            tree: A parsed ast.Module node.

        Returns:
            Tuple of (symbols list, edges list).
        """
        self.visit(tree)
        return self.symbols, self.edges

    # -- Visitors --

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        parent = self._current_scope()
        qualified = self._qualify(node.name)
        bases = [self._name_from_node(b) for b in node.bases]
        docstring = ast.get_docstring(node)

        sym = Symbol(
            name=node.name,
            qualified_name=qualified,
            symbol_type="class",
            file_path=self.file_path,
            start_line=node.lineno,
            end_line=getattr(node, "end_lineno", node.lineno),
            parent=parent,
            bases=bases,
            docstring=docstring,
        )
        self.symbols.append(sym)

        for base in bases:
            self.edges.append(
                SymbolEdge(
                    source=qualified,
                    target=base,
                    edge_type="inherits",
                    file_path=self.file_path,
                )
            )

        if parent:
            self.edges.append(
                SymbolEdge(
                    source=parent,
                    target=qualified,
                    edge_type="contains",
                    file_path=self.file_path,
                )
            )

        self._scope_stack.append(node.name)
        self.generic_visit(node)
        self._scope_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        parent = self._current_scope()
        sym_type = "method" if parent else "function"
        qualified = self._qualify(node.name)
        docstring = ast.get_docstring(node)
        sig = self._build_signature(node)

        sym = Symbol(
            name=node.name,
            qualified_name=qualified,
            symbol_type=sym_type,
            file_path=self.file_path,
            start_line=node.lineno,
            end_line=getattr(node, "end_lineno", node.lineno),
            parent=parent,
            docstring=docstring,
            signature=sig,
        )
        self.symbols.append(sym)

        if parent:
            self.edges.append(
                SymbolEdge(
                    source=parent,
                    target=qualified,
                    edge_type="contains",
                    file_path=self.file_path,
                )
            )

        self._scope_stack.append(node.name)
        self.generic_visit(node)
        self._scope_stack.pop()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            name = alias.asname or alias.name
            sym = Symbol(
                name=name,
                qualified_name=self._qualify(name),
                symbol_type="import",
                file_path=self.file_path,
                start_line=node.lineno,
                end_line=node.lineno,
                parent=self._current_scope(),
            )
            self.symbols.append(sym)
            self.edges.append(
                SymbolEdge(
                    source=self.file_path,
                    target=alias.name,
                    edge_type="imports",
                    file_path=self.file_path,
                )
            )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        for alias in node.names:
            name = alias.asname or alias.name
            sym = Symbol(
                name=name,
                qualified_name=self._qualify(name),
                symbol_type="import",
                file_path=self.file_path,
                start_line=node.lineno,
                end_line=node.lineno,
                parent=self._current_scope(),
            )
            self.symbols.append(sym)
        self.edges.append(
            SymbolEdge(
                source=self.file_path,
                target=module,
                edge_type="imports",
                file_path=self.file_path,
            )
        )

    def visit_Assign(self, node: ast.Assign) -> None:
        """Capture module-level or class-level constant assignments."""
        if self._scope_stack:
            return  # only top-level / class-body variables
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id.isupper():
                sym = Symbol(
                    name=target.id,
                    qualified_name=self._qualify(target.id),
                    symbol_type="variable",
                    file_path=self.file_path,
                    start_line=node.lineno,
                    end_line=getattr(node, "end_lineno", node.lineno),
                )
                self.symbols.append(sym)

    # -- Helpers --

    def _current_scope(self) -> Optional[str]:
        return ".".join(self._scope_stack) if self._scope_stack else None

    def _qualify(self, name: str) -> str:
        if self._scope_stack:
            return ".".join(self._scope_stack) + "." + name
        return name

    @staticmethod
    def _name_from_node(node: ast.expr) -> str:
        """Best-effort name extraction from a base class expression."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parts: List[str] = []
            n: ast.expr = node
            while isinstance(n, ast.Attribute):
                parts.append(n.attr)
                n = n.value
            if isinstance(n, ast.Name):
                parts.append(n.id)
            return ".".join(reversed(parts))
        return ast.unparse(node) if hasattr(ast, "unparse") else "<expr>"

    @staticmethod
    def _build_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        """Produce a parameter signature string for a function/method.

        Args:
            node: AST FunctionDef or AsyncFunctionDef node.

        Returns:
            Signature string like ``"(self, x: int, y: str = 'hi')"``
        """
        args = node.args
        parts: List[str] = []

        # positional-only
        for i, arg in enumerate(args.posonlyargs):
            parts.append(_format_arg(arg, args, i))

        # regular args
        n_pos = len(args.posonlyargs)
        n_defaults = len(args.defaults)
        n_args = len(args.args)
        default_start = n_args - n_defaults

        for i, arg in enumerate(args.args):
            default_idx = n_pos + i - default_start
            default: Optional[ast.expr] = None
            if n_pos + i >= default_start:
                default = args.defaults[n_pos + i - default_start]
            parts.append(_format_arg(arg, args, i, default))

        if args.vararg:
            parts.append("*" + args.vararg.arg)
        elif args.kwonlyargs:
            parts.append("*")

        for i, arg in enumerate(args.kwonlyargs):
            default = args.kw_defaults[i]
            parts.append(_format_arg(arg, args, 0, default))

        if args.kwarg:
            parts.append("**" + args.kwarg.arg)

        return "(" + ", ".join(parts) + ")"


def _format_arg(
    arg: ast.arg,
    _args: ast.arguments,
    _index: int,
    default: Optional[ast.expr] = None,
) -> str:
    """Format a single argument node to a string."""
    s = arg.arg
    if arg.annotation and hasattr(ast, "unparse"):
        s += f": {ast.unparse(arg.annotation)}"
    if default and hasattr(ast, "unparse"):
        s += f" = {ast.unparse(default)}"
    return s
