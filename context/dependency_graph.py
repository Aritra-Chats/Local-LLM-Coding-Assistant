"""dependency_graph.py — Sentinel module import/dependency graph.

Parses each Python file's import statements (using the built-in 'ast'
module) and builds a directed graph of module-level dependencies.

Provides:
  - Import edge collection per file
  - Forward/reverse adjacency lookups
  - Cycle detection (DFS-based)
  - Identification of third-party vs first-party vs stdlib imports

No external dependencies required — stdlib only (ast, pathlib, sys).
"""

import ast
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterator, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Stdlib module names (subset used for first-party detection)
# ---------------------------------------------------------------------------

_STDLIB_TOP_LEVEL: FrozenSet[str] = frozenset(
    sys.stdlib_module_names
    if hasattr(sys, "stdlib_module_names")   # Python 3.10+
    else {
        "abc", "asyncio", "ast", "builtins", "collections", "contextlib",
        "copy", "dataclasses", "datetime", "enum", "functools", "hashlib",
        "importlib", "inspect", "io", "itertools", "json", "logging",
        "math", "os", "pathlib", "pickle", "platform", "queue", "random",
        "re", "shutil", "signal", "socket", "sqlite3", "string", "struct",
        "subprocess", "sys", "tempfile", "textwrap", "threading", "time",
        "traceback", "typing", "unittest", "urllib", "uuid", "warnings",
        "weakref", "xml", "zipfile",
    }
)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class ImportEdge:
    """A single import relationship extracted from a Python file.

    Attributes:
        importer: Repository-relative path of the importing file.
        module: Fully qualified module being imported (e.g. ``"os.path"``).
        names: Specific names imported via ``from X import Y``.  Empty for
            plain ``import X`` statements.
        line: Source line number (1-based).
        level: Relative import level (0 = absolute, 1 = from., 2 = from..).
        import_kind: One of ``"stdlib"``, ``"first_party"``,
            ``"third_party"``, or ``"relative"``.
    """

    importer: str
    module: str
    names: List[str]
    line: int
    level: int
    import_kind: str


# ---------------------------------------------------------------------------
# Dependency Graph
# ---------------------------------------------------------------------------


class DependencyGraph:
    """Directed dependency graph over a Python codebase.

    Each node is a file path (repository-relative).  Each edge is an
    ImportEdge from importer to imported module.

    Usage::

        dg = DependencyGraph()
        dg.add_file("sentinel/core/bootstrap.py", source_text, first_party_prefix="sentinel")
        dg.add_directory("sentinel/", first_party_prefix="sentinel")

        deps = dg.get_dependencies("sentinel/core/bootstrap.py")
        rdeps = dg.get_reverse_dependencies("sentinel/system/system_check.py")
        cycles = dg.detect_cycles()

    Attributes:
        edges: All ImportEdge objects collected.
    """

    def __init__(self) -> None:
        self.edges: List[ImportEdge] = []
        # file_path → list of modules this file imports
        self._forward: Dict[str, List[str]] = defaultdict(list)
        # module/file → list of files that import it
        self._reverse: Dict[str, List[str]] = defaultdict(list)
        # file_path → ImportEdge list
        self._file_edges: Dict[str, List[ImportEdge]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Building the graph
    # ------------------------------------------------------------------

    def add_file(
        self,
        file_path: str,
        source: str,
        first_party_prefix: str = "",
    ) -> int:
        """Parse a Python source file and add its imports to the graph.

        Args:
            file_path: Repository-relative path of the file (node identifier).
            source: Full Python source text.
            first_party_prefix: Module name prefix considered first-party
                (e.g. ``"sentinel"``).  Used to classify import_kind.

        Returns:
            Number of import edges added.
        """
        try:
            tree = ast.parse(source, filename=file_path)
        except SyntaxError:
            return 0

        new_edges = list(
            self._extract_imports(tree, file_path, first_party_prefix)
        )

        for edge in new_edges:
            self.edges.append(edge)
            self._file_edges[file_path].append(edge)
            self._forward[file_path].append(edge.module)
            self._reverse[edge.module].append(file_path)

        return len(new_edges)

    def add_directory(
        self,
        root: str,
        first_party_prefix: str = "",
    ) -> int:
        """Recursively index all Python files under a directory.

        Args:
            root: Absolute or relative path to the directory root.
            first_party_prefix: Module prefix for first-party classification.

        Returns:
            Total number of import edges added.
        """
        root_path = Path(root).resolve()
        skip_dirs = {"__pycache__", ".venv", "venv", ".git", "node_modules"}
        total = 0

        for py_file in root_path.rglob("*.py"):
            if any(part in skip_dirs for part in py_file.parts):
                continue
            try:
                source = py_file.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            try:
                rel = str(py_file.relative_to(root_path))
            except ValueError:
                rel = str(py_file)

            total += self.add_file(rel, source, first_party_prefix)

        return total

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get_dependencies(self, file_path: str) -> List[str]:
        """Return all module names directly imported by a file.

        Args:
            file_path: Repository-relative path of the importing file.

        Returns:
            Sorted, de-duplicated list of module name strings.
        """
        return sorted(set(self._forward.get(file_path, [])))

    def get_reverse_dependencies(self, module_or_file: str) -> List[str]:
        """Return all files that import the given module or file.

        Args:
            module_or_file: Module name (e.g. ``"os.path"``) or
                repository-relative file path.

        Returns:
            Sorted list of importer file paths.
        """
        return sorted(set(self._reverse.get(module_or_file, [])))

    def get_import_edges(self, file_path: str) -> List[ImportEdge]:
        """Return full ImportEdge objects for a specific file.

        Args:
            file_path: Repository-relative file path.

        Returns:
            List of ImportEdge objects.
        """
        return self._file_edges.get(file_path, [])

    def get_all_files(self) -> List[str]:
        """Return all known importer file paths.

        Returns:
            Sorted list of file path strings.
        """
        return sorted(self._file_edges.keys())

    def get_transitive_dependencies(
        self, file_path: str, include_stdlib: bool = False
    ) -> Set[str]:
        """BFS/DFS to collect all transitive imports from a file.

        Only traverses edges where the module name can be resolved to a
        file in the graph.

        Args:
            file_path: Starting file path.
            include_stdlib: If False, stdlib modules are excluded.

        Returns:
            Set of module/file strings reachable from the start.
        """
        visited: Set[str] = set()
        queue: deque[str] = deque([file_path])
        all_files = set(self._file_edges.keys())

        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            for dep in self._forward.get(current, []):
                if dep in visited:
                    continue
                if not include_stdlib and self._is_stdlib(dep):
                    continue
                if dep in all_files:
                    queue.append(dep)
                else:
                    visited.add(dep)

        visited.discard(file_path)
        return visited

    # ------------------------------------------------------------------
    # Cycle detection
    # ------------------------------------------------------------------

    def detect_cycles(self) -> List[List[str]]:
        """Find all import cycles in the graph (DFS-based).

        Only file-to-file edges are considered (skips unresolved third-party
        imports).

        Returns:
            List of cycles, where each cycle is a list of file path strings
            forming a loop (first element == last element).
        """
        known_files = set(self._file_edges.keys())
        visited: Set[str] = set()
        in_stack: Set[str] = set()
        stack: List[str] = []
        cycles: List[List[str]] = []

        def dfs(node: str) -> None:
            visited.add(node)
            in_stack.add(node)
            stack.append(node)

            for dep in self._forward.get(node, []):
                if dep not in known_files:
                    continue
                if dep not in visited:
                    dfs(dep)
                elif dep in in_stack:
                    cycle_start = stack.index(dep)
                    cycle = stack[cycle_start:] + [dep]
                    # Normalise to avoid duplicates
                    min_idx = cycle.index(min(cycle))
                    normalised = cycle[min_idx:] + cycle[1:min_idx + 1]
                    if normalised not in cycles:
                        cycles.append(normalised)

            stack.pop()
            in_stack.discard(node)

        for file in known_files:
            if file not in visited:
                dfs(file)

        return cycles

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return aggregate statistics about the graph.

        Returns:
            Dict with file count, edge count, cycle count, and kind breakdown.
        """
        from collections import Counter
        kind_counts = Counter(e.import_kind for e in self.edges)
        cycles = self.detect_cycles()
        return {
            "files": len(self._file_edges),
            "total_edges": len(self.edges),
            "cycles": len(cycles),
            "by_kind": dict(kind_counts),
        }

    def most_imported(self, top_n: int = 10) -> List[Tuple[str, int]]:
        """Return the most-imported modules ranked by importer count.

        Args:
            top_n: How many results to return.

        Returns:
            List of (module_name, importer_count) tuples, descending.
        """
        counts = {mod: len(importers) for mod, importers in self._reverse.items()}
        return sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_n]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_imports(
        self,
        tree: ast.AST,
        file_path: str,
        first_party_prefix: str,
    ) -> Iterator[ImportEdge]:
        """Yield ImportEdge objects from an AST's import statements.

        Args:
            tree: Parsed AST module.
            file_path: Importer file path.
            first_party_prefix: Prefix for first-party classification.

        Yields:
            ImportEdge for each import statement.
        """
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    kind = self._classify(alias.name, 0, first_party_prefix)
                    yield ImportEdge(
                        importer=file_path,
                        module=alias.name,
                        names=[alias.asname or alias.name],
                        line=node.lineno,
                        level=0,
                        import_kind=kind,
                    )

            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                level = node.level or 0
                names = [alias.name for alias in node.names]
                kind = self._classify(module, level, first_party_prefix)
                yield ImportEdge(
                    importer=file_path,
                    module=module,
                    names=names,
                    line=node.lineno,
                    level=level,
                    import_kind=kind,
                )

    def _classify(
        self,
        module: str,
        level: int,
        first_party_prefix: str,
    ) -> str:
        """Determine the kind of an import.

        Args:
            module: Module name string.
            level: Relative import level (0 = absolute).
            first_party_prefix: First-party module prefix.

        Returns:
            One of ``"relative"``, ``"stdlib"``, ``"first_party"``,
            ``"third_party"``.
        """
        if level > 0:
            return "relative"
        top = module.split(".")[0] if module else ""
        if top in _STDLIB_TOP_LEVEL:
            return "stdlib"
        if first_party_prefix and top == first_party_prefix:
            return "first_party"
        return "third_party"

    @staticmethod
    def _is_stdlib(module: str) -> bool:
        """Return True if the top-level module name is part of stdlib."""
        return module.split(".")[0] in _STDLIB_TOP_LEVEL
