"""find_files.py — Sentinel find_files tool.

Searches for files by name pattern (glob) or content search, returning
matching file paths. Useful for discovering files when exact paths are unknown.

Registered name: ``"find_files"``
"""

import re
from pathlib import Path
from typing import Any, List, Optional

from tools.tool_registry import Tool, ToolResult

_DEFAULT_MAX_RESULTS = 100
_HARD_MAX_RESULTS = 1000


class FindFilesTool(Tool):
    """Find files by name pattern (glob) or content match.

    Parameters:
        pattern (str, required): Filename glob pattern (e.g. "*.test.js") or
            content search term (when search_content=True).
        path (str, optional): Root directory to search.
            Defaults to ``"."`` (current directory).
        search_content (bool, optional): If True, search file contents instead of
            filenames. Defaults to False.
        max_results (int, optional): Maximum matches to return.
            Defaults to 100.
    """

    name = "find_files"
    description = (
        "Find files by name pattern (glob) or content search. "
        "Returns a sorted list of matching file paths relative to the search root."
    )
    parameters_schema = {
        "pattern": {
            "type": "string",
            "description": (
                "Filename glob pattern (e.g. '*.test.js' or '**/App.js') "
                "or content search term when search_content=True."
            ),
            "required": True,
        },
        "path": {
            "type": "string",
            "description": "Root directory to search. Defaults to '.'.",
            "required": False,
            "default": ".",
        },
        "search_content": {
            "type": "bool",
            "description": "If True, search file contents instead of filenames.",
            "required": False,
            "default": False,
        },
        "max_results": {
            "type": "int",
            "description": f"Maximum matches to return. Defaults to {_DEFAULT_MAX_RESULTS}.",
            "required": False,
            "default": _DEFAULT_MAX_RESULTS,
        },
    }

    def run(  # type: ignore[override]
        self,
        pattern: str,
        path: str = ".",
        search_content: bool = False,
        max_results: int = _DEFAULT_MAX_RESULTS,
        project_root: str = "",
        **_: Any,
    ) -> ToolResult:
        """Execute file search by name or content.

        Args:
            pattern: Glob pattern (e.g. "*.js") or search term.
            path: Root directory to search. Defaults to ".".
            search_content: Search file contents if True, else filenames.
            max_results: Cap on total matches returned.
            project_root: Optional project root for path resolution.

        Returns:
            ToolResult with ``output`` as a dict containing ``matches`` (list of
            matching file paths relative to the search root) and ``count``.
        """
        max_results = min(max(1, int(max_results)), _HARD_MAX_RESULTS)

        # --- Resolve root directory ---
        root = Path(path).expanduser()
        if project_root and not root.is_absolute():
            root = Path(project_root) / path
        root = root.resolve()

        if not root.exists():
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"Path does not exist: {root}",
            )

        matches: List[str] = []

        if search_content:
            # --- Search file contents ---
            try:
                flags = re.IGNORECASE
                regex = re.compile(re.escape(pattern), flags)
            except re.error as exc:
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    error=f"Invalid search pattern '{pattern}': {exc}",
                )

            for fpath in root.rglob("*"):
                if not fpath.is_file() or len(matches) >= max_results:
                    break

                # Skip large files and binary extensons
                try:
                    if fpath.stat().st_size > 10 * 1024 * 1024:  # 10 MB
                        continue
                except OSError:
                    continue

                # Skip common binary extensions
                if fpath.suffix in (
                    ".pyc",
                    ".o",
                    ".so",
                    ".dll",
                    ".exe",
                    ".zip",
                    ".tar",
                    ".gz",
                ):
                    continue

                try:
                    content = fpath.read_text(errors="ignore")
                    if regex.search(content):
                        rel_path = fpath.relative_to(root)
                        matches.append(str(rel_path).replace("\\", "/"))
                except Exception:
                    pass

        else:
            # --- Search by filename pattern (glob) ---
            try:
                for fpath in root.rglob(pattern):
                    if fpath.is_file() and len(matches) < max_results:
                        rel_path = fpath.relative_to(root)
                        matches.append(str(rel_path).replace("\\", "/"))
            except ValueError as exc:
                # rglob may fail on invalid patterns
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    error=f"Invalid glob pattern '{pattern}': {exc}",
                )

        matches = sorted(matches)
        return ToolResult(
            tool_name=self.name,
            success=True,
            output={"matches": matches, "count": len(matches)},
        )
