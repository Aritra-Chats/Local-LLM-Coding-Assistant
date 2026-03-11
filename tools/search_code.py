"""search_code.py — Sentinel search_code tool.

Searches files under a directory for a literal string or regex pattern,
mirroring the behaviour of ``grep -r``.  Results are returned as a list of
match objects containing the file path, line number, and matched line.

Registered name: ``"search_code"``
"""

import re
from pathlib import Path
from typing import Any, List, Optional

from tools.tool_registry import Tool, ToolResult

# Cap total matches to avoid huge payloads
_DEFAULT_MAX_RESULTS = 200
_HARD_MAX_RESULTS = 2000

# Files larger than this are skipped to keep things fast
_MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB


class SearchCodeTool(Tool):
    """Search source files for a literal string or regex pattern.

    Parameters:
        query (str, required): The search term or regular expression.
        path (str, optional): Root directory (or single file) to search.
            Defaults to the current working directory.
        glob (str, optional): File glob filter, e.g. ``"*.py"`` or
            ``"**/*.ts"``.  Defaults to ``"**/*"`` (all files).
        is_regex (bool, optional): Treat *query* as a Python regex when
            ``True``.  Defaults to ``False`` (literal match).
        case_sensitive (bool, optional): Case-sensitive match.
            Defaults to ``False``.
        max_results (int, optional): Maximum number of matches to return.
            Defaults to 200.
    """

    name = "search_code"
    description = (
        "Search files in a directory for a literal string or regex pattern.  "
        "Returns a list of matching lines with file path and line number."
    )
    parameters_schema = {
        "query": {
            "type": "string",
            "description": "The search term or regular expression to look for.",
            "required": True,
        },
        "path": {
            "type": "string",
            "description": "Root directory or specific file to search.  Defaults to '.'.",
            "required": False,
            "default": ".",
        },
        "glob": {
            "type": "string",
            "description": "Glob pattern to filter files, e.g. '*.py'.  Defaults to '**/*'.",
            "required": False,
            "default": "**/*",
        },
        "is_regex": {
            "type": "bool",
            "description": "Treat query as a Python regex expression when True.",
            "required": False,
            "default": False,
        },
        "case_sensitive": {
            "type": "bool",
            "description": "Perform a case-sensitive match when True.",
            "required": False,
            "default": False,
        },
        "max_results": {
            "type": "int",
            "description": f"Maximum number of matches to return.  Defaults to {_DEFAULT_MAX_RESULTS}.",
            "required": False,
            "default": _DEFAULT_MAX_RESULTS,
        },
    }

    def run(  # type: ignore[override]
        self,
        query: str,
        path: str = ".",
        glob: str = "**/*",
        is_regex: bool = False,
        case_sensitive: bool = False,
        max_results: int = _DEFAULT_MAX_RESULTS,
        **_: Any,
    ) -> ToolResult:
        """Execute the search and return matching lines.

        Args:
            query: Search term or regex.
            path: Root directory or file to search.
            glob: Glob pattern to restrict which files are searched.
            is_regex: If ``True`` treat *query* as a regex.
            case_sensitive: If ``False`` (default) match case-insensitively.
            max_results: Cap on total matches returned.

        Returns:
            ToolResult whose ``output`` is a list of dicts:
            ``{"file": str, "line_number": int, "line": str}``.
        """
        max_results = min(max(1, int(max_results)), _HARD_MAX_RESULTS)

        # --- compile pattern ---
        flags = 0 if case_sensitive else re.IGNORECASE
        if is_regex:
            try:
                pattern = re.compile(query, flags)
            except re.error as exc:
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    error=f"Invalid regex '{query}': {exc}",
                )
        else:
            pattern = re.compile(re.escape(query), flags)

        # --- resolve root ---
        root = Path(path).expanduser().resolve()
        if not root.exists():
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"Path does not exist: {root}",
            )

        # --- collect candidate files ---
        if root.is_file():
            candidates = [root]
        else:
            candidates = [p for p in root.glob(glob) if p.is_file()]

        matches: List[dict] = []
        files_searched = 0
        files_skipped = 0

        for candidate in candidates:
            if len(matches) >= max_results:
                break
            if candidate.stat().st_size > _MAX_FILE_BYTES:
                files_skipped += 1
                continue

            try:
                text = candidate.read_text(encoding="utf-8", errors="replace")
            except Exception:  # noqa: BLE001 — unreadable file
                files_skipped += 1
                continue

            files_searched += 1
            for lineno, line in enumerate(text.splitlines(), start=1):
                if pattern.search(line):
                    matches.append(
                        {
                            "file": str(candidate),
                            "line_number": lineno,
                            "line": line.rstrip("\r\n"),
                        }
                    )
                    if len(matches) >= max_results:
                        break

        truncated = len(matches) >= max_results

        return ToolResult(
            tool_name=self.name,
            success=True,
            output=matches,
            metadata={
                "query": query,
                "root": str(root),
                "files_searched": files_searched,
                "files_skipped": files_skipped,
                "total_matches": len(matches),
                "truncated": truncated,
            },
        )
