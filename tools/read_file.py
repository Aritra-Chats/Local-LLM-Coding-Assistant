"""read_file.py — Sentinel read_file tool.

Reads a file from the local filesystem and returns its content as a string.
Supports optional line-range extraction and encoding override.

Registered name: ``"read_file"``
"""

from pathlib import Path
from typing import Any, Optional

from tools.tool_registry import Tool, ToolResult

# Maximum bytes to read before truncating (5 MB)
_MAX_BYTES = 5 * 1024 * 1024


class ReadFileTool(Tool):
    """Read a local file and return its text content.

    Parameters:
        path (str, required): Absolute or project-relative path to the file.
        start_line (int, optional): First line to return (1-based, inclusive).
        end_line (int, optional): Last line to return (1-based, inclusive).
        encoding (str, optional): File encoding.  Defaults to ``"utf-8"``.
    """

    name = "read_file"
    description = "Read a local file and return its text content, optionally restricted to a line range."
    parameters_schema = {
        "path": {
            "type": "string",
            "description": "Absolute or relative path to the file to read.",
            "required": True,
        },
        "start_line": {
            "type": "int",
            "description": "First line number to return (1-based, inclusive).",
            "required": False,
            "default": None,
        },
        "end_line": {
            "type": "int",
            "description": "Last line number to return (1-based, inclusive).",
            "required": False,
            "default": None,
        },
        "encoding": {
            "type": "string",
            "description": "File encoding.  Defaults to utf-8.",
            "required": False,
            "default": "utf-8",
        },
        "project_root": {
            "type": "string",
            "description": "Project root directory for resolving relative paths (injected automatically).",
            "required": False,
            "default": "",
        },
    }

    def run(  # type: ignore[override]
        self,
        path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        encoding: str = "utf-8",
        project_root: str = "",
        **_: Any,
    ) -> ToolResult:
        """Read the file and return its contents.

        Args:
            path: Path to the file (absolute or relative to project_root).
            start_line: Optional start line (1-based).
            end_line: Optional end line (1-based).
            encoding: Text encoding.
            project_root: Optional project root directory for resolving relative paths.

        Returns:
            ToolResult with ``output`` set to the file content string.
        """
        # Resolve path relative to project_root if provided and path is relative
        target_path = Path(path).expanduser()
        if project_root and not target_path.is_absolute():
            target_path = (Path(project_root) / path).resolve()
        else:
            target_path = target_path.resolve()
            
        if not target_path.exists():
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"File not found: {target_path}",
            )
        if not target_path.is_file():
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"Path is not a file: {target_path}",
            )

        size = target_path.stat().st_size
        raw = target_path.read_bytes()[:_MAX_BYTES]
        truncated = size > _MAX_BYTES

        try:
            text = raw.decode(encoding, errors="replace")
        except LookupError:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"Unknown encoding: {encoding}",
            )

        lines = text.splitlines(keepends=True)
        total_lines = len(lines)

        if start_line is not None or end_line is not None:
            s = max(0, (start_line or 1) - 1)
            e = (end_line or total_lines)
            lines = lines[s:e]
            text = "".join(lines)

        return ToolResult(
            tool_name=self.name,
            success=True,
            output=text,
            metadata={
                "path": str(target_path),
                "size_bytes": size,
                "total_lines": total_lines,
                "truncated": truncated,
                "encoding": encoding,
            },
        )
