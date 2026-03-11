"""write_file.py — Sentinel write_file tool.

Writes or appends text content to a local file.  Parent directories are
created automatically when they do not exist.

Registered name: ``"write_file"``
"""

from pathlib import Path
from typing import Any

from tools.tool_registry import Tool, ToolResult


class WriteFileTool(Tool):
    """Write text to a local file (overwrite or append).

    Parameters:
        path (str, required): Path to the file to write.
        content (str, required): Text content to write.
        mode (str, optional): ``"overwrite"`` (default) or ``"append"``.
        encoding (str, optional): File encoding.  Defaults to ``"utf-8"``.
        create_parents (bool, optional): Create missing parent directories.
            Defaults to ``True``.
    """

    name = "write_file"
    description = (
        "Write or append text content to a local file.  "
        "Parent directories are created automatically by default."
    )
    parameters_schema = {
        "path": {
            "type": "string",
            "description": "Absolute or relative path to the destination file.",
            "required": True,
        },
        "content": {
            "type": "string",
            "description": "Text content to write to the file.",
            "required": True,
        },
        "mode": {
            "type": "string",
            "description": "Write mode: 'overwrite' (default) or 'append'.",
            "required": False,
            "default": "overwrite",
        },
        "encoding": {
            "type": "string",
            "description": "File encoding.  Defaults to utf-8.",
            "required": False,
            "default": "utf-8",
        },
        "create_parents": {
            "type": "bool",
            "description": "Create missing parent directories when True (default).",
            "required": False,
            "default": True,
        },
    }

    def run(  # type: ignore[override]
        self,
        path: str,
        content: str,
        mode: str = "overwrite",
        encoding: str = "utf-8",
        create_parents: bool = True,
        **_: Any,
    ) -> ToolResult:
        """Write *content* to *path*.

        Args:
            path: Destination file path.
            content: Text to write.
            mode: ``"overwrite"`` or ``"append"``.
            encoding: Text encoding.
            create_parents: If ``True`` (default), create missing parent dirs.

        Returns:
            ToolResult with ``output`` set to a brief summary string.
        """
        if mode not in ("overwrite", "append"):
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"Invalid mode '{mode}'.  Must be 'overwrite' or 'append'.",
            )

        resolved = Path(path).expanduser()

        if create_parents:
            resolved.parent.mkdir(parents=True, exist_ok=True)
        elif not resolved.parent.exists():
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"Parent directory does not exist: {resolved.parent}",
            )

        file_mode = "a" if mode == "append" else "w"
        try:
            resolved.write_text(content, encoding=encoding) if mode == "overwrite" else \
                resolved.open("a", encoding=encoding).write(content)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=str(exc),
            )

        bytes_written = len(content.encode(encoding, errors="replace"))
        action = "appended" if mode == "append" else "written"
        return ToolResult(
            tool_name=self.name,
            success=True,
            output=f"Successfully {action} {bytes_written} bytes to {resolved}.",
            metadata={
                "path": str(resolved),
                "mode": mode,
                "bytes_written": bytes_written,
                "encoding": encoding,
            },
        )
