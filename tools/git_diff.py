"""git_diff.py — Sentinel git_diff tool.

Returns the output of ``git diff`` (working tree vs. HEAD by default).
Supports diffing a specific commit range or an individual file/directory.

Registered name: ``"git_diff"``
"""

import subprocess
from pathlib import Path
from typing import Any, Optional

from tools.tool_registry import Tool, ToolResult

_DEFAULT_TIMEOUT = 30


class GitDiffTool(Tool):
    """Show git diff output for the current repository.

    Parameters:
        path (str, optional): Restrict the diff to this file or directory.
        cwd (str, optional): Repository root.  Defaults to ``"."``.
        staged (bool, optional): Show staged changes (``--cached``) instead
            of unstaged.  Defaults to ``False``.
        base (str, optional): Base ref, e.g. ``"HEAD~1"`` or a commit SHA.
            When omitted the working-tree vs. HEAD diff is shown.
        compare (str, optional): Comparison ref.  When both *base* and
            *compare* are given, runs ``git diff <base> <compare>``.
        timeout (int, optional): Timeout in seconds.  Defaults to 30.
    """

    name = "git_diff"
    description = (
        "Return the output of git diff for the current repository.  "
        "Supports staged, unstaged, and ref-range diffs."
    )
    parameters_schema = {
        "path": {
            "type": "string",
            "description": "Restrict diff to a specific file or directory.",
            "required": False,
            "default": None,
        },
        "cwd": {
            "type": "string",
            "description": "Repository root directory.  Defaults to '.'.",
            "required": False,
            "default": ".",
        },
        "staged": {
            "type": "bool",
            "description": "Show staged (cached) diff when True.  Defaults to False.",
            "required": False,
            "default": False,
        },
        "base": {
            "type": "string",
            "description": "Base ref/commit for the diff, e.g. 'HEAD~1'.",
            "required": False,
            "default": None,
        },
        "compare": {
            "type": "string",
            "description": "Comparison ref.  Used together with 'base' for a range diff.",
            "required": False,
            "default": None,
        },
        "timeout": {
            "type": "int",
            "description": f"Timeout in seconds.  Defaults to {_DEFAULT_TIMEOUT}.",
            "required": False,
            "default": _DEFAULT_TIMEOUT,
        },
    }

    def run(  # type: ignore[override]
        self,
        path: Optional[str] = None,
        cwd: str = ".",
        staged: bool = False,
        base: Optional[str] = None,
        compare: Optional[str] = None,
        timeout: int = _DEFAULT_TIMEOUT,
        **_: Any,
    ) -> ToolResult:
        """Run ``git diff`` and return the patch text.

        Returns:
            ToolResult with ``output`` as the diff string.
        """
        cmd = ["git", "diff"]

        if staged:
            cmd.append("--cached")

        if base and compare:
            cmd += [base, compare]
        elif base:
            cmd.append(base)

        if path:
            cmd += ["--", str(Path(path).expanduser())]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(Path(cwd).expanduser()),
                capture_output=True,
                timeout=int(timeout),
            )
        except FileNotFoundError:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error="git executable not found.  Ensure git is installed and on PATH.",
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"git diff timed out after {timeout}s.",
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=str(exc),
            )

        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace")
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=stderr or f"git diff exited with code {result.returncode}.",
            )

        diff_text = result.stdout.decode("utf-8", errors="replace")
        return ToolResult(
            tool_name=self.name,
            success=True,
            output=diff_text,
            metadata={
                "cmd": cmd,
                "cwd": str(cwd),
                "has_changes": bool(diff_text.strip()),
            },
        )
