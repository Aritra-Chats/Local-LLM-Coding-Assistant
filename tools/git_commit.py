"""git_commit.py — Sentinel git_commit tool.

Stages files and creates a git commit.  Optionally accepts a list of paths
to stage (``git add``); when no paths are provided every tracked modification
is staged (``git add -u``).

Registered name: ``"git_commit"``
"""

import subprocess
from pathlib import Path
from typing import Any, List, Optional

from tools.tool_registry import Tool, ToolResult

_DEFAULT_TIMEOUT = 30


class GitCommitTool(Tool):
    """Stage files and create a git commit.

    Parameters:
        message (str, required): Commit message.
        paths (list[str], optional): Specific paths to stage with
            ``git add``.  When omitted, runs ``git add -u`` to stage all
            tracked changed files.
        add_all (bool, optional): Run ``git add -A`` (stage everything
            including untracked files) when ``True``.  Defaults to ``False``.
        cwd (str, optional): Repository root.  Defaults to ``"."``.
        timeout (int, optional): Timeout in seconds.  Defaults to 30.
    """

    name = "git_commit"
    description = (
        "Stage files and create a git commit with the given message.  "
        "Optionally specify which paths to stage."
    )
    parameters_schema = {
        "message": {
            "type": "string",
            "description": "Commit message.",
            "required": True,
        },
        "paths": {
            "type": "list",
            "description": "Paths to stage.  Omit to stage all tracked modified files.",
            "required": False,
            "default": None,
        },
        "add_all": {
            "type": "bool",
            "description": "Stage all changes including untracked files when True.",
            "required": False,
            "default": False,
        },
        "cwd": {
            "type": "string",
            "description": "Repository root directory.  Defaults to '.'.",
            "required": False,
            "default": ".",
        },
        "timeout": {
            "type": "int",
            "description": f"Timeout per git subcommand in seconds.  Defaults to {_DEFAULT_TIMEOUT}.",
            "required": False,
            "default": _DEFAULT_TIMEOUT,
        },
    }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _run(cmd: List[str], cwd: str, timeout: int):
        """Run a git subcommand; return (returncode, stdout, stderr)."""
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            timeout=timeout,
        )
        stdout = result.stdout.decode("utf-8", errors="replace")
        stderr = result.stderr.decode("utf-8", errors="replace")
        return result.returncode, stdout, stderr

    # ------------------------------------------------------------------

    def run(  # type: ignore[override]
        self,
        message: str,
        paths: Optional[List[str]] = None,
        add_all: bool = False,
        cwd: str = ".",
        timeout: int = _DEFAULT_TIMEOUT,
        **_: Any,
    ) -> ToolResult:
        """Stage and commit.

        Args:
            message: Commit message.
            paths: Files/directories to stage explicitly.
            add_all: Stage everything (``git add -A``) when ``True``.
            cwd: Working directory.
            timeout: Per-command timeout.

        Returns:
            ToolResult with ``output`` containing the commit SHA and summary.
        """
        cwd_str = str(Path(cwd).expanduser())
        t = max(1, int(timeout))

        # --- stage ---
        try:
            if add_all:
                add_cmd = ["git", "add", "-A"]
            elif paths:
                add_cmd = ["git", "add", "--"] + [str(p) for p in paths]
            else:
                add_cmd = ["git", "add", "-u"]

            rc, _, stderr = self._run(add_cmd, cwd_str, t)
        except FileNotFoundError:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error="git executable not found.",
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"git add timed out after {t}s.",
            )

        if rc != 0:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"git add failed: {stderr}",
            )

        # --- commit ---
        try:
            rc, stdout, stderr = self._run(
                ["git", "commit", "-m", message], cwd_str, t
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"git commit timed out after {t}s.",
            )

        if rc != 0:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"git commit failed: {stderr or stdout}",
            )

        # Extract short SHA from stdout ("master 1a2b3c4 …")
        sha = ""
        for token in stdout.split():
            if len(token) >= 7 and all(c in "0123456789abcdef" for c in token.lower()):
                sha = token
                break

        return ToolResult(
            tool_name=self.name,
            success=True,
            output=stdout.strip(),
            metadata={"sha": sha, "message": message, "cwd": cwd_str},
        )
