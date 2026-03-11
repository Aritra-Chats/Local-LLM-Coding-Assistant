"""run_tests.py — Sentinel run_tests tool.

Runs the project test suite via ``pytest`` (or ``unittest``) and returns a
structured summary including pass/fail/error counts and captured output.

Registered name: ``"run_tests"``
"""

import re
import subprocess
import sys
from pathlib import Path
from typing import Any, List, Optional

from tools.tool_registry import Tool, ToolResult

_DEFAULT_TIMEOUT = 120  # seconds
_MAX_OUTPUT_BYTES = 2 * 1024 * 1024  # 2 MB

# Patterns to extract pytest summary line, e.g.
#   "3 passed, 1 failed, 2 errors in 0.42s"
_SUMMARY_RE = re.compile(
    r"(?P<passed>\d+) passed"
    r"|(?P<failed>\d+) failed"
    r"|(?P<error>\d+) error",
    re.IGNORECASE,
)


def _parse_summary(text: str) -> dict:
    """Extract pass/fail/error counts from pytest terminal output."""
    counts = {"passed": 0, "failed": 0, "errors": 0}
    for m in _SUMMARY_RE.finditer(text):
        if m.group("passed"):
            counts["passed"] = int(m.group("passed"))
        if m.group("failed"):
            counts["failed"] = int(m.group("failed"))
        if m.group("error"):
            counts["errors"] = int(m.group("error"))
    return counts


class RunTestsTool(Tool):
    """Run the project test suite with pytest and return results.

    Parameters:
        path (str, optional): Test directory or file to run.
            Defaults to ``"."`` (project root — pytest auto-discovers).
        args (list[str], optional): Extra pytest CLI arguments, e.g.
            ``["-v", "--tb=short"]``.
        runner (str, optional): Test runner executable.
            Defaults to ``"pytest"``.  Use ``"unittest"`` to fall back to
            ``python -m unittest``.
        timeout (int, optional): Timeout in seconds.  Defaults to 120.
        python (str, optional): Python interpreter path.
            Defaults to the current interpreter (``sys.executable``).
    """

    name = "run_tests"
    description = (
        "Run the project test suite via pytest (or unittest) and return a "
        "structured summary with pass/fail/error counts and captured output."
    )
    parameters_schema = {
        "path": {
            "type": "string",
            "description": "Test directory or file to run.  Defaults to '.' (auto-discover).",
            "required": False,
            "default": ".",
        },
        "args": {
            "type": "list",
            "description": "Extra pytest CLI arguments, e.g. ['-v', '--tb=short'].",
            "required": False,
            "default": None,
        },
        "runner": {
            "type": "string",
            "description": "Test runner: 'pytest' (default) or 'unittest'.",
            "required": False,
            "default": "pytest",
        },
        "timeout": {
            "type": "int",
            "description": f"Timeout in seconds.  Defaults to {_DEFAULT_TIMEOUT}.",
            "required": False,
            "default": _DEFAULT_TIMEOUT,
        },
        "python": {
            "type": "string",
            "description": "Python interpreter to use.  Defaults to the current interpreter.",
            "required": False,
            "default": None,
        },
    }

    def run(  # type: ignore[override]
        self,
        path: str = ".",
        args: Optional[List[str]] = None,
        runner: str = "pytest",
        timeout: int = _DEFAULT_TIMEOUT,
        python: Optional[str] = None,
        **_: Any,
    ) -> ToolResult:
        """Execute the test suite.

        Args:
            path: Test path.
            args: Extra runner arguments.
            runner: ``"pytest"`` or ``"unittest"``.
            timeout: Max execution time in seconds.
            python: Python interpreter path.

        Returns:
            ToolResult with ``output`` as a dict containing ``stdout``,
            ``stderr``, ``returncode``, ``summary`` (pass/fail/error counts).
        """
        timeout = max(1, int(timeout))
        interpreter = python or sys.executable
        extra_args = list(args) if args else []

        resolved_path = Path(path).expanduser()

        if runner == "pytest":
            cmd = [interpreter, "-m", "pytest", str(resolved_path)] + extra_args
        elif runner == "unittest":
            cmd = [interpreter, "-m", "unittest", "discover", "-s", str(resolved_path)] + extra_args
        else:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"Unknown runner '{runner}'.  Use 'pytest' or 'unittest'.",
            )

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"Test run timed out after {timeout}s.",
                metadata={"cmd": cmd},
            )
        except FileNotFoundError as exc:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"Interpreter or runner not found: {exc}",
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=str(exc),
            )

        stdout = result.stdout[:_MAX_OUTPUT_BYTES].decode("utf-8", errors="replace")
        stderr = result.stderr[:_MAX_OUTPUT_BYTES].decode("utf-8", errors="replace")
        combined = stdout + stderr
        summary = _parse_summary(combined)
        success = result.returncode == 0

        return ToolResult(
            tool_name=self.name,
            success=success,
            output={
                "stdout": stdout,
                "stderr": stderr,
                "returncode": result.returncode,
                "summary": summary,
            },
            error=None if success else f"Tests failed (exit code {result.returncode}).",
            metadata={"cmd": cmd, "path": str(resolved_path)},
        )
