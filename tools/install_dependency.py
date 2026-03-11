"""install_dependency.py — Sentinel install_dependency tool.

Installs one or more Python packages into the active (or a specified)
Python environment using ``pip install``.  Returns structured output
including the packages that were installed or already satisfied.

Registered name: ``"install_dependency"``
"""

import re
import subprocess
import sys
from typing import Any, List, Optional

from tools.tool_registry import Tool, ToolResult

_DEFAULT_TIMEOUT = 120
_MAX_OUTPUT_BYTES = 2 * 1024 * 1024

# pip success patterns
_ALREADY_SATISFIED_RE = re.compile(r"Requirement already satisfied", re.IGNORECASE)
_SUCCESSFULLY_INSTALLED_RE = re.compile(
    r"Successfully installed (.+)", re.IGNORECASE
)


class InstallDependencyTool(Tool):
    """Install one or more Python packages with pip.

    Parameters:
        packages (list[str] or str, required): Package name(s) to install,
            e.g. ``["requests", "rich>=13"]`` or ``"numpy"``.
        upgrade (bool, optional): Pass ``--upgrade`` to pip.
            Defaults to ``False``.
        python (str, optional): Python interpreter whose pip to use.
            Defaults to the current interpreter.
        index_url (str, optional): Custom PyPI index URL.
        extra_args (list[str], optional): Additional raw pip arguments.
        timeout (int, optional): Timeout in seconds.  Defaults to 120.
    """

    name = "install_dependency"
    description = (
        "Install Python packages using pip and return a structured summary "
        "of what was installed or already satisfied."
    )
    parameters_schema = {
        "packages": {
            "type": "list",
            "description": "Package name(s) to install, e.g. ['requests', 'rich>=13'].",
            "required": True,
        },
        "upgrade": {
            "type": "bool",
            "description": "Pass --upgrade to pip when True.  Defaults to False.",
            "required": False,
            "default": False,
        },
        "python": {
            "type": "string",
            "description": "Python interpreter path.  Defaults to the current interpreter.",
            "required": False,
            "default": None,
        },
        "index_url": {
            "type": "string",
            "description": "Custom PyPI index URL, e.g. 'https://pypi.org/simple'.",
            "required": False,
            "default": None,
        },
        "extra_args": {
            "type": "list",
            "description": "Extra raw arguments to pass to pip install.",
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
        packages,
        upgrade: bool = False,
        python: Optional[str] = None,
        index_url: Optional[str] = None,
        extra_args: Optional[List[str]] = None,
        timeout: int = _DEFAULT_TIMEOUT,
        **_: Any,
    ) -> ToolResult:
        """Install *packages* via pip.

        Args:
            packages: Package spec(s) — string or list of strings.
            upgrade: Add ``--upgrade`` flag.
            python: Interpreter path.
            index_url: Custom index URL.
            extra_args: Raw extra pip arguments.
            timeout: Max execution time.

        Returns:
            ToolResult with ``output`` as a dict containing ``stdout``,
            ``stderr``, ``installed`` list, and ``already_satisfied`` flag.
        """
        # normalise packages to a list
        if isinstance(packages, str):
            packages = [packages]
        packages = [str(p).strip() for p in packages if str(p).strip()]
        if not packages:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error="No packages specified.",
            )

        interpreter = python or sys.executable
        cmd = [interpreter, "-m", "pip", "install"]

        if upgrade:
            cmd.append("--upgrade")

        if index_url:
            cmd += ["--index-url", index_url]

        cmd += packages

        if extra_args:
            cmd += [str(a) for a in extra_args]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=max(1, int(timeout)),
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"pip install timed out after {timeout}s.",
                metadata={"packages": packages},
            )
        except FileNotFoundError as exc:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"Interpreter not found: {exc}",
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

        # Parse installed package names
        installed: List[str] = []
        m = _SUCCESSFULLY_INSTALLED_RE.search(combined)
        if m:
            installed = m.group(1).split()

        already_satisfied = bool(_ALREADY_SATISFIED_RE.search(combined)) and not installed
        success = result.returncode == 0

        return ToolResult(
            tool_name=self.name,
            success=success,
            output={
                "stdout": stdout,
                "stderr": stderr,
                "returncode": result.returncode,
                "installed": installed,
                "already_satisfied": already_satisfied,
            },
            error=None if success else f"pip exited with code {result.returncode}.",
            metadata={"packages": packages, "cmd": cmd},
        )
