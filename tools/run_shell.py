"""run_shell.py — Sentinel run_shell tool.

Executes a shell command in a subprocess and returns its stdout, stderr, and
return code.  A configurable timeout prevents runaway processes.

Security note
-------------
To reduce the risk of command injection a denylist of high-impact command
prefixes is checked against the resolved argv before execution.  Callers
should still avoid passing untrusted input directly to this tool.

Registered name: ``"run_shell"``
"""

import os
import shlex
import subprocess
from typing import Any, Dict, Optional

from tools.tool_registry import Tool, ToolResult

# Commands (first token of argv) that are blocked regardless of arguments.
_BLOCKED_COMMANDS = frozenset(
    {
        "rm",
        "rmdir",
        "del",
        "format",
        "mkfs",
        "dd",
        "shutdown",
        "reboot",
        "halt",
        "poweroff",
        "kill",
        "killall",
        "pkill",
        "taskkill",
        "reg",
        "regedit",
        "net",
        "netsh",
        "iptables",
        "chmod",
        "chown",
        "passwd",
        "sudo",
        "su",
        "runas",
        "curl",   # prefer the web_search or url_fetcher tools
        "wget",
    }
)

_DEFAULT_TIMEOUT = 30  # seconds
_MAX_OUTPUT_BYTES = 1 * 1024 * 1024  # 1 MB per stream


class RunShellTool(Tool):
    """Run an arbitrary shell command and capture its output.

    Parameters:
        command (str, required): The shell command to execute.
        cwd (str, optional): Working directory.  Defaults to ``"."``.
        timeout (int, optional): Timeout in seconds.  Defaults to 30.
        env_extra (dict, optional): Extra environment variables to merge with
            the current environment.
        shell (bool, optional): Pass command to the OS shell (``/bin/sh -c``
            on Unix, ``cmd.exe /c`` on Windows) when ``True``.
            Defaults to ``False`` (safer tokenised exec).
    """

    name = "run_shell"
    description = (
        "Execute a shell command and return its stdout, stderr, and exit code.  "
        "Dangerous commands (rm, sudo, etc.) are blocked."
    )
    parameters_schema = {
        "command": {
            "type": "string",
            "description": "The shell command to execute.",
            "required": True,
        },
        "cwd": {
            "type": "string",
            "description": "Working directory for the command.  Defaults to '.'.",
            "required": False,
            "default": ".",
        },
        "timeout": {
            "type": "int",
            "description": f"Timeout in seconds.  Defaults to {_DEFAULT_TIMEOUT}.",
            "required": False,
            "default": _DEFAULT_TIMEOUT,
        },
        "env_extra": {
            "type": "dict",
            "description": "Extra environment variables to overlay on the current environment.",
            "required": False,
            "default": None,
        },
        "shell": {
            "type": "bool",
            "description": (
                "Pass command to the OS shell when True (less safe).  "
                "Defaults to False (tokenised exec)."
            ),
            "required": False,
            "default": False,
        },
    }

    def run(  # type: ignore[override]
        self,
        command: str,
        cwd: str = ".",
        timeout: int = _DEFAULT_TIMEOUT,
        env_extra: Optional[Dict[str, str]] = None,
        shell: bool = False,
        **_: Any,
    ) -> ToolResult:
        """Run *command* and return captured output.

        Args:
            command: Shell command string.
            cwd: Working directory.
            timeout: Maximum execution time in seconds.
            env_extra: Additional environment variables.
            shell: Whether to invoke via the OS shell.

        Returns:
            ToolResult with ``output`` as a dict containing ``stdout``,
            ``stderr``, and ``returncode``.
        """
        timeout = max(1, int(timeout))

        # --- security: denylist check ---
        if shell:
            # When shell=True we still try to extract the first token for checking.
            first_token = shlex.split(command)[0] if command.strip() else ""
        else:
            try:
                args = shlex.split(command)
            except ValueError as exc:
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    error=f"Could not parse command: {exc}",
                )
            first_token = args[0] if args else ""

        base_cmd = os.path.basename(first_token).lower()
        if base_cmd in _BLOCKED_COMMANDS:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=(
                    f"Command '{base_cmd}' is blocked for safety reasons.  "
                    "Use a dedicated tool or request explicit user approval."
                ),
            )

        # --- environment ---
        env = dict(os.environ)
        if env_extra:
            env.update({str(k): str(v) for k, v in env_extra.items()})

        # --- execution ---
        cmd_arg = command if shell else args  # type: ignore[possibly-undefined]
        try:
            result = subprocess.run(
                cmd_arg,
                shell=shell,
                cwd=cwd,
                env=env,
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"Command timed out after {timeout}s.",
                metadata={"command": command, "timeout": timeout},
            )
        except FileNotFoundError as exc:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"Command not found: {exc}",
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=str(exc),
            )

        stdout = result.stdout[:_MAX_OUTPUT_BYTES].decode("utf-8", errors="replace")
        stderr = result.stderr[:_MAX_OUTPUT_BYTES].decode("utf-8", errors="replace")
        success = result.returncode == 0

        return ToolResult(
            tool_name=self.name,
            success=success,
            output={"stdout": stdout, "stderr": stderr, "returncode": result.returncode},
            error=None if success else f"Process exited with code {result.returncode}.",
            metadata={"command": command, "cwd": str(cwd), "timeout": timeout},
        )
