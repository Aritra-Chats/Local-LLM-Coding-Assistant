"""sandbox.py — Sandboxed execution for shell and test tools.

Provides a restricted subprocess wrapper that:
- Blocks known dangerous commands (rm -rf /, format, etc.)
- Enforces timeouts
- Captures stdout/stderr
- Restricts working directory to project root when provided
"""
from __future__ import annotations
import os
import re
import subprocess
import shlex
from typing import Any, Dict, List, Optional

# Commands that are always blocked regardless of context
_BLOCKED_PATTERNS = [
    r"\brm\s+-rf\s+/",          # rm -rf /
    r"\bmkfs\b",                 # filesystem format
    r"\bdd\s+if=",               # disk destroyer
    r"\bshutdown\b",             # system shutdown
    r"\breboot\b",               # reboot
    r":(){ :|:& };:",            # fork bomb
    r"\bchmod\s+777\s+/",        # chmod 777 /
    r"\b>\s*/dev/sd",            # overwrite disk
]

_BLOCKED_RE = re.compile("|".join(_BLOCKED_PATTERNS))


class Sandbox:
    """Restricted subprocess executor for Sentinel tool invocations.

    Args:
        project_root: If set, shell commands run with this as cwd.
        allowed_commands: If set, only these base commands are permitted.
    """

    def __init__(
        self,
        project_root: Optional[str] = None,
        allowed_commands: Optional[List[str]] = None,
    ) -> None:
        self.project_root = project_root
        self.allowed_commands = set(allowed_commands) if allowed_commands else None

    def run(self, command: str, timeout: int = 30) -> Dict[str, Any]:
        """Run a shell command with safety checks.

        Args:
            command: Shell command string to execute.
            timeout: Maximum seconds before killing the process.

        Returns:
            Dict with stdout, stderr, returncode, success, blocked keys.
        """
        # Safety check: block dangerous patterns
        if _BLOCKED_RE.search(command):
            return {
                "stdout": "",
                "stderr": f"Blocked: command matches a restricted pattern.",
                "returncode": -1,
                "success": False,
                "blocked": True,
            }

        # If allowed_commands is set, restrict to those base commands
        if self.allowed_commands:
            try:
                parts = shlex.split(command)
                base_cmd = os.path.basename(parts[0]) if parts else ""
                if base_cmd not in self.allowed_commands:
                    return {
                        "stdout": "",
                        "stderr": f"Blocked: '{base_cmd}' is not in allowed commands list.",
                        "returncode": -1,
                        "success": False,
                        "blocked": True,
                    }
            except ValueError:
                pass  # malformed quoting — let subprocess handle it

        cwd = self.project_root if self.project_root and os.path.isdir(self.project_root) else None

        try:
            result = subprocess.run(
                shlex.split(command),
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
            )
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "success": result.returncode == 0,
                "blocked": False,
            }
        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": f"Command timed out after {timeout}s.",
                "returncode": -1,
                "success": False,
                "blocked": False,
            }
        except Exception as e:
            return {
                "stdout": "",
                "stderr": str(e),
                "returncode": -1,
                "success": False,
                "blocked": False,
            }
