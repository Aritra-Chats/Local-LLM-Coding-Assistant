"""open_application.py — Sentinel open_application tool.

Opens a file, URL, or application using the host operating system's default
handler (``os.startfile`` on Windows, ``xdg-open`` on Linux, ``open`` on
macOS).

Security note
-------------
To prevent SSRF and unintended process launches, the tool validates that the
target is either a local filesystem path or an ``http``/``https`` URL.
Other URI schemes (``file://``, ``ftp://``, etc.) and raw command strings
are rejected.

Registered name: ``"open_application"``
"""

import os
import platform
import subprocess
import urllib.parse
from pathlib import Path
from typing import Any, Optional

from tools.tool_registry import Tool, ToolResult

# Allowed URL schemes when target looks like a URL
_ALLOWED_SCHEMES = {"http", "https"}


def _is_url(target: str) -> bool:
    try:
        parsed = urllib.parse.urlparse(target)
        return parsed.scheme.lower() in _ALLOWED_SCHEMES
    except Exception:  # noqa: BLE001
        return False


class OpenApplicationTool(Tool):
    """Open a local file or URL with the system default application.

    Parameters:
        target (str, required): Local file path or an ``http``/``https`` URL
            to open.
        application (str, optional): Specific application to launch.
            When provided, the tool runs ``<application> <target>`` directly
            instead of the system default handler.
        timeout (int, optional): Wait timeout when *wait* is ``True``.
            Defaults to 10 seconds.
        wait (bool, optional): Block until the launched process exits when
            ``True``.  Defaults to ``False``.
    """

    name = "open_application"
    description = (
        "Open a local file or HTTP/HTTPS URL with the OS default application "
        "or a specified program."
    )
    parameters_schema = {
        "target": {
            "type": "string",
            "description": "Local file path or http/https URL to open.",
            "required": True,
        },
        "application": {
            "type": "string",
            "description": "Specific application executable to use instead of the system default.",
            "required": False,
            "default": None,
        },
        "timeout": {
            "type": "int",
            "description": "Wait timeout in seconds (only used when wait=True).  Defaults to 10.",
            "required": False,
            "default": 10,
        },
        "wait": {
            "type": "bool",
            "description": "Block until the launched process exits when True.  Defaults to False.",
            "required": False,
            "default": False,
        },
    }

    def run(  # type: ignore[override]
        self,
        target: str,
        application: Optional[str] = None,
        timeout: int = 10,
        wait: bool = False,
        **_: Any,
    ) -> ToolResult:
        """Open *target* with the appropriate system handler.

        Args:
            target: File path or URL.
            application: Optional explicit application.
            timeout: Wait timeout.
            wait: Whether to block.

        Returns:
            ToolResult indicating launch success.
        """
        target = target.strip()
        timeout = max(1, int(timeout))

        # --- validate target ---
        is_url = _is_url(target)
        if not is_url:
            # Treat as filesystem path
            resolved = Path(target).expanduser().resolve()
            if not resolved.exists():
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    error=f"Target path does not exist: {resolved}",
                )
            target_str = str(resolved)
        else:
            # Reject non-http/https schemes (already filtered by _is_url)
            target_str = target

        system = platform.system()

        try:
            if application:
                proc = subprocess.Popen([application, target_str])
                if wait:
                    proc.wait(timeout=timeout)
                pid = proc.pid
            elif system == "Windows":
                os.startfile(target_str)  # type: ignore[attr-defined]
                pid = None
            elif system == "Darwin":
                proc = subprocess.Popen(["open", target_str])
                if wait:
                    proc.wait(timeout=timeout)
                pid = proc.pid
            else:
                # Linux / other POSIX
                proc = subprocess.Popen(["xdg-open", target_str])
                if wait:
                    proc.wait(timeout=timeout)
                pid = proc.pid
        except FileNotFoundError as exc:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"Application not found: {exc}",
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"Process did not finish within {timeout}s.",
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=str(exc),
            )

        return ToolResult(
            tool_name=self.name,
            success=True,
            output=f"Opened: {target_str}",
            metadata={
                "target": target_str,
                "application": application,
                "pid": pid,
                "system": system,
            },
        )
