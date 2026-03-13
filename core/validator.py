"""validator.py — Pipeline and step output validation utilities.

Provides two levels of validation:

Structural validation
    Quick key-presence checks to ensure dicts have the required shape.
    These run on every step result and pipeline before execution starts.

Semantic validation
    Deeper checks on the *content* of LLM-generated outputs:
    - Python syntax checking for ``write_file`` payloads
    - JSON well-formedness checks for JSON file writes
    - Parameter schema validation for tool calls
    - Basic safety check: blocks write_file paths outside the project root

All functions return ``ValidationResult`` namedtuples so callers can
inspect both pass/fail status and a human-readable reason without
catching exceptions.
"""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


class ValidationResult(NamedTuple):
    """Result of a validation check.

    Attributes:
        ok:     True if validation passed.
        reason: Human-readable explanation.  Empty string when ok=True.
    """
    ok: bool
    reason: str = ""

    @classmethod
    def passed(cls) -> "ValidationResult":
        return cls(ok=True, reason="")

    @classmethod
    def failed(cls, reason: str) -> "ValidationResult":
        return cls(ok=False, reason=reason)


# ---------------------------------------------------------------------------
# Structural validation (original API preserved)
# ---------------------------------------------------------------------------


def validate_step_output(output: Dict[str, Any]) -> bool:
    """Structural check: output dict must have a 'status' key."""
    return isinstance(output, dict) and "status" in output


def validate_pipeline(steps: List[Dict[str, Any]]) -> bool:
    """Structural check: pipeline list must be non-empty and well-formed.

    Each step must have at minimum 'name' and 'agent' keys.
    """
    if not steps:
        return False
    required = {"name", "agent"}
    return all(required.issubset(s.keys()) for s in steps)


# ---------------------------------------------------------------------------
# Semantic validation
# ---------------------------------------------------------------------------


def validate_tool_call(
    tool_name: str,
    params: Dict[str, Any],
    project_root: Optional[str] = None,
) -> ValidationResult:
    """Validate a single tool call before it is dispatched.

    Checks performed (varies by tool):

    - ``write_file``:  path present; if project_root given, path must be
      inside it; content present; if ``.py`` → valid Python syntax; if
      ``.json`` → valid JSON.
    - ``run_shell``:   command present and non-empty.
    - ``read_file`` / ``search_code``:  path present.
    - All other tools: params dict presence only.

    Args:
        tool_name:    Tool identifier string.
        params:       Tool parameter dict.
        project_root: Optional absolute path for path-scoping.

    Returns:
        :class:`ValidationResult`.
    """
    if not isinstance(params, dict):
        return ValidationResult.failed(
            f"Tool '{tool_name}': params must be a dict, got {type(params).__name__}"
        )

    if tool_name == "write_file":
        return _validate_write_file(params, project_root)

    if tool_name == "run_shell":
        cmd = params.get("command", "")
        if not cmd or not str(cmd).strip():
            return ValidationResult.failed("run_shell: 'command' param is missing or empty")
        return _validate_shell_command(str(cmd))

    if tool_name in ("read_file", "search_code"):
        path = params.get("path", "")
        if not path:
            return ValidationResult.failed(f"{tool_name}: 'path' param is missing or empty")
        return ValidationResult.passed()

    return ValidationResult.passed()


def validate_agent_actions(
    actions: List[Any],
    project_root: Optional[str] = None,
) -> List[ValidationResult]:
    """Validate a list of AgentAction objects before dispatch.

    Runs :func:`validate_tool_call` on ``tool_call`` actions.
    Non-tool actions (message, delegate, abort) are always passed through.

    Args:
        actions:      List of AgentAction objects or dicts.
        project_root: Optional absolute path for path-scoping.

    Returns:
        List of :class:`ValidationResult`, one per action.
    """
    results: List[ValidationResult] = []
    for action in actions:
        if hasattr(action, "action_type"):
            action_type = action.action_type
            payload = getattr(action, "payload", {}) or {}
        elif isinstance(action, dict):
            action_type = action.get("action_type", "")
            payload = action.get("payload", {}) or {}
        else:
            results.append(ValidationResult.passed())
            continue

        if action_type == "tool_call":
            tool = payload.get("tool", "")
            params = payload.get("params", {}) or {}
            results.append(validate_tool_call(tool, params, project_root))
        else:
            results.append(ValidationResult.passed())

    return results


def validate_write_file_content(path: str, content: str) -> ValidationResult:
    """Validate the content that would be written to *path*.

    Language-aware checks:

    - ``*.py``  → Python AST parse (catches syntax errors)
    - ``*.json`` → JSON decode (catches malformed JSON)
    - others    → content presence only

    Args:
        path:    Target file path (suffix used to detect language).
        content: File content string.

    Returns:
        :class:`ValidationResult`.
    """
    suffix = Path(path).suffix.lower()

    if suffix == ".py":
        try:
            ast.parse(content)
        except SyntaxError as exc:
            return ValidationResult.failed(
                f"write_file '{path}': Python syntax error at line {exc.lineno} — {exc.msg}"
            )

    elif suffix == ".json":
        try:
            json.loads(content)
        except json.JSONDecodeError as exc:
            return ValidationResult.failed(
                f"write_file '{path}': invalid JSON — {exc.msg} (line {exc.lineno})"
            )

    return ValidationResult.passed()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_write_file(
    params: Dict[str, Any],
    project_root: Optional[str],
) -> ValidationResult:
    path = params.get("path", "")
    content = params.get("content")

    if not path:
        return ValidationResult.failed("write_file: 'path' param is missing or empty")

    if content is None:
        return ValidationResult.failed(f"write_file '{path}': 'content' param is missing")

    # Path-scoping safety check
    if project_root:
        try:
            resolved = (Path(project_root).resolve() / path).resolve()
            resolved.relative_to(Path(project_root).resolve())
        except ValueError:
            return ValidationResult.failed(
                f"write_file '{path}': path escapes project root — rejected for safety"
            )

    return validate_write_file_content(path, str(content))


# ---------------------------------------------------------------------------
# Shell command safety filter
# ---------------------------------------------------------------------------

import re as _re
import shlex as _shlex

# Patterns for dangerous commands.  Each entry is a (pattern, reason) pair.
# Patterns are matched against the full command string (case-insensitive).
_DANGEROUS_PATTERNS: List[tuple] = [
    # Destructive file system operations
    (_re.compile(r"\brm\s+(-\w*\s+)*-[rRfF]*r[rRfF]*\b"), "rm -rf is not permitted"),
    (_re.compile(r"\brm\s+(-\w*\s+)*-[rRfF]*f[rRfF]*\b"), "rm -f on recursive delete is not permitted"),
    (_re.compile(r"\brmdir\s+/"), "rmdir on root path is not permitted"),
    # System shutdown / reboot
    (_re.compile(r"\bshutdown\b"), "shutdown command is not permitted"),
    (_re.compile(r"\breboot\b"), "reboot command is not permitted"),
    (_re.compile(r"\binit\s+[06]\b"), "init 0/6 (halt/reboot) is not permitted"),
    (_re.compile(r"\bpoweroff\b"), "poweroff command is not permitted"),
    (_re.compile(r"\bhalt\b"), "halt command is not permitted"),
    # Disk formatting
    (_re.compile(r"\bmkfs\b"), "mkfs (format) command is not permitted"),
    (_re.compile(r"\bformat\s+[a-zA-Z]:"), "Windows format command is not permitted"),
    (_re.compile(r"\bdd\b.*\bof=/dev/"), "dd to a block device is not permitted"),
    # Fork bombs and infinite loops that could exhaust resources
    (_re.compile(r":\(\)\{.*:\|:&"), "fork bomb pattern is not permitted"),
    # Writing to /etc /sys /proc
    (_re.compile(r"\bchmod\s+777\s+/"), "chmod 777 on root path is not permitted"),
    (_re.compile(r"\bchown\s+.*\s+/\s*$"), "chown on root path is not permitted"),
    # Privilege escalation
    (_re.compile(r"\bsudo\s+rm\b"), "sudo rm is not permitted"),
    (_re.compile(r"\bsudo\s+shutdown\b"), "sudo shutdown is not permitted"),
    (_re.compile(r"\bsudo\s+reboot\b"), "sudo reboot is not permitted"),
    (_re.compile(r"\bsudo\s+mkfs\b"), "sudo mkfs is not permitted"),
    # Windows equivalents
    (_re.compile(r"\brd\s+/s\s+/q\b", _re.IGNORECASE), "rd /s /q is not permitted"),
    (_re.compile(r"\bdel\s+/[fF]\s+/[sS]", _re.IGNORECASE), "del /f /s is not permitted"),
]

# Exact command names that are always blocked regardless of arguments
_BLOCKED_COMMANDS: frozenset = frozenset({
    "shutdown", "reboot", "poweroff", "halt", "mkfs",
    "fdisk", "parted", "gdisk", "wipefs",
})


def _validate_shell_command(command: str) -> ValidationResult:
    """Check a shell command string against the safety block-list.

    Args:
        command: Raw shell command string.

    Returns:
        :class:`ValidationResult` — failed if the command matches a
        dangerous pattern, passed otherwise.
    """
    stripped = command.strip()

    # Check dangerous regex patterns
    for pattern, reason in _DANGEROUS_PATTERNS:
        if pattern.search(stripped):
            return ValidationResult.failed(f"run_shell blocked: {reason}. Command: {stripped!r}")

    # Check blocked command names (first token of the command)
    try:
        tokens = _shlex.split(stripped)
        if tokens:
            # Extract the base name (strip path)
            base = tokens[0].split("/")[-1].split("\\")[-1].lower()
            if base in _BLOCKED_COMMANDS:
                return ValidationResult.failed(
                    f"run_shell blocked: '{base}' command is not permitted. "
                    f"Command: {stripped!r}"
                )
    except ValueError:
        pass  # shlex parsing failed on complex quoting; skip name check

    return ValidationResult.passed()
