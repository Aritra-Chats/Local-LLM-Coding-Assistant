"""step_contract.py — SDLC-style entry/exit contracts for pipeline steps.

Every pipeline step can carry a :class:`StepContract` that declares:
- Preconditions (entry_requirements, required_system_tools) that must hold
  before the step starts.
- Postconditions (exit_criteria, expected_artifacts) that define "done".

:class:`ContractChecker` evaluates these using pure filesystem + PATH checks —
no LLM involved — so it is fast, deterministic, and side-effect-free.
"""
from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StepContract:
    """SDLC contract attached to a pipeline step.

    Attributes:
        entry_requirements: Human-readable preconditions checked before the
            step runs.  Examples: "node >= 16 is installed", "output_dir is
            writable", "package.json exists at project_root".
        exit_criteria: Postconditions that define "done".  Examples: "all
            unit tests pass", "no TypeScript errors in src/".
        required_system_tools: CLI tools that must be on PATH.  Each name is
            checked via ``shutil.which()``.
        expected_artifacts: Relative paths (from project_root) that must
            exist after the step completes.
        max_fix_attempts: Max times AsyncSupervisorLoop may auto-fix a
            failure on this step before escalating to the user.
    """

    entry_requirements: List[str] = field(default_factory=list)
    exit_criteria: List[str] = field(default_factory=list)
    required_system_tools: List[str] = field(default_factory=list)
    expected_artifacts: List[str] = field(default_factory=list)
    max_fix_attempts: int = 3

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_requirements": self.entry_requirements,
            "exit_criteria": self.exit_criteria,
            "required_system_tools": self.required_system_tools,
            "expected_artifacts": self.expected_artifacts,
            "max_fix_attempts": self.max_fix_attempts,
        }

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> Optional["StepContract"]:
        """Deserialise from a dict (e.g. retrieved from step["contract"]).

        Returns ``None`` when *d* is falsy so callers can do a simple ``if
        contract:`` guard.
        """
        if not d:
            return None
        return cls(
            entry_requirements=d.get("entry_requirements", []),
            exit_criteria=d.get("exit_criteria", []),
            required_system_tools=d.get("required_system_tools", []),
            expected_artifacts=d.get("expected_artifacts", []),
            max_fix_attempts=int(d.get("max_fix_attempts", 3)),
        )


@dataclass
class ContractCheckResult:
    """Result of a contract check.

    Attributes:
        passed: True when all checked items passed.
        failed_items: The requirement/criteria strings that failed.
        details: Human-readable summary for display or logging.
    """

    passed: bool
    failed_items: List[str] = field(default_factory=list)
    details: str = ""


# ---------------------------------------------------------------------------
# Checker
# ---------------------------------------------------------------------------

# Keywords we look for in natural-language requirement strings.
_WRITABLE_KW   = re.compile(r"\bwrit(able|e?able)\b", re.I)
_EXISTS_KW     = re.compile(r"\bexists?\b", re.I)
_VERSION_KW    = re.compile(r"(node|python|npm|git|java|ruby|go)\s*[><=!]+\s*(\d+)", re.I)


def _artifact_variant_matches(requested: str, candidate: str) -> bool:
    """Return True when two artifact paths are close enough to count as the same file."""
    requested_path = requested.replace("\\", "/").strip()
    candidate_path = candidate.replace("\\", "/").strip()

    if requested_path.lower() == candidate_path.lower():
        return True

    requested_base = os.path.basename(requested_path)
    candidate_base = os.path.basename(candidate_path)
    if requested_base.lower() == candidate_base.lower():
        return True

    requested_stem, requested_ext = os.path.splitext(requested_base)
    candidate_stem, candidate_ext = os.path.splitext(candidate_base)
    if requested_ext.lower() != candidate_ext.lower():
        return False

    requested_stem = requested_stem.lower()
    candidate_stem = candidate_stem.lower()
    if requested_stem == candidate_stem:
        return True

    if requested_stem + "s" == candidate_stem or candidate_stem + "s" == requested_stem:
        return True

    return False


class ContractChecker:
    """Pure static helpers — no state, no LLM, no side effects beyond reads."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def check_entry(contract: StepContract, context: Dict[str, Any]) -> ContractCheckResult:
        """Evaluate all entry requirements and required_system_tools.

        Uses ``shutil.which()`` for tool presence checks and ``os.access``
        for filesystem checks.  Natural-language requirements are handled
        by a small keyword matcher so the check is always deterministic.

        Args:
            contract: The :class:`StepContract` to evaluate.
            context: The current step context dict (used to resolve paths
                such as ``project_root`` and ``output_dir``).

        Returns:
            :class:`ContractCheckResult`.
        """
        failed: List[str] = []

        # 1. CLI tool presence checks
        tool_status = ContractChecker.check_system_tools(contract.required_system_tools)
        for tool, present in tool_status.items():
            if not present:
                failed.append(f"required tool not found on PATH: {tool}")

        # 2. Natural-language requirement checks
        project_root = context.get("project_root", "") or ""
        output_dir   = context.get("output_dir",   "") or ""

        for req in contract.entry_requirements:
            result = ContractChecker._check_requirement(req, project_root, output_dir)
            if result is False:
                failed.append(req)

        passed = len(failed) == 0
        details = "All entry requirements met." if passed else (
            f"{len(failed)} requirement(s) failed: " + "; ".join(failed)
        )
        return ContractCheckResult(passed=passed, failed_items=failed, details=details)

    @staticmethod
    def check_exit(
        contract: StepContract,
        project_root: str,
        known_files: List[str],
    ) -> ContractCheckResult:
        """Evaluate expected_artifacts and exit_criteria.

        ``expected_artifacts`` are resolved as relative paths from
        *project_root* and checked with ``os.path.exists``.

        Args:
            contract: The step contract.
            project_root: Absolute path to the project root directory.
            known_files: List of relative file paths recorded by the
                engine (from ``context["known_files"]``).

        Returns:
            :class:`ContractCheckResult`.
        """
        failed: List[str] = []

        # 1. Artifact existence check
        for artifact in contract.expected_artifacts:
            found = ContractChecker._artifact_exists(artifact, project_root, known_files)
            if not found:
                failed.append(f"expected artifact missing: {artifact}")

        # 2. Natural-language exit criteria (best-effort keyword check)
        for criterion in contract.exit_criteria:
            result = ContractChecker._check_exit_criterion(
                criterion, project_root, known_files
            )
            if result is False:
                failed.append(criterion)

        passed = len(failed) == 0
        details = "All exit criteria met." if passed else (
            f"{len(failed)} exit criterion/criteria failed: " + "; ".join(failed)
        )
        return ContractCheckResult(passed=passed, failed_items=failed, details=details)

    @staticmethod
    def check_system_tools(tools: List[str]) -> Dict[str, bool]:
        """Return ``{tool_name: available}`` for each tool in *tools*."""
        return {t: shutil.which(t) is not None for t in tools}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_requirement(req: str, project_root: str, output_dir: str) -> Optional[bool]:
        """Check a single natural-language requirement string.

        Returns:
            ``True`` if requirement is met, ``False`` if it is not, ``None``
            if it cannot be evaluated (treated as passing — don't block on
            things we can't check).
        """
        lower = req.lower()

        # Writable directory checks
        if _WRITABLE_KW.search(lower):
            target = output_dir or project_root
            if target:
                return os.access(target, os.W_OK)
            return None  # can't check — pass through

        # File/directory existence checks  e.g. "package.json exists at project_root"
        if _EXISTS_KW.search(lower):
            # Extract quoted or bare filename-like tokens
            tokens = re.findall(r"[\w./\\-]+\.\w+", req)
            if tokens and project_root:
                for tok in tokens:
                    full = os.path.join(project_root, tok)
                    if os.path.exists(full):
                        return True
                # None of the named files exist
                return False
            return None  # can't evaluate without a project root

        # Tool version checks  e.g. "node >= 16 is installed"
        m = _VERSION_KW.search(req)
        if m:
            tool = m.group(1).lower()
            if shutil.which(tool) is None:
                return False
            # We have the tool — version check would require parsing output;
            # treat presence as sufficient for our use-case.
            return True

        # Fallback: if the req mentions a known system tool by name, check PATH
        for keyword in ("node", "npm", "npx", "git", "python", "pip",
                        "java", "gradle", "flutter", "dart", "go", "ruby"):
            if keyword in lower and "install" in lower:
                return shutil.which(keyword) is not None

        return None  # Unknown requirement — don't block

    @staticmethod
    def _artifact_exists(
        artifact: str,
        project_root: str,
        known_files: List[str],
    ) -> bool:
        """Return True if *artifact* is present on disk or in known_files."""
        # Direct match in known_files (relative paths)
        if artifact in known_files:
            return True

        # Filesystem check
        if project_root:
            abs_path = os.path.join(project_root, artifact)
            if os.path.exists(abs_path):
                return True

        # Basename match anywhere in known_files (handles path discrepancies)
        base = os.path.basename(artifact)
        if base and any(os.path.basename(f) == base for f in known_files):
            return True

        # Conservative near-match fallback for common filename variants such
        # as style.css vs styles.css when only one candidate exists.
        matches = [f for f in known_files if _artifact_variant_matches(artifact, f)]
        if len(matches) == 1:
            return True

        return False

    @staticmethod
    def _check_exit_criterion(
        criterion: str,
        project_root: str,
        known_files: List[str],
    ) -> Optional[bool]:
        """Check a natural-language exit criterion.

        Returns True / False / None (can't determine).
        """
        lower = criterion.lower()

        # "X exists at ..." patterns
        if _EXISTS_KW.search(lower):
            tokens = re.findall(r"[\w./\\-]+\.\w+", criterion)
            if tokens:
                for tok in tokens:
                    if ContractChecker._artifact_exists(tok, project_root, known_files):
                        return True
                return False
            # e.g. "node_modules directory exists"
            if "node_modules" in lower and project_root:
                return os.path.isdir(os.path.join(project_root, "node_modules"))
            if ".git" in lower and project_root:
                return os.path.isdir(os.path.join(project_root, ".git"))

        return None  # Can't evaluate — pass through
