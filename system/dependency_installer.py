"""dependency_installer.py — Python package dependency installation helpers."""
from __future__ import annotations
import subprocess
import sys
from typing import List


def install_packages(packages: List[str], quiet: bool = True) -> bool:
    """Install *packages* via pip into the current Python environment."""
    args = [sys.executable, "-m", "pip", "install"] + packages
    if quiet:
        args.append("-q")
    result = subprocess.run(args, capture_output=True, text=True)
    return result.returncode == 0


def is_installed(package: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(package) is not None


def ensure_packages(packages: List[str]) -> List[str]:
    """Install any packages that are not yet present. Returns list of missing."""
    missing = [p for p in packages if not is_installed(p)]
    if missing:
        install_packages(missing)
    return [p for p in missing if not is_installed(p)]  # still missing after install
