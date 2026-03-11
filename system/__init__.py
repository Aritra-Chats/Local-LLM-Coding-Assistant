"""system — Hardware detection, dependency installation, and Ollama management."""
from system.hardware_detector import SystemCheck, SystemInfo, GPUInfo
from system.dependency_installer import install_packages, is_installed, ensure_packages
from system.ollama_manager import (
    is_ollama_installed, pull_model, list_local_models, is_model_available,
)

__all__ = [
    "SystemCheck", "SystemInfo", "GPUInfo",
    "install_packages", "is_installed", "ensure_packages",
    "is_ollama_installed", "pull_model", "list_local_models", "is_model_available",
]
