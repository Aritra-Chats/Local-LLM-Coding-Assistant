"""system_check.py — Sentinel cross-platform hardware detection.

Detects CPU, RAM, and GPU capabilities on Windows, Linux, and macOS.
Returns a structured SystemInfo dataclass used by HardwareProfile to
assign the appropriate runtime mode.
"""

import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class GPUInfo:
    """Information about a single detected GPU.

    Attributes:
        name: GPU model name string.
        vram_gb: Total VRAM in gigabytes, or None if unknown.
        driver: Driver version string, or None if unavailable.
        backend: Detection backend used ('nvidia-smi', 'rocm', 'metal', 'wmic').
    """

    name: str
    vram_gb: Optional[float]
    driver: Optional[str]
    backend: str


@dataclass
class SystemInfo:
    """Full hardware snapshot of the current machine.

    Attributes:
        os_name: Operating system name ('Windows', 'Linux', 'Darwin').
        os_version: OS version string.
        cpu_name: Processor model name.
        cpu_cores: Physical CPU core count.
        cpu_threads: Logical CPU thread count.
        ram_gb: Total installed RAM in gigabytes.
        ram_available_gb: Currently available RAM in gigabytes.
        gpus: List of detected GPU devices.
        has_cuda: True if an NVIDIA CUDA-capable GPU is present.
        has_rocm: True if an AMD ROCm-capable GPU is detected.
        has_metal: True if Apple Metal is available (macOS only).
        ollama_installed: True if the `ollama` binary is on PATH.
        git_installed: True if the `git` binary is on PATH.
        python_version: Running Python version string.
    """

    os_name: str
    os_version: str
    cpu_name: str
    cpu_cores: int
    cpu_threads: int
    ram_gb: float
    ram_available_gb: float
    gpus: List[GPUInfo] = field(default_factory=list)
    has_cuda: bool = False
    has_rocm: bool = False
    has_metal: bool = False
    ollama_installed: bool = False
    git_installed: bool = False
    python_version: str = ""

    @property
    def total_vram_gb(self) -> float:
        """Sum of VRAM across all detected GPUs in gigabytes."""
        return sum(g.vram_gb for g in self.gpus if g.vram_gb is not None)

    @property
    def has_gpu(self) -> bool:
        """True if at least one GPU was detected."""
        return bool(self.gpus)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class SystemCheck:
    """Cross-platform hardware detector.

    Call run() to obtain a populated SystemInfo instance.
    """

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> SystemInfo:
        """Run all hardware detection routines and return a SystemInfo.

        Returns:
            A fully populated SystemInfo dataclass.
        """
        import psutil  # imported here so the module is importable without psutil

        os_name = platform.system()  # 'Windows', 'Linux', 'Darwin'
        os_version = platform.version()
        python_version = platform.python_version()

        cpu_name = self._get_cpu_name()
        cpu_cores = psutil.cpu_count(logical=False) or 1
        cpu_threads = psutil.cpu_count(logical=True) or cpu_cores

        mem = psutil.virtual_memory()
        ram_gb = round(mem.total / (1024 ** 3), 2)
        ram_available_gb = round(mem.available / (1024 ** 3), 2)

        gpus = self._detect_gpus(os_name)
        has_cuda = any(g.backend == "nvidia-smi" for g in gpus)
        has_rocm = any(g.backend == "rocm" for g in gpus)
        has_metal = os_name == "Darwin" and self._check_metal()

        ollama_installed = shutil.which("ollama") is not None
        git_installed = shutil.which("git") is not None

        return SystemInfo(
            os_name=os_name,
            os_version=os_version,
            cpu_name=cpu_name,
            cpu_cores=cpu_cores,
            cpu_threads=cpu_threads,
            ram_gb=ram_gb,
            ram_available_gb=ram_available_gb,
            gpus=gpus,
            has_cuda=has_cuda,
            has_rocm=has_rocm,
            has_metal=has_metal,
            ollama_installed=ollama_installed,
            git_installed=git_installed,
            python_version=python_version,
        )

    # ------------------------------------------------------------------
    # CPU
    # ------------------------------------------------------------------

    def _get_cpu_name(self) -> str:
        """Retrieve the CPU model name in a cross-platform manner.

        Returns:
            CPU model name string, or 'Unknown CPU' if not detectable.
        """
        os_name = platform.system()

        if os_name == "Windows":
            return self._run_cmd(
                ["wmic", "cpu", "get", "name", "/value"],
                parse_key="Name",
            ) or platform.processor()

        if os_name == "Linux":
            try:
                with open("/proc/cpuinfo", encoding="utf-8") as f:
                    for line in f:
                        if line.lower().startswith("model name"):
                            return line.split(":", 1)[1].strip()
            except OSError:
                pass
            return platform.processor() or "Unknown CPU"

        if os_name == "Darwin":
            result = self._run_raw(["sysctl", "-n", "machdep.cpu.brand_string"])
            return result.strip() if result else platform.processor() or "Unknown CPU"

        return platform.processor() or "Unknown CPU"

    # ------------------------------------------------------------------
    # GPU
    # ------------------------------------------------------------------

    def _detect_gpus(self, os_name: str) -> List[GPUInfo]:
        """Detect GPUs using platform-appropriate methods.

        Tries multiple backends in order of preference.

        Args:
            os_name: The platform name ('Windows', 'Linux', 'Darwin').

        Returns:
            List of GPUInfo instances for each detected GPU.
        """
        gpus: List[GPUInfo] = []

        # NVIDIA — nvidia-smi (Windows + Linux)
        gpus.extend(self._detect_nvidia())

        # AMD ROCm (Linux)
        if os_name == "Linux" and not gpus:
            gpus.extend(self._detect_rocm())

        # Apple Metal / integrated (macOS)
        if os_name == "Darwin" and not gpus:
            gpus.extend(self._detect_metal())

        # Windows fallback via WMIC if nvidia-smi missed it
        if os_name == "Windows" and not gpus:
            gpus.extend(self._detect_wmic())

        return gpus

    def _detect_nvidia(self) -> List[GPUInfo]:
        """Query nvidia-smi for NVIDIA GPU details.

        Returns:
            List of GPUInfo from nvidia-smi, empty if not available.
        """
        if not shutil.which("nvidia-smi"):
            return []

        output = self._run_raw(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ]
        )
        if not output:
            return []

        gpus: List[GPUInfo] = []
        for line in output.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            name, vram_mib_str, driver = parts[0], parts[1], parts[2]
            try:
                vram_gb = round(int(vram_mib_str) / 1024, 2)
            except ValueError:
                vram_gb = None
            gpus.append(
                GPUInfo(name=name, vram_gb=vram_gb, driver=driver, backend="nvidia-smi")
            )
        return gpus

    def _detect_rocm(self) -> List[GPUInfo]:
        """Query rocm-smi for AMD ROCm GPU details.

        Returns:
            List of GPUInfo from rocm-smi, empty if not available.
        """
        if not shutil.which("rocm-smi"):
            return []

        output = self._run_raw(["rocm-smi", "--showproductname", "--csv"])
        if not output:
            return []

        gpus: List[GPUInfo] = []
        lines = output.strip().splitlines()
        for line in lines[1:]:  # skip header
            parts = [p.strip() for p in line.split(",")]
            name = parts[-1] if parts else "AMD GPU"
            gpus.append(GPUInfo(name=name, vram_gb=None, driver=None, backend="rocm"))
        return gpus

    def _detect_metal(self) -> List[GPUInfo]:
        """Detect Apple Metal GPU via system_profiler.

        Returns:
            List of GPUInfo for Apple Silicon / discrete GPU, empty if not found.
        """
        output = self._run_raw(
            ["system_profiler", "SPDisplaysDataType", "-detailLevel", "mini"]
        )
        if not output:
            return []

        gpus: List[GPUInfo] = []
        current_name: Optional[str] = None
        current_vram: Optional[float] = None

        for line in output.splitlines():
            stripped = line.strip()
            if stripped.startswith("Chipset Model:"):
                current_name = stripped.split(":", 1)[1].strip()
                current_vram = None
            elif stripped.startswith("VRAM") and ":" in stripped:
                raw = stripped.split(":", 1)[1].strip()
                current_vram = self._parse_vram(raw)
            elif stripped == "" and current_name:
                gpus.append(
                    GPUInfo(
                        name=current_name,
                        vram_gb=current_vram,
                        driver=None,
                        backend="metal",
                    )
                )
                current_name = None
                current_vram = None

        if current_name:
            gpus.append(
                GPUInfo(name=current_name, vram_gb=current_vram, driver=None, backend="metal")
            )

        return gpus

    def _detect_wmic(self) -> List[GPUInfo]:
        """Detect GPUs on Windows via WMIC as a fallback.

        Returns:
            List of GPUInfo from WMIC, empty if not available.
        """
        output = self._run_cmd(["wmic", "path", "win32_videocontroller", "get", "name", "/value"])
        if not output:
            return []

        gpus: List[GPUInfo] = []
        for line in output.strip().splitlines():
            if "=" in line:
                name = line.split("=", 1)[1].strip()
                if name:
                    gpus.append(
                        GPUInfo(name=name, vram_gb=None, driver=None, backend="wmic")
                    )
        return gpus

    def _check_metal(self) -> bool:
        """Check if Apple Metal is available on this macOS system.

        Returns:
            True if Metal is likely available (macOS 10.11+).
        """
        try:
            major = int(platform.mac_ver()[0].split(".")[0])
            return major >= 11 or (major == 10 and int(platform.mac_ver()[0].split(".")[1]) >= 11)
        except (ValueError, IndexError):
            return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _run_raw(cmd: List[str]) -> str:
        """Run a subprocess command and return stdout as a string.

        Args:
            cmd: Command list to execute.

        Returns:
            stdout string, or empty string on failure.
        """
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.stdout or ""
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return ""

    @staticmethod
    def _run_cmd(cmd: List[str], parse_key: Optional[str] = None) -> str:
        """Run a subprocess command and optionally extract a key=value pair.

        Args:
            cmd: Command list to execute.
            parse_key: If provided, find and return the value for this key
                       from key=value output lines.

        Returns:
            Extracted value string or raw stdout, empty string on failure.
        """
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
            )
            output = result.stdout or ""
            if parse_key:
                for line in output.splitlines():
                    if line.strip().lower().startswith(parse_key.lower() + "="):
                        return line.split("=", 1)[1].strip()
                return ""
            return output
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return ""

    @staticmethod
    def _parse_vram(raw: str) -> Optional[float]:
        """Parse a VRAM string like '8 GB' or '4096 MB' into gigabytes.

        Args:
            raw: Raw VRAM string from system output.

        Returns:
            VRAM in gigabytes as float, or None if not parseable.
        """
        raw = raw.strip().upper()
        try:
            if "GB" in raw:
                return round(float(raw.replace("GB", "").strip()), 2)
            if "MB" in raw:
                return round(float(raw.replace("MB", "").strip()) / 1024, 2)
        except ValueError:
            pass
        return None
