# Installation Guide

This guide covers all required steps to get Sentinel running on Windows, Linux, and macOS.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Standard Installation](#2-standard-installation)
3. [Windows Quick Start](#3-windows-quick-start)
4. [Linux / macOS Quick Start](#4-linux--macos-quick-start)
5. [Ollama Setup](#5-ollama-setup)
6. [Model Selection](#6-model-selection)
7. [Environment Variables](#7-environment-variables)
8. [Offline Installation](#8-offline-installation)
9. [GPU Setup](#9-gpu-setup)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| **Python** | 3.11 or later | 3.12 / 3.13 also supported |
| **Ollama** | Latest | [ollama.ai/download](https://ollama.ai/download) |
| **RAM** | ≥ 8 GB | 16 GB+ recommended for a good experience |
| **Disk** | ≥ 10 GB free | Model storage: 7B ≈ 4 GB, 13B ≈ 8 GB, 34B ≈ 20 GB |
| **Git** | Any recent | Optional; required for git-related tools |

### Python version check

```bash
python --version
# Should print: Python 3.11.x or later
```

---

## 2. Standard Installation

```bash
# 1. Clone
git clone https://github.com/Aritra-Chats/Local-LLM-Coding-Assistant.git
cd Local-LLM-Coding-Assistant

# 2. Add the Sentinel folder to your PATH
#    Windows example: setx PATH "%PATH%;C:\tools\local-llm-assistant"

# 3. Open a new PowerShell window and launch Sentinel
sentinel --project C:\work\my-project

# 4. Copy environment template (optional, only if you want to override defaults)
cp .env.example .env           # Linux / macOS
copy .env.example .env         # Windows
```

On first launch Sentinel will automatically:
- Detect your hardware and select a hardware mode
- Create a local Python virtual environment under the Sentinel folder
- Install any missing Python packages into that venv
- Install Ollama with winget if it is missing
- Prompt you to choose ONLINE or OFFLINE mode with arrow-key navigation
- Pull only the Ollama models required for the chosen startup mode
- Create `~/.sentinel/` workspace directories

---

## 3. Windows Quick Start

```powershell
# Clone
git clone https://github.com/Aritra-Chats/Local-LLM-Coding-Assistant.git
cd Local-LLM-Coding-Assistant

# Add the folder to PATH, then open a new PowerShell window

# Launch
sentinel --project C:\work\my-project

# Fallback if you want to run the repo directly from its folder
python main.py
```

---

## 4. Linux / macOS Quick Start

```bash
git clone https://github.com/Aritra-Chats/Local-LLM-Coding-Assistant.git
cd Local-LLM-Coding-Assistant

sentinel
```

Make the `sentinel` script executable and add it to your PATH (one-time):

```bash
chmod +x sentinel
# Option A: symlink into a PATH directory (recommended)
sudo ln -s "$(pwd)/sentinel" /usr/local/bin/sentinel

# Option B: add the repository folder to your PATH in your shell profile:
# echo 'export PATH="$PATH:$(pwd)"' >> ~/.bashrc && source ~/.bashrc
```

---

## 5. Use Sentinel From Another Project Repo

If your goal is to work on a separate project repository, keep Sentinel installed in its own folder and point it at the project you want to edit.

### One-time setup

```powershell
# 1. Download or clone Sentinel somewhere stable, for example:
cd C:\tools\local-llm-assistant

# 2. Add that folder to PATH, then open a new PowerShell window

# 3. Verify the command is available
Get-Command sentinel
```

### Start Sentinel against your project repo

```powershell
# Option A: pass the project repo each time
sentinel --project C:\work\my-project

# Option B: make one project the default for this shell session
$env:SENTINEL_PROJECT_DIR = "C:\work\my-project"
sentinel
```

If you want the command available in future PowerShell windows, keep the Sentinel folder on PATH. The batch file is what makes PATH-based launching work.

---

## 6. Ollama Setup

Sentinel requires [Ollama](https://ollama.ai) running locally.

### Install Ollama

**Windows:** Download and run the installer from [ollama.ai/download](https://ollama.ai/download).

**macOS:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### Start Ollama

Ollama runs as a background service after installation. To start it manually:

```bash
ollama serve
```

Verify it is running:

```bash
ollama list        # should print available models (empty on first install)
curl http://localhost:11434/api/tags   # should return {"models":[...]}
```

### Pull models manually (optional)

Sentinel's bootstrap step pulls models automatically. To pull them yourself:

```bash
# Minimal mode (8–12 GB RAM)
ollama pull codellama:7b
ollama pull mistral:7b
ollama pull nomic-embed-text

# Standard mode (12–20 GB RAM)
ollama pull codellama:13b
ollama pull mixtral:8x7b
ollama pull nomic-embed-text

# Advanced mode (≥ 20 GB RAM or GPU ≥ 6 GB VRAM)
ollama pull codellama:34b
ollama pull mixtral:8x7b
ollama pull nomic-embed-text
```

---

## 7. Model Selection

Sentinel auto-selects models based on your hardware. You can override:

On first launch, the operating-mode prompt uses keyboard navigation instead of numeric input.

### Override for the current session

```bash
sentinel --hw-mode minimal        # force minimal models
sentinel --hw-mode standard       # force standard models
sentinel --hw-mode advanced       # force advanced models
```

### Make mode override your default

There is currently no dedicated environment variable for hardware mode.
Use a shell alias or launcher script to always start with your preferred mode:

```bash
sentinel --hw-mode standard
```

### Alternative models

Any Ollama-compatible model works. Well-tested alternatives:

| Purpose | Alternatives |
|---|---|
| Code generation | `deepseek-coder:6.7b`, `qwen2.5-coder:7b`, `starcoder2:7b` |
| Reasoning | `llama3:8b`, `phi3:medium`, `gemma2:9b` |
| Large context | `llama3:70b` (requires ≥ 40 GB RAM or large GPU) |

---

## 8. Environment Variables

Copy `.env.example` to `.env` and adjust as needed:

```env
# Where Sentinel stores sessions, metrics, and index files
# Default: %USERPROFILE%\.sentinel  (Windows)
#          ~/.sentinel              (Linux / macOS)
SENTINEL_HOME=C:\Users\you\.sentinel

# Ollama API URL (change if Ollama runs on a different host/port)
OLLAMA_BASE_URL=http://localhost:11434

# Embedding model
SENTINEL_EMBEDDING_MODEL=nomic-embed-text

# Context window token budget (per pipeline step)
SENTINEL_TOKEN_BUDGET=3000

# Optional default project directory (used when --project is not passed)
SENTINEL_PROJECT_DIR=C:\code\my-project
```

---

## 9. Offline Installation

For air-gapped environments:

### Step 1 — Download models on a connected machine

```bash
ollama pull codellama:7b
ollama pull nomic-embed-text

# Locate the model files (default Ollama model directory)
# Linux / macOS: ~/.ollama/models/
# Windows:       %USERPROFILE%\.ollama\models\
```

Copy the entire `~/.ollama/models/` directory to the target machine.

### Step 2 — Pre-download Python packages

```bash
pip download -r requirements.txt -d ./packages
```

Copy `packages/` to the target machine, then install offline:

```bash
pip install --no-index --find-links=./packages -r requirements.txt
```

### Step 3 — Skip bootstrap network checks

```bash
sentinel --no-bootstrap
```

---

## 10. GPU Setup

### NVIDIA (CUDA)

Ensure CUDA drivers are installed. Ollama detects CUDA automatically. Verify:

```bash
nvidia-smi          # should list your GPU
ollama run codellama:7b "hello"    # should show VRAM usage
```

### AMD (ROCm) — Linux only

Install ROCm drivers from [rocm.docs.amd.com](https://rocm.docs.amd.com). Ollama includes ROCm support; verify with `ollama run`.

### Apple Silicon (Metal)

Ollama uses Metal automatically on Apple Silicon Macs. Sentinel will detect `has_metal = True` and set Advanced mode.

---

## 11. Troubleshooting

### `ollama: command not found`

Ollama is not on your PATH. Either install it or set:

```env
OLLAMA_BASE_URL=http://localhost:11434
```

and start Ollama manually with `ollama serve`.

### `ConnectionRefusedError` when connecting to Ollama

Ollama is not running. Start it:

```bash
ollama serve     # foreground, or
ollama start     # background (macOS)
```

### Models not found after bootstrap

Pull manually:

```bash
ollama pull codellama:7b
ollama pull nomic-embed-text
```

### `prompt_toolkit` or `rich` import errors

Ensure your virtualenv is activated:

```bash
# Windows
.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### PowerShell execution policy error

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Low RAM — Sentinel chooses the wrong mode

Force a specific mode:

```bash
sentinel --hw-mode minimal
```

### Session data location

All session files, metrics, and indexes are stored under:
- Windows: `%USERPROFILE%\.sentinel\`
- Linux / macOS: `~/.sentinel/`

Override with the `SENTINEL_HOME` environment variable.
