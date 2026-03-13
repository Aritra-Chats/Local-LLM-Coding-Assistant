# Usage Guide

Complete reference for running Sentinel, using the interactive REPL, and integrating it into your development workflow.

---

## Table of Contents

1. [Starting Sentinel](#1-starting-sentinel)
2. [Command-Line Arguments](#2-command-line-arguments)
3. [Interactive REPL](#3-interactive-repl)
4. [Slash Commands](#4-slash-commands)
5. [Task Prompts](#5-task-prompts)
6. [Session Management](#6-session-management)
7. [Attachment System](#7-attachment-system)
8. [Pipeline Inspection](#8-pipeline-inspection)
9. [Hardware Modes](#9-hardware-modes)
10. [Environment Variables](#10-environment-variables)
11. [Tips and Workflow Patterns](#11-tips-and-workflow-patterns)

---

## 1. Starting Sentinel

```bash
# From the project root, with the virtualenv active:
python main.py

# Windows launcher (handles venv activation automatically):
sentinel.bat
```

On first launch, Sentinel runs its bootstrap sequence (hardware detection, model pulling, workspace setup). Subsequent launches skip this and start in a few seconds.

---

## 2. Command-Line Arguments

```
python main.py [OPTIONS]
```

| Argument | Type | Description |
|---|---|---|
| `--resume SESSION_ID` | string | Resume a previous session by its ID |
| `--project PATH` | path | Set the project root Sentinel operates on (default: current directory) |
| `--mode MODE` | `minimal` \| `standard` \| `advanced` | Override the auto-detected hardware mode |
| `--no-bootstrap` | flag | Skip the bootstrap sequence (faster startup) |

### Examples

```bash
# New session on the current directory
python main.py

# Resume a previous session
python main.py --resume 5f63c8db-7f49-4fa4-a42f-6d53ec2d4b5f

# Assist a specific project
python main.py --project C:\code\my-api

# Force minimal mode on a constrained machine
python main.py --mode minimal

# Fast startup (bootstrap already done)
python main.py --no-bootstrap

# Combine flags
python main.py --project ./my-project --mode standard --no-bootstrap
```

---

## 3. Interactive REPL

Once Sentinel starts you will see the banner and the prompt:

```
sentinel ›
```

Type a **task prompt** or a **slash command**:

- **Task prompts** — plain English descriptions of what you want done
- **Slash commands** — start with `/` and control the session

Use **Tab** for command completion and **Up / Down** arrows to navigate input history.

Press **Ctrl+C** once to interrupt a running pipeline.  
Press **Ctrl+C** twice (or type `/exit`) to save the session and quit.

---

## 4. Slash Commands

| Command | Description |
|---|---|
| `/help` | Show all available commands with descriptions |
| `/status` | Show current session and system status |
| `/pipeline` | Display the last pipeline as a Rich table |
| `/models` | List available local Ollama models |
| `/session` | Show current session ID, project, start time, and turn count |
| `/resume <id>` | Load a saved session by ID |
| `/context` | Show what the context engine assembled for the last step |
| `/index` | Rebuild the project index for the current workspace |
| `/syscheck` | Run hardware and dependency checks |
| `/tasks` | List user tasks in the current session |
| `/mode <mode>` | Switch hardware mode (`minimal` / `standard` / `advanced`) |
| `/diff` | Render a syntax-highlighted diff of the last file edit |
| `/clear` | Clear the terminal |
| `/exit` | Save the current session and exit |

### `/pipeline` output example

```
┌──────────────────────────────────────────────────────────────────────────┐
│ Pipeline  7 steps · solo · standard · complexity: medium                 │
├───┬──────────────────────────┬──────────────┬───────────┬────────────────┤
│ # │ Step                     │ Agent        │ Status    │ Elapsed        │
├───┼──────────────────────────┼──────────────┼───────────┼────────────────┤
│ 1 │ Classify task            │ supervisor   │ ✔ done    │ 0.3 s          │
│ 2 │ Read existing auth code  │ coding       │ ✔ done    │ 0.1 s          │
│ 3 │ Generate JWT middleware  │ coding       │ ✔ done    │ 4.2 s          │
│ 4 │ Write middleware file    │ coding       │ ✔ done    │ 0.2 s          │
│ 5 │ Update route definitions │ coding       │ ✔ done    │ 3.1 s          │
│ 6 │ Write updated routes     │ coding       │ ✔ done    │ 0.2 s          │
│ 7 │ Run tests                │ debugging    │ ✔ done    │ 2.8 s          │
└───┴──────────────────────────┴──────────────┴───────────┴────────────────┘
```

---

## 5. Task Prompts

Type any natural-language development request. Sentinel classifies it into one of six categories and builds an appropriate pipeline:

| Category | Example prompts |
|---|---|
| **coding** | `Write a FastAPI endpoint that validates a JWT` |
| **debugging** | `Debug the failing tests in tests/test_auth.py` |
| **reasoning** | `Explain the dependency graph of this project` |
| **research** | `Find best practices for async SQLAlchemy with FastAPI` |
| **devops** | `Add GitHub Actions CI for running pytest on push` |
| **system** | `Open the project in VS Code` |

### Good prompt patterns

**Be specific about the goal:**
```
sentinel › Refactor src/database.py to use async SQLAlchemy 2.0 with connection pooling
```

**Reference files directly:**
```
sentinel › Fix the type errors in @file:src/models/user.py
```

**Scope the task clearly:**
```
sentinel › Write unit tests for the UserService class in src/services/user.py
```

**Chain related tasks:**
```
sentinel › First read the README, then add a docker-compose.yml that matches the setup instructions
```

---

## 6. Session Management

### Sessions are automatically saved

Each session is saved as a JSON file under `~/.sentinel/sessions/`. The session ID is a UUID and is printed at startup:

```
Session ID: 5f63c8db-7f49-4fa4-a42f-6d53ec2d4b5f
```

### Resume a session

```bash
python main.py --resume 5f63c8db-7f49-4fa4-a42f-6d53ec2d4b5f
```

Or from within the REPL:

```
sentinel › /resume 5f63c8db-7f49-4fa4-a42f-6d53ec2d4b5f
```

### Session data

Each session stores:
- All conversation turns (user and assistant)
- Pipeline state from the last run
- Project root path
- Start time and last-active time

---

## 7. Attachment System

You can include file content, URLs, and PDF text directly in prompts using `@token` syntax:

| Token | Example | Description |
|---|---|---|
| `@file:PATH` | `@file:src/main.py` | Include file contents inline |
| `@url:URL` | `@url:https://docs.fastapi.tiangolo.com` | Fetch and include a web page |
| `@pdf:PATH` | `@pdf:spec.pdf` | Extract and include PDF text |

Image and snippet parsing helpers exist in the context package, but the runtime prompt expansion path currently guarantees `@file:`, `@url:`, and `@pdf:` handling.

### Examples

```
sentinel › Review @file:src/auth.py and suggest improvements to the token validation logic

sentinel › Implement the API described at @url:https://jsonplaceholder.typicode.com

sentinel › Summarise the requirements in @pdf:docs/spec.pdf and create a task breakdown
```

---

## 8. Pipeline Inspection

After any task completes, use `/pipeline` to inspect what was executed:

```
sentinel › /pipeline
```

The table shows each step's name, assigned agent, status, and elapsed time.

To see the context assembled for the last step:

```
sentinel › /context
```

To see the diff produced by the last file edit:

```
sentinel › /diff
```

---

## 9. Hardware Modes

Sentinel auto-detects your hardware on startup. You can view the detected mode in the startup output or check it with `/session`.

| Mode | RAM | Models used | Pipeline concurrency |
|---|---|---|---|
| **minimal** | 8–12 GB | `codellama:7b` + `mistral:7b` | 1 (sequential) |
| **standard** | 12–20 GB | `codellama:13b` + `mixtral:8x7b` | 2 |
| **advanced** | ≥ 20 GB or GPU | `codellama:34b` + `mixtral:8x7b` | 4 |

### Override for one session

```bash
python main.py --mode minimal
```

### Override at runtime

```
sentinel › /mode standard
```

`/mode` updates the current session metadata. Restart Sentinel with `--mode` to apply the mode change to model routing.

---

## 10. Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SENTINEL_HOME` | `~/.sentinel` | Root directory for sessions, metrics, index |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API base URL |
| `SENTINEL_EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model for RAG |
| `SENTINEL_TOKEN_BUDGET` | `3000` | Maximum context tokens per pipeline step |
| `SENTINEL_PROJECT_DIR` | current directory | Default project root when `--project` is not passed |

Set these in `.env` (copy from `.env.example`) or in your shell environment.

---

## 11. Tips and Workflow Patterns

### Use `--project` to keep Sentinel focused

```bash
python main.py --project ./my-api
```

Sentinel will index and search only that directory, giving much more relevant context.

### Skip bootstrap for daily use

After the first run, bootstrap is unnecessary:

```bash
python main.py --no-bootstrap
```

Add this to your shell alias:

```bash
alias sentinel="python /path/to/local-llm-assistant/main.py --no-bootstrap"
```

### Resume long tasks across sessions

For large refactors or multi-day features, save your session ID and resume:

```bash
# Day 1
python main.py
# note the session ID: 5f63c8db-7f49-4fa4-a42f-6d53ec2d4b5f

# Day 2
python main.py --resume 5f63c8db-7f49-4fa4-a42f-6d53ec2d4b5f
```

### Let Sentinel self-correct

If a first attempt is close but not quite right, just say what's wrong:

```
sentinel › The generated function doesn't handle None inputs — fix that
sentinel › The test should use pytest fixtures, not setUp/tearDown
```

Sentinel incorporates context from the previous pipeline.

### Use `minimal` mode for quick questions

For research or explanation tasks that don't need heavy models:

```bash
python main.py --mode minimal

sentinel › Explain what a context manager is in Python
sentinel › What does the walrus operator do?
```

### Check metrics after a long session

Metrics are saved to `~/.sentinel/metrics/<session_id>.json`. They record pipeline success rates, model latency, tool reliability, and edit acceptance rates.
