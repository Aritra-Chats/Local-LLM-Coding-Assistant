# Sentinel — Roadmap

This document describes planned features, milestones, and long-term direction for the
Sentinel local-LLM assistant project.

> **Note:** Dates are targets only. This is a community-driven project and timelines may
> shift. Follow the GitHub Milestones page for the most current status.

---

## Current Release — v0.1 (Stable)

**Theme: Local Foundation**

Everything needed to run a fully functional, local AI assistant on consumer hardware.

### Delivered

- [x] Hardware auto-detection (Minimal / Standard / Advanced modes)
- [x] One-command bootstrap (Ollama install, model pull, workspace creation)
- [x] 9-agent pipeline system (Supervisor, Planner, Pipeline Generator, Coding,
      Debugging, Reasoning, DevOps, Research, System)
- [x] 10 built-in tools (file I/O, shell, search, git, web, dependency management)
- [x] Rich interactive REPL with slash commands
- [x] Attachment system (`@file:`, `@url:`, `@image:`, `@pdf:`, `@snippet:`)
- [x] Session persistence (`ses_YYYYMMDD_HHMMSS`)
- [x] Learning system with EMA performance tracking
- [x] Flat package architecture with full type annotations

---

## v0.2 — Enhanced Interfaces

**Theme: Meet the user where they are**

| Feature | Status | Notes |
|---|---|---|
| VS Code extension | Planned | Side-panel REPL, inline suggestion accept/reject |
| Web UI (FastAPI + HTMX) | Planned | Runs locally on `localhost:8765` |
| Markdown rendering in REPL | Planned | Syntax-highlighted code blocks |
| Session diff view | Planned | `sentinel › /diff <session_id>` command |
| Configurable REPL theme | Planned | Light / dark / custom colour schemes |

---

## v0.3 — Project Awareness

**Theme: Understand the codebase, not just the file**

| Feature | Status | Notes |
|---|---|---|
| Persistent vector store | Planned | ChromaDB or FAISS — survives restarts |
| Project-scoped memory | Planned | Per-project `.sentinel/` directory |
| Multi-file context window | Planned | Automatically include relevant files |
| Symbol-aware search | Planned | AST-level search for classes / functions |
| Incremental index updates | Planned | Re-index only changed files on startup |
| Multi-project switching | Planned | `sentinel --project /path/to/project` |

---

## v0.4 — Extension System

**Theme: Composable, community-driven tools and agents**

| Feature | Status | Notes |
|---|---|---|
| Plugin manifest format | Planned | `plugin.json` describing a tool or agent |
| Plugin CLI installer | Planned | `sentinel plugin install <name>` |
| Plugin registry (local) | Planned | Curated index of community plugins |
| Custom agent builder UI | Planned | Define agent roles in a YAML config |
| Tool sandboxing | Planned | Namespace-isolated execution for plugins |
| Hot reload for plugins | Planned | Load/unload without restarting Sentinel |

---

## v0.5 — Quality & Reliability

**Theme: Production hardening**

| Feature | Status | Notes |
|---|---|---|
| Full test suite (≥90% coverage) | Planned | Covers all 12 packages |
| Integration test harness | Planned | End-to-end pipeline tests with mock Ollama |
| Property-based tests | Planned | `hypothesis` for parser and planner |
| Structured logging | Planned | JSON log output, configurable verbosity |
| Graceful shutdown / interrupt | Planned | Ctrl-C saves state and exits cleanly |
| Health check endpoint | Planned | `/health` for the web UI backend |
| CI / CD pipeline | Planned | GitHub Actions: lint, test, release |

---

## v1.0 — General Availability

**Theme: Stable, documented, and community-ready**

| Feature | Status | Notes |
|---|---|---|
| Stable public API | Planned | Semantic versioning, deprecation policy |
| Full API reference docs | Planned | Auto-generated from docstrings (Sphinx) |
| Comprehensive test coverage | Planned | ≥90% across all packages |
| Windows / Linux / macOS parity | Planned | Verified on all three platforms |
| Long-term support policy | Planned | LTS branch, security patches |
| Official plugin registry | Planned | Public hosted index |

---

## Post-1.0 Horizon

These are ideas on the long-term wishlist. No timeline or commitment is implied.

- **Team collaboration mode** — shared session server, multi-user REPL
- **Local fine-tuning** — lightweight LoRA adapter training on user-accepted edits
- **Voice interface** — push-to-talk/STT frontend, TTS response readback
- **Mobile companion app** — iOS / Android app tunnelling to local Sentinel instance
- **Agent marketplace** — community agents with signatures and ratings
- **Direct IDE plugins** — JetBrains, Neovim, Emacs integrations

---

## How to Influence the Roadmap

1. **Vote** on existing GitHub Issues with a 👍 reaction
2. **Open a feature request** tagged `enhancement` with a clear use-case description
3. **Join the discussion** in GitHub Discussions
4. **Submit a PR** — working code always accelerates prioritisation

---

## Version History

| Version | Date | Highlights |
|---|---|---|
| v0.1.0 | 2026 | Initial release — local inference, 9 agents, 10 tools |
