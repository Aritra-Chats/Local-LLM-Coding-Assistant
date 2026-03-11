# Security Policy

## Overview

Sentinel is a **local-first, offline-capable** AI assistant. Its core security guarantee
is that your data and conversations never leave your machine by default. This document
explains the security model, known outbound network vectors, shell tool considerations,
and how to report vulnerabilities.

---

## Table of Contents

1. [Security Model](#1-security-model)
2. [Outbound Network Vectors](#2-outbound-network-vectors)
3. [Shell Tool Considerations](#3-shell-tool-considerations)
4. [Data Storage](#4-data-storage)
5. [Supported Versions](#5-supported-versions)
6. [Reporting a Vulnerability](#6-reporting-a-vulnerability)
7. [Response Timeline](#7-response-timeline)
8. [Security Best Practices for Users](#8-security-best-practices-for-users)

---

## 1. Security Model

| Principle | Implementation |
|---|---|
| **Local inference** | All LLM inference runs against a local Ollama instance (`localhost:11434`). No prompts or responses are sent to any cloud provider. |
| **No telemetry** | Sentinel does not collect usage statistics, crash reports, or any other telemetry. |
| **No automatic updates** | Sentinel does not phone home for updates. All updates are explicit (`pip install --upgrade`). |
| **File system isolation** | Sentinel reads and writes only within the paths you explicitly provide. The workspace directory is `~/.sentinel/` by default. |
| **No credential storage** | Sentinel does not store passwords, tokens, or API keys. Secrets are read from environment variables at runtime. |

---

## 2. Outbound Network Vectors

While Sentinel is local-first, two features can make outbound network requests. Both
are **user-initiated** and can be disabled.

### 2.1 `web_search` Tool

The `WebSearchTool` issues HTTP GET requests to a configured search endpoint when an
agent calls it. By default this tool is disabled until a search provider URL is set in
`.env`.

**Mitigation:**
- Do not set `SENTINEL_SEARCH_URL` in `.env` if you want a fully air-gapped setup.
- Review the query before confirming any agent step that invokes `web_search`.

### 2.2 `@url:` Attachment Token

When you type `@url:https://example.com` in the REPL, Sentinel fetches the content of
that URL and includes it in the model context. The request is made from your machine
using your IP address.

**Mitigation:**
- Only attach URLs you trust.
- Sentinel does not follow redirects to `localhost` or private RFC-1918 addresses
  (SSRF protection). Attempts to attach internal URLs will be rejected.

### 2.3 Ollama API

All model inference communicates with Ollama over `http://localhost:11434`. This is a
loopback address and is not accessible from the network unless you explicitly bind
Ollama to a non-loopback interface.

---

## 3. Shell Tool Considerations

The `run_shell` tool (`RunShellTool`) executes arbitrary shell commands. This is
intentionally powerful — and therefore the highest-risk component.

### Default behaviour

- Commands are executed with the **same privileges as the user running Sentinel**.
- There is no containerisation or sandboxing by default.
- The working directory defaults to the project root.

### Recommendations

1. **Review every command** before confirming agent steps that use `run_shell`.
2. **Run Sentinel as a non-privileged user**, never as root or Administrator.
3. For higher-assurance environments, consider wrapping Sentinel in a container or VM.
4. The `SENTINEL_ALLOW_SHELL` environment variable can be set to `false` to disable the
   `run_shell` tool entirely.

```bash
# .env — disable shell execution
SENTINEL_ALLOW_SHELL=false
```

---

## 4. Data Storage

All Sentinel data is stored locally under `~/.sentinel/`:

| Path | Contents |
|---|---|
| `~/.sentinel/sessions/` | Conversation history (plaintext JSON) |
| `~/.sentinel/index/` | Vector embeddings of your project files |
| `~/.sentinel/memory/` | Persistent agent memory snapshots |
| `~/.sentinel/logs/` | Operational logs |
| `~/.sentinel/models/` | Model metadata cache |

**Recommendations:**
- Apply appropriate file-system permissions to `~/.sentinel/` to restrict access to
  your user account.
- Exclude `~/.sentinel/` from cloud sync if conversations contain sensitive information.
- The `~/.sentinel/sessions/` directory may contain any text you typed into Sentinel.
  Delete sessions you no longer need with `/clear` or by removing the JSON files.

---

## 5. Supported Versions

Security patches are applied to the **latest release only**. We do not backport fixes
to older versions.

| Version | Status |
|---|---|
| v0.1.x (current) | ✅ Supported |
| Pre-release | ❌ Not supported |

---

## 6. Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub Issues.**

Instead, use one of the following private channels:

- **GitHub Private Security Advisory:** Use the "Report a vulnerability" button on the
  repository's Security tab (preferred).
- **Email:** sentinel-security@example.com *(replace with your actual contact)*

### What to include

A good vulnerability report contains:

1. **Description** — what the vulnerability is and why it is a security issue
2. **Reproduction steps** — minimal sequence of actions to trigger the issue
3. **Impact** — what an attacker could achieve (e.g., arbitrary code execution, data
   exfiltration)
4. **Affected versions** — which Sentinel version(s) you tested against
5. **Suggested fix** (optional but appreciated)

---

## 7. Response Timeline

| Stage | Target |
|---|---|
| Acknowledgement of report | Within 48 hours |
| Initial triage and severity assessment | Within 5 business days |
| Fix developed and tested | Within 30 days (critical: within 7 days) |
| Release and public disclosure | Coordinated with reporter |

We follow **coordinated disclosure**: we will credit you in the release notes and
Security Advisory unless you prefer to remain anonymous.

---

## 8. Security Best Practices for Users

- **Run Sentinel as a non-root, non-Administrator user.** It does not require elevated
  privileges.
- **Keep Ollama updated.** Ollama itself has its own update cadence; run
  `ollama pull <model>` regularly to get patched model binaries.
- **Review agent steps before confirmation.** Sentinel shows each pipeline step before
  executing; take a moment to read `run_shell` commands.
- **Audit `.env` regularly.** Remove any keys or URLs you no longer use.
- **Do not expose Ollama to the network** unless you have a firewall rule restricting
  access. The default `localhost:11434` binding is safe.
- **Back up `~/.sentinel/`** if your sessions contain irreplaceable context, and store
  the backup securely.
