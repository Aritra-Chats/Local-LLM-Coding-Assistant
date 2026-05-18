

# Local-LLM-Coding-Assistant (Sentinel)

Local-LLM-Coding-Assistant ("Sentinel") is a local-first, extensible framework for building developer-facing assistants powered by locally-hosted large language models (primarily via Ollama). Sentinel provides modular orchestration, specialized agents, a dynamic pipeline execution engine, and schema-driven tools for safe, auditable repository edits and developer workflows.

## Purpose and audience

- Purpose: enable reproducible, auditable developer automation driven by local LLMs.
- Audience: contributors, maintainers, and teams that want a local assistant for code generation, debugging, research, and repository-aware automation.

## At a glance

- Local-first: default integrations target Ollama to reduce external data exposure.
- Auditable: pipelines, agent actions, and tool invocations are structured for traceability.
- Extensible: add agents, tools, and model adapters with clear extension points.

## Table of contents

- [Prerequisites](#prerequisites)
- [Quickstart (developer)](#quickstart-developer)
- [Development workflow](#development-workflow)
- [Where to look next](#where-to-look-next)

## Prerequisites

- Python 3.11 or later
- Ollama (recommended) for local inference
- Git (for contributing)

## Quickstart (developer)

1. Clone and enter the repo:

```bash
git clone https://github.com/your-org/Local-LLM-Coding-Assistant.git
cd Local-LLM-Coding-Assistant
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate    # macOS / Linux
.venv\Scripts\Activate.ps1  # Windows PowerShell
pip install -r requirements.txt
```

3. Run the project (development):

```bash
python main.py
```

4. Run tests:

```bash
pytest -q
```

## Development workflow

- Branching: create topic branches using `feature/<short-desc>` or `fix/<short-desc>`.
- Commit messages: use conventional styles (`feat:`, `fix:`, `docs:`, `chore:`).
- Code style: run `black`, `ruff`, and `isort` before opening a PR.
- Tests: add or update tests that cover behavior changes.

### PR checklist

- [ ] Branch follows naming convention
- [ ] Tests added/updated and pass locally
- [ ] Linting and formatting applied
- [ ] Documentation updated if public APIs changed

## Where to look next

- Architecture overview: [ARCHITECTURE.md](ARCHITECTURE.md)
- Installation and bootstrap: [INSTALLATION.md](INSTALLATION.md)
- Contribution process: [CONTRIBUTING.md](CONTRIBUTING.md)
- Usage examples: [USAGE.md](USAGE.md)

## Support & communication

Open issues for bugs and feature requests. For sensitive reports (security or abuse), use the address referenced in [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## License

See the repository `LICENSE` file for terms and attribution.

---

## Project structure (quick reference)

This list helps contributors find implementation locations quickly:

- `main.py` — CLI entrypoint and bootstrap
- `agents/` — agent implementations (coding_agent.py, debugging_agent.py, etc.)
- `core/` — supervisor, model router, execution engine
- `execution/` — pipeline types and step runner
- `context/` — RAG, loaders, symbol graph
- `models/` — Ollama clients and model adapters
- `tools/` — file, git, shell tools
- `memory/` — session and conversation persistence
- `tests/` — unit and integration tests

If you are contributing code, start by reading `CONTRIBUTING.md` and `ARCHITECTURE.md`.

