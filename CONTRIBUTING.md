
# Contributing Guide

Thank you for contributing to the Local-LLM-Coding-Assistant project. This document outlines the project's contribution workflow, code style, testing expectations, and guidance for larger design proposals.

## Quick links

- [Getting started](#getting-started)
- [Code style and linters](#code-style-and-linters)
- [Tests](#tests)
- [Commit messages and PRs](#commit-messages-and-prs)
- [Large changes and design proposals](#large-changes-and-design-proposals)

## Getting started

1. Fork the repository and create a feature branch named `topic/short-description`.
2. Keep your branch targeted at `main` (or `develop` if the project uses a development branch).
3. If the change is non-trivial (design, major refactor, API changes), open an issue first describing the motivation and proposed approach.

## Code style and linters

- Python code should follow PEP 8 with 4-space indentation.
- Run formatters and linters before committing:

```bash
black .
ruff check .
isort .
```

Add or update unit tests for behavioral changes.

## Tests

- Run the test suite with `pytest -q`.
- For CI, ensure your branch passes the same test matrix configured in `.github/workflows/`.

### Running tests locally

- Run a single test file:

```bash
pytest tests/test_context/test_rag_search.py -q
```

- Run a single test function:

```bash
pytest tests/test_context/test_rag_search.py::test_search_returns_relevant -q
```

- Measure coverage locally:

```bash
pip install coverage
coverage run -m pytest
coverage report -m
```

### Linting and pre-commit

Run linters and formatters before committing:

```bash
black .
ruff check .
isort .
```

Optionally install a pre-commit hook to enforce these checks automatically.

## Commit messages and PRs

- Keep commits small and focused.
- Use conventional commits style where possible: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`.
- Submit a pull request describing the problem, design choices, and any backward-compatibility considerations.

## Large changes and design proposals

For larger architecture or API changes, propose a design document in the `docs/` folder or an issue with a design outline. Include:

- Motivation and problem statement
- Alternatives considered
- Proposed API/manifest
- Migration guide (if applicable)

<details>
<summary>Security-sensitive contributions (expand)</summary>

If your change touches security-sensitive code (sandboxing, network, secrets handling), include additional tests and a short security rationale in the PR.

</details>

## Code ownership and reviews

PRs require at least one approving reviewer from the core team. The `CriticAgent` and automated linters will run in CI to surface issues.

---

We appreciate your careful, well-documented contributions — they're what make this project sustainable.
