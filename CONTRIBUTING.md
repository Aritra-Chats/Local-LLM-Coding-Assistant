# Contributing to Sentinel

Thank you for your interest in contributing! This document covers how to set up a
development environment, follow project conventions, and submit changes.

---

## Table of Contents

1. [Code of Conduct](#1-code-of-conduct)
2. [How to Contribute](#2-how-to-contribute)
3. [Development Setup](#3-development-setup)
4. [Project Structure](#4-project-structure)
5. [Coding Standards](#5-coding-standards)
6. [Writing Tests](#6-writing-tests)
7. [Adding a New Agent](#7-adding-a-new-agent)
8. [Adding a New Tool](#8-adding-a-new-tool)
9. [Pull Request Process](#9-pull-request-process)
10. [Commit Message Format](#10-commit-message-format)

---

## 1. Code of Conduct

This project follows the [Code of Conduct](CODE_OF_CONDUCT.md). By participating you agree
to abide by its terms.

---

## 2. How to Contribute

| Type | Action |
|---|---|
| **Bug reports** | Open a GitHub Issue with a minimal reproduction |
| **Feature requests** | Open a GitHub Issue tagged `enhancement`; discuss before implementing large changes |
| **Documentation** | PRs against any `.md` file are always welcome |
| **Tests** | Tests for existing behaviour are especially appreciated |
| **New agents / tools** | See dedicated sections below |

---

## 3. Development Setup

```bash
git clone https://github.com/your-org/Local-LLM-Coding-Assistant.git
cd Local-LLM-Coding-Assistant

python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\Activate.ps1      # Windows PowerShell

pip install -r requirements.txt
pip install pytest pytest-cov ruff
```

Verify the setup:

```bash
python -c "from agents import build_agent_registry; print('OK')"
python -m pytest tests/ -q
```

---

## 4. Project Structure

Every top-level package follows the same pattern:

```
package/
├── __init__.py       Public API — exports the stable names
├── base_X.py         Abstract base class (ABC) where applicable
└── concrete_X.py     Concrete implementation
```

ABCs and their concrete implementations live in **separate files** (`base_*.py` vs
`concrete_*.py`). Keep that boundary intact.

---

## 5. Coding Standards

### Python version

Python **3.11+**. Every file must begin with:

```python
from __future__ import annotations
```

### Style & linting

- Follow [PEP 8](https://peps.python.org/pep-0008/)
- Run `ruff check .` before committing — fix all warnings
- Maximum line length: **100 characters**

### Type annotations

All public functions and methods **must** have full type annotations, including return types:

```python
def classify(self, info: SystemInfo) -> HardwareProfile:
    ...
```

### Docstrings

All public classes, methods, and module-level functions require **Google-style** docstrings:

```python
def build(self, step: dict[str, Any]) -> dict[str, Any]:
    """Assemble the context payload for a pipeline step.

    Args:
        step: The pipeline step dict.

    Returns:
        Context payload dict ready for the model.
    """
```

### Local-first rule

Sentinel is local-first. Do **not** add code that calls an external API, telemetry
service, or update server without explicit user initiation. All outbound network access
must be either:

1. User-initiated (e.g., the `web_search` tool or the `@url:` attachment token)
2. To the local Ollama instance (`http://localhost:11434` by default)

### Secrets and credentials

- Never hardcode credentials, tokens, or API keys
- Read secrets from environment variables
- Add new env vars to `.env.example` with a descriptive comment

---

## 6. Writing Tests

Tests live in `tests/` with one sub-package per top-level package:

```
tests/
├── test_agents/
├── test_core/
├── test_execution/
├── test_context/
└── test_tools/
```

Run the full suite:

```bash
python -m pytest tests/ -v
```

With coverage report:

```bash
python -m pytest tests/ --cov=. --cov-report=term-missing
```

### Conventions

- Name test files `test_<module>.py`
- Name test functions `test_<what_it_tests>()`
- Use `pytest.fixture` for shared setup
- **Mock Ollama calls** with `unittest.mock.patch` — tests must not require a running
  Ollama instance
- Each PR should maintain or improve overall coverage

### Example

```python
# tests/test_agents/test_task_manager.py
import pytest
from tasks.task_manager import TaskClassifier


def test_classifier_returns_coding_for_write_prompt():
    clf = TaskClassifier()
    result = clf.classify("write a Python function that parses JSON")
    assert result.category == "coding"
    assert result.confidence > 0.5


def test_classifier_returns_debugging_for_fix_prompt():
    clf = TaskClassifier()
    result = clf.classify("fix the failing test in test_auth.py")
    assert result.category == "debugging"
```

---

## 7. Adding a New Agent

**Step 1 — Create** `agents/my_agent.py`

```python
from __future__ import annotations

from typing import Any
from agents.base_agent import BaseAgent


class MyAgent(BaseAgent):
    """One-line description of what this agent does."""

    name = "my_agent"
    description = "Handles X, Y, and Z tasks."

    def run(self, step: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        # Your implementation
        ...

    def validate_output(self, output: dict[str, Any]) -> bool:
        return "result" in output

    def handle_error(self, error: Exception, step: dict[str, Any]) -> dict[str, Any]:
        return {"status": "failed", "error": str(error)}

    def describe(self) -> str:
        return self.description
```

**Step 2 — Export** from `agents/__init__.py`:

```python
from agents.my_agent import MyAgent

__all__ = [..., "MyAgent"]
```

**Step 3 — Register** in `build_agent_registry()` inside `agents/__init__.py`:

```python
"my_agent": MyAgent(),
```

**Step 4 — Map task category** in `tasks/task_manager.py` `TASK_CATEGORIES` if needed.

**Step 5 — Write tests** in `tests/test_agents/test_my_agent.py`.

---

## 8. Adding a New Tool

**Step 1 — Create** `tools/my_tool.py`

```python
from __future__ import annotations

from typing import Any
from tools.tool_registry import Tool, ToolResult


class MyTool(Tool):
    name = "my_tool"
    description = "Does X given Y."
    parameters_schema = {
        "type": "object",
        "properties": {
            "input": {"type": "string", "description": "The input to process"}
        },
        "required": ["input"],
    }

    def execute(self, params: dict[str, Any]) -> ToolResult:
        value = params["input"]
        # Your implementation
        return ToolResult(success=True, output={"result": value})
```

**Step 2 — Register** in `tools/__init__.py`:

```python
from tools.my_tool import MyTool

_BUILTIN_TOOLS = [
    ...,
    MyTool(),
]
```

**Step 3 — Write tests** in `tests/test_tools/test_my_tool.py`.

---

## 9. Pull Request Process

1. **Fork** the repository and create a descriptive branch:
   ```bash
   git checkout -b feat/my-feature
   ```

2. Make your changes following all standards above.

3. Run linting and tests:
   ```bash
   ruff check .
   python -m pytest tests/ -q
   ```

4. Push and open a **Pull Request against `main`**.

5. Fill in the PR description:
   - What does this PR do?
   - Why is this change needed?
   - How was it tested?
   - Any breaking changes?

6. Address review comments. Once approved and CI passes, a maintainer will merge.

---

## 10. Commit Message Format

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short description>

[optional body]

[optional footer]
```

| Type | When to use |
|---|---|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `refactor` | Code restructuring (no behaviour change) |
| `test` | Adding or updating tests |
| `chore` | Build, CI, dependencies |
| `perf` | Performance improvement |

### Examples

```
feat(agents): add DataAnalysisAgent for tabular data tasks

fix(core): handle None pipeline_state in supervisor.recover()

docs: add CONTRIBUTING.md with agent/tool extension guides

test(tools): add unit tests for RunShellTool timeout handling
```
