"""categories.py — Sentinel canonical category definitions.

Single source of truth for task/agent categories, aliases, and the
keyword-to-agent routing map.  Both the model router and the task
manager import from here so the two subsystems never drift apart.

Category names
--------------
The six canonical agent/task categories are::

    coding      — source code generation, editing, search
    debugging   — error diagnosis, test fixing, traceback analysis
    reasoning   — planning, analysis, explanation, comparison
    devops      — shell ops, git, CI/CD, dependency management
    research    — web search, documentation lookup
    system      — OS-level operations, launching applications

Alias map
---------
``CATEGORY_ALIASES`` maps raw strings (from LLM output, step dicts,
user flags, etc.) to one of the six canonical names above.  Both the
model router's ``_step_category()`` helper and the task manager's
``_route_by_keywords()`` function resolve through this map.
"""
from __future__ import annotations

from typing import Dict, FrozenSet, List, Tuple

# ---------------------------------------------------------------------------
# Canonical category names
# ---------------------------------------------------------------------------

TASK_CATEGORIES: Tuple[str, ...] = (
    "coding",
    "debugging",
    "reasoning",
    "devops",
    "research",
    "system",
)

# ---------------------------------------------------------------------------
# Alias map  (raw token → canonical category)
# ---------------------------------------------------------------------------
# Add new aliases here; both model_router and task_manager pick them up.

CATEGORY_ALIASES: Dict[str, str] = {
    # coding
    "code":         "coding",
    "implement":    "coding",
    "write":        "coding",
    "refactor":     "coding",
    "generate":     "coding",
    "build":        "coding",
    "edit":         "coding",

    # debugging
    "debug":        "debugging",
    "fix":          "debugging",
    "bug":          "debugging",
    "error":        "debugging",
    "test":         "debugging",

    # reasoning
    "plan":         "reasoning",
    "reason":       "reasoning",
    "analyse":      "reasoning",
    "analyze":      "reasoning",
    "explain":      "reasoning",
    "summarise":    "reasoning",
    "summarize":    "reasoning",
    "decide":       "reasoning",
    "compare":      "reasoning",
    "evaluate":     "reasoning",

    # devops
    "ops":          "devops",
    "deploy":       "devops",
    "docker":       "devops",
    "ci":           "devops",
    "cd":           "devops",
    "pipeline":     "devops",
    "shell":        "devops",
    "install":      "devops",
    "dependency":   "devops",
    "git":          "devops",
    "commit":       "devops",
    "diff":         "devops",

    # research
    "search":       "research",
    "research":     "research",
    "find":         "research",
    "lookup":       "research",
    "web":          "research",
    "docs":         "research",
    "documentation": "research",

    # system
    "open":         "system",
    "launch":       "system",
    "start":        "system",
    "process":      "system",
    "os":           "system",
    "platform":     "system",
}

# ---------------------------------------------------------------------------
# Keyword → agent routing table
# ---------------------------------------------------------------------------
# Each entry is (keyword_set, canonical_category).  A description string
# is routed to the first category whose keyword set has any match.

KEYWORD_AGENT_MAP: List[Tuple[FrozenSet[str], str]] = [
    (
        frozenset({"write", "implement", "create", "code", "function", "class",
                   "module", "refactor", "edit", "generate", "build"}),
        "coding",
    ),
    (
        frozenset({"debug", "fix", "error", "traceback", "exception",
                   "failing", "bug", "test"}),
        "debugging",
    ),
    (
        frozenset({"deploy", "docker", "ci", "cd", "pipeline", "shell",
                   "install", "dependency", "git", "commit", "diff"}),
        "devops",
    ),
    (
        frozenset({"search", "research", "find", "look up", "web",
                   "documentation", "docs"}),
        "research",
    ),
    (
        frozenset({"reason", "explain", "analyse", "analyze", "summarise",
                   "summarize", "decide", "compare", "evaluate"}),
        "reasoning",
    ),
    (
        frozenset({"open", "launch", "start", "system", "process", "os", "platform"}),
        "system",
    ),
]

# ---------------------------------------------------------------------------
# Default tools per agent category
# ---------------------------------------------------------------------------

AGENT_DEFAULT_TOOLS: Dict[str, List[str]] = {
    "coding":    ["read_file", "write_file", "search_code"],
    "debugging": ["run_tests", "run_shell", "read_file", "search_code"],
    "devops":    ["run_shell", "git_diff", "git_commit", "install_dependency"],
    "research":  ["web_search", "read_file"],
    "reasoning": [],
    "system":    ["open_application", "run_shell", "install_dependency"],
}

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def normalise_category(raw: str) -> str:
    """Resolve *raw* to a canonical category name.

    Strips ``_agent`` / ``-agent`` suffixes (e.g. ``"coding_agent"``
    → ``"coding"``), lower-cases, then looks up in ``CATEGORY_ALIASES``.
    Falls back to ``"reasoning"`` for unknown values.

    Args:
        raw: Raw category string (from LLM output, step dict, etc.).

    Returns:
        One of the six canonical category names.
    """
    cat = (
        str(raw)
        .lower()
        .replace("_agent", "")
        .replace("-agent", "")
        .replace("-", "_")
        .strip()
    )
    # Direct hit on canonical name
    if cat in TASK_CATEGORIES:
        return cat
    # Alias lookup
    return CATEGORY_ALIASES.get(cat, "reasoning")


def route_by_keywords(description: str) -> str:
    """Route a natural-language description to an agent category.

    Iterates ``KEYWORD_AGENT_MAP`` and returns the category of the first
    matching entry.  Defaults to ``"coding"`` when no keywords match.

    Args:
        description: Natural-language step description.

    Returns:
        Canonical agent category string.
    """
    lower = description.lower()
    for keywords, category in KEYWORD_AGENT_MAP:
        if any(kw in lower for kw in keywords):
            return category
    return "coding"


def default_tools_for_agent(agent: str) -> List[str]:
    """Return the default tool list for a given agent category.

    Args:
        agent: Canonical agent category string.

    Returns:
        List of tool name strings (may be empty).
    """
    return list(AGENT_DEFAULT_TOOLS.get(agent, []))
