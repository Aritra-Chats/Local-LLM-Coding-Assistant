"""tools/__init__.py

Public API for the Sentinel tool system.

Exports
-------
Tool classes
    ReadFileTool, WriteFileTool, SearchCodeTool, FindFilesTool, RunShellTool, RunTestsTool,
    GitDiffTool, GitCommitTool, InstallDependencyTool, WebSearchTool,
    OpenApplicationTool, ProjectInitializerTool

Registry
    ConcreteToolRegistry, ToolResult, Tool (base)

Convenience
    build_default_registry() — returns a pre-populated ConcreteToolRegistry
    with all eleven built-in tools registered.

Usage
-----
    from tools import build_default_registry

    registry = build_default_registry()
    result = registry.invoke("read_file", {"path": "README.md"})
"""

from tools.tool_registry import (
    ConcreteToolRegistry,
    Tool,
    ToolResult,
    ToolRegistry,
)

from tools.read_file import ReadFileTool
from tools.write_file import WriteFileTool
from tools.search_code import SearchCodeTool
from tools.find_files import FindFilesTool
from tools.run_shell import RunShellTool
from tools.run_tests import RunTestsTool
from tools.git_diff import GitDiffTool
from tools.git_commit import GitCommitTool
from tools.install_dependency import InstallDependencyTool
from tools.web_search import WebSearchTool
from tools.open_application import OpenApplicationTool
from tools.project_initializer import ProjectInitializerTool

__all__ = [
    # Base / registry
    "ToolRegistry",
    "Tool",
    "ToolResult",
    "ConcreteToolRegistry",
    # Built-in tools
    "ReadFileTool",
    "WriteFileTool",
    "SearchCodeTool",
    "FindFilesTool",
    "RunShellTool",
    "RunTestsTool",
    "GitDiffTool",
    "GitCommitTool",
    "InstallDependencyTool",
    "WebSearchTool",
    "OpenApplicationTool",
    "ProjectInitializerTool",
    # Factory
    "build_default_registry",
]

# Ordered list of all built-in tool classes — used by build_default_registry.
_BUILTIN_TOOLS = [
    ReadFileTool,
    WriteFileTool,
    SearchCodeTool,
    FindFilesTool,
    RunShellTool,
    RunTestsTool,
    GitDiffTool,
    GitCommitTool,
    InstallDependencyTool,
    WebSearchTool,
    OpenApplicationTool,
    ProjectInitializerTool,
]


def build_default_registry() -> ConcreteToolRegistry:
    """Create and return a :class:`ConcreteToolRegistry` with all built-in tools.

    Each of the ten tools is instantiated and registered under its canonical
    ``name`` attribute.

    Returns:
        A ready-to-use :class:`ConcreteToolRegistry` instance.

    Example::

        registry = build_default_registry()
        print(registry.list_tools())
        # ['read_file', 'write_file', 'search_code', ...]

        result = registry.invoke("read_file", {"path": "README.md"})
    """
    registry = ConcreteToolRegistry()
    for tool_cls in _BUILTIN_TOOLS:
        instance = tool_cls()
        registry.register(instance.name, instance)
    return registry
