"""tool_registry.py — Sentinel tool registration and invocation system.

Contains:
    Tool                  — Base class for all Sentinel tools.
    ToolResult            — Standardised structured result dataclass.
    ToolRegistry          — Abstract base class (ABC).
    ConcreteToolRegistry  — Full implementation with JSON-schema validation,
                            execution timing, and error capture.
"""

import json
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Type


class ToolRegistry(ABC):
    """Abstract base class for the Tool Registry.

    Manages the registration, discovery, and invocation of all tools
    available to Sentinel agents. Tools are invoked through structured
    model outputs and routed here for safe, sandboxed execution.
    """

    @abstractmethod
    def register(self, name: str, tool: Any) -> None:
        """Register a tool under a given name.

        Args:
            name: The unique string identifier for the tool (e.g. 'read_file').
            tool: The tool instance or callable to register.
        """
        ...

    @abstractmethod
    def unregister(self, name: str) -> None:
        """Remove a tool from the registry.

        Args:
            name: The identifier of the tool to remove.
        """
        ...

    @abstractmethod
    def get(self, name: str) -> Any:
        """Retrieve a registered tool by name.

        Args:
            name: The tool identifier.

        Returns:
            The registered tool instance or callable.

        Raises:
            KeyError: If no tool with the given name is registered.
        """
        ...

    @abstractmethod
    def invoke(self, name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke a registered tool with the given parameters.

        Args:
            name: The tool identifier to invoke.
            params: A dict of parameters to pass to the tool.

        Returns:
            A result dict containing the tool output and execution metadata.
        """
        ...

    @abstractmethod
    def list_tools(self) -> List[str]:
        """Return the names of all currently registered tools.

        Returns:
            A list of registered tool name strings.
        """
        ...

    @abstractmethod
    def describe_tool(self, name: str) -> Dict[str, Any]:
        """Return the schema and description of a registered tool.

        Args:
            name: The tool identifier.

        Returns:
            A dict containing the tool's name, description, and parameter schema.
        """
        ...


# ---------------------------------------------------------------------------
# Structured result
# ---------------------------------------------------------------------------


@dataclass
class ToolResult:
    """Standardised result envelope returned by every tool invocation.

    Attributes:
        tool_name:    Name of the tool that was called.
        success:      True if the tool completed without an unhandled exception.
        output:       The primary payload — string, list, or dict depending on tool.
        error:        Error message string when success is False.
        elapsed_ms:   Wall-clock execution time in milliseconds.
        metadata:     Arbitrary extra fields the tool may populate.
    """

    tool_name: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    elapsed_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialise to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


# ---------------------------------------------------------------------------
# Base Tool
# ---------------------------------------------------------------------------


class Tool:
    """Base class for all Sentinel tools.

    Subclasses must set ``name``, ``description``, and ``parameters_schema``
    as class attributes and implement ``run(**kwargs) -> ToolResult``.

    ``parameters_schema`` format::

        {
            "param_name": {
                "type": "string",       # str | int | bool | list | dict
                "description": "...",
                "required": True,
                "default": None,        # omit when required=True
            }
        }
    """

    name: str = ""
    description: str = ""
    parameters_schema: Dict[str, Any] = {}

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the tool.  Subclasses must override."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement run()")

    def validate(self, params: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate params dict against the schema.

        Returns:
            (True, None) if valid; (False, error_message) otherwise.
        """
        schema = self.parameters_schema
        for pname, pdef in schema.items():
            if pdef.get("required") and pname not in params:
                return False, f"Missing required parameter '{pname}' for tool '{self.name}'"
        unknown = set(params) - set(schema)
        if unknown:
            return False, f"Unknown parameters for tool '{self.name}': {sorted(unknown)}"
        return True, None

    def schema_dict(self) -> Dict[str, Any]:
        """Return the tool's full schema as a JSON-compatible dict."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema,
        }


# ---------------------------------------------------------------------------
# Concrete Registry
# ---------------------------------------------------------------------------


class ConcreteToolRegistry(ToolRegistry):
    """Full tool registry: registers Tool instances, validates params,
    captures exceptions, and measures execution time.

    Usage::

        registry = ConcreteToolRegistry()
        registry.register("read_file", ReadFileTool())
        result = registry.invoke("read_file", {"path": "src/main.py"})
        # also accepts raw JSON strings for params:
        result = registry.invoke("read_file", '{"path": "src/main.py"}')
    """

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, name: str, tool: Any) -> None:
        """Register a tool under the given name.

        Args:
            name: Unique tool identifier.
            tool: A Tool instance or plain callable (auto-wrapped).
        """
        if isinstance(tool, Tool):
            self._tools[name] = tool
        elif callable(tool):
            self._tools[name] = _CallableWrapper(name=name, fn=tool)
        else:
            raise TypeError(f"Tool must be a Tool instance or callable, got {type(tool)}")

    def unregister(self, name: str) -> None:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered")
        del self._tools[name]

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered")
        return self._tools[name]

    def invoke(self, name: str, params: Any) -> Dict[str, Any]:
        """Invoke a tool by name with structured parameters.

        Args:
            name: Tool identifier.
            params: Dict of parameters, or a JSON string.

        Returns:
            ToolResult serialised as a plain dict.
        """
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except json.JSONDecodeError as exc:
                return ToolResult(tool_name=name, success=False,
                                  error=f"Invalid JSON params: {exc}").to_dict()

        if name not in self._tools:
            return ToolResult(
                tool_name=name, success=False,
                error=f"Unknown tool '{name}'. Available: {self.list_tools()}",
            ).to_dict()

        tool = self._tools[name]
        valid, err = tool.validate(params)
        if not valid:
            return ToolResult(tool_name=name, success=False, error=err).to_dict()

        t0 = time.perf_counter()
        try:
            result: ToolResult = tool.run(**params)
        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            return ToolResult(
                tool_name=name, success=False, error=str(exc),
                elapsed_ms=elapsed,
                metadata={"traceback": traceback.format_exc()},
            ).to_dict()

        result.elapsed_ms = (time.perf_counter() - t0) * 1000
        return result.to_dict()

    def list_tools(self) -> List[str]:
        return sorted(self._tools.keys())

    def describe_tool(self, name: str) -> Dict[str, Any]:
        return self.get(name).schema_dict()

    def describe_all(self) -> List[Dict[str, Any]]:
        """Return schema dicts for every registered tool."""
        return [self._tools[n].schema_dict() for n in self.list_tools()]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _CallableWrapper(Tool):
    """Wraps a plain callable as a no-schema Tool."""

    def __init__(self, name: str, fn: Callable) -> None:
        self.name = name
        self.fn = fn
        self.description = getattr(fn, "__doc__", "") or ""
        self.parameters_schema = {}

    def run(self, **kwargs: Any) -> ToolResult:
        output = self.fn(**kwargs)
        return ToolResult(tool_name=self.name, success=True, output=output)

    def validate(self, params: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        return True, None
