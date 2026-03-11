"""command_parser.py — Sentinel slash-command parser.

Parses raw user input and dispatches recognised slash commands.
Unrecognised input is treated as a task prompt and returned for
the agent pipeline to handle.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class Command:
    """Represents a registered slash command.

    Attributes:
        name: The command name without the leading slash (e.g. 'help').
        description: One-line description shown in /help output.
        usage: Optional usage hint string (e.g. '/resume <session_id>').
        handler: Callable invoked when the command is matched.
        aliases: Optional list of alternative command names.
    """

    name: str
    description: str
    handler: Callable[["ParsedInput"], None]
    usage: Optional[str] = None
    aliases: List[str] = field(default_factory=list)


@dataclass
class ParsedInput:
    """The result of parsing a single line of user input.

    Attributes:
        is_command: True if the input is a slash command.
        command: The command name (without slash), if is_command is True.
        args: Positional arguments following the command name.
        raw: The original unmodified input string.
        is_task: True if the input should be treated as a task prompt.
    """

    raw: str
    is_command: bool = False
    command: Optional[str] = None
    args: List[str] = field(default_factory=list)
    is_task: bool = False


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class CommandParser:
    """Parses user input and dispatches slash commands.

    Register handlers for each supported command via register().
    Call parse() to classify a raw input string, then dispatch() to
    invoke the appropriate handler.
    """

    COMMAND_PREFIX = "/"

    def __init__(self) -> None:
        self._registry: Dict[str, Command] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, command: Command) -> None:
        """Register a command and its aliases.

        Args:
            command: The Command instance to register.
        """
        self._registry[command.name] = command
        for alias in command.aliases:
            self._registry[alias] = command

    def register_many(self, commands: List[Command]) -> None:
        """Register multiple commands at once.

        Args:
            commands: List of Command instances to register.
        """
        for cmd in commands:
            self.register(cmd)

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def parse(self, raw: str) -> ParsedInput:
        """Parse a raw input string into a ParsedInput.

        Args:
            raw: The raw string entered by the user.

        Returns:
            A ParsedInput describing whether the input is a command or task.
        """
        stripped = raw.strip()

        if not stripped.startswith(self.COMMAND_PREFIX):
            return ParsedInput(raw=raw, is_task=bool(stripped), is_command=False)

        parts = stripped[len(self.COMMAND_PREFIX):].split()
        if not parts:
            return ParsedInput(raw=raw, is_task=False, is_command=False)

        name = parts[0].lower()
        args = parts[1:]
        return ParsedInput(raw=raw, is_command=True, command=name, args=args)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def dispatch(self, parsed: ParsedInput) -> bool:
        """Dispatch a parsed command to its registered handler.

        Args:
            parsed: A ParsedInput with is_command=True.

        Returns:
            True if a handler was found and invoked, False if the command
            is not recognised.
        """
        if not parsed.is_command or parsed.command is None:
            return False

        cmd = self._registry.get(parsed.command)
        if cmd is None:
            return False

        cmd.handler(parsed)
        return True

    def is_known(self, name: str) -> bool:
        """Check whether a command name is registered.

        Args:
            name: Command name without the leading slash.

        Returns:
            True if the command is registered.
        """
        return name in self._registry

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_commands(self) -> List[Command]:
        """Return the list of unique registered commands (no alias duplicates).

        Returns:
            List of unique Command instances.
        """
        seen = set()
        unique: List[Command] = []
        for cmd in self._registry.values():
            if cmd.name not in seen:
                seen.add(cmd.name)
                unique.append(cmd)
        return sorted(unique, key=lambda c: c.name)

    def get_help_table(self) -> List[Tuple[str, str]]:
        """Return a list of (command, description) pairs for /help output.

        Returns:
            Sorted list of (name_with_slash, description) tuples.
        """
        return [
            (f"/{cmd.name}", cmd.description)
            for cmd in self.get_commands()
        ]
