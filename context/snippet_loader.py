"""snippet_loader.py — Sentinel @snippet: attachment handler.

Captures an inline code or text snippet typed directly in the prompt and
returns it as a structured attachment dict.

Prompt syntax variants:

  Inline (single-line):
      @snippet: print("hello world")

  Fenced (multi-line, terminated by a closing fence):
      @snippet:python
      def greet(name):
          return f"Hello, {name}!"
      ---

  Fenced without explicit language tag:
      @snippet:
      some multi-line
      text here
      ---

The ``SnippetLoader`` class is the stateful entry point used when the
context loader is parsing a multi-line prompt interactively.  The module-
level ``parse()`` function extracts all snippets already embedded in a
complete prompt string.

No external dependencies.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Marker used to close a fenced snippet block
_FENCE_END = "---"

# Regex to match the opening of a @snippet: token on a line
# Captures optional language tag: @snippet:python or @snippet:
_SNIPPET_OPEN_RE = re.compile(
    r"@snippet:([a-zA-Z0-9_.\-]*)\s*(.*)",
    re.DOTALL,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load(
    raw: str,
    label: str = "",
    language: str = "",
) -> Dict[str, Any]:
    """Wrap a raw text/code snippet in a structured attachment dict.

    This is the direct builder used when the snippet content is already
    known (i.e. fully parsed from the prompt).

    Args:
        raw: The raw snippet text.
        label: Optional human-readable label for this snippet.
        language: Optional language hint (e.g. ``"python"``).

    Returns:
        Attachment dict with keys:
            - ``type``     — always ``"snippet"``
            - ``label``    — provided label or empty string
            - ``language`` — language hint string
            - ``content``  — the raw snippet text
            - ``encoding`` — always ``"text"``
            - ``lines``    — number of lines in the snippet
    """
    return {
        "type": "snippet",
        "label": label,
        "language": language,
        "content": raw,
        "encoding": "text",
        "lines": raw.count("\n") + 1,
    }


def parse(prompt: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Extract all @snippet: attachments from a complete prompt string.

    Handles both inline and fenced variants.  Replaces each @snippet: token
    in the prompt with a ``[snippet:<n>]`` placeholder preserving surrounding
    text intact.

    Args:
        prompt: Full prompt string potentially containing snippet tokens.

    Returns:
        Tuple of (cleaned_prompt, list_of_attachment_dicts).
        ``cleaned_prompt`` has snippet tokens replaced with placeholders.
        Each attachment dict matches the format returned by ``load()``.
    """
    attachments: List[Dict[str, Any]] = []
    lines = prompt.splitlines(keepends=True)
    output_lines: List[str] = []
    snippet_index = 0
    i = 0

    while i < len(lines):
        line = lines[i]
        match = _SNIPPET_OPEN_RE.match(line.rstrip("\n"))

        if match is None:
            output_lines.append(line)
            i += 1
            continue

        language = match.group(1).strip()
        inline_rest = match.group(2).strip()

        if inline_rest:
            # Inline snippet — content is on the same line
            att = load(inline_rest, label=f"snippet:{snippet_index}", language=language)
            attachments.append(att)
            output_lines.append(f"[snippet:{snippet_index}]\n")
            snippet_index += 1
            i += 1
        else:
            # Fenced snippet — collect until _FENCE_END
            i += 1
            body_lines: List[str] = []
            while i < len(lines):
                body_line = lines[i].rstrip("\n")
                if body_line.strip() == _FENCE_END:
                    i += 1
                    break
                body_lines.append(lines[i])
                i += 1
            content = "".join(body_lines).rstrip("\n")
            att = load(content, label=f"snippet:{snippet_index}", language=language)
            attachments.append(att)
            output_lines.append(f"[snippet:{snippet_index}]\n")
            snippet_index += 1

    return "".join(output_lines), attachments


# ---------------------------------------------------------------------------
# Stateful interactive snippet collector
# ---------------------------------------------------------------------------


class SnippetLoader:
    """Stateful collector for multi-line @snippet: blocks entered interactively.

    Used by the REPL / context loader when parsing user input line by line.
    Call ``feed(line)`` for each line; when ``is_collecting`` becomes False
    the completed snippet is available via ``take()``.

    Example::

        loader = SnippetLoader()
        loader.start("python")         # user typed @snippet:python
        loader.feed("def foo(): pass")
        loader.feed("---")             # closes the block
        assert not loader.is_collecting
        att = loader.take()
    """

    def __init__(self) -> None:
        self._lines: List[str] = []
        self._language: str = ""
        self._label: str = ""
        self.is_collecting: bool = False

    def start(self, language: str = "", label: str = "") -> None:
        """Begin collecting a new snippet block.

        Args:
            language: Language hint for the snippet.
            label: Optional human-readable label.
        """
        self._lines = []
        self._language = language
        self._label = label
        self.is_collecting = True

    def feed(self, line: str) -> bool:
        """Add a line to the current snippet block.

        Args:
            line: A single line of input (without trailing newline).

        Returns:
            True if the snippet is now complete, False if still collecting.
        """
        if not self.is_collecting:
            return False
        if line.strip() == _FENCE_END:
            self.is_collecting = False
            return True
        self._lines.append(line)
        return False

    def take(self) -> Optional[Dict[str, Any]]:
        """Return the completed snippet attachment dict and reset state.

        Returns:
            Attachment dict or None if not yet complete.
        """
        if self.is_collecting:
            return None
        content = "\n".join(self._lines)
        att = load(content, label=self._label, language=self._language)
        self._lines = []
        self._language = ""
        self._label = ""
        return att
