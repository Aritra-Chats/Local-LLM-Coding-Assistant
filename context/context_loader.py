"""context_loader.py — Sentinel context attachment orchestrator.

Parses @attachment tokens from user prompts and dispatches to the
appropriate sub-loader.  Returns a list of structured attachment dicts
ready for injection into the context payload via ConcreteContextBuilder.

Supported syntax
----------------
  @file:path/to/file.py
  @file:src/**/*.py          (glob expansion)
  @image:path/to/image.png
  @url:https://example.com/page
  @pdf:path/to/document.pdf
  @snippet:python            (opens a fenced block, closed by ---)
  @snippet: inline text      (inline single-line variant)

The ``ContextLoader.load(prompt)`` method:
  1. Scans the prompt for @token lines / inline tokens.
  2. Resolves each token to its sub-loader.
  3. Collects all attachment dicts.
  4. Returns a ``LoadResult`` with the cleaned prompt and attachments.

Errors from individual loaders are caught and stored in
``LoadResult.errors`` — they never abort the full load.

No external dependencies beyond the context sub-modules.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from context import file_loader, image_parser, url_fetcher, pdf_parser, snippet_loader

# ---------------------------------------------------------------------------
# Token patterns
# ---------------------------------------------------------------------------

# Matches @file:, @image:, @url:, @pdf: — value must be a single non-whitespace token.
_PATH_ATTACHMENT_RE = re.compile(
    r"(?<![`\\])@(file|image|url|pdf):(\S+)",
    re.MULTILINE,
)

# Matches @snippet: — optional language tag then optional inline content to EOL.
_SNIPPET_ATTACHMENT_RE = re.compile(
    r"(?<![`\\])@snippet:([a-zA-Z0-9_.\-]*)\s*([^\n]*)",
    re.MULTILINE,
)

# Token type → loader callable signature: (value, project_root) → attachment(s)
_LoaderFn = Callable[[str, str], Any]


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class LoadResult:
    """Result of parsing and loading all @attachments from a prompt.

    Attributes:
        prompt: The original prompt with @attachment tokens replaced by
            ``[attachment:<index>]`` placeholders.
        attachments: Ordered list of attachment dicts (one per token, with
            file/glob tokens potentially expanding to multiple entries).
        errors: List of (token_str, error_message) tuples for any tokens
            that failed to load.
    """

    prompt: str
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        """True if any attachment failed to load."""
        return bool(self.errors)

    def inject_into(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Merge attachment list into an existing context payload dict.

        Attachments are stored under the ``"attachments"`` key.  Existing
        attachment lists are extended rather than replaced.

        Args:
            context: An existing context dict (e.g. from ConcreteContextBuilder).

        Returns:
            The same dict, mutated in place, for convenience.
        """
        existing = context.get("attachments", [])
        context["attachments"] = existing + self.attachments
        return context


# ---------------------------------------------------------------------------
# ContextLoader
# ---------------------------------------------------------------------------


class ContextLoader:
    """Parse @attachment tokens and dispatch to the correct sub-loader.

    Args:
        project_root: Absolute path to the project root.  Used to resolve
            relative ``@file:`` and ``@pdf:`` paths.
    """

    def __init__(self, project_root: str = "") -> None:
        self.project_root = project_root
        self._snippet_collector = snippet_loader.SnippetLoader()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def load(self, prompt: str) -> LoadResult:
        """Parse all @attachment tokens from *prompt* and load their content.

        Handles multi-line fenced ``@snippet:`` blocks by treating the block
        as a single token spanning multiple lines before invoking the loader.

        Args:
            prompt: Raw user prompt string.

        Returns:
            LoadResult with cleaned prompt string and attachment list.
        """
        # First pass: collect fenced @snippet: blocks so they are not
        # misidentified as separate tokens.
        prompt, snippet_attachments = snippet_loader.parse(prompt)

        # Second pass: process remaining @token occurrences.
        # Merge both regex iterators sorted by position in the string.
        path_matches = [
            (m.start(), m.group(1).lower(), m.group(2).strip(), m.group(0))
            for m in _PATH_ATTACHMENT_RE.finditer(prompt)
        ]
        snippet_matches = [
            (m.start(), "snippet", (m.group(1).strip(), m.group(2).strip()), m.group(0))
            for m in _SNIPPET_ATTACHMENT_RE.finditer(prompt)
        ]
        all_matches = sorted(path_matches + snippet_matches, key=lambda x: x[0])

        attachments: List[Dict[str, Any]] = []
        errors: List[Tuple[str, str]] = []
        offset = 0
        cleaned_parts: List[str] = []

        for start, token_type, value, full_token in all_matches:
            if start < offset:
                continue

            match_end = start + len(full_token)
            placeholder_idx = len(attachments) + len(snippet_attachments)

            cleaned_parts.append(prompt[offset:start])
            offset = match_end

            loaded, err = self._dispatch(token_type, value)
            if err:
                errors.append((full_token, err))
                cleaned_parts.append(f"[attachment:{placeholder_idx}:error]")
            else:
                for att in loaded:
                    attachments.append(att)
                cleaned_parts.append(f"[attachment:{placeholder_idx}]")

        # Append any text after the last token
        cleaned_parts.append(prompt[offset:])

        all_attachments = snippet_attachments + attachments

        return LoadResult(
            prompt="".join(cleaned_parts),
            attachments=all_attachments,
            errors=errors,
        )

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _dispatch(
        self, token_type: str, value: str
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Route a parsed token to the correct sub-loader.

        Args:
            token_type: One of ``file``, ``image``, ``url``, ``pdf``.
                (``snippet`` tokens are handled before dispatch.)
            value: The value portion of the token.

        Returns:
            Tuple of (list_of_attachment_dicts, error_message_or_None).
        """
        try:
            if token_type == "file":
                results = file_loader.load(value, self.project_root)
                if not results:
                    return [], f"No files matched: {value}"
                return results, None

            elif token_type == "image":
                att = image_parser.load(value, self.project_root)
                return [att], None

            elif token_type == "url":
                att = url_fetcher.load(value)
                return [att], None

            elif token_type == "pdf":
                att = pdf_parser.load(value, self.project_root)
                return [att], None

            elif token_type == "snippet":
                # value is a (lang, content) tuple from _SNIPPET_ATTACHMENT_RE
                if isinstance(value, tuple):
                    lang, content = value
                else:
                    lang_match = re.match(r"^([a-zA-Z0-9_.\-]*)\s*(.*)", value, re.DOTALL)
                    lang = lang_match.group(1) if lang_match else ""
                    content = lang_match.group(2).strip() if lang_match else value
                att = snippet_loader.load(content, language=lang)
                return [att], None

            else:
                return [], f"Unknown attachment type: @{token_type}"

        except Exception as exc:
            return [], str(exc)

    # ------------------------------------------------------------------
    # Convenience: parse + inject in one call
    # ------------------------------------------------------------------

    def build_context(
        self,
        prompt: str,
        base_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Load attachments and inject them into a context dict.

        Args:
            prompt: Raw user prompt containing @attachment tokens.
            base_context: Existing context dict to merge into.  If None,
                a new dict is created.

        Returns:
            Tuple of (cleaned_prompt, context_dict_with_attachments).
        """
        result = self.load(prompt)
        ctx = base_context if base_context is not None else {}
        result.inject_into(ctx)
        if result.has_errors:
            ctx["attachment_errors"] = result.errors
        return result.prompt, ctx


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------


def load_attachments(
    prompt: str,
    project_root: str = "",
) -> LoadResult:
    """Parse and load all @attachment tokens from a prompt string.

    Convenience wrapper around ``ContextLoader.load()`` for one-off use.

    Args:
        prompt: Raw prompt string.
        project_root: Project root for path resolution.

    Returns:
        LoadResult instance.
    """
    return ContextLoader(project_root=project_root).load(prompt)
