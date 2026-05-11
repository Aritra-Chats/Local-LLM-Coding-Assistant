"""api_key_manager.py — Unskippable API key collection for online mode.

Reads API keys from ``~/.sentinel/.env``, checks for missing mandatory
keys, and prompts the user interactively.  The mandatory key check is a
blocking loop — online mode cannot proceed without ``OLLAMA_API_KEY``.

Usage::

    from core.api_key_manager import APIKeyManager
    from config.settings import SENTINEL_HOME
    mgr = APIKeyManager(env_file=SENTINEL_HOME / ".env")
    safe = mgr.check_and_collect()   # returns False if user chose offline
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table


# ---------------------------------------------------------------------------
# Key definitions
# ---------------------------------------------------------------------------

REQUIRED_KEYS: Dict[str, Dict[str, Any]] = {
    "OLLAMA_API_KEY": {
        "label":       "Ollama Cloud API Key",
        "url":         "https://ollama.com/settings/keys",
        "mandatory":   True,
        "description": "Required for Ollama Cloud model access.",
    },
}

OPTIONAL_KEYS: Dict[str, Dict[str, Any]] = {
    "ANTHROPIC_API_KEY": {
        "label":       "Anthropic Claude API Key",
        "url":         "https://console.anthropic.com/settings/keys",
        "mandatory":   False,
        "description": "Used as fallback for reasoning and research tasks.",
    },
    "OPENAI_API_KEY": {
        "label":       "OpenAI API Key",
        "url":         "https://platform.openai.com/api-keys",
        "mandatory":   False,
        "description": "Used as fallback for coding and debugging tasks.",
    },
    "GOOGLE_API_KEY": {
        "label":       "Google Gemini API Key",
        "url":         "https://aistudio.google.com/app/apikey",
        "mandatory":   False,
        "description": "Used as fallback for math and data science tasks.",
    },
}


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class APIKeyManager:
    """Manages API key loading, display, and interactive collection.

    Parameters
    ----------
    env_file:
        Path to ``~/.sentinel/.env``; keys are persisted here across runs.
    """

    def __init__(self, env_file: Path) -> None:
        self.env_file = env_file
        self.console = Console()

    def load_env_file(self) -> None:
        """Load ``key=value`` pairs from ``env_file`` into ``os.environ``.

        Existing env vars are NOT overwritten, so shell-set keys take
        priority over file-stored ones.
        """
        if not self.env_file.exists():
            return
        for line in self.env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key not in os.environ:
                    os.environ[key] = value

    def save_key(self, key: str, value: str) -> None:
        """Persist ``key=value`` to ``~/.sentinel/.env`` and set in os.environ.

        Args:
            key:   Environment variable name.
            value: Secret value (stored in the file with double-quotes).
        """
        self.env_file.parent.mkdir(parents=True, exist_ok=True)
        lines: list = []
        if self.env_file.exists():
            lines = self.env_file.read_text(encoding="utf-8").splitlines()
        # Remove any existing line for this key
        lines = [l for l in lines if not l.startswith(f"{key}=")]
        lines.append(f'{key}="{value}"')
        self.env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.environ[key] = value

    def check_and_collect(self) -> bool:
        """Check for missing keys; collect mandatory keys interactively.

        This method is **blocking and unskippable** for mandatory keys.
        The user must either provide the mandatory key or type ``offline``
        to switch modes.

        Returns:
            ``True``  if all mandatory keys are present — safe to proceed.
            ``False`` if the user chose to switch to offline mode.
        """
        self.load_env_file()
        self._print_key_status_table()

        # ── Mandatory key loop ─────────────────────────────────────────
        for key, meta in REQUIRED_KEYS.items():
            while not os.environ.get(key):
                self.console.print(Panel(
                    f"[bold red]✗  {meta['label']} is missing.[/bold red]\n\n"
                    f"{meta['description']}\n\n"
                    f"Generate your key at: [link={meta['url']}]{meta['url']}[/link]\n\n"
                    "[dim]You must provide this key to use online mode.\n"
                    "Type [bold]offline[/bold] to switch to offline mode instead.[/dim]",
                    title="[bold red]Required API Key Missing[/bold red]",
                    border_style="red",
                ))

                value = Prompt.ask(
                    f"  Enter [bold]{meta['label']}[/bold] (or type [yellow]offline[/yellow] to switch mode)",
                    password=True,
                )

                if value.strip().lower() == "offline":
                    self.console.print("[yellow]Switching to OFFLINE mode.[/yellow]")
                    os.environ["SENTINEL_MODE"] = "offline"
                    return False

                if value.strip():
                    self.save_key(key, value.strip())
                    self.console.print(f"  [green]✔[/green] {meta['label']} saved.")
                else:
                    self.console.print("  [red]Empty value not accepted. Please try again.[/red]")

        # ── Optional key prompts ───────────────────────────────────────
        missing_optional = [
            (k, m) for k, m in OPTIONAL_KEYS.items()
            if not os.environ.get(k)
        ]

        if missing_optional:
            self.console.print(Panel(
                "[dim]The following optional API keys are not set.\n"
                "These are used as fallback providers when Ollama Cloud\n"
                "has no suitable model for a task. You can skip any of them.[/dim]",
                title="Optional API Keys",
                border_style="dim",
            ))
            for key, meta in missing_optional:
                value = Prompt.ask(
                    f"  [dim]{meta['label']}[/dim] (Enter to skip)",
                    default="",
                    password=True,
                )
                if value.strip():
                    self.save_key(key, value.strip())
                    self.console.print(f"  [green]✔[/green] {meta['label']} saved.")
                else:
                    self.console.print(f"  [dim]Skipped {meta['label']}.[/dim]")

        return True  # All mandatory keys present

    def _print_key_status_table(self) -> None:
        table = Table(title="API Key Status", border_style="dim")
        table.add_column("Key",      style="bold")
        table.add_column("Status")
        table.add_column("Required")
        table.add_column("Purpose")

        for key, meta in {**REQUIRED_KEYS, **OPTIONAL_KEYS}.items():
            present = bool(os.environ.get(key))
            status  = "[green]✔ Set[/green]" if present else "[red]✗ Missing[/red]"
            req     = "[bold red]Mandatory[/bold red]" if meta["mandatory"] else "[dim]Optional[/dim]"
            table.add_row(key, status, req, meta["description"])

        self.console.print(table)
