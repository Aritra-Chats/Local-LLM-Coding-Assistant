"""selection_menu.py — keyboard-driven terminal option selection."""

from __future__ import annotations

from typing import Sequence, Tuple

from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.containers import Window, HSplit
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML


Option = Tuple[str, str]


def select_option(title: str, prompt: str, options: Sequence[Option], default_index: int = 0) -> str:
    """Modern arrow-key option selector with a Rich-like look.

    - Uses prompt_toolkit to render a centered list with highlighting.
    - Up/Down to move, Enter to select, Esc to accept default.
    """
    if not options:
        raise ValueError("select_option requires at least one option")

    index = max(0, min(default_index, len(options) - 1))

    kb = KeyBindings()

    @kb.add("up")
    def _up(event):
        nonlocal index
        index = (index - 1) % len(options)
        event.app.invalidate()

    @kb.add("down")
    def _down(event):
        nonlocal index
        index = (index + 1) % len(options)
        event.app.invalidate()

    @kb.add("escape")
    def _escape(event):
        event.app.exit(result=options[default_index][0])

    @kb.add("enter")
    def _enter(event):
        event.app.exit(result=options[index][0])

    def get_text() -> HTML:
        lines = [f"<b>{title}</b>\n", f"{prompt}\n\n"]
        for i, (val, label) in enumerate(options):
            if i == index:
                lines.append(f"<ansicyan>▶ {label}</ansicyan>\n")
            else:
                lines.append(f"  {label}\n")
        return HTML("".join(lines))

    body_control = FormattedTextControl(get_text)
    body_window = Window(content=body_control, wrap_lines=True)

    root_container = HSplit([body_window])

    style = Style.from_dict({
        "": "bg:#0f1720 #cbd5e1",
    })

    app = Application(layout=Layout(root_container), key_bindings=kb, full_screen=False, mouse_support=False, style=style)

    return app.run()