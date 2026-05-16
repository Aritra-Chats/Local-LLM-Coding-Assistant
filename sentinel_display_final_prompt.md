# Sentinel Display / Output System — Final Implementation Prompt
## Grounded on: `sentinel_codebase_current.zip` + `sentinel_supervisor_implemented.zip`

---

## What Changed in the Supervisor Implementation

The second zip added five new modules and updated four existing ones. Every change affects the display prompt:

| New / Changed File | Display Impact |
|---|---|
| `core/supervisor_bus.py` | New event types (`TOOL_FAILED`, `STEP_ENTRY_FAILED`, `STEP_EXIT_FAILED`) need visual treatment |
| `core/async_supervisor.py` | Supervisor daemon thread calls `tracker.update_step_action()` — tracker must be wired |
| `core/step_contract.py` | Entry/exit contract failures need their own display state |
| `core/execution_engine.py` | Calls `tracker.update_step_action()` directly (no `ToolRegistry` hook needed); approval prompt already implemented using `tracker.paused_for_input()` |
| `execution/step_runner.py` | Defined but not yet wired into the engine — no display action needed |
| `agents/supervisor.py` | Added `diagnose_failure()` + `start_async_monitoring()` |
| `agents/planner.py` | Now generates `StepContract` fields per step |
| `agents/pipeline_generator.py` | `enrich_step()` now attaches `StepContract` |
| `main.py` | Added `engine.attach_supervisor(supervisor)` call after init |

### Critical finding: approval prompt is already implemented

`ConcreteExecutionEngine._request_approval()` already exists and works correctly. It uses `tracker.paused_for_input()` and `tracker.console`. **Do not rewrite it.** The only requirement is that the engine's `_progress_tracker` points to `SentinelDisplay`'s tracker instance so the approval UI routes through the correct console.

### Critical finding: supervisor tracker reference is stale

`attach_supervisor()` is called once at startup with `tracker=None` because the `ProgressTracker` doesn't exist yet — it's created per-run inside `run_pipeline()`. The `AsyncSupervisorLoop` caches `self.tracker = None` and its `_update_label()` method therefore never updates the UI during supervisor fix cycles. **This is a bug that must be fixed.**

---

## Architecture: One New File, Targeted Changes to Four Existing Files

### New file: `cli/sentinel_display.py`

```python
"""sentinel_display.py — Unified display context for Sentinel.

Single source of truth for all terminal output. Owns one Console,
one Live context, one ProgressTracker subclass, and all rendering state.
All other modules receive this instance instead of creating their own consoles.
"""
```

#### The `SentinelProgressTracker` subclass

The execution engine creates its own `ProgressTracker` via `self._make_tracker()` during `run_pipeline()`. Rather than monkey-patching the engine, we **subclass `ProgressTracker`** and inject the subclass into the engine before each run. The subclass overrides `update_step_action()` to also feed the tool log and trigger phase transitions in `SentinelDisplay`.

```python
class SentinelProgressTracker(ProgressTracker):
    """ProgressTracker subclass that feeds SentinelDisplay's live layout.

    The engine calls update_step_action() before every tool dispatch and
    before agent.run(). This subclass intercepts those calls and routes
    them into the display's tool log and spinner phase.
    """

    def __init__(self, display: "SentinelDisplay", console: Console) -> None:
        super().__init__(console=console)
        self._display = display

    def update_step_action(self, step_index: int, action_message: str) -> None:
        # Feed through to the base progress bar label
        super().update_step_action(step_index, action_message)
        # Also route into the display's tool log and spinner
        self._display._on_step_action(step_index, action_message)
```

The `_on_step_action()` method in `SentinelDisplay` inspects the message:

```python
def _on_step_action(self, step_index: int, message: str) -> None:
    """Called by SentinelProgressTracker on every engine label update."""
    msg_lower = message.lower()

    # Detect supervisor intervention
    if "supervisor fixing" in msg_lower or "supervisor fixing entry" in msg_lower:
        self.set_phase(AgentPhase.SUPERVISOR, message[:60])
        return

    # Detect tool dispatch (engine labels these as "→ tool_name" or "Writing X")
    tool_name = self._extract_tool_name(message)
    if tool_name:
        # Close any previously open tool entry for the same step
        self._close_running_tool_entries()
        call_id = self.begin_tool_call(tool_name, message)
        self._active_call_id = call_id
        return

    # Default: thinking/planning label
    self.set_phase(AgentPhase.THINKING, message[:60])

def _extract_tool_name(self, label: str) -> str:
    """Extract a tool name from an engine-generated label string."""
    # Engine labels tool calls as "→ tool_name" or "Writing …" etc.
    known_tools = {
        "read_file", "write_file", "search_code", "find_files",
        "run_shell", "run_tests", "git_commit", "install_dependency",
        "project_initializer",
    }
    label_lower = label.lower()
    for tool in known_tools:
        if tool.replace("_", " ") in label_lower or tool in label_lower:
            return tool
    if label.startswith("→ "):
        return label[2:].split()[0]
    return ""
```

#### Injecting the tracker into the engine

In `SentinelDisplay.live_session()`, after entering the Live context, inject the tracker into the engine:

```python
@contextmanager
def live_session(self, engine: Optional[Any] = None) -> Generator[None, None, None]:
    if not self._tty:
        yield
        return

    self._layout = self._build_layout()
    self._sentinel_tracker = SentinelProgressTracker(display=self, console=self._console)

    # Inject into the engine so it uses our tracker instead of creating its own
    if engine is not None:
        engine.show_progress = False               # prevent engine from calling _make_tracker()
        engine._progress_tracker = self._sentinel_tracker  # inject directly

    with Live(self._layout, console=self._console,
              refresh_per_second=10, screen=False,
              vertical_overflow="visible") as live:
        self._live = live

        # Start the pipeline display
        self._sentinel_tracker._live = self._live  # so tracker's Live = our Live

        try:
            yield
        finally:
            self._live = None
            if engine is not None:
                engine._progress_tracker = None
```

Pass `engine=self._runtime._engine` when calling `live_session()` in `_handle()`.

#### Fix the supervisor tracker staleness bug

In `core/async_supervisor.py`, `_update_label()` uses `self.tracker` which is set to `None` at startup. Replace:

```python
# BEFORE (broken):
def _update_label(self, event, label):
    if self.tracker is not None:
        self.tracker.update_step_action(event.step_index, label)

# AFTER (fixed):
def _update_label(self, event, label):
    # Prefer the engine's current per-run tracker (set by run_pipeline())
    # over the cached reference (which may be None from startup).
    tracker = (
        getattr(self.engine, "_progress_tracker", None)
        or self.tracker
    )
    if tracker is not None:
        try:
            tracker.update_step_action(event.step_index, label)
        except Exception:
            pass
```

This is a one-line fix in `core/async_supervisor.py`.

---

## Phase Enum

```python
class AgentPhase(Enum):
    IDLE       = auto()   # waiting for input
    BOOTING    = auto()   # 6-step init sequence
    THINKING   = auto()   # LLM call in-flight
    TOOL_CALL  = auto()   # tool being dispatched
    SUPERVISOR = auto()   # async supervisor diagnosing/fixing a failure
    CONTRACT   = auto()   # entry/exit contract check failed, supervisor fixing
    DONE       = auto()   # turn complete
    ERROR      = auto()   # unrecoverable error or abort
```

Phase badge rendering per phase:

```python
_PHASE_BADGES = {
    AgentPhase.IDLE:       ("[dim]○ Idle[/dim]",                          "dim"),
    AgentPhase.BOOTING:    ("[cyan]◑ Starting[/cyan]",                    "cyan"),
    AgentPhase.THINKING:   ("<spinner>",                                   "cyan"),
    AgentPhase.TOOL_CALL:  ("[yellow]⟳ {detail}[/yellow]",               "yellow"),
    AgentPhase.SUPERVISOR: ("[bold magenta]⚙ Supervisor: {detail}[/bold magenta]", "magenta"),
    AgentPhase.CONTRACT:   ("[bold yellow]◎ Contract check: {detail}[/bold yellow]", "yellow"),
    AgentPhase.DONE:       ("[bold green]✓ Done[/bold green]",            "green"),
    AgentPhase.ERROR:      ("[bold red]✗ Error[/bold red]",               "red"),
}
```

---

## Layout Structure

```python
def _build_layout(self) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header",   size=3),
        Layout(name="tool_log", size=16),   # scrolling tool call log
        Layout(name="status",   size=1),    # persistent bottom bar
    )
    return layout
```

The `tool_log` zone height is capped at 16 rows. When the log has fewer entries it shrinks naturally via `Panel` height.

---

## Tool Call Log — Entry States

The log records each call as a dict. The `begin_tool_call()` / `end_tool_call()` API is unchanged from the earlier prompt spec, with one addition: a `"supervisor_fix"` status for fix actions injected by the async supervisor.

```python
# Status values:
#   "running"       — tool call in-flight (spinning icon)
#   "done"          — completed successfully
#   "error"         — failed
#   "supervisor_fix"— this is a supervisor-injected fix action (magenta)

def _render_tool_log(self) -> Panel:
    table = Table(box=None, show_header=False, padding=(0, 1))
    table.add_column("icon",   width=3,  no_wrap=True)
    table.add_column("tool",   width=18, no_wrap=True)
    table.add_column("arg",    width=44, no_wrap=True)
    table.add_column("result", no_wrap=True)

    visible = self._tool_log[-12:]
    for i, entry in enumerate(visible):
        is_recent = i >= len(visible) - 3
        dim_pfx = "" if is_recent else "dim "

        status = entry["status"]
        if status == "running":
            icon = Text(_SPINNER_FRAMES[self._spinner_tick % 4], style="yellow")
            tool_s = f"{dim_pfx}yellow"
        elif status == "done":
            icon = Text("✓", style="bold green")
            tool_s = f"{dim_pfx}green"
        elif status == "supervisor_fix":
            icon = Text("⚙", style="bold magenta")
            tool_s = f"{dim_pfx}magenta"
        else:
            icon = Text("✗", style="bold red")
            tool_s = f"{dim_pfx}red"

        elapsed = ""
        if entry.get("end_time") and entry.get("start_time"):
            s = entry["end_time"] - entry["start_time"]
            elapsed = f" [dim]({s:.1f}s)[/dim]"

        table.add_row(
            icon,
            Text(entry["tool_name"],    style=tool_s),
            Text(entry["arg_summary"],  style=f"{dim_pfx}white"),
            Text(entry["result_summary"] + elapsed, style=f"{dim_pfx}cyan"),
        )

    # Show contract failure summary if active
    subtitle = ""
    if self._phase == AgentPhase.CONTRACT and self._contract_failed_items:
        subtitle = f"[yellow]Contract: {'; '.join(self._contract_failed_items[:2])}[/yellow]"

    return Panel(
        table,
        title="[dim]Activity[/dim]",
        subtitle=subtitle,
        border_style="dim",
    )
```

#### Supervisor fix entries

When the supervisor injects fix actions (visible in the label "Supervisor fixing: …"), add a special entry to the log:

```python
def _on_step_action(self, step_index: int, message: str) -> None:
    msg_lower = message.lower()
    if "supervisor fixing" in msg_lower:
        # Extract rationale from "Supervisor fixing: <rationale>  (attempt N/M)"
        rationale = message.split(":", 1)[-1].strip().split("(attempt")[0].strip()
        self._tool_log.append({
            "id": f"sup_{self._spinner_tick}",
            "tool_name": "supervisor",
            "arg_summary": rationale[:44],
            "status": "supervisor_fix",
            "result_summary": "",
            "start_time": time.monotonic(),
            "end_time": None,
        })
        self.set_phase(AgentPhase.SUPERVISOR, rationale[:50])
        return
    # ... rest of existing _on_step_action logic
```

#### Contract failure display

The engine emits `STEP_ENTRY_FAILED` / `STEP_EXIT_FAILED` events to `supervisor_bus`. The supervisor loop calls `_update_label()` with a "Supervisor fixing entry: …" message, which goes through `update_step_action()` → `_on_step_action()`. Capture the failed contract items from the label text and store them in `self._contract_failed_items: list[str]` so the tool log subtitle can display them.

```python
# In _on_step_action:
if "supervisor fixing entry" in msg_lower or "supervisor fixing exit" in msg_lower:
    self._phase = AgentPhase.CONTRACT
    # Parse failed items from the label (engine includes them in the message)
    self._contract_failed_items = [message.split("fixing entry:")[-1].strip()[:60]]
    self._tool_log.append({
        "id": f"contract_{self._spinner_tick}",
        "tool_name": "contract",
        "arg_summary": self._contract_failed_items[0][:44],
        "status": "supervisor_fix",
        "result_summary": "",
        "start_time": time.monotonic(),
        "end_time": None,
    })
    return
```

---

## Spinner

```python
_SPINNER_FRAMES  = ["◐", "◓", "◑", "◒"]
_SHIMMER_CHARS   = ["·", "✢", "✳", "✦", "✧", "·"]
_THINKING_VERBS  = [
    "Thinking", "Planning", "Reasoning", "Analyzing", "Decomposing",
    "Exploring", "Evaluating", "Mapping", "Synthesizing", "Verifying",
    "Drafting", "Tracing",
]
# Supervisor-specific verbs — shown when phase == SUPERVISOR
_SUPERVISOR_VERBS = [
    "Diagnosing", "Analyzing failure", "Proposing fix",
    "Checking contracts", "Recovering",
]
```

Spinner verb selection:

```python
def _spinner_text(self) -> str:
    if self._phase == AgentPhase.SUPERVISOR:
        verbs = _SUPERVISOR_VERBS
    else:
        verbs = _THINKING_VERBS

    verb_idx = (self._spinner_tick // 40) % len(verbs)
    verb     = verbs[verb_idx]
    frame    = _SPINNER_FRAMES[self._spinner_tick % 4]
    shimmer  = _SHIMMER_CHARS[self._spinner_tick % 6]
    return f"[bold cyan]{frame}[/bold cyan] [bold white]{verb}[/bold white][dim cyan]{shimmer}[/dim cyan]"
```

---

## Status Bar

```python
def _render_status_bar(self) -> Text:
    model = self._model_name or "no model"
    branch = self._get_git_branch()   # cached, 1s TTL
    
    parts = [
        f"[bold blue]{model}[/bold blue]",
        f"[dim magenta]{branch}[/dim magenta]" if branch else None,
        "[dim]sentinel active[/dim]",
    ]
    return Text("  " + "  │  ".join(p for p in parts if p))
```

---

## Safe Logging During Live

```python
def log(self, renderable: Any) -> None:
    """Print a renderable safely whether or not Live is active.
    
    When Live is active, uses live.console.print() which correctly
    inserts output above the Live region without tearing. When Live
    is not active, prints directly.
    """
    if self._live is not None:
        self._live.console.print(renderable)
    else:
        self._console.print(renderable)
```

---

## Non-Interactive / Pipe Mode

```python
def __init__(self) -> None:
    self._console = Console()
    self._tty = sys.stdout.isatty()
    # ... other fields
```

When `_tty` is False:
- `live_session()` is a transparent `yield` — no `Live` entered, no layout built
- `set_phase()`, `begin_tool_call()`, `end_tool_call()` are no-ops
- `log()` falls back to `self._console.print()`
- No spinner, no status bar

---

## Exact Changes to Existing Files

### `main.py` (3 changes)

**Change 1 — Shared console.** Replace:
```python
# Line 27-31 in main.py:
from rich.console import Console
from rich.panel import Panel
console = Console()
```
With:
```python
from rich.console import Console
from rich.panel import Panel
from cli.sentinel_display import SentinelDisplay
display = SentinelDisplay()
console = display.console   # alias: all 80+ existing console.print() calls still work
```

**Change 2 — Attach supervisor with tracker.** Replace:
```python
# Lines 323-334 in main.py (the new attach_supervisor block):
try:
    self._engine.attach_supervisor(self._supervisor)
except Exception as _sup_err:
    ...
```
With:
```python
try:
    # Pass the display's tracker so supervisor loop label updates
    # are visible in the UI from the first run.
    # Note: the tracker may be None here (created per-run); the
    # async_supervisor.py _update_label() fix handles this by falling
    # back to engine._progress_tracker at runtime.
    self._engine.attach_supervisor(self._supervisor, tracker=None)
except Exception as _sup_err:
    import sys as _sys
    print(f"[Sentinel] Warning: async supervisor could not start: {_sup_err}",
          file=_sys.stderr)
```
*(The real fix is in `async_supervisor.py`, not here.)*

**Change 3 — Wrap `_handle()` in a Live session and wire engine.** In `make_task_handler()`, the returned `_handle` function:
```python
def _handle(prompt: str) -> None:
    with display.live_session(engine=self._engine):
        display.set_phase(AgentPhase.THINKING, "Parsing prompt")
        # ... existing _handle body unchanged below this point ...
        # After out = self.process_prompt(...):
        display.set_phase(AgentPhase.DONE)
        # Replace all console.print(Panel(...)) in the tool result loop:
        for sr in result.step_results:
            ...
            for tr in sr.tool_results:
                display.render_tool_result(tr, sr, diff_viewer)
```

Add `render_tool_result()` to `SentinelDisplay` (moves the existing if/elif chain from `main.py` verbatim, replacing `console.print(...)` with `self.log(...)`):

```python
def render_tool_result(
    self,
    tool_result: dict,
    step_result: Any,
    diff_viewer: "DiffViewer",
) -> None:
    """Route a completed tool result to the appropriate display output."""
    tool    = tool_result.get("tool_name", "")
    success = tool_result.get("success", False)
    meta    = tool_result.get("metadata", {})

    # Close the running tool entry in the log
    if self._active_call_id:
        result_summary = _summarize_tool_result(tool, tool_result)
        self.end_tool_call(self._active_call_id, result_summary, success=success)
        self._active_call_id = None

    if tool == "write_file" and success:
        # ... existing diff rendering logic, replacing console.print with self.log
        pass
    elif tool in ("run_tests", "run_shell") and tool_result.get("output"):
        output_text = str(tool_result["output"])[:3000]
        style = "green" if success else "red"
        self.log(Panel(
            f"[{style}]{output_text}[/{style}]",
            title=f"[bold]{tool}[/bold] · {step_result.step_name}",
            border_style=style, expand=False,
        ))
    elif not success and tool_result.get("error"):
        self.log(f"  [red]✗[/red] {tool} failed: [dim]{tool_result['error']}[/dim]")
```

Also move the pipeline summary panel to use `display.log()`:
```python
# Replace final console.print(Panel(...)) summary with:
display.log(Panel(out["summary"], title="[bold cyan]Pipeline Complete[/bold cyan]",
                  border_style=border))
```

### `cli/interface.py` (2 changes)

**Change 1 — Remove module-level console.** Replace:
```python
console = Console()
```
With:
```python
# console is received from SentinelDisplay — set in __init__
```

**Change 2 — Accept display parameter.** Update `__init__`:
```python
def __init__(self, session: SessionManager, display: "SentinelDisplay") -> None:
    self.display = display
    self.console = display.console   # used by all existing command handlers
    self.pipeline_viewer = PipelineViewer(console=display.console)
    self.diff_viewer     = DiffViewer(console=display.console)
    self.progress        = ProgressTracker(console=display.console)
    # ... rest unchanged
```

In `main.py` where `InteractiveUI` is instantiated, pass `display`:
```python
ui = InteractiveUI(session=session, display=display)
```

### `cli/progress_tracker.py` (1 change)

**Fix `pause()` / `resume()` / `paused_for_input()`.** The engine's `_request_approval()` calls `tracker.paused_for_input()`, which currently has the broken stop/restart pattern. Replace the three methods:

```python
def pause(self) -> None:
    """No-op: Live is managed by SentinelDisplay, not by this tracker."""
    pass

def resume(self) -> None:
    """No-op: Live is managed by SentinelDisplay, not by this tracker."""
    pass

@contextmanager
def paused_for_input(self) -> Generator[None, None, None]:
    """Yield cleanly. Live.console.print() and input() work during Live.
    
    Rich >= 13 supports printing to live.console while Live is active.
    No stop/restart is needed. The engine's _request_approval() uses
    con.print(panel) followed by input() — both work correctly.
    """
    yield
```

### `core/async_supervisor.py` (1 change — bug fix)

**Fix stale tracker reference in `_update_label()`.** Replace:
```python
def _update_label(self, event: "BusEvent", label: str) -> None:
    if self.tracker is not None:
        try:
            self.tracker.update_step_action(event.step_index, label)
        except Exception:
            pass
```
With:
```python
def _update_label(self, event: "BusEvent", label: str) -> None:
    # Prefer the engine's current per-run tracker (set by run_pipeline())
    # over self.tracker which may be None (set at startup before any run).
    tracker = (
        getattr(self.engine, "_progress_tracker", None)
        or self.tracker
    )
    if tracker is not None:
        try:
            tracker.update_step_action(event.step_index, label)
        except Exception:
            pass
```

---

## `SentinelDisplay` — Complete Public API

```python
class SentinelDisplay:
    # Properties
    console: Console                    # the single shared console

    # Lifecycle
    def live_session(self, engine=None) -> ContextManager[None]: ...
    def stop(self) -> None: ...         # called on KeyboardInterrupt

    # Phase
    def set_phase(self, phase: AgentPhase, detail: str = "") -> None: ...

    # Tool call log
    def begin_tool_call(self, tool_name: str, arg_summary: str) -> str: ...
    def end_tool_call(self, call_id: str, result_summary: str, success: bool) -> None: ...

    # Status bar
    def set_status(self, **kwargs) -> None: ...   # accepts model_name=, task_name=

    # Output
    def log(self, renderable: Any) -> None: ...
    def render_tool_result(self, tool_result, step_result, diff_viewer) -> None: ...

    # Internal callbacks (called by SentinelProgressTracker)
    def _on_step_action(self, step_index: int, message: str) -> None: ...
```

---

## Boot Sequence

```python
BOOT_STEPS = [
    "Loading hardware profile",
    "Initialising model router",
    "Building agent registry",
    "Building tool registry",
    "Initialising context engine",
    "Initialising learning system",
]

def _initialize(self) -> None:
    with display.live_session():   # no engine yet at boot
        display.set_phase(AgentPhase.BOOTING)
        for i, step_name in enumerate(BOOT_STEPS, 1):
            display.set_phase(AgentPhase.BOOTING, f"[{i}/{len(BOOT_STEPS)}] {step_name}")
            # ... existing init block for this step ...

        display.set_phase(AgentPhase.IDLE)
        display.log("[bold green]✔ Sentinel runtime initialised.[/bold green]")
```

---

## Files Changed Summary

| File | Type | Change |
|---|---|---|
| `cli/sentinel_display.py` | **NEW** | `SentinelDisplay`, `SentinelProgressTracker`, `AgentPhase`, spinner, layout |
| `main.py` | Modified | Shared console, wrap `_handle` in `live_session`, boot sequence, summary panel |
| `cli/interface.py` | Modified | Remove module console, accept `display` param |
| `cli/progress_tracker.py` | Modified | Fix `pause()`/`resume()`/`paused_for_input()` to no-ops |
| `core/async_supervisor.py` | Modified | Fix stale tracker reference in `_update_label()` |
| `cli/display.py` | Unchanged | Shim still works |
| `cli/diff_viewer.py` | Unchanged | |
| `cli/pipeline_viewer.py` | Unchanged | |
| `core/step_contract.py` | Unchanged | New from supervisor zip |
| `core/supervisor_bus.py` | Unchanged | New from supervisor zip |
| `core/execution_engine.py` | Unchanged | New from supervisor zip |
| `execution/step_runner.py` | Unchanged | New from supervisor zip |
| `agents/supervisor.py` | Unchanged | New from supervisor zip |
| `agents/planner.py` | Unchanged | New from supervisor zip |
| `agents/pipeline_generator.py` | Unchanged | New from supervisor zip |
| All other original files | Unchanged | |

---

## Acceptance Criteria

**Normal run:**
```
╭──────────────────────────────────────────────────────────╮
│  Sentinel   ◑ Synthesizing✦                             │
╰──────────────────────────────────────────────────────────╯
  ✓  read_file        src/main.py              247 lines  (0.3s)
  ✓  search_code      auth middleware           3 hits    (0.8s)
  ✓  write_file       src/auth/routes.py        written   (1.2s)

  qwen2.5-coder  │  main  │  sentinel active
```

**Supervisor fixing a failure:**
```
╭──────────────────────────────────────────────────────────╮
│  Sentinel   [bold magenta]⚙ Supervisor: npm not installed[/bold magenta]   │
╰──────────────────────────────────────────────────────────╯
  ✗  run_shell        npm install              exit 1    (2.1s)
  ⚙  supervisor       npm not installed                  …
  ✓  install_dep      nodejs                   installed (4.8s)

  qwen2.5-coder  │  main  │  sentinel active
```

**Contract entry failure:**
```
╭──────────────────────────────────────────────────────────╮
│  Sentinel   [yellow]◎ Contract check: node is installed[/yellow]    │
│                                                          │
│  Contract: node is installed; npm is installed           │
╰──────────────────────────────────────────────────────────╯
```

**Approval prompt (rendered by engine's `_request_approval()`, no changes needed):**
```
╭─ ⚠  Approval Required ──────────────────────────────────╮
│  Sentinel wants to execute:                             │
│  Write → src/auth/routes.py                             │
╰──────────────────────────────────────────────────────────╯
  Apply? [Y/n/A] ›
```

The display is continuous — no tearing between the approval prompt and the Live region.
