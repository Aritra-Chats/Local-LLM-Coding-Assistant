"""async_supervisor.py — Background watchdog thread for the Sentinel supervisor.

:class:`AsyncSupervisorLoop` runs as a daemon thread alongside the
:class:`~core.execution_engine.ConcreteExecutionEngine`.  It consumes
:class:`~core.supervisor_bus.BusEvent` objects, diagnoses failures via
:class:`~agents.supervisor.ConcreteSupervisorAgent`, and injects fix actions
back into the engine — all without blocking the main execution thread.

Lifecycle::

    loop = AsyncSupervisorLoop(supervisor, bus, engine, tracker)
    loop.start()                   # before engine.run_pipeline()
    engine.run_pipeline(pipeline)  # main thread
    loop.stop()                    # called by engine after pipeline ends

Fix attempt tracking
--------------------
Each distinct error is fingerprinted as the first 60 lowercase chars of the
error message (spaces → underscores).  Once ``fix_counts[fingerprint]`` reaches
``contract.max_fix_attempts`` (default 3) the loop calls ``engine.abort()``
with a structured user-facing message.
"""
from __future__ import annotations

import re
import threading
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agents.supervisor import ConcreteSupervisorAgent
    from cli.progress_tracker import ConcreteProgressTracker
    from core.execution_engine import ConcreteExecutionEngine
    from core.supervisor_bus import BusEvent, SupervisorBus


def _fingerprint(error: str) -> str:
    """Stable short key for an error string."""
    cleaned = error.lower().replace(" ", "_")
    return re.sub(r"[^a-z0-9_]", "", cleaned)[:60]


class AsyncSupervisorLoop:
    """Background watchdog thread.

    Consumes :class:`~core.supervisor_bus.BusEvent` objects and reacts:

    * ``TOOL_FAILED``        → diagnose → fix → inject → engine resumes
    * ``STEP_ENTRY_FAILED``  → diagnose entry failure → fix → re-check
    * ``STEP_EXIT_FAILED``   → diagnose missing artifact → fix → re-check
    * ``PIPELINE_DONE``      → exit cleanly
    * ``ABORT_REQUESTED``    → exit immediately

    Args:
        supervisor: The :class:`~agents.supervisor.ConcreteSupervisorAgent`
            instance used for LLM-driven diagnosis.
        bus: The shared :class:`~core.supervisor_bus.SupervisorBus`.
        engine: The running :class:`~core.execution_engine.ConcreteExecutionEngine`.
        tracker: Optional progress tracker for live label updates.
    """

    def __init__(
        self,
        supervisor: "ConcreteSupervisorAgent",
        bus: "SupervisorBus",
        engine: "ConcreteExecutionEngine",
        tracker: Optional["ConcreteProgressTracker"] = None,
    ) -> None:
        self.supervisor = supervisor
        self.bus = bus
        self.engine = engine
        self.tracker = tracker

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # fix_counts keyed by  "{step_id}:{fingerprint}"
        self._fix_counts: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the daemon thread."""
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="sentinel-supervisor",
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the loop to exit cleanly."""
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Main event loop (runs in daemon thread)
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Main event loop."""
        from core.supervisor_bus import BusEventType

        while not self._stop_event.is_set():
            event = self.bus.get(timeout=0.5)
            if event is None:
                continue

            if event.type in (BusEventType.PIPELINE_DONE, BusEventType.ABORT_REQUESTED):
                break

            if event.type == BusEventType.TOOL_FAILED:
                self._handle_tool_failure(event)
            elif event.type == BusEventType.STEP_ENTRY_FAILED:
                self._handle_entry_failure(event)
            elif event.type == BusEventType.STEP_EXIT_FAILED:
                self._handle_exit_failure(event)
            # STEP_COMPLETED events are informational — no action needed.

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _handle_tool_failure(self, event: "BusEvent") -> None:
        """Diagnose a tool failure and inject a fix or abort."""
        fp  = _fingerprint(event.error)
        key = f"{event.step_id}:{fp}"

        # Determine max_fix_attempts from the step contract (default 3).
        max_attempts = self._max_fix_attempts(event)

        current = self._fix_counts.get(key, 0)
        if current >= max_attempts:
            self._abort_with_message(
                event,
                reason=(
                    f"Cannot fix '{event.tool_name}' failure after "
                    f"{max_attempts} attempt(s).\n"
                    f"Error: {event.error[:300]}\n"
                    f"Please resolve manually and retry."
                ),
            )
            return

        proposal = self.supervisor.diagnose_failure(event)

        if not proposal.fix_possible:
            self._abort_with_message(
                event,
                reason=(
                    f"Supervisor cannot fix '{event.tool_name}' failure.\n"
                    f"Reason: {proposal.rationale}\n"
                    f"Error: {event.error[:300]}"
                ),
            )
            return

        attempt_display = current + 1
        self._update_label(
            event,
            f"Supervisor fixing: {proposal.rationale}  (attempt {attempt_display}/{max_attempts})",
        )

        self._fix_counts[key] = current + 1
        self.engine.inject_fix_and_retry(event.step_id, proposal)

    def _handle_entry_failure(self, event: "BusEvent") -> None:
        """Diagnose an entry contract failure and attempt to fix it."""
        fp  = _fingerprint(event.error)
        key = f"{event.step_id}:entry:{fp}"

        max_attempts = self._max_fix_attempts(event)
        current = self._fix_counts.get(key, 0)

        if current >= max_attempts:
            self._abort_with_message(
                event,
                reason=(
                    f"Step '{event.step_name}' entry requirements could not be "
                    f"satisfied after {max_attempts} attempt(s).\n"
                    f"Failed: {event.error[:300]}"
                ),
            )
            return

        proposal = self.supervisor.diagnose_failure(event)

        if not proposal.fix_possible:
            self._abort_with_message(
                event,
                reason=(
                    f"Step '{event.step_name}' cannot start: entry requirements unmet.\n"
                    f"Reason: {proposal.rationale}"
                ),
            )
            return

        attempt_display = current + 1
        self._update_label(
            event,
            f"Supervisor fixing entry: {proposal.rationale}  (attempt {attempt_display}/{max_attempts})",
        )

        self._fix_counts[key] = current + 1
        self.engine.inject_fix_and_retry(event.step_id, proposal)

    def _handle_exit_failure(self, event: "BusEvent") -> None:
        """Diagnose an exit contract failure and attempt to fix it."""
        fp  = _fingerprint(event.error)
        key = f"{event.step_id}:exit:{fp}"

        max_attempts = self._max_fix_attempts(event)
        current = self._fix_counts.get(key, 0)

        if current >= max_attempts:
            self._abort_with_message(
                event,
                reason=(
                    f"Step '{event.step_name}' exit criteria could not be satisfied "
                    f"after {max_attempts} attempt(s).\n"
                    f"Failed: {event.error[:300]}"
                ),
            )
            return

        proposal = self.supervisor.diagnose_failure(event)

        if not proposal.fix_possible:
            self._abort_with_message(
                event,
                reason=(
                    f"Step '{event.step_name}' exit criteria not met.\n"
                    f"Reason: {proposal.rationale}"
                ),
            )
            return

        attempt_display = current + 1
        self._update_label(
            event,
            f"Supervisor fixing exit: {proposal.rationale}  (attempt {attempt_display}/{max_attempts})",
        )

        self._fix_counts[key] = current + 1
        self.engine.inject_fix_and_retry(event.step_id, proposal)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _max_fix_attempts(self, event: "BusEvent") -> int:
        """Extract max_fix_attempts from the event's context or default to 3."""
        ctx = event.context or {}
        step = ctx.get("_step", {}) or {}
        contract_dict = step.get("contract") or {}
        return int(contract_dict.get("max_fix_attempts", 3))

    def _update_label(self, event: "BusEvent", label: str) -> None:
        """Push a label update to the progress tracker (thread-safe via Rich).

        Prefers engine._progress_tracker (set per-run) over self.tracker
        (which may be None from startup before the first run_pipeline call).
        """
        tracker = (
            getattr(self.engine, "_progress_tracker", None)
            or self.tracker
        )
        if tracker is not None:
            try:
                tracker.update_step_action(event.step_index, label)
            except Exception:
                pass

    def _abort_with_message(self, event: "BusEvent", reason: str) -> None:
        """Tell the engine to abort and stop the supervisor loop."""
        try:
            self.engine.abort(reason)
        except Exception:
            pass
        self._stop_event.set()
