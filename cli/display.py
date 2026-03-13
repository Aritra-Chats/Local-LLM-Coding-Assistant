"""display.py — Backward-compatibility re-export shim.

The two classes that previously lived in this file have been moved to
their own dedicated modules:

    PipelineViewer  →  cli.pipeline_viewer
    ProgressTracker →  cli.progress_tracker

This shim re-exports both so that any code importing from ``cli.display``
continues to work without modification.
"""
from __future__ import annotations

from cli.pipeline_viewer import PipelineViewer       # noqa: F401
from cli.progress_tracker import ProgressTracker     # noqa: F401

__all__ = ["PipelineViewer", "ProgressTracker"]
