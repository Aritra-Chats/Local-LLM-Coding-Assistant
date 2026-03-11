"""learning — Sentinel learning and self-improvement system.

Public API
----------
PerformanceTracker           — Central metrics store (all four families).
PipelineMetric               — Rolling stats for a (category, mode) pipeline bucket.
ModelMetric                  — Rolling latency + reliability for a (model, category) pair.
EditMetric                   — Edit acceptance rate per agent.
ToolMetric                   — Tool reliability per registered tool.
PerformanceSnapshot          — Point-in-time snapshot of all metrics.

LearningPipelineOptimizer    — Apply data-driven patches to pipeline steps.
OptimizationSuggestion       — A single actionable recommendation.
PipelineOptimizationReport   — Collection of suggestions for one pipeline.

PromptOptimizer              — Track variant performance; surface improvements.
PromptTemplate               — Named (agent, category) template with variants.
PromptVariant                — A single prompt variant with rolling stats.
PromptObservation            — A single recorded invocation outcome.
"""

from learning.metrics_tracker import (  # noqa: F401
    PerformanceTracker,
    PipelineMetric,
    ModelMetric,
    EditMetric,
    ToolMetric,
    PerformanceSnapshot,
)

from learning.feedback_loop import (  # noqa: F401
    LearningPipelineOptimizer,
    OptimizationSuggestion,
    PipelineOptimizationReport,
    RETRY_BOOST_THRESHOLD,
    LATENCY_THRESHOLD_MS,
    COUNCIL_DROP_THRESHOLD,
)

from learning.prompt_optimizer import (  # noqa: F401
    PromptOptimizer,
    PromptTemplate,
    PromptVariant,
    PromptObservation,
    WEIGHT_ACCEPTANCE,
    WEIGHT_SUCCESS,
    WEIGHT_SPEED,
    MIN_OBSERVATIONS,
    REFERENCE_LATENCY_MS,
)

__all__ = [
    # performance_tracker
    "PerformanceTracker",
    "PipelineMetric",
    "ModelMetric",
    "EditMetric",
    "ToolMetric",
    "PerformanceSnapshot",
    # pipeline_optimizer
    "LearningPipelineOptimizer",
    "OptimizationSuggestion",
    "PipelineOptimizationReport",
    "RETRY_BOOST_THRESHOLD",
    "LATENCY_THRESHOLD_MS",
    "COUNCIL_DROP_THRESHOLD",
    # prompt_optimizer
    "PromptOptimizer",
    "PromptTemplate",
    "PromptVariant",
    "PromptObservation",
    "WEIGHT_ACCEPTANCE",
    "WEIGHT_SUCCESS",
    "WEIGHT_SPEED",
    "MIN_OBSERVATIONS",
    "REFERENCE_LATENCY_MS",
]
