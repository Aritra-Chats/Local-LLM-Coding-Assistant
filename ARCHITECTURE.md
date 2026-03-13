# Sentinel Architecture

This document describes the internal design of every Sentinel subsystem. It is intended for contributors and developers who want to understand how the pieces fit together.

---

## Table of Contents

1. [Design Principles](#1-design-principles)
2. [Top-Level Structure](#2-top-level-structure)
3. [Request Lifecycle](#3-request-lifecycle)
4. [Agents Subsystem](#4-agents-subsystem)
5. [Task Planning](#5-task-planning)
6. [Pipeline Generator](#6-pipeline-generator)
7. [Execution Engine](#7-execution-engine)
8. [Context Engine](#8-context-engine)
9. [Tool System](#9-tool-system)
10. [Model Router](#10-model-router)
11. [Learning System](#11-learning-system)
12. [Memory & Session](#12-memory--session)
13. [Hardware Profiler](#13-hardware-profiler)
14. [CLI Layer](#14-cli-layer)
15. [Bootstrap](#15-bootstrap)
16. [Data Flow Diagram](#16-data-flow-diagram)

---

## 1. Design Principles

| Principle | Implementation |
|---|---|
| **Local-first** | All inference via Ollama; no external API calls |
| **Hardware-adaptive** | Three runtime modes (minimal / standard / advanced) selected automatically |
| **Pipeline-oriented** | Every task becomes an explicit, inspectable JSON pipeline |
| **ABC + concrete** | Each major component is defined as an ABC with one concrete implementation, making substitution trivial |
| **Self-improving** | EMA-based metrics drive pipeline and prompt optimisation across sessions |
| **Fail-safe** | Every step has retry budgets, fallback models, and supervisor recovery |

---

## 2. Top-Level Structure

```
local-llm-assistant/
├── agents/       ABC + concrete: supervisor, planner, pipeline_generator + 7 specialists
├── cli/          Interactive REPL, display, diff, command palette
├── config/       Hardware profile, model config, environment settings
├── context/      RAG engine, symbol graph, dependency graph, attachment loader
├── core/         Bootstrap, execution engine, model router, validator
├── execution/    Dynamic pipeline, step runner, retry handler, sandbox
├── learning/     Metrics tracker, pipeline optimiser, prompt A/B engine
├── memory/       Session store, conversation memory, project index
├── models/       Ollama HTTP client, embedding client, model registry
├── system/       Hardware detection, dependency installer, Ollama manager
├── tasks/        Task planner, classifier, schema re-exports
├── tools/        10 built-in tools + registry
└── main.py       SentinelRuntime orchestrator + entry point
```

---

## 3. Request Lifecycle

```
User types a prompt
        │
        ▼
InteractiveUI  ──────────────────────────────────────────────────
        │  CommandParser intercepts /commands; passes tasks onward
        ▼
SentinelRuntime.process_prompt()
        │
        ├─► ConcreteSupervisorAgent.parse_prompt()
        │       └─ classify goal, extract metadata
        │
        ├─► ConcretePlannerAgent (via supervisor.delegate())
        │       └─ TaskPlanner: classify → decompose → ExecutionPlan
        │
        ├─► ConcretePipelineGeneratorAgent
        │       └─ DynamicPipelineGenerator: plan → Pipeline JSON
        │
        ├─► LearningPipelineOptimizer.optimize()
        │       └─ apply data-driven patches to pipeline steps
        │
        ├─► ConcreteExecutionEngine.run_pipeline()
        │       └─ for each step:
        │               ConcreteContextBuilder.build()
        │               ConcreteModelRouter.select()
        │               agent.run(step, context)
        │               ToolRegistry.invoke(tool, params)
        │               ProgressTracker update
        │
        └─► PerformanceTracker.record_pipeline_run()
```

---

## 4. Agents Subsystem

**Package:** `agents/`

### Class hierarchy

```
BaseAgent (ABC)  ──  agents/base_agent.py
    ├── SupervisorAgent (ABC)          ──  agents/supervisor.py
    │       └── ConcreteSupervisorAgent
    ├── PlannerAgent (ABC)             ──  agents/planner.py
    │       └── ConcretePlannerAgent
    ├── PipelineGenerator (ABC)        ──  agents/pipeline_generator.py
    │       └── ConcretePipelineGeneratorAgent
    ├── CodingAgent                    ──  agents/coding_agent.py
    ├── DebuggingAgent                 ──  agents/debugging_agent.py
    ├── ReasoningAgent                 ──  agents/reasoning_agent.py
    ├── DevOpsAgent                    ──  agents/devops_agent.py
    ├── ResearchAgent                  ──  agents/research_agent.py
    ├── SystemAgent                    ──  agents/system_agent.py
    └── CriticAgent                    ──  agents/critic_agent.py
```

### `BaseAgent` interface

```python
run(step, context) -> Dict          # primary execution entry point
validate_output(output) -> bool     # post-run sanity check
handle_error(error, step) -> Dict   # per-agent error recovery
describe() -> str                   # human-readable agent description
```

### Agent roles

| Agent | Category | Responsibilities |
|---|---|---|
| `ConcreteSupervisorAgent` | orchestration | Parse prompts, delegate, monitor, recover |
| `ConcretePlannerAgent` | planning | Decompose goals into subtasks, assign agents |
| `ConcretePipelineGeneratorAgent` | planning | Convert `ExecutionPlan` → `Pipeline` |
| `CodingAgent` | coding | Code generation, file editing |
| `DebuggingAgent` | debugging | Error diagnosis, patch generation |
| `ReasoningAgent` | reasoning | Analysis, explanation, comparison |
| `DevOpsAgent` | devops | Git, CI/CD, shell commands |
| `ResearchAgent` | research | Web search, documentation lookup |
| `SystemAgent` | system | OS operations, application control |
| `CriticAgent` | review | Senior-reviewer pass on write_file actions before dispatch |

### `AgentAction`

All agent outputs are expressed as `AgentAction` objects (`agents/agent_action.py`). Action types:

```
tool_call   — invoke a registered tool
file_edit   — write or patch a file
reasoning   — internal reasoning trace
response    — final natural-language output
clarify     — request clarification from user
```

---

## 5. Task Planning

**Package:** `tasks/`

### Pipeline

```
Raw prompt
    │
    ▼
TaskClassifier.classify(goal) → TaskClassification
    scores goal against weighted keyword sets for 6 categories:
    reasoning · coding · debugging · research · devops · system
    │
    ▼
SubtaskDecomposer.decompose(goal, classification) → List[Subtask]
    applies per-category templates, assigns agents and tools per subtask
    │
    ▼
ExecutionPlanGenerator.generate(goal, subtasks) → ExecutionPlan
    adds dependency annotations, context hints, and complexity estimate
    │
    ▼
TaskPlanner.plan(task) → ExecutionPlan   (public API)
```

### `ExecutionPlan` schema

```json
{
  "plan_id": "<uuid>",
  "goal": "<string>",
  "category": "coding|debugging|reasoning|research|devops|system",
  "complexity": "low|medium|high",
  "subtasks": [
    {
      "subtask_id": "<uuid>",
      "name": "<string>",
      "agent": "<agent_name>",
      "tools": ["<tool_name>"],
      "context_hints": ["<hint>"],
      "dependencies": ["<subtask_id>"]
    }
  ],
  "estimated_steps": 5,
  "created_at": "<ISO-8601>"
}
```

---

## 6. Pipeline Generator

**Package:** `execution/`  
**Key file:** `execution/pipeline.py` (`DynamicPipelineGenerator`)

Converts an `ExecutionPlan` into an executable `Pipeline` by:

1. Mapping subtasks → `PipelineStep` objects
2. Assigning model hints per-step based on hardware mode
3. Applying retry budgets and timeout values
4. Resolving dependency ordering
5. Tagging steps eligible for council review (multi-agent voting)

### Pipeline modes

| Mode | Description |
|---|---|
| `solo` | Single agent handles each step |
| `council` | High-stakes steps reviewed by multiple agents |

### System modes (hardware-driven)

| System Mode | Token budgets | Concurrency |
|---|---|---|
| `minimal` | 4096 | 1 |
| `standard` | 8192 | 2 |
| `advanced` | 16384 | 4 |

---

## 7. Execution Engine

**Package:** `core/`  
**Key file:** `core/execution_engine.py`

`ConcreteExecutionEngine` drives a `Pipeline` to completion:

```
for each step (respecting depends_on):
    1. ConcreteContextBuilder.build(step)          → context payload
    2. ConcreteModelRouter.select(step, context)   → model tag
    3. agent.run(step, context)                    → AgentAction list
    4. for action in actions:
           if action.type == "tool_call":
               ToolRegistry.invoke(tool, params)
           if action.type == "file_edit":
               write / patch file
    5. ProgressTracker.complete_step(step_id)
    6. if step failed: supervisor.recover(failure, state)
```

Returns a `PipelineRunResult` containing:
- `status` — `completed | partial | failed`
- `step_results` — list of `StepResult` per step
- `total_elapsed_ms`
- `failed_steps` count
- `summary()` — human-readable outcome

---

## 8. Context Engine

**Package:** `context/`

`ConcreteContextBuilder` assembles a token-efficient context payload for each pipeline step from multiple sources:

| Source | Module | Content |
|---|---|---|
| RAG search | `context/rag_search.py` | Relevant code chunks from repository |
| Symbol graph | `context/symbol_graph.py` | Class/function cross-references (AST-based) |
| Dependency graph | `context/dependency_graph.py` | Module import relationships |
| Project synopsis | `context/project_synopsis.py` | LLM-generated codebase summary (cached) |
| Conversation memory | `memory/conversation_memory.py` | Recent turns from the current session |
| Attachments | `context/context_loader.py` | `@file` / `@url` / `@image` / `@pdf` / `@snippet` tokens |

The builder respects a per-profile **token budget** (`DEFAULT_TOKEN_BUDGET` in `config/settings.py`), ranking sources by relevance and truncating to fit.

---

## 9. Tool System

**Package:** `tools/`

All tools extend `Tool` (base class in `tools/tool_registry.py`):

```python
class Tool:
    name: str                              # registry key
    description: str                       # shown in /help and to models
    parameters_schema: Dict                # JSON Schema for input validation
    def execute(self, params) -> ToolResult
```

### Built-in tools

| Tool | Key | Description |
|---|---|---|
| `ReadFileTool` | `read_file` | Read file content with optional line range |
| `WriteFileTool` | `write_file` | Write or overwrite a file |
| `SearchCodeTool` | `search_code` | Regex or text search across a project |
| `RunShellTool` | `run_shell` | Execute a shell command |
| `RunTestsTool` | `run_tests` | Run pytest / unittest and capture output |
| `GitDiffTool` | `git_diff` | Show uncommitted changes |
| `GitCommitTool` | `git_commit` | Stage and commit files |
| `WebSearchTool` | `web_search` | DuckDuckGo search (no API key needed) |
| `InstallDependencyTool` | `install_dependency` | pip install a package |
| `OpenApplicationTool` | `open_application` | Open a URL or application |

`ConcreteToolRegistry` wraps every invocation with:
- JSON Schema input validation
- Execution timing
- Structured error capture (no exceptions escape)

---

## 10. Model Router

**Package:** `core/`  
**Key file:** `core/model_router.py`

`ConcreteModelRouter` selects the Ollama model for each pipeline step:

1. Reads the `HardwareProfile` to know available model tier
2. Consults step `model_hint` (set by the pipeline generator)
3. Falls back to the profile's `recommended_model` if the hint is unavailable
4. Records per-(model, category) latency via `PerformanceTracker`
5. Degrades to smaller model if `PERFORMANCE_DEGRADED_THRESHOLD` is breached

Exposes:
- `select(step, context)` — returns model tag string
- `select_coding_model()` — profile default for coding steps
- `select_reasoning_model()` — profile default for reasoning steps
- `fallback(model)` — returns next-smaller model in the tier

---

## 11. Learning System

**Package:** `learning/`

Three components work together:

### `PerformanceTracker` (`learning/metrics_tracker.py`)

Tracks four metric families using exponential moving averages (α = 0.2):

| Family | Key | Description |
|---|---|---|
| Pipeline success rate | `(category, mode)` | Pass/fail EMA per pipeline type |
| Model latency | `(model, category)` | Latency EMA, first-token latency |
| Edit acceptance | `(agent)` | User accept/reject rate per agent |
| Tool reliability | `(tool)` | Success rate and consecutive-failure streak |

Persists to `~/.sentinel/metrics/<session_id>.json`.

### `LearningPipelineOptimizer` (`learning/feedback_loop.py`)

Reads `PerformanceTracker` snapshots and patches open pipelines:
- Increases retry budgets for unreliable tools
- Swaps models on latency violations
- Removes council overhead when solo performance is already high
- Promotes high-reliability tools to front of step tool lists

### `PromptOptimizer` (`learning/prompt_optimizer.py`)

A/B tests prompt variants per `(agent, category)` pair:
- Maintains `PromptTemplate` objects with multiple `PromptVariant` entries
- Scores variants by weighted combination of acceptance rate, success rate, and speed
- Surfaces suggestions when variant count reaches `MIN_OBSERVATIONS = 20`

---

## 12. Memory & Session

**Package:** `memory/`

### `SessionManager` (`memory/session_store.py`)

Manages a single session lifecycle:
- `start()` — create or resume a session
- `add_turn(role, content)` — append a conversation turn
- `save()` — persist to `~/.sentinel/sessions/<session_id>.json`
- `pipeline_state` — dict updated after each pipeline run

Sessions are stored as JSON files and survive restarts.

### `ConversationMemory` (`memory/conversation_memory.py`)

In-process ring buffer (default 200 turns) of conversation history. Used by `ConcreteContextBuilder` as a context source.

### `ProjectIndex` (`memory/project_index.py`)

File tree index for fast path search. Rebuilt on demand; persisted to `~/.sentinel/index/`.

---

## 13. Hardware Profiler

**Package:** `config/` + `system/`

```
system/hardware_detector.py  ─►  SystemCheck.run() → SystemInfo
                                      RAM, CPU count, GPU list,
                                      CUDA/ROCm/Metal flags, VRAM

config/hardware_profile.py   ─►  HardwareProfiler.classify(info) → HardwareProfile
                                      mode (minimal/standard/advanced)
                                      recommended_model, context_limit,
                                      max_pipeline_concurrency,
                                      embedding_model, reasoning_model
```

### Classification thresholds

| Mode | RAM | GPU |
|---|---|---|
| Minimal | < 12 GB | No qualifying GPU |
| Standard | 12–20 GB | GPU < 6 GB VRAM, or none |
| Advanced | ≥ 20 GB, OR CUDA/ROCm GPU ≥ 6 GB VRAM, OR Apple Metal | Any |

---

## 14. CLI Layer

**Package:** `cli/`

| Module | Class/Function | Role |
|---|---|---|
| `cli/interface.py` | `InteractiveUI` | Main REPL loop using `prompt_toolkit` |
| `cli/interface.py` | `launch()`, `main()` | Entry-point helpers |
| `cli/pipeline_viewer.py` | `PipelineViewer` | Rich tree/table rendering of pipeline steps |
| `cli/progress_tracker.py` | `ProgressTracker` | Rich Live progress bar during execution |
| `cli/display.py` | re-export shim | Backward-compat re-export of `PipelineViewer` and `ProgressTracker` |
| `cli/command_palette.py` | `CommandParser` | Slash-command registration and dispatch |
| `cli/diff_viewer.py` | `DiffViewer` | Syntax-highlighted unified diff output |

`InteractiveUI` exposes a `_handle_task` hook that `SentinelRuntime.make_task_handler()` wires to the full pipeline at startup.

---

## 15. Bootstrap

**Package:** `core/bootstrap.py`

Seven-step first-launch sequence:

| Step | Action |
|---|---|
| 1 | Detect hardware → `HardwareProfile` |
| 2 | Install required Python packages via pip |
| 3 | Install Ollama if not on PATH |
| 4 | Pull primary language model |
| 5 | Pull embedding model |
| 6 | Create workspace directories under `~/.sentinel/` |
| 7 | Build initial project index |

On subsequent launches, presence of `~/.sentinel/.bootstrapped` skips all steps. Pass `--no-bootstrap` to skip unconditionally.

---

## 16. Data Flow Diagram

```
┌──────────┐    prompt     ┌─────────────────┐
│   User   │──────────────▶│  InteractiveUI  │
└──────────┘               └────────┬────────┘
                                    │ task string
                                    ▼
                           ┌─────────────────┐   parse_prompt   ┌──────────────────┐
                           │ SentinelRuntime │─────────────────▶│ SupervisorAgent  │
                           └────────┬────────┘                  └──────────────────┘
                                    │ structured task
                                    ▼
                           ┌─────────────────┐    plan()        ┌──────────────────┐
                           │  SentinelRuntime│─────────────────▶│   TaskPlanner    │
                           └────────┬────────┘                  └──────────────────┘
                                    │ ExecutionPlan
                                    ▼
                           ┌─────────────────┐   generate()     ┌──────────────────┐
                           │  SentinelRuntime│─────────────────▶│ PipelineGenerator│
                           └────────┬────────┘                  └──────────────────┘
                                    │ Pipeline
                                    ▼
                           ┌─────────────────┐   optimize()     ┌──────────────────┐
                           │  SentinelRuntime│─────────────────▶│ PipelineOptimizer│
                           └────────┬────────┘                  └──────────────────┘
                                    │ Pipeline (patched)
                                    ▼
                           ┌─────────────────┐                  ┌──────────────────┐
                           │ ExecutionEngine │◀────────────────▶│  ContextBuilder  │
                           │                 │                  ├──────────────────┤
                           │  step loop      │◀────────────────▶│   ModelRouter    │
                           │                 │                  ├──────────────────┤
                           │                 │◀────────────────▶│  AgentRegistry   │
                           │                 │                  ├──────────────────┤
                           └────────┬────────┘◀────────────────▶│  ToolRegistry    │
                                    │ PipelineRunResult          └──────────────────┘
                                    ▼
                           ┌─────────────────┐
                           │PerformanceTracker│
                           └─────────────────┘
```
