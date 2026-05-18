# Project Roadmap

This roadmap outlines the planned milestones, short-term and long-term objectives, and expected timelines. It is a living document and will be updated as priorities evolve.

## Themes

- Reliability & safety
- Performance & model routing
- Developer ergonomics
- Extensibility & plugin model

## Interactive view

Expand items for details and owner notes.

<details>
<summary>High-level milestones (Gantt)</summary>

```mermaid
gantt
    title Sentinel Roadmap
    dateFormat  YYYY-MM-DD
    section Foundation
    Core architecture stabilization  :done,    des1, 2024-06-01, 2024-08-01
    Test coverage and CI              :done,    des2, 2024-07-01, 2024-09-01
    section Short-term
    Improved context ranking         :active,  des3, 2024-10-01, 2025-01-01
    Better model fallbacks           :         des4, 2025-01-10, 2025-03-01
    section Mid-term
    Plugin system & tool API         :         des5, 2025-04-01, 2025-09-01
    Multi-host orchestration         :         des6, 2025-10-01, 2026-02-01
```

</details>

## How to influence the roadmap

Open issues tagged `roadmap` and join architecture discussions on PRs. Major feature proposals should include a migration plan and backwards-compatibility notes.
