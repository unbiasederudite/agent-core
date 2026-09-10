# core

All agent intelligence.

## Contents

- `models/` — Pydantic domain and startup-config models
- `protocols/` — interfaces for interchangeable implementations
- `registries/` — runtime name-to-instance maps
- `exceptions/` — the `AgentError` hierarchy
- `session_stores/` — per-conversation message history storage implementations
- `run_context/` — per-run `(agent, session_id)` correlation context, threaded through logging
- `tracing/` — process-wide content-capture flag
