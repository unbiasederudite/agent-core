# api

Inbound HTTP adapter. Thin translation layer between HTTP and `core/services/` — contains no reasoning of its own.

## Contents

- `schemas.py` — backend-native request/response models for the agent-run and registry-listing routes.
- `app.py` — `create_app(config_path)` builds and wires the FastAPI app: routes, middleware, and exception handlers.
- `logging_setup.py` — JSON/text log formatting, handlers, and correlation filters; stamps `trace_id`/`span_id` onto log records from the currently active span.
- `request_context.py` — request-id correlation middleware and its logging filter; opens the root HTTP span (named `"{method} {route}"`) and derives `request_id` from its trace id.
- `__main__.py` — CLI entrypoint: parses `--config`/`--host`/`--port` and starts the server.

## Running

    uv run python -m agent.api --config path/to/config.json

`--host` (default `127.0.0.1`) and `--port` (default `8000`) are optional.

## Endpoints

Full request/response schemas: `/docs` (Swagger UI) or `/redoc`, once the server is running.

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/agents/{agent_name}` | Run an agent against a message, continuing a session or starting a new one. |
| `GET` | `/v1/agents/{agent_name}/sessions/{session_id}` | Read a session's stored history. |
| `GET` | `/v1/agents/{agent_name}/sessions/{session_id}/usage` | Read a session's token/cost usage. |
| `DELETE` | `/v1/agents/{agent_name}/sessions/{session_id}` | Delete a session and its history. |
| `GET` | `/v1/agents` | List registered agents. |
| `GET` | `/v1/agents/{agent_name}/usage` | Read an agent's cumulative token/cost usage. |
| `GET` | `/v1/tools` | List registered tools. |
| `GET` | `/v1/models` | List registered model id strings. |
| `GET` | `/v1/strategies` | List registered reasoning strategy names. |
| `GET` | `/health` | Liveness check. |

## Errors

Every error body is `{"detail": {"message": "...", "code": "...", "request_id": "..."}}` —
`code` and `request_id` are present only where noted below.

| Status | `code` | Meaning |
|---|---|---|
| 403 | `tool_not_allowed` / `model_not_allowed` / `strategy_not_allowed` | Not permitted for this agent. |
| 404 | `agent_not_found` / `model_not_found` / `strategy_not_found` / `session_not_found` / `tool_not_found` / `guardrail_not_found` | Named resource not registered. |
| 404 / 405 | *(none)* | Unmatched route or method. |
| 409 | `session_busy` | Session already in use by another request. |
| 413 | `input_too_large` | `message` exceeds the agent's `max_input_chars`. |
| 413 | `context_window_exceeded` | Overflowed the model's context window; compaction unavailable. |
| 413 | `compaction_exhausted` | Overflowed the context window; compaction was tried and didn't help. |
| 422 | `guardrail_blocked` | A block-action input or output guardrail triggered. |
| 400 | *(none)* | Safety-net fallback for a `ClientError` not covered by any row above. |
| 429 | *(none)* | Provider rate-limited the request. Carries a `Retry-After` header. |
| 429 | `too_many_concurrent_requests` | `AppConfig`'s `max_concurrent_requests` cap reached. Carries a `Retry-After` header. |
| 500 | *(none)* | Unhandled server error. Body includes `request_id`. |
| 502 | *(none)* | Other LLM call failure. Body includes `request_id`. |
| 503 | *(none)* | Model's `max_concurrent_requests` cap reached. Carries a `Retry-After` header. |
| 504 | *(none)* | Provider request timed out, or the agent's `max_request_seconds` was exceeded. |
