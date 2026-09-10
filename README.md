# agent-core

[![CI](https://github.com/unbiasederudite/agent/actions/workflows/ci.yml/badge.svg)](https://github.com/unbiasederudite/agent/actions/workflows/ci.yml)

A lightweight, extensible Python core for building AI agents — config declares the process-wide
set of available LLMs, tools, reasoning strategies, and guardrails, plus process behavior like
compaction, logging, and session storage; each agent then picks which of those it uses. None of
it is hardcoded; see [Plugins](#plugins) below for what ships today and how each one swaps out.

## Contents

- [Plugins](#plugins)
- [Requirements](#requirements)
- [Install](#install)
- [Running](#running)
- [Endpoints](#endpoints)
- [Guardrails](#guardrails)
- [Security](#security)
- [Development](#development)
- [License](#license)

## Plugins

Each of these is a swappable domain — this is what ships in each one today; see
`ARCHITECTURE.md` for how to add another.

| Plugin | Ships today |
|---|---|
| LLM Adapter | [litellm](https://github.com/BerriAI/litellm) |
| Reasoning Strategy | `react` |
| Tool | `get_current_time` |
| Guardrails | [Guardrails AI](https://guardrailsai.com/) |
| Session Store | `in_memory` |
| Transport | [FastAPI](https://fastapi.tiangolo.com/) |

## Requirements

- Python 3.13+
- [uv](https://github.com/astral-sh/uv)

## Install

```bash
uv sync
```

For development, also install the git hooks that run lint/format checks before each commit:

```bash
uv run pre-commit install
```

## Running

Copy the example config and env files, and fill in your provider API key(s):

```bash
cp config.example.json config.json
cp .env.example .env
```

`config.json` is the process's full startup configuration — which models, tools, strategies,
agents, and guardrails are available, plus process-wide settings like compaction, logging,
session storage, tracing, and more. The example works as-is; see [CONFIG.md](CONFIG.md) when
you're ready to customize it. `.env` holds the provider API key(s) — see `.env.example`'s comments for how a
model's provider prefix (e.g. `anthropic/...`) maps to its expected env var name.

Then start the API:

```bash
uv run python -m agent.api --config config.json
```

`--host` (default `127.0.0.1`) and `--port` (default `8000`) are optional.

## Endpoints

With the server running, browse the full interactive API reference at
`http://127.0.0.1:8000/docs` (Swagger UI) or `/redoc` — every route, request/response schema,
and field description, generated live from the API.

Run an agent (`POST /v1/agents/{agent_name}`), using the `researcher` agent from
`config.example.json`:

```bash
curl http://127.0.0.1:8000/v1/agents/researcher \
  -H "Content-Type: application/json" \
  -d '{"message": "What time is it?"}'
```

To continue the conversation, pass the response's `session_id` back in a follow-up request's
body.

## Guardrails

To add guardrails:

1. Browse the [Hub](https://guardrailsai.com/hub) and pick a validator — a Hub id looks like
   `namespace/name` (e.g. `guardrails/regex_match`).
2. Install it. Most validators are published directly to PyPI, named after the part of
   the Hub id after the `/`:

   ```bash
   uv pip install guardrails-ai-<name>
   ```

   A validator not yet migrated to that packaging still needs the older, deprecated CLI
   instead: `uv run guardrails hub install <hub-id>`.
3. Add a `guardrails` entry referencing its Hub id to `config.json` (see `config.example.json`
   for a working example), then list its name under an agent's `input_guardrails`,
   `tool_output_guardrails`, or `output_guardrails`.

Not every Hub validator can be used here. A validator that doesn't call an LLM at all (a regex,
a length check, a local classifier) just works, no matter which one you pick, once it's
installed by either path above. A validator that *does* call an LLM internally — an
"LLM-as-judge" style check — is only usable if its constructor accepts a parameter named
exactly `llm_callable`; that's the one convention this project can act on to redirect that call
through its own configured LLM instead of a provider called directly, tracking its cost like any
other completion. Before configuring an LLM-based validator, open its source and check its
constructor signature: if `llm_callable` is there, it's fully usable; if the validator calls an
LLM some other way — a differently named parameter, or a provider SDK called directly — this
project has no way to route or account for that call, so treat it as unsupported rather than
configuring it.

## Security

This project has no built-in authentication or authorization — it's designed for a single-person,
private deployment, not a multi-tenant or publicly exposed one. Every configured agent, and every
endpoint, is reachable by anyone who can reach the process. If you deploy this anywhere other
than `127.0.0.1`, put your own access control (a reverse proxy, a VPN, network-level
restrictions) in front of it.

## Development

Read `ARCHITECTURE.md` before any change that touches more than one folder — it's the target
design this codebase follows.

Once the git hooks are installed (see [Install](#install)), run in this order before considering
any change done:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy --strict src/
uv run pytest tests/unit/ tests/integration/ -q
E2E_TESTS=1 uv run pytest tests/e2e/ -q
```

## License

[MIT](LICENSE)
