# Architecture

This project runs AI agents defined entirely by config: config declares the process-wide set
of available LLMs, tools, reasoning strategies, and guardrails, plus process behavior like
compaction, logging, session storage, and tracing; each agent picks which of those it uses, and is
exposed to callers through a transport. Every one of those pieces is swappable — a new backend
means implementing the matching protocol, not touching the code that orchestrates a run.
`agent/core` is what makes that possible: it separates all agent reasoning from the transports
that expose it and the concrete backends it calls out to. This document describes that
separation and the invariants that hold it together — read it before any change that touches
more than one folder.

## Building Blocks

| Folder | Purpose |
|---|---|
| `src/agent/cli/` | Inbound adapter — terminal interaction (planned, not yet implemented) |
| `src/agent/api/` | Inbound adapter — HTTP interaction |
| `src/agent/core/` | All agent intelligence |
| `src/agent/core/exceptions/` | Full exception hierarchy rooted at `AgentError` |
| `src/agent/core/factories/` | Factories that construct registered instances from configuration |
| `src/agent/core/models/` | All Pydantic data models: domain models and startup config |
| `src/agent/core/protocols/` | Protocol interfaces for anything with interchangeable implementations |
| `src/agent/core/registries/` | Runtime registries for agents, LLMs, tools, strategies, and guardrails |
| `src/agent/core/run_context/` | Per-run `(agent, session_id)` correlation context, threaded through logging, plus a per-run accumulator for supporting-LLM-call usage |
| `src/agent/core/services/` | Use-case orchestration |
| `src/agent/core/session_stores/` | Per-conversation message history storage implementations |
| `src/agent/core/strategies/` | Reasoning and selection algorithm implementations |
| `src/agent/core/tools/` | Concrete tool implementations |
| `src/agent/adapters/` | Outbound adapters — the clients that talk to external systems |
| `tests/unit/` | Pure logic tests — no external deps |
| `tests/integration/` | Adapter wiring tests — all externals mocked |
| `tests/e2e/` | Real-runtime smoke tests — never in CI |

`core/` is a deliberate boundary, not just another folder: it owns all agent intelligence.
`cli/`, `api/`, and `adapters/` are thin — they translate at the edges and contain no
reasoning of their own.

---

## Boundaries

- **Inbound adapters** (`cli/`, `api/`) — the outside world calls **into** the core through
  these. Each translates its own input into a call against `core/services/`, and the result
  back into its own output format. They are siblings, never nested inside one another, and
  `core/` never depends on any of them.
- **Outbound adapters** (`adapters/`) — the core calls **out** through these, to an LLM
  provider, a tool's REST client, a database. They implement `core/protocols/` interfaces.
- Each inbound adapter is meant to ship as its own installable extra, so a deployment
  installs only the transport it needs — not yet true of the current `pyproject.toml`,
  where `api/`'s dependencies (FastAPI, uvicorn) are unconditional.
- **This runs as a single process.** Every registry, the session store, cost/context
  tracking, and `core/run_context/`'s per-run state are in-process objects with no
  cross-process coordination. Running more than one instance (multiple workers, multiple
  replicas behind a load balancer) gives each its own independent state, not a shared view —
  a session created on one instance doesn't exist on another.

```
inbound adapter -> services -> strategies -> protocols -> outbound adapters
```

Cross-cutting (available to every layer): `models`, `exceptions`, `logging.Logger`,
`core/run_context/`. The arrow above is the runtime call direction, not the import
direction — adapters import from `core/protocols/` to implement it. No circular imports; no
layer calls backwards.

---

## Config System

Everything configurable is declared in a single JSON file, loaded and validated once at
startup; nothing changes after the process starts. The configurable surface: the LLM, agent,
strategy, tool, and guardrail registries; the session store; compaction (summarizer model,
budget, keep-window); logging (level, format); and tracing (console/otlp destinations, content
capture).

Registries and factories are the binding layer between that config and runtime:

- **Config declares names**; **factories** (`core/factories/`) resolve those names to
  concrete instances once at startup; **registries** answer runtime lookups by name and
  raise immediately if a name isn't registered — never at first use.
- A name present in code but absent from config is silently unavailable. A name in config
  with no matching implementation fails fast at startup, not at invocation time.
- **An agent's `model`, `strategy`, and `tools` are defaults, not fixed choices.** A request
  may override any of them; each has a matching ceiling (`allowed_models`,
  `allowed_strategies`, `allowed_tools`) bounding what a request may override *to* — `None`
  means unrestricted, and a default outside its own ceiling fails validation at config-load
  time. Guardrail lists (`input_guardrails`, `tool_output_guardrails`, `output_guardrails`)
  have no such override: they're fixed per agent, never chosen per request.
- A component with no config surface of its own (nothing per-instance to resolve, e.g.
  logging setup) is constructed directly by its caller instead of routed through
  `core/factories/`. Follow this precedent rather than growing a factory for a component
  with nothing to configure.

---

## Extending

Adding a new backend for a config-driven protocol (`ILLM`, `IStrategy`, `ITool`, `IGuardrail`,
`ISessionStore`) means implementing it and registering the implementation in
`core/factories/app.py`; no other file in `core/` needs to change.

---

## Wire Format

The internal wire format for LLM messages and tool-calling is OpenAI-compatible.
`ILLM` implementations are contractually required to flatten tool-call/tool-result content
out of any outbound request that declares no `tools` — some providers reject tool-shaped
content on a request with no tool schema attached, even when it's just replaying history.
The leading system message is assembled from the optional root `base_prompt` followed by
the selected agent's own `system_prompt`, concatenated unconditionally and never overridden
by client-supplied messages.

---

## Conversation Model

Each conversation is a session, identified by a server-generated `session_id` and locked to
the agent it was created under. Session history lives only in memory (`ISessionStore`) — not
persisted, lost on restart; durability, not growth, is the store's known limitation.

- **Evicted by count, not time.** Sessions have no TTL; the store evicts LRU once
  `max_sessions` is exceeded, and cascades that eviction to cost/context tracking so neither
  outlives its session. Eviction skips a session currently busy or locked.
- **Two concurrency guards.** `lock()` serializes a multi-call read-modify-write; `busy()`
  rejects a second concurrent operation on the same session outright. A run marks its session
  busy before doing anything else, including one it just created, so it's protected from
  eviction from its first moment. A session a run created is discarded if that run fails
  before completing; a session the caller already had is never touched.
- **Compaction bounds cross-turn growth**: once stored history crosses a configured token
  budget, the older portion is summarized and replaced. A retry after an overflow reruns a
  strategy's whole loop, so an already-executed tool call can run again — safe today only
  because the one existing tool is read-only.
- **Cost and context tracking are independent** of each other and of compaction. Cumulative
  cost includes a run's supporting LLM calls (compaction's summarizer, a guardrail's own
  check), not just its main turn, but isn't recorded for a run that fails partway through —
  reported cost is a floor, not an exact total.
- **Per-run growth caps** (tool-result length, tool calls per round, aggregate tool-result
  size), optional per agent, bound one in-progress run independent of compaction. Both are
  visible to the LLM rather than silent — a skipped-call message or an omission marker, the
  same "let it react" contract guardrails follow.
- **Tool calls within one round execute concurrently**, not in sequence. If the loop
  exhausts its iteration cap before the model stops requesting tools, one final call is
  forced with no tools offered, so the loop always ends in an answer.
- **A run's timeout doesn't roll back other side effects** — a proactive compaction that
  already committed before the timeout fires leaves its rewrite in place even though the run
  itself still raises `RequestTimeoutError`.

---

## Guardrails

A guardrail is a named reference to a dynamically-resolved [Guardrails AI](https://guardrailsai.com/)
Hub validator — this codebase ships zero built-in checks. Resolution falls back from the
current PyPI-package convention to the older Hub CLI/registry mechanism when needed; an
unresolvable `validator_id` fails at startup, not at first use.

Three checkpoints, independently configured per agent: the incoming message
(`input_guardrails`), each tool result (`tool_output_guardrails`), and the final response
(`output_guardrails`). Each guardrail carries its own action — `block` (raise
`GuardrailBlockedError`), `redact` (substitute corrected content and continue), or `warn` (log
and continue unchanged). A validator that raises internally is always treated as a block
regardless of its configured action — a check that couldn't run can't tell whether its content
was safe. The tool-output checkpoint is the one asymmetric case: a block there never raises —
it becomes a tool-result error message the LLM can react to, rather than aborting the run.

A triggered finding's `reason` is free text from the validator, which can echo back an excerpt
of the very content it just caught. It's never included in `GuardrailBlockedError`'s own
message — that reaches the HTTP caller directly, and, from the tool-output checkpoint, the
LLM's own context — only the guardrail's name is. Logs get the full `reason` unconditionally,
since they never leave this process.

Some Hub validators call an LLM internally via a constructor argument named `llm_callable`;
when declared, that call is redirected through a registered `ILLM` instead of a provider
directly, sharing its retry/timeout/cost behavior with every other completion
(`adapters/llm_registry_provider.py`).

---

## Logging

Every log line — this app's own, uvicorn's, and litellm's — goes through one formatter and one
stream: both libraries' own default handlers are neutralized at startup so their records fall
through to root instead. This gives one consistent format, and lets every line — including
uvicorn's access log — carry this app's `request_id`/`trace_id`/`span_id` correlation fields via
`RunContextFilter`. `trace_id`/`span_id` are stamped only when the current span is both valid
and sampled, the same distinction `request_id` derivation makes, since a partial sampler can
hand back a structurally valid id for a span it never exports.

An exception's log severity follows OTel's own exception-recording guidance by span kind, not
one fixed level: a genuine unhandled failure in the root HTTP span logs at ERROR, an LLM-call
failure at WARNING, and a client-initiated cancellation at DEBUG. This mirrors the span-level
judgment: a `ClientError` or `CancelledError` never marks a span as an error either, since
neither is an operational problem.

Logs never leave this process — there's no log-shipping mechanism anywhere in this codebase —
so a caught exception's full text and any tool-execution failure always logs unconditionally,
even where the same content stays behind `capture_content_enabled()` on a span attribute or a
tool-result message. `duration_ms` and `exception_type` are passed via `extra=`, not just
interpolated into the message text, so they're queryable without parsing it. `JsonFormatter`'s
timestamp is an explicit UTC ISO-8601 string, not `logging.Formatter`'s locale-dependent default;
`TextFormatter` appends any `extra` field the fixed format string doesn't already render, so
nothing set via `extra=` silently vanishes in text mode.

---

## Tracing

`core/` and `api/` instrument against `opentelemetry-api` only — the same
`trace.get_tracer(__name__)` pattern every module already uses for `logging.getLogger(__name__)`
— so tracing is a genuine no-op with zero runtime cost when nothing configures it.
`opentelemetry-sdk` (the `TracerProvider`, sampler, exporter) is imported in exactly one place,
`core/factories/app.py`'s `configure_tracing()`.

`trace.set_tracer_provider()` silently no-ops after the first call in a process, so
`configure_tracing()` returns the provider it registered (or `None`) specifically so its caller
can tell later whether it actually won that race; `create_app()`'s shutdown hook only shuts down
the provider it registered if the global still points at that exact object. Without this check,
a second `create_app()` call in the same process (two app instances, or a test building more
than one) could shut down another instance's still-running provider, or double-shut-down a
shared one.

`console` and `endpoint` (OTLP) are independent toggles, not a choice between them — both can
export at once. Neither logging nor tracing ever writes to a file: this is a single-person,
directly-run deployment with no supervisor capturing stdout, so shell redirection covers the
rare case a file is wanted.

One root span opens per request (`api/request_context.py`'s `RequestIdMiddleware`,
`SpanKind.SERVER`); every other span nests under it in the order a run actually executes:

```
GET /sessions/{session_id}         (root — api/, RequestIdMiddleware; SpanKind.SERVER)
└─ invoke_agent <agent>             (core/services/agent_run.py)
   ├─ input_guardrails             (if configured)
   │  └─ guardrail <name>
   ├─ chat <model>                 (LLM call; SpanKind.CLIENT)
   ├─ execute_tool <name>          (one span per tool call; concurrent calls in the same
   ├─ execute_tool <name>           round are sibling spans with overlapping timestamps)
   │  └─ tool_output_guardrails
   │     └─ guardrail <name>
   ├─ chat <model>                 (further rounds, if the loop iterates again)
   ├─ output_guardrails
   │  └─ guardrail <name>
   └─ compaction                   (core/services/compaction.py — only when triggered)
      └─ chat <model>              (the summarizer's own call)
```

Span names follow the GenAI semantic conventions' `{operation} {name}` pattern (`chat <model>`,
`execute_tool <tool name>`, `invoke_agent <agent name>`); `guardrail <name>` matches the same
shape for consistency, even though guardrail spans aren't a GenAI-defined operation. `chat` and
`execute_tool` both stamp `gen_ai.conversation.id`/`gen_ai.agent.name` via one shared
`stamp_run_context()` helper (`core/tracing/__init__.py`) whenever a run is active, rather than
each reading `RunContext` and setting the pair by hand. `chat` opens inside each `ILLM`'s own
`complete()`, not at each call site, so every caller — the ReAct loop, compaction's summarizer,
a guardrail's `llm_callable` redirect — is covered by construction.

The root span follows OTel's stable HTTP semantic conventions (method, route, status code,
client/server address, protocol version, user agent), named `"{method} {route}"` once a route
matches. `url.query` is the one exception, gated behind `capture_content_enabled()` rather than
recorded unconditionally, since a query string can carry a token a misconfigured client
appended to any URL.

`request_id` is the root span's trace id, hex-formatted, whenever tracing is on and no caller
supplied an `X-Request-ID` header; with tracing off, it falls back to a fresh `uuid.uuid4().hex`
instead, since a no-op span's trace id is always zero. A caller-supplied, valid `X-Request-ID` is
recorded separately as `agent_core.client_request_id`, distinct from the span's own trace id.

Content — prompt/response text, tool arguments and results, a tool's own description — is never
attached to a span unless `TracingConfig.capture_content` is set; by default spans carry only
metadata (model, token counts, tool name, duration, finish reason). This applies uniformly
regardless of a value's source — a tool's own description is operator-written, not user input,
but it's gated the same as everything else, since the rule is "free-text content is gated, full
stop," not "gated only when plausibly sensitive." When enabled, message content is JSON-encoded
per the GenAI semantic conventions' `{"role", "parts", ...}` schema, since a span attribute can
only hold a scalar or a homogeneous array of scalars, never a native nested structure. `chat`
also carries `gen_ai.tool.definitions` under the same flag — the only way to see which tools were
available but never called, since an `execute_tool` span exists only for one actually invoked.

Attribute names reuse a real OTel/GenAI convention wherever one exists (`gen_ai.*`, `http.*`,
`url.*`, `error.type`); a concept with no convention of its own (a guardrail's action, a
compaction's before/after size) gets an `agent_core.*` prefix, since an unprefixed custom name
risks colliding with whatever a future semantic-convention release standardizes.

Every span that can genuinely fail sets `error.type` (a low-cardinality identifier) and, only
when `capture_content` is on, records the actual exception and status description via one shared
`record_gated_exception()` helper (`core/tracing/__init__.py`). This is deliberately not applied
to a guardrail's own `block` trigger or an `asyncio.CancelledError` — both are expected outcomes,
not operational failures, so neither should inflate an error-rate dashboard the way a genuine
unhandled exception would; the same judgment logging makes, above. Recording the same exception
at both its origin span (`chat`) and an ancestor (`invoke_agent`) is deliberate, not duplication
by accident — root-cause detail at the origin, failure visibility at the span an operator
actually watches — and each is logged exactly once, at the innermost frame that catches it.

The root HTTP span follows the same rule from the other direction: a 4xx is the client's fault,
so its status and `error.type` stay unset even when a registered exception handler converts the
failure into a response before it ever reaches the middleware that owns the span.

`configure_tracing()` pairs the console exporter with a synchronous `SimpleSpanProcessor` —
correct for its dev-debugging purpose, but blocking I/O in the request path — and the OTLP
exporter with a `BatchSpanProcessor` instead, batching and exporting off a background thread.

`service.name` defaults to `"agent-core"` only when the operator hasn't already set
`OTEL_SERVICE_NAME` — passing the default unconditionally would silently override a real
operator choice, since `Resource.create()` merges an explicit attribute over an env-detected one.

---

## Exception Hierarchy

All exceptions are subclasses of `AgentError`; `core/` raises only these, and adapters
re-raise third-party exceptions as the appropriate subclass at the boundary. `api/`'s request-id
middleware wraps the whole request in its own exception handler, ahead of Starlette's own
catch-all — so it, not the app's registered `Exception` handler, is what actually fires for a
genuinely unhandled exception during a request.

`ClientError` marks a deliberate, expected rejection, distinct from a genuine operational
failure like an `LLMError` or an unexpected bug — excluded from span error-tracking the same
way a 4xx is (see Tracing), and always mapped to a 4xx by the route handler that catches it.

---

## Naming Conventions

| Pattern | Usage |
|---------|-------|
| `BaseX` | Abstract base class |
| `IX` | Interface / Protocol in `core/protocols/` |
| `XStrategy` | Interchangeable algorithm implementations in `core/strategies/` |
| `XService` | Orchestration and use-case logic in `core/services/` |
| `XRegistry` | Runtime name-to-instance maps in `core/registries/` |
| `XFactory` | Object construction from config in `core/factories/`, when a class earns its keep (state, multiple methods). A single construction step is a plain function instead. |
| `XAdapter` | Concrete outbound adapter for an external system in `adapters/` |
| `XSessionStore` | Interchangeable session-history storage implementations in `core/session_stores/` |
| `XConfig` | Pydantic config model in `core/models/` |
| `XError` | Typed exception in `core/exceptions/` |
| `logging.getLogger(__name__)` | Standard logger (one per module) |
| `trace.get_tracer(__name__)` | Standard tracer (one per module) |
| Plain noun | Data-only `BaseModel` |
