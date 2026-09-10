# Config Reference

Every field `config.json` accepts, by section. `required` fields have no default. See the
root [README](README.md#running) for how `config.json` fits into running the process, and
[Guardrails](README.md#guardrails) for how to find and install a validator referenced under
`guardrails[]`.

## Top level

| Field | Default | Description |
|---|---|---|
| `llms` | required | The allow-list of LLMs available to this process. |
| `agents` | `[]` | The allow-list of agents available to this process. |
| `tools` | `[]` | The allow-list of tools available to this process. |
| `strategies` | `[]` | The allow-list of reasoning strategies available to this process. |
| `guardrails` | `[]` | The allow-list of guardrails available to this process. |
| `base_prompt` | `null` | Prepended before every agent's own `system_prompt`. |
| `compaction` | `null` | Compaction settings. Omitted disables compaction. |
| `logging` | defaults below | Logging configuration. |
| `session_store` | defaults below | Session-history storage settings. |
| `tracing` | defaults below | Distributed tracing configuration. |
| `max_sessions` | `null` | Cap on how many distinct sessions are kept at once. |
| `max_concurrent_requests` | `null` | Cap on requests running at once across every agent and model combined. |

## `llms[]`

| Field | Default | Description |
|---|---|---|
| `model` | required | litellm-format provider/model id, e.g. `"anthropic/claude-sonnet-5"`. |
| `temperature` | `null` | Default sampling temperature, if set. |
| `top_p` | `null` | Default nucleus sampling value, if set. |
| `max_tokens` | `null` | Default max output tokens, if set. |
| `context_window` | `null` | Overrides the model's looked-up context-window size. |
| `num_retries` | `2` | Retries for retriable failures before giving up. |
| `timeout` | `null` | Per-attempt timeout in seconds. |
| `retry_base_delay` | `1.0` | Delay before the first retry, in seconds. |
| `retry_max_delay` | `30.0` | Cap on delay between retries, in seconds. |
| `retry_multiplier` | `2.0` | Backoff multiplier applied to the delay after each retry. |
| `max_concurrent_requests` | `null` | Cap on concurrent in-flight calls to this model. |

## `tools[]`

| Field | Default | Description |
|---|---|---|
| `name` | required | The tool's lookup key, matching a code-level implementation. |

## `strategies[]`

| Field | Default | Description |
|---|---|---|
| `name` | required | The strategy's lookup key, matching a code-level implementation. |

## `guardrails[]`

| Field | Default | Description |
|---|---|---|
| `name` | required | The guardrail's lookup key, referenced by agents. |
| `validator_id` | required | Guardrails AI Hub validator id, e.g. `"guardrails/detect_pii"`. |
| `validator_params` | `{}` | Keyword arguments passed to the resolved validator's constructor. |
| `action` | `"block"` | What happens when this guardrail triggers: `block`, `redact`, or `warn`. |

## `agents[]`

| Field | Default | Description |
|---|---|---|
| `name` | required | The agent's lookup key. |
| `system_prompt` | required | The agent's leading system prompt. |
| `model` | required | The model used when a request doesn't override `model`. |
| `strategy` | required | The reasoning strategy used when a request doesn't override `strategy`. |
| `temperature` | `null` | Default sampling temperature, if set. |
| `top_p` | `null` | Default nucleus sampling value, if set. |
| `max_tokens` | `null` | Default max output tokens, if set. |
| `tools` | `[]` | Tool names available to this agent by default. |
| `input_guardrails` | `[]` | Guardrails checked against the incoming message. |
| `tool_output_guardrails` | `[]` | Guardrails checked against each tool result. |
| `output_guardrails` | `[]` | Guardrails checked against the final response. |
| `max_tool_iterations` | `10` | Cap on tool-calling loop rounds. |
| `max_tool_result_chars` | `null` | Cap on a single tool result's length, in characters. |
| `max_tool_calls_per_round` | `null` | Cap on tool calls executed per LLM response. |
| `max_tool_results_total_chars` | `null` | Cap on combined tool-result length across a run, in characters. |
| `max_input_chars` | `null` | Cap on a request's `message` length, in characters. |
| `allowed_tools` | `null` | Ceiling on tool names a request may specify. `null` means unrestricted; `[]` forbids all tools. |
| `allowed_models` | `null` | Ceiling on model names a request may override to. `null` means unrestricted. |
| `allowed_strategies` | `null` | Ceiling on strategy names a request may override to. `null` means unrestricted. |
| `max_request_seconds` | `null` | Wall-clock budget for one agent run. |

## `compaction`

Optional — omit entirely to disable compaction.

| Field | Default | Description |
|---|---|---|
| `model` | required | Model used to generate summaries. |
| `token_budget_pct` | `0.8` | Fraction of the resolved model's max input tokens that triggers compaction. |
| `keep_recent_turns` | `4` | Most recent turns kept verbatim, never summarized. |
| `chunk_turns` | `4` | Turns per chunk in the map-reduce fallback. |
| `prompt` | (built-in default) | Appended as a final user-role message after the old messages being summarized. |

## `logging`

| Field | Default | Description |
|---|---|---|
| `level` | `"INFO"` | Minimum log level to emit: `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL`. |
| `format` | `"text"` | Log output format: `text` or `json`. |

## `session_store`

| Field | Default | Description |
|---|---|---|
| `type` | `"in_memory"` | The session store's lookup key, matching a code-level implementation. |

## `tracing`

Every field is off by default — omit `tracing` entirely to keep the process untraced.

| Field | Default | Description |
|---|---|---|
| `console` | `false` | Whether to export spans to the console as JSON Lines. |
| `endpoint` | `null` | OTLP HTTP collector endpoint, or `null` to skip OTLP export. |
| `capture_content` | `false` | Whether content text is attached to spans, not just metadata. |
