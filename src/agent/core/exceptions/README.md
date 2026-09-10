# exceptions

The full exception hierarchy, rooted at `AgentError`. `core/` raises only `AgentError` subclasses; adapters catch third-party exceptions at the boundary and re-raise as one of these. `ClientError` marks the subset that are deliberate, expected rejections (not found, not allowed, busy, blocked, too large) rather than operational failures.

## Contents

- `__init__.py` — `AgentError` (the root), `ClientError` (deliberate-rejection marker), `ConfigError`, `RequestTimeoutError`, `LLMError` with its subclasses `LLMRateLimitedError`, `LLMTimeoutError`, `LLMOverloadedError`, `LLMContextWindowExceededError`, and `CompactionExhaustedError`; and, under `ClientError`: `LLMNotFoundError`, `AgentNotFoundError`, `ToolNotFoundError`, `StrategyNotFoundError`, `SessionNotFoundError`, `InputTooLargeError`, `SessionBusyError`, `ToolNotAllowedError`, `ModelNotAllowedError`, `StrategyNotAllowedError`, `GuardrailNotFoundError`, and `GuardrailBlockedError`
