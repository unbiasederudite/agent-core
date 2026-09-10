"""The full exception hierarchy, rooted at AgentError."""


class AgentError(Exception):
    """Root of the agent-core exception hierarchy."""


class ClientError(AgentError):
    """A deliberate, expected rejection based on the request or current state."""


class ConfigError(AgentError):
    """Raised when startup configuration is invalid."""


class LLMNotFoundError(ClientError):
    """Raised when a requested LLM name is not registered."""


class AgentNotFoundError(ClientError):
    """Raised when a requested agent name is not registered."""


class ToolNotFoundError(ClientError):
    """Raised when a requested tool name is not registered."""


class StrategyNotFoundError(ClientError):
    """Raised when a requested reasoning strategy name is not registered."""


class GuardrailNotFoundError(ClientError):
    """Raised when a requested guardrail name is not registered."""


class SessionNotFoundError(ClientError):
    """Raised when a requested (agent, session_id) pair is not registered."""


class LLMError(AgentError):
    """Raised when an outbound LLM call fails."""


class LLMRateLimitedError(LLMError):
    """Raised when the upstream LLM provider rate-limited the request."""


class LLMTimeoutError(LLMError):
    """Raised when the upstream LLM provider request timed out."""


class LLMOverloadedError(LLMError):
    """Raised when a model's configured `max_concurrent_requests` is already reached."""


class LLMContextWindowExceededError(LLMError):
    """Raised when the provider rejects a request for exceeding the model's context window."""


class CompactionExhaustedError(LLMContextWindowExceededError):
    """Raised when compaction was tried and the request still doesn't fit."""


class InputTooLargeError(ClientError):
    """Raised when a request's `message` exceeds the agent's configured `max_input_chars`."""


class SessionBusyError(ClientError):
    """Raised when a request targets a session another operation is currently using."""


class RequestTimeoutError(AgentError):
    """Raised when a single request exceeds its agent's configured `max_request_seconds`."""


class ToolNotAllowedError(ClientError):
    """Raised when a request names a tool that is registered but not permitted for this agent."""


class ModelNotAllowedError(ClientError):
    """Raised when a request names a model that is registered but not permitted for this agent."""


class StrategyNotAllowedError(ClientError):
    """Raised when a request names a strategy that is registered but not permitted for the agent."""


class GuardrailBlockedError(ClientError):
    """Raised when a block-action guardrail triggers on the input or final-output checkpoint."""
