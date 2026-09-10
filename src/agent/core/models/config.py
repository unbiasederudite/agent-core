"""Pydantic config models for application startup."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SamplingDefaults(BaseModel):
    """Shared sampling-default fields: temperature, top_p, and max output tokens."""

    model_config = ConfigDict(extra="forbid")

    temperature: float | None = Field(
        default=None, ge=0, description="Default sampling temperature, if set."
    )
    top_p: float | None = Field(
        default=None, ge=0, le=1, description="Default nucleus sampling value, if set."
    )
    max_tokens: int | None = Field(
        default=None, ge=1, description="Default max output tokens, if set."
    )


class LLMConfig(SamplingDefaults):
    """One entry in the startup LLM allow-list."""

    model: str = Field(
        description='litellm-format provider/model id, e.g. "anthropic/claude-sonnet-5".'
    )
    context_window: int | None = Field(
        default=None, ge=1, description="Overrides the model's looked-up context-window size."
    )
    num_retries: int = Field(
        default=2, ge=0, description="Retries for retriable failures before giving up."
    )
    timeout: float | None = Field(default=None, gt=0, description="Per-attempt timeout in seconds.")
    retry_base_delay: float = Field(
        default=1.0, gt=0, description="Delay before the first retry, in seconds."
    )
    retry_max_delay: float = Field(
        default=30.0, gt=0, description="Cap on delay between retries, in seconds."
    )
    retry_multiplier: float = Field(
        default=2.0, ge=1, description="Backoff multiplier applied to the delay after each retry."
    )
    max_concurrent_requests: int | None = Field(
        default=None, ge=1, description="Cap on concurrent in-flight calls to this model."
    )

    @model_validator(mode="after")
    def _max_delay_not_below_base(self) -> "LLMConfig":
        """Reject a retry_max_delay smaller than retry_base_delay.

        Raises:
            ValueError: if retry_max_delay is less than retry_base_delay.
        """
        if self.retry_max_delay < self.retry_base_delay:
            raise ValueError(
                f"retry_max_delay ({self.retry_max_delay}) must be >= "
                f"retry_base_delay ({self.retry_base_delay})"
            )
        return self


class LoggingConfig(BaseModel):
    """Logging configuration."""

    model_config = ConfigDict(extra="forbid")

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Minimum log level to emit."
    )
    format: Literal["text", "json"] = Field(default="text", description="Log output format.")


class ToolConfig(BaseModel):
    """One entry in the startup tool allow-list."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="The tool's lookup key, matching a code-level implementation.")


class StrategyConfig(BaseModel):
    """One entry in the startup strategy allow-list."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        description="The strategy's lookup key, matching a code-level implementation."
    )


class GuardrailConfig(BaseModel):
    """One entry in the startup guardrail allow-list."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="The guardrail's lookup key, referenced by agents.")
    validator_id: str = Field(
        description='Guardrails AI Hub validator id, e.g. "guardrails/detect_pii".'
    )
    validator_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword arguments passed to the resolved validator's constructor.",
    )
    action: Literal["block", "redact", "warn"] = Field(
        default="block", description="What happens when this guardrail triggers."
    )


class AgentConfig(SamplingDefaults):
    """One entry in the startup agent allow-list."""

    name: str = Field(description="The agent's lookup key.")
    system_prompt: str = Field(description="The agent's leading system prompt.")
    model: str = Field(description="The model used when a request doesn't override `model`.")
    strategy: str = Field(
        description="The reasoning strategy used when a request doesn't override `strategy`."
    )
    tools: list[str] = Field(
        default_factory=list, description="Tool names available to this agent by default."
    )
    input_guardrails: list[str] = Field(
        default_factory=list, description="Guardrails checked against the incoming message."
    )
    tool_output_guardrails: list[str] = Field(
        default_factory=list, description="Guardrails checked against each tool result."
    )
    output_guardrails: list[str] = Field(
        default_factory=list, description="Guardrails checked against the final response."
    )
    max_tool_iterations: int = Field(
        default=10, ge=1, description="Cap on tool-calling loop rounds."
    )
    max_tool_result_chars: int | None = Field(
        default=None, ge=1, description="Cap on a single tool result's length, in characters."
    )
    max_tool_calls_per_round: int | None = Field(
        default=None, ge=1, description="Cap on tool calls executed per LLM response."
    )
    max_tool_results_total_chars: int | None = Field(
        default=None,
        ge=1,
        description="Cap on combined tool-result length across a run, in characters.",
    )
    max_input_chars: int | None = Field(
        default=None, ge=1, description="Cap on a request's `message` length, in characters."
    )
    allowed_tools: list[str] | None = Field(
        default=None,
        description="Ceiling on tool names a request may specify. `None` means unrestricted; "
        "`[]` forbids all tools.",
    )
    allowed_models: list[str] | None = Field(
        default=None,
        description="Ceiling on model names a request may override to. `None` means unrestricted.",
    )
    allowed_strategies: list[str] | None = Field(
        default=None,
        description="Ceiling on strategy names a request may override to. `None` means "
        "unrestricted.",
    )
    max_request_seconds: float | None = Field(
        default=None, gt=0, description="Wall-clock budget for one agent run."
    )

    @model_validator(mode="after")
    def _defaults_within_ceilings(self) -> "AgentConfig":
        """Reject a default that falls outside this agent's own configured ceiling.

        Raises:
            ValueError: if a default falls outside its matching ceiling.
        """
        if self.allowed_tools is not None:
            outside = [name for name in self.tools if name not in self.allowed_tools]
            if outside:
                raise ValueError(f"tools {outside} are not in allowed_tools {self.allowed_tools}")
        if self.allowed_models is not None and self.model not in self.allowed_models:
            raise ValueError(f"model '{self.model}' is not in allowed_models {self.allowed_models}")
        if self.allowed_strategies is not None and self.strategy not in self.allowed_strategies:
            raise ValueError(
                f"strategy '{self.strategy}' is not in allowed_strategies {self.allowed_strategies}"
            )
        return self


class CompactionConfig(BaseModel):
    """Global settings for summarizing a session's older history once it crosses budget."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(description="Model used to generate summaries.")
    token_budget_pct: float = Field(
        default=0.8,
        gt=0,
        le=1,
        description="Fraction of the resolved model's max input tokens that triggers compaction.",
    )
    keep_recent_turns: int = Field(
        default=4, ge=0, description="Most recent turns kept verbatim, never summarized."
    )
    chunk_turns: int = Field(
        default=4, ge=1, description="Turns per chunk in the map-reduce fallback."
    )
    prompt: str = Field(
        default=(
            "Summarize the conversation above concisely. Preserve the task's goal, key "
            "decisions, established constraints, and any facts or results later turns may "
            "need to reference. Omit small talk and resolved intermediate steps."
        ),
        description=(
            "Appended as a final user-role message after the old messages being summarized."
        ),
    )


class SessionStoreConfig(BaseModel):
    """Session-history storage settings."""

    model_config = ConfigDict(extra="forbid")

    type: str = Field(
        default="in_memory",
        description="The session store's lookup key, matching a code-level implementation.",
    )


class TracingConfig(BaseModel):
    """Distributed tracing configuration."""

    model_config = ConfigDict(extra="forbid")

    console: bool = Field(default=False, description="Whether to export spans to the console.")
    endpoint: str | None = Field(
        default=None, description="OTLP collector endpoint, or None to skip OTLP export."
    )
    capture_content: bool = Field(
        default=False,
        description="Whether content text is attached to spans, not just metadata.",
    )


class AppConfig(BaseModel):
    """Root startup configuration, loaded once from a JSON file."""

    model_config = ConfigDict(extra="forbid")

    llms: list[LLMConfig] = Field(description="The allow-list of LLMs available to this process.")
    agents: list[AgentConfig] = Field(
        default_factory=list, description="The allow-list of agents available to this process."
    )
    tools: list[ToolConfig] = Field(
        default_factory=list, description="The allow-list of tools available to this process."
    )
    strategies: list[StrategyConfig] = Field(
        default_factory=list,
        description="The allow-list of reasoning strategies available to this process.",
    )
    guardrails: list[GuardrailConfig] = Field(
        default_factory=list,
        description="The allow-list of guardrails available to this process.",
    )
    base_prompt: str | None = Field(
        default=None, description="Prepended before every agent's own `system_prompt`."
    )
    compaction: CompactionConfig | None = Field(
        default=None, description="Compaction settings. Omitted disables compaction."
    )
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig, description="Logging configuration."
    )
    session_store: SessionStoreConfig = Field(
        default_factory=SessionStoreConfig, description="Session-history storage settings."
    )
    tracing: TracingConfig = Field(
        default_factory=TracingConfig, description="Distributed tracing configuration."
    )
    max_sessions: int | None = Field(
        default=None, ge=1, description="Cap on how many distinct sessions are kept at once."
    )
    max_concurrent_requests: int | None = Field(
        default=None,
        ge=1,
        description="Cap on requests running at once across every agent and model combined.",
    )
