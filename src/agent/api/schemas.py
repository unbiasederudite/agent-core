"""Backend-native request/response schemas for the agent-run and registry-listing routes."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent.core.models.message import Message
from agent.core.models.usage import Usage


class AgentRunRequest(BaseModel):
    """Request body for POST /v1/agents/{agent_name}."""

    model_config = ConfigDict(extra="forbid")

    message: str = Field(description="The user's message to send to the agent.")
    model: str | None = Field(
        default=None,
        description="litellm-format provider/model string overriding the agent's configured model.",
    )
    strategy: str | None = Field(
        default=None,
        description="Reasoning strategy name overriding the agent's configured strategy.",
    )
    temperature: float | None = Field(
        default=None,
        ge=0,
        description="Sampling temperature override.",
    )
    top_p: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Nucleus sampling override.",
    )
    max_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Max output tokens override.",
    )
    tools: list[str] | None = Field(
        default=None,
        description="Tool names to offer the LLM, overriding the agent's configured tools.",
    )
    session_id: str | None = Field(
        default=None,
        description="Session id to continue an existing conversation.",
    )


class AgentRunResponse(BaseModel):
    """Response body for POST /v1/agents/{agent_name}."""

    model: str = Field(
        description="The litellm-format provider/model string that ran this request."
    )
    message: Message = Field(description="The generated reply message.")
    usage: Usage = Field(description="Token usage for this run's own turn.")
    supporting_usage: Usage = Field(
        description="Token usage from this run's own supporting LLM calls — a guardrail "
        "check, a compaction summary — separate from the turn above."
    )
    finish_reason: str = Field(description="Why generation stopped.")
    session_id: str = Field(description="Session id this response belongs to.")


class SessionHistoryResponse(BaseModel):
    """Response body for GET /v1/agents/{agent_name}/sessions/{session_id}."""

    session_id: str = Field(description="The session this history belongs to.")
    messages: list[Message] = Field(description="Every stored message, in order.")


class SessionUsageResponse(BaseModel):
    """Response body for GET /v1/agents/{agent_name}/sessions/{session_id}/usage."""

    session_id: str = Field(description="The session this usage belongs to.")
    cumulative: Usage | None = Field(
        description="Token and cost totals summed across every run against this session, "
        "or None if this session's own usage record was evicted to bound memory."
    )
    context_tokens: int | None = Field(
        description="Token footprint of the full stored history as of the last run, "
        "or None if this session's own footprint record was evicted to bound memory."
    )


class AgentUsageResponse(BaseModel):
    """Response body for GET /v1/agents/{agent_name}/usage."""

    agent: str = Field(description="The agent this usage belongs to.")
    cumulative: Usage = Field(description="Token and cost totals across all this agent's sessions.")


class AgentSummary(BaseModel):
    """One entry in the GET /v1/agents listing."""

    name: str = Field(description="The agent's lookup key.")
    model: str = Field(description="The agent's default LLM.")
    strategy: str = Field(description="The agent's default reasoning strategy.")
    tools: list[str] = Field(description="Tool names available to this agent by default.")


class ToolSummary(BaseModel):
    """One entry in the GET /v1/tools listing."""

    name: str = Field(description="The tool's lookup key.")
    description: str = Field(description="Human-readable description of what the tool does.")
    parameters: dict[str, Any] = Field(description="JSON schema for the tool's call arguments.")
