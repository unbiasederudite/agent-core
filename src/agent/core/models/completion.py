"""LLM completion result model."""

from pydantic import BaseModel, Field

from agent.core.models.message import Message
from agent.core.models.usage import Usage


class Completion(BaseModel):
    """The result of one model completion call."""

    message: Message = Field(description="The assistant's reply.")
    usage: Usage = Field(description="Token usage for this completion.")
    finish_reason: str = Field(
        description='Why generation stopped (e.g. "stop", "length", "content_filter").'
    )
    response_id: str | None = Field(
        default=None, description="The provider's own identifier for this completion, if any."
    )
    response_model: str | None = Field(
        default=None,
        description="The exact model that generated this completion, if the provider reports "
        "one distinct from the requested model.",
    )
