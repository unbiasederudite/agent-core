import asyncio
import json
import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import litellm
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode

from agent.adapters import litellm as litellm_adapter
from agent.adapters.litellm import LiteLLMAdapter
from agent.core.exceptions import (
    LLMContextWindowExceededError,
    LLMError,
    LLMOverloadedError,
    LLMRateLimitedError,
    LLMTimeoutError,
)
from agent.core.models.config import LLMConfig
from agent.core.models.message import Message, ToolCall, ToolCallFunction
from agent.core.run_context import run_context
from agent.core.tracing import set_capture_content


def test_module_import_clears_litellm_verbose_logger_handler_and_enables_propagation():
    assert litellm.verbose_logger.handlers == []  # type: ignore[attr-defined]
    assert litellm.verbose_logger.propagate is True  # type: ignore[attr-defined]


class _FakeProviderError(Exception):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


def _fake_litellm_response(finish_reason: str = "stop") -> SimpleNamespace:
    return SimpleNamespace(
        id="chatcmpl-test123",
        model="gpt-4o-0613",
        choices=[
            SimpleNamespace(message=SimpleNamespace(content="hello!"), finish_reason=finish_reason)
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def _tool_exchange_history() -> list[Message]:
    """A turn whose assistant message carries `tool_calls`, answered by a role="tool" result."""
    return [
        Message(role="user", content="what time is it?"),
        Message(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="call_1", function=ToolCallFunction(name="get_current_time", arguments="{}")
                )
            ],
        ),
        Message(role="tool", tool_call_id="call_1", name="get_current_time", content="12:00"),
        Message(role="assistant", content="it is noon"),
    ]


_FOLDED_TOOL_EXCHANGE = (
    "[called tool 'get_current_time' with {}]\n"
    "[tool 'get_current_time' returned: 12:00]\n"
    "it is noon"
)

_A_TOOL_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {"name": "get_current_time", "description": "...", "parameters": {}},
    }
]


def test_from_config_maps_every_field_onto_the_adapter():
    config = LLMConfig(
        model="openai/gpt-4o",
        temperature=0.2,
        top_p=0.9,
        max_tokens=512,
        context_window=32000,
        num_retries=5,
        timeout=10.0,
        retry_base_delay=0.5,
        retry_max_delay=8.0,
        retry_multiplier=3.0,
        max_concurrent_requests=4,
    )

    adapter = LiteLLMAdapter.from_config(config)

    assert adapter._model == "openai/gpt-4o"
    assert adapter._temperature == 0.2
    assert adapter._top_p == 0.9
    assert adapter._max_tokens == 512
    assert adapter._context_window == 32000
    assert adapter._num_retries == 5
    assert adapter._timeout == 10.0
    assert adapter._retry_base_delay == 0.5
    assert adapter._retry_max_delay == 8.0
    assert adapter._retry_multiplier == 3.0
    assert adapter._max_concurrent == 4


def test_from_config_given_only_model_uses_llm_config_defaults():
    adapter = LiteLLMAdapter.from_config(LLMConfig(model="openai/gpt-4o"))

    assert adapter._temperature is None
    assert adapter._num_retries == 2
    assert adapter._retry_base_delay == 1.0
    assert adapter._max_concurrent is None


async def test_complete_given_successful_response_returns_completion(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("litellm.acompletion", AsyncMock(return_value=_fake_litellm_response()))
    adapter = LiteLLMAdapter(model="openai/gpt-4o")

    completion = await adapter.complete([Message(role="user", content="hi")])

    assert completion.message.role == "assistant"
    assert completion.message.content == "hello!"
    assert completion.usage.total_tokens == 15
    assert completion.finish_reason == "stop"


async def test_complete_opens_a_chat_span_with_usage_and_operation_attributes(
    monkeypatch: pytest.MonkeyPatch,
):
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(litellm_adapter, "tracer", provider.get_tracer("test"))
    monkeypatch.setattr("litellm.acompletion", AsyncMock(return_value=_fake_litellm_response()))
    adapter = LiteLLMAdapter(model="openai/gpt-4o")

    await adapter.complete([Message(role="user", content="hi")])

    [span] = [s for s in exporter.get_finished_spans() if s.name == "chat openai/gpt-4o"]
    assert span.kind == SpanKind.CLIENT
    assert span.attributes["gen_ai.operation.name"] == "chat"
    assert span.attributes["gen_ai.request.model"] == "openai/gpt-4o"
    assert span.attributes["gen_ai.usage.input_tokens"] == 10
    assert span.attributes["gen_ai.usage.output_tokens"] == 5
    assert span.attributes["gen_ai.response.finish_reasons"] == ("stop",)
    assert span.attributes["gen_ai.response.id"] == "chatcmpl-test123"
    assert span.attributes["gen_ai.response.model"] == "gpt-4o-0613"


async def test_complete_given_sampling_params_chat_span_has_request_attributes(
    monkeypatch: pytest.MonkeyPatch,
):
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(litellm_adapter, "tracer", provider.get_tracer("test"))
    monkeypatch.setattr("litellm.acompletion", AsyncMock(return_value=_fake_litellm_response()))
    adapter = LiteLLMAdapter(model="openai/gpt-4o")

    await adapter.complete(
        [Message(role="user", content="hi")], temperature=0.5, top_p=0.9, max_tokens=100
    )

    [span] = [s for s in exporter.get_finished_spans() if s.name == "chat openai/gpt-4o"]
    assert span.attributes["gen_ai.request.temperature"] == 0.5
    assert span.attributes["gen_ai.request.top_p"] == 0.9
    assert span.attributes["gen_ai.request.max_tokens"] == 100


async def test_complete_given_no_sampling_params_chat_span_omits_request_attributes(
    monkeypatch: pytest.MonkeyPatch,
):
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(litellm_adapter, "tracer", provider.get_tracer("test"))
    monkeypatch.setattr("litellm.acompletion", AsyncMock(return_value=_fake_litellm_response()))
    adapter = LiteLLMAdapter(model="openai/gpt-4o")

    await adapter.complete([Message(role="user", content="hi")])

    [span] = [s for s in exporter.get_finished_spans() if s.name == "chat openai/gpt-4o"]
    assert "gen_ai.request.temperature" not in span.attributes
    assert "gen_ai.request.top_p" not in span.attributes
    assert "gen_ai.request.max_tokens" not in span.attributes


async def test_complete_given_capture_content_disabled_chat_span_has_no_message_content(
    monkeypatch: pytest.MonkeyPatch,
):
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(litellm_adapter, "tracer", provider.get_tracer("test"))
    set_capture_content(False)
    monkeypatch.setattr("litellm.acompletion", AsyncMock(return_value=_fake_litellm_response()))
    adapter = LiteLLMAdapter(model="openai/gpt-4o")

    await adapter.complete([Message(role="user", content="hi")])

    [span] = [s for s in exporter.get_finished_spans() if s.name == "chat openai/gpt-4o"]
    assert not any("content" in key for key in span.attributes)


async def test_complete_given_capture_content_enabled_chat_span_has_output_content(
    monkeypatch: pytest.MonkeyPatch,
):
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(litellm_adapter, "tracer", provider.get_tracer("test"))
    set_capture_content(True)
    try:
        monkeypatch.setattr("litellm.acompletion", AsyncMock(return_value=_fake_litellm_response()))
        adapter = LiteLLMAdapter(model="openai/gpt-4o")

        await adapter.complete([Message(role="user", content="hi")])

        [span] = [s for s in exporter.get_finished_spans() if s.name == "chat openai/gpt-4o"]
        assert json.loads(span.attributes["gen_ai.output.messages"]) == [
            {
                "role": "assistant",
                "parts": [{"type": "text", "content": "hello!"}],
                "finish_reason": "stop",
            }
        ]
    finally:
        set_capture_content(False)


async def test_complete_given_capture_content_enabled_chat_span_has_input_messages(
    monkeypatch: pytest.MonkeyPatch,
):
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(litellm_adapter, "tracer", provider.get_tracer("test"))
    set_capture_content(True)
    try:
        monkeypatch.setattr("litellm.acompletion", AsyncMock(return_value=_fake_litellm_response()))
        adapter = LiteLLMAdapter(model="openai/gpt-4o")

        await adapter.complete(_tool_exchange_history())

        [span] = [s for s in exporter.get_finished_spans() if s.name == "chat openai/gpt-4o"]
        assert json.loads(span.attributes["gen_ai.input.messages"]) == [
            {"role": "user", "parts": [{"type": "text", "content": "what time is it?"}]},
            {
                "role": "assistant",
                "parts": [
                    {
                        "type": "tool_call",
                        "id": "call_1",
                        "name": "get_current_time",
                        "arguments": {},
                    }
                ],
            },
            {
                "role": "tool",
                "parts": [{"type": "tool_call_response", "id": "call_1", "result": "12:00"}],
            },
            {"role": "assistant", "parts": [{"type": "text", "content": "it is noon"}]},
        ]
    finally:
        set_capture_content(False)


async def test_complete_given_oversized_history_input_messages_keeps_the_most_recent_part(
    monkeypatch: pytest.MonkeyPatch,
):
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(litellm_adapter, "tracer", provider.get_tracer("test"))
    monkeypatch.setattr(litellm_adapter, "_MAX_MESSAGE_ATTRIBUTE_CHARS", 40)
    set_capture_content(True)
    try:
        monkeypatch.setattr("litellm.acompletion", AsyncMock(return_value=_fake_litellm_response()))
        adapter = LiteLLMAdapter(model="openai/gpt-4o")

        await adapter.complete([Message(role="user", content=f"message {i}") for i in range(20)])

        [span] = [s for s in exporter.get_finished_spans() if s.name == "chat openai/gpt-4o"]
        input_messages = span.attributes["gen_ai.input.messages"]
        assert "earlier characters" in input_messages
        assert "message 19" in input_messages
        assert "message 0" not in input_messages
    finally:
        set_capture_content(False)


async def test_complete_given_capture_content_enabled_and_tools_chat_span_has_tool_definitions(
    monkeypatch: pytest.MonkeyPatch,
):
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(litellm_adapter, "tracer", provider.get_tracer("test"))
    set_capture_content(True)
    try:
        monkeypatch.setattr("litellm.acompletion", AsyncMock(return_value=_fake_litellm_response()))
        adapter = LiteLLMAdapter(model="openai/gpt-4o")
        tool_schema = {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Get the current time.",
                "parameters": {"type": "object", "properties": {}},
            },
        }

        await adapter.complete([Message(role="user", content="hi")], tools=[tool_schema])

        [span] = [s for s in exporter.get_finished_spans() if s.name == "chat openai/gpt-4o"]
        assert json.loads(span.attributes["gen_ai.tool.definitions"]) == [
            {
                "type": "function",
                "name": "get_current_time",
                "description": "Get the current time.",
                "parameters": {"type": "object", "properties": {}},
            }
        ]
    finally:
        set_capture_content(False)


async def test_complete_given_capture_content_enabled_and_no_tools_chat_span_omits_tool_definitions(
    monkeypatch: pytest.MonkeyPatch,
):
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(litellm_adapter, "tracer", provider.get_tracer("test"))
    set_capture_content(True)
    try:
        monkeypatch.setattr("litellm.acompletion", AsyncMock(return_value=_fake_litellm_response()))
        adapter = LiteLLMAdapter(model="openai/gpt-4o")

        await adapter.complete([Message(role="user", content="hi")])

        [span] = [s for s in exporter.get_finished_spans() if s.name == "chat openai/gpt-4o"]
        assert "gen_ai.tool.definitions" not in span.attributes
    finally:
        set_capture_content(False)


async def test_complete_given_concurrency_cap_reached_chat_span_records_the_error(
    monkeypatch: pytest.MonkeyPatch,
):
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(litellm_adapter, "tracer", provider.get_tracer("test"))
    adapter = LiteLLMAdapter(model="openai/gpt-4o", max_concurrent_requests=1)
    adapter._in_flight = 1

    with pytest.raises(LLMOverloadedError):
        await adapter.complete([Message(role="user", content="hi")])

    [span] = [s for s in exporter.get_finished_spans() if s.name == "chat openai/gpt-4o"]
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes["error.type"] == "LLMOverloadedError"


async def test_complete_given_provider_error_chat_span_omits_description_by_default(
    monkeypatch: pytest.MonkeyPatch,
):
    # str(exc) here can carry provider-echoed request content (a classified LLMError wraps the
    # raw provider error), so it's gated like gen_ai content elsewhere on this same span —
    # error.type alone (already content-free) identifies the failure by default. The span must
    # also disable record_exception/set_status_on_exception defaults, or OTel's own SDK would
    # auto-record the raw message on span exit regardless of this gate.
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(litellm_adapter, "tracer", provider.get_tracer("test"))
    monkeypatch.setattr("litellm.acompletion", AsyncMock(side_effect=RuntimeError("provider down")))
    adapter = LiteLLMAdapter(model="openai/gpt-4o", num_retries=0)

    with pytest.raises(LLMError):
        await adapter.complete([Message(role="user", content="hi")])

    [span] = [s for s in exporter.get_finished_spans() if s.name == "chat openai/gpt-4o"]
    assert span.attributes["error.type"] == "LLMError"
    assert span.status.status_code == StatusCode.ERROR
    assert span.status.description is None
    assert len(span.events) == 0


async def test_complete_given_provider_error_chat_span_records_it_when_capturing_content(
    monkeypatch: pytest.MonkeyPatch,
):
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(litellm_adapter, "tracer", provider.get_tracer("test"))
    monkeypatch.setattr("litellm.acompletion", AsyncMock(side_effect=RuntimeError("provider down")))
    adapter = LiteLLMAdapter(model="openai/gpt-4o", num_retries=0)
    set_capture_content(True)
    try:
        with pytest.raises(LLMError):
            await adapter.complete([Message(role="user", content="hi")])
    finally:
        set_capture_content(False)

    [span] = [s for s in exporter.get_finished_spans() if s.name == "chat openai/gpt-4o"]
    assert span.attributes["error.type"] == "LLMError"
    assert span.status.status_code == StatusCode.ERROR
    assert span.status.description is not None
    assert "provider down" in span.status.description
    assert len(span.events) == 1


async def test_complete_opens_a_chat_span_with_provider_name(monkeypatch: pytest.MonkeyPatch):
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(litellm_adapter, "tracer", provider.get_tracer("test"))
    monkeypatch.setattr("litellm.acompletion", AsyncMock(return_value=_fake_litellm_response()))
    adapter = LiteLLMAdapter(model="openai/gpt-4o")

    await adapter.complete([Message(role="user", content="hi")])

    [span] = [s for s in exporter.get_finished_spans() if s.name == "chat openai/gpt-4o"]
    assert span.attributes["gen_ai.provider.name"] == "openai"


async def test_complete_given_active_run_context_chat_span_has_conversation_id(
    monkeypatch: pytest.MonkeyPatch,
):
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(litellm_adapter, "tracer", provider.get_tracer("test"))
    monkeypatch.setattr("litellm.acompletion", AsyncMock(return_value=_fake_litellm_response()))
    adapter = LiteLLMAdapter(model="openai/gpt-4o")

    with run_context("researcher", "sess-1"):
        await adapter.complete([Message(role="user", content="hi")])

    [span] = [s for s in exporter.get_finished_spans() if s.name == "chat openai/gpt-4o"]
    assert span.attributes["gen_ai.conversation.id"] == "sess-1"
    assert span.attributes["gen_ai.agent.name"] == "researcher"


async def test_complete_given_no_active_run_chat_span_has_no_conversation_id(
    monkeypatch: pytest.MonkeyPatch,
):
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(litellm_adapter, "tracer", provider.get_tracer("test"))
    monkeypatch.setattr("litellm.acompletion", AsyncMock(return_value=_fake_litellm_response()))
    adapter = LiteLLMAdapter(model="openai/gpt-4o")

    await adapter.complete([Message(role="user", content="hi")])

    [span] = [s for s in exporter.get_finished_spans() if s.name == "chat openai/gpt-4o"]
    assert "gen_ai.conversation.id" not in span.attributes
    assert "gen_ai.agent.name" not in span.attributes


async def test_complete_given_length_finish_reason_is_not_hardcoded_to_stop(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "litellm.acompletion",
        AsyncMock(return_value=_fake_litellm_response(finish_reason="length")),
    )
    adapter = LiteLLMAdapter(model="openai/gpt-4o")

    completion = await adapter.complete([Message(role="user", content="hi")])

    assert completion.finish_reason == "length"


async def test_complete_given_litellm_raises_wraps_as_llm_error(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("litellm.acompletion", AsyncMock(side_effect=RuntimeError("provider down")))
    adapter = LiteLLMAdapter(model="openai/gpt-4o", num_retries=0)

    with pytest.raises(LLMError):
        await adapter.complete([Message(role="user", content="hi")])


async def test_complete_given_empty_choices_raises_llm_error(monkeypatch: pytest.MonkeyPatch):
    response = SimpleNamespace(
        choices=[],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    mock_acompletion = AsyncMock(return_value=response)
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    adapter = LiteLLMAdapter(model="openai/gpt-4o", num_retries=3)

    with pytest.raises(LLMError):
        await adapter.complete([Message(role="user", content="hi")])

    # A malformed response is a model/prompt issue, not a transient failure — it must
    # not be retried (an empty `choices` list has no .status_code, so the retry loop's
    # own classification would otherwise treat it as one and burn the whole budget).
    assert mock_acompletion.call_count == 1


async def test_complete_given_none_content_raises_llm_error(monkeypatch: pytest.MonkeyPatch):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    mock_acompletion = AsyncMock(return_value=response)
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    adapter = LiteLLMAdapter(model="openai/gpt-4o", num_retries=3)

    with pytest.raises(LLMError):
        await adapter.complete([Message(role="user", content="hi")])

    assert mock_acompletion.call_count == 1


async def test_complete_given_empty_string_content_and_no_tool_calls_raises_llm_error(
    monkeypatch: pytest.MonkeyPatch,
):
    # An empty-string (not None) reply with no tool_calls would otherwise be stored verbatim
    # and later rejected by Anthropic/Bedrock ("text content blocks must be non-empty").
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=""))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    monkeypatch.setattr("litellm.acompletion", AsyncMock(return_value=response))
    adapter = LiteLLMAdapter(model="openai/gpt-4o")

    with pytest.raises(LLMError):
        await adapter.complete([Message(role="user", content="hi")])


async def test_complete_given_empty_string_content_alongside_tool_calls_normalizes_to_none(
    monkeypatch: pytest.MonkeyPatch,
):
    # An empty-string (not None) reply accompanying real tool_calls must not be stored as-is:
    # `exclude_none=True` keeps an empty string on a later replay, unlike `None`, which could
    # trip the same empty-text-block rejection this file's `content=None`-only guard prevents.
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="",
                    tool_calls=[
                        SimpleNamespace(
                            id="call_1",
                            function=SimpleNamespace(name="get_current_time", arguments="{}"),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    monkeypatch.setattr("litellm.acompletion", AsyncMock(return_value=response))
    adapter = LiteLLMAdapter(model="openai/gpt-4o")

    completion = await adapter.complete([Message(role="user", content="hi")])

    assert completion.message.content is None


async def test_complete_given_status_code_429_raises_llm_rate_limited_error(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "litellm.acompletion",
        AsyncMock(side_effect=_FakeProviderError("rate limited", status_code=429)),
    )
    adapter = LiteLLMAdapter(model="openai/gpt-4o", num_retries=0)

    with pytest.raises(LLMRateLimitedError):
        await adapter.complete([Message(role="user", content="hi")])


async def test_complete_given_status_code_408_raises_llm_timeout_error(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "litellm.acompletion",
        AsyncMock(side_effect=_FakeProviderError("timed out", status_code=408)),
    )
    adapter = LiteLLMAdapter(model="openai/gpt-4o", num_retries=0)

    with pytest.raises(LLMTimeoutError):
        await adapter.complete([Message(role="user", content="hi")])


async def test_complete_given_no_params_and_no_defaults_omits_sampling_kwargs(
    monkeypatch: pytest.MonkeyPatch,
):
    mock_acompletion = AsyncMock(return_value=_fake_litellm_response())
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    adapter = LiteLLMAdapter(model="openai/gpt-4o")

    await adapter.complete([Message(role="user", content="hi")])

    _, kwargs = mock_acompletion.call_args
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs
    assert "max_completion_tokens" not in kwargs


async def test_complete_given_constructed_defaults_forwards_them(
    monkeypatch: pytest.MonkeyPatch,
):
    mock_acompletion = AsyncMock(return_value=_fake_litellm_response())
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    adapter = LiteLLMAdapter(model="openai/gpt-4o", temperature=0.2, top_p=0.9, max_tokens=512)

    await adapter.complete([Message(role="user", content="hi")])

    _, kwargs = mock_acompletion.call_args
    assert kwargs["temperature"] == 0.2
    assert kwargs["top_p"] == 0.9
    assert kwargs["max_completion_tokens"] == 512


async def test_complete_given_call_value_overrides_constructed_default(
    monkeypatch: pytest.MonkeyPatch,
):
    mock_acompletion = AsyncMock(return_value=_fake_litellm_response())
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    adapter = LiteLLMAdapter(model="openai/gpt-4o", temperature=0.2)

    await adapter.complete([Message(role="user", content="hi")], temperature=0.9)

    _, kwargs = mock_acompletion.call_args
    assert kwargs["temperature"] == 0.9


async def test_complete_given_call_value_zero_overrides_constructed_default(
    monkeypatch: pytest.MonkeyPatch,
):
    mock_acompletion = AsyncMock(return_value=_fake_litellm_response())
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    adapter = LiteLLMAdapter(model="openai/gpt-4o", temperature=0.5)

    await adapter.complete([Message(role="user", content="hi")], temperature=0.0)

    _, kwargs = mock_acompletion.call_args
    assert kwargs["temperature"] == 0.0


async def test_complete_given_tool_calls_in_response_maps_them_and_allows_none_content(
    monkeypatch: pytest.MonkeyPatch,
):
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="call_1",
                            function=SimpleNamespace(name="get_current_time", arguments="{}"),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    monkeypatch.setattr("litellm.acompletion", AsyncMock(return_value=response))
    adapter = LiteLLMAdapter(model="openai/gpt-4o")

    completion = await adapter.complete([Message(role="user", content="hi")])

    assert completion.message.content is None
    assert completion.message.tool_calls == [
        ToolCall(id="call_1", function=ToolCallFunction(name="get_current_time", arguments="{}"))
    ]
    assert completion.finish_reason == "tool_calls"


async def test_complete_given_tools_param_forwards_to_litellm(monkeypatch: pytest.MonkeyPatch):
    mock_acompletion = AsyncMock(return_value=_fake_litellm_response())
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    adapter = LiteLLMAdapter(model="openai/gpt-4o")
    tools = [
        {
            "type": "function",
            "function": {"name": "get_current_time", "description": "...", "parameters": {}},
        }
    ]

    await adapter.complete([Message(role="user", content="hi")], tools=tools)

    _, kwargs = mock_acompletion.call_args
    assert kwargs["tools"] == tools


async def test_complete_given_no_tools_omits_tools_kwarg(monkeypatch: pytest.MonkeyPatch):
    mock_acompletion = AsyncMock(return_value=_fake_litellm_response())
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    adapter = LiteLLMAdapter(model="openai/gpt-4o")

    await adapter.complete([Message(role="user", content="hi")])

    _, kwargs = mock_acompletion.call_args
    assert "tools" not in kwargs


async def test_complete_excludes_none_fields_from_outbound_messages(
    monkeypatch: pytest.MonkeyPatch,
):
    mock_acompletion = AsyncMock(return_value=_fake_litellm_response())
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    adapter = LiteLLMAdapter(model="openai/gpt-4o")

    await adapter.complete([Message(role="system", content="hi")])

    _, kwargs = mock_acompletion.call_args
    assert kwargs["messages"][0] == {"role": "system", "content": "hi"}


async def test_complete_given_outbound_tool_calls_serializes_nested_wire_shape(
    monkeypatch: pytest.MonkeyPatch,
):
    mock_acompletion = AsyncMock(return_value=_fake_litellm_response())
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    adapter = LiteLLMAdapter(model="openai/gpt-4o")
    message = Message(
        role="assistant",
        tool_calls=[
            ToolCall(
                id="call_1", function=ToolCallFunction(name="get_current_time", arguments="{}")
            )
        ],
    )

    # Declares tools: `tool_calls` only ever reach the wire on a tool-declaring request, since
    # a no-tools request has them folded into plain text first.
    await adapter.complete([message, Message(role="user", content="hi")], tools=_A_TOOL_SCHEMA)

    _, kwargs = mock_acompletion.call_args
    assert kwargs["messages"][0] == {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_current_time", "arguments": "{}"},
            }
        ],
    }


async def test_complete_given_no_tools_flattens_tool_exchanges_out_of_outbound_messages(
    monkeypatch: pytest.MonkeyPatch,
):
    # Bedrock's Converse API rejects toolUse/toolResult blocks in a request with no toolConfig,
    # even when they only replay an earlier exchange — so a no-tools call must never send them.
    mock_acompletion = AsyncMock(return_value=_fake_litellm_response())
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    adapter = LiteLLMAdapter(model="openai/gpt-4o")

    await adapter.complete(_tool_exchange_history())

    _, kwargs = mock_acompletion.call_args
    assert kwargs["messages"] == [
        {"role": "user", "content": "what time is it?"},
        {"role": "assistant", "content": _FOLDED_TOOL_EXCHANGE},
    ]


async def test_complete_given_empty_tools_list_flattens_tool_exchanges_too(
    monkeypatch: pytest.MonkeyPatch,
):
    # `[]` is as much "no tools declared" as `None` is — the adapter must not rely on callers
    # having normalized one to the other.
    mock_acompletion = AsyncMock(return_value=_fake_litellm_response())
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    adapter = LiteLLMAdapter(model="openai/gpt-4o")

    await adapter.complete(_tool_exchange_history(), tools=[])

    _, kwargs = mock_acompletion.call_args
    assert kwargs["messages"] == [
        {"role": "user", "content": "what time is it?"},
        {"role": "assistant", "content": _FOLDED_TOOL_EXCHANGE},
    ]


async def test_complete_given_no_tools_history_ending_on_tool_result_flattens_to_trailing_assistant(
    monkeypatch: pytest.MonkeyPatch,
):
    # Unlike `_tool_exchange_history()` (which ends on a real assistant reply), history cut off
    # mid-tool-use has nothing after the last tool result — `flush()` folds it into a brand-new
    # synthetic assistant message with nothing after it. `ReactStrategy`'s forced-final call is
    # the one caller that can reach this shape (see its own docstring); this proves what actually
    # goes out on the wire in that case, which is why that caller appends a trailing instruction.
    mock_acompletion = AsyncMock(return_value=_fake_litellm_response())
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    adapter = LiteLLMAdapter(model="openai/gpt-4o")
    history = [
        Message(role="user", content="what time is it?"),
        Message(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="call_1", function=ToolCallFunction(name="get_current_time", arguments="{}")
                )
            ],
        ),
        Message(role="tool", tool_call_id="call_1", name="get_current_time", content="12:00"),
    ]

    await adapter.complete(history)

    _, kwargs = mock_acompletion.call_args
    assert kwargs["messages"] == [
        {"role": "user", "content": "what time is it?"},
        {
            "role": "assistant",
            "content": (
                "[called tool 'get_current_time' with {}]\n"
                "[tool 'get_current_time' returned: 12:00]"
            ),
        },
    ]


async def test_complete_given_tools_declared_sends_the_tool_exchange_unflattened(
    monkeypatch: pytest.MonkeyPatch,
):
    # The regression that matters: flattening is for no-tools requests only. A tool-declaring
    # request must still replay the genuine toolUse/toolResult blocks, or the provider loses
    # the pairing between a tool call and its result.
    mock_acompletion = AsyncMock(return_value=_fake_litellm_response())
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    adapter = LiteLLMAdapter(model="openai/gpt-4o")

    await adapter.complete(_tool_exchange_history(), tools=_A_TOOL_SCHEMA)

    _, kwargs = mock_acompletion.call_args
    assert kwargs["messages"] == [
        {"role": "user", "content": "what time is it?"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_current_time", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "content": "12:00",
            "tool_call_id": "call_1",
            "name": "get_current_time",
        },
        {"role": "assistant", "content": "it is noon"},
    ]


async def test_complete_given_no_tools_and_no_tool_content_leaves_messages_unchanged(
    monkeypatch: pytest.MonkeyPatch,
):
    # Making the flattening unconditional for no-tools calls costs nothing in the common case.
    mock_acompletion = AsyncMock(return_value=_fake_litellm_response())
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    adapter = LiteLLMAdapter(model="openai/gpt-4o")

    await adapter.complete(
        [Message(role="user", content="hi"), Message(role="assistant", content="hello")]
    )

    _, kwargs = mock_acompletion.call_args
    assert kwargs["messages"] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


async def test_complete_given_context_window_exceeded_raises_llm_context_window_exceeded_error(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "litellm.acompletion",
        AsyncMock(
            side_effect=litellm.ContextWindowExceededError(
                message="context exceeded", model="openai/gpt-4o", llm_provider="openai"
            )
        ),
    )
    adapter = LiteLLMAdapter(model="openai/gpt-4o")

    with pytest.raises(LLMContextWindowExceededError):
        await adapter.complete([Message(role="user", content="hi")])


def test_max_input_tokens_given_known_model_returns_it(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("litellm.get_model_info", lambda model: {"max_input_tokens": 128000})
    adapter = LiteLLMAdapter(model="openai/gpt-4o")

    assert adapter.max_input_tokens() == 128000


def test_max_input_tokens_given_unrecognized_model_raises_llm_error(
    monkeypatch: pytest.MonkeyPatch,
):
    def _raise(model: str) -> dict[str, int]:
        raise ValueError("model isn't mapped")

    monkeypatch.setattr("litellm.get_model_info", _raise)
    adapter = LiteLLMAdapter(model="openai/does-not-exist")

    with pytest.raises(LLMError):
        adapter.max_input_tokens()


def test_max_input_tokens_given_missing_field_raises_llm_error(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("litellm.get_model_info", lambda model: {})
    adapter = LiteLLMAdapter(model="openai/gpt-4o")

    with pytest.raises(LLMError):
        adapter.max_input_tokens()


def test_max_input_tokens_given_context_window_override_returns_it_without_litellm_lookup(
    monkeypatch: pytest.MonkeyPatch,
):
    mock_get_model_info = Mock(side_effect=AssertionError("should not be called"))
    monkeypatch.setattr("litellm.get_model_info", mock_get_model_info)
    adapter = LiteLLMAdapter(model="self-hosted/my-model", context_window=32000)

    assert adapter.max_input_tokens() == 32000
    mock_get_model_info.assert_not_called()


def test_max_input_tokens_given_context_window_override_takes_precedence_over_real_litellm_data(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("litellm.get_model_info", lambda model: {"max_input_tokens": 128000})
    adapter = LiteLLMAdapter(model="openai/gpt-4o", context_window=999)

    assert adapter.max_input_tokens() == 999


async def test_complete_given_status_code_401_raises_immediately_without_retry(
    monkeypatch: pytest.MonkeyPatch,
):
    mock_acompletion = AsyncMock(side_effect=_FakeProviderError("bad key", status_code=401))
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    adapter = LiteLLMAdapter(model="openai/gpt-4o", num_retries=3)

    with pytest.raises(LLMError):
        await adapter.complete([Message(role="user", content="hi")])

    assert mock_acompletion.call_count == 1


async def test_complete_given_status_code_400_raises_immediately_without_retry(
    monkeypatch: pytest.MonkeyPatch,
):
    mock_acompletion = AsyncMock(side_effect=_FakeProviderError("bad request", status_code=400))
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    adapter = LiteLLMAdapter(model="openai/gpt-4o", num_retries=3)

    with pytest.raises(LLMError):
        await adapter.complete([Message(role="user", content="hi")])

    assert mock_acompletion.call_count == 1


async def test_complete_given_status_code_404_raises_immediately_without_retry(
    monkeypatch: pytest.MonkeyPatch,
):
    mock_acompletion = AsyncMock(side_effect=_FakeProviderError("not found", status_code=404))
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    adapter = LiteLLMAdapter(model="openai/gpt-4o", num_retries=3)

    with pytest.raises(LLMError):
        await adapter.complete([Message(role="user", content="hi")])

    assert mock_acompletion.call_count == 1


async def test_complete_given_status_code_429_retries_up_to_num_retries_then_raises(
    monkeypatch: pytest.MonkeyPatch,
):
    mock_acompletion = AsyncMock(side_effect=_FakeProviderError("rate limited", status_code=429))
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    monkeypatch.setattr("agent.adapters.litellm.asyncio.sleep", AsyncMock())
    adapter = LiteLLMAdapter(model="openai/gpt-4o", num_retries=2)

    with pytest.raises(LLMRateLimitedError):
        await adapter.complete([Message(role="user", content="hi")])

    assert mock_acompletion.call_count == 3  # the initial attempt plus 2 retries


async def test_complete_given_status_code_500_is_retriable(monkeypatch: pytest.MonkeyPatch):
    mock_acompletion = AsyncMock(side_effect=_FakeProviderError("server error", status_code=500))
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    monkeypatch.setattr("agent.adapters.litellm.asyncio.sleep", AsyncMock())
    adapter = LiteLLMAdapter(model="openai/gpt-4o", num_retries=1)

    with pytest.raises(LLMError):
        await adapter.complete([Message(role="user", content="hi")])

    assert mock_acompletion.call_count == 2


async def test_complete_given_connection_error_with_no_status_code_is_retriable(
    monkeypatch: pytest.MonkeyPatch,
):
    mock_acompletion = AsyncMock(
        side_effect=[RuntimeError("connection reset"), _fake_litellm_response()]
    )
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    monkeypatch.setattr("agent.adapters.litellm.asyncio.sleep", AsyncMock())
    adapter = LiteLLMAdapter(model="openai/gpt-4o", num_retries=1)

    completion = await adapter.complete([Message(role="user", content="hi")])

    assert completion.message.content == "hello!"
    assert mock_acompletion.call_count == 2


async def test_complete_given_transient_failure_then_success_returns_normally(
    monkeypatch: pytest.MonkeyPatch,
):
    mock_acompletion = AsyncMock(
        side_effect=[_FakeProviderError("rate limited", status_code=429), _fake_litellm_response()]
    )
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    monkeypatch.setattr("agent.adapters.litellm.asyncio.sleep", AsyncMock())
    adapter = LiteLLMAdapter(model="openai/gpt-4o", num_retries=2)

    completion = await adapter.complete([Message(role="user", content="hi")])

    assert completion.message.content == "hello!"
    assert mock_acompletion.call_count == 2


async def test_complete_given_retries_uses_configured_exponential_backoff(
    monkeypatch: pytest.MonkeyPatch,
):
    mock_acompletion = AsyncMock(side_effect=_FakeProviderError("rate limited", status_code=429))
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    mock_sleep = AsyncMock()
    monkeypatch.setattr("agent.adapters.litellm.asyncio.sleep", mock_sleep)
    adapter = LiteLLMAdapter(
        model="openai/gpt-4o",
        num_retries=3,
        retry_base_delay=1.0,
        retry_multiplier=2.0,
        retry_max_delay=30.0,
    )

    with pytest.raises(LLMRateLimitedError):
        await adapter.complete([Message(role="user", content="hi")])

    delays = [call.args[0] for call in mock_sleep.call_args_list]
    expected = [1.0, 2.0, 4.0]
    assert len(delays) == len(expected)
    for actual, base in zip(delays, expected, strict=True):
        assert base * 0.5 <= actual <= base


async def test_complete_given_retry_delay_caps_at_max_delay(monkeypatch: pytest.MonkeyPatch):
    mock_acompletion = AsyncMock(side_effect=_FakeProviderError("rate limited", status_code=429))
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    mock_sleep = AsyncMock()
    monkeypatch.setattr("agent.adapters.litellm.asyncio.sleep", mock_sleep)
    adapter = LiteLLMAdapter(
        model="openai/gpt-4o",
        num_retries=4,
        retry_base_delay=10.0,
        retry_multiplier=3.0,
        retry_max_delay=15.0,
    )

    with pytest.raises(LLMRateLimitedError):
        await adapter.complete([Message(role="user", content="hi")])

    delays = [call.args[0] for call in mock_sleep.call_args_list]
    expected = [10.0, 15.0, 15.0, 15.0]
    assert len(delays) == len(expected)
    for actual, base in zip(delays, expected, strict=True):
        assert base * 0.5 <= actual <= base


async def test_complete_given_malformed_response_raises_without_retry(
    monkeypatch: pytest.MonkeyPatch,
):
    # A message with neither content nor tool_calls is our own raised LLMError — a
    # persistently malformed response is a model/prompt issue, not a transient failure.
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    mock_acompletion = AsyncMock(return_value=response)
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    adapter = LiteLLMAdapter(model="openai/gpt-4o", num_retries=3)

    with pytest.raises(LLMError):
        await adapter.complete([Message(role="user", content="hi")])

    assert mock_acompletion.call_count == 1


async def test_complete_given_unexpected_response_shape_raises_llm_error_without_retry(
    monkeypatch: pytest.MonkeyPatch,
):
    # response.usage is missing entirely — an AttributeError while constructing
    # Completion, not the two already-covered shape problems (empty choices, no
    # content/tool_calls). Same "malformed response, don't retry" contract should apply.
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hi"), finish_reason="stop")]
    )
    mock_acompletion = AsyncMock(return_value=response)
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    adapter = LiteLLMAdapter(model="openai/gpt-4o", num_retries=3)

    with pytest.raises(LLMError):
        await adapter.complete([Message(role="user", content="hi")])

    assert mock_acompletion.call_count == 1


async def test_complete_never_passes_num_retries_kwarg_to_litellm(monkeypatch: pytest.MonkeyPatch):
    mock_acompletion = AsyncMock(return_value=_fake_litellm_response())
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    adapter = LiteLLMAdapter(model="openai/gpt-4o", num_retries=5)

    await adapter.complete([Message(role="user", content="hi")])

    _, kwargs = mock_acompletion.call_args
    assert "num_retries" not in kwargs


async def test_complete_given_timeout_set_forwards_it(monkeypatch: pytest.MonkeyPatch):
    mock_acompletion = AsyncMock(return_value=_fake_litellm_response())
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    adapter = LiteLLMAdapter(model="openai/gpt-4o", timeout=15.0)

    await adapter.complete([Message(role="user", content="hi")])

    _, kwargs = mock_acompletion.call_args
    assert kwargs["timeout"] == 15.0


async def test_complete_given_no_timeout_omits_it(monkeypatch: pytest.MonkeyPatch):
    mock_acompletion = AsyncMock(return_value=_fake_litellm_response())
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    adapter = LiteLLMAdapter(model="openai/gpt-4o")

    await adapter.complete([Message(role="user", content="hi")])

    _, kwargs = mock_acompletion.call_args
    assert "timeout" not in kwargs


async def test_complete_given_retry_logs_warning_with_structured_exception_fields(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    caplog.set_level(logging.WARNING, logger="agent.adapters.litellm")
    mock_acompletion = AsyncMock(
        side_effect=[_FakeProviderError("rate limited", status_code=429), _fake_litellm_response()]
    )
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    monkeypatch.setattr("agent.adapters.litellm.asyncio.sleep", AsyncMock())
    adapter = LiteLLMAdapter(model="openai/gpt-4o", num_retries=1)

    await adapter.complete([Message(role="user", content="hi")])

    [record] = caplog.records
    assert record.exception_type == "_FakeProviderError"
    assert record.status_code == 429
    assert record.attempt == 1


async def test_complete_given_exhausted_retries_logs_warning_with_traceback(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    # WARN, not ERROR: per OTel's exception-recording guidance, a CLIENT-kind span's
    # exceptions (the whole call happens inside the `chat` span) are WARN severity.
    caplog.set_level(logging.WARNING, logger="agent.adapters.litellm")
    mock_acompletion = AsyncMock(side_effect=_FakeProviderError("rate limited", status_code=429))
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    monkeypatch.setattr("agent.adapters.litellm.asyncio.sleep", AsyncMock())
    adapter = LiteLLMAdapter(model="openai/gpt-4o", num_retries=1)

    with pytest.raises(LLMRateLimitedError):
        await adapter.complete([Message(role="user", content="hi")])

    [record] = [r for r in caplog.records if r.levelno == logging.WARNING and r.exc_info]
    assert record.exc_info is not None
    assert record.status_code == 429


async def test_complete_given_at_capacity_raises_immediately_without_calling_litellm(
    monkeypatch: pytest.MonkeyPatch,
):
    mock_acompletion = AsyncMock(return_value=_fake_litellm_response())
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    adapter = LiteLLMAdapter(model="openai/gpt-4o", max_concurrent_requests=1)
    adapter._in_flight = 1

    with pytest.raises(LLMOverloadedError):
        await adapter.complete([Message(role="user", content="hi")])

    mock_acompletion.assert_not_called()


async def test_complete_given_max_concurrent_requests_none_never_rejects(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("litellm.acompletion", AsyncMock(return_value=_fake_litellm_response()))
    adapter = LiteLLMAdapter(model="openai/gpt-4o")
    adapter._in_flight = 1000

    completion = await adapter.complete([Message(role="user", content="hi")])

    assert completion.message.content == "hello!"


async def test_complete_given_two_concurrent_calls_below_cap_both_proceed(
    monkeypatch: pytest.MonkeyPatch,
):
    release = asyncio.Event()

    async def _slow_acompletion(**kwargs: object) -> SimpleNamespace:
        await release.wait()
        return _fake_litellm_response()

    monkeypatch.setattr("litellm.acompletion", _slow_acompletion)
    adapter = LiteLLMAdapter(model="openai/gpt-4o", max_concurrent_requests=2)

    task1 = asyncio.create_task(adapter.complete([Message(role="user", content="a")]))
    task2 = asyncio.create_task(adapter.complete([Message(role="user", content="b")]))
    await asyncio.sleep(0)
    assert adapter._in_flight == 2
    release.set()
    result1, result2 = await asyncio.gather(task1, task2)

    assert result1.message.content == "hello!"
    assert result2.message.content == "hello!"


async def test_complete_given_third_call_while_two_in_flight_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
):
    release = asyncio.Event()

    async def _slow_acompletion(**kwargs: object) -> SimpleNamespace:
        await release.wait()
        return _fake_litellm_response()

    monkeypatch.setattr("litellm.acompletion", _slow_acompletion)
    adapter = LiteLLMAdapter(model="openai/gpt-4o", max_concurrent_requests=2)
    task1 = asyncio.create_task(adapter.complete([Message(role="user", content="a")]))
    task2 = asyncio.create_task(adapter.complete([Message(role="user", content="b")]))
    await asyncio.sleep(0)

    with pytest.raises(LLMOverloadedError):
        await adapter.complete([Message(role="user", content="c")])

    release.set()
    await asyncio.gather(task1, task2)


async def test_complete_in_flight_count_returns_to_zero_after_success(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("litellm.acompletion", AsyncMock(return_value=_fake_litellm_response()))
    adapter = LiteLLMAdapter(model="openai/gpt-4o", max_concurrent_requests=1)

    await adapter.complete([Message(role="user", content="hi")])

    assert adapter._in_flight == 0


async def test_complete_in_flight_count_returns_to_zero_after_exhausted_retries(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "litellm.acompletion",
        AsyncMock(side_effect=_FakeProviderError("bad key", status_code=401)),
    )
    adapter = LiteLLMAdapter(model="openai/gpt-4o", max_concurrent_requests=1, num_retries=0)

    with pytest.raises(LLMError):
        await adapter.complete([Message(role="user", content="hi")])

    assert adapter._in_flight == 0


async def test_complete_given_at_capacity_logs_warning_with_structured_fields(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    caplog.set_level(logging.WARNING, logger="agent.adapters.litellm")
    adapter = LiteLLMAdapter(model="openai/gpt-4o", max_concurrent_requests=1)
    adapter._in_flight = 1

    with pytest.raises(LLMOverloadedError):
        await adapter.complete([Message(role="user", content="hi")])

    [record] = caplog.records
    assert record.model == "openai/gpt-4o"
    assert record.in_flight == 1
    assert record.max_concurrent == 1


async def test_complete_given_retriable_failure_applies_jitter_to_the_delay(
    monkeypatch: pytest.MonkeyPatch,
):
    mock_acompletion = AsyncMock(
        side_effect=[_FakeProviderError("rate limited", status_code=429), _fake_litellm_response()]
    )
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    mock_sleep = AsyncMock()
    monkeypatch.setattr("agent.adapters.litellm.asyncio.sleep", mock_sleep)
    adapter = LiteLLMAdapter(
        model="openai/gpt-4o", num_retries=1, retry_base_delay=10.0, retry_multiplier=2.0
    )

    await adapter.complete([Message(role="user", content="hi")])

    delay = mock_sleep.call_args_list[0].args[0]
    assert 5.0 <= delay < 10.0  # base_delay=10.0 * uniform(0.5, 1.0), never the bare 10.0


async def test_complete_given_known_model_sets_cost_usd(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("litellm.acompletion", AsyncMock(return_value=_fake_litellm_response()))
    monkeypatch.setattr("litellm.completion_cost", lambda **kwargs: 0.00123)
    adapter = LiteLLMAdapter("openai/gpt-4o")

    completion = await adapter.complete([Message(role="user", content="hi")])

    assert completion.usage.cost_usd == pytest.approx(0.00123)


async def test_complete_given_unpriced_model_sets_cost_usd_to_none(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("litellm.acompletion", AsyncMock(return_value=_fake_litellm_response()))

    def _raise(**kwargs: Any) -> float:
        raise Exception("model not mapped yet")

    monkeypatch.setattr("litellm.completion_cost", _raise)
    adapter = LiteLLMAdapter("openai/gpt-4o")

    completion = await adapter.complete([Message(role="user", content="hi")])

    assert completion.usage.cost_usd is None
