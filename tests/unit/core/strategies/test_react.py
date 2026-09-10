"""Tests for ReactStrategy — the ReAct tool-calling loop."""

import asyncio
import json
import logging
from typing import Any

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from pydantic import BaseModel, ConfigDict, field_validator

from agent.core.models.completion import Completion
from agent.core.models.guardrail import GuardrailFinding
from agent.core.models.message import Message, ToolCall, ToolCallFunction
from agent.core.models.usage import Usage
from agent.core.protocols.itool import ITool
from agent.core.run_context import run_context
from agent.core.strategies import react
from agent.core.strategies.react import ReactStrategy
from agent.core.tracing import set_capture_content


class _FakeLLM:
    """Returns queued completions in order, one per call; records every call's args."""

    def __init__(self, completions: list[Completion]) -> None:
        self._completions = list(completions)
        self.calls: list[dict[str, Any]] = []
        self.model = "test-model"

    async def complete(
        self,
        messages: list[Message],
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> Completion:
        self.calls.append(
            {
                "messages": list(messages),
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "tools": tools,
            }
        )
        return self._completions[len(self.calls) - 1]


class _EchoParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: str


class _EchoTool:
    name = "echo"
    description = "Echoes its input."
    parameters_model: type[BaseModel] = _EchoParams

    async def execute(self, **kwargs: Any) -> str:
        return str(kwargs["value"])


class _EmptyParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _RaisingTool:
    name = "boom"
    description = "Always raises."
    parameters_model: type[BaseModel] = _EmptyParams

    async def execute(self, **kwargs: Any) -> str:
        raise RuntimeError("tool exploded")


class _FakeGuardrail:
    """Returns a fixed finding for every check() call."""

    def __init__(self, name: str, action: str, finding: GuardrailFinding) -> None:
        self.name = name
        self.action = action
        self._finding = finding

    async def check(self, content: str) -> GuardrailFinding:
        return self._finding


class _ConcurrentTool:
    """Records the max number of overlapping in-flight executions."""

    name = "slow"
    description = "Sleeps briefly to prove concurrent execution."
    parameters_model: type[BaseModel] = _EmptyParams

    def __init__(self, counter: list[int], max_seen: list[int]) -> None:
        self._counter = counter
        self._max_seen = max_seen

    async def execute(self, **kwargs: Any) -> str:
        self._counter[0] += 1
        self._max_seen[0] = max(self._max_seen[0], self._counter[0])
        await asyncio.sleep(0.01)
        self._counter[0] -= 1
        return "done"


def _usage(n: int = 1) -> Usage:
    return Usage(prompt_tokens=n, completion_tokens=n, total_tokens=2 * n)


def _final_completion(content: str = "final answer") -> Completion:
    return Completion(
        message=Message(role="assistant", content=content), usage=_usage(), finish_reason="stop"
    )


def _tool_call_completion(calls: list[ToolCall]) -> Completion:
    return Completion(
        message=Message(role="assistant", content=None, tool_calls=calls),
        usage=_usage(),
        finish_reason="tool_calls",
    )


def _call(id_: str, name: str, arguments: str) -> ToolCall:
    return ToolCall(id=id_, function=ToolCallFunction(name=name, arguments=arguments))


async def test_run_given_no_tool_calls_returns_after_one_llm_call():
    llm = _FakeLLM([_final_completion()])
    strategy = ReactStrategy()

    turn = await strategy.run([Message(role="user", content="hi")], llm, {}, max_iterations=10)

    assert len(llm.calls) == 1
    assert turn.message.content == "final answer"
    assert turn.finish_reason == "stop"


async def test_run_given_no_tool_calls_turn_messages_is_just_the_final_answer():
    llm = _FakeLLM([_final_completion()])
    strategy = ReactStrategy()

    turn = await strategy.run([Message(role="user", content="hi")], llm, {}, max_iterations=10)

    assert turn.messages == [Message(role="assistant", content="final answer")]


async def test_run_given_one_tool_call_executes_it_and_calls_llm_again():
    tools: dict[str, ITool] = {"echo": _EchoTool()}
    llm = _FakeLLM(
        [
            _tool_call_completion([_call("call_1", "echo", '{"value": "hi"}')]),
            _final_completion("you said hi"),
        ]
    )
    strategy = ReactStrategy()

    turn = await strategy.run(
        [Message(role="user", content="echo hi")], llm, tools, max_iterations=10
    )

    assert len(llm.calls) == 2
    assert turn.message.content == "you said hi"
    second_call_messages = llm.calls[1]["messages"]
    assert second_call_messages[-1] == Message(
        role="tool", tool_call_id="call_1", name="echo", content="hi"
    )
    assert second_call_messages[-2].tool_calls == [_call("call_1", "echo", '{"value": "hi"}')]


async def test_run_given_tools_declares_schema_derived_from_parameters_model():
    tools: dict[str, ITool] = {"echo": _EchoTool()}
    llm = _FakeLLM([_final_completion()])
    strategy = ReactStrategy()

    await strategy.run([Message(role="user", content="hi")], llm, tools, max_iterations=10)

    assert llm.calls[0]["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "Echoes its input.",
                "parameters": _EchoParams.model_json_schema(),
            },
        }
    ]


async def test_run_given_one_tool_call_turn_messages_is_the_full_generated_delta():
    tools: dict[str, ITool] = {"echo": _EchoTool()}
    llm = _FakeLLM(
        [
            _tool_call_completion([_call("call_1", "echo", '{"value": "hi"}')]),
            _final_completion("you said hi"),
        ]
    )
    strategy = ReactStrategy()

    turn = await strategy.run(
        [Message(role="user", content="echo hi")], llm, tools, max_iterations=10
    )

    assert turn.messages == [
        Message(
            role="assistant",
            content=None,
            tool_calls=[_call("call_1", "echo", '{"value": "hi"}')],
        ),
        Message(role="tool", tool_call_id="call_1", name="echo", content="hi"),
        Message(role="assistant", content="you said hi"),
    ]


async def test_run_given_multiple_tool_calls_executes_them_concurrently():
    counter = [0]
    max_seen = [0]
    tools: dict[str, ITool] = {"slow": _ConcurrentTool(counter, max_seen)}
    llm = _FakeLLM(
        [
            _tool_call_completion([_call("call_1", "slow", "{}"), _call("call_2", "slow", "{}")]),
            _final_completion(),
        ]
    )
    strategy = ReactStrategy()

    await strategy.run([Message(role="user", content="go")], llm, tools, max_iterations=10)

    assert max_seen[0] == 2


async def test_run_given_multiple_tool_calls_returns_one_result_message_per_call_in_order():
    tools: dict[str, ITool] = {"echo": _EchoTool()}
    llm = _FakeLLM(
        [
            _tool_call_completion(
                [
                    _call("call_1", "echo", '{"value": "a"}'),
                    _call("call_2", "echo", '{"value": "b"}'),
                ]
            ),
            _final_completion(),
        ]
    )
    strategy = ReactStrategy()

    await strategy.run([Message(role="user", content="go")], llm, tools, max_iterations=10)

    result_messages = llm.calls[1]["messages"][-2:]
    assert result_messages[0] == Message(
        role="tool", tool_call_id="call_1", name="echo", content="a"
    )
    assert result_messages[1] == Message(
        role="tool", tool_call_id="call_2", name="echo", content="b"
    )


async def test_run_given_tool_call_names_an_unoffered_tool_returns_error_content_and_continues():
    llm = _FakeLLM([_tool_call_completion([_call("call_1", "missing", "{}")]), _final_completion()])
    strategy = ReactStrategy()

    turn = await strategy.run([Message(role="user", content="go")], llm, {}, max_iterations=10)

    result_message = llm.calls[1]["messages"][-1]
    assert result_message.tool_call_id == "call_1"
    assert result_message.content == "Error: tool 'missing' was not offered for this call"
    assert turn.message.content == "final answer"


async def test_run_given_tool_raises_returns_error_content_and_continues():
    tools: dict[str, ITool] = {"boom": _RaisingTool()}
    llm = _FakeLLM([_tool_call_completion([_call("call_1", "boom", "{}")]), _final_completion()])
    strategy = ReactStrategy()

    turn = await strategy.run([Message(role="user", content="go")], llm, tools, max_iterations=10)

    result_message = llm.calls[1]["messages"][-1]
    assert result_message.content == "Error: tool 'boom' failed: tool exploded"
    assert turn.message.content == "final answer"


async def test_run_given_malformed_json_arguments_returns_error_content_without_invoking_tool():
    tools: dict[str, ITool] = {"echo": _EchoTool()}
    llm = _FakeLLM(
        [_tool_call_completion([_call("call_1", "echo", "{not json")]), _final_completion()]
    )
    strategy = ReactStrategy()

    turn = await strategy.run([Message(role="user", content="go")], llm, tools, max_iterations=10)

    result_message = llm.calls[1]["messages"][-1]
    assert result_message.content is not None
    assert result_message.content.startswith("Error:")
    assert turn.message.content == "final answer"


async def test_run_given_non_object_json_arguments_returns_error_content_without_invoking_tool():
    tools: dict[str, ITool] = {"echo": _EchoTool()}
    llm = _FakeLLM(
        [_tool_call_completion([_call("call_1", "echo", "[1, 2]")]), _final_completion()]
    )
    strategy = ReactStrategy()

    turn = await strategy.run([Message(role="user", content="go")], llm, tools, max_iterations=10)

    result_message = llm.calls[1]["messages"][-1]
    assert result_message.content == "Error: arguments must be a JSON object"
    assert turn.message.content == "final answer"


async def test_run_given_max_iterations_exhausted_forces_one_final_call_without_tools():
    tools: dict[str, ITool] = {"echo": _EchoTool()}
    always_calls_tool = _tool_call_completion([_call("call_1", "echo", '{"value": "x"}')])
    llm = _FakeLLM([always_calls_tool, always_calls_tool, _final_completion("gave up")])
    strategy = ReactStrategy()

    turn = await strategy.run([Message(role="user", content="go")], llm, tools, max_iterations=2)

    assert len(llm.calls) == 3
    assert llm.calls[2]["tools"] is None
    assert turn.message.content == "gave up"
    assert turn.finish_reason == "stop"


async def test_run_given_max_iterations_exhausted_final_call_passes_messages_through_unmodified():
    # The forced final call declares no tools, so what actually reaches the provider may not
    # carry toolUse/toolResult blocks — but folding them out is the `ILLM` implementation's
    # contractual job, proven against the real outbound payload in
    # `tests/integration/adapters/test_litellm.py`. This strategy doesn't pre-flatten anything
    # itself, and a fake LLM here could never prove the provider-safety property anyway. It does
    # append one scoped instruction message to this one outbound request (see `ReactStrategy`'s
    # own docstring note on why) — everything before that is untouched.
    tools: dict[str, ITool] = {"echo": _EchoTool()}
    always_calls_tool = _tool_call_completion([_call("call_1", "echo", '{"value": "x"}')])
    llm = _FakeLLM([always_calls_tool, always_calls_tool, _final_completion("gave up")])
    strategy = ReactStrategy()

    await strategy.run([Message(role="user", content="go")], llm, tools, max_iterations=2)

    requested = Message(
        role="assistant", content=None, tool_calls=[_call("call_1", "echo", '{"value": "x"}')]
    )
    returned = Message(role="tool", tool_call_id="call_1", name="echo", content="x")
    assert llm.calls[2]["messages"] == [
        Message(role="user", content="go"),
        requested,
        returned,
        requested,
        returned,
        Message(
            role="user",
            content="No further tool calls are available. Provide your final answer now.",
        ),
    ]


async def test_run_given_max_iterations_exhausted_turn_messages_keeps_the_real_tool_exchange():
    # What gets returned (and stored as session history) keeps the genuine tool-call/tool-result
    # messages — the strategy's own bookkeeping is never rewritten for any one call's needs.
    tools: dict[str, ITool] = {"echo": _EchoTool()}
    always_calls_tool = _tool_call_completion([_call("call_1", "echo", '{"value": "x"}')])
    llm = _FakeLLM([always_calls_tool, always_calls_tool, _final_completion("gave up")])
    strategy = ReactStrategy()

    turn = await strategy.run([Message(role="user", content="go")], llm, tools, max_iterations=2)

    assert [m.role for m in turn.messages] == [
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]
    assert turn.messages[0].tool_calls == [_call("call_1", "echo", '{"value": "x"}')]
    assert turn.messages[1] == Message(role="tool", tool_call_id="call_1", name="echo", content="x")


async def test_run_given_max_iterations_exhausted_turn_messages_includes_every_round():
    tools: dict[str, ITool] = {"echo": _EchoTool()}
    always_calls_tool = _tool_call_completion([_call("call_1", "echo", '{"value": "x"}')])
    llm = _FakeLLM([always_calls_tool, always_calls_tool, _final_completion("gave up")])
    strategy = ReactStrategy()

    turn = await strategy.run([Message(role="user", content="go")], llm, tools, max_iterations=2)

    # 2 rounds x (assistant tool-call + tool result) + 1 forced final answer.
    assert len(turn.messages) == 5
    assert turn.messages[-1] == turn.message


async def test_run_sums_usage_across_every_llm_call():
    tools: dict[str, ITool] = {"echo": _EchoTool()}
    llm = _FakeLLM(
        [
            Completion(
                message=Message(
                    role="assistant",
                    content=None,
                    tool_calls=[_call("call_1", "echo", '{"value": "x"}')],
                ),
                usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                finish_reason="tool_calls",
            ),
            Completion(
                message=Message(role="assistant", content="done"),
                usage=Usage(prompt_tokens=20, completion_tokens=3, total_tokens=23),
                finish_reason="stop",
            ),
        ]
    )
    strategy = ReactStrategy()

    turn = await strategy.run([Message(role="user", content="go")], llm, tools, max_iterations=10)

    assert turn.usage == Usage(prompt_tokens=30, completion_tokens=8, total_tokens=38)


async def test_run_forwards_sampling_params_to_every_llm_call():
    llm = _FakeLLM([_final_completion()])
    strategy = ReactStrategy()

    await strategy.run(
        [Message(role="user", content="hi")],
        llm,
        {},
        max_iterations=10,
        temperature=0.3,
        top_p=0.8,
        max_tokens=100,
    )

    assert llm.calls[0]["temperature"] == 0.3
    assert llm.calls[0]["top_p"] == 0.8
    assert llm.calls[0]["max_tokens"] == 100


async def test_run_given_no_tools_never_offers_tools_to_llm():
    llm = _FakeLLM([_final_completion()])
    strategy = ReactStrategy()

    await strategy.run([Message(role="user", content="hi")], llm, {}, max_iterations=10)

    assert llm.calls[0]["tools"] is None


async def test_run_given_no_tools_and_tool_history_still_returns_a_sensible_turn():
    # The previously-unhandled third gap: an empty `tools` dict makes `tool_schemas` None, so
    # even the MAIN LOOP call declares no tools — on a session whose history already holds a
    # real tool exchange from an earlier turn (e.g. AgentRunService's documented `tools=[]`
    # override). This is the unit half of the fix: the strategy hands that history straight to
    # the LLM and returns a normal Turn, no crash and no special-casing. A fake LLM cannot
    # prove the actual provider-safety property — that the outbound request is flattened is
    # proven in `tests/integration/adapters/test_litellm.py` against the real adapter.
    history = [
        Message(role="user", content="what time is it?"),
        Message(
            role="assistant", content=None, tool_calls=[_call("call_0", "echo", '{"value": "x"}')]
        ),
        Message(role="tool", tool_call_id="call_0", name="echo", content="x"),
        Message(role="assistant", content="it was x"),
        Message(role="user", content="and now?"),
    ]
    llm = _FakeLLM([_final_completion("still x")])
    strategy = ReactStrategy()

    turn = await strategy.run(history, llm, {}, max_iterations=10)

    assert llm.calls[0]["tools"] is None
    assert llm.calls[0]["messages"] == history
    assert turn.messages == [Message(role="assistant", content="still x")]
    assert turn.finish_reason == "stop"


async def test_run_given_original_messages_list_is_not_mutated():
    llm = _FakeLLM(
        [_tool_call_completion([_call("call_1", "echo", '{"value": "x"}')]), _final_completion()]
    )
    tools: dict[str, ITool] = {"echo": _EchoTool()}
    strategy = ReactStrategy()
    original = [Message(role="user", content="go")]

    await strategy.run(original, llm, tools, max_iterations=10)

    assert original == [Message(role="user", content="go")]


async def test_run_returned_turn_messages_excludes_the_input_messages():
    llm = _FakeLLM(
        [_tool_call_completion([_call("call_1", "echo", '{"value": "x"}')]), _final_completion()]
    )
    tools: dict[str, ITool] = {"echo": _EchoTool()}
    strategy = ReactStrategy()
    original = [Message(role="user", content="go")]

    turn = await strategy.run(original, llm, tools, max_iterations=10)

    assert original[0] not in turn.messages


async def test_run_given_no_tool_calls_final_total_tokens_equals_the_only_calls_total():
    llm = _FakeLLM([_final_completion()])
    strategy = ReactStrategy()

    turn = await strategy.run([Message(role="user", content="hi")], llm, {}, max_iterations=10)

    assert turn.final_total_tokens == _usage().total_tokens


async def test_run_given_tool_call_final_total_tokens_equals_the_last_calls_total_not_summed():
    tools: dict[str, ITool] = {"echo": _EchoTool()}
    llm = _FakeLLM(
        [
            Completion(
                message=Message(
                    role="assistant",
                    content=None,
                    tool_calls=[_call("call_1", "echo", '{"value": "x"}')],
                ),
                usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                finish_reason="tool_calls",
            ),
            Completion(
                message=Message(role="assistant", content="done"),
                usage=Usage(prompt_tokens=20, completion_tokens=3, total_tokens=23),
                finish_reason="stop",
            ),
        ]
    )
    strategy = ReactStrategy()

    turn = await strategy.run([Message(role="user", content="go")], llm, tools, max_iterations=10)

    assert turn.final_total_tokens == 23
    assert turn.usage.total_tokens == 38


async def test_run_given_tool_result_over_max_tool_result_chars_is_truncated_with_marker():
    tools: dict[str, ITool] = {"echo": _EchoTool()}
    long_value = "x" * 100
    llm = _FakeLLM(
        [
            _tool_call_completion([_call("call_1", "echo", f'{{"value": "{long_value}"}}')]),
            _final_completion(),
        ]
    )
    strategy = ReactStrategy()

    await strategy.run(
        [Message(role="user", content="go")],
        llm,
        tools,
        max_iterations=10,
        max_tool_result_chars=10,
    )

    result_message = llm.calls[1]["messages"][-1]
    assert result_message.content == "x" * 10 + "\n...[truncated, 90 more characters]"


async def test_run_given_tool_result_under_max_tool_result_chars_is_not_modified():
    tools: dict[str, ITool] = {"echo": _EchoTool()}
    llm = _FakeLLM(
        [
            _tool_call_completion([_call("call_1", "echo", '{"value": "short"}')]),
            _final_completion(),
        ]
    )
    strategy = ReactStrategy()

    await strategy.run(
        [Message(role="user", content="go")],
        llm,
        tools,
        max_iterations=10,
        max_tool_result_chars=1000,
    )

    result_message = llm.calls[1]["messages"][-1]
    assert result_message.content == "short"


async def test_run_given_max_tool_result_chars_none_leaves_long_result_uncapped():
    tools: dict[str, ITool] = {"echo": _EchoTool()}
    long_value = "x" * 100_000
    llm = _FakeLLM(
        [
            _tool_call_completion([_call("call_1", "echo", f'{{"value": "{long_value}"}}')]),
            _final_completion(),
        ]
    )
    strategy = ReactStrategy()

    await strategy.run([Message(role="user", content="go")], llm, tools, max_iterations=10)

    result_message = llm.calls[1]["messages"][-1]
    assert result_message.content == long_value


async def test_run_given_tool_raises_error_content_is_truncated_when_over_max_tool_result_chars():
    tools: dict[str, ITool] = {"boom": _RaisingTool()}
    llm = _FakeLLM([_tool_call_completion([_call("call_1", "boom", "{}")]), _final_completion()])
    strategy = ReactStrategy()

    await strategy.run(
        [Message(role="user", content="go")],
        llm,
        tools,
        max_iterations=10,
        max_tool_result_chars=10,
    )

    result_message = llm.calls[1]["messages"][-1]
    assert result_message.content is not None
    assert result_message.content.startswith("Error: too")
    assert "[truncated," in result_message.content


async def test_run_given_unoffered_tool_error_is_not_truncated_even_with_small_max_tool_result_chars():  # noqa: E501
    llm = _FakeLLM([_tool_call_completion([_call("call_1", "missing", "{}")]), _final_completion()])
    strategy = ReactStrategy()

    await strategy.run(
        [Message(role="user", content="go")],
        llm,
        {},
        max_iterations=10,
        max_tool_result_chars=1,
    )

    result_message = llm.calls[1]["messages"][-1]
    assert result_message.content == "Error: tool 'missing' was not offered for this call"


async def test_run_given_bad_json_error_is_not_truncated_even_with_small_max_tool_result_chars():
    tools: dict[str, ITool] = {"echo": _EchoTool()}
    llm = _FakeLLM(
        [_tool_call_completion([_call("call_1", "echo", "{not json")]), _final_completion()]
    )
    strategy = ReactStrategy()

    await strategy.run(
        [Message(role="user", content="go")],
        llm,
        tools,
        max_iterations=10,
        max_tool_result_chars=1,
    )

    result_message = llm.calls[1]["messages"][-1]
    assert result_message.content is not None
    assert "[truncated," not in result_message.content


async def test_run_given_non_object_args_error_is_not_truncated_even_with_small_max_tool_result_chars():  # noqa: E501
    tools: dict[str, ITool] = {"echo": _EchoTool()}
    llm = _FakeLLM(
        [_tool_call_completion([_call("call_1", "echo", "[1, 2]")]), _final_completion()]
    )
    strategy = ReactStrategy()

    await strategy.run(
        [Message(role="user", content="go")],
        llm,
        tools,
        max_iterations=10,
        max_tool_result_chars=1,
    )

    result_message = llm.calls[1]["messages"][-1]
    assert result_message.content == "Error: arguments must be a JSON object"


async def test_run_given_max_iterations_exhausted_final_total_tokens_is_the_forced_calls_total():
    tools: dict[str, ITool] = {"echo": _EchoTool()}
    always_calls_tool = _tool_call_completion([_call("call_1", "echo", '{"value": "x"}')])
    forced_final = Completion(
        message=Message(role="assistant", content="gave up"),
        usage=Usage(prompt_tokens=50, completion_tokens=4, total_tokens=54),
        finish_reason="stop",
    )
    llm = _FakeLLM([always_calls_tool, always_calls_tool, forced_final])
    strategy = ReactStrategy()

    turn = await strategy.run([Message(role="user", content="go")], llm, tools, max_iterations=2)

    assert turn.final_total_tokens == 54


_SKIP_MARKER = "Error: skipped — this round requested more than the 2 tool calls allowed at once"
_OMITTED_MARKER = "Error: tool result omitted — aggregate tool-output budget exhausted"


def _three_echo_calls() -> list[ToolCall]:
    return [
        _call("call_1", "echo", '{"value": "a"}'),
        _call("call_2", "echo", '{"value": "b"}'),
        _call("call_3", "echo", '{"value": "c"}'),
    ]


async def test_run_given_more_calls_than_max_tool_calls_per_round_executes_only_the_first_n():
    tools: dict[str, ITool] = {"echo": _EchoTool()}
    llm = _FakeLLM([_tool_call_completion(_three_echo_calls()), _final_completion()])
    strategy = ReactStrategy()

    await strategy.run(
        [Message(role="user", content="go")],
        llm,
        tools,
        max_iterations=10,
        max_tool_calls_per_round=2,
    )

    results = llm.calls[1]["messages"][-3:]
    assert [(m.tool_call_id, m.content) for m in results[:2]] == [("call_1", "a"), ("call_2", "b")]


async def test_run_given_more_calls_than_max_tool_calls_per_round_skipped_calls_say_why():
    tools: dict[str, ITool] = {"echo": _EchoTool()}
    llm = _FakeLLM([_tool_call_completion(_three_echo_calls()), _final_completion()])
    strategy = ReactStrategy()

    await strategy.run(
        [Message(role="user", content="go")],
        llm,
        tools,
        max_iterations=10,
        max_tool_calls_per_round=2,
    )

    assert llm.calls[1]["messages"][-1] == Message(
        role="tool", tool_call_id="call_3", name="echo", content=_SKIP_MARKER
    )


async def test_run_given_exactly_max_tool_calls_per_round_calls_executes_all_of_them():
    tools: dict[str, ITool] = {"echo": _EchoTool()}
    llm = _FakeLLM([_tool_call_completion(_three_echo_calls()), _final_completion()])
    strategy = ReactStrategy()

    await strategy.run(
        [Message(role="user", content="go")],
        llm,
        tools,
        max_iterations=10,
        max_tool_calls_per_round=3,
    )

    assert [m.content for m in llm.calls[1]["messages"][-3:]] == ["a", "b", "c"]


async def test_run_given_max_tool_calls_per_round_none_executes_every_call():
    tools: dict[str, ITool] = {"echo": _EchoTool()}
    llm = _FakeLLM([_tool_call_completion(_three_echo_calls()), _final_completion()])
    strategy = ReactStrategy()

    await strategy.run([Message(role="user", content="go")], llm, tools, max_iterations=10)

    assert [m.content for m in llm.calls[1]["messages"][-3:]] == ["a", "b", "c"]


async def test_run_given_results_crossing_max_tool_results_total_chars_omits_the_rest():
    tools: dict[str, ITool] = {"echo": _EchoTool()}
    value = "x" * 50
    llm = _FakeLLM(
        [
            _tool_call_completion(
                [_call(f"call_{n}", "echo", f'{{"value": "{value}"}}') for n in (1, 2, 3)]
            ),
            _final_completion(),
        ]
    )
    strategy = ReactStrategy()

    await strategy.run(
        [Message(role="user", content="go")],
        llm,
        tools,
        max_iterations=10,
        max_tool_results_total_chars=60,
    )

    assert [m.content for m in llm.calls[1]["messages"][-3:]] == [value, value, _OMITTED_MARKER]


async def test_run_given_max_tool_results_total_chars_none_keeps_every_result():
    tools: dict[str, ITool] = {"echo": _EchoTool()}
    value = "x" * 50
    llm = _FakeLLM(
        [
            _tool_call_completion(
                [_call(f"call_{n}", "echo", f'{{"value": "{value}"}}') for n in (1, 2, 3)]
            ),
            _final_completion(),
        ]
    )
    strategy = ReactStrategy()

    await strategy.run([Message(role="user", content="go")], llm, tools, max_iterations=10)

    assert [m.content for m in llm.calls[1]["messages"][-3:]] == [value, value, value]


async def test_run_given_budget_spent_in_an_earlier_round_omits_the_next_rounds_results():
    tools: dict[str, ITool] = {"echo": _EchoTool()}
    value = "x" * 50
    llm = _FakeLLM(
        [
            _tool_call_completion([_call("call_1", "echo", f'{{"value": "{value}"}}')]),
            _tool_call_completion(
                [_call(f"call_{n}", "echo", f'{{"value": "{value}"}}') for n in (2, 3)]
            ),
            _final_completion(),
        ]
    )
    strategy = ReactStrategy()

    await strategy.run(
        [Message(role="user", content="go")],
        llm,
        tools,
        max_iterations=10,
        max_tool_results_total_chars=10,
    )

    assert llm.calls[1]["messages"][-1].content == value  # round 1 spent the whole budget
    assert [m.content for m in llm.calls[2]["messages"][-2:]] == [_OMITTED_MARKER, _OMITTED_MARKER]


async def test_run_given_a_skipped_call_its_marker_is_still_subject_to_the_aggregate_budget():
    tools: dict[str, ITool] = {"echo": _EchoTool()}
    value = "x" * 50
    llm = _FakeLLM(
        [
            _tool_call_completion([_call("call_0", "echo", f'{{"value": "{value}"}}')]),
            _tool_call_completion(_three_echo_calls()),
            _final_completion(),
        ]
    )
    strategy = ReactStrategy()

    await strategy.run(
        [Message(role="user", content="go")],
        llm,
        tools,
        max_iterations=10,
        max_tool_calls_per_round=2,
        max_tool_results_total_chars=10,
    )

    # The skipped call's own marker is a result like any other — it never bypasses the budget.
    assert llm.calls[2]["messages"][-1].content == _OMITTED_MARKER


async def test_run_counts_the_per_call_truncated_length_not_the_original_toward_the_budget():
    tools: dict[str, ITool] = {"echo": _EchoTool()}
    value = "x" * 100
    llm = _FakeLLM(
        [
            _tool_call_completion(
                [_call(f"call_{n}", "echo", f'{{"value": "{value}"}}') for n in (1, 2)]
            ),
            _final_completion(),
        ]
    )
    strategy = ReactStrategy()

    # Each result truncates to 45 chars (10 kept + a 35-char marker), so both fit in 100.
    # Counting the untruncated 100 chars instead would omit the second result.
    await strategy.run(
        [Message(role="user", content="go")],
        llm,
        tools,
        max_iterations=10,
        max_tool_result_chars=10,
        max_tool_results_total_chars=100,
    )

    truncated = "x" * 10 + "\n...[truncated, 90 more characters]"
    assert [m.content for m in llm.calls[1]["messages"][-2:]] == [truncated, truncated]


async def test_run_given_max_iterations_exhausted_logs_warning(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.WARNING, logger="agent.core.strategies.react")
    tools: dict[str, ITool] = {"echo": _EchoTool()}
    always_calls_tool = _tool_call_completion([_call("call_1", "echo", '{"value": "x"}')])
    llm = _FakeLLM([always_calls_tool, _final_completion("gave up")])
    strategy = ReactStrategy()

    await strategy.run([Message(role="user", content="go")], llm, tools, max_iterations=1)

    assert any("max_iterations" in r.message for r in caplog.records)


async def test_run_given_unoffered_tool_logs_warning(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.WARNING, logger="agent.core.strategies.react")
    llm = _FakeLLM([_tool_call_completion([_call("call_1", "missing", "{}")]), _final_completion()])
    strategy = ReactStrategy()

    await strategy.run([Message(role="user", content="go")], llm, {}, max_iterations=10)

    assert any("missing" in r.message for r in caplog.records)


async def test_run_given_malformed_json_logs_warning(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.WARNING, logger="agent.core.strategies.react")
    tools: dict[str, ITool] = {"echo": _EchoTool()}
    llm = _FakeLLM(
        [_tool_call_completion([_call("call_1", "echo", "{not json")]), _final_completion()]
    )
    strategy = ReactStrategy()

    await strategy.run([Message(role="user", content="go")], llm, tools, max_iterations=10)

    assert any(r.levelno == logging.WARNING for r in caplog.records)


async def test_run_given_tool_raises_logs_warning(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.WARNING, logger="agent.core.strategies.react")
    tools: dict[str, ITool] = {"boom": _RaisingTool()}
    llm = _FakeLLM([_tool_call_completion([_call("call_1", "boom", "{}")]), _final_completion()])
    strategy = ReactStrategy()

    await strategy.run([Message(role="user", content="go")], llm, tools, max_iterations=10)

    assert any("boom" in r.message for r in caplog.records)


async def test_run_given_tool_raises_logs_warning_with_exception_type(
    caplog: pytest.LogCaptureFixture,
):
    caplog.set_level(logging.WARNING, logger="agent.core.strategies.react")
    tools: dict[str, ITool] = {"boom": _RaisingTool()}
    llm = _FakeLLM([_tool_call_completion([_call("call_1", "boom", "{}")]), _final_completion()])
    strategy = ReactStrategy()

    await strategy.run([Message(role="user", content="go")], llm, tools, max_iterations=10)

    [record] = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert record.exception_type == "RuntimeError"


async def test_run_given_successful_tool_call_logs_info_start_and_completed(
    caplog: pytest.LogCaptureFixture,
):
    caplog.set_level(logging.INFO, logger="agent.core.strategies.react")
    tools: dict[str, ITool] = {"echo": _EchoTool()}
    llm = _FakeLLM(
        [_tool_call_completion([_call("call_1", "echo", '{"value": "hi"}')]), _final_completion()]
    )
    strategy = ReactStrategy()

    await strategy.run([Message(role="user", content="go")], llm, tools, max_iterations=10)

    info_records = [r for r in caplog.records if r.levelno == logging.INFO]
    assert any("executing" in r.message for r in info_records)
    [completed] = [r for r in info_records if "completed" in r.message]
    assert isinstance(completed.duration_ms, float)


async def test_run_given_tool_raises_does_not_also_log_info_completed(
    caplog: pytest.LogCaptureFixture,
):
    caplog.set_level(logging.DEBUG, logger="agent.core.strategies.react")
    tools: dict[str, ITool] = {"boom": _RaisingTool()}
    llm = _FakeLLM([_tool_call_completion([_call("call_1", "boom", "{}")]), _final_completion()])
    strategy = ReactStrategy()

    await strategy.run([Message(role="user", content="go")], llm, tools, max_iterations=10)

    assert not any(r.levelno == logging.INFO and "completed" in r.message for r in caplog.records)


async def test_run_given_max_tool_calls_per_round_truncation_logs_info(
    caplog: pytest.LogCaptureFixture,
):
    caplog.set_level(logging.INFO, logger="agent.core.strategies.react")
    tools: dict[str, ITool] = {"echo": _EchoTool()}
    llm = _FakeLLM([_tool_call_completion(_three_echo_calls()), _final_completion()])
    strategy = ReactStrategy()

    await strategy.run(
        [Message(role="user", content="go")],
        llm,
        tools,
        max_iterations=10,
        max_tool_calls_per_round=2,
    )

    assert any(
        r.levelno == logging.INFO and "max_tool_calls_per_round" in r.message
        for r in caplog.records
    )


async def test_run_given_max_tool_results_total_chars_truncation_logs_info(
    caplog: pytest.LogCaptureFixture,
):
    caplog.set_level(logging.INFO, logger="agent.core.strategies.react")
    tools: dict[str, ITool] = {"echo": _EchoTool()}
    value = "x" * 50
    llm = _FakeLLM(
        [
            _tool_call_completion(
                [_call(f"call_{n}", "echo", f'{{"value": "{value}"}}') for n in (1, 2, 3)]
            ),
            _final_completion(),
        ]
    )
    strategy = ReactStrategy()

    await strategy.run(
        [Message(role="user", content="go")],
        llm,
        tools,
        max_iterations=10,
        max_tool_results_total_chars=60,
    )

    assert any(r.levelno == logging.INFO and "budget" in r.message for r in caplog.records)


async def test_run_given_no_truncation_does_not_log_info_about_it(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO, logger="agent.core.strategies.react")
    tools: dict[str, ITool] = {"echo": _EchoTool()}
    llm = _FakeLLM([_tool_call_completion(_three_echo_calls()), _final_completion()])
    strategy = ReactStrategy()

    await strategy.run([Message(role="user", content="go")], llm, tools, max_iterations=10)

    assert not any(
        "budget" in r.message or "max_tool_calls_per_round" in r.message for r in caplog.records
    )


async def test_run_given_no_tool_calls_logs_info_calling_and_responded(
    caplog: pytest.LogCaptureFixture,
):
    caplog.set_level(logging.INFO, logger="agent.core.strategies.react")
    llm = _FakeLLM([_final_completion()])
    strategy = ReactStrategy()

    await strategy.run([Message(role="user", content="hi")], llm, {}, max_iterations=10)

    info_records = [r for r in caplog.records if r.levelno == logging.INFO]
    assert any("calling" in r.message.lower() for r in info_records)
    [responded] = [r for r in info_records if "responded" in r.message.lower()]
    assert isinstance(responded.duration_ms, float)


async def test_run_given_multiple_iterations_logs_info_once_per_iteration(
    caplog: pytest.LogCaptureFixture,
):
    caplog.set_level(logging.INFO, logger="agent.core.strategies.react")
    tools: dict[str, ITool] = {"echo": _EchoTool()}
    always_calls_tool = _tool_call_completion([_call("call_1", "echo", '{"value": "x"}')])
    llm = _FakeLLM([always_calls_tool, _final_completion("done")])
    strategy = ReactStrategy()

    await strategy.run([Message(role="user", content="go")], llm, tools, max_iterations=10)

    calling_lines = [r for r in caplog.records if "calling" in r.message.lower()]
    assert len(calling_lines) == 2  # once per llm.complete() call, including this one's success


async def test_run_given_missing_required_argument_returns_validation_error_content():
    tools: dict[str, ITool] = {"echo": _EchoTool()}
    llm = _FakeLLM([_tool_call_completion([_call("call_1", "echo", "{}")]), _final_completion()])
    strategy = ReactStrategy()

    turn = await strategy.run([Message(role="user", content="go")], llm, tools, max_iterations=10)

    result_message = llm.calls[1]["messages"][-1]
    assert result_message.content == "Error: invalid arguments: value: Field required"
    assert turn.message.content == "final answer"


async def test_run_given_wrong_type_argument_returns_validation_error_content():
    tools: dict[str, ITool] = {"echo": _EchoTool()}
    llm = _FakeLLM(
        [_tool_call_completion([_call("call_1", "echo", '{"value": 5}')]), _final_completion()]
    )
    strategy = ReactStrategy()

    await strategy.run([Message(role="user", content="go")], llm, tools, max_iterations=10)

    result_message = llm.calls[1]["messages"][-1]
    assert (
        result_message.content == "Error: invalid arguments: value: Input should be a valid string"
    )


async def test_run_given_extra_argument_returns_validation_error_content():
    tools: dict[str, ITool] = {"echo": _EchoTool()}
    llm = _FakeLLM(
        [
            _tool_call_completion([_call("call_1", "echo", '{"value": "hi", "extra": "x"}')]),
            _final_completion(),
        ]
    )
    strategy = ReactStrategy()

    await strategy.run([Message(role="user", content="go")], llm, tools, max_iterations=10)

    result_message = llm.calls[1]["messages"][-1]
    assert (
        result_message.content == "Error: invalid arguments: extra: Extra inputs are not permitted"
    )


async def test_run_given_valid_arguments_executes_normally():
    tools: dict[str, ITool] = {"echo": _EchoTool()}
    llm = _FakeLLM(
        [_tool_call_completion([_call("call_1", "echo", '{"value": "hi"}')]), _final_completion()]
    )
    strategy = ReactStrategy()

    turn = await strategy.run([Message(role="user", content="go")], llm, tools, max_iterations=10)

    result_message = llm.calls[1]["messages"][-1]
    assert result_message.content == "hi"
    assert turn.message.content == "final answer"


async def test_run_given_valid_arguments_execute_receives_the_validated_value():
    # Proves execute() gets validated.model_dump(), not the raw parsed JSON — a default
    # the caller never supplied must still reach execute() as an explicit kwarg.
    class _DefaultingParams(BaseModel):
        model_config = ConfigDict(extra="forbid")
        value: str = "default"

    class _DefaultingTool:
        name = "defaulting"
        description = "Returns its value argument, or a default."
        parameters_model: type[BaseModel] = _DefaultingParams
        received: dict[str, Any] | None = None

        async def execute(self, **kwargs: Any) -> str:
            self.received = kwargs
            return kwargs["value"]

    tool = _DefaultingTool()
    tools: dict[str, ITool] = {"defaulting": tool}
    llm = _FakeLLM(
        [_tool_call_completion([_call("call_1", "defaulting", "{}")]), _final_completion()]
    )
    strategy = ReactStrategy()

    await strategy.run([Message(role="user", content="go")], llm, tools, max_iterations=10)

    assert tool.received == {"value": "default"}


async def test_run_given_validation_error_is_not_truncated_even_with_small_max_tool_result_chars():  # noqa: E501
    tools: dict[str, ITool] = {"echo": _EchoTool()}
    llm = _FakeLLM([_tool_call_completion([_call("call_1", "echo", "{}")]), _final_completion()])
    strategy = ReactStrategy()

    await strategy.run(
        [Message(role="user", content="go")],
        llm,
        tools,
        max_iterations=10,
        max_tool_result_chars=1,
    )

    result_message = llm.calls[1]["messages"][-1]
    assert result_message.content == "Error: invalid arguments: value: Field required"


async def test_run_given_validation_error_logs_warning(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.WARNING, logger="agent.core.strategies.react")
    tools: dict[str, ITool] = {"echo": _EchoTool()}
    llm = _FakeLLM([_tool_call_completion([_call("call_1", "echo", "{}")]), _final_completion()])
    strategy = ReactStrategy()

    await strategy.run([Message(role="user", content="go")], llm, tools, max_iterations=10)

    assert any("validation" in r.message for r in caplog.records)


async def test_run_given_validator_raises_non_validation_error_returns_error_content():
    class _ExplodingParams(BaseModel):
        model_config = ConfigDict(extra="forbid")
        value: str

        @field_validator("value")
        @classmethod
        def _explode(cls, v: str) -> str:
            raise KeyError("boom")

    class _ExplodingValidatorTool:
        name = "exploding_validator"
        description = "A tool whose own validator raises something Pydantic doesn't wrap."
        parameters_model: type[BaseModel] = _ExplodingParams

        async def execute(self, **kwargs: Any) -> str:
            return "should never reach here"

    tools: dict[str, ITool] = {"exploding_validator": _ExplodingValidatorTool()}
    llm = _FakeLLM(
        [
            _tool_call_completion([_call("call_1", "exploding_validator", '{"value": "x"}')]),
            _final_completion(),
        ]
    )
    strategy = ReactStrategy()

    turn = await strategy.run([Message(role="user", content="go")], llm, tools, max_iterations=10)

    result_message = llm.calls[1]["messages"][-1]
    assert result_message.content == "Error: invalid arguments for tool 'exploding_validator'"
    assert turn.message.content == "final answer"


async def test_run_given_many_extra_arguments_caps_the_error_count_shown():
    tools: dict[str, ITool] = {"echo": _EchoTool()}
    payload = {"value": "hi", **{f"extra_{i}": "x" for i in range(10)}}
    llm = _FakeLLM(
        [
            _tool_call_completion([_call("call_1", "echo", json.dumps(payload))]),
            _final_completion(),
        ]
    )
    strategy = ReactStrategy()

    await strategy.run([Message(role="user", content="go")], llm, tools, max_iterations=10)

    result_message = llm.calls[1]["messages"][-1]
    assert result_message.content is not None
    # 5 errors shown joined by "; " (4 separators) plus the "; ...and N more" suffix (1 more).
    assert result_message.content.count(";") == 5
    assert "...and 5 more error(s)" in result_message.content


async def test_run_given_long_extra_argument_name_truncates_the_field_reference():
    tools: dict[str, ITool] = {"echo": _EchoTool()}
    long_key = "x" * 300
    llm = _FakeLLM(
        [
            _tool_call_completion(
                [_call("call_1", "echo", json.dumps({"value": "hi", long_key: "y"}))]
            ),
            _final_completion(),
        ]
    )
    strategy = ReactStrategy()

    await strategy.run([Message(role="user", content="go")], llm, tools, max_iterations=10)

    result_message = llm.calls[1]["messages"][-1]
    assert result_message.content is not None
    assert "[truncated," in result_message.content


async def test_run_given_validation_error_log_does_not_contain_pydantic_doc_url(
    caplog: pytest.LogCaptureFixture,
):
    caplog.set_level(logging.WARNING, logger="agent.core.strategies.react")
    tools: dict[str, ITool] = {"echo": _EchoTool()}
    llm = _FakeLLM([_tool_call_completion([_call("call_1", "echo", "{}")]), _final_completion()])
    strategy = ReactStrategy()

    await strategy.run([Message(role="user", content="go")], llm, tools, max_iterations=10)

    assert not any("errors.pydantic.dev" in r.message for r in caplog.records)


@pytest.mark.parametrize(
    ("tool_name", "tools", "arguments", "expected_detail"),
    [
        ("echo", {"echo": _EchoTool()}, "{not json", "Expecting property name"),
        ("echo", {"echo": _EchoTool()}, "{}", "Field required"),
        ("boom", {"boom": _RaisingTool()}, "{}", "tool exploded"),
    ],
)
async def test_execute_call_exception_logs_always_include_detail_and_exc_info(
    caplog: pytest.LogCaptureFixture,
    tool_name: str,
    tools: dict[str, ITool],
    arguments: str,
    expected_detail: str,
) -> None:
    # Unlike this exception's span/tool-result-message content (opt-in via
    # capture_content_enabled()), the log always gets full detail: logs never leave this
    # process, so they're the one place full detail is safe by default.
    caplog.set_level(logging.WARNING, logger="agent.core.strategies.react")

    await react._execute_call(tools, _call("call_1", tool_name, arguments), None, None)

    assert len(caplog.records) == 1
    assert caplog.records[0].exc_info is not None
    assert expected_detail in caplog.records[0].message


async def test_execute_call_given_validator_raises_non_validation_error_logs_detail(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _ExplodingParams(BaseModel):
        model_config = ConfigDict(extra="forbid")
        value: str

        @field_validator("value")
        @classmethod
        def _explode(cls, v: str) -> str:
            raise KeyError("boom")

    class _ExplodingValidatorTool:
        name = "exploding_validator"
        description = "A tool whose own validator raises something Pydantic doesn't wrap."
        parameters_model: type[BaseModel] = _ExplodingParams

        async def execute(self, **kwargs: Any) -> str:
            return "should never reach here"

    caplog.set_level(logging.WARNING, logger="agent.core.strategies.react")
    tools: dict[str, ITool] = {"exploding_validator": _ExplodingValidatorTool()}

    await react._execute_call(
        tools, _call("call_1", "exploding_validator", '{"value": "x"}'), None, None
    )

    assert len(caplog.records) == 1
    assert caplog.records[0].exc_info is not None
    assert "boom" in caplog.records[0].message


async def test_run_given_tool_output_guardrail_blocks_returns_error_tool_result_not_raise():
    guardrail = _FakeGuardrail(
        "no-secrets", "block", GuardrailFinding(triggered=True, reason="leaked a key")
    )
    tool = _EchoTool()
    llm = _FakeLLM(
        [
            Completion(
                message=Message(
                    role="assistant",
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            function=ToolCallFunction(
                                name="echo", arguments=json.dumps({"value": "sk-abc123"})
                            ),
                        )
                    ],
                ),
                usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                finish_reason="tool_calls",
            ),
            Completion(
                message=Message(role="assistant", content="done"),
                usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                finish_reason="stop",
            ),
        ]
    )

    turn = await ReactStrategy().run(
        [Message(role="user", content="echo the secret")],
        llm,
        {"echo": tool},
        max_iterations=5,
        tool_output_guardrails=[guardrail],
    )

    tool_result_message = turn.messages[1]
    assert tool_result_message.role == "tool"
    assert tool_result_message.content is not None
    assert "guardrail 'no-secrets' blocked this content" in tool_result_message.content


async def test_run_given_tool_output_guardrail_blocks_truncates_error_to_max_tool_result_chars():
    guardrail = _FakeGuardrail(
        "no-secrets", "block", GuardrailFinding(triggered=True, reason="leaked a key")
    )
    tool = _EchoTool()
    llm = _FakeLLM(
        [
            Completion(
                message=Message(
                    role="assistant",
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            function=ToolCallFunction(
                                name="echo", arguments=json.dumps({"value": "sk-abc123"})
                            ),
                        )
                    ],
                ),
                usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                finish_reason="tool_calls",
            ),
            Completion(
                message=Message(role="assistant", content="done"),
                usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                finish_reason="stop",
            ),
        ]
    )

    turn = await ReactStrategy().run(
        [Message(role="user", content="echo the secret")],
        llm,
        {"echo": tool},
        max_iterations=5,
        max_tool_result_chars=10,
        tool_output_guardrails=[guardrail],
    )

    tool_result_message = turn.messages[1]
    content = tool_result_message.content
    assert content is not None
    assert content.startswith("Error: too")
    assert "...[truncated," in content
    untruncated_length = len("Error: tool result guardrail 'no-secrets' blocked this content")
    assert len(content) < untruncated_length


async def test_run_given_tool_output_guardrail_redacts_replaces_result_content():
    guardrail = _FakeGuardrail(
        "no-secrets",
        "redact",
        GuardrailFinding(triggered=True, reason="secret", redacted_content="[REDACTED]"),
    )
    tool = _EchoTool()
    llm = _FakeLLM(
        [
            Completion(
                message=Message(
                    role="assistant",
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            function=ToolCallFunction(
                                name="echo", arguments=json.dumps({"value": "sk-abc123"})
                            ),
                        )
                    ],
                ),
                usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                finish_reason="tool_calls",
            ),
            Completion(
                message=Message(role="assistant", content="done"),
                usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                finish_reason="stop",
            ),
        ]
    )

    turn = await ReactStrategy().run(
        [Message(role="user", content="echo the secret")],
        llm,
        {"echo": tool},
        max_iterations=5,
        tool_output_guardrails=[guardrail],
    )

    assert turn.messages[1].content == "[REDACTED]"


async def test_execute_call_opens_a_span_named_after_the_tool(monkeypatch: pytest.MonkeyPatch):
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(react, "tracer", provider.get_tracer("test"))
    tools: dict[str, ITool] = {"echo": _EchoTool()}

    await react._execute_call(tools, _call("call_1", "echo", '{"value": "hi"}'), None, None)

    [span] = [s for s in exporter.get_finished_spans() if s.name == "execute_tool echo"]
    assert span.attributes["gen_ai.tool.name"] == "echo"
    assert span.attributes["gen_ai.tool.call.id"] == "call_1"
    assert "gen_ai.tool.description" not in span.attributes
    assert span.attributes["gen_ai.tool.type"] == "function"
    assert "gen_ai.agent.name" not in span.attributes
    assert "gen_ai.conversation.id" not in span.attributes


async def test_execute_call_given_capture_content_span_has_tool_description(
    monkeypatch: pytest.MonkeyPatch,
):
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(react, "tracer", provider.get_tracer("test"))
    tools: dict[str, ITool] = {"echo": _EchoTool()}
    set_capture_content(True)
    try:
        await react._execute_call(tools, _call("call_1", "echo", '{"value": "hi"}'), None, None)
    finally:
        set_capture_content(False)

    [span] = [s for s in exporter.get_finished_spans() if s.name == "execute_tool echo"]
    assert span.attributes["gen_ai.tool.description"] == "Echoes its input."


async def test_execute_call_given_active_run_context_span_has_agent_name(
    monkeypatch: pytest.MonkeyPatch,
):
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(react, "tracer", provider.get_tracer("test"))
    tools: dict[str, ITool] = {"echo": _EchoTool()}

    with run_context("clock-bot", "sess-1"):
        await react._execute_call(tools, _call("call_1", "echo", '{"value": "hi"}'), None, None)

    [span] = [s for s in exporter.get_finished_spans() if s.name == "execute_tool echo"]
    assert span.attributes["gen_ai.agent.name"] == "clock-bot"
    assert span.attributes["gen_ai.conversation.id"] == "sess-1"


async def test_execute_call_given_unoffered_tool_span_status_is_error(
    monkeypatch: pytest.MonkeyPatch,
):
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(react, "tracer", provider.get_tracer("test"))

    await react._execute_call({}, _call("call_1", "missing", "{}"), None, None)

    [span] = [s for s in exporter.get_finished_spans() if s.name == "execute_tool missing"]
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes["error.type"] == "tool_not_offered"


async def test_execute_call_given_tool_raises_span_status_description_omitted_by_default(
    monkeypatch: pytest.MonkeyPatch,
):
    # The status description is a second place this same content could reach an exported
    # span — gen_ai.tool.call.result is correctly gated a few lines away, but set_status()
    # is a separate call that must be gated identically, not just the attribute.
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(react, "tracer", provider.get_tracer("test"))
    tools: dict[str, ITool] = {"boom": _RaisingTool()}

    await react._execute_call(tools, _call("call_1", "boom", "{}"), None, None)

    [span] = [s for s in exporter.get_finished_spans() if s.name == "execute_tool boom"]
    assert span.status.status_code == StatusCode.ERROR
    assert span.status.description is None
    assert "tool exploded" not in (span.attributes.get("gen_ai.tool.call.result") or "")


async def test_execute_call_given_tool_raises_span_status_description_set_when_capturing_content(
    monkeypatch: pytest.MonkeyPatch,
):
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(react, "tracer", provider.get_tracer("test"))
    tools: dict[str, ITool] = {"boom": _RaisingTool()}
    set_capture_content(True)
    try:
        await react._execute_call(tools, _call("call_1", "boom", "{}"), None, None)
    finally:
        set_capture_content(False)

    [span] = [s for s in exporter.get_finished_spans() if s.name == "execute_tool boom"]
    assert span.status.description is not None
    assert "tool exploded" in span.status.description


async def test_execute_call_given_success_span_status_is_not_error(
    monkeypatch: pytest.MonkeyPatch,
):
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(react, "tracer", provider.get_tracer("test"))
    tools: dict[str, ITool] = {"echo": _EchoTool()}

    await react._execute_call(tools, _call("call_1", "echo", '{"value": "hi"}'), None, None)

    [span] = [s for s in exporter.get_finished_spans() if s.name == "execute_tool echo"]
    assert span.status.status_code != StatusCode.ERROR


async def test_execute_call_given_success_content_starting_with_error_text_span_is_not_error(
    monkeypatch: pytest.MonkeyPatch,
):
    # Proves is_error is an explicit signal, not a content-sniff: a legitimate successful
    # result whose text happens to start with "Error:" must not mark the span ERROR.
    class _MisleadingSuccessTool:
        name = "misleading"
        description = "Returns a legitimate result that happens to start with 'Error:'."
        parameters_model: type[BaseModel] = _EmptyParams

        async def execute(self, **kwargs: Any) -> str:
            return "Error: connection refused"

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(react, "tracer", provider.get_tracer("test"))
    tools: dict[str, ITool] = {"misleading": _MisleadingSuccessTool()}

    result = await react._execute_call(tools, _call("call_1", "misleading", "{}"), None, None)

    assert result.content == "Error: connection refused"
    [span] = [s for s in exporter.get_finished_spans() if s.name == "execute_tool misleading"]
    assert span.status.status_code != StatusCode.ERROR


async def test_execute_call_given_tool_raises_span_omits_exception_event_by_default(
    monkeypatch: pytest.MonkeyPatch,
):
    # A tool's own exception message can embed the tool-argument/tool-output content this
    # is reporting on, so record_exception() is gated the same as every other span-content
    # site — error.type alone (already content-free) identifies the failure by default.
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(react, "tracer", provider.get_tracer("test"))
    tools: dict[str, ITool] = {"boom": _RaisingTool()}

    await react._execute_call(tools, _call("call_1", "boom", "{}"), None, None)

    [span] = [s for s in exporter.get_finished_spans() if s.name == "execute_tool boom"]
    assert span.events == ()
    assert span.attributes["error.type"] == "RuntimeError"


async def test_execute_call_given_tool_raises_span_records_the_exception_when_capturing_content(
    monkeypatch: pytest.MonkeyPatch,
):
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(react, "tracer", provider.get_tracer("test"))
    tools: dict[str, ITool] = {"boom": _RaisingTool()}
    set_capture_content(True)
    try:
        await react._execute_call(tools, _call("call_1", "boom", "{}"), None, None)
    finally:
        set_capture_content(False)

    [span] = [s for s in exporter.get_finished_spans() if s.name == "execute_tool boom"]
    assert len(span.events) >= 1
    assert any(event.name == "exception" for event in span.events)
    assert span.attributes["error.type"] == "RuntimeError"


async def test_execute_call_given_long_arguments_span_content_is_truncated(
    monkeypatch: pytest.MonkeyPatch,
):
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(react, "tracer", provider.get_tracer("test"))
    set_capture_content(True)
    try:
        tools: dict[str, ITool] = {"echo": _EchoTool()}
        long_value = "x" * 100
        arguments = f'{{"value": "{long_value}"}}'

        await react._execute_call(tools, _call("call_1", "echo", arguments), 10, None)

        [span] = [s for s in exporter.get_finished_spans() if s.name == "execute_tool echo"]
        assert span.attributes["gen_ai.tool.call.arguments"] == (
            arguments[:10] + f"\n...[truncated, {len(arguments) - 10} more characters]"
        )
    finally:
        set_capture_content(False)


async def test_execute_call_span_has_operation_name(monkeypatch: pytest.MonkeyPatch):
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(react, "tracer", provider.get_tracer("test"))
    tools: dict[str, ITool] = {"echo": _EchoTool()}

    await react._execute_call(tools, _call("call_1", "echo", '{"value": "hi"}'), None, None)

    [span] = [s for s in exporter.get_finished_spans() if s.name == "execute_tool echo"]
    assert span.attributes["gen_ai.operation.name"] == "execute_tool"


async def test_run_given_concurrent_tool_calls_produces_sibling_spans(
    monkeypatch: pytest.MonkeyPatch,
):
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(react, "tracer", provider.get_tracer("test"))
    counter, max_seen = [0], [0]
    tools: dict[str, ITool] = {"slow": _ConcurrentTool(counter, max_seen)}
    llm = _FakeLLM(
        [
            _tool_call_completion([_call("c1", "slow", "{}"), _call("c2", "slow", "{}")]),
            _final_completion(),
        ]
    )
    strategy = ReactStrategy()

    # ReactStrategy.run() is exercised standalone here, with no AgentRunService/agent_run.py
    # wrapping it — in the real call chain, agent_run.py's "invoke_agent" span (Task 5) is
    # already active for the whole call, which is what makes the two concurrent tool spans
    # siblings under a common parent. In isolation there is no such parent, so the test opens
    # its own enclosing span to stand in for it — this is a test-only span, not a change to
    # production code.
    with provider.get_tracer("test").start_as_current_span("test_root") as root:
        await strategy.run([Message(role="user", content="go")], llm, tools, max_iterations=10)

    tool_spans = [s for s in exporter.get_finished_spans() if s.name == "execute_tool slow"]
    assert len(tool_spans) == 2
    assert tool_spans[0].parent.span_id == tool_spans[1].parent.span_id
    assert tool_spans[0].parent.span_id == root.get_span_context().span_id
