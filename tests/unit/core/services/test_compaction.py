"""Tests for CompactionService."""

import logging
from typing import Any

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agent.core.exceptions import (
    LLMContextWindowExceededError,
    LLMError,
    LLMRateLimitedError,
    SessionNotFoundError,
)
from agent.core.models.completion import Completion
from agent.core.models.config import CompactionConfig
from agent.core.models.message import Message, ToolCall, ToolCallFunction
from agent.core.models.usage import Usage
from agent.core.registries.llm import LLMRegistry
from agent.core.run_context import collect_extra_usage, run_context
from agent.core.services import compaction as compaction_module
from agent.core.services.compaction import (
    CompactionService,
    _chunk_by_turns,
    _split_at_turn_boundary,
)
from agent.core.services.context_tracker import ContextFootprintTracker
from agent.core.session_stores.in_memory import InMemorySessionStore
from agent.core.tracing import set_capture_content


class _FakeLLM:
    """A stand-in ILLM: fixed max_input_tokens, queued complete() outcomes.

    `completion` takes either one outcome (used for every call) or a list returned in
    order, one per `complete()` call, with the last one repeating — same convention as
    `_FakeStrategy` in `test_agent_run.py`. The list form is what exercises the
    summarizer's retry-once and its chunked-summarization fallback, both of which turn on
    *consecutive* calls differing.
    """

    def __init__(
        self,
        max_input_tokens: int = 1000,
        completion: Completion | Exception | list[Completion | Exception] | None = None,
        max_input_tokens_error: Exception | None = None,
    ) -> None:
        self._max_input_tokens = max_input_tokens
        self._completions = completion if isinstance(completion, list) else [completion]
        self._max_input_tokens_error = max_input_tokens_error
        self.complete_calls: list[list[Message]] = []

    async def complete(
        self,
        messages: list[Message],
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> Completion:
        self.complete_calls.append(list(messages))
        outcome = self._completions[min(len(self.complete_calls) - 1, len(self._completions) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        assert outcome is not None
        return outcome

    def max_input_tokens(self) -> int:
        if self._max_input_tokens_error is not None:
            raise self._max_input_tokens_error
        return self._max_input_tokens


def _summary_completion(content: str = "summary", finish_reason: str = "stop") -> Completion:
    return Completion(
        message=Message(role="assistant", content=content),
        usage=Usage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
        finish_reason=finish_reason,
    )


def _config(**overrides: object) -> CompactionConfig:
    return CompactionConfig(model="summarizer", **overrides)


def _turn(n: int) -> list[Message]:
    """One turn: a user message and an assistant reply, both tagged with `n`.

    Deliberately wordy: `compact` refuses to replace content it wouldn't actually shrink, and
    the synthetic summary pair it substitutes costs a fixed ~67 characters of prefix and
    acknowledgment. A toy-sized turn would be *grown* by compaction, not shrunk, so every turn
    here is comfortably larger than that floor.
    """
    return [
        Message(
            role="user",
            content=f"question {n}: what should we do about the deployment and its timeline?",
        ),
        Message(
            role="assistant",
            content=f"answer {n}: ship it on friday, once the review and the smoke tests pass.",
        ),
    ]


def _tool_turn(n: int) -> list[Message]:
    """One turn with a tool exchange: user, tool-call request, tool result, final answer."""
    return [
        Message(role="user", content=f"question {n}"),
        Message(
            role="assistant",
            content=None,
            tool_calls=[
                ToolCall(
                    id=f"call_{n}",
                    function=ToolCallFunction(name="echo", arguments='{"text":"hi"}'),
                )
            ],
        ),
        Message(role="tool", tool_call_id=f"call_{n}", name="echo", content="hi"),
        Message(role="assistant", content=f"answer {n}"),
    ]


def _compacted(recent: list[Message], content: str = "summary") -> list[Message]:
    """The history `compact` stores: a synthetic summary turn prepended before `recent`.

    Always the same two-message shape (`role="user"` summary, `role="assistant"`
    acknowledgment) regardless of how many turns `recent` holds, including none — this
    keeps the stored history starting on `role="user"` (the shape Anthropic requires) and
    never produces two consecutive same-role messages, since the acknowledgment always sits
    between the summary and whatever comes next.
    """
    summary_text = f"[Summary of earlier conversation]\n{content}"
    return [
        Message(role="user", content=summary_text),
        Message(role="assistant", content="Understood — I have that context."),
        *recent,
    ]


def _assert_anthropic_safe_shape(messages: list[Message]) -> None:
    """Structural invariants any message list `compact` stores or replays must satisfy.

    Checked independently of any single test's own hand-built expected value (e.g.
    `_compacted()`), so a shared wrong assumption in both the implementation and a test's
    fixture can't hide a real violation — this is exactly how the assistant-first-message
    and consecutive-user-message bugs slipped through earlier reviews. Tool exchanges are not
    checked here: both stored history and a summarizer request may legitimately carry them —
    folding them out of a no-tools request is the `ILLM` implementation's job, covered in
    `tests/integration/adapters/test_litellm.py`. Empty: trivially safe.
    """
    if not messages:
        return
    assert messages[0].role == "user"
    roles = [message.role for message in messages]
    assert all(before != after for before, after in zip(roles, roles[1:], strict=False))
    assert all((message.content or "").strip() or message.tool_calls for message in messages)


async def _seeded_store(turns: list[list[Message]]) -> tuple[InMemorySessionStore, str]:
    """Create a session and append each element of `turns` as one atomic turn."""
    store = InMemorySessionStore()
    session_id = await store.create("researcher")
    for turn in turns:
        await store.append("researcher", session_id, turn)
    return store, session_id


# — _split_at_turn_boundary ----------------------------------------------------------------


def test_split_given_empty_history_returns_zero():
    assert _split_at_turn_boundary([], keep_recent_turns=4) == 0


def test_split_given_fewer_turns_than_keep_returns_zero():
    history = _turn(1) + _turn(2)

    assert _split_at_turn_boundary(history, keep_recent_turns=4) == 0


def test_split_given_exactly_keep_many_turns_returns_zero():
    history = _turn(1) + _turn(2)

    assert _split_at_turn_boundary(history, keep_recent_turns=2) == 0


def test_split_given_more_turns_than_keep_splits_at_turn_boundary():
    history = _turn(1) + _turn(2) + _turn(3)

    split = _split_at_turn_boundary(history, keep_recent_turns=1)

    assert history[:split] == _turn(1) + _turn(2)
    assert history[split:] == _turn(3)


def test_split_given_keep_zero_and_nonempty_history_returns_full_length():
    history = _turn(1) + _turn(2)

    assert _split_at_turn_boundary(history, keep_recent_turns=0) == len(history)


def test_split_never_separates_a_tool_call_from_its_result():
    tool_call = Message(
        role="assistant",
        content=None,
        tool_calls=[ToolCall(id="call_1", function=ToolCallFunction(name="echo", arguments="{}"))],
    )
    tool_result = Message(role="tool", tool_call_id="call_1", name="echo", content="hi")
    history = _turn(1) + [
        Message(role="user", content="q2"),
        tool_call,
        tool_result,
        Message(role="assistant", content="a2"),
    ]

    split = _split_at_turn_boundary(history, keep_recent_turns=1)

    kept = history[split:]
    assert tool_call in kept
    assert tool_result in kept


# — _chunk_by_turns ------------------------------------------------------------------------


def test_chunk_by_turns_given_empty_messages_returns_no_chunks():
    assert _chunk_by_turns([], 4) == []


def test_chunk_by_turns_splits_into_groups_of_the_given_turn_count():
    messages = _turn(1) + _turn(2) + _turn(3) + _turn(4)

    assert _chunk_by_turns(messages, 2) == [_turn(1) + _turn(2), _turn(3) + _turn(4)]


def test_chunk_by_turns_given_a_partial_last_group_keeps_the_remainder():
    messages = _turn(1) + _turn(2) + _turn(3)

    assert _chunk_by_turns(messages, 2) == [_turn(1) + _turn(2), _turn(3)]


def test_chunk_by_turns_given_a_single_turn_returns_one_chunk():
    # `_summarize_chunked` refuses to chunk further when this returns one chunk — a single
    # turn that alone overflows the summarizer isn't recoverable by splitting it.
    assert _chunk_by_turns(_turn(1), 4) == [_turn(1)]


def test_chunk_by_turns_never_separates_a_tool_call_from_its_result():
    chunks = _chunk_by_turns(_tool_turn(1) + _tool_turn(2), 1)

    assert chunks == [_tool_turn(1), _tool_turn(2)]


# — context_tracker / maybe_compact ---------------------------------------------------------


async def test_maybe_compact_given_no_recorded_usage_is_a_noop():
    llm_registry = LLMRegistry()
    llm = _FakeLLM()
    llm_registry.register("agent-model", llm)
    store, session_id = await _seeded_store([_turn(1), _turn(2), _turn(3)])
    tracker = ContextFootprintTracker()
    service = CompactionService(llm_registry, store, _config(keep_recent_turns=1), tracker)

    await service.maybe_compact("researcher", session_id, "agent-model")

    assert llm.complete_calls == []
    assert await store.get("researcher", session_id) == _turn(1) + _turn(2) + _turn(3)


async def test_maybe_compact_given_usage_under_budget_is_a_noop():
    llm_registry = LLMRegistry()
    llm = _FakeLLM(max_input_tokens=1000)
    llm_registry.register("agent-model", llm)
    store, session_id = await _seeded_store([_turn(1), _turn(2), _turn(3)])
    tracker = ContextFootprintTracker()
    service = CompactionService(
        llm_registry, store, _config(keep_recent_turns=1, token_budget_pct=0.8), tracker
    )
    tracker.record("researcher", session_id, 100)

    await service.maybe_compact("researcher", session_id, "agent-model")

    assert await store.get("researcher", session_id) == _turn(1) + _turn(2) + _turn(3)


async def test_maybe_compact_given_usage_over_budget_compacts():
    llm_registry = LLMRegistry()
    llm = _FakeLLM(max_input_tokens=1000)
    summarizer = _FakeLLM(completion=_summary_completion())
    llm_registry.register("agent-model", llm)
    llm_registry.register("summarizer", summarizer)
    store, session_id = await _seeded_store([_turn(1), _turn(2), _turn(3)])
    tracker = ContextFootprintTracker()
    service = CompactionService(
        llm_registry, store, _config(keep_recent_turns=1, token_budget_pct=0.8), tracker
    )
    tracker.record("researcher", session_id, 900)

    await service.maybe_compact("researcher", session_id, "agent-model")

    stored = await store.get("researcher", session_id)
    assert stored == _compacted(_turn(3))
    _assert_anthropic_safe_shape(stored)


async def test_maybe_compact_given_per_call_model_uses_that_models_budget():
    llm_registry = LLMRegistry()
    small_window_llm = _FakeLLM(max_input_tokens=100)
    summarizer = _FakeLLM(completion=_summary_completion())
    llm_registry.register("small-model", small_window_llm)
    llm_registry.register("summarizer", summarizer)
    store, session_id = await _seeded_store([_turn(1), _turn(2), _turn(3)])
    tracker = ContextFootprintTracker()
    service = CompactionService(
        llm_registry, store, _config(keep_recent_turns=1, token_budget_pct=0.8), tracker
    )
    tracker.record("researcher", session_id, 90)  # over 100 * 0.8 = 80

    await service.maybe_compact("researcher", session_id, "small-model")

    stored = await store.get("researcher", session_id)
    assert stored == _compacted(_turn(3))
    _assert_anthropic_safe_shape(stored)


async def test_maybe_compact_given_unknown_context_window_is_a_noop():
    llm_registry = LLMRegistry()
    llm = _FakeLLM(max_input_tokens_error=LLMError("litellm has no max_input_tokens for model"))
    llm_registry.register("agent-model", llm)
    store, session_id = await _seeded_store([_turn(1), _turn(2), _turn(3)])
    tracker = ContextFootprintTracker()
    service = CompactionService(
        llm_registry, store, _config(keep_recent_turns=1, token_budget_pct=0.8), tracker
    )
    tracker.record("researcher", session_id, 900)

    await service.maybe_compact("researcher", session_id, "agent-model")

    assert llm.complete_calls == []
    assert await store.get("researcher", session_id) == _turn(1) + _turn(2) + _turn(3)


# — compact -----------------------------------------------------------------------------


async def test_compact_given_nothing_older_than_keep_window_returns_false():
    llm_registry = LLMRegistry()
    llm_registry.register("summarizer", _FakeLLM(completion=_summary_completion()))
    store, session_id = await _seeded_store([_turn(1)])
    tracker = ContextFootprintTracker()
    service = CompactionService(llm_registry, store, _config(keep_recent_turns=4), tracker)

    compacted = await service.compact("researcher", session_id)

    assert compacted is False
    assert await store.get("researcher", session_id) == _turn(1)


async def test_compact_builds_summarizer_call_from_old_messages_and_prompt():
    llm_registry = LLMRegistry()
    summarizer = _FakeLLM(completion=_summary_completion())
    llm_registry.register("summarizer", summarizer)
    store, session_id = await _seeded_store([_turn(1), _turn(2), _turn(3)])
    tracker = ContextFootprintTracker()
    service = CompactionService(
        llm_registry, store, _config(keep_recent_turns=1, prompt="Summarize this."), tracker
    )

    await service.compact("researcher", session_id)

    sent = summarizer.complete_calls[0]
    assert sent == [*_turn(1), *_turn(2), Message(role="user", content="Summarize this.")]
    _assert_anthropic_safe_shape(sent)


async def test_compact_prepends_a_summary_turn_before_the_recent_turns():
    llm_registry = LLMRegistry()
    llm_registry.register("summarizer", _FakeLLM(completion=_summary_completion("gist")))
    store, session_id = await _seeded_store([_turn(1), _turn(2), _turn(3)])
    tracker = ContextFootprintTracker()
    service = CompactionService(llm_registry, store, _config(keep_recent_turns=1), tracker)

    compacted = await service.compact("researcher", session_id)

    assert compacted is True
    stored = await store.get("researcher", session_id)
    assert stored == _compacted(_turn(3), "gist")
    _assert_anthropic_safe_shape(stored)


async def test_compact_leaves_recent_turns_content_untouched():
    # The summary turn is prepended, not merged into recent[0]'s own content.
    llm_registry = LLMRegistry()
    llm_registry.register("summarizer", _FakeLLM(completion=_summary_completion()))
    store, session_id = await _seeded_store([_turn(1), _turn(2), _turn(3)])
    tracker = ContextFootprintTracker()
    service = CompactionService(llm_registry, store, _config(keep_recent_turns=1), tracker)

    await service.compact("researcher", session_id)

    stored = await store.get("researcher", session_id)
    assert stored[-len(_turn(3)) :] == _turn(3)
    assert len(stored) == len(_turn(3)) + 2
    _assert_anthropic_safe_shape(stored)


async def test_compact_given_keep_zero_stores_the_summary_and_a_placeholder():
    llm_registry = LLMRegistry()
    llm_registry.register("summarizer", _FakeLLM(completion=_summary_completion("gist")))
    store, session_id = await _seeded_store([_turn(1), _turn(2)])
    tracker = ContextFootprintTracker()
    service = CompactionService(llm_registry, store, _config(keep_recent_turns=0), tracker)

    await service.compact("researcher", session_id)

    stored = await store.get("researcher", session_id)
    assert stored == _compacted([], "gist")
    _assert_anthropic_safe_shape(stored)


async def test_compact_given_keep_zero_ends_history_on_an_assistant_message():
    # AgentRunService builds every call as [system, *history, new user message], so stored
    # history must end on role="assistant" or the next turn's append makes two consecutive
    # role="user" messages. ["user", "assistant"] both starts and ends correctly.
    llm_registry = LLMRegistry()
    llm_registry.register("summarizer", _FakeLLM(completion=_summary_completion()))
    store, session_id = await _seeded_store([_turn(1), _turn(2)])
    tracker = ContextFootprintTracker()
    service = CompactionService(llm_registry, store, _config(keep_recent_turns=0), tracker)

    await service.compact("researcher", session_id)

    stored = await store.get("researcher", session_id)
    assert [message.role for message in stored] == ["user", "assistant"]


async def test_compact_given_keep_zero_survives_the_next_turns_appended_user():
    # Replay what AgentRunService does on the very next call: history + one new user message.
    llm_registry = LLMRegistry()
    llm_registry.register("summarizer", _FakeLLM(completion=_summary_completion()))
    store, session_id = await _seeded_store([_turn(1), _turn(2)])
    tracker = ContextFootprintTracker()
    service = CompactionService(llm_registry, store, _config(keep_recent_turns=0), tracker)

    await service.compact("researcher", session_id)

    next_call = [
        *await store.get("researcher", session_id),
        Message(role="user", content="next question"),
    ]
    _assert_anthropic_safe_shape(next_call)


async def test_compact_leaves_history_starting_with_a_user_message():
    # The invariant that matters: Anthropic rejects a conversation whose first message
    # is role="assistant", and stored history is replayed verbatim on every later turn.
    llm_registry = LLMRegistry()
    llm_registry.register("summarizer", _FakeLLM(completion=_summary_completion()))
    store, session_id = await _seeded_store([_turn(1), _tool_turn(2), _turn(3)])
    tracker = ContextFootprintTracker()
    service = CompactionService(llm_registry, store, _config(keep_recent_turns=1), tracker)

    await service.compact("researcher", session_id)

    _assert_anthropic_safe_shape(await store.get("researcher", session_id))


async def test_compact_leaves_no_two_consecutive_same_role_messages():
    # Anthropic-via-Bedrock hard-rejects consecutive same-role messages, so the summary
    # turn's acknowledgment (role="assistant") always sits between the summary and recent[0]
    # (also role="user").
    llm_registry = LLMRegistry()
    llm_registry.register("summarizer", _FakeLLM(completion=_summary_completion()))
    store, session_id = await _seeded_store([_turn(1), _tool_turn(2), _turn(3)])
    tracker = ContextFootprintTracker()
    service = CompactionService(llm_registry, store, _config(keep_recent_turns=1), tracker)

    await service.compact("researcher", session_id)

    _assert_anthropic_safe_shape(await store.get("researcher", session_id))


async def test_compact_sends_the_old_portion_to_the_summarizer_with_tool_exchanges_intact():
    # `compact` deliberately does no pre-processing of its own: the summarizer call declares no
    # `tools`, and folding tool exchanges out of such a request is the `ILLM` implementation's
    # contractual job (proven against the real outbound payload in
    # `tests/integration/adapters/test_litellm.py`). Asserting the absence of tool content here
    # would test a property this layer no longer provides.
    llm_registry = LLMRegistry()
    summarizer = _FakeLLM(completion=_summary_completion())
    llm_registry.register("summarizer", summarizer)
    store, session_id = await _seeded_store([_turn(1), _tool_turn(2), _turn(3)])
    tracker = ContextFootprintTracker()
    service = CompactionService(
        llm_registry, store, _config(keep_recent_turns=1, prompt="Sum up."), tracker
    )

    await service.compact("researcher", session_id)

    assert summarizer.complete_calls[0] == [
        *_turn(1),
        *_tool_turn(2),
        Message(role="user", content="Sum up."),
    ]


async def test_compact_given_truncated_summary_returns_false_and_leaves_history():
    llm_registry = LLMRegistry()
    llm_registry.register(
        "summarizer", _FakeLLM(completion=_summary_completion(finish_reason="length"))
    )
    store, session_id = await _seeded_store([_turn(1), _turn(2), _turn(3)])
    tracker = ContextFootprintTracker()
    service = CompactionService(llm_registry, store, _config(keep_recent_turns=1), tracker)

    compacted = await service.compact("researcher", session_id)

    assert compacted is False
    assert await store.get("researcher", session_id) == _turn(1) + _turn(2) + _turn(3)


async def test_compact_given_contentless_summary_returns_false_and_leaves_history():
    # `Message` allows role="assistant" with content=None when tool_calls is set. The
    # summarizer declares no tools so this shouldn't happen, but nothing enforces it, and
    # formatting `None` into the summary would silently store the literal text "None".
    llm_registry = LLMRegistry()
    contentless = Completion(
        message=Message(
            role="assistant",
            content=None,
            tool_calls=[
                ToolCall(id="call_1", function=ToolCallFunction(name="echo", arguments="{}"))
            ],
        ),
        usage=Usage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
        finish_reason="stop",
    )
    llm_registry.register("summarizer", _FakeLLM(completion=contentless))
    store, session_id = await _seeded_store([_turn(1), _turn(2), _turn(3)])
    tracker = ContextFootprintTracker()
    service = CompactionService(llm_registry, store, _config(keep_recent_turns=1), tracker)

    compacted = await service.compact("researcher", session_id)

    assert compacted is False
    assert await store.get("researcher", session_id) == _turn(1) + _turn(2) + _turn(3)


async def test_compact_given_empty_summary_returns_false_and_leaves_history():
    llm_registry = LLMRegistry()
    llm_registry.register("summarizer", _FakeLLM(completion=_summary_completion("")))
    store, session_id = await _seeded_store([_turn(1), _turn(2), _turn(3)])
    tracker = ContextFootprintTracker()
    service = CompactionService(llm_registry, store, _config(keep_recent_turns=1), tracker)

    compacted = await service.compact("researcher", session_id)

    assert compacted is False
    assert await store.get("researcher", session_id) == _turn(1) + _turn(2) + _turn(3)


async def test_compact_given_whitespace_summary_returns_false_and_leaves_history():
    # A non-empty but whitespace-only summary would otherwise pass the emptiness check
    # (the stored text isn't literally "" once the "[Summary of earlier conversation]" prefix
    # is added) and silently discard the whole old portion of history for nothing.
    llm_registry = LLMRegistry()
    llm_registry.register("summarizer", _FakeLLM(completion=_summary_completion("   \n  ")))
    store, session_id = await _seeded_store([_turn(1), _turn(2), _turn(3)])
    tracker = ContextFootprintTracker()
    service = CompactionService(llm_registry, store, _config(keep_recent_turns=1), tracker)

    compacted = await service.compact("researcher", session_id)

    assert compacted is False
    assert await store.get("researcher", session_id) == _turn(1) + _turn(2) + _turn(3)


async def test_compact_given_summarizer_overflow_returns_false_and_leaves_history():
    llm_registry = LLMRegistry()
    llm_registry.register(
        "summarizer", _FakeLLM(completion=LLMContextWindowExceededError("too big"))
    )
    store, session_id = await _seeded_store([_turn(1), _turn(2), _turn(3)])
    tracker = ContextFootprintTracker()
    service = CompactionService(llm_registry, store, _config(keep_recent_turns=1), tracker)

    compacted = await service.compact("researcher", session_id)

    assert compacted is False
    assert await store.get("researcher", session_id) == _turn(1) + _turn(2) + _turn(3)


async def test_compact_given_summarizer_rate_limited_returns_false():
    llm_registry = LLMRegistry()
    llm_registry.register("summarizer", _FakeLLM(completion=LLMRateLimitedError("rate limited")))
    store, session_id = await _seeded_store([_turn(1), _turn(2), _turn(3)])
    tracker = ContextFootprintTracker()
    service = CompactionService(llm_registry, store, _config(keep_recent_turns=1), tracker)

    compacted = await service.compact("researcher", session_id)

    assert compacted is False
    assert await store.get("researcher", session_id) == _turn(1) + _turn(2) + _turn(3)


async def test_compact_clears_stale_usage_estimate():
    llm_registry = LLMRegistry()
    agent_model = _FakeLLM(max_input_tokens=1000)
    summarizer = _FakeLLM(completion=_summary_completion())
    llm_registry.register("agent-model", agent_model)
    llm_registry.register("summarizer", summarizer)
    store, session_id = await _seeded_store([_turn(1), _turn(2), _turn(3)])
    tracker = ContextFootprintTracker()
    service = CompactionService(
        llm_registry, store, _config(keep_recent_turns=1, token_budget_pct=0.8), tracker
    )
    tracker.record("researcher", session_id, 900)  # over 1000 * 0.8 = 800

    compacted = await service.compact("researcher", session_id)
    assert compacted is True

    # A successful compact() resets the shared context_tracker's footprint directly — the old
    # footprint no longer describes the now-shrunk history.
    assert tracker.get("researcher", session_id) is None

    # The pre-compaction estimate (900) must not linger: with no fresh usage recorded since
    # the compaction, maybe_compact has nothing to compare against and must no-op, even
    # though a stale 900 would have tripped this budget (1000 * 0.8 = 800).
    await service.maybe_compact("researcher", session_id, "agent-model")

    assert len(summarizer.complete_calls) == 1  # only the first compaction's call


_RESUMMARY = "the earlier conversation was itself already summarized; little detail remains"


async def test_compact_given_only_a_previous_summary_turn_to_resummarize_returns_false():
    # The degenerate shape: a session compacted before, with exactly keep_recent_turns of new
    # turns since. The split lands `old` on the previous synthetic pair alone — re-summarizing
    # it re-adds the prefix and acknowledgment on top of near-zero information, growing history
    # while the bulky recent turns (the actual reason it's over budget) go untouched.
    llm_registry = LLMRegistry()
    summarizer = _FakeLLM(completion=_summary_completion(_RESUMMARY))
    llm_registry.register("summarizer", summarizer)
    store, session_id = await _seeded_store([_compacted([]), _turn(1), _turn(2)])
    original = await store.get("researcher", session_id)
    tracker = ContextFootprintTracker()
    service = CompactionService(llm_registry, store, _config(keep_recent_turns=2), tracker)

    compacted = await service.compact("researcher", session_id)

    assert compacted is False
    assert await store.get("researcher", session_id) == original


async def test_compact_given_only_a_previous_summary_turn_never_calls_the_summarizer():
    # The same degenerate shape, recognized structurally from our own fixed prefix and
    # acknowledgment text — so it costs nothing at all, not one wasted summarizer call
    # discovering after the fact that the result wouldn't shrink.
    llm_registry = LLMRegistry()
    summarizer = _FakeLLM(completion=_summary_completion(_RESUMMARY))
    llm_registry.register("summarizer", summarizer)
    store, session_id = await _seeded_store([_compacted([]), _turn(1), _turn(2)])
    tracker = ContextFootprintTracker()
    service = CompactionService(llm_registry, store, _config(keep_recent_turns=2), tracker)

    await service.compact("researcher", session_id)

    assert summarizer.complete_calls == []


async def test_compact_given_only_a_previous_summary_and_no_recent_turns_still_refuses():
    # Unconditional: even with nothing else in history (this pair IS the entire history,
    # reached e.g. with keep_recent_turns configured as 0), there's still nothing new to
    # fold in — re-summarizing our own prior output alone is refused for free, the same as
    # when other `recent` turns are present. Real compaction resumes automatically once
    # genuinely new content becomes part of `old` again.
    llm_registry = LLMRegistry()
    summarizer = _FakeLLM(completion=_summary_completion("much shorter"))
    llm_registry.register("summarizer", summarizer)
    store, session_id = await _seeded_store([_compacted([], "an unusually large first summary")])
    original = await store.get("researcher", session_id)
    tracker = ContextFootprintTracker()
    service = CompactionService(llm_registry, store, _config(keep_recent_turns=0), tracker)

    compacted = await service.compact("researcher", session_id)

    assert compacted is False
    assert summarizer.complete_calls == []
    assert await store.get("researcher", session_id) == original


async def test_compact_given_a_non_shrinking_summary_never_clears_the_usage_estimate():
    # The wasted-effort half of the same bug: returning True here would also drop the estimate,
    # so the next maybe_compact would no-op instead of retrying once the session grows further.
    llm_registry = LLMRegistry()
    agent_model = _FakeLLM(max_input_tokens=1000)
    llm_registry.register("agent-model", agent_model)
    summarizer = _FakeLLM(completion=_summary_completion(_RESUMMARY))
    llm_registry.register("summarizer", summarizer)
    store, session_id = await _seeded_store([_compacted([]), _turn(1), _turn(2)])
    tracker = ContextFootprintTracker()
    service = CompactionService(
        llm_registry, store, _config(keep_recent_turns=2, token_budget_pct=0.8), tracker
    )
    tracker.record("researcher", session_id, 900)  # over 1000 * 0.8 = 800

    assert await service.compact("researcher", session_id) is False

    await service.maybe_compact("researcher", session_id, "agent-model")

    # The estimate survived the refusal, so the proactive check still fires and calls compact
    # a second time — what matters here. History is unchanged between the two, so both hit
    # the structural fast path and neither spends a summarizer call at all.
    assert tracker.get("researcher", session_id) == 900
    assert summarizer.complete_calls == []


async def test_compact_given_an_ordinary_two_message_old_portion_still_uses_the_shrink_guard():
    # Same message *count* as the synthetic summary pair the fast path recognizes, but not its
    # shape — so the fast path must not swallow it: the summarizer really runs, and the
    # general post-hoc content-length guard is what refuses the non-shrinking result.
    llm_registry = LLMRegistry()
    summarizer = _FakeLLM(completion=_summary_completion("y" * 1000))
    llm_registry.register("summarizer", summarizer)
    store, session_id = await _seeded_store([_turn(1), _turn(2)])
    original = await store.get("researcher", session_id)
    tracker = ContextFootprintTracker()
    service = CompactionService(llm_registry, store, _config(keep_recent_turns=1), tracker)

    compacted = await service.compact("researcher", session_id)

    assert compacted is False
    assert len(summarizer.complete_calls) == 1
    assert await store.get("researcher", session_id) == original


async def test_compact_given_a_genuinely_smaller_summary_still_replaces_history():
    llm_registry = LLMRegistry()
    llm_registry.register("summarizer", _FakeLLM(completion=_summary_completion("gist")))
    store, session_id = await _seeded_store([_turn(n) for n in range(1, 6)])
    tracker = ContextFootprintTracker()
    service = CompactionService(llm_registry, store, _config(keep_recent_turns=1), tracker)

    compacted = await service.compact("researcher", session_id)

    assert compacted is True
    assert await store.get("researcher", session_id) == _compacted(_turn(5), "gist")


async def test_compact_given_a_summary_smaller_by_only_one_character_still_replaces_history():
    # The guard rejects "equal or bigger", not "not much smaller" — a marginal but real
    # reduction must still go through.
    llm_registry = LLMRegistry()
    store, session_id = await _seeded_store([_turn(1), _turn(2)])
    old_chars = sum(len(message.content or "") for message in _turn(1))
    # 67 fixed chars: the 33-char prefix, its newline, and the 33-char acknowledgment.
    summary = "x" * (old_chars - 67 - 1)
    llm_registry.register("summarizer", _FakeLLM(completion=_summary_completion(summary)))
    tracker = ContextFootprintTracker()
    service = CompactionService(llm_registry, store, _config(keep_recent_turns=1), tracker)

    compacted = await service.compact("researcher", session_id)

    assert compacted is True
    assert await store.get("researcher", session_id) == _compacted(_turn(2), summary)


async def test_compact_given_active_run_records_summarizer_usage_into_it():
    llm_registry = LLMRegistry()
    llm_registry.register("summarizer", _FakeLLM(completion=_summary_completion("gist")))
    store, session_id = await _seeded_store([_turn(n) for n in range(1, 6)])
    tracker = ContextFootprintTracker()
    service = CompactionService(llm_registry, store, _config(keep_recent_turns=1), tracker)

    with run_context("researcher", session_id):
        await service.compact("researcher", session_id)

        assert collect_extra_usage().total_tokens == 10


async def test_compact_given_unknown_session_raises_session_not_found_error():
    llm_registry = LLMRegistry()
    llm_registry.register("summarizer", _FakeLLM(completion=_summary_completion()))
    store = InMemorySessionStore()
    tracker = ContextFootprintTracker()
    service = CompactionService(llm_registry, store, _config(keep_recent_turns=1), tracker)

    with pytest.raises(SessionNotFoundError):
        await service.compact("researcher", "does-not-exist")


# — compact: summarizer retried once --------------------------------------------------------


async def test_compact_given_truncated_summary_then_success_stores_the_retried_summary():
    llm_registry = LLMRegistry()
    summarizer = _FakeLLM(
        completion=[
            _summary_completion("cut off", finish_reason="length"),
            _summary_completion("complete"),
        ]
    )
    llm_registry.register("summarizer", summarizer)
    store, session_id = await _seeded_store([_turn(1), _turn(2), _turn(3)])
    tracker = ContextFootprintTracker()
    service = CompactionService(llm_registry, store, _config(keep_recent_turns=1), tracker)

    compacted = await service.compact("researcher", session_id)

    assert compacted is True
    assert await store.get("researcher", session_id) == _compacted(_turn(3), "complete")


async def test_compact_given_empty_summary_then_success_stores_the_retried_summary():
    llm_registry = LLMRegistry()
    summarizer = _FakeLLM(completion=[_summary_completion(""), _summary_completion("gist")])
    llm_registry.register("summarizer", summarizer)
    store, session_id = await _seeded_store([_turn(1), _turn(2), _turn(3)])
    tracker = ContextFootprintTracker()
    service = CompactionService(llm_registry, store, _config(keep_recent_turns=1), tracker)

    compacted = await service.compact("researcher", session_id)

    assert compacted is True
    assert await store.get("researcher", session_id) == _compacted(_turn(3), "gist")


async def test_compact_given_summarizer_llm_error_gives_up_after_exactly_one_attempt():
    llm_registry = LLMRegistry()
    summarizer = _FakeLLM(completion=LLMRateLimitedError("rate limited"))
    llm_registry.register("summarizer", summarizer)
    store, session_id = await _seeded_store([_turn(1), _turn(2), _turn(3)])
    tracker = ContextFootprintTracker()
    service = CompactionService(llm_registry, store, _config(keep_recent_turns=1), tracker)

    compacted = await service.compact("researcher", session_id)

    assert compacted is False
    assert len(summarizer.complete_calls) == 1
    assert await store.get("researcher", session_id) == _turn(1) + _turn(2) + _turn(3)


# — compact: chunked (map-reduce) fallback --------------------------------------------------


def _nine_turns() -> list[list[Message]]:
    return [_turn(n) for n in range(1, 10)]


async def test_compact_given_summarizer_overflow_falls_back_to_smaller_chunked_calls():
    # 9 turns, keeping 1: the old portion is 8 turns, split into 2 chunks of 4.
    llm_registry = LLMRegistry()
    summarizer = _FakeLLM(
        completion=[LLMContextWindowExceededError("too big"), _summary_completion()]
    )
    llm_registry.register("summarizer", summarizer)
    store, session_id = await _seeded_store(_nine_turns())
    tracker = ContextFootprintTracker()
    service = CompactionService(llm_registry, store, _config(keep_recent_turns=1), tracker)

    await service.compact("researcher", session_id)

    single_pass, *chunked = summarizer.complete_calls
    assert len(chunked) == 3  # two chunk summaries (map) plus one combining call (reduce)
    assert all(len(call) < len(single_pass) for call in chunked)
    for call in chunked:
        _assert_anthropic_safe_shape(call)


async def test_compact_given_summarizer_overflow_stores_the_reduced_summary():
    llm_registry = LLMRegistry()
    summarizer = _FakeLLM(
        completion=[LLMContextWindowExceededError("too big"), _summary_completion("gist")]
    )
    llm_registry.register("summarizer", summarizer)
    store, session_id = await _seeded_store(_nine_turns())
    tracker = ContextFootprintTracker()
    service = CompactionService(llm_registry, store, _config(keep_recent_turns=1), tracker)

    compacted = await service.compact("researcher", session_id)

    assert compacted is True
    stored = await store.get("researcher", session_id)
    assert stored == _compacted(_turn(9), "gist")
    _assert_anthropic_safe_shape(stored)


async def test_compact_given_a_chunk_produces_unusable_content_once_then_succeeds_stores_the_reduced_summary():  # noqa: E501
    llm_registry = LLMRegistry()
    summarizer = _FakeLLM(
        completion=[
            LLMContextWindowExceededError("too big"),  # single pass overflows
            _summary_completion(""),  # chunk 1, attempt 1: empty (unusable) content
            _summary_completion("chunk1"),  # chunk 1, attempt 2 (retry): succeeds
            _summary_completion("chunk2"),  # chunk 2: succeeds first try
            _summary_completion("final"),  # reduce: succeeds first try
        ]
    )
    llm_registry.register("summarizer", summarizer)
    store, session_id = await _seeded_store(_nine_turns())
    tracker = ContextFootprintTracker()
    service = CompactionService(llm_registry, store, _config(keep_recent_turns=1), tracker)

    compacted = await service.compact("researcher", session_id)

    assert compacted is True
    assert len(summarizer.complete_calls) == 5  # the retry, not a second failure, is why
    stored = await store.get("researcher", session_id)
    assert stored == _compacted(_turn(9), "final")


async def test_compact_given_the_reduce_call_produces_unusable_content_once_then_succeeds_stores_the_reduced_summary():  # noqa: E501
    llm_registry = LLMRegistry()
    summarizer = _FakeLLM(
        completion=[
            LLMContextWindowExceededError("too big"),  # single pass overflows
            _summary_completion("chunk1"),  # chunk 1: succeeds first try
            _summary_completion("chunk2"),  # chunk 2: succeeds first try
            _summary_completion("cut off", finish_reason="length"),  # reduce, attempt 1: truncated
            _summary_completion("final"),  # reduce, attempt 2 (retry): succeeds
        ]
    )
    llm_registry.register("summarizer", summarizer)
    store, session_id = await _seeded_store(_nine_turns())
    tracker = ContextFootprintTracker()
    service = CompactionService(llm_registry, store, _config(keep_recent_turns=1), tracker)

    compacted = await service.compact("researcher", session_id)

    assert compacted is True
    assert len(summarizer.complete_calls) == 5
    stored = await store.get("researcher", session_id)
    assert stored == _compacted(_turn(9), "final")


async def test_compact_given_chunked_fallback_sends_each_chunk_summary_to_the_reduce_call():
    llm_registry = LLMRegistry()
    summarizer = _FakeLLM(
        completion=[LLMContextWindowExceededError("too big"), _summary_completion("part")]
    )
    llm_registry.register("summarizer", summarizer)
    store, session_id = await _seeded_store(_nine_turns())
    tracker = ContextFootprintTracker()
    service = CompactionService(
        llm_registry, store, _config(keep_recent_turns=1, prompt="Sum up."), tracker
    )

    await service.compact("researcher", session_id)

    reduce_call = summarizer.complete_calls[-1]
    # A single merged message, not one per chunk plus a separate prompt message — multiple
    # consecutive role="user" messages here would violate the alternation Anthropic-via-
    # Bedrock requires, the same bug class this file's other fixes exist to prevent.
    assert reduce_call == [
        Message(
            role="user",
            content=(
                "Summary of an earlier part of the conversation:\npart\n\n"
                "Summary of an earlier part of the conversation:\npart\n\n"
                "Sum up."
            ),
        )
    ]
    _assert_anthropic_safe_shape(reduce_call)


async def test_compact_given_a_chunk_that_also_overflows_returns_false_and_leaves_history():
    llm_registry = LLMRegistry()
    llm_registry.register(
        "summarizer", _FakeLLM(completion=LLMContextWindowExceededError("too big"))
    )
    store, session_id = await _seeded_store(_nine_turns())
    original = [message for turn in _nine_turns() for message in turn]
    tracker = ContextFootprintTracker()
    service = CompactionService(llm_registry, store, _config(keep_recent_turns=1), tracker)

    compacted = await service.compact("researcher", session_id)

    assert compacted is False
    assert await store.get("researcher", session_id) == original


async def test_compact_given_a_chunk_that_errors_returns_false_and_leaves_history():
    llm_registry = LLMRegistry()
    summarizer = _FakeLLM(
        completion=[LLMContextWindowExceededError("too big"), LLMRateLimitedError("rate limited")]
    )
    llm_registry.register("summarizer", summarizer)
    store, session_id = await _seeded_store(_nine_turns())
    original = [message for turn in _nine_turns() for message in turn]
    tracker = ContextFootprintTracker()
    service = CompactionService(llm_registry, store, _config(keep_recent_turns=1), tracker)

    compacted = await service.compact("researcher", session_id)

    assert compacted is False
    assert await store.get("researcher", session_id) == original


async def test_compact_given_the_reduce_call_overflows_returns_false_and_leaves_history():
    llm_registry = LLMRegistry()
    summarizer = _FakeLLM(
        completion=[
            LLMContextWindowExceededError("too big"),
            _summary_completion(),
            _summary_completion(),
            LLMContextWindowExceededError("still too big"),
        ]
    )
    llm_registry.register("summarizer", summarizer)
    store, session_id = await _seeded_store(_nine_turns())
    original = [message for turn in _nine_turns() for message in turn]
    tracker = ContextFootprintTracker()
    service = CompactionService(llm_registry, store, _config(keep_recent_turns=1), tracker)

    compacted = await service.compact("researcher", session_id)

    assert compacted is False
    assert await store.get("researcher", session_id) == original


async def test_compact_given_a_single_old_turn_that_overflows_never_chunks_further():
    # One turn can't be split into more than one chunk, so there's nothing left to try —
    # exactly one summarizer call, and the overflow is not retried with the same input.
    llm_registry = LLMRegistry()
    summarizer = _FakeLLM(completion=LLMContextWindowExceededError("too big"))
    llm_registry.register("summarizer", summarizer)
    store, session_id = await _seeded_store([_turn(1), _turn(2)])
    tracker = ContextFootprintTracker()
    service = CompactionService(llm_registry, store, _config(keep_recent_turns=1), tracker)

    compacted = await service.compact("researcher", session_id)

    assert compacted is False
    assert len(summarizer.complete_calls) == 1


async def test_compact_given_a_smaller_chunk_turns_splits_the_old_portion_into_more_chunks():
    # 9 turns, keeping 1: the old portion is 8 turns — 2 chunks of 4 by default, 4 of 2 here.
    llm_registry = LLMRegistry()
    summarizer = _FakeLLM(
        completion=[LLMContextWindowExceededError("too big"), _summary_completion()]
    )
    llm_registry.register("summarizer", summarizer)
    store, session_id = await _seeded_store(_nine_turns())
    tracker = ContextFootprintTracker()
    service = CompactionService(
        llm_registry, store, _config(keep_recent_turns=1, chunk_turns=2), tracker
    )

    await service.compact("researcher", session_id)

    _single_pass, *chunked = summarizer.complete_calls
    # Four map calls of 2 turns each (2 messages per turn plus the appended prompt), then the
    # single merged reduce call — two map calls of 5 messages each with the default of 4.
    assert [len(call) for call in chunked] == [5, 5, 5, 5, 1]


# — logging --------------------------------------------------------------------------------


async def test_maybe_compact_given_triggered_logs_info(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO, logger="agent.core.services.compaction")
    llm_registry = LLMRegistry()
    llm_registry.register("agent-model", _FakeLLM(max_input_tokens=1000))
    llm_registry.register("summarizer", _FakeLLM(completion=_summary_completion()))
    store, session_id = await _seeded_store([_turn(1), _turn(2), _turn(3)])
    tracker = ContextFootprintTracker()
    service = CompactionService(
        llm_registry, store, _config(keep_recent_turns=1, token_budget_pct=0.8), tracker
    )
    tracker.record("researcher", session_id, 900)

    await service.maybe_compact("researcher", session_id, "agent-model")

    assert any(r.levelno == logging.INFO and "triggered" in r.message for r in caplog.records)


async def test_maybe_compact_given_under_budget_logs_debug(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.DEBUG, logger="agent.core.services.compaction")
    llm_registry = LLMRegistry()
    llm_registry.register("agent-model", _FakeLLM(max_input_tokens=1000))
    store, session_id = await _seeded_store([_turn(1), _turn(2), _turn(3)])
    tracker = ContextFootprintTracker()
    service = CompactionService(
        llm_registry, store, _config(keep_recent_turns=1, token_budget_pct=0.8), tracker
    )
    tracker.record("researcher", session_id, 100)

    await service.maybe_compact("researcher", session_id, "agent-model")

    assert any(r.levelno == logging.DEBUG and "under budget" in r.message for r in caplog.records)


async def test_maybe_compact_given_unresolvable_window_logs_debug(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.DEBUG, logger="agent.core.services.compaction")
    llm_registry = LLMRegistry()
    llm_registry.register(
        "agent-model", _FakeLLM(max_input_tokens_error=LLMError("no data for this model"))
    )
    store, session_id = await _seeded_store([_turn(1), _turn(2), _turn(3)])
    tracker = ContextFootprintTracker()
    service = CompactionService(llm_registry, store, _config(keep_recent_turns=1), tracker)
    tracker.record("researcher", session_id, 900)

    await service.maybe_compact("researcher", session_id, "agent-model")

    assert any(r.levelno == logging.DEBUG and "agent-model" in r.message for r in caplog.records)


async def test_compact_given_nothing_older_than_keep_window_logs_debug_not_error(
    caplog: pytest.LogCaptureFixture,
):
    caplog.set_level(logging.DEBUG, logger="agent.core.services.compaction")
    llm_registry = LLMRegistry()
    llm_registry.register("summarizer", _FakeLLM(completion=_summary_completion()))
    store, session_id = await _seeded_store([_turn(1)])
    tracker = ContextFootprintTracker()
    service = CompactionService(llm_registry, store, _config(keep_recent_turns=4), tracker)

    await service.compact("researcher", session_id)

    # The exact bug this milestone's spec review caught: this benign, expected no-op must
    # never log at ERROR (or WARNING) — only DEBUG.
    assert not any(r.levelno >= logging.WARNING for r in caplog.records)
    assert any(r.levelno == logging.DEBUG for r in caplog.records)


async def test_compact_given_only_a_previous_summary_turn_logs_debug_not_error(
    caplog: pytest.LogCaptureFixture,
):
    caplog.set_level(logging.DEBUG, logger="agent.core.services.compaction")
    llm_registry = LLMRegistry()
    llm_registry.register("summarizer", _FakeLLM(completion=_summary_completion(_RESUMMARY)))
    store, session_id = await _seeded_store([_compacted([]), _turn(1), _turn(2)])
    tracker = ContextFootprintTracker()
    service = CompactionService(llm_registry, store, _config(keep_recent_turns=2), tracker)

    await service.compact("researcher", session_id)

    assert not any(r.levelno >= logging.WARNING for r in caplog.records)
    assert any(r.levelno == logging.DEBUG for r in caplog.records)


async def test_compact_given_chunked_fallback_engaged_logs_warning(
    caplog: pytest.LogCaptureFixture,
):
    caplog.set_level(logging.WARNING, logger="agent.core.services.compaction")
    llm_registry = LLMRegistry()
    summarizer = _FakeLLM(
        completion=[LLMContextWindowExceededError("too big"), _summary_completion()]
    )
    llm_registry.register("summarizer", summarizer)
    store, session_id = await _seeded_store(_nine_turns())
    tracker = ContextFootprintTracker()
    service = CompactionService(llm_registry, store, _config(keep_recent_turns=1), tracker)

    await service.compact("researcher", session_id)

    assert any("chunked" in r.message for r in caplog.records)


async def test_compact_given_non_shrinking_summary_logs_warning(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.WARNING, logger="agent.core.services.compaction")
    llm_registry = LLMRegistry()
    summarizer = _FakeLLM(completion=_summary_completion("y" * 1000))
    llm_registry.register("summarizer", summarizer)
    store, session_id = await _seeded_store([_turn(1), _turn(2)])
    tracker = ContextFootprintTracker()
    service = CompactionService(llm_registry, store, _config(keep_recent_turns=1), tracker)

    await service.compact("researcher", session_id)

    assert any("smaller" in r.message for r in caplog.records)


async def test_compact_given_success_logs_info(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO, logger="agent.core.services.compaction")
    llm_registry = LLMRegistry()
    llm_registry.register("summarizer", _FakeLLM(completion=_summary_completion("gist")))
    store, session_id = await _seeded_store([_turn(1), _turn(2), _turn(3)])
    tracker = ContextFootprintTracker()
    service = CompactionService(llm_registry, store, _config(keep_recent_turns=1), tracker)

    await service.compact("researcher", session_id)

    assert any(r.levelno == logging.INFO and "compacted" in r.message for r in caplog.records)


# — compact: tracing ------------------------------------------------------------------------


async def test_compact_given_something_to_summarize_opens_a_compaction_span(
    monkeypatch: pytest.MonkeyPatch,
):
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(compaction_module, "tracer", provider.get_tracer("test"))
    llm_registry = LLMRegistry()
    llm_registry.register("summarizer", _FakeLLM(completion=_summary_completion()))
    store, session_id = await _seeded_store([_turn(1), _turn(2), _turn(3)])
    tracker = ContextFootprintTracker()
    service = CompactionService(llm_registry, store, _config(keep_recent_turns=1), tracker)

    await service.compact("researcher", session_id)

    assert any(s.name == "compaction" for s in exporter.get_finished_spans())


async def test_compact_given_something_to_summarize_sets_span_attributes(
    monkeypatch: pytest.MonkeyPatch,
):
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(compaction_module, "tracer", provider.get_tracer("test"))
    llm_registry = LLMRegistry()
    llm_registry.register("summarizer", _FakeLLM(completion=_summary_completion("gist")))
    store, session_id = await _seeded_store([_turn(1), _turn(2), _turn(3)])
    tracker = ContextFootprintTracker()
    service = CompactionService(llm_registry, store, _config(keep_recent_turns=1), tracker)

    await service.compact("researcher", session_id)

    [span] = [s for s in exporter.get_finished_spans() if s.name == "compaction"]
    assert span.attributes is not None
    assert span.attributes["gen_ai.agent.name"] == "researcher"
    assert span.attributes["gen_ai.conversation.id"] == session_id
    assert span.attributes["agent_core.compaction.old_content_chars"] > 0
    assert span.attributes["agent_core.compaction.new_content_chars"] > 0
    assert (
        span.attributes["agent_core.compaction.new_content_chars"]
        < span.attributes["agent_core.compaction.old_content_chars"]
    )


async def test_compact_given_unexpected_error_span_records_error_type(
    monkeypatch: pytest.MonkeyPatch,
):
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(compaction_module, "tracer", provider.get_tracer("test"))
    llm_registry = LLMRegistry()
    llm_registry.register("summarizer", _FakeLLM(completion=RuntimeError("boom")))
    store, session_id = await _seeded_store([_turn(1), _turn(2), _turn(3)])
    tracker = ContextFootprintTracker()
    service = CompactionService(llm_registry, store, _config(keep_recent_turns=1), tracker)

    with pytest.raises(RuntimeError):
        await service.compact("researcher", session_id)

    [span] = [s for s in exporter.get_finished_spans() if s.name == "compaction"]
    assert span.attributes["error.type"] == "RuntimeError"
    assert span.status.description is None
    assert span.events == ()


async def test_compact_given_unexpected_error_span_records_it_when_capturing_content(
    monkeypatch: pytest.MonkeyPatch,
):
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(compaction_module, "tracer", provider.get_tracer("test"))
    llm_registry = LLMRegistry()
    llm_registry.register("summarizer", _FakeLLM(completion=RuntimeError("boom")))
    store, session_id = await _seeded_store([_turn(1), _turn(2), _turn(3)])
    tracker = ContextFootprintTracker()
    service = CompactionService(llm_registry, store, _config(keep_recent_turns=1), tracker)

    set_capture_content(True)
    try:
        with pytest.raises(RuntimeError):
            await service.compact("researcher", session_id)
    finally:
        set_capture_content(False)

    [span] = [s for s in exporter.get_finished_spans() if s.name == "compaction"]
    assert span.attributes["error.type"] == "RuntimeError"
    assert span.status.description == "boom"
    assert len(span.events) == 1


async def test_compact_given_nothing_to_summarize_opens_no_compaction_span(
    monkeypatch: pytest.MonkeyPatch,
):
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(compaction_module, "tracer", provider.get_tracer("test"))
    llm_registry = LLMRegistry()
    llm_registry.register("summarizer", _FakeLLM(completion=_summary_completion()))
    store, session_id = await _seeded_store([_turn(1)])
    tracker = ContextFootprintTracker()
    service = CompactionService(llm_registry, store, _config(keep_recent_turns=4), tracker)

    await service.compact("researcher", session_id)

    assert not any(s.name == "compaction" for s in exporter.get_finished_spans())


async def test_compact_given_truncated_summary_logs_error(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.ERROR, logger="agent.core.services.compaction")
    llm_registry = LLMRegistry()
    llm_registry.register(
        "summarizer", _FakeLLM(completion=_summary_completion(finish_reason="length"))
    )
    store, session_id = await _seeded_store([_turn(1), _turn(2), _turn(3)])
    tracker = ContextFootprintTracker()
    service = CompactionService(llm_registry, store, _config(keep_recent_turns=1), tracker)

    await service.compact("researcher", session_id)

    [record] = caplog.records
    assert "max_tokens" in record.message
