"""Compaction: keeps a session's stored history under a configured token budget."""

import logging
from dataclasses import dataclass
from typing import Literal

from opentelemetry import trace

from agent.core.exceptions import LLMContextWindowExceededError, LLMError
from agent.core.models.config import CompactionConfig
from agent.core.models.message import Message
from agent.core.protocols.illm import ILLM
from agent.core.protocols.isession_store import ISessionStore
from agent.core.registries.llm import LLMRegistry
from agent.core.run_context import record_extra_usage
from agent.core.services.context_tracker import ContextFootprintTracker
from agent.core.tracing import record_gated_exception

logger = logging.getLogger(__name__)

tracer = trace.get_tracer(__name__)

_SUMMARY_PREFIX = "[Summary of earlier conversation]\n"
_ACKNOWLEDGMENT = "Understood — I have that context."


def _user_message_indices(messages: list[Message]) -> list[int]:
    """Indices of every role="user" message in `messages` — each marks one turn's start.

    Args:
        messages: Messages to scan.

    Returns:
        list[int]: indices of every `role="user"` message.
    """
    return [i for i, message in enumerate(messages) if message.role == "user"]


def _split_at_turn_boundary(history: list[Message], keep_recent_turns: int) -> int:
    """Split `history` into an old (summarizable) part and a recent (kept verbatim) part.

    Args:
        history: Full message history to split.
        keep_recent_turns: How many trailing turns to keep verbatim.

    Returns:
        int: index where the recent portion begins.
    """
    user_indices = _user_message_indices(history)
    turn_count = len(user_indices)
    if keep_recent_turns >= turn_count:
        return 0
    if keep_recent_turns == 0:
        return len(history)
    return user_indices[turn_count - keep_recent_turns]


def _chunk_by_turns(messages: list[Message], turns_per_chunk: int) -> list[list[Message]]:
    """Split `messages` into consecutive groups of up to `turns_per_chunk` turns each.

    Args:
        messages: Messages to split.
        turns_per_chunk: Turns per group.

    Returns:
        list[list[Message]]: the message groups, one per chunk.
    """
    user_indices = _user_message_indices(messages)
    if not user_indices:
        return [messages] if messages else []
    boundaries = user_indices[::turns_per_chunk]
    ends = [*boundaries[1:], len(messages)]
    return [messages[start:end] for start, end in zip(boundaries, ends, strict=True)]


def _is_previous_summary_turn(old: list[Message]) -> bool:
    """True if `old` is exactly the synthetic summary pair a prior compaction produced.

    Args:
        old: Candidate message pair to check.

    Returns:
        bool: whether `old` is a prior summary pair.
    """
    if len(old) != 2:
        return False
    first, second = old
    return (
        first.role == "user"
        and (first.content or "").startswith(_SUMMARY_PREFIX)
        and second.role == "assistant"
        and second.content == _ACKNOWLEDGMENT
    )


@dataclass(frozen=True)
class _SummaryOutcome:
    """Result of one summarization attempt."""

    status: Literal["ok", "llm_error", "unusable"]  # Outcome of the attempt.
    text: str | None = None  # The summary text, if `status` is "ok".


class CompactionService:
    """Keeps a session's stored history under its configured token budget."""

    def __init__(
        self,
        llm_registry: LLMRegistry,
        session_store: ISessionStore,
        config: CompactionConfig,
        context_tracker: ContextFootprintTracker,
    ) -> None:
        """Initialize with its registry, store, config, and tracker dependencies.

        Args:
            llm_registry: Registry of available LLM implementations.
            session_store: Per-conversation message history storage.
            config: Compaction settings (summarizer model, budget, keep-window, prompt).
            context_tracker: Each session's current context-token footprint.
        """
        self._llm_registry = llm_registry
        self._session_store = session_store
        self._config = config
        self._context_tracker = context_tracker

    async def maybe_compact(self, agent: str, session_id: str, model: str) -> None:
        """Compact now if the last-recorded context size is over budget for `model`.

        Args:
            agent: The agent this session belongs to.
            session_id: The session to check.
            model: Model whose context window the budget is measured against.
        """
        last = self._context_tracker.get(agent, session_id)
        if last is None:
            return
        try:
            max_input_tokens = self._llm_registry.get(model).max_input_tokens()
        except LLMError:
            logger.debug(
                "skipping proactive compaction check for agent=%s session=%s: context window "
                "unknown for model %s",
                agent,
                session_id,
                model,
            )
            return
        budget = max_input_tokens * self._config.token_budget_pct
        if last > budget:
            logger.info(
                "proactive compaction triggered for agent=%s session=%s: %d tokens over "
                "budget %.0f",
                agent,
                session_id,
                last,
                budget,
            )
            await self.compact(agent, session_id)
        else:
            logger.debug(
                "checked agent=%s session=%s: %d tokens, still under budget %.0f",
                agent,
                session_id,
                last,
                budget,
            )

    async def compact(self, agent: str, session_id: str) -> bool:
        """Summarize the old portion of `(agent, session_id)`'s history, if there is one.

        Args:
            agent: The agent this session belongs to.
            session_id: The session to compact.

        Returns:
            bool: whether the stored history was replaced with a summary.
        """
        async with self._session_store.lock(agent, session_id):
            history = await self._session_store.get(agent, session_id)
            split = _split_at_turn_boundary(history, self._config.keep_recent_turns)
            if split == 0:
                logger.debug(
                    "nothing old enough to summarize for agent=%s session=%s given "
                    "keep_recent_turns=%d",
                    agent,
                    session_id,
                    self._config.keep_recent_turns,
                )
                return False
            old, recent = history[:split], history[split:]
            if _is_previous_summary_turn(old):
                logger.debug(
                    "agent=%s session=%s: old portion is already a summary turn, nothing new "
                    "to fold in",
                    agent,
                    session_id,
                )
                return False
            with tracer.start_as_current_span(
                "compaction", record_exception=False, set_status_on_exception=False
            ) as span:
                span.set_attribute("gen_ai.agent.name", agent)
                span.set_attribute("gen_ai.conversation.id", session_id)
                try:
                    summarizer = self._llm_registry.get(self._config.model)
                    try:
                        summary_content = await self._summarize_with_retry(summarizer, old)
                    except LLMContextWindowExceededError:
                        logger.warning(
                            "single-pass summary overflowed the summarizer for agent=%s "
                            "session=%s, falling back to chunked map-reduce",
                            agent,
                            session_id,
                        )
                        summary_content = await self._summarize_chunked(summarizer, old)
                    if summary_content is None:
                        return False
                    summary_text = f"{_SUMMARY_PREFIX}{summary_content}"
                    new_history = [
                        Message(role="user", content=summary_text),
                        Message(role="assistant", content=_ACKNOWLEDGMENT),
                        *recent,
                    ]
                    # Message *count* would hide this: both sides can be 2 messages. Content
                    # length is what actually answers "did this shrink anything".
                    old_content_chars = sum(len(message.content or "") for message in old)
                    new_content_chars = sum(
                        len(message.content or "") for message in new_history[:2]
                    )
                    span.set_attribute("agent_core.compaction.old_content_chars", old_content_chars)
                    span.set_attribute("agent_core.compaction.new_content_chars", new_content_chars)
                    if new_content_chars >= old_content_chars:
                        logger.warning(
                            "summary for agent=%s session=%s was not smaller than the original "
                            "content (%d >= %d chars), discarding",
                            agent,
                            session_id,
                            new_content_chars,
                            old_content_chars,
                        )
                        return False
                    # Commits immediately on success; a subsequent failure in the caller's own
                    # LLM call does not roll this rewrite back.
                    await self._session_store.replace(agent, session_id, new_history)
                    self._context_tracker.forget(agent, session_id)
                    logger.info(
                        "compacted agent=%s session=%s: %d -> %d chars, %d turn(s) kept verbatim",
                        agent,
                        session_id,
                        old_content_chars,
                        new_content_chars,
                        len(_user_message_indices(recent)),
                    )
                    return True
                except Exception as exc:
                    record_gated_exception(span, exc)
                    raise

    async def _try_summarize(
        self, summarizer: ILLM, messages: list[Message], *, is_retry: bool = False
    ) -> _SummaryOutcome:
        """One summarizer call attempt over `messages` plus the configured prompt.

        Args:
            summarizer: LLM used to generate the summary.
            messages: Conversation slice to summarize.
            is_retry: Whether this is the retry attempt.

        Returns:
            _SummaryOutcome: the call's status and, if successful, the summary text.

        Raises:
            LLMContextWindowExceededError: if the summarizer call itself overflows.
        """
        if messages and messages[-1].role == "user":
            request = [
                *messages[:-1],
                Message(role="user", content=f"{messages[-1].content}\n\n{self._config.prompt}"),
            ]
        else:
            request = [*messages, Message(role="user", content=self._config.prompt)]
        try:
            completion = await summarizer.complete(request)
        except LLMContextWindowExceededError:
            raise
        except LLMError as exc:
            logger.warning(
                "summarizer call failed, giving up (adapter already retried what was "
                "retriable): %s",
                exc,
                extra={"exception_type": type(exc).__name__},
            )
            return _SummaryOutcome(status="llm_error")
        record_extra_usage(completion.usage)
        if completion.finish_reason == "length":
            if is_retry:
                logger.error(
                    "summary was truncated (finish_reason=length) for model %s — "
                    "its own max_tokens is likely too low",
                    self._config.model,
                )
            return _SummaryOutcome(status="unusable")
        content = (completion.message.content or "").strip()
        if not content:
            return _SummaryOutcome(status="unusable")
        return _SummaryOutcome(status="ok", text=completion.message.content)

    async def _summarize_with_retry(self, summarizer: ILLM, messages: list[Message]) -> str | None:
        """One summarization attempt, retried once if the result was unusable, not if it raised.

        Args:
            summarizer: LLM used to generate the summary.
            messages: Conversation slice to summarize.

        Returns:
            str | None: the summary text, or `None` on failure.

        Raises:
            LLMContextWindowExceededError: if a summarizer call overflows.
        """
        outcome = await self._try_summarize(summarizer, messages)
        if outcome.status == "unusable":
            outcome = await self._try_summarize(summarizer, messages, is_retry=True)
        return outcome.text if outcome.status == "ok" else None

    async def _summarize_chunked(self, summarizer: ILLM, messages: list[Message]) -> str | None:
        """Summarize `messages` in chunks, for when a single pass overflows the summarizer.

        Args:
            summarizer: LLM used to generate each chunk's summary.
            messages: Conversation slice to summarize.

        Returns:
            str | None: the final summary text, or `None` on failure.
        """
        chunks = _chunk_by_turns(messages, self._config.chunk_turns)
        if len(chunks) <= 1:
            return None
        chunk_summaries: list[str] = []
        for chunk in chunks:
            try:
                summary = await self._summarize_with_retry(summarizer, chunk)
            except LLMContextWindowExceededError:
                logger.warning(
                    "a chunk's own content still overflowed the summarizer even after chunking"
                )
                return None
            if summary is None:
                return None
            chunk_summaries.append(summary)
        combined_text = "\n\n".join(
            f"Summary of an earlier part of the conversation:\n{s}" for s in chunk_summaries
        )
        combined = [Message(role="user", content=combined_text)]
        try:
            return await self._summarize_with_retry(summarizer, combined)
        except LLMContextWindowExceededError:
            logger.warning("the reduce step (combining chunk summaries) overflowed the summarizer")
            return None
