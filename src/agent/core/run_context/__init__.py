"""Per-run correlation context and supporting-LLM-call usage accumulator."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import NamedTuple

from agent.core.models.usage import ZERO_USAGE, Usage, sum_usage


class RunContext(NamedTuple):
    """The (agent, session_id) pair one run executes under."""

    agent: str  # Agent the run executes under.
    session_id: str  # Session the run executes under.


_run_context: ContextVar[RunContext | None] = ContextVar("run_context", default=None)
_extra_usage: ContextVar[list[Usage] | None] = ContextVar("extra_usage", default=None)


def current_run_context() -> RunContext | None:
    """Return the current run's (agent, session_id), or None outside a run.

    Returns:
        RunContext | None: the current run context, or `None`.
    """
    return _run_context.get()


@contextmanager
def run_context(agent: str, session_id: str) -> Iterator[None]:
    """Bind `(agent, session_id)` as the current run context for this block.

    Args:
        agent: Agent the run executes under.
        session_id: Session the run executes under.
    """
    context_token = _run_context.set(RunContext(agent, session_id))
    usage_token = _extra_usage.set([])
    try:
        yield
    finally:
        _run_context.reset(context_token)
        _extra_usage.reset(usage_token)


def record_extra_usage(usage: Usage) -> None:
    """Add a supporting LLM call's usage to the current run's accumulator.

    Args:
        usage: The supporting call's usage.
    """
    bucket = _extra_usage.get()
    if bucket is not None:
        bucket.append(usage)


def collect_extra_usage() -> Usage:
    """Sum every usage recorded via `record_extra_usage()` so far this run.

    Returns:
        Usage: the combined usage, or all-zero if none was recorded or no run is active.
    """
    bucket = _extra_usage.get()
    if not bucket:
        return ZERO_USAGE
    total = ZERO_USAGE
    for usage in bucket:
        total = sum_usage(total, usage)
    return total
