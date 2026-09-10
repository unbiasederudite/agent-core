"""Tests for core/run_context.py — the (agent, session_id) correlation ContextVar."""

from agent.core.models.usage import Usage
from agent.core.run_context import (
    collect_extra_usage,
    current_run_context,
    record_extra_usage,
    run_context,
)


def test_current_run_context_given_no_active_run_returns_none():
    assert current_run_context() is None


def test_run_context_given_active_block_returns_agent_and_session_id():
    with run_context("researcher", "sess-1"):
        assert current_run_context() == ("researcher", "sess-1")


def test_run_context_given_block_exits_restores_none():
    with run_context("researcher", "sess-1"):
        pass

    assert current_run_context() is None


def test_run_context_given_nested_blocks_restores_outer_on_exit():
    with run_context("researcher", "sess-1"):
        with run_context("writer", "sess-2"):
            assert current_run_context() == ("writer", "sess-2")
        assert current_run_context() == ("researcher", "sess-1")


def _usage(total_tokens: int, cost_usd: float | None = None) -> Usage:
    return Usage(
        prompt_tokens=total_tokens,
        completion_tokens=0,
        total_tokens=total_tokens,
        cost_usd=cost_usd,
    )


def test_collect_extra_usage_given_no_active_run_returns_zero():
    assert collect_extra_usage() == Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)


def test_collect_extra_usage_given_active_run_with_no_recordings_returns_zero():
    with run_context("researcher", "sess-1"):
        assert collect_extra_usage() == Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)


def test_record_extra_usage_given_no_active_run_is_a_no_op():
    record_extra_usage(_usage(10))

    assert collect_extra_usage() == Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)


def test_collect_extra_usage_given_multiple_recordings_sums_them():
    with run_context("researcher", "sess-1"):
        record_extra_usage(_usage(10, cost_usd=0.01))
        record_extra_usage(_usage(5, cost_usd=0.02))

        total = collect_extra_usage()

    assert total.total_tokens == 15
    assert total.cost_usd == 0.03


def test_record_extra_usage_given_nested_blocks_isolates_each_runs_accumulator():
    with run_context("researcher", "sess-1"):
        record_extra_usage(_usage(10))
        with run_context("writer", "sess-2"):
            record_extra_usage(_usage(3))
            assert collect_extra_usage().total_tokens == 3
        assert collect_extra_usage().total_tokens == 10
