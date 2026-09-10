"""Tests for CostTracker."""

import pytest

from agent.core.models.usage import Usage
from agent.core.services.cost_tracker import CostTracker


def _usage(total: int, cost: float | None = None) -> Usage:
    return Usage(prompt_tokens=total, completion_tokens=0, total_tokens=total, cost_usd=cost)


def test_session_usage_given_nothing_recorded_returns_none():
    service = CostTracker()

    assert service.session_usage("researcher", "s1") is None


def test_agent_usage_given_nothing_recorded_returns_zero_usage():
    service = CostTracker()

    usage = service.agent_usage("researcher")

    assert usage == Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0, cost_usd=None)


def test_record_accumulates_session_usage_across_calls():
    service = CostTracker()

    service.record("researcher", "s1", _usage(10, cost=0.01))
    service.record("researcher", "s1", _usage(20, cost=0.02))

    result = service.session_usage("researcher", "s1")
    assert result is not None
    assert result.total_tokens == 30
    assert result.cost_usd == pytest.approx(0.03)


def test_record_accumulates_agent_usage_across_sessions():
    service = CostTracker()

    service.record("researcher", "s1", _usage(10, cost=0.01))
    service.record("researcher", "s2", _usage(20, cost=0.02))

    usage = service.agent_usage("researcher")
    assert usage.total_tokens == 30
    assert usage.cost_usd == pytest.approx(0.03)


def test_record_keeps_sessions_of_different_agents_independent():
    service = CostTracker()

    service.record("researcher", "s1", _usage(10))
    service.record("writer", "s1", _usage(5))

    assert service.agent_usage("researcher").total_tokens == 10
    assert service.agent_usage("writer").total_tokens == 5


def test_forget_discards_session_state():
    service = CostTracker()
    service.record("researcher", "s1", _usage(10))

    service.forget("researcher", "s1")

    assert service.session_usage("researcher", "s1") is None


def test_forget_does_not_affect_agent_usage():
    service = CostTracker()
    service.record("researcher", "s1", _usage(10, cost=0.01))

    service.forget("researcher", "s1")

    assert service.agent_usage("researcher").total_tokens == 10


def test_forget_given_unknown_session_does_not_raise():
    service = CostTracker()

    service.forget("researcher", "does-not-exist")  # must not raise


def test_record_never_evicts_on_its_own():
    """Eviction is the session store's call alone, cascaded via `forget()`."""
    service = CostTracker()

    for i in range(50):
        service.record("researcher", f"s{i}", _usage(1))

    assert service.session_usage("researcher", "s0") is not None
