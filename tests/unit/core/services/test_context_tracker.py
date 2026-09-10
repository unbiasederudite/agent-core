"""Tests for ContextFootprintTracker."""

from agent.core.services.context_tracker import ContextFootprintTracker


def test_get_given_nothing_recorded_returns_none():
    tracker = ContextFootprintTracker()

    assert tracker.get("researcher", "s1") is None


def test_get_after_record_returns_the_recorded_value():
    tracker = ContextFootprintTracker()

    tracker.record("researcher", "s1", 100)

    assert tracker.get("researcher", "s1") == 100


def test_record_twice_overwrites_rather_than_sums():
    tracker = ContextFootprintTracker()
    tracker.record("researcher", "s1", 100)

    tracker.record("researcher", "s1", 50)

    assert tracker.get("researcher", "s1") == 50


def test_record_keeps_sessions_of_different_agents_independent():
    tracker = ContextFootprintTracker()

    tracker.record("researcher", "s1", 100)
    tracker.record("writer", "s1", 50)

    assert tracker.get("researcher", "s1") == 100
    assert tracker.get("writer", "s1") == 50


def test_forget_removes_an_entry():
    tracker = ContextFootprintTracker()
    tracker.record("researcher", "s1", 100)

    tracker.forget("researcher", "s1")

    assert tracker.get("researcher", "s1") is None


def test_forget_given_unknown_session_does_not_raise():
    tracker = ContextFootprintTracker()

    tracker.forget("researcher", "does-not-exist")  # must not raise


def test_record_never_evicts_on_its_own():
    """Eviction is the session store's call alone, cascaded via `forget()`."""
    tracker = ContextFootprintTracker()

    for i in range(50):
        tracker.record("researcher", f"s{i}", 1)

    assert tracker.get("researcher", "s0") is not None
