import logging

import pytest

from agent.core.exceptions import SessionBusyError, SessionNotFoundError
from agent.core.models.message import Message
from agent.core.models.usage import Usage
from agent.core.services.context_tracker import ContextFootprintTracker
from agent.core.services.cost_tracker import CostTracker
from agent.core.services.session_service import SessionService
from agent.core.session_stores.in_memory import InMemorySessionStore


async def test_get_history_returns_stored_messages():
    store = InMemorySessionStore()
    session_id = await store.create("researcher")
    await store.append("researcher", session_id, [Message(role="user", content="hi")])
    service = SessionService(store)

    messages = await service.get_history("researcher", session_id)

    assert messages == [Message(role="user", content="hi")]


async def test_get_history_given_unknown_session_raises_session_not_found_error():
    service = SessionService(InMemorySessionStore())

    with pytest.raises(SessionNotFoundError):
        await service.get_history("researcher", "does-not-exist")


async def test_delete_removes_the_session():
    store = InMemorySessionStore()
    session_id = await store.create("researcher")
    service = SessionService(store)

    await service.delete("researcher", session_id)

    with pytest.raises(SessionNotFoundError):
        await store.get("researcher", session_id)


async def test_delete_given_unknown_session_raises_session_not_found_error():
    service = SessionService(InMemorySessionStore())

    with pytest.raises(SessionNotFoundError):
        await service.delete("researcher", "does-not-exist")


async def test_delete_given_currently_busy_raises_session_busy_error():
    store = InMemorySessionStore()
    session_id = await store.create("researcher")
    service = SessionService(store)

    async with store.busy("researcher", session_id):
        with pytest.raises(SessionBusyError):
            await service.delete("researcher", session_id)


async def test_delete_logs_info(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO, logger="agent.core.services.session_service")
    store = InMemorySessionStore()
    session_id = await store.create("researcher")
    service = SessionService(store)

    await service.delete("researcher", session_id)

    assert any("session deleted" in r.message for r in caplog.records)


async def test_delete_given_unknown_session_does_not_log_success(
    caplog: pytest.LogCaptureFixture,
):
    caplog.set_level(logging.INFO, logger="agent.core.services.session_service")
    service = SessionService(InMemorySessionStore())

    with pytest.raises(SessionNotFoundError):
        await service.delete("researcher", "does-not-exist")

    assert caplog.records == []


async def test_delete_forgets_cost_tracker_state():
    store = InMemorySessionStore()
    session_id = await store.create("researcher")
    cost_tracker = CostTracker()
    cost_tracker.record(
        "researcher",
        session_id,
        Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )
    context_tracker = ContextFootprintTracker()
    context_tracker.record("researcher", session_id, 2)
    service = SessionService(store, cost_tracker=cost_tracker, context_tracker=context_tracker)

    await service.delete("researcher", session_id)

    assert cost_tracker.session_usage("researcher", session_id) is None
    assert context_tracker.get("researcher", session_id) is None


async def test_get_usage_returns_recorded_usage():
    store = InMemorySessionStore()
    session_id = await store.create("researcher")
    cost_tracker = CostTracker()
    cost_tracker.record(
        "researcher",
        session_id,
        Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2, cost_usd=0.01),
    )
    context_tracker = ContextFootprintTracker()
    context_tracker.record("researcher", session_id, 2)
    service = SessionService(store, cost_tracker=cost_tracker, context_tracker=context_tracker)

    cumulative, context_tokens = await service.get_usage("researcher", session_id)

    assert cumulative is not None
    assert cumulative.total_tokens == 2
    assert cumulative.cost_usd == pytest.approx(0.01)
    assert context_tokens == 2


async def test_get_usage_given_unknown_session_raises_session_not_found_error():
    service = SessionService(InMemorySessionStore())

    with pytest.raises(SessionNotFoundError):
        await service.get_usage("researcher", "does-not-exist")


async def test_get_usage_given_session_with_no_recorded_usage_returns_none():
    """Reachable for a session that exists but has never completed a run yet."""
    store = InMemorySessionStore()
    session_id = await store.create("researcher")
    service = SessionService(store)

    cumulative, context_tokens = await service.get_usage("researcher", session_id)

    assert cumulative is None
    assert context_tokens is None


async def test_get_usage_given_usage_recorded_but_no_context_footprint_defaults_footprint_only():
    """`get_usage()` composes two independent reads — one missing must not blank the other."""
    store = InMemorySessionStore()
    session_id = await store.create("researcher")
    cost_tracker = CostTracker()
    cost_tracker.record(
        "researcher",
        session_id,
        Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2, cost_usd=0.01),
    )
    service = SessionService(store, cost_tracker=cost_tracker)

    cumulative, context_tokens = await service.get_usage("researcher", session_id)

    assert cumulative is not None
    assert cumulative.total_tokens == 2
    assert cumulative.cost_usd == pytest.approx(0.01)
    assert context_tokens is None


async def test_get_usage_given_context_footprint_recorded_but_no_usage_defaults_usage_only():
    """Same independence, the other way round: footprint known, cumulative usage unknown."""
    store = InMemorySessionStore()
    session_id = await store.create("researcher")
    context_tracker = ContextFootprintTracker()
    context_tracker.record("researcher", session_id, 42)
    service = SessionService(store, context_tracker=context_tracker)

    cumulative, context_tokens = await service.get_usage("researcher", session_id)

    assert cumulative is None
    assert context_tokens == 42


async def test_delete_forgets_context_tracker_state_independently_of_cost_tracker():
    store = InMemorySessionStore()
    session_id = await store.create("researcher")
    context_tracker = ContextFootprintTracker()
    context_tracker.record("researcher", session_id, 42)
    service = SessionService(store, context_tracker=context_tracker)

    await service.delete("researcher", session_id)

    assert context_tracker.get("researcher", session_id) is None
