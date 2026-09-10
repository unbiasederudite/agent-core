"""Tests for InMemorySessionStore."""

import asyncio
import logging

import pytest

from agent.core.exceptions import SessionBusyError, SessionNotFoundError
from agent.core.models.message import Message
from agent.core.session_stores.in_memory import InMemorySessionStore


async def test_create_returns_an_id_with_empty_history():
    store = InMemorySessionStore()

    session_id = await store.create("researcher")

    assert await store.get("researcher", session_id) == []


async def test_append_then_get_round_trips_messages():
    store = InMemorySessionStore()
    session_id = await store.create("researcher")
    messages = [Message(role="user", content="hi"), Message(role="assistant", content="hello")]

    await store.append("researcher", session_id, messages)

    assert await store.get("researcher", session_id) == messages


async def test_append_twice_accumulates_in_order():
    store = InMemorySessionStore()
    session_id = await store.create("researcher")
    first = [Message(role="user", content="hi")]
    second = [Message(role="assistant", content="hello")]

    await store.append("researcher", session_id, first)
    await store.append("researcher", session_id, second)

    assert await store.get("researcher", session_id) == first + second


async def test_get_given_unknown_session_id_raises_session_not_found_error():
    store = InMemorySessionStore()

    with pytest.raises(SessionNotFoundError):
        await store.get("researcher", "does-not-exist")


async def test_append_given_unknown_session_id_raises_session_not_found_error():
    store = InMemorySessionStore()

    with pytest.raises(SessionNotFoundError):
        await store.append("researcher", "does-not-exist", [Message(role="user", content="hi")])


async def test_get_given_session_created_under_a_different_agent_raises_session_not_found_error():
    store = InMemorySessionStore()
    session_id = await store.create("researcher")

    with pytest.raises(SessionNotFoundError):
        await store.get("scheduler", session_id)


async def test_create_given_two_calls_returns_different_ids():
    store = InMemorySessionStore()

    first_id = await store.create("researcher")
    second_id = await store.create("researcher")

    assert first_id != second_id


async def test_sessions_under_different_agents_are_independent():
    store = InMemorySessionStore()
    researcher_id = await store.create("researcher")
    scheduler_id = await store.create("scheduler")
    await store.append("researcher", researcher_id, [Message(role="user", content="a")])

    assert await store.get("researcher", researcher_id) == [Message(role="user", content="a")]
    assert await store.get("scheduler", scheduler_id) == []


async def test_replace_overwrites_existing_history():
    store = InMemorySessionStore()
    session_id = await store.create("researcher")
    await store.append("researcher", session_id, [Message(role="user", content="old")])

    await store.replace("researcher", session_id, [Message(role="assistant", content="summary")])

    assert await store.get("researcher", session_id) == [
        Message(role="assistant", content="summary")
    ]


async def test_replace_given_unknown_session_id_raises_session_not_found_error():
    store = InMemorySessionStore()

    with pytest.raises(SessionNotFoundError):
        await store.replace("researcher", "does-not-exist", [Message(role="user", content="hi")])


async def test_lock_serializes_two_concurrent_holders_of_the_same_session():
    store = InMemorySessionStore()
    session_id = await store.create("researcher")
    order: list[str] = []

    async def _holder(label: str, hold_seconds: float) -> None:
        async with store.lock("researcher", session_id):
            order.append(f"{label}-start")
            await asyncio.sleep(hold_seconds)
            order.append(f"{label}-end")

    await asyncio.gather(_holder("first", 0.05), _holder("second", 0))

    assert order == ["first-start", "first-end", "second-start", "second-end"]


async def test_lock_given_different_session_ids_does_not_block_each_other():
    store = InMemorySessionStore()
    first_id = await store.create("researcher")
    second_id = await store.create("researcher")
    order: list[str] = []

    async def _holder(session_id: str, label: str, hold_seconds: float) -> None:
        async with store.lock("researcher", session_id):
            order.append(f"{label}-start")
            await asyncio.sleep(hold_seconds)
            order.append(f"{label}-end")

    await asyncio.gather(_holder(first_id, "first", 0.05), _holder(second_id, "second", 0))

    # "second" (no hold time, different key) must not wait behind "first"'s longer hold.
    assert order.index("second-end") < order.index("first-end")


async def test_lock_given_contended_logs_debug(caplog: pytest.LogCaptureFixture):
    store = InMemorySessionStore()
    session_id = await store.create("researcher")
    caplog.set_level(logging.DEBUG, logger="agent.core.session_stores.in_memory")
    release = asyncio.Event()

    async def _hold() -> None:
        async with store.lock("researcher", session_id):
            await release.wait()

    async def _wait_for_it() -> None:
        async with store.lock("researcher", session_id):
            pass

    holder = asyncio.create_task(_hold())
    await asyncio.sleep(0)  # let the holder acquire the lock first
    waiter = asyncio.create_task(_wait_for_it())
    await asyncio.sleep(0)  # let the waiter observe the lock as held
    release.set()
    await waiter
    await holder

    assert any("waiting for session lock" in record.message for record in caplog.records)


async def test_lock_given_uncontended_does_not_log(caplog: pytest.LogCaptureFixture):
    store = InMemorySessionStore()
    session_id = await store.create("researcher")
    caplog.set_level(logging.DEBUG, logger="agent.core.session_stores.in_memory")

    async with store.lock("researcher", session_id):
        pass

    assert caplog.records == []


async def test_lock_given_session_already_evicted_does_not_leave_an_orphaned_lock():
    store = InMemorySessionStore()
    session_id = "does-not-exist-in-sessions"

    async with store.lock("researcher", session_id):
        pass

    assert ("researcher", session_id) not in store._locks


async def test_busy_given_uncontended_allows_entry():
    store = InMemorySessionStore()
    session_id = await store.create("researcher")
    entered = False

    async with store.busy("researcher", session_id):
        entered = True

    assert entered


async def test_busy_given_already_held_raises_session_busy_error_immediately():
    store = InMemorySessionStore()
    session_id = await store.create("researcher")
    entered_second = False

    async with store.busy("researcher", session_id):
        with pytest.raises(SessionBusyError):
            async with store.busy("researcher", session_id):
                entered_second = True

    assert not entered_second


async def test_busy_released_after_context_exit_allows_reentry():
    store = InMemorySessionStore()
    session_id = await store.create("researcher")

    async with store.busy("researcher", session_id):
        pass
    async with store.busy("researcher", session_id):
        entered_again = True

    assert entered_again


async def test_busy_released_after_exception_inside_allows_reentry():
    store = InMemorySessionStore()
    session_id = await store.create("researcher")

    with pytest.raises(ValueError, match="boom"):
        async with store.busy("researcher", session_id):
            raise ValueError("boom")

    async with store.busy("researcher", session_id):
        entered_after_error = True

    assert entered_after_error


async def test_busy_given_different_session_ids_does_not_conflict():
    store = InMemorySessionStore()
    first_id = await store.create("researcher")
    second_id = await store.create("researcher")

    async with store.busy("researcher", first_id):
        async with store.busy("researcher", second_id):
            both_entered = True

    assert both_entered


async def test_delete_removes_the_session():
    store = InMemorySessionStore()
    session_id = await store.create("researcher")

    await store.delete("researcher", session_id)

    with pytest.raises(SessionNotFoundError):
        await store.get("researcher", session_id)


async def test_delete_given_unknown_session_id_raises_session_not_found_error():
    store = InMemorySessionStore()

    with pytest.raises(SessionNotFoundError):
        await store.delete("researcher", "does-not-exist")


async def test_delete_does_not_affect_a_different_session():
    store = InMemorySessionStore()
    kept_id = await store.create("researcher")
    deleted_id = await store.create("researcher")
    await store.append("researcher", kept_id, [Message(role="user", content="keep me")])

    await store.delete("researcher", deleted_id)

    assert await store.get("researcher", kept_id) == [Message(role="user", content="keep me")]


async def test_given_max_sessions_none_never_evicts():
    store = InMemorySessionStore(max_sessions=None)
    ids = [await store.create("researcher") for _ in range(50)]

    for session_id in ids:
        assert await store.get("researcher", session_id) == []


async def test_given_over_max_sessions_evicts_the_least_recently_touched():
    store = InMemorySessionStore(max_sessions=2)
    first_id = await store.create("researcher")
    second_id = await store.create("researcher")

    third_id = await store.create("researcher")

    with pytest.raises(SessionNotFoundError):
        await store.get("researcher", first_id)
    assert await store.get("researcher", second_id) == []
    assert await store.get("researcher", third_id) == []


async def test_given_over_max_sessions_calls_on_evict_with_the_evicted_key():
    evicted: list[tuple[str, str]] = []
    store = InMemorySessionStore(
        max_sessions=2, on_evict=lambda agent, sid: evicted.append((agent, sid))
    )
    first_id = await store.create("researcher")
    await store.create("researcher")

    await store.create("researcher")

    assert evicted == [("researcher", first_id)]


async def test_given_no_on_evict_eviction_still_proceeds():
    store = InMemorySessionStore(max_sessions=1)
    first_id = await store.create("researcher")

    await store.create("researcher")

    with pytest.raises(SessionNotFoundError):
        await store.get("researcher", first_id)


async def test_touching_a_session_protects_it_from_the_next_eviction():
    store = InMemorySessionStore(max_sessions=2)
    first_id = await store.create("researcher")
    second_id = await store.create("researcher")
    await store.get("researcher", first_id)  # touch first — now second is oldest

    await store.create("researcher")

    with pytest.raises(SessionNotFoundError):
        await store.get("researcher", second_id)
    assert await store.get("researcher", first_id) == []


async def test_append_counts_as_a_touch_protecting_from_eviction():
    store = InMemorySessionStore(max_sessions=2)
    first_id = await store.create("researcher")
    second_id = await store.create("researcher")
    await store.append("researcher", first_id, [Message(role="user", content="hi")])

    await store.create("researcher")

    with pytest.raises(SessionNotFoundError):
        await store.get("researcher", second_id)
    assert await store.get("researcher", first_id) == [Message(role="user", content="hi")]


async def test_eviction_skips_a_session_whose_lock_is_currently_held():
    store = InMemorySessionStore(max_sessions=1)
    held_id = await store.create("researcher")
    release = asyncio.Event()

    async def _hold() -> None:
        async with store.lock("researcher", held_id):
            await release.wait()

    holder = asyncio.create_task(_hold())
    await asyncio.sleep(0)  # let the holder actually acquire the lock

    new_id = await store.create("researcher")

    assert await store.get("researcher", held_id) == []
    assert await store.get("researcher", new_id) == []
    release.set()
    await holder


async def test_eviction_skips_a_session_currently_marked_busy():
    store = InMemorySessionStore(max_sessions=1)
    busy_id = await store.create("researcher")
    release = asyncio.Event()

    async def _hold() -> None:
        async with store.busy("researcher", busy_id):
            await release.wait()

    holder = asyncio.create_task(_hold())
    await asyncio.sleep(0)

    new_id = await store.create("researcher")

    assert await store.get("researcher", busy_id) == []
    assert await store.get("researcher", new_id) == []
    release.set()
    await holder


async def test_eviction_removes_the_evicted_session_lock_too():
    store = InMemorySessionStore(max_sessions=1)
    first_id = await store.create("researcher")
    async with store.lock("researcher", first_id):
        pass  # create a lock entry for first_id, then release it

    await store.create("researcher")

    assert (("researcher", first_id)) not in store._locks
