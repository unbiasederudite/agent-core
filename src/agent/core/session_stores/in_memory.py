"""In-process, non-durable session-history store."""

import asyncio
import logging
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from agent.core.exceptions import SessionBusyError, SessionNotFoundError
from agent.core.models.message import Message

logger = logging.getLogger(__name__)


class InMemorySessionStore:
    """Session history backed by a process-local dict. Lost on restart."""

    def __init__(
        self,
        max_sessions: int | None = None,
        on_evict: Callable[[str, str], None] | None = None,
    ) -> None:
        """Initialize an empty session store.

        Args:
            max_sessions: Maximum number of sessions kept at once. `None` means unbounded.
            on_evict: Called with `(agent, session_id)` when this store evicts a session,
                so other per-session state keyed the same way can be kept in sync.
        """
        self._sessions: OrderedDict[tuple[str, str], list[Message]] = OrderedDict()
        self._locks: OrderedDict[tuple[str, str], asyncio.Lock] = OrderedDict()
        self._busy: set[tuple[str, str]] = set()
        self._max_sessions = max_sessions
        self._on_evict = on_evict

    def _evict_if_over_capacity(self, protect: tuple[str, str]) -> None:
        """Evict the oldest-touched session once the session count exceeds `max_sessions`.

        Args:
            protect: Session key to never evict.
        """
        if self._max_sessions is None or len(self._sessions) <= self._max_sessions:
            return
        for key in list(self._sessions.keys()):
            if key == protect:
                continue
            lock = self._locks.get(key)
            if key in self._busy or (lock is not None and lock.locked()):
                continue
            del self._sessions[key]
            self._locks.pop(key, None)
            if self._on_evict is not None:
                self._on_evict(*key)
            return

    async def create(self, agent: str) -> str:
        """Allocate a new session under `agent`, with empty history, and return its id.

        Args:
            agent: Agent the new session belongs to.

        Returns:
            str: the new session's id.
        """
        session_id = uuid.uuid4().hex
        key = (agent, session_id)
        self._sessions[key] = []
        self._evict_if_over_capacity(protect=key)
        return session_id

    async def get(self, agent: str, session_id: str) -> list[Message]:
        """Return the stored history for `(agent, session_id)`.

        Args:
            agent: Agent the session belongs to.
            session_id: Session to read.

        Returns:
            list[Message]: the session's stored history.

        Raises:
            SessionNotFoundError: if no session exists for this exact pair.
        """
        key = (agent, session_id)
        if key not in self._sessions:
            raise SessionNotFoundError(f"no session '{session_id}' for agent '{agent}'")
        self._sessions.move_to_end(key)
        return list(self._sessions[key])

    async def append(self, agent: str, session_id: str, messages: list[Message]) -> None:
        """Extend the stored history for `(agent, session_id)` with `messages`, in order.

        Args:
            agent: Agent the session belongs to.
            session_id: Session to extend.
            messages: Messages to append.

        Raises:
            SessionNotFoundError: if no session exists for this exact pair.
        """
        key = (agent, session_id)
        if key not in self._sessions:
            raise SessionNotFoundError(f"no session '{session_id}' for agent '{agent}'")
        self._sessions[key].extend(messages)
        self._sessions.move_to_end(key)

    async def replace(self, agent: str, session_id: str, messages: list[Message]) -> None:
        """Overwrite the stored history for `(agent, session_id)` with `messages` entirely.

        Args:
            agent: Agent the session belongs to.
            session_id: Session to overwrite.
            messages: Messages to store in place of the existing history.

        Raises:
            SessionNotFoundError: if no session exists for this exact pair.
        """
        key = (agent, session_id)
        if key not in self._sessions:
            raise SessionNotFoundError(f"no session '{session_id}' for agent '{agent}'")
        self._sessions[key] = list(messages)

    @asynccontextmanager
    async def lock(self, agent: str, session_id: str) -> AsyncIterator[None]:
        """Hold an exclusive, per-`(agent, session_id)` lock for the duration of the block.

        Args:
            agent: Agent the session belongs to.
            session_id: Session to lock.
        """
        key = (agent, session_id)
        session_lock = self._locks.get(key)
        if session_lock is None:
            session_lock = asyncio.Lock()
            self._locks[key] = session_lock
        else:
            self._locks.move_to_end(key)
        if session_lock.locked():
            logger.debug(
                "waiting for session lock agent=%s session=%s, currently held", agent, session_id
            )
        try:
            async with session_lock:
                yield
        finally:
            if key not in self._sessions:
                self._locks.pop(key, None)

    @asynccontextmanager
    async def busy(self, agent: str, session_id: str) -> AsyncIterator[None]:
        """Reject a second concurrent operation on `(agent, session_id)`, without waiting.

        Args:
            agent: Agent the session belongs to.
            session_id: Session to mark busy.

        Raises:
            SessionBusyError: if the pair is already marked busy.
        """
        key = (agent, session_id)
        if key in self._busy:
            raise SessionBusyError(f"session '{session_id}' for agent '{agent}' is busy")
        self._busy.add(key)
        try:
            yield
        finally:
            self._busy.discard(key)

    async def delete(self, agent: str, session_id: str) -> None:
        """Permanently remove `(agent, session_id)` and all its stored history.

        Args:
            agent: Agent the session belongs to.
            session_id: Session to delete.

        Raises:
            SessionNotFoundError: if no session exists for this exact pair.
        """
        key = (agent, session_id)
        if key not in self._sessions:
            raise SessionNotFoundError(f"no session '{session_id}' for agent '{agent}'")
        del self._sessions[key]
        self._locks.pop(key, None)
