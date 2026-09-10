"""Session-lifecycle use cases: reading history and usage, and deleting a session."""

import logging

from agent.core.models.message import Message
from agent.core.models.usage import Usage
from agent.core.protocols.isession_store import ISessionStore
from agent.core.services.context_tracker import ContextFootprintTracker
from agent.core.services.cost_tracker import CostTracker

logger = logging.getLogger(__name__)


class SessionService:
    """Orchestrates session-lifecycle operations: reading history/usage, deleting a session."""

    def __init__(
        self,
        session_store: ISessionStore,
        cost_tracker: CostTracker | None = None,
        context_tracker: ContextFootprintTracker | None = None,
    ) -> None:
        """Initialize with its storage and tracking dependencies.

        Args:
            session_store: Per-conversation message history storage.
            cost_tracker: Cumulative per-session/per-agent token and cost usage.
            context_tracker: Each session's current context-token footprint.
        """
        self._session_store = session_store
        self._cost_tracker = cost_tracker if cost_tracker is not None else CostTracker()
        self._context_tracker = (
            context_tracker if context_tracker is not None else ContextFootprintTracker()
        )

    async def get_history(self, agent: str, session_id: str) -> list[Message]:
        """Return the full stored history for `(agent, session_id)`.

        Args:
            agent: Agent the session belongs to.
            session_id: Session to read.

        Returns:
            list[Message]: the session's stored history.

        Raises:
            SessionNotFoundError: no session exists for this exact pair.
        """
        return await self._session_store.get(agent, session_id)

    async def get_usage(self, agent: str, session_id: str) -> tuple[Usage | None, int | None]:
        """Return (cumulative usage, context_tokens) for this session.

        Args:
            agent: Agent the session belongs to.
            session_id: Session to look up.

        Returns:
            tuple[Usage | None, int | None]: cumulative usage and current context token
            count, each `None` if this session's own tracking record was evicted.

        Raises:
            SessionNotFoundError: no session exists for this exact pair.
        """
        await self._session_store.get(agent, session_id)
        usage = self._cost_tracker.session_usage(agent, session_id)
        context_tokens = self._context_tracker.get(agent, session_id)
        return (usage, context_tokens)

    async def delete(self, agent: str, session_id: str) -> None:
        """Permanently remove `(agent, session_id)` and forget its recorded usage state.

        Args:
            agent: Agent the session belongs to.
            session_id: Session to delete.

        Raises:
            SessionNotFoundError: no session exists for this exact pair.
            SessionBusyError: another operation already holds this session.
        """
        async with self._session_store.busy(agent, session_id):
            await self._session_store.delete(agent, session_id)
            self._cost_tracker.forget(agent, session_id)
            self._context_tracker.forget(agent, session_id)
        logger.info("session deleted: agent=%s session=%s", agent, session_id)
