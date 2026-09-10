"""ContextFootprintTracker: the single source of truth for a session's current context size."""


class ContextFootprintTracker:
    """Tracks each session's current context-token footprint."""

    def __init__(self) -> None:
        """Initialize with no recorded footprints."""
        self._footprints: dict[tuple[str, str], int] = {}

    def record(self, agent: str, session_id: str, context_tokens: int) -> None:
        """Record this turn's ending context size for this session.

        Args:
            agent: Agent the session belongs to.
            session_id: Session to record for.
            context_tokens: The turn's ending context size, in tokens.
        """
        self._footprints[(agent, session_id)] = context_tokens

    def get(self, agent: str, session_id: str) -> int | None:
        """Return the context-token footprint last recorded for this session.

        Args:
            agent: Agent the session belongs to.
            session_id: Session to look up.

        Returns:
            int | None: the token count, or `None` if unknown.
        """
        return self._footprints.get((agent, session_id))

    def forget(self, agent: str, session_id: str) -> None:
        """Discard the recorded footprint for (agent, session_id). No-op if never recorded.

        Args:
            agent: Agent the session belongs to.
            session_id: Session to forget.
        """
        self._footprints.pop((agent, session_id), None)
