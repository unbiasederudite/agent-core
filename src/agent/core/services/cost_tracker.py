"""CostTracker: cumulative token/cost tracking per session and per agent."""

from agent.core.models.usage import ZERO_USAGE, Usage, sum_usage


class CostTracker:
    """Tracks cumulative token/cost usage per session and per agent."""

    def __init__(self) -> None:
        """Initialize with no recorded usage."""
        self._session_usage: dict[tuple[str, str], Usage] = {}
        self._agent_usage: dict[str, Usage] = {}

    def record(self, agent: str, session_id: str, turn_usage: Usage) -> None:
        """Fold one run's usage into both the session's and the agent's cumulative totals.

        Args:
            agent: Agent the session belongs to.
            session_id: Session to record for.
            turn_usage: This run's usage.
        """
        key = (agent, session_id)
        self._session_usage[key] = sum_usage(self._session_usage.get(key, ZERO_USAGE), turn_usage)
        self._agent_usage[agent] = sum_usage(self._agent_usage.get(agent, ZERO_USAGE), turn_usage)

    def session_usage(self, agent: str, session_id: str) -> Usage | None:
        """Cumulative usage for this session.

        Args:
            agent: Agent the session belongs to.
            session_id: Session to look up.

        Returns:
            Usage | None: the cumulative usage, or `None` if unknown.
        """
        return self._session_usage.get((agent, session_id))

    def agent_usage(self, agent: str) -> Usage:
        """Cumulative usage for this agent, across every session. All-zero if never run.

        Args:
            agent: Agent to look up.

        Returns:
            Usage: the agent's cumulative usage.
        """
        return self._agent_usage.get(agent, ZERO_USAGE)

    def forget(self, agent: str, session_id: str) -> None:
        """Discard this session's cumulative-usage entry.

        Args:
            agent: Agent the session belongs to.
            session_id: Session to forget.
        """
        self._session_usage.pop((agent, session_id), None)
