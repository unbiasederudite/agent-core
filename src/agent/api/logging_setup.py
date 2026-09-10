"""Logging setup: JSON/text formatting, a console handler, and filters."""

import json
import logging
from datetime import UTC, datetime

from opentelemetry import trace

from agent.core.models.config import LoggingConfig
from agent.core.run_context import current_run_context
from agent.core.tracing import has_exported_span

_LOGRECORD_DEFAULT_ATTRS = frozenset(vars(logging.LogRecord("", 0, "", 0, "", (), None)).keys())

_FORMATTER_OWN_KEYS = frozenset(
    {
        "message",
        "asctime",
        "exc_text",
        "timestamp",
        "level",
        "logger",
        "request_id",
        "agent",
        "session_id",
        "trace_id",
        "span_id",
    }
)

_STANDARD_ATTRS = _LOGRECORD_DEFAULT_ATTRS | _FORMATTER_OWN_KEYS


def _rfc3339(record: logging.LogRecord) -> str:
    """Render a record's creation time as an RFC3339/ISO-8601 UTC timestamp.

    Args:
        record: The log record to read the timestamp from.

    Returns:
        str: the timestamp, e.g. `"2026-09-08T12:34:56.789012+00:00"`.
    """
    return datetime.fromtimestamp(record.created, tz=UTC).isoformat()


def _extra_fields(record: logging.LogRecord) -> dict[str, object]:
    """Return the record's `extra=` fields — everything not already a standard/formatter key.

    Args:
        record: The log record to read fields from.

    Returns:
        dict[str, object]: the non-standard fields, keyed by name.
    """
    return {key: value for key, value in record.__dict__.items() if key not in _STANDARD_ATTRS}


class JsonFormatter(logging.Formatter):
    """Renders each `LogRecord` as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as a JSON string.

        Args:
            record: The log record to format.

        Returns:
            str: the log record as a JSON line.
        """
        payload: dict[str, object] = {
            "timestamp": _rfc3339(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", None),
            "agent": getattr(record, "agent", None),
            "session_id": getattr(record, "session_id", None),
            "trace_id": getattr(record, "trace_id", None),
            "span_id": getattr(record, "span_id", None),
        }
        payload.update(_extra_fields(record))
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class TextFormatter(logging.Formatter):
    """Renders each `LogRecord` as one line, appending any non-standard `extra=` fields."""

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as one line, with any extra fields appended.

        Args:
            record: The log record to format.

        Returns:
            str: the formatted line.
        """
        line = super().format(record)
        extras = _extra_fields(record)
        if not extras:
            return line
        rendered = " ".join(f"{key}={value}" for key, value in extras.items())
        return f"{line} [{rendered}]"


class RunContextFilter(logging.Filter):
    """Stamps run-context and active-span identifiers onto each log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Stamp the record's agent/session_id and trace_id/span_id fields.

        Args:
            record: The log record to stamp.

        Returns:
            bool: always `True` — never filters a record out.
        """
        context = current_run_context()
        record.agent, record.session_id = context if context is not None else (None, None)
        span_context = trace.get_current_span().get_span_context()
        exported = has_exported_span(span_context)
        record.trace_id = format(span_context.trace_id, "032x") if exported else None
        record.span_id = format(span_context.span_id, "016x") if exported else None
        return True


def configure_logging(config: LoggingConfig, *extra_filters: logging.Filter) -> None:
    """Wire up the console handler, the chosen formatter, and the level from `config`.

    Args:
        config: Logging settings (level, format).
        extra_filters: Additional log filters to attach.
    """
    formatter: logging.Formatter = (
        JsonFormatter()
        if config.format == "json"
        else TextFormatter(
            "%(asctime)s %(levelname)s %(name)s "
            "[request_id=%(request_id)s agent=%(agent)s session=%(session_id)s "
            "trace_id=%(trace_id)s span_id=%(span_id)s] %(message)s",
            defaults={
                "request_id": None,
                "agent": None,
                "session_id": None,
                "trace_id": None,
                "span_id": None,
            },
        )
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    for extra_filter in extra_filters:
        handler.addFilter(extra_filter)
    logging.basicConfig(level=config.level, handlers=[handler], force=True)
