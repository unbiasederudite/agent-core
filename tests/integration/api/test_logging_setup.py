"""Tests for api/logging_setup.py's configure_logging(): handler construction."""

import logging

import pytest

from agent.api.logging_setup import JsonFormatter, configure_logging
from agent.core.models.config import LoggingConfig


class _NullFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = None  # type: ignore[attr-defined]
        if not hasattr(record, "agent"):
            record.agent = None  # type: ignore[attr-defined]
        if not hasattr(record, "session_id"):
            record.session_id = None  # type: ignore[attr-defined]
        return True


@pytest.fixture(autouse=True)
def _reset_root_logger():
    yield
    root = logging.getLogger()
    for handler in root.handlers[:]:
        handler.close()
        root.removeHandler(handler)
    root.setLevel(logging.WARNING)


def test_configure_logging_given_text_format_uses_plain_formatter():
    configure_logging(LoggingConfig(format="text"), _NullFilter())

    root_handler = logging.getLogger().handlers[0]
    assert not isinstance(root_handler.formatter, JsonFormatter)


def test_configure_logging_given_json_format_uses_json_formatter():
    configure_logging(LoggingConfig(format="json"), _NullFilter())

    root_handler = logging.getLogger().handlers[0]
    assert isinstance(root_handler.formatter, JsonFormatter)


def test_configure_logging_always_produces_one_stream_handler():
    configure_logging(LoggingConfig(), _NullFilter())

    handlers = logging.getLogger().handlers
    assert len(handlers) == 1
    assert isinstance(handlers[0], logging.StreamHandler)


def test_configure_logging_called_twice_does_not_accumulate_handlers():
    config = LoggingConfig()

    configure_logging(config, _NullFilter())
    configure_logging(config, _NullFilter())

    assert len(logging.getLogger().handlers) == 1  # not 2


def test_configure_logging_given_multiple_filters_attaches_all_to_every_handler():
    first, second = _NullFilter(), _NullFilter()

    configure_logging(LoggingConfig(), first, second)

    [handler] = logging.getLogger().handlers
    assert first in handler.filters
    assert second in handler.filters


def test_configure_logging_given_text_format_and_no_filters_does_not_raise():
    # No RequestIdFilter/RunContextFilter attached — record.request_id/agent/session_id
    # are never set. The text format string references all three; formatter `defaults`
    # must cover them or this crashes with KeyError the first time anything logs.
    configure_logging(LoggingConfig(format="text"))

    logging.getLogger("smoke").info("no filters attached at all")  # must not raise
