"""Tests for api/logging_setup.py's JsonFormatter and RunContextFilter."""

import json
import logging
import sys
from datetime import datetime

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

from agent.api.logging_setup import JsonFormatter, RunContextFilter, TextFormatter
from agent.core.run_context import run_context


def _record(**extra: object) -> logging.LogRecord:
    record = logging.LogRecord(
        name="test.logger",
        level=logging.WARNING,
        pathname="",
        lineno=0,
        msg="something failed: %s",
        args=("detail",),
        exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_json_formatter_renders_valid_json_with_expected_keys():
    formatter = JsonFormatter()
    record = _record(request_id="req-1")

    parsed = json.loads(formatter.format(record))

    assert parsed["level"] == "WARNING"
    assert parsed["logger"] == "test.logger"
    assert parsed["message"] == "something failed: detail"
    assert parsed["request_id"] == "req-1"
    assert "timestamp" in parsed


def test_json_formatter_given_no_request_id_includes_it_as_null():
    formatter = JsonFormatter()
    record = _record(request_id=None)

    parsed = json.loads(formatter.format(record))

    assert parsed["request_id"] is None


def test_json_formatter_promotes_extra_fields_to_top_level_keys():
    formatter = JsonFormatter()
    record = _record(request_id="req-1", exception_type="RateLimitError", status_code=429)

    parsed = json.loads(formatter.format(record))

    assert parsed["exception_type"] == "RateLimitError"
    assert parsed["status_code"] == 429


def test_json_formatter_given_exc_info_includes_traceback():
    formatter = JsonFormatter()
    try:
        raise ValueError("boom")
    except ValueError:
        record = _record(request_id="req-1")
        record.exc_info = sys.exc_info()

    parsed = json.loads(formatter.format(record))

    assert "ValueError" in parsed["exc_info"]


def test_json_formatter_given_extra_field_matching_output_key_does_not_clobber_it():
    formatter = JsonFormatter()
    record = _record(
        request_id="req-1",
        level="CUSTOM_LEVEL",
        logger="CUSTOM_LOGGER",
        timestamp="CUSTOM_TS",
    )

    parsed = json.loads(formatter.format(record))

    assert parsed["level"] == "WARNING"
    assert parsed["logger"] == "test.logger"
    assert parsed["timestamp"] != "CUSTOM_TS"


def test_json_formatter_given_agent_and_session_id_includes_them():
    formatter = JsonFormatter()
    record = _record(request_id=None, agent="researcher", session_id="sess-1")

    parsed = json.loads(formatter.format(record))

    assert parsed["agent"] == "researcher"
    assert parsed["session_id"] == "sess-1"


def test_json_formatter_timestamp_is_rfc3339():
    formatter = JsonFormatter()
    record = _record(request_id=None)

    parsed = json.loads(formatter.format(record))

    # datetime.fromisoformat round-trips any valid RFC3339/ISO-8601 string; a format lacking
    # the "T" separator or a UTC offset would raise here instead.
    parsed_ts = datetime.fromisoformat(parsed["timestamp"])
    assert parsed_ts.tzinfo is not None


def test_text_formatter_appends_extra_fields_not_in_the_format_string():
    formatter = TextFormatter("%(levelname)s %(message)s")
    record = _record(attempt=2)

    line = formatter.format(record)

    assert "attempt=2" in line


def test_text_formatter_given_no_extra_fields_appends_nothing():
    formatter = TextFormatter("%(levelname)s %(message)s")
    record = _record()

    line = formatter.format(record)

    assert "[" not in line


def test_json_formatter_given_no_agent_or_session_id_includes_them_as_null():
    formatter = JsonFormatter()
    record = _record(request_id=None, agent=None, session_id=None)

    parsed = json.loads(formatter.format(record))

    assert parsed["agent"] is None
    assert parsed["session_id"] is None


def test_run_context_filter_given_active_run_stamps_agent_and_session_id():
    record = _record()
    with run_context("researcher", "sess-1"):
        RunContextFilter().filter(record)

    assert record.agent == "researcher"  # type: ignore[attr-defined]
    assert record.session_id == "sess-1"  # type: ignore[attr-defined]


def test_run_context_filter_given_no_active_run_stamps_none():
    record = _record()

    RunContextFilter().filter(record)

    assert record.agent is None  # type: ignore[attr-defined]
    assert record.session_id is None  # type: ignore[attr-defined]


def test_json_formatter_given_trace_id_includes_it():
    formatter = JsonFormatter()
    record = _record(request_id=None, trace_id="abc123", span_id="def456")

    parsed = json.loads(formatter.format(record))

    assert parsed["trace_id"] == "abc123"
    assert parsed["span_id"] == "def456"


def test_json_formatter_given_no_trace_id_includes_it_as_null():
    formatter = JsonFormatter()
    record = _record(request_id=None)

    parsed = json.loads(formatter.format(record))

    assert parsed["trace_id"] is None
    assert parsed["span_id"] is None


def test_run_context_filter_given_active_span_stamps_trace_and_span_id():
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    record = _record()

    with tracer.start_as_current_span("op") as span:
        RunContextFilter().filter(record)
        expected_trace_id = format(span.get_span_context().trace_id, "032x")
        expected_span_id = format(span.get_span_context().span_id, "016x")

    assert record.trace_id == expected_trace_id  # type: ignore[attr-defined]
    assert record.span_id == expected_span_id  # type: ignore[attr-defined]


def test_run_context_filter_given_no_active_span_omits_trace_and_span_id():
    record = _record()

    RunContextFilter().filter(record)

    assert getattr(record, "trace_id", None) is None
    assert getattr(record, "span_id", None) is None


def test_run_context_filter_given_sampled_out_span_omits_trace_and_span_id():
    # A partial sampler still yields a structurally valid (non-zero) trace/span id for a
    # span it decided not to export — is_valid alone can't tell that apart from a real,
    # exported span, only trace_flags.sampled can. Without this check, log lines would
    # reference a trace id that exists in no tracing backend anywhere.
    provider = TracerProvider(sampler=TraceIdRatioBased(0.0))
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    record = _record()

    with tracer.start_as_current_span("op"):
        RunContextFilter().filter(record)

    assert getattr(record, "trace_id", None) is None
    assert getattr(record, "span_id", None) is None
