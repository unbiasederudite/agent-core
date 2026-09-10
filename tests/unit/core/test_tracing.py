"""Tests for the process-wide content-capture flag and its supporting span helpers."""

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import INVALID_SPAN_CONTEXT, SpanContext, StatusCode, TraceFlags

from agent.core.tracing import (
    capture_content_enabled,
    has_exported_span,
    record_gated_exception,
    set_capture_content,
    truncate_keeping_recent,
)


def test_capture_content_enabled_defaults_to_false():
    assert capture_content_enabled() is False


def test_set_capture_content_true_then_false_round_trips():
    set_capture_content(True)
    assert capture_content_enabled() is True

    set_capture_content(False)
    assert capture_content_enabled() is False


def test_record_gated_exception_by_default_omits_description_and_event():
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")

    with tracer.start_as_current_span(
        "op", record_exception=False, set_status_on_exception=False
    ) as span:
        record_gated_exception(span, RuntimeError("boom"))

    [finished] = exporter.get_finished_spans()
    assert finished.status.status_code == StatusCode.ERROR
    assert finished.status.description is None
    assert finished.attributes["error.type"] == "RuntimeError"
    assert finished.events == ()


def test_record_gated_exception_when_capturing_content_includes_description_and_event():
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")

    set_capture_content(True)
    try:
        with tracer.start_as_current_span(
            "op", record_exception=False, set_status_on_exception=False
        ) as span:
            record_gated_exception(span, RuntimeError("boom"))
    finally:
        set_capture_content(False)

    [finished] = exporter.get_finished_spans()
    assert finished.status.status_code == StatusCode.ERROR
    assert finished.status.description == "boom"
    assert finished.attributes["error.type"] == "RuntimeError"
    assert len(finished.events) == 1


def test_has_exported_span_given_valid_and_sampled_is_true():
    context = SpanContext(trace_id=1, span_id=1, is_remote=False, trace_flags=TraceFlags(0x01))
    assert has_exported_span(context) is True


def test_has_exported_span_given_valid_but_not_sampled_is_false():
    context = SpanContext(trace_id=1, span_id=1, is_remote=False, trace_flags=TraceFlags(0x00))
    assert has_exported_span(context) is False


def test_has_exported_span_given_invalid_is_false():
    assert has_exported_span(INVALID_SPAN_CONTEXT) is False


def test_truncate_keeping_recent_given_under_cap_returns_content_unchanged():
    assert truncate_keeping_recent("hello", 10) == "hello"


def test_truncate_keeping_recent_given_none_cap_returns_content_unchanged():
    assert truncate_keeping_recent("hello", None) == "hello"


def test_truncate_keeping_recent_given_over_cap_keeps_the_end():
    result = truncate_keeping_recent("0123456789", 4)

    assert result.endswith("6789")
    assert "[truncated, 6 earlier characters]" in result
