"""Process-wide content-capture flag, set once at startup."""

from opentelemetry import trace

from agent.core.run_context import current_run_context

_capture_content = False


def set_capture_content(value: bool) -> None:
    """Set whether span-producing call sites attach prompt/response/tool content.

    Args:
        value: `True` to attach content, `False` to attach metadata only.
    """
    global _capture_content
    _capture_content = value


def capture_content_enabled() -> bool:
    """Return whether span-producing call sites should attach content.

    Returns:
        bool: the current content-capture setting.
    """
    return _capture_content


def truncate(content: str, max_chars: int | None) -> str:
    """Cap `content` at `max_chars`, appending a marker noting what was cut.

    Args:
        content: Text to cap.
        max_chars: Cap in characters. `None` means uncapped.

    Returns:
        str: `content`, truncated if it exceeded `max_chars`.
    """
    if max_chars is None or len(content) <= max_chars:
        return content
    return content[:max_chars] + f"\n...[truncated, {len(content) - max_chars} more characters]"


def truncate_keeping_recent(content: str, max_chars: int | None) -> str:
    """Cap `content` at `max_chars`, keeping its end and marking what was cut from the start.

    Args:
        content: Text to cap.
        max_chars: Cap in characters. `None` means uncapped.

    Returns:
        str: `content`, truncated from the start if it exceeded `max_chars`.
    """
    if max_chars is None or len(content) <= max_chars:
        return content
    return f"[truncated, {len(content) - max_chars} earlier characters]...\n" + content[-max_chars:]


def has_exported_span(span_context: trace.SpanContext) -> bool:
    """Return whether `span_context` is both valid and sampled for export.

    Args:
        span_context: The span context to check.

    Returns:
        bool: whether this span context was actually exported somewhere.
    """
    return span_context.is_valid and span_context.trace_flags.sampled


def stamp_run_context(span: trace.Span) -> None:
    """Attach the active run's agent/session ids to `span`, if a run is active.

    Args:
        span: The span to stamp.
    """
    run_context = current_run_context()
    if run_context is not None:
        span.set_attribute("gen_ai.agent.name", run_context.agent)
        span.set_attribute("gen_ai.conversation.id", run_context.session_id)


def record_gated_exception(span: trace.Span, exc: Exception) -> None:
    """Mark `span` as ERROR for `exc`, gating the exception detail behind content capture.

    Args:
        span: The span to mark.
        exc: The exception the span failed with.
    """
    if _capture_content:
        span.record_exception(exc)
        span.set_status(trace.Status(trace.StatusCode.ERROR, str(exc)))
    else:
        span.set_status(trace.Status(trace.StatusCode.ERROR))
    span.set_attribute("error.type", type(exc).__name__)
