import logging

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from agent.core.exceptions import GuardrailBlockedError
from agent.core.models.guardrail import GuardrailFinding
from agent.core.protocols import iguardrail
from agent.core.protocols.iguardrail import run_guardrails
from agent.core.tracing import set_capture_content


class _Guardrail:
    def __init__(self, name: str, action: str, finding: GuardrailFinding) -> None:
        self.name = name
        self.action = action
        self._finding = finding
        self.checked_with: list[str] = []

    async def check(self, content: str) -> GuardrailFinding:
        self.checked_with.append(content)
        return self._finding


async def test_run_guardrails_given_nothing_triggers_returns_content_unchanged():
    guardrail = _Guardrail("g", "block", GuardrailFinding(triggered=False))

    result = await run_guardrails("hello", [guardrail], "input_guardrails")

    assert result == "hello"


async def test_run_guardrails_given_nothing_triggers_logs_the_passed_check(
    caplog: pytest.LogCaptureFixture,
):
    guardrail = _Guardrail("no-secrets", "block", GuardrailFinding(triggered=False))

    with caplog.at_level(logging.DEBUG):
        await run_guardrails("hello", [guardrail], "input_guardrails")

    assert "no-secrets" in caplog.text
    [record] = [r for r in caplog.records if "no-secrets" in r.message]
    assert isinstance(record.duration_ms, float)


async def test_run_guardrails_given_block_action_triggers_raises_guardrail_blocked_error():
    guardrail = _Guardrail(
        "no-secrets", "block", GuardrailFinding(triggered=True, reason="looks like a key")
    )

    with pytest.raises(GuardrailBlockedError, match="no-secrets"):
        await run_guardrails("sk-abc123", [guardrail], "input_guardrails")


async def test_run_guardrails_given_block_action_error_never_carries_the_reason():
    # finding.reason is free text from a third-party validator this codebase doesn't
    # control — for a secrets/PII detector it can echo back an excerpt of the exact
    # content the guardrail exists to catch. This message reaches the HTTP caller
    # verbatim (GuardrailBlockedError maps to a 4xx via _UNIFORM_ERROR_MAP) and the
    # LLM's own context (as a tool-result message), so it must never carry the reason —
    # not even with content capture on, unlike span/log content elsewhere. The
    # guardrail's own name is the always-safe, generic explanation both still get.
    guardrail = _Guardrail(
        "no-secrets", "block", GuardrailFinding(triggered=True, reason="sk-abc123-the-real-secret")
    )
    set_capture_content(True)
    try:
        with pytest.raises(GuardrailBlockedError) as exc_info:
            await run_guardrails("sk-abc123-the-real-secret", [guardrail], "input_guardrails")
    finally:
        set_capture_content(False)

    assert "sk-abc123-the-real-secret" not in str(exc_info.value)
    assert "no-secrets" in str(exc_info.value)


async def test_run_guardrails_given_triggered_action_always_logs_the_reason(
    caplog: pytest.LogCaptureFixture,
):
    # Logs stay local to this process (never shipped anywhere by this codebase), unlike
    # the exception message above — so the full reason belongs here, unconditionally,
    # for whoever operates this deployment to actually see what was flagged and why.
    caplog.set_level(logging.INFO)
    guardrail = _Guardrail(
        "no-pii", "warn", GuardrailFinding(triggered=True, reason="email: real.user@example.com")
    )

    await run_guardrails("content", [guardrail], "input_guardrails")

    assert any("real.user@example.com" in r.message for r in caplog.records)


async def test_run_guardrails_given_block_action_triggers_logs_before_raising(
    caplog: pytest.LogCaptureFixture,
):
    guardrail = _Guardrail(
        "no-secrets", "block", GuardrailFinding(triggered=True, reason="looks like a key")
    )

    with caplog.at_level(logging.INFO), pytest.raises(GuardrailBlockedError):
        await run_guardrails("sk-abc123", [guardrail], "input_guardrails")

    assert "no-secrets" in caplog.text


async def test_run_guardrails_given_redact_action_triggers_returns_redacted_content():
    guardrail = _Guardrail(
        "no-pii",
        "redact",
        GuardrailFinding(triggered=True, reason="pii", redacted_content="[REDACTED]"),
    )

    result = await run_guardrails("my email is a@b.com", [guardrail], "input_guardrails")

    assert result == "[REDACTED]"


async def test_run_guardrails_given_redact_action_triggers_logs_the_redaction(
    caplog: pytest.LogCaptureFixture,
):
    guardrail = _Guardrail(
        "no-pii",
        "redact",
        GuardrailFinding(triggered=True, reason="pii", redacted_content="[REDACTED]"),
    )

    with caplog.at_level(logging.INFO):
        await run_guardrails("my email is a@b.com", [guardrail], "input_guardrails")

    assert "no-pii" in caplog.text


async def test_run_guardrails_given_redact_action_with_no_fix_returns_content_unchanged():
    guardrail = _Guardrail("no-pii", "redact", GuardrailFinding(triggered=True, reason="pii"))

    result = await run_guardrails("content", [guardrail], "input_guardrails")

    assert result == "content"


async def test_run_guardrails_given_warn_action_triggers_logs_and_returns_content_unchanged(
    caplog: pytest.LogCaptureFixture,
):
    guardrail = _Guardrail(
        "toxicity", "warn", GuardrailFinding(triggered=True, reason="mildly rude")
    )

    with caplog.at_level(logging.WARNING):
        result = await run_guardrails("content", [guardrail], "input_guardrails")

    assert result == "content"
    assert "toxicity" in caplog.text


async def test_run_guardrails_given_multiple_guardrails_feeds_redacted_content_forward():
    first = _Guardrail(
        "first",
        "redact",
        GuardrailFinding(triggered=True, reason="a", redacted_content="stage-1"),
    )
    second = _Guardrail("second", "block", GuardrailFinding(triggered=False))

    result = await run_guardrails("original", [first, second], "input_guardrails")

    assert result == "stage-1"
    assert second.checked_with == ["stage-1"]


async def test_run_guardrails_given_empty_list_returns_content_unchanged():
    result = await run_guardrails("content", [], "input_guardrails")

    assert result == "content"


async def test_run_guardrails_given_warn_then_block_still_evaluates_and_raises_on_the_second():
    first = _Guardrail("first", "warn", GuardrailFinding(triggered=True, reason="mild"))
    second = _Guardrail("second", "block", GuardrailFinding(triggered=True, reason="severe"))

    with pytest.raises(GuardrailBlockedError, match="second"):
        await run_guardrails("original", [first, second], "input_guardrails")

    assert second.checked_with == ["original"]


async def test_run_guardrails_opens_a_checkpoint_span_and_a_child_per_guardrail(
    monkeypatch: pytest.MonkeyPatch,
):
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(iguardrail, "tracer", provider.get_tracer("test"))
    guardrail = _Guardrail("no-secrets", "block", GuardrailFinding(triggered=False))

    await run_guardrails("hello", [guardrail], "input_guardrails")

    spans = exporter.get_finished_spans()
    [checkpoint_span] = [s for s in spans if s.name == "input_guardrails"]
    [validator_span] = [s for s in spans if s.name == "guardrail no-secrets"]
    assert validator_span.parent.span_id == checkpoint_span.context.span_id
    assert validator_span.attributes["agent_core.guardrail.triggered"] is False
    assert validator_span.attributes["agent_core.guardrail.action"] == "block"


async def test_run_guardrails_given_block_action_triggers_spans_are_not_error_status(
    monkeypatch: pytest.MonkeyPatch,
):
    # A block is expected, working behavior, not a system failure — the propagating
    # GuardrailBlockedError should not inflate error-rate dashboards by marking these spans
    # ERROR (OTel's own default set_status_on_exception=True would do that otherwise).
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(iguardrail, "tracer", provider.get_tracer("test"))
    guardrail = _Guardrail(
        "no-secrets", "block", GuardrailFinding(triggered=True, reason="looks like a key")
    )

    with pytest.raises(GuardrailBlockedError):
        await run_guardrails("sk-abc123", [guardrail], "input_guardrails")

    spans = exporter.get_finished_spans()
    [checkpoint_span] = [s for s in spans if s.name == "input_guardrails"]
    [validator_span] = [s for s in spans if s.name == "guardrail no-secrets"]
    assert checkpoint_span.status.status_code != StatusCode.ERROR
    assert validator_span.status.status_code != StatusCode.ERROR
    # set_status_on_exception=False alone doesn't stop the SDK from auto-recording an
    # "exception" event on propagation — record_exception=False is needed too, or a block
    # (expected, not a failure) still gets an unwanted exception event on both spans.
    assert len(checkpoint_span.events) == 0
    assert len(validator_span.events) == 0


class _RaisingGuardrail:
    name = "broken"
    action = "block"

    async def check(self, content: str) -> GuardrailFinding:
        raise RuntimeError("validator backend unreachable")


async def test_run_guardrails_given_check_raises_spans_omit_description_by_default(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    # Unlike a deliberate block, a guardrail's own check() crashing is a genuine failure and
    # must be visible in tracing — distinct from the block case above. A custom check()
    # implementation can raise with content-bearing text, so the description is gated the
    # same as `finding.reason` a few lines below.
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(iguardrail, "tracer", provider.get_tracer("test"))
    caplog.set_level(logging.ERROR, logger="agent.core.protocols.iguardrail")

    with pytest.raises(RuntimeError, match="validator backend unreachable"):
        await run_guardrails("hello", [_RaisingGuardrail()], "input_guardrails")

    assert len(caplog.records) == 1
    assert caplog.records[0].exc_info is not None
    assert caplog.records[0].exception_type == "RuntimeError"

    spans = exporter.get_finished_spans()
    [checkpoint_span] = [s for s in spans if s.name == "input_guardrails"]
    [validator_span] = [s for s in spans if s.name == "guardrail broken"]
    assert checkpoint_span.status.status_code == StatusCode.ERROR
    assert checkpoint_span.status.description is None
    assert checkpoint_span.attributes["error.type"] == "RuntimeError"
    assert validator_span.status.status_code == StatusCode.ERROR
    assert validator_span.status.description is None
    assert validator_span.attributes["error.type"] == "RuntimeError"
    assert validator_span.events == ()
    assert checkpoint_span.events == ()


async def test_run_guardrails_given_check_raises_spans_record_the_error_when_capturing_content(
    monkeypatch: pytest.MonkeyPatch,
):
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(iguardrail, "tracer", provider.get_tracer("test"))
    set_capture_content(True)
    try:
        with pytest.raises(RuntimeError, match="validator backend unreachable"):
            await run_guardrails("hello", [_RaisingGuardrail()], "input_guardrails")
    finally:
        set_capture_content(False)

    spans = exporter.get_finished_spans()
    [checkpoint_span] = [s for s in spans if s.name == "input_guardrails"]
    [validator_span] = [s for s in spans if s.name == "guardrail broken"]
    assert checkpoint_span.status.description == "validator backend unreachable"
    assert validator_span.status.description == "validator backend unreachable"
    # Exactly one, not two: record_exception=False on span creation stops the SDK from
    # auto-recording a second "exception" event on top of the manual record_exception() call.
    assert [e.name for e in validator_span.events] == ["exception"]
    assert [e.name for e in checkpoint_span.events] == ["exception"]
