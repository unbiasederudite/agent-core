"""Tests for api/request_context.py — the request correlation id middleware and filter."""

import asyncio
import logging
import re
from typing import Any

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
from opentelemetry.semconv.schemas import Schemas
from opentelemetry.trace import SpanKind
from starlette.types import Message, Receive, Scope, Send

from agent.api import request_context
from agent.api.request_context import RequestIdFilter, RequestIdMiddleware, current_request_id
from agent.core.tracing import set_capture_content

_HEX32 = re.compile(r"^[0-9a-f]{32}$")


def _app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(RequestIdMiddleware)

    @app.get("/ok")
    async def ok() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/boom")
    async def boom() -> None:
        raise RuntimeError("boom")

    return app


def _app_with_handled_errors() -> FastAPI:
    app = FastAPI()
    app.add_middleware(RequestIdMiddleware)

    @app.get("/handled-503")
    async def handled_503() -> None:
        raise RuntimeError("handled without propagating")

    @app.exception_handler(RuntimeError)
    async def handle_runtime_error(_request: Any, _exc: RuntimeError) -> JSONResponse:
        # Mirrors app.py's own registered exception handlers: a failure gets converted to a
        # response INSIDE self.app(), so no exception ever reaches RequestIdMiddleware itself.
        return JSONResponse(status_code=503, content={"detail": {"message": "unavailable"}})

    @app.get("/not-found-but-fine")
    async def not_found_but_fine() -> None:
        raise HTTPException(status_code=404, detail="nope")

    return app


def _traced_app(exporter: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(request_context, "tracer", provider.get_tracer("test"))
    return _app()


def test_request_id_middleware_given_no_client_header_request_id_is_the_trace_id(
    monkeypatch: pytest.MonkeyPatch,
):
    exporter = InMemorySpanExporter()
    client = TestClient(_traced_app(exporter, monkeypatch))

    response = client.get("/ok")

    request_id = response.headers["X-Request-ID"]
    assert _HEX32.fullmatch(request_id)
    [span] = exporter.get_finished_spans()
    assert span.name == "GET /ok"
    assert span.kind == SpanKind.SERVER
    assert format(span.context.trace_id, "032x") == request_id
    assert span.attributes["http.request.method"] == "GET"
    assert span.attributes["url.path"] == "/ok"
    assert span.attributes["url.scheme"] == "http"
    assert span.attributes["http.route"] == "/ok"
    assert span.attributes["http.response.status_code"] == 200
    assert span.attributes["client.address"] == "testclient"
    assert span.attributes["network.peer.address"] == "testclient"
    assert span.attributes["network.peer.port"] == 50000
    assert span.attributes["server.address"] == "testserver"
    assert span.attributes["server.port"] == 80
    assert span.attributes["network.protocol.version"] == "1.1"


def test_request_id_middleware_given_user_agent_header_records_it(
    monkeypatch: pytest.MonkeyPatch,
):
    exporter = InMemorySpanExporter()
    client = TestClient(_traced_app(exporter, monkeypatch))

    client.get("/ok", headers={"User-Agent": "test-agent/1.0"})

    [span] = exporter.get_finished_spans()
    assert span.attributes["user_agent.original"] == "test-agent/1.0"


def test_request_id_middleware_given_query_string_omits_url_query_by_default(
    monkeypatch: pytest.MonkeyPatch,
):
    # No current route reads a query parameter, but nothing stops a caller from appending
    # one to any URL (a stray ?api_key=... from a misconfigured client) — gated the same as
    # this span's other content-bearing attributes, per OTel's own HTTP semantic conventions.
    exporter = InMemorySpanExporter()
    client = TestClient(_traced_app(exporter, monkeypatch))

    client.get("/ok?a=1&b=2")

    [span] = exporter.get_finished_spans()
    assert "url.query" not in span.attributes


def test_request_id_middleware_given_query_string_records_it_when_capturing_content(
    monkeypatch: pytest.MonkeyPatch,
):
    exporter = InMemorySpanExporter()
    client = TestClient(_traced_app(exporter, monkeypatch))

    set_capture_content(True)
    try:
        client.get("/ok?a=1&b=2")
    finally:
        set_capture_content(False)

    [span] = exporter.get_finished_spans()
    assert span.attributes["url.query"] == "a=1&b=2"


def test_request_id_middleware_given_no_query_string_omits_url_query(
    monkeypatch: pytest.MonkeyPatch,
):
    exporter = InMemorySpanExporter()
    client = TestClient(_traced_app(exporter, monkeypatch))

    client.get("/ok")

    [span] = exporter.get_finished_spans()
    assert "url.query" not in span.attributes


def test_request_id_middleware_given_unmatched_route_span_name_is_method_only(
    monkeypatch: pytest.MonkeyPatch,
):
    exporter = InMemorySpanExporter()
    client = TestClient(_traced_app(exporter, monkeypatch), raise_server_exceptions=False)

    client.get("/no-such-route")

    [span] = exporter.get_finished_spans()
    assert span.name == "GET"
    assert "http.route" not in span.attributes
    assert span.attributes["http.response.status_code"] == 404


def test_request_id_middleware_given_nonstandard_method_normalizes_to_other(
    monkeypatch: pytest.MonkeyPatch,
):
    # Per HTTP semantic conventions, http.request.method MUST be normalized to "_OTHER" for a
    # method not in the known-methods list, with the raw value preserved separately. The span
    # NAME's method placeholder is different: the spec requires the literal "HTTP" there, not
    # "_OTHER" — the two diverge in exactly this one case.
    exporter = InMemorySpanExporter()
    client = TestClient(_traced_app(exporter, monkeypatch), raise_server_exceptions=False)

    client.request("PROPFIND", "/no-such-route")

    [span] = exporter.get_finished_spans()
    assert span.name == "HTTP"
    assert span.attributes["http.request.method"] == "_OTHER"
    assert span.attributes["http.request.method_original"] == "PROPFIND"


def test_request_id_middleware_given_standard_method_omits_method_original(
    monkeypatch: pytest.MonkeyPatch,
):
    exporter = InMemorySpanExporter()
    client = TestClient(_traced_app(exporter, monkeypatch))

    client.get("/ok")

    [span] = exporter.get_finished_spans()
    assert "http.request.method_original" not in span.attributes


def test_request_id_middleware_given_handled_5xx_response_records_error_type(
    monkeypatch: pytest.MonkeyPatch,
):
    # A failure fully handled by a registered exception handler inside self.app() never
    # raises into this middleware at all — only the resulting status code is visible here.
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(request_context, "tracer", provider.get_tracer("test"))
    client = TestClient(_app_with_handled_errors(), raise_server_exceptions=False)

    response = client.get("/handled-503")

    assert response.status_code == 503
    [span] = exporter.get_finished_spans()
    assert span.attributes["http.response.status_code"] == 503
    assert span.attributes["error.type"] == "503"
    assert span.status.status_code == trace.StatusCode.ERROR


def test_request_id_middleware_given_handled_4xx_response_has_no_error_type(
    monkeypatch: pytest.MonkeyPatch,
):
    # Per HTTP semantic conventions, a 4xx is the client's fault, not the server's — it must
    # not be treated as a server-span error.
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(request_context, "tracer", provider.get_tracer("test"))
    client = TestClient(_app_with_handled_errors(), raise_server_exceptions=False)

    response = client.get("/not-found-but-fine")

    assert response.status_code == 404
    [span] = exporter.get_finished_spans()
    assert span.attributes["http.response.status_code"] == 404
    assert span.status.status_code != trace.StatusCode.ERROR
    assert "error.type" not in span.attributes


def test_request_id_middleware_given_client_header_root_span_records_it_as_distinct(
    monkeypatch: pytest.MonkeyPatch,
):
    exporter = InMemorySpanExporter()
    client = TestClient(_traced_app(exporter, monkeypatch))

    response = client.get("/ok", headers={"X-Request-ID": "client-supplied-id"})

    assert response.headers["X-Request-ID"] == "client-supplied-id"
    [span] = exporter.get_finished_spans()
    assert span.attributes["agent_core.client_request_id"] == "client-supplied-id"
    assert format(span.context.trace_id, "032x") != "client-supplied-id"


def test_request_id_middleware_given_span_not_sampled_request_id_is_not_the_trace_id(
    monkeypatch: pytest.MonkeyPatch,
):
    # A partial-sampling sampler still yields a valid (non-zero) trace/span id even when the
    # span is sampled out — is_valid alone can't tell the two cases apart, only trace_flags.sampled
    # can. Force "never sampled" and confirm the fallback fresh uuid kicks in instead of the
    # (unexported, unobservable) trace id.
    exporter = InMemorySpanExporter()
    provider = TracerProvider(sampler=TraceIdRatioBased(0.0))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(request_context, "tracer", provider.get_tracer("test"))
    client = TestClient(_app())

    response = client.get("/ok")

    request_id = response.headers["X-Request-ID"]
    assert _HEX32.fullmatch(request_id)
    assert exporter.get_finished_spans() == ()


def test_request_id_middleware_given_tracing_disabled_ids_are_still_unique_per_request():
    # No monkeypatched tracer here — this exercises the real default no-op tracer, i.e. the
    # out-of-the-box (tracing-off) configuration every other test in this file already runs
    # under. This must keep passing: it is the pre-existing
    # `test_request_id_middleware_given_two_requests_generates_different_ids` behavior, restated
    # explicitly here because it's the regression this task's fix protects.
    client = TestClient(_app())

    first = client.get("/ok").headers["X-Request-ID"]
    second = client.get("/ok").headers["X-Request-ID"]

    assert first != second
    assert first != "0" * 32


async def test_request_id_middleware_given_response_started_exception_omits_description(
    monkeypatch: pytest.MonkeyPatch,
):
    # Proves the fix for a real bug found in Task 5's review: start_as_current_span's own
    # default (record_exception=True, set_status_on_exception=True) would record the same
    # exception a second time when this except clause's manual record_exception/set_status
    # calls let it propagate further — which only happens on the response_started branch (the
    # swallowed-into-a-500 path never lets the exception escape the `with` block, so the span's
    # own defaults never get a chance to fire there). The span must disable those defaults so
    # this except clause's manual calls are the only recording, on this path too. The exception
    # message/event are also gated behind capture_content_enabled(), same as every other span
    # exception site — nothing rules out a future exception whose message echoes request content.
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(request_context, "tracer", provider.get_tracer("test"))

    async def _streaming_then_broken_app(scope: Scope, receive: Receive, send: Send) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        raise RuntimeError("boom mid-stream")

    middleware = RequestIdMiddleware(_streaming_then_broken_app)
    scope: Scope = {"type": "http", "headers": []}

    async def _receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def _send(message: Message) -> None:
        pass

    with pytest.raises(RuntimeError, match="boom mid-stream"):
        await middleware(scope, _receive, _send)

    [span] = exporter.get_finished_spans()
    assert len(span.events) == 0
    assert span.status.status_code == trace.StatusCode.ERROR
    assert span.status.description is None
    assert span.attributes["error.type"] == "RuntimeError"


async def test_request_id_middleware_given_response_started_exception_records_when_capturing(
    monkeypatch: pytest.MonkeyPatch,
):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(request_context, "tracer", provider.get_tracer("test"))

    async def _streaming_then_broken_app(scope: Scope, receive: Receive, send: Send) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        raise RuntimeError("boom mid-stream")

    middleware = RequestIdMiddleware(_streaming_then_broken_app)
    scope: Scope = {"type": "http", "headers": []}

    async def _receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def _send(message: Message) -> None:
        pass

    set_capture_content(True)
    try:
        with pytest.raises(RuntimeError, match="boom mid-stream"):
            await middleware(scope, _receive, _send)
    finally:
        set_capture_content(False)

    [span] = exporter.get_finished_spans()
    assert len(span.events) == 1
    assert span.status.status_code == trace.StatusCode.ERROR
    assert span.status.description == "boom mid-stream"
    assert span.attributes["error.type"] == "RuntimeError"


async def test_request_id_middleware_given_cancelled_error_propagates_without_marking_span_error(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    caplog.set_level(logging.DEBUG, logger="agent.api.request_context")
    # A client-initiated cancellation isn't evidence of an operational problem, the same
    # reasoning that already excludes a ClientError from invoke_agent's span error tracking
    # — so the span is left untouched (no exception event, no ERROR status, no error.type),
    # even though the exception is still logged (at DEBUG, with full detail) and re-raised.
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(request_context, "tracer", provider.get_tracer("test"))

    async def _cancelled_app(scope: Scope, receive: Receive, send: Send) -> None:
        raise asyncio.CancelledError

    middleware = RequestIdMiddleware(_cancelled_app)
    scope: Scope = {"type": "http", "headers": []}

    async def _receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def _send(message: Message) -> None:
        pass

    with pytest.raises(asyncio.CancelledError):
        await middleware(scope, _receive, _send)

    [span] = exporter.get_finished_spans()
    assert not any(event.name == "exception" for event in span.events)
    assert span.status.status_code != trace.StatusCode.ERROR
    assert "error.type" not in span.attributes
    assert len(caplog.records) == 1
    assert caplog.records[0].exc_info is not None


def test_request_id_middleware_given_no_client_header_generates_one_and_echoes_it():
    client = TestClient(_app())

    response = client.get("/ok")

    assert response.headers["X-Request-ID"]


def test_request_id_middleware_given_client_header_echoes_it_unchanged():
    client = TestClient(_app())

    response = client.get("/ok", headers={"X-Request-ID": "client-supplied-id"})

    assert response.headers["X-Request-ID"] == "client-supplied-id"


def test_request_id_middleware_given_two_requests_generates_different_ids():
    client = TestClient(_app())

    first = client.get("/ok").headers["X-Request-ID"]
    second = client.get("/ok").headers["X-Request-ID"]

    assert first != second


def test_request_id_middleware_given_malicious_header_generates_a_fresh_id_instead():
    client = TestClient(_app())

    response = client.get(
        "/ok", headers={"X-Request-ID": "abc] injected-agent=evil session=fake ["}
    )

    echoed = response.headers["X-Request-ID"]
    assert echoed != "abc] injected-agent=evil session=fake ["
    assert "]" not in echoed
    assert "[" not in echoed


def test_request_id_middleware_given_header_with_disallowed_characters_ignores_it():
    client = TestClient(_app())

    response = client.get("/ok", headers={"X-Request-ID": "has spaces"})

    assert response.headers["X-Request-ID"] != "has spaces"


def test_request_id_middleware_given_safe_header_characters_echoes_it_unchanged():
    client = TestClient(_app())

    response = client.get("/ok", headers={"X-Request-ID": "req-123.abc_DEF"})

    assert response.headers["X-Request-ID"] == "req-123.abc_DEF"


def test_request_id_middleware_present_on_error_response_too():
    client = TestClient(_app(), raise_server_exceptions=False)

    response = client.get("/boom")

    assert response.status_code == 500
    assert response.headers["X-Request-ID"]
    request_id = response.headers["X-Request-ID"]
    assert response.json() == {
        "detail": {
            "message": "An unexpected error occurred.",
            "request_id": request_id,
        }
    }


def test_tracer_declares_the_stable_http_semconv_schema_it_follows():
    # This module only ever emits stable general/HTTP semantic-convention attributes, unlike
    # every other tracer in this codebase — so it's the one place a schema_url can be
    # declared accurately. Compared against the installed opentelemetry-semantic-conventions
    # package's own enum, not a duplicated literal, so this can't silently drift from it.
    assert request_context.tracer._schema_url == Schemas.V1_43_0.value


def test_current_request_id_given_no_active_request_returns_none():
    assert current_request_id() is None


def test_request_id_middleware_sets_context_var_visible_during_the_request():
    seen: list[str | None] = []
    app = FastAPI()
    app.add_middleware(RequestIdMiddleware)

    @app.get("/check")
    async def check() -> dict[str, str]:
        seen.append(current_request_id())
        return {"status": "ok"}

    client = TestClient(app)
    client.get("/check", headers={"X-Request-ID": "abc123"})

    assert seen == ["abc123"]


def test_request_id_filter_given_no_active_request_stamps_none():
    filter_ = RequestIdFilter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0, msg="hi", args=(), exc_info=None
    )

    filter_.filter(record)

    assert record.request_id is None


def test_request_id_filter_stamps_record_with_the_active_request_id():
    app = FastAPI()
    app.add_middleware(RequestIdMiddleware)
    captured: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record)

    handler = _Capture()
    handler.addFilter(RequestIdFilter())
    test_logger = logging.getLogger("agent.api.request_context.test")
    test_logger.addHandler(handler)
    test_logger.setLevel(logging.INFO)

    @app.get("/log")
    async def log_something() -> dict[str, str]:
        test_logger.info("inside a request")
        return {"status": "ok"}

    client = TestClient(app)
    client.get("/log", headers={"X-Request-ID": "req-xyz"})
    test_logger.removeHandler(handler)

    [record] = captured
    assert record.request_id == "req-xyz"


def test_request_id_middleware_given_unhandled_exception_logs_error(caplog):
    caplog.set_level(logging.ERROR, logger="agent.api.request_context")
    client = TestClient(_app(), raise_server_exceptions=False)

    client.get("/boom")

    assert len(caplog.records) == 1
    assert caplog.records[0].exc_info is not None
    assert caplog.records[0].exception_type == "RuntimeError"


async def test_request_id_middleware_given_non_http_scope_passes_through_unchanged():
    calls: list[Scope] = []

    async def _stub_app(scope: Scope, receive: Receive, send: Send) -> None:
        calls.append(scope)

    middleware = RequestIdMiddleware(_stub_app)
    scope: Scope = {"type": "lifespan"}

    async def _receive() -> Message:
        raise AssertionError("should not be called")

    async def _send(message: Message) -> None:
        raise AssertionError("should not be called")

    await middleware(scope, _receive, _send)

    assert calls == [scope]
    assert current_request_id() is None


async def test_request_id_middleware_given_exception_after_response_started_reraises():
    async def _streaming_then_broken_app(scope: Scope, receive: Receive, send: Send) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        raise RuntimeError("boom mid-stream")

    middleware = RequestIdMiddleware(_streaming_then_broken_app)
    scope: Scope = {"type": "http", "headers": []}
    sent: list[Message] = []

    async def _receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def _send(message: Message) -> None:
        sent.append(message)

    with pytest.raises(RuntimeError, match="boom mid-stream"):
        await middleware(scope, _receive, _send)

    # The real response.start already went out; no second one was attempted.
    assert len(sent) == 1
