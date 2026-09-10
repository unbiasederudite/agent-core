"""Request correlation id, threaded through logging and set once per request."""

import asyncio
import logging
import re
import uuid
from contextvars import ContextVar

from opentelemetry import trace
from opentelemetry.semconv.schemas import Schemas
from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from agent.core.tracing import capture_content_enabled, has_exported_span, record_gated_exception

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__, schema_url=Schemas.V1_43_0.value)

INTERNAL_ERROR_MESSAGE = "An unexpected error occurred."

_SAFE_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,128}")

_KNOWN_HTTP_METHODS = frozenset(
    {"GET", "HEAD", "POST", "PUT", "DELETE", "CONNECT", "OPTIONS", "TRACE", "PATCH", "QUERY"}
)


def current_request_id() -> str | None:
    """Return the current request's correlation id, or `None` outside a request context.

    Returns:
        str | None: the current request id, or `None`.
    """
    return _request_id.get()


class RequestIdMiddleware:
    """Assign or echo an `X-Request-ID` per request."""

    def __init__(self, app: ASGIApp) -> None:
        """Wrap `app`, the next layer in the ASGI middleware/router chain.

        Args:
            app: The next ASGI layer.
        """
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Bind the request id for one HTTP request; pass everything else through unchanged.

        Args:
            scope: The ASGI connection scope.
            receive: The ASGI receive channel.
            send: The ASGI send channel.

        Raises:
            asyncio.CancelledError: the request was cancelled.
            Exception: whatever `self.app` raised, if a response was already partway out
                when it failed.
        """
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        raw_method = scope.get("method", "")
        method = raw_method if raw_method in _KNOWN_HTTP_METHODS else "_OTHER"
        span_method = "HTTP" if method == "_OTHER" else method
        path = scope.get("path", "")
        with tracer.start_as_current_span(
            span_method,  # renamed to "{span_method} {route}" below, once the route is known
            kind=trace.SpanKind.SERVER,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            span.set_attribute("http.request.method", method)
            if method != raw_method:
                span.set_attribute("http.request.method_original", raw_method)
            span.set_attribute("url.path", path)
            span.set_attribute("url.scheme", scope.get("scheme", "http"))
            query_string = scope.get("query_string", b"")
            if query_string and capture_content_enabled():
                span.set_attribute("url.query", query_string.decode("latin-1"))
            client = scope.get("client")
            if client is not None:
                span.set_attribute("client.address", client[0])
                span.set_attribute("network.peer.address", client[0])
                span.set_attribute("network.peer.port", client[1])
            server = scope.get("server")
            if server is not None:
                span.set_attribute("server.address", server[0])
                if server[1] is not None:
                    span.set_attribute("server.port", server[1])
            http_version = scope.get("http_version")
            if http_version:
                span.set_attribute("network.protocol.version", http_version)
            request_headers = Headers(scope=scope)
            user_agent = request_headers.get("user-agent")
            if user_agent is not None:
                span.set_attribute("user_agent.original", user_agent)
            span_context = span.get_span_context()
            client_request_id = request_headers.get("x-request-id")
            client_id_is_safe = client_request_id is not None and bool(
                _SAFE_REQUEST_ID.fullmatch(client_request_id)
            )
            if client_id_is_safe and client_request_id is not None:
                request_id = client_request_id
            elif has_exported_span(span_context):
                request_id = format(span_context.trace_id, "032x")
            else:
                request_id = uuid.uuid4().hex
            if (
                client_id_is_safe
                and has_exported_span(span_context)
                and client_request_id is not None
            ):
                span.set_attribute("agent_core.client_request_id", client_request_id)
            token = _request_id.set(request_id)
            response_started = False
            status_code: int | None = None
            error_type_recorded = False

            async def send_wrapper(message: Message) -> None:
                nonlocal response_started, status_code
                if message["type"] == "http.response.start":
                    response_started = True
                    status_code = message["status"]
                    MutableHeaders(scope=message)["X-Request-ID"] = request_id
                await send(message)

            try:
                await self.app(scope, receive, send_wrapper)
            except asyncio.CancelledError:
                logger.debug("request cancelled", exc_info=True)
                raise
            except Exception as exc:
                logger.error(
                    "unhandled exception",
                    exc_info=True,
                    extra={"exception_type": type(exc).__name__},
                )
                record_gated_exception(span, exc)
                error_type_recorded = True
                if response_started:
                    raise
                status_code = 500
                response = JSONResponse(
                    status_code=status_code,
                    content={
                        "detail": {"message": INTERNAL_ERROR_MESSAGE, "request_id": request_id}
                    },
                )
                response.headers["X-Request-ID"] = request_id
                await response(scope, receive, send)
            finally:
                route = scope.get("route")
                path_format = getattr(route, "path_format", None)
                if path_format is not None:
                    span.set_attribute("http.route", path_format)
                    span.update_name(f"{span_method} {path_format}")
                if status_code is not None:
                    span.set_attribute("http.response.status_code", status_code)
                    if status_code >= 500 and not error_type_recorded:
                        span.set_status(trace.Status(trace.StatusCode.ERROR))
                        span.set_attribute("error.type", str(status_code))
                _request_id.reset(token)


class RequestIdFilter(logging.Filter):
    """Stamps `record.request_id` from the current request's `ContextVar` onto every `LogRecord`."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Add the current request_id to the log record, or None if outside a request.

        Args:
            record: The log record to stamp.

        Returns:
            bool: always `True` — never filters a record out.
        """
        record.request_id = current_request_id()
        return True
