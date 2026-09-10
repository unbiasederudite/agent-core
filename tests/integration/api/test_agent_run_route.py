import asyncio
import logging

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent.api import app as app_module
from agent.api.app import add_agent_run_route, add_exception_handlers
from agent.api.request_context import RequestIdMiddleware
from agent.core.exceptions import (
    AgentError,
    AgentNotFoundError,
    ClientError,
    GuardrailBlockedError,
    GuardrailNotFoundError,
    LLMError,
    LLMNotFoundError,
    LLMOverloadedError,
    LLMRateLimitedError,
    LLMTimeoutError,
    ModelNotAllowedError,
    RequestTimeoutError,
    SessionBusyError,
    SessionNotFoundError,
    StrategyNotAllowedError,
    StrategyNotFoundError,
    ToolNotAllowedError,
    ToolNotFoundError,
)
from agent.core.models.message import Message, ToolCall, ToolCallFunction
from agent.core.models.run import Run
from agent.core.models.usage import Usage
from agent.core.services.agent_run import AgentRunService


class _StubAgentRunService(AgentRunService):
    def __init__(self, result: Run | Exception) -> None:
        self._result = result
        self.last_agent: str | None = None
        self.last_message: str | None = None
        self.last_strategy: str | None = None
        self.last_session_id: str | None = None

    async def run(
        self,
        message: str,
        agent: str,
        *,
        model: str | None = None,
        strategy: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        tools: list[str] | None = None,
        session_id: str | None = None,
    ) -> Run:
        self.last_agent = agent
        self.last_message = message
        self.last_strategy = strategy
        self.last_session_id = session_id
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class _BlockingAgentRunService(AgentRunService):
    """Signals `started` on entry, then blocks until `release` is set."""

    def __init__(self, started: asyncio.Event, release: asyncio.Event) -> None:
        self._started = started
        self._release = release

    async def run(
        self,
        message: str,
        agent: str,
        *,
        model: str | None = None,
        strategy: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        tools: list[str] | None = None,
        session_id: str | None = None,
    ) -> Run:
        self._started.set()
        await self._release.wait()
        return _run()


def _run(finish_reason: str = "stop", session_id: str = "sess_1") -> Run:
    return Run(
        model="openai/gpt-4o",
        response=Message(role="assistant", content="hello!"),
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        finish_reason=finish_reason,
        session_id=session_id,
    )


def _client_for(result: Run | Exception) -> TestClient:
    app = FastAPI()
    app.add_middleware(RequestIdMiddleware)
    add_exception_handlers(app)
    add_agent_run_route(app, _StubAgentRunService(result))
    return TestClient(app, raise_server_exceptions=False)


def test_run_agent_given_success_returns_200_with_message():
    client = _client_for(_run())

    response = client.post("/v1/agents/researcher", json={"message": "hi"})

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "openai/gpt-4o"
    assert body["message"]["content"] == "hello!"
    assert body["usage"]["total_tokens"] == 2
    assert body["finish_reason"] == "stop"
    assert body["message"]["tool_calls"] is None
    assert body["session_id"] == "sess_1"


def test_run_agent_uses_the_path_segment_as_the_agent_name():
    service = _StubAgentRunService(_run())
    app = FastAPI()
    add_exception_handlers(app)
    add_agent_run_route(app, service)
    client = TestClient(app)

    client.post("/v1/agents/researcher", json={"message": "hi"})

    assert service.last_agent == "researcher"


def test_run_agent_passes_the_body_message_through():
    service = _StubAgentRunService(_run())
    app = FastAPI()
    add_exception_handlers(app)
    add_agent_run_route(app, service)
    client = TestClient(app)

    client.post("/v1/agents/researcher", json={"message": "what time is it?"})

    assert service.last_message == "what time is it?"


def test_run_agent_passes_the_body_strategy_through():
    service = _StubAgentRunService(_run())
    app = FastAPI()
    add_exception_handlers(app)
    add_agent_run_route(app, service)
    client = TestClient(app)

    client.post("/v1/agents/researcher", json={"message": "hi", "strategy": "rewoo"})

    assert service.last_strategy == "rewoo"


def test_run_agent_passes_the_body_session_id_through():
    service = _StubAgentRunService(_run())
    app = FastAPI()
    add_exception_handlers(app)
    add_agent_run_route(app, service)
    client = TestClient(app)

    client.post("/v1/agents/researcher", json={"message": "hi", "session_id": "sess_9"})

    assert service.last_session_id == "sess_9"


def test_run_agent_given_no_session_id_in_body_passes_none_through():
    service = _StubAgentRunService(_run())
    app = FastAPI()
    add_exception_handlers(app)
    add_agent_run_route(app, service)
    client = TestClient(app)

    client.post("/v1/agents/researcher", json={"message": "hi"})

    assert service.last_session_id is None


def test_run_agent_given_non_stop_finish_reason_is_not_hardcoded():
    client = _client_for(_run(finish_reason="length"))

    response = client.post("/v1/agents/researcher", json={"message": "hi"})

    assert response.json()["finish_reason"] == "length"


def test_run_agent_given_unknown_agent_returns_404():
    client = _client_for(AgentNotFoundError("nope"))

    response = client.post("/v1/agents/nope", json={"message": "hi"})

    assert response.status_code == 404
    detail = response.json()["detail"]
    assert detail["message"] == "nope"
    assert detail["code"] == "agent_not_found"


def test_run_agent_given_unknown_model_override_returns_404():
    client = _client_for(LLMNotFoundError("nope/nope"))

    response = client.post("/v1/agents/researcher", json={"message": "hi", "model": "nope/nope"})

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "model_not_found"


def test_run_agent_given_unknown_strategy_override_returns_404():
    client = _client_for(StrategyNotFoundError("nope"))

    response = client.post("/v1/agents/researcher", json={"message": "hi", "strategy": "nope"})

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "strategy_not_found"


def test_run_agent_given_unknown_tool_returns_404():
    client = _client_for(ToolNotFoundError("nope"))

    response = client.post("/v1/agents/researcher", json={"message": "hi", "tools": ["nope"]})

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "tool_not_found"


def test_run_agent_given_unknown_guardrail_returns_404():
    client = _client_for(GuardrailNotFoundError("nope"))

    response = client.post("/v1/agents/researcher", json={"message": "hi"})

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "guardrail_not_found"


def test_run_agent_given_unknown_session_id_returns_404():
    client = _client_for(SessionNotFoundError("nope"))

    response = client.post(
        "/v1/agents/researcher", json={"message": "hi", "session_id": "does-not-exist"}
    )

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "session_not_found"


def test_run_agent_given_llm_failure_returns_502():
    client = _client_for(LLMError("provider down"))

    response = client.post("/v1/agents/researcher", json={"message": "hi"})

    assert response.status_code == 502
    detail = response.json()["detail"]
    assert detail["message"] == "The request to the model provider failed. Please try again."
    assert "provider down" not in detail["message"]
    assert detail["request_id"]


def test_run_agent_given_rate_limited_error_returns_429():
    client = _client_for(LLMRateLimitedError("rate limited"))

    response = client.post("/v1/agents/researcher", json={"message": "hi"})

    assert response.status_code == 429


def test_run_agent_given_rate_limited_error_sets_retry_after_header():
    client = _client_for(LLMRateLimitedError("rate limited"))

    response = client.post("/v1/agents/researcher", json={"message": "hi"})

    assert response.headers["Retry-After"]


def test_run_agent_given_timeout_error_returns_504():
    client = _client_for(LLMTimeoutError("timed out"))

    response = client.post("/v1/agents/researcher", json={"message": "hi"})

    assert response.status_code == 504


def test_run_agent_given_client_error_not_in_the_uniform_map_returns_400_not_500():
    # Safety net: a ClientError subtype not (yet) listed in _UNIFORM_ERROR_MAP must still
    # degrade to a generic 4xx, never the 500 the final AgentError catch-all would otherwise
    # produce — a deliberate rejection must never look like a server failure to the caller.
    client = _client_for(ClientError("some deliberate rejection"))

    response = client.post("/v1/agents/researcher", json={"message": "hi"})

    assert response.status_code == 400
    assert response.json()["detail"]["message"] == "some deliberate rejection"


def _all_subclasses(cls: type) -> set[type]:
    return set(cls.__subclasses__()).union(
        *(_all_subclasses(subclass) for subclass in cls.__subclasses__())
    )


def test_uniform_error_map_covers_every_client_error_subtype():
    # The `except ClientError` safety net in add_agent_run_route exists specifically so a
    # ClientError subtype missing from _UNIFORM_ERROR_MAP still degrades to a generic 4xx
    # instead of the safety net becoming the LIVE path for a type someone forgot to map.
    # This test makes that omission fail loudly instead of silently falling through.
    assert _all_subclasses(ClientError) == set(app_module._UNIFORM_ERROR_MAP)


def test_run_agent_given_generic_agent_error_returns_500(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.ERROR, logger="agent.api.app")
    client = _client_for(AgentError("something unexpected"))

    response = client.post("/v1/agents/researcher", json={"message": "hi"})

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert detail["message"] == "An unexpected error occurred."
    assert "something unexpected" not in detail["message"]
    assert detail["request_id"]
    [record] = [r for r in caplog.records if "unhandled AgentError" in r.message]
    assert record.exception_type == "AgentError"


def test_run_agent_given_missing_message_field_returns_400():
    client = _client_for(_run())

    response = client.post("/v1/agents/researcher", json={})

    assert response.status_code == 400
    assert "param" in response.json()["detail"]


def test_run_agent_given_unexpected_exception_returns_500(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.ERROR, logger="agent.api.app")
    client = _client_for(RuntimeError("boom"))

    response = client.post("/v1/agents/researcher", json={"message": "hi"})

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert detail["message"] == "An unexpected error occurred."
    assert "boom" not in detail["message"]
    assert detail["request_id"]
    [record] = [r for r in caplog.records if "unhandled exception" in r.message]
    assert record.exception_type == "RuntimeError"


def test_run_agent_given_tool_calls_message_serializes_correctly():
    run = Run(
        model="openai/gpt-4o",
        response=Message(
            role="assistant",
            content=None,
            tool_calls=[
                ToolCall(
                    id="call_1",
                    function=ToolCallFunction(name="get_current_time", arguments="{}"),
                )
            ],
        ),
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        finish_reason="tool_calls",
        session_id="sess_1",
    )
    client = _client_for(run)

    response = client.post(
        "/v1/agents/researcher", json={"message": "what time is it?", "tools": ["get_current_time"]}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["finish_reason"] == "tool_calls"
    assert body["message"]["content"] is None
    assert body["message"]["tool_calls"][0]["function"]["name"] == "get_current_time"


def test_unmatched_route_returns_404():
    client = _client_for(_run())

    response = client.get("/does-not-exist")

    assert response.status_code == 404
    assert response.json() == {"detail": {"message": "Not Found"}}


def test_wrong_http_method_returns_405():
    client = _client_for(_run())

    response = client.get("/v1/agents/researcher")

    assert response.status_code == 405
    assert response.json() == {"detail": {"message": "Method Not Allowed"}}


def test_run_agent_given_overloaded_error_returns_503():
    client = _client_for(LLMOverloadedError("model at capacity"))

    response = client.post("/v1/agents/researcher", json={"message": "hi"})

    assert response.status_code == 503
    assert (
        response.json()["detail"]["message"]
        == "This model is at capacity. Please try again shortly."
    )


def test_run_agent_given_overloaded_error_sets_retry_after_header():
    client = _client_for(LLMOverloadedError("model at capacity"))

    response = client.post("/v1/agents/researcher", json={"message": "hi"})

    assert response.headers["Retry-After"]


def test_run_agent_given_unknown_agent_logs_info(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO, logger="agent.api.app")
    client = _client_for(AgentNotFoundError("nope"))

    client.post("/v1/agents/nope", json={"message": "hi"})

    assert any(r.levelno == logging.INFO and "404" in r.message for r in caplog.records)


def test_run_agent_given_rate_limited_does_not_log_in_app_layer(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.DEBUG, logger="agent.api.app")
    client = _client_for(LLMRateLimitedError("rate limited"))

    client.post("/v1/agents/researcher", json={"message": "hi"})

    # Already logged upstream in litellm.py with the same request_id — app.py adds nothing.
    assert [r for r in caplog.records if r.name == "agent.api.app"] == []


def test_run_agent_given_generic_agent_error_logs_error_with_traceback(
    caplog: pytest.LogCaptureFixture,
):
    caplog.set_level(logging.ERROR, logger="agent.api.app")
    client = _client_for(AgentError("something unexpected"))

    client.post("/v1/agents/researcher", json={"message": "hi"})

    [record] = caplog.records
    assert record.exc_info is not None


def test_run_agent_given_unexpected_exception_logs_error_with_traceback(
    caplog: pytest.LogCaptureFixture,
):
    # A bare RuntimeError matches none of add_agent_run_route's except clauses (they only
    # list AgentError and its subtypes), so it propagates past app.py entirely and is
    # caught by RequestIdMiddleware's own fallback — Starlette's ServerErrorMiddleware
    # (which owns handlers for the literal Exception class) sits outside every middleware
    # added via app.add_middleware(), so app.py's own @app.exception_handler(Exception) is
    # structurally unreachable for this exact case (see api/README.md). This test checks
    # the logger that actually fires, not app.py's — checking agent.api.app here would
    # silently pass by capturing nothing, giving false confidence in dead code.
    caplog.set_level(logging.ERROR, logger="agent.api.request_context")
    client = _client_for(RuntimeError("boom"))

    client.post("/v1/agents/researcher", json={"message": "hi"})

    [record] = caplog.records
    assert record.exc_info is not None


def test_unmatched_route_returns_404_and_logs_info(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO, logger="agent.api.app")
    client = _client_for(_run())

    response = client.get("/does-not-exist")

    assert response.status_code == 404
    assert any("404" in r.message for r in caplog.records)


def test_wrong_http_method_returns_405_and_logs_info(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO, logger="agent.api.app")
    client = _client_for(_run())

    response = client.get("/v1/agents/researcher")

    assert response.status_code == 405
    assert any("405" in r.message for r in caplog.records)


def test_wrong_http_method_returns_405_with_allow_header():
    client = _client_for(_run())

    response = client.get("/v1/agents/researcher")

    assert response.status_code == 405
    assert "Allow" in response.headers


def test_run_agent_given_missing_message_field_logs_info(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO, logger="agent.api.app")
    client = _client_for(_run())

    client.post("/v1/agents/researcher", json={})

    assert any(r.levelno == logging.INFO for r in caplog.records)


def test_run_agent_given_session_busy_returns_409():
    client = _client_for(SessionBusyError("busy"))

    response = client.post("/v1/agents/researcher", json={"message": "hi"})

    assert response.status_code == 409


def test_run_agent_given_tool_not_allowed_returns_403():
    client = _client_for(ToolNotAllowedError("nope"))

    response = client.post("/v1/agents/researcher", json={"message": "hi"})

    assert response.status_code == 403


def test_run_agent_given_model_not_allowed_returns_403():
    client = _client_for(ModelNotAllowedError("nope"))

    response = client.post("/v1/agents/researcher", json={"message": "hi"})

    assert response.status_code == 403


def test_run_agent_given_strategy_not_allowed_returns_403():
    client = _client_for(StrategyNotAllowedError("nope"))

    response = client.post("/v1/agents/researcher", json={"message": "hi"})

    assert response.status_code == 403


def test_run_agent_given_request_timeout_returns_504():
    client = _client_for(RequestTimeoutError("too slow"))

    response = client.post("/v1/agents/researcher", json={"message": "hi"})

    assert response.status_code == 504


def test_run_agent_given_guardrail_blocked_error_returns_422():
    client = _client_for(GuardrailBlockedError("blocked by guardrail 'no_profanity'"))

    response = client.post("/v1/agents/researcher", json={"message": "trigger"})

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["message"] == "blocked by guardrail 'no_profanity'"
    assert detail["code"] == "guardrail_blocked"


async def test_run_agent_given_max_concurrent_requests_reached_returns_429():
    started = asyncio.Event()
    release = asyncio.Event()
    app = FastAPI()
    add_exception_handlers(app)
    add_agent_run_route(app, _BlockingAgentRunService(started, release), max_concurrent_requests=1)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        first = asyncio.create_task(client.post("/v1/agents/researcher", json={"message": "hi"}))
        await started.wait()

        second = await client.post("/v1/agents/researcher", json={"message": "hi"})
        release.set()
        first_response = await first

    assert second.status_code == 429
    assert second.headers["Retry-After"] == "5"
    assert second.json()["detail"]["code"] == "too_many_concurrent_requests"
    assert first_response.status_code == 200


async def test_run_agent_given_a_slot_frees_up_a_later_request_proceeds():
    started = asyncio.Event()
    release = asyncio.Event()
    app = FastAPI()
    add_exception_handlers(app)
    add_agent_run_route(app, _BlockingAgentRunService(started, release), max_concurrent_requests=1)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        first = asyncio.create_task(client.post("/v1/agents/researcher", json={"message": "hi"}))
        await started.wait()
        release.set()
        await first

        started.clear()
        release.clear()
        second = asyncio.create_task(client.post("/v1/agents/researcher", json={"message": "hi"}))
        await started.wait()
        release.set()
        second_response = await second

    assert second_response.status_code == 200


def test_run_agent_given_no_max_concurrent_requests_configured_stays_unbounded():
    client = _client_for(_run())

    response = client.post("/v1/agents/researcher", json={"message": "hi"})

    assert response.status_code == 200
