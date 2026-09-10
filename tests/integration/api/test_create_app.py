"""Hermetic wiring tests for create_app(): build_registries -> AgentRunService -> routes."""

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider

from agent.api.app import create_app
from agent.core.exceptions import ConfigError


def _fake_litellm_response() -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hello!"), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def _agent_config_path(tmp_path: Path, **agent_overrides: object) -> Path:
    config_path = tmp_path / "app_config.json"
    agent = {
        "name": "researcher",
        "system_prompt": "You are a research assistant.",
        "model": "openai/gpt-4o",
        "strategy": "react",
        **agent_overrides,
    }
    config_path.write_text(
        json.dumps(
            {
                "llms": [{"model": "openai/gpt-4o"}],
                "strategies": [{"name": "react"}],
                "agents": [agent],
            }
        )
    )
    return config_path


def test_create_app_given_missing_config_file_raises_config_error(tmp_path: Path):
    config_path = tmp_path / "does_not_exist.json"

    with pytest.raises(ConfigError):
        create_app(config_path)


def test_create_app_given_invalid_json_config_raises_config_error(tmp_path: Path):
    config_path = tmp_path / "app_config.json"
    config_path.write_text("not json")

    with pytest.raises(ConfigError):
        create_app(config_path)


def test_create_app_given_agent_serves_run_with_prepended_system_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    mock_acompletion = AsyncMock(return_value=_fake_litellm_response())
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    config_path = _agent_config_path(tmp_path)

    app = create_app(config_path)
    client = TestClient(app)

    response = client.post("/v1/agents/researcher", json={"message": "hi"})

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "openai/gpt-4o"
    assert body["message"]["content"] == "hello!"
    assert body["usage"]["total_tokens"] == 15

    _, kwargs = mock_acompletion.call_args
    assert kwargs["messages"][0] == {"role": "system", "content": "You are a research assistant."}


def test_create_app_given_tracing_disabled_shuts_down_without_raising(tmp_path: Path):
    app = create_app(_agent_config_path(tmp_path))

    with TestClient(app):
        pass


def test_create_app_given_tracing_enabled_uses_sdk_provider_and_shuts_down_without_raising(
    tmp_path: Path,
):
    config_path = tmp_path / "app_config.json"
    config_path.write_text(
        json.dumps(
            {
                "llms": [{"model": "openai/gpt-4o"}],
                "strategies": [{"name": "react"}],
                "agents": [
                    {
                        "name": "researcher",
                        "system_prompt": "You are a research assistant.",
                        "model": "openai/gpt-4o",
                        "strategy": "react",
                    }
                ],
                "tracing": {"console": True},
            }
        )
    )

    app = create_app(config_path)

    with TestClient(app):
        assert isinstance(trace.get_tracer_provider(), SDKTracerProvider)


def test_create_app_given_a_second_tracing_enabled_app_does_not_shut_down_the_first(
    tmp_path: Path,
):
    # opentelemetry.trace.set_tracer_provider() silently no-ops on any call after the first
    # in a process — a second create_app() with tracing enabled must not tear down the
    # first app's still-registered provider when IT shuts down.
    config_path = tmp_path / "app_config.json"
    config_path.write_text(
        json.dumps({"llms": [{"model": "openai/gpt-4o"}], "tracing": {"console": True}})
    )

    first_app = create_app(config_path)
    with TestClient(first_app):
        pass
    registered_provider = trace.get_tracer_provider()

    second_app = create_app(config_path)
    with TestClient(second_app):
        pass

    assert trace.get_tracer_provider() is registered_provider


def test_create_app_given_unknown_agent_returns_404(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("litellm.acompletion", AsyncMock(return_value=_fake_litellm_response()))
    config_path = tmp_path / "app_config.json"
    config_path.write_text(json.dumps({"llms": [{"model": "openai/gpt-4o"}]}))

    app = create_app(config_path)
    client = TestClient(app)

    response = client.post("/v1/agents/missing", json={"message": "hi"})

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "agent_not_found"


def test_create_app_given_unmatched_route_returns_message_shaped_detail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # Starlette's own framework-raised 404 (no matching route) has a plain-string `detail`
    # by default — handle_http_exception normalizes it to the same {"message": ...} shape
    # every app-raised error uses, so callers can always read detail["message"].
    monkeypatch.setattr("litellm.acompletion", AsyncMock(return_value=_fake_litellm_response()))
    config_path = tmp_path / "app_config.json"
    config_path.write_text(json.dumps({"llms": [{"model": "openai/gpt-4o"}]}))
    app = create_app(config_path)
    client = TestClient(app)

    response = client.get("/this-route-does-not-exist")

    assert response.status_code == 404
    assert response.json() == {"detail": {"message": "Not Found"}}


def test_create_app_given_wrong_http_method_returns_message_shaped_detail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("litellm.acompletion", AsyncMock(return_value=_fake_litellm_response()))
    config_path = tmp_path / "app_config.json"
    config_path.write_text(json.dumps({"llms": [{"model": "openai/gpt-4o"}]}))
    app = create_app(config_path)
    client = TestClient(app)

    response = client.delete("/health")

    assert response.status_code == 405
    assert response.json() == {"detail": {"message": "Method Not Allowed"}}


def test_create_app_given_request_tools_reaches_litellm_as_function_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    mock_acompletion = AsyncMock(return_value=_fake_litellm_response())
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    config_path = tmp_path / "app_config.json"
    config_path.write_text(
        json.dumps(
            {
                "llms": [{"model": "openai/gpt-4o"}],
                "tools": [{"name": "get_current_time"}],
                "strategies": [{"name": "react"}],
                "agents": [
                    {
                        "name": "researcher",
                        "system_prompt": "You are a research assistant.",
                        "model": "openai/gpt-4o",
                        "strategy": "react",
                    }
                ],
            }
        )
    )

    app = create_app(config_path)
    client = TestClient(app)

    response = client.post(
        "/v1/agents/researcher", json={"message": "hi", "tools": ["get_current_time"]}
    )

    assert response.status_code == 200
    _, kwargs = mock_acompletion.call_args
    assert kwargs["tools"][0]["function"]["name"] == "get_current_time"


def test_create_app_given_empty_tools_list_suppresses_agent_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    mock_acompletion = AsyncMock(return_value=_fake_litellm_response())
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    config_path = tmp_path / "app_config.json"
    config_path.write_text(
        json.dumps(
            {
                "llms": [{"model": "openai/gpt-4o"}],
                "tools": [{"name": "get_current_time"}],
                "strategies": [{"name": "react"}],
                "agents": [
                    {
                        "name": "researcher",
                        "system_prompt": "You are a research assistant.",
                        "model": "openai/gpt-4o",
                        "strategy": "react",
                        "tools": ["get_current_time"],
                    }
                ],
            }
        )
    )

    app = create_app(config_path)
    client = TestClient(app)

    response = client.post("/v1/agents/researcher", json={"message": "hi", "tools": []})

    assert response.status_code == 200
    _, kwargs = mock_acompletion.call_args
    assert "tools" not in kwargs


def test_create_app_given_sampling_params_forwards_them_to_litellm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    mock_acompletion = AsyncMock(return_value=_fake_litellm_response())
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    config_path = _agent_config_path(tmp_path)

    app = create_app(config_path)
    client = TestClient(app)

    response = client.post(
        "/v1/agents/researcher",
        json={"message": "hi", "temperature": 0.2, "top_p": 0.9, "max_tokens": 512},
    )

    assert response.status_code == 200
    _, kwargs = mock_acompletion.call_args
    assert kwargs["temperature"] == 0.2
    assert kwargs["top_p"] == 0.9
    assert kwargs["max_completion_tokens"] == 512


def test_create_app_given_unknown_strategy_override_returns_404(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("litellm.acompletion", AsyncMock(return_value=_fake_litellm_response()))
    config_path = _agent_config_path(tmp_path)

    app = create_app(config_path)
    client = TestClient(app)

    response = client.post("/v1/agents/researcher", json={"message": "hi", "strategy": "rewoo"})

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "strategy_not_found"


def test_create_app_given_llm_requests_a_tool_call_executes_it_and_returns_final_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    tool_call_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="call_1",
                            function=SimpleNamespace(name="get_current_time", arguments="{}"),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    mock_acompletion = AsyncMock(side_effect=[tool_call_response, _fake_litellm_response()])
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    config_path = tmp_path / "app_config.json"
    config_path.write_text(
        json.dumps(
            {
                "llms": [{"model": "openai/gpt-4o"}],
                "tools": [{"name": "get_current_time"}],
                "strategies": [{"name": "react"}],
                "agents": [
                    {
                        "name": "clock-bot",
                        "system_prompt": "You have a clock tool.",
                        "model": "openai/gpt-4o",
                        "strategy": "react",
                        "tools": ["get_current_time"],
                    }
                ],
            }
        )
    )

    app = create_app(config_path)
    client = TestClient(app)

    response = client.post("/v1/agents/clock-bot", json={"message": "what time is it?"})

    assert response.status_code == 200
    body = response.json()
    assert body["message"]["content"] == "hello!"
    assert body["message"]["tool_calls"] is None
    assert body["finish_reason"] == "stop"
    assert mock_acompletion.call_count == 2
    second_call_messages = mock_acompletion.call_args_list[1].kwargs["messages"]
    assert second_call_messages[-1]["role"] == "tool"
    assert second_call_messages[-1]["tool_call_id"] == "call_1"
    assert second_call_messages[-1]["name"] == "get_current_time"


def test_create_app_given_valid_config_lists_agents_tools_llms_and_strategies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("litellm.acompletion", AsyncMock(return_value=_fake_litellm_response()))
    config_path = tmp_path / "app_config.json"
    config_path.write_text(
        json.dumps(
            {
                "llms": [{"model": "openai/gpt-4o"}],
                "tools": [{"name": "get_current_time"}],
                "strategies": [{"name": "react"}],
                "agents": [
                    {
                        "name": "researcher",
                        "system_prompt": "You are a research assistant.",
                        "model": "openai/gpt-4o",
                        "strategy": "react",
                        "tools": ["get_current_time"],
                    }
                ],
            }
        )
    )

    app = create_app(config_path)
    client = TestClient(app)

    assert client.get("/v1/agents").json() == [
        {
            "name": "researcher",
            "model": "openai/gpt-4o",
            "strategy": "react",
            "tools": ["get_current_time"],
        }
    ]
    assert client.get("/v1/tools").json()[0]["name"] == "get_current_time"
    assert client.get("/v1/models").json() == ["openai/gpt-4o"]
    assert client.get("/v1/strategies").json() == ["react"]


def test_create_app_given_no_session_id_returns_a_new_one_and_second_call_reuses_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    mock_acompletion = AsyncMock(return_value=_fake_litellm_response())
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    config_path = _agent_config_path(tmp_path)

    app = create_app(config_path)
    client = TestClient(app)

    first = client.post("/v1/agents/researcher", json={"message": "my name is Sam"})
    session_id = first.json()["session_id"]
    assert session_id

    client.post(
        "/v1/agents/researcher",
        json={"message": "what's my name?", "session_id": session_id},
    )

    second_call_messages = mock_acompletion.call_args_list[1].kwargs["messages"]
    assert second_call_messages[1] == {"role": "user", "content": "my name is Sam"}
    assert second_call_messages[2] == {"role": "assistant", "content": "hello!"}
    assert second_call_messages[3] == {"role": "user", "content": "what's my name?"}


def test_create_app_given_unknown_session_id_returns_404(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("litellm.acompletion", AsyncMock(return_value=_fake_litellm_response()))
    config_path = _agent_config_path(tmp_path)

    app = create_app(config_path)
    client = TestClient(app)

    response = client.post(
        "/v1/agents/researcher", json={"message": "hi", "session_id": "does-not-exist"}
    )

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "session_not_found"


def test_create_app_given_session_id_reused_under_a_different_agent_returns_404(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("litellm.acompletion", AsyncMock(return_value=_fake_litellm_response()))
    config_path = tmp_path / "app_config.json"
    config_path.write_text(
        json.dumps(
            {
                "llms": [{"model": "openai/gpt-4o"}],
                "strategies": [{"name": "react"}],
                "agents": [
                    {
                        "name": "researcher",
                        "system_prompt": "You are a research assistant.",
                        "model": "openai/gpt-4o",
                        "strategy": "react",
                    },
                    {
                        "name": "scheduler",
                        "system_prompt": "You are a scheduling assistant.",
                        "model": "openai/gpt-4o",
                        "strategy": "react",
                    },
                ],
            }
        )
    )

    app = create_app(config_path)
    client = TestClient(app)

    first = client.post("/v1/agents/researcher", json={"message": "hi"})
    session_id = first.json()["session_id"]

    response = client.post("/v1/agents/scheduler", json={"message": "hi", "session_id": session_id})

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "session_not_found"


def test_create_app_given_message_exceeds_max_input_chars_returns_413(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("litellm.acompletion", AsyncMock(return_value=_fake_litellm_response()))
    config_path = _agent_config_path(tmp_path, max_input_chars=5)

    app = create_app(config_path)
    client = TestClient(app)

    response = client.post("/v1/agents/researcher", json={"message": "this is too long"})

    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "input_too_large"


def test_create_app_given_message_exceeds_max_input_chars_logs_only_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # agent_run.py already logs this at INFO before raising InputTooLargeError —
    # handle_http_exception must not log a second INFO line for the same 400 response.
    monkeypatch.setattr("litellm.acompletion", AsyncMock(return_value=_fake_litellm_response()))
    config_path = _agent_config_path(tmp_path, max_input_chars=5)
    app = create_app(config_path)
    app_handler, app_records = _capture("agent.api.app")
    run_handler, run_records = _capture("agent.core.services.agent_run")
    client = TestClient(app)

    client.post("/v1/agents/researcher", json={"message": "this is too long"})

    logging.getLogger("agent.api.app").removeHandler(app_handler)
    logging.getLogger("agent.core.services.agent_run").removeHandler(run_handler)
    assert len(run_records) == 1
    assert app_records == []


def test_create_app_given_compaction_configured_wires_and_invokes_compaction_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Once usage crosses the tiny configured budget, the next call compacts first."""
    summary_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="summary", tool_calls=None), finish_reason="stop"
            )
        ],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=5, total_tokens=10),
    )
    mock_acompletion = AsyncMock(
        side_effect=[_fake_litellm_response(), summary_response, _fake_litellm_response()]
    )
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    config_path = tmp_path / "app_config.json"
    config_path.write_text(
        json.dumps(
            {
                "llms": [{"model": "openai/gpt-4o"}, {"model": "openai/gpt-4o-mini"}],
                "strategies": [{"name": "react"}],
                "agents": [
                    {
                        "name": "researcher",
                        "system_prompt": "You are a research assistant.",
                        "model": "openai/gpt-4o",
                        "strategy": "react",
                    }
                ],
                "compaction": {
                    "model": "openai/gpt-4o-mini",
                    "token_budget_pct": 0.0001,
                    "keep_recent_turns": 0,
                    "prompt": "Summarize this.",
                },
            }
        )
    )

    app = create_app(config_path)
    client = TestClient(app)

    first = client.post("/v1/agents/researcher", json={"message": "hi"})
    session_id = first.json()["session_id"]

    second = client.post(
        "/v1/agents/researcher",
        json={"message": "what's next?", "session_id": session_id},
    )

    assert second.status_code == 200
    assert mock_acompletion.call_count == 3
    summarizer_call = mock_acompletion.call_args_list[1]
    assert summarizer_call.kwargs["model"] == "openai/gpt-4o-mini"
    assert summarizer_call.kwargs["messages"] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello!"},
        {"role": "user", "content": "Summarize this."},
    ]


def test_create_app_given_context_window_exceeded_with_no_compaction_returns_413(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import litellm

    monkeypatch.setattr(
        "litellm.acompletion",
        AsyncMock(
            side_effect=litellm.ContextWindowExceededError(
                message="too big", model="openai/gpt-4o", llm_provider="openai"
            )
        ),
    )
    config_path = _agent_config_path(tmp_path)

    app = create_app(config_path)
    client = TestClient(app)

    response = client.post("/v1/agents/researcher", json={"message": "hi"})

    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "context_window_exceeded"


def test_create_app_given_compaction_cannot_help_returns_413_compaction_exhausted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # Second call: the agent call overflows, and every compaction tier fails too (the
    # summarizer overflows as well, and a one-turn history can't be chunked) — the
    # exhausted case, which must map to its own code, not the generic overflow one.
    import litellm

    overflow = litellm.ContextWindowExceededError(
        message="too big", model="openai/gpt-4o", llm_provider="openai"
    )
    mock_acompletion = AsyncMock(side_effect=[_fake_litellm_response(), *[overflow] * 20])
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    config_path = tmp_path / "app_config.json"
    config_path.write_text(
        json.dumps(
            {
                "llms": [{"model": "openai/gpt-4o"}, {"model": "openai/gpt-4o-mini"}],
                "strategies": [{"name": "react"}],
                "agents": [
                    {
                        "name": "researcher",
                        "system_prompt": "You are a research assistant.",
                        "model": "openai/gpt-4o",
                        "strategy": "react",
                    }
                ],
                "compaction": {
                    "model": "openai/gpt-4o-mini",
                    "keep_recent_turns": 1,
                    "prompt": "Summarize this.",
                },
            }
        )
    )

    app = create_app(config_path)
    client = TestClient(app)
    session_id = client.post("/v1/agents/researcher", json={"message": "hi"}).json()["session_id"]

    response = client.post(
        "/v1/agents/researcher",
        json={"message": "what's next?", "session_id": session_id},
    )

    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "compaction_exhausted"


def _capture(logger_name: str) -> tuple[logging.Handler, list[logging.LogRecord]]:
    """A handler attached directly to a named logger.

    Safe against configure_logging()'s own logging.basicConfig(force=True), which only
    replaces the *root* logger's handler list, never a named child logger's own
    directly-attached ones.
    """
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Capture()
    target = logging.getLogger(logger_name)
    target.addHandler(handler)
    target.setLevel(logging.DEBUG)
    return handler, records


def test_create_app_logs_startup_info_with_wiring_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    handler, records = _capture("agent.api.app")
    monkeypatch.setattr("litellm.acompletion", AsyncMock(return_value=_fake_litellm_response()))
    config_path = _agent_config_path(tmp_path)

    create_app(config_path)

    logging.getLogger("agent.api.app").removeHandler(handler)
    [record] = [r for r in records if r.levelno == logging.INFO]
    assert "agent-core started" in record.message
    assert "1" in record.message  # one agent configured by _agent_config_path


def test_create_app_health_endpoint_returns_ok(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("litellm.acompletion", AsyncMock(return_value=_fake_litellm_response()))
    config_path = _agent_config_path(tmp_path)
    app = create_app(config_path)
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_create_app_health_endpoint_logs_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("litellm.acompletion", AsyncMock(return_value=_fake_litellm_response()))
    config_path = _agent_config_path(tmp_path)
    app = create_app(config_path)  # startup log already fired; irrelevant to this test
    handler, records = _capture("agent.api.app")
    client = TestClient(app)

    client.get("/health")

    logging.getLogger("agent.api.app").removeHandler(handler)
    assert records == []


def test_create_app_registry_listing_logs_debug(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("litellm.acompletion", AsyncMock(return_value=_fake_litellm_response()))
    config_path = _agent_config_path(tmp_path)
    app = create_app(config_path)
    handler, records = _capture("agent.api.app")
    client = TestClient(app)

    client.get("/v1/agents")

    logging.getLogger("agent.api.app").removeHandler(handler)
    [record] = records
    assert record.levelno == logging.DEBUG
    assert "agents" in record.message


def test_create_app_response_carries_x_request_id_header(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("litellm.acompletion", AsyncMock(return_value=_fake_litellm_response()))
    config_path = _agent_config_path(tmp_path)
    app = create_app(config_path)
    client = TestClient(app)

    response = client.post("/v1/agents/researcher", json={"message": "hi"})

    assert response.headers["X-Request-ID"]


def test_create_app_given_json_logging_format_uses_json_formatter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("litellm.acompletion", AsyncMock(return_value=_fake_litellm_response()))
    config_path = tmp_path / "app_config.json"
    config_path.write_text(
        json.dumps(
            {
                "llms": [{"model": "openai/gpt-4o"}],
                "strategies": [{"name": "react"}],
                "agents": [
                    {
                        "name": "researcher",
                        "system_prompt": "You are a research assistant.",
                        "model": "openai/gpt-4o",
                        "strategy": "react",
                    }
                ],
                "logging": {"format": "json"},
            }
        )
    )

    create_app(config_path)

    from agent.api.logging_setup import JsonFormatter

    assert isinstance(logging.getLogger().handlers[0].formatter, JsonFormatter)


def test_create_app_given_max_sessions_wires_it_into_the_session_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("litellm.acompletion", AsyncMock(return_value=_fake_litellm_response()))
    config_path = tmp_path / "app_config.json"
    config_path.write_text(
        json.dumps(
            {
                "llms": [{"model": "openai/gpt-4o"}],
                "strategies": [{"name": "react"}],
                "agents": [
                    {
                        "name": "researcher",
                        "system_prompt": "You are a research assistant.",
                        "model": "openai/gpt-4o",
                        "strategy": "react",
                    }
                ],
                "max_sessions": 1,
            }
        )
    )
    app = create_app(config_path)
    client = TestClient(app)

    first = client.post("/v1/agents/researcher", json={"message": "hi"})
    second = client.post("/v1/agents/researcher", json={"message": "hi again"})

    get_first = client.get(f"/v1/agents/researcher/sessions/{first.json()['session_id']}")
    get_second = client.get(f"/v1/agents/researcher/sessions/{second.json()['session_id']}")
    assert get_first.status_code == 404  # evicted — max_sessions=1
    assert get_second.status_code == 200


def test_get_session_usage_returns_cumulative_and_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("litellm.acompletion", AsyncMock(return_value=_fake_litellm_response()))
    monkeypatch.setattr("litellm.completion_cost", lambda **kwargs: 0.001)
    app = create_app(_agent_config_path(tmp_path))
    client = TestClient(app)
    run_response = client.post("/v1/agents/researcher", json={"message": "hello"})
    session_id = run_response.json()["session_id"]

    response = client.get(f"/v1/agents/researcher/sessions/{session_id}/usage")

    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == session_id
    assert body["cumulative"]["total_tokens"] == 15
    assert float(body["cumulative"]["cost_usd"]) == pytest.approx(0.001)
    assert body["context_tokens"] == 15


def test_get_session_usage_given_unknown_session_returns_404(tmp_path: Path):
    app = create_app(_agent_config_path(tmp_path))
    client = TestClient(app)

    response = client.get("/v1/agents/researcher/sessions/does-not-exist/usage")

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "session_not_found"


def test_get_agent_usage_returns_cumulative(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("litellm.acompletion", AsyncMock(return_value=_fake_litellm_response()))
    monkeypatch.setattr("litellm.completion_cost", lambda **kwargs: 0.001)
    app = create_app(_agent_config_path(tmp_path))
    client = TestClient(app)
    client.post("/v1/agents/researcher", json={"message": "hello"})

    response = client.get("/v1/agents/researcher/usage")

    assert response.status_code == 200
    body = response.json()
    assert body["agent"] == "researcher"
    assert body["cumulative"]["total_tokens"] == 15
    assert float(body["cumulative"]["cost_usd"]) == pytest.approx(0.001)


def test_get_agent_usage_given_unregistered_agent_returns_404(tmp_path: Path):
    app = create_app(_agent_config_path(tmp_path))
    client = TestClient(app)

    response = client.get("/v1/agents/nonexistent/usage")

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "agent_not_found"


def test_get_agent_usage_given_never_run_returns_zero_usage(tmp_path: Path):
    app = create_app(_agent_config_path(tmp_path))
    client = TestClient(app)

    response = client.get("/v1/agents/researcher/usage")

    assert response.status_code == 200
    assert response.json()["cumulative"]["total_tokens"] == 0
