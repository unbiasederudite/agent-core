"""Tests for building and registering the process-wide TracerProvider."""

from importlib.metadata import PackageNotFoundError, version
from unittest.mock import patch

import pytest

from agent.core.factories.app import configure_tracing
from agent.core.models.config import TracingConfig
from agent.core.tracing import capture_content_enabled, set_capture_content


@pytest.fixture(autouse=True)
def _reset_capture_content():
    set_capture_content(False)
    yield
    set_capture_content(False)


def test_configure_tracing_given_no_destination_does_not_set_a_tracer_provider():
    with patch("agent.core.factories.app.trace.set_tracer_provider") as mock_set:
        configure_tracing(TracingConfig())

    mock_set.assert_not_called()


def test_configure_tracing_given_no_destination_returns_none():
    result = configure_tracing(TracingConfig())

    assert result is None


def test_configure_tracing_given_console_returns_the_registered_provider():
    with patch("agent.core.factories.app.trace.set_tracer_provider") as mock_set:
        result = configure_tracing(TracingConfig(console=True))

    assert result is not None
    mock_set.assert_called_once_with(result)


def test_configure_tracing_given_no_destination_leaves_capture_content_false():
    configure_tracing(TracingConfig(capture_content=True))

    assert capture_content_enabled() is False


def test_configure_tracing_given_no_destination_resets_a_prior_calls_capture_content():
    # capture_content is a process-global, not tied to any one TracerProvider — a later,
    # differently-configured call in the same process must not inherit an earlier one's flag.
    with patch("agent.core.factories.app.trace.set_tracer_provider"):
        configure_tracing(TracingConfig(console=True, capture_content=True))
    assert capture_content_enabled() is True

    configure_tracing(TracingConfig())

    assert capture_content_enabled() is False


def test_configure_tracing_given_endpoint_does_not_raise():
    with patch("agent.core.factories.app.trace.set_tracer_provider") as mock_set:
        configure_tracing(TracingConfig(endpoint="http://localhost:4318"))

    mock_set.assert_called_once()


def test_configure_tracing_given_console_and_endpoint_does_not_raise():
    with patch("agent.core.factories.app.trace.set_tracer_provider") as mock_set:
        configure_tracing(TracingConfig(console=True, endpoint="http://localhost:4318"))

    mock_set.assert_called_once()


def test_configure_tracing_given_console_sets_capture_content_flag():
    with patch("agent.core.factories.app.trace.set_tracer_provider"):
        configure_tracing(TracingConfig(console=True, capture_content=True))

    assert capture_content_enabled() is True


def test_configure_tracing_given_no_otel_service_name_env_defaults_to_agent_core(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("OTEL_SERVICE_NAME", raising=False)
    with patch("agent.core.factories.app.trace.set_tracer_provider") as mock_set:
        configure_tracing(TracingConfig(console=True))

    provider = mock_set.call_args[0][0]
    assert provider.resource.attributes["service.name"] == "agent-core"


def test_configure_tracing_given_otel_service_name_env_set_does_not_override_it(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("OTEL_SERVICE_NAME", "operator-chosen-name")
    with patch("agent.core.factories.app.trace.set_tracer_provider") as mock_set:
        configure_tracing(TracingConfig(console=True))

    provider = mock_set.call_args[0][0]
    assert provider.resource.attributes["service.name"] == "operator-chosen-name"


def test_configure_tracing_given_no_otel_service_name_env_sets_service_version(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("OTEL_SERVICE_NAME", raising=False)
    with patch("agent.core.factories.app.trace.set_tracer_provider") as mock_set:
        configure_tracing(TracingConfig(console=True))

    provider = mock_set.call_args[0][0]
    # Read from the installed package's own metadata, not hand-typed — must match exactly.
    assert provider.resource.attributes["service.version"] == version("agent-core")


def test_configure_tracing_given_package_metadata_missing_omits_service_version(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("OTEL_SERVICE_NAME", raising=False)
    with (
        patch("agent.core.factories.app.trace.set_tracer_provider") as mock_set,
        patch("agent.core.factories.app._package_version", side_effect=PackageNotFoundError),
    ):
        configure_tracing(TracingConfig(console=True))

    provider = mock_set.call_args[0][0]
    assert "service.version" not in provider.resource.attributes
    assert provider.resource.attributes["service.name"] == "agent-core"
