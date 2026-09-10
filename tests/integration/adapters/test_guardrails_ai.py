from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import litellm
import pytest
from guardrails.validator_base import FailResult, PassResult, Validator, register_validator

from agent.adapters.guardrails_ai import GuardrailsAIAdapter, resolve_validator
from agent.adapters.llm_registry_provider import (
    LLM_REGISTRY_PROVIDER,
    register_llm_registry_provider,
)
from agent.core.exceptions import GuardrailBlockedError
from agent.core.models.completion import Completion
from agent.core.models.message import Message
from agent.core.models.usage import Usage
from agent.core.registries.llm import LLMRegistry
from agent.core.run_context import collect_extra_usage, run_context


def _mock_validator_cls() -> MagicMock:
    validator_cls = MagicMock()
    validator_cls.return_value = MagicMock()
    return validator_cls


def _mock_summary(failure_reason: str = "pii detected") -> MagicMock:
    return MagicMock(validator_name="MockValidator", failure_reason=failure_reason)


@patch("agent.adapters.guardrails_ai.resolve_validator")
@patch("agent.adapters.guardrails_ai.Guard")
async def test_check_given_no_validation_summaries_returns_not_triggered(
    mock_guard_cls: MagicMock, mock_resolve: MagicMock
) -> None:
    mock_resolve.return_value = _mock_validator_cls()
    mock_guard = MagicMock()
    mock_guard.parse.return_value = MagicMock(validation_summaries=[])
    mock_guard_cls.return_value.use.return_value = mock_guard

    adapter = GuardrailsAIAdapter("no-pii", "guardrails/detect_pii", {}, "block")
    finding = await adapter.check("hello")

    assert finding.triggered is False
    assert finding.reason is None


@patch("agent.adapters.guardrails_ai.resolve_validator")
@patch("agent.adapters.guardrails_ai.Guard")
async def test_check_given_block_action_and_summaries_present_returns_triggered_no_redaction(
    mock_guard_cls: MagicMock, mock_resolve: MagicMock
) -> None:
    mock_resolve.return_value = _mock_validator_cls()
    mock_guard = MagicMock()
    mock_guard.parse.return_value = MagicMock(
        validation_summaries=[_mock_summary("pii detected")],
        validated_output="my ssn is 123-45-6789",
    )
    mock_guard_cls.return_value.use.return_value = mock_guard

    adapter = GuardrailsAIAdapter("no-pii", "guardrails/detect_pii", {}, "block")
    finding = await adapter.check("my ssn is 123-45-6789")

    assert finding.triggered is True
    assert finding.redacted_content is None
    assert finding.reason == "pii detected"


@patch("agent.adapters.guardrails_ai.resolve_validator")
@patch("agent.adapters.guardrails_ai.Guard")
async def test_check_given_block_action_and_parse_raises_raises_guardrail_blocked(
    mock_guard_cls: MagicMock, mock_resolve: MagicMock
) -> None:
    mock_resolve.return_value = _mock_validator_cls()
    mock_guard = MagicMock()
    mock_guard.parse.side_effect = RuntimeError("Error getting response from the LLM: boom")
    mock_guard_cls.return_value.use.return_value = mock_guard

    adapter = GuardrailsAIAdapter("judge", "guardrails/response_evaluator", {}, "block")

    with pytest.raises(GuardrailBlockedError):
        await adapter.check("hello")


@patch("agent.adapters.guardrails_ai.resolve_validator")
@patch("agent.adapters.guardrails_ai.Guard")
async def test_check_given_warn_action_and_parse_raises_raises_guardrail_blocked(
    mock_guard_cls: MagicMock, mock_resolve: MagicMock
) -> None:
    mock_resolve.return_value = _mock_validator_cls()
    mock_guard = MagicMock()
    mock_guard.parse.side_effect = RuntimeError("Error getting response from the LLM: boom")
    mock_guard_cls.return_value.use.return_value = mock_guard

    adapter = GuardrailsAIAdapter("judge", "guardrails/response_evaluator", {}, "warn")

    with pytest.raises(GuardrailBlockedError):
        await adapter.check("hello")


@patch("agent.adapters.guardrails_ai.resolve_validator")
@patch("agent.adapters.guardrails_ai.Guard")
async def test_check_given_redact_action_and_parse_raises_raises_guardrail_blocked(
    mock_guard_cls: MagicMock, mock_resolve: MagicMock
) -> None:
    mock_resolve.return_value = _mock_validator_cls()
    mock_guard = MagicMock()
    mock_guard.parse.side_effect = RuntimeError("Error getting response from the LLM: boom")
    mock_guard_cls.return_value.use.return_value = mock_guard

    adapter = GuardrailsAIAdapter("judge", "guardrails/response_evaluator", {}, "redact")

    with pytest.raises(GuardrailBlockedError):
        await adapter.check("hello")


@patch("agent.adapters.guardrails_ai.resolve_validator")
@patch("agent.adapters.guardrails_ai.Guard")
async def test_check_given_redact_action_and_summaries_present_returns_fixed_content(
    mock_guard_cls: MagicMock, mock_resolve: MagicMock
) -> None:
    mock_resolve.return_value = _mock_validator_cls()
    mock_guard = MagicMock()
    mock_guard.parse.return_value = MagicMock(
        validation_summaries=[_mock_summary("pii detected")],
        validated_output="my ssn is [REDACTED]",
    )
    mock_guard_cls.return_value.use.return_value = mock_guard

    adapter = GuardrailsAIAdapter("no-pii", "guardrails/detect_pii", {}, "redact")
    finding = await adapter.check("my ssn is 123-45-6789")

    assert finding.triggered is True
    assert finding.redacted_content == "my ssn is [REDACTED]"
    assert finding.reason == "pii detected"


@patch("agent.adapters.guardrails_ai.resolve_validator")
@patch("agent.adapters.guardrails_ai.Guard")
async def test_check_given_redact_action_and_unfixed_output_returns_no_redaction(
    mock_guard_cls: MagicMock, mock_resolve: MagicMock
) -> None:
    """A NOOP-style outcome under `redact`: flagged, but the output was never actually changed."""
    mock_resolve.return_value = _mock_validator_cls()
    mock_guard = MagicMock()
    mock_guard.parse.return_value = MagicMock(
        validation_summaries=[_mock_summary("pii detected")],
        validated_output="my ssn is 123-45-6789",
    )
    mock_guard_cls.return_value.use.return_value = mock_guard

    adapter = GuardrailsAIAdapter("no-pii", "guardrails/detect_pii", {}, "redact")
    finding = await adapter.check("my ssn is 123-45-6789")

    assert finding.triggered is True
    assert finding.redacted_content is None


@patch("agent.adapters.guardrails_ai.resolve_validator")
def test_init_given_unresolvable_validator_id_raises_value_error(mock_resolve: MagicMock) -> None:
    mock_resolve.side_effect = ValueError("validator package not installed: guardrails/bogus")

    with pytest.raises(ValueError, match="not installed"):
        GuardrailsAIAdapter("bad", "guardrails/bogus", {}, "block")


@patch("agent.adapters.guardrails_ai.resolve_validator")
@patch("agent.adapters.guardrails_ai.Guard")
def test_init_sets_name_and_action(mock_guard_cls: MagicMock, mock_resolve: MagicMock) -> None:
    mock_resolve.return_value = _mock_validator_cls()

    adapter = GuardrailsAIAdapter("no-pii", "guardrails/detect_pii", {}, "warn")

    assert adapter.name == "no-pii"
    assert adapter.action == "warn"


@patch("agent.adapters.guardrails_ai.resolve_validator")
@patch("agent.adapters.guardrails_ai.Guard")
def test_init_disables_hub_metrics_collection_on_the_guard(
    mock_guard_cls: MagicMock, mock_resolve: MagicMock
) -> None:
    mock_resolve.return_value = _mock_validator_cls()
    mock_guard = mock_guard_cls.return_value

    GuardrailsAIAdapter("no-pii", "guardrails/detect_pii", {}, "block")

    mock_guard.configure.assert_called_once_with(allow_metrics_collection=False)


def test_resolve_validator_given_invalid_hub_id_raises_value_error() -> None:
    with pytest.raises(ValueError, match="not a valid Hub id"):
        resolve_validator("no-slash-here")


def test_resolve_validator_given_uninstalled_package_raises_value_error() -> None:
    with patch("agent.adapters.guardrails_ai.importlib.import_module", side_effect=ImportError):
        with pytest.raises(ValueError, match="not installed"):
            resolve_validator("guardrails/bogus")


def test_resolve_validator_given_no_resolution_path_raises_value_error() -> None:
    # Neither the PyPI-package import nor guardrails-ai's own registry (including its
    # internal Hub-registry fallback) resolves the id.
    with (
        patch("agent.adapters.guardrails_ai.importlib.import_module"),
        patch("agent.adapters.guardrails_ai.get_validator_class", return_value=None),
    ):
        with pytest.raises(ValueError, match="not installed"):
            resolve_validator("guardrails/bogus")


# --- Real `Guard` + a locally-registered fake validator, no mocking, no network/Hub install ---


@register_validator(name="agent-core-tests/replaces-secret", data_type="string")
class _ReplacesSecretValidator(Validator):
    """Fails whenever `value` contains the literal substring "secret", fixing it when asked."""

    def validate(self, value: object, metadata: dict[str, object]) -> PassResult | FailResult:
        text = str(value)
        if "secret" in text:
            return FailResult(
                error_message="found the word 'secret'",
                fix_value=text.replace("secret", "[REDACTED]"),
            )
        return PassResult()


def test_resolve_validator_given_real_registered_validator_returns_it() -> None:
    # Exercises the real get_validator_class lookup against guardrails-ai's own registry;
    # only the package import step is mocked, since there's no real installable package here.
    with patch("agent.adapters.guardrails_ai.importlib.import_module"):
        resolved = resolve_validator("agent-core-tests/replaces-secret")

    assert resolved is _ReplacesSecretValidator


async def test_check_given_real_guard_fixing_validator_under_redact_returns_fixed_content() -> None:
    with patch(
        "agent.adapters.guardrails_ai.resolve_validator",
        return_value=_ReplacesSecretValidator,
    ):
        adapter = GuardrailsAIAdapter(
            "no-secrets", "agent-core-tests/replaces-secret", {}, "redact"
        )

        finding = await adapter.check("my secret here")

    assert finding.triggered is True
    assert finding.redacted_content == "my [REDACTED] here"


async def test_check_given_real_guard_with_noop_validator_under_block_flags_without_redaction() -> (
    None
):
    with patch(
        "agent.adapters.guardrails_ai.resolve_validator",
        return_value=_ReplacesSecretValidator,
    ):
        adapter = GuardrailsAIAdapter("no-secrets", "agent-core-tests/replaces-secret", {}, "block")

        finding = await adapter.check("my secret here")

    assert finding.triggered is True
    assert finding.redacted_content is None
    assert finding.reason is not None
    assert finding.reason != "None"


async def test_check_given_real_guard_with_clean_content_returns_not_triggered() -> None:
    with patch(
        "agent.adapters.guardrails_ai.resolve_validator",
        return_value=_ReplacesSecretValidator,
    ):
        adapter = GuardrailsAIAdapter("no-secrets", "agent-core-tests/replaces-secret", {}, "block")

        finding = await adapter.check("nothing sensitive here")

    assert finding.triggered is False


@register_validator(name="agent-core-tests/calls-llm-callable", data_type="string")
class _CallsLLMCallableValidator(Validator):
    """Calls its `llm_callable` synchronously via litellm, mirroring a real Hub validator."""

    def __init__(
        self, llm_callable: str = "gpt-4o-mini", on_fail: object = None, **kwargs: Any
    ) -> None:
        super().__init__(on_fail, llm_callable=llm_callable, **kwargs)
        self.llm_callable = llm_callable

    def validate(self, value: object, metadata: dict[str, object]) -> PassResult | FailResult:
        litellm.completion(
            model=self.llm_callable, messages=[{"role": "user", "content": str(value)}]
        )
        return PassResult()


async def test_check_given_llm_callable_validator_records_its_usage_on_the_active_run() -> None:
    # Proves the fix for Guard.parse()'s own contextvars isolation: without
    # guardrail_call_scope(), this call's usage would silently vanish instead of landing here.
    litellm.custom_provider_map = None
    llm = AsyncMock()
    llm.complete.return_value = Completion(
        message=Message(role="assistant", content="ok"),
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        finish_reason="stop",
    )
    registry = LLMRegistry()
    registry.register("openai/gpt-4o-mini", llm)
    register_llm_registry_provider(registry)

    with patch(
        "agent.adapters.guardrails_ai.resolve_validator",
        return_value=_CallsLLMCallableValidator,
    ):
        adapter = GuardrailsAIAdapter(
            "unusual-prompt-check",
            "agent-core-tests/calls-llm-callable",
            {"llm_callable": f"{LLM_REGISTRY_PROVIDER}/openai/gpt-4o-mini"},
            "block",
        )

        with run_context("researcher", "sess-1"):
            finding = await adapter.check("hello")

            assert collect_extra_usage().total_tokens == 15

    assert finding.triggered is False
