"""Guardrail adapter backed by one resolved Guardrails AI Hub validator."""

import asyncio
import importlib
import inspect
import logging
import os
from typing import Any, Literal, cast

from guardrails import Guard, OnFailAction
from guardrails.classes.validation.validation_summary import ValidationSummary
from guardrails.settings import settings as guardrails_settings
from guardrails.validator_base import get_validator_class

from agent.adapters.llm_registry_provider import guardrail_call_scope
from agent.core.exceptions import GuardrailBlockedError
from agent.core.models.guardrail import GuardrailFinding

guardrails_settings.disable_tracing = True

# Guardrails AI's default validator dispatch hops to a thread that doesn't inherit
# contextvars, silently dropping any per-call state threaded through from the caller. Forcing
# synchronous dispatch keeps a validator's own LLM call on the calling thread instead, at the
# cost of validators no longer running concurrently within one check.
os.environ["GUARDRAILS_RUN_SYNC"] = "true"

logger = logging.getLogger(__name__)

_MAX_REASONS_SHOWN = 5
_MAX_REASON_CHARS = 200


def _truncate_reason(text: str, max_chars: int) -> str:
    """Cap `text` at `max_chars`, appending a marker noting what was cut.

    Args:
        text: Text to cap.
        max_chars: Cap in characters.

    Returns:
        str: `text`, truncated if it exceeded `max_chars`.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"...[truncated, {len(text) - max_chars} more characters]"


def _format_validation_summaries(summaries: list[ValidationSummary]) -> str:
    """Turn validation summaries into a short, bounded reason string.

    Args:
        summaries: The triggered validators' summaries.

    Returns:
        str: a short, readable, length-bounded description of what triggered.
    """
    parts = []
    for summary in summaries[:_MAX_REASONS_SHOWN]:
        reason = summary.failure_reason or f"{summary.validator_name} flagged this content"
        parts.append(_truncate_reason(reason, _MAX_REASON_CHARS))
    message = "; ".join(parts)
    omitted = len(summaries) - _MAX_REASONS_SHOWN
    if omitted > 0:
        message += f"; ...and {omitted} more reason(s)"
    return message


def resolve_validator(validator_id: str) -> type:
    """Resolve a Guardrails AI Hub id to its validator class.

    Args:
        validator_id: Hub id, e.g. "guardrails/detect_pii".

    Returns:
        type: the resolved validator class.

    Raises:
        ValueError: `validator_id` isn't a valid Hub id, or no installed package resolves it.
    """
    _, separator, short_name = validator_id.partition("/")
    if not separator or not short_name:
        raise ValueError(f"'{validator_id}' is not a valid Hub id (expected 'namespace/name')")
    try:
        importlib.import_module(f"guardrails_ai.{short_name}")
    except ImportError:
        # Not published under the newer PyPI convention — get_validator_class() below falls
        # back to the deprecated Hub-registry mechanism for validators only available that way.
        pass
    validator_cls = get_validator_class(validator_id)
    if validator_cls is None:
        package_name = f"guardrails-ai-{short_name.replace('_', '-')}"
        raise ValueError(
            f"validator '{validator_id}' is not installed — try `pip install {package_name}` "
            f"if it's been migrated to PyPI, or `guardrails hub install {validator_id}` "
            f"if it hasn't"
        )
    return cast(type, validator_cls)


def declares_llm_callable(validator_cls: type) -> bool:
    """Check whether a validator class's constructor declares an `llm_callable` parameter.

    Args:
        validator_cls: Validator class to inspect.

    Returns:
        bool: whether the class's constructor accepts `llm_callable`.
    """
    return "llm_callable" in inspect.signature(validator_cls.__init__).parameters  # type: ignore[misc]


class GuardrailsAIAdapter:
    """Checks content using one resolved Guardrails AI Hub validator."""

    def __init__(
        self,
        name: str,
        validator_id: str,
        validator_params: dict[str, Any],
        action: Literal["block", "redact", "warn"],
        *,
        validator_cls: type | None = None,
    ) -> None:
        """Resolve `validator_id`, unless already resolved, and build the underlying `Guard`.

        Args:
            name: This guardrail's lookup key.
            validator_id: Guardrails AI Hub id to resolve.
            validator_params: Keyword arguments for the resolved validator's constructor.
            action: What happens when this guardrail triggers.
            validator_cls: The already-resolved validator class, if a caller resolved
                `validator_id` itself. Resolved here otherwise.

        Raises:
            ValueError: `validator_id` doesn't resolve to an installed validator.
        """
        resolved_cls = (
            validator_cls if validator_cls is not None else resolve_validator(validator_id)
        )
        on_fail = OnFailAction.FIX if action == "redact" else OnFailAction.NOOP
        guard = Guard()
        guard.configure(allow_metrics_collection=False)
        self._guard = guard.use(resolved_cls(**validator_params, on_fail=on_fail))
        self.name = name
        self.action: Literal["block", "redact", "warn"] = action

    async def check(self, content: str) -> GuardrailFinding:
        """Run the resolved validator against `content`.

        Args:
            content: Text to check.

        Returns:
            GuardrailFinding: the check's result.

        Raises:
            GuardrailBlockedError: the underlying validator raised.
        """
        try:
            with guardrail_call_scope() as call_kwargs:
                result = await asyncio.to_thread(self._guard.parse, content, **call_kwargs)
        except Exception as exc:  # noqa: BLE001 — a validator's internals can raise anything
            logger.warning(
                "guardrail '%s' check failed: %s",
                self.name,
                exc,
                exc_info=True,
                extra={"exception_type": type(exc).__name__},
            )
            # Regardless of this guardrail's own action (block, redact, warn): a failed
            # check can't tell whether its content was safe, so it's treated as if it had
            # blocked rather than let unverified content through unfiltered.
            raise GuardrailBlockedError(f"guardrail '{self.name}' check failed") from exc
        summaries = result.validation_summaries or []
        if not summaries:
            return GuardrailFinding(triggered=False)
        redacted_content = None
        if (
            self.action == "redact"
            and isinstance(result.validated_output, str)
            and result.validated_output != content
        ):
            redacted_content = result.validated_output
        return GuardrailFinding(
            triggered=True,
            reason=_format_validation_summaries(summaries),
            redacted_content=redacted_content,
        )
