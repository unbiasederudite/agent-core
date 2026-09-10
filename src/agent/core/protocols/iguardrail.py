"""Protocol interface for guardrail implementations, and the pipeline that runs them."""

import logging
import time
from typing import Literal, Protocol

from opentelemetry import trace

from agent.core.exceptions import GuardrailBlockedError
from agent.core.models.guardrail import GuardrailFinding
from agent.core.tracing import record_gated_exception

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class IGuardrail(Protocol):
    """Interface for a check that inspects content and decides whether to flag it."""

    name: str  # The guardrail's lookup key, matching its config entry's `name`.
    action: Literal["block", "redact", "warn"]  # What happens when this guardrail triggers.

    async def check(self, content: str) -> GuardrailFinding:
        """Check `content` and report whether it should be flagged.

        Args:
            content: Text to check.

        Returns:
            GuardrailFinding: the check's result.
        """
        ...


async def _check_one(guardrail: IGuardrail, content: str) -> tuple[GuardrailFinding, float]:
    """Run one guardrail's `check()` and time it.

    Args:
        guardrail: The guardrail to run.
        content: Text to check.

    Returns:
        The check's finding, and how long the check took in milliseconds.
    """
    with tracer.start_as_current_span(
        f"guardrail {guardrail.name}", record_exception=False, set_status_on_exception=False
    ) as span:
        start = time.monotonic()
        try:
            finding = await guardrail.check(content)
        except Exception as exc:
            logger.error(
                "guardrail '%s' check raised",
                guardrail.name,
                exc_info=True,
                extra={"exception_type": type(exc).__name__},
            )
            record_gated_exception(span, exc)
            raise
        duration_ms = (time.monotonic() - start) * 1000
        span.set_attribute("agent_core.guardrail.triggered", finding.triggered)
        span.set_attribute("agent_core.guardrail.action", guardrail.action)
        return finding, duration_ms


async def run_guardrails(
    content: str,
    guardrails: list[IGuardrail],
    checkpoint: Literal["input_guardrails", "output_guardrails", "tool_output_guardrails"],
) -> str:
    """Run `guardrails` against `content` in order, applying each one's configured action.

    Args:
        content: Text to check.
        guardrails: Guardrails to run, in order.
        checkpoint: Name of this checkpoint, used as the parent span's name.

    Returns:
        str: `content`, possibly redacted by one or more guardrails.

    Raises:
        GuardrailBlockedError: a block-action guardrail triggered.
    """
    with tracer.start_as_current_span(
        checkpoint, record_exception=False, set_status_on_exception=False
    ) as checkpoint_span:
        try:
            for guardrail in guardrails:
                finding, duration_ms = await _check_one(guardrail, content)
                if not finding.triggered:
                    logger.debug(
                        "guardrail '%s' checked in %.1fms, no findings",
                        guardrail.name,
                        duration_ms,
                        extra={"duration_ms": duration_ms},
                    )
                    continue
                if guardrail.action == "block":
                    logger.info(
                        "guardrail '%s' blocked this content after %.1fms: %s",
                        guardrail.name,
                        duration_ms,
                        finding.reason,
                        extra={"duration_ms": duration_ms},
                    )
                    raise GuardrailBlockedError(
                        f"guardrail '{guardrail.name}' blocked this content"
                    )
                if guardrail.action == "redact" and finding.redacted_content is not None:
                    logger.info(
                        "guardrail '%s' redacted content after %.1fms: %s",
                        guardrail.name,
                        duration_ms,
                        finding.reason,
                        extra={"duration_ms": duration_ms},
                    )
                    content = finding.redacted_content
                else:
                    logger.warning(
                        "guardrail '%s' triggered (%s) after %.1fms: %s",
                        guardrail.name,
                        guardrail.action,
                        duration_ms,
                        finding.reason,
                        extra={"duration_ms": duration_ms},
                    )
        except GuardrailBlockedError:
            raise
        except Exception as exc:
            record_gated_exception(checkpoint_span, exc)
            raise
        return content
