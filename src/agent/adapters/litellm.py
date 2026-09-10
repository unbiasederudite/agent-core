"""LLM adapter backed by litellm, giving access to any provider litellm supports."""

import asyncio
import json
import logging
import random
from typing import Any, cast

import litellm
from opentelemetry import trace

from agent.core.exceptions import (
    LLMContextWindowExceededError,
    LLMError,
    LLMOverloadedError,
    LLMRateLimitedError,
    LLMTimeoutError,
)
from agent.core.models.completion import Completion
from agent.core.models.config import LLMConfig
from agent.core.models.message import (
    Message,
    ToolCall,
    ToolCallFunction,
    flatten_tool_exchanges_for_no_tools_request,
)
from agent.core.models.usage import Usage
from agent.core.tracing import (
    capture_content_enabled,
    record_gated_exception,
    stamp_run_context,
    truncate,
    truncate_keeping_recent,
)

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

_MAX_MESSAGE_ATTRIBUTE_CHARS = 32_000

litellm.telemetry = False
litellm.verbose_logger.handlers.clear()  # type: ignore[attr-defined]
litellm.verbose_logger.propagate = True  # type: ignore[attr-defined]


def _first_not_none[T](a: T | None, b: T | None) -> T | None:
    """Return `a`, or `b` if `a` is `None`.

    Args:
        a: Preferred value.
        b: Fallback value.

    Returns:
        T | None: `a` if not `None`, else `b`.
    """
    return a if a is not None else b


def _is_retriable(status_code: int | None) -> bool:
    """Return whether a failure (by HTTP status) is worth retrying: no response, or a 5xx.

    Args:
        status_code: The failure's HTTP status code, if any.

    Returns:
        bool: whether the failure is worth retrying.
    """
    return status_code is None or status_code >= 500


def _classify(exc: Exception, status_code: int | None) -> LLMError:
    """Map a caught provider exception to its typed equivalent, by status code.

    Args:
        exc: The caught exception.
        status_code: The failure's HTTP status code, if any.

    Returns:
        LLMError: the typed equivalent.
    """
    if status_code == 429:
        return LLMRateLimitedError(str(exc))
    if status_code == 408:  # litellm's own marker for a timeout, not real HTTP 408
        return LLMTimeoutError(str(exc))
    return LLMError(str(exc))


def _resolve_provider_name(model: str) -> str:
    """Resolve `model`'s provider via litellm's own model-to-provider mapping.

    Args:
        model: The litellm model string.

    Returns:
        str: the resolved provider name, or "unknown" if resolution fails.
    """
    try:
        return cast(str, litellm.get_llm_provider(model)[1])
    except Exception:
        return "unknown"


def _message_parts(message: Message) -> list[dict[str, object]]:
    """Map `message` to the GenAI semantic conventions' `parts` array shape.

    Args:
        message: The message to convert.

    Returns:
        list[dict[str, object]]: one part per content item, per the input/output messages
            JSON schema.
    """
    if message.role == "tool":
        return [
            {
                "type": "tool_call_response",
                "id": message.tool_call_id,
                "result": message.content,
            }
        ]
    parts: list[dict[str, object]] = []
    if message.content is not None:
        parts.append({"type": "text", "content": message.content})
    for call in message.tool_calls or []:
        try:
            arguments: object = json.loads(call.function.arguments)
        except json.JSONDecodeError:
            arguments = call.function.arguments
        parts.append(
            {
                "type": "tool_call",
                "id": call.id,
                "name": call.function.name,
                "arguments": arguments,
            }
        )
    return parts


def _to_gen_ai_message(message: Message) -> dict[str, object]:
    """Map `message` to the GenAI semantic conventions' message-object shape.

    Args:
        message: The message to convert.

    Returns:
        dict[str, object]: `{"role": ..., "parts": [...]}`, per the input/output messages
            JSON schema.
    """
    return {"role": message.role, "parts": _message_parts(message)}


def _to_gen_ai_tool_definition(tool: dict[str, Any]) -> dict[str, object]:
    """Map an OpenAI-format function schema to the GenAI tool-definition shape.

    Args:
        tool: An OpenAI-format `{"type": "function", "function": {...}}` tool schema.

    Returns:
        dict[str, object]: the same schema flattened to `{"type", "name", "description",
            "parameters"}`, per the tool definitions JSON schema.
    """
    function = tool["function"]
    return {
        "type": tool["type"],
        "name": function["name"],
        "description": function["description"],
        "parameters": function["parameters"],
    }


class LiteLLMAdapter:
    """Turns messages into a completion via litellm, supporting any litellm provider."""

    def __init__(
        self,
        model: str,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        context_window: int | None = None,
        num_retries: int = 2,
        timeout: float | None = None,
        retry_base_delay: float = 1.0,
        retry_max_delay: float = 30.0,
        retry_multiplier: float = 2.0,
        max_concurrent_requests: int | None = None,
    ) -> None:
        """Initialize adapter with a model identifier and its configured sampling defaults.

        Args:
            model: The litellm model string (e.g., 'openai/gpt-4o').
            temperature: Default sampling temperature.
            top_p: Default nucleus sampling value.
            max_tokens: Default max output tokens.
            context_window: Override for the model's context-window size.
            num_retries: Retry count for retriable failures.
            timeout: Per-attempt timeout in seconds.
            retry_base_delay: Delay before the first retry, in seconds.
            retry_max_delay: Cap on delay between retries, in seconds.
            retry_multiplier: Backoff multiplier for retry delay.
            max_concurrent_requests: Cap on concurrent in-flight calls to this model.
        """
        self._model = model
        self._provider_name = _resolve_provider_name(model)
        self._temperature = temperature
        self._top_p = top_p
        self._max_tokens = max_tokens
        self._context_window = context_window
        self._num_retries = num_retries
        self._timeout = timeout
        self._retry_base_delay = retry_base_delay
        self._retry_max_delay = retry_max_delay
        self._retry_multiplier = retry_multiplier
        self._max_concurrent = max_concurrent_requests
        self._in_flight = 0

    @classmethod
    def from_config(cls, config: LLMConfig) -> "LiteLLMAdapter":
        """Build an adapter from a model's configuration, mapping its fields onto `__init__`.

        Args:
            config: The model's startup configuration.

        Returns:
            LiteLLMAdapter: the built adapter.
        """
        return cls(
            config.model,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            context_window=config.context_window,
            num_retries=config.num_retries,
            timeout=config.timeout,
            retry_base_delay=config.retry_base_delay,
            retry_max_delay=config.retry_max_delay,
            retry_multiplier=config.retry_multiplier,
            max_concurrent_requests=config.max_concurrent_requests,
        )

    async def complete(
        self,
        messages: list[Message],
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> Completion:
        """Send messages to litellm and map the result to a Completion.

        Args:
            messages: Conversation history to send.
            temperature: Temperature override for this call.
            top_p: Top-p override for this call.
            max_tokens: Max-tokens override for this call.
            tools: OpenAI-format function schemas to offer the model.

        Returns:
            Completion: the mapped model response.

        Raises:
            LLMOverloadedError: if the concurrency cap is already reached.
            LLMRateLimitedError: if the provider rate-limited the request.
            LLMTimeoutError: if the request timed out.
            LLMError: if the call fails, or the response has neither content nor tool_calls.
        """
        with tracer.start_as_current_span(
            f"chat {self._model}",
            kind=trace.SpanKind.CLIENT,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            span.set_attribute("gen_ai.operation.name", "chat")
            span.set_attribute("gen_ai.request.model", self._model)
            span.set_attribute("gen_ai.provider.name", self._provider_name)
            stamp_run_context(span)
            resolved_temperature = _first_not_none(temperature, self._temperature)
            resolved_top_p = _first_not_none(top_p, self._top_p)
            resolved_max_tokens = _first_not_none(max_tokens, self._max_tokens)
            if resolved_temperature is not None:
                span.set_attribute("gen_ai.request.temperature", resolved_temperature)
            if resolved_top_p is not None:
                span.set_attribute("gen_ai.request.top_p", resolved_top_p)
            if resolved_max_tokens is not None:
                span.set_attribute("gen_ai.request.max_tokens", resolved_max_tokens)
            if capture_content_enabled():
                span.set_attribute(
                    "gen_ai.input.messages",
                    truncate_keeping_recent(
                        json.dumps([_to_gen_ai_message(m) for m in messages]),
                        _MAX_MESSAGE_ATTRIBUTE_CHARS,
                    ),
                )
                if tools:
                    span.set_attribute(
                        "gen_ai.tool.definitions",
                        json.dumps([_to_gen_ai_tool_definition(t) for t in tools]),
                    )
            if self._max_concurrent is not None and self._in_flight >= self._max_concurrent:
                logger.warning(
                    "model %s at capacity (%d/%d), rejecting",
                    self._model,
                    self._in_flight,
                    self._max_concurrent,
                    extra={
                        "model": self._model,
                        "in_flight": self._in_flight,
                        "max_concurrent": self._max_concurrent,
                    },
                )
                error = LLMOverloadedError(
                    f"model '{self._model}' is at capacity "
                    f"({self._max_concurrent} concurrent requests)"
                )
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(error)))
                span.set_attribute("error.type", type(error).__name__)
                raise error
            self._in_flight += 1
            try:
                completion = await self._complete_with_retries(
                    messages, resolved_temperature, resolved_top_p, resolved_max_tokens, tools
                )
            except Exception as exc:
                record_gated_exception(span, exc)
                raise
            finally:
                self._in_flight -= 1
            span.set_attribute("gen_ai.usage.input_tokens", completion.usage.prompt_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", completion.usage.completion_tokens)
            span.set_attribute("gen_ai.response.finish_reasons", (completion.finish_reason,))
            if completion.response_id is not None:
                span.set_attribute("gen_ai.response.id", completion.response_id)
            if completion.response_model is not None:
                span.set_attribute("gen_ai.response.model", completion.response_model)
            if capture_content_enabled():
                output_message = _to_gen_ai_message(completion.message)
                output_message["finish_reason"] = completion.finish_reason
                span.set_attribute(
                    "gen_ai.output.messages",
                    truncate(json.dumps([output_message]), _MAX_MESSAGE_ATTRIBUTE_CHARS),
                )
            return completion

    async def _complete_with_retries(
        self,
        messages: list[Message],
        temperature: float | None,
        top_p: float | None,
        max_tokens: int | None,
        tools: list[dict[str, Any]] | None,
    ) -> Completion:
        """Send messages to litellm with retries, and map the result to a Completion.

        Args:
            messages: Conversation history to send.
            temperature: Resolved sampling temperature for this call, if any.
            top_p: Resolved nucleus sampling value for this call, if any.
            max_tokens: Resolved max output tokens for this call, if any.
            tools: OpenAI-format function schemas to offer the model.

        Returns:
            Completion: the mapped model response.

        Raises:
            LLMRateLimitedError: if the provider rate-limited the request.
            LLMTimeoutError: if the request timed out.
            LLMError: if the call fails, or the response has neither content nor tool_calls.
        """
        params: dict[str, Any] = {
            key: value
            for key, value in (
                ("temperature", temperature),
                ("top_p", top_p),
                ("max_completion_tokens", max_tokens),
                ("timeout", self._timeout),
            )
            if value is not None
        }
        if tools:
            params["tools"] = tools
        else:
            messages = flatten_tool_exchanges_for_no_tools_request(messages)

        outbound_messages = [m.model_dump(exclude_none=True) for m in messages]

        for attempt in range(self._num_retries + 1):
            try:
                response = await litellm.acompletion(
                    model=self._model,
                    messages=outbound_messages,
                    **params,
                )
            except Exception as exc:
                if isinstance(exc, litellm.ContextWindowExceededError):  # type: ignore[attr-defined]
                    raise LLMContextWindowExceededError(str(exc)) from exc
                status_code = getattr(exc, "status_code", None)
                classified = _classify(exc, status_code)
                retriable = status_code in (429, 408) or _is_retriable(status_code)
                extra = {
                    "exception_type": type(exc).__name__,
                    "status_code": status_code,
                    "attempt": attempt + 1,
                }
                if not retriable or attempt == self._num_retries:
                    logger.warning(
                        "LLM call failed permanently: %s (status=%s)",
                        type(exc).__name__,
                        status_code,
                        exc_info=True,
                        extra=extra,
                    )
                    raise classified from exc
                delay = min(
                    self._retry_base_delay * (self._retry_multiplier**attempt),
                    self._retry_max_delay,
                ) * random.uniform(0.5, 1.0)
                logger.warning(
                    "LLM call failed: %s (status=%s), attempt %d/%d, retrying in %.1fs: %s",
                    type(exc).__name__,
                    status_code,
                    attempt + 1,
                    self._num_retries + 1,
                    delay,
                    exc,
                    extra=extra,
                )
                await asyncio.sleep(delay)
                continue

            # Parsing/constructing the result from a *successful* call is kept outside the
            # except block above: a response-shape problem here has no .status_code, so
            # treating it as retriable would silently retry a deterministically-malformed
            # response num_retries times before reporting it, instead of failing once.
            try:
                choice = response.choices[0]
                raw_tool_calls = getattr(choice.message, "tool_calls", None)
                tool_calls = (
                    [
                        ToolCall(
                            id=tc.id,
                            function=ToolCallFunction(
                                name=tc.function.name, arguments=tc.function.arguments
                            ),
                        )
                        for tc in raw_tool_calls
                    ]
                    if raw_tool_calls
                    else None
                )
                content = choice.message.content or None
                if content is None and tool_calls is None:
                    raise LLMError("litellm returned a message with neither content nor tool_calls")
                try:
                    cost_usd = round(litellm.completion_cost(completion_response=response), 10)
                except Exception:
                    cost_usd = None
                return Completion(
                    message=Message(role="assistant", content=content, tool_calls=tool_calls),
                    usage=Usage(
                        prompt_tokens=response.usage.prompt_tokens,
                        completion_tokens=response.usage.completion_tokens,
                        total_tokens=response.usage.total_tokens,
                        cost_usd=cost_usd,
                    ),
                    finish_reason=choice.finish_reason,
                    response_id=getattr(response, "id", None),
                    response_model=getattr(response, "model", None),
                )
            except LLMError as malformed:
                logger.warning(
                    "provider returned a malformed response: %s",
                    malformed,
                    exc_info=True,
                    extra={"exception_type": type(malformed).__name__},
                )
                raise
            except (IndexError, AttributeError, KeyError, TypeError) as exc:
                shape_error = LLMError(f"litellm returned an unexpected response shape: {exc}")
                logger.warning(
                    "provider returned a malformed response: %s",
                    shape_error,
                    exc_info=True,
                    extra={"exception_type": type(shape_error).__name__},
                )
                raise shape_error from exc
        raise AssertionError("unreachable — the loop above always returns or raises")

    def max_input_tokens(self) -> int:
        """Return this model's maximum input token count.

        Returns:
            int: the model's maximum input token count.

        Raises:
            LLMError: if no known limit is found for this model.
        """
        if self._context_window is not None:
            return self._context_window
        try:
            info = litellm.get_model_info(self._model)
        except Exception as exc:
            raise LLMError(str(exc)) from exc
        max_input = info.get("max_input_tokens")
        if max_input is None:
            raise LLMError(f"litellm has no max_input_tokens for model '{self._model}'")
        return max_input
