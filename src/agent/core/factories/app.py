"""Factory for building runtime registries from configuration."""

import os
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version
from typing import Any

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)

from agent.adapters import guardrails_ai
from agent.adapters.guardrails_ai import GuardrailsAIAdapter
from agent.adapters.litellm import LiteLLMAdapter
from agent.adapters.llm_registry_provider import (
    LLM_REGISTRY_PROVIDER,
    register_llm_registry_provider,
)
from agent.core.exceptions import ConfigError
from agent.core.models.config import (
    AgentConfig,
    AppConfig,
    CompactionConfig,
    GuardrailConfig,
    LLMConfig,
    LoggingConfig,
    SessionStoreConfig,
    StrategyConfig,
    ToolConfig,
    TracingConfig,
)
from agent.core.protocols.isession_store import ISessionStore
from agent.core.protocols.istrategy import IStrategy
from agent.core.protocols.itool import ITool
from agent.core.registries.agent import AgentRegistry
from agent.core.registries.guardrail import GuardrailRegistry
from agent.core.registries.llm import LLMRegistry
from agent.core.registries.strategy import StrategyRegistry
from agent.core.registries.tool import ToolRegistry
from agent.core.session_stores.in_memory import InMemorySessionStore
from agent.core.strategies.react import ReactStrategy
from agent.core.tools.get_current_time import GetCurrentTimeTool
from agent.core.tracing import set_capture_content

_TOOL_IMPLEMENTATIONS: dict[str, Callable[[], ITool]] = {
    "get_current_time": GetCurrentTimeTool,
}

_STRATEGY_IMPLEMENTATIONS: dict[str, Callable[[], IStrategy]] = {
    "react": ReactStrategy,
}

_SESSION_STORE_IMPLEMENTATIONS: dict[str, Callable[..., ISessionStore]] = {
    "in_memory": InMemorySessionStore,
}

_DEFAULT_SERVICE_NAME = "agent-core"


def _build_llm_registry(llm_configs: list[LLMConfig]) -> tuple[LLMRegistry, set[str]]:
    """Build the LLM registry.

    Args:
        llm_configs: Startup LLM allow-list.

    Returns:
        tuple[LLMRegistry, set[str]]: the registry, and its known model names.

    Raises:
        ConfigError: on a duplicate model name.
    """
    llm_registry = LLMRegistry()
    seen_models: set[str] = set()
    for llm_config in llm_configs:
        if llm_config.model in seen_models:
            raise ConfigError(f"duplicate LLM model in config: {llm_config.model}")
        seen_models.add(llm_config.model)
        llm_registry.register(llm_config.model, LiteLLMAdapter.from_config(llm_config))
    return llm_registry, seen_models


def _build_tool_registry(tool_configs: list[ToolConfig]) -> tuple[ToolRegistry, set[str]]:
    """Build the tool registry.

    Args:
        tool_configs: Startup tool allow-list.

    Returns:
        tuple[ToolRegistry, set[str]]: the registry, and its known tool names.

    Raises:
        ConfigError: on a duplicate name or missing implementation.
    """
    tool_registry = ToolRegistry()
    seen_tools: set[str] = set()
    for tool_config in tool_configs:
        if tool_config.name in seen_tools:
            raise ConfigError(f"duplicate tool name in config: {tool_config.name}")
        seen_tools.add(tool_config.name)
        implementation = _TOOL_IMPLEMENTATIONS.get(tool_config.name)
        if implementation is None:
            raise ConfigError(f"no tool implementation registered for: {tool_config.name}")
        tool_registry.register(tool_config.name, implementation())
    return tool_registry, seen_tools


def _build_strategy_registry(
    strategy_configs: list[StrategyConfig],
) -> tuple[StrategyRegistry, set[str]]:
    """Build the strategy registry.

    Args:
        strategy_configs: Startup strategy allow-list.

    Returns:
        tuple[StrategyRegistry, set[str]]: the registry, and its known strategy names.

    Raises:
        ConfigError: on a duplicate name or missing implementation.
    """
    strategy_registry = StrategyRegistry()
    seen_strategies: set[str] = set()
    for strategy_config in strategy_configs:
        if strategy_config.name in seen_strategies:
            raise ConfigError(f"duplicate strategy name in config: {strategy_config.name}")
        seen_strategies.add(strategy_config.name)
        implementation = _STRATEGY_IMPLEMENTATIONS.get(strategy_config.name)
        if implementation is None:
            raise ConfigError(f"no strategy implementation registered for: {strategy_config.name}")
        strategy_registry.register(strategy_config.name, implementation())
    return strategy_registry, seen_strategies


def _enforce_llm_callable(
    validator_cls: type, validator_id: str, validator_params: dict[str, Any], known_models: set[str]
) -> dict[str, Any]:
    """Require and redirect a validator's `llm_callable` constructor argument, if it has one.

    Args:
        validator_cls: The validator's already-resolved class.
        validator_id: Guardrails AI Hub id, for error messages.
        validator_params: The guardrail's configured validator constructor arguments.
        known_models: Model names registered in this process's LLM registry.

    Returns:
        dict[str, Any]: `validator_params`, with `llm_callable` rewritten to route through
            the registered LLM, if the resolved validator declares that parameter.
            Unchanged otherwise.

    Raises:
        ConfigError: the validator declares `llm_callable`, but `validator_params` omits
            it or sets it to a name not in `known_models`.
    """
    if not guardrails_ai.declares_llm_callable(validator_cls):
        return validator_params
    llm_callable = validator_params.get("llm_callable")
    if llm_callable not in known_models:
        raise ConfigError(
            f"validator '{validator_id}' declares an 'llm_callable' constructor argument — "
            f"validator_params must set it to a model name registered in this process's "
            f"'llms' config (got: {llm_callable!r})"
        )
    # The prefix is what makes litellm dispatch this call to this codebase's own registered
    # custom provider instead of a real one — litellm strips it before handing the rest of
    # the string to the handler, which resolves what's left against the same LLM registry.
    return {**validator_params, "llm_callable": f"{LLM_REGISTRY_PROVIDER}/{llm_callable}"}


def _build_guardrail_registry(
    guardrail_configs: list[GuardrailConfig],
    known_models: set[str],
    llm_registry: LLMRegistry,
) -> tuple[GuardrailRegistry, set[str]]:
    """Build the guardrail registry.

    Args:
        guardrail_configs: Startup guardrail allow-list.
        known_models: Model names already registered, for `llm_callable` enforcement.
        llm_registry: Registry `llm_callable`-declaring validators are redirected through.

    Returns:
        tuple[GuardrailRegistry, set[str]]: the registry, and its known guardrail names.

    Raises:
        ConfigError: on a duplicate name, an unresolvable validator id, or a validator's
            `llm_callable` argument missing or unregistered.
    """
    guardrail_registry = GuardrailRegistry()
    seen_guardrails: set[str] = set()
    if guardrail_configs:
        register_llm_registry_provider(llm_registry)
    for guardrail_config in guardrail_configs:
        if guardrail_config.name in seen_guardrails:
            raise ConfigError(f"duplicate guardrail name in config: {guardrail_config.name}")
        seen_guardrails.add(guardrail_config.name)
        try:
            validator_cls = guardrails_ai.resolve_validator(guardrail_config.validator_id)
            validator_params = _enforce_llm_callable(
                validator_cls,
                guardrail_config.validator_id,
                guardrail_config.validator_params,
                known_models,
            )
            adapter = GuardrailsAIAdapter(
                guardrail_config.name,
                guardrail_config.validator_id,
                validator_params,
                guardrail_config.action,
                validator_cls=validator_cls,
            )
        except ConfigError:
            raise
        except Exception as exc:  # noqa: BLE001 — a Hub validator's constructor can raise anything
            raise ConfigError(f"guardrail '{guardrail_config.name}': {exc}") from exc
        guardrail_registry.register(guardrail_config.name, adapter)
    return guardrail_registry, seen_guardrails


def _build_agent_registry(
    agent_configs: list[AgentConfig],
    known_models: set[str],
    known_tools: set[str],
    known_strategies: set[str],
) -> AgentRegistry:
    """Build the agent registry.

    Args:
        agent_configs: Startup agent allow-list.
        known_models: Model names already registered.
        known_tools: Tool names already registered.
        known_strategies: Strategy names already registered.

    Returns:
        AgentRegistry: the built registry.

    Raises:
        ConfigError: on any invalid agent declaration.
    """
    agent_registry = AgentRegistry()
    seen_agents: set[str] = set()
    for agent_config in agent_configs:
        if agent_config.name in seen_agents:
            raise ConfigError(f"duplicate agent name in config: {agent_config.name}")
        seen_agents.add(agent_config.name)
        if agent_config.model not in known_models:
            raise ConfigError(
                f"agent '{agent_config.name}' declares unknown model: {agent_config.model}"
            )
        if agent_config.strategy not in known_strategies:
            raise ConfigError(
                f"agent '{agent_config.name}' declares unknown strategy: {agent_config.strategy}"
            )
        seen_agent_tools: set[str] = set()
        for tool_name in agent_config.tools:
            if tool_name not in known_tools:
                raise ConfigError(f"agent '{agent_config.name}' declares unknown tool: {tool_name}")
            if tool_name in seen_agent_tools:
                raise ConfigError(
                    f"agent '{agent_config.name}' declares duplicate tool: {tool_name}"
                )
            seen_agent_tools.add(tool_name)
        agent_registry.register(agent_config.name, agent_config)
    return agent_registry


def _build_session_store(
    config: SessionStoreConfig,
    max_sessions: int | None,
    on_evict: Callable[[str, str], None] | None,
) -> ISessionStore:
    """Build the session store.

    Args:
        config: Session store settings.
        max_sessions: Cap on how many distinct sessions are kept at once.
        on_evict: Called with `(agent, session_id)` when the store evicts a session, so
            other per-session state keyed the same way can be kept in sync.

    Returns:
        ISessionStore: the built session store.

    Raises:
        ConfigError: no implementation is registered for the configured type.
    """
    implementation = _SESSION_STORE_IMPLEMENTATIONS.get(config.type)
    if implementation is None:
        raise ConfigError(f"no session store implementation registered for: {config.type}")
    return implementation(max_sessions, on_evict=on_evict)


def _validate_guardrail_list(
    agent_name: str, list_name: str, guardrail_names: list[str], known_guardrails: set[str]
) -> None:
    """Validate one of an agent's guardrail-checkpoint lists against what's registered.

    Args:
        agent_name: The agent declaring this list, for error messages.
        list_name: Which checkpoint list this is, for error messages.
        guardrail_names: The names to validate.
        known_guardrails: Guardrail names already registered.

    Raises:
        ConfigError: a name is unknown, or a name is duplicated within the list.
    """
    seen: set[str] = set()
    for guardrail_name in guardrail_names:
        if guardrail_name not in known_guardrails:
            raise ConfigError(
                f"agent '{agent_name}' declares unknown guardrail in {list_name}: {guardrail_name}"
            )
        if guardrail_name in seen:
            raise ConfigError(
                f"agent '{agent_name}' declares duplicate guardrail in {list_name}: "
                f"{guardrail_name}"
            )
        seen.add(guardrail_name)


def _span_to_jsonl(span: ReadableSpan) -> str:
    """Format one finished span as a single JSON Lines record.

    Args:
        span: The finished span to format.

    Returns:
        str: the span as one line of JSON, newline-terminated.
    """
    json_line: str = span.to_json(indent=None)
    return json_line + "\n"


def configure_tracing(config: TracingConfig) -> TracerProvider | None:
    """Build and register the process-wide TracerProvider.

    Args:
        config: Tracing settings (destinations, content capture).

    Returns:
        TracerProvider | None: the provider this call built and registered, or `None` if no
            destination was configured.
    """
    if not config.console and config.endpoint is None:
        set_capture_content(False)
        return None
    set_capture_content(config.capture_content)
    resource = None
    if "OTEL_SERVICE_NAME" not in os.environ:
        attributes: dict[str, str] = {SERVICE_NAME: _DEFAULT_SERVICE_NAME}
        try:
            attributes[SERVICE_VERSION] = _package_version("agent-core")
        except PackageNotFoundError:
            pass
        resource = Resource.create(attributes)
    provider = TracerProvider(resource=resource)
    if config.console:
        provider.add_span_processor(
            SimpleSpanProcessor(ConsoleSpanExporter(formatter=_span_to_jsonl))
        )
    if config.endpoint is not None:
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=config.endpoint)))
    trace.set_tracer_provider(provider)
    return provider


def build_registries(
    config: AppConfig,
    on_session_evict: Callable[[str, str], None] | None = None,
) -> tuple[
    LLMRegistry,
    AgentRegistry,
    ToolRegistry,
    StrategyRegistry,
    GuardrailRegistry,
    ISessionStore,
    str | None,
    CompactionConfig | None,
    LoggingConfig,
    TracingConfig,
]:
    """Build populated registries from already-parsed startup configuration.

    Args:
        config: The startup configuration.
        on_session_evict: Called with `(agent, session_id)` when the session store evicts
            a session, so other per-session state keyed the same way can be kept in sync.

    Returns:
        The five registries, the built session store, the process-wide `base_prompt`, and the
        raw `compaction`, `logging`, and `tracing` config values.

    Raises:
        ConfigError: if `config` is invalid.
    """
    llm_registry, known_models = _build_llm_registry(config.llms)
    tool_registry, known_tools = _build_tool_registry(config.tools)
    strategy_registry, known_strategies = _build_strategy_registry(config.strategies)
    guardrail_registry, known_guardrails = _build_guardrail_registry(
        config.guardrails, known_models, llm_registry
    )
    agent_registry = _build_agent_registry(
        config.agents, known_models, known_tools, known_strategies
    )
    session_store = _build_session_store(
        config.session_store, config.max_sessions, on_session_evict
    )
    if config.compaction is not None and config.compaction.model not in known_models:
        raise ConfigError(f"compaction declares unknown model: {config.compaction.model}")

    for agent_config in config.agents:
        if agent_config.allowed_tools is not None:
            for tool_name in agent_config.allowed_tools:
                if tool_name not in known_tools:
                    raise ConfigError(
                        f"agent '{agent_config.name}' declares unknown tool in "
                        f"allowed_tools: {tool_name}"
                    )
        if agent_config.allowed_models is not None:
            for model_name in agent_config.allowed_models:
                if model_name not in known_models:
                    raise ConfigError(
                        f"agent '{agent_config.name}' declares unknown model in "
                        f"allowed_models: {model_name}"
                    )
        if agent_config.allowed_strategies is not None:
            for strategy_name in agent_config.allowed_strategies:
                if strategy_name not in known_strategies:
                    raise ConfigError(
                        f"agent '{agent_config.name}' declares unknown strategy in "
                        f"allowed_strategies: {strategy_name}"
                    )
        _validate_guardrail_list(
            agent_config.name, "input_guardrails", agent_config.input_guardrails, known_guardrails
        )
        _validate_guardrail_list(
            agent_config.name,
            "tool_output_guardrails",
            agent_config.tool_output_guardrails,
            known_guardrails,
        )
        _validate_guardrail_list(
            agent_config.name,
            "output_guardrails",
            agent_config.output_guardrails,
            known_guardrails,
        )

    return (
        llm_registry,
        agent_registry,
        tool_registry,
        strategy_registry,
        guardrail_registry,
        session_store,
        config.base_prompt,
        config.compaction,
        config.logging,
        config.tracing,
    )
