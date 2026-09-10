import pytest
from pydantic import ValidationError

from agent.core.models.config import (
    AgentConfig,
    AppConfig,
    CompactionConfig,
    GuardrailConfig,
    LLMConfig,
    LoggingConfig,
    StrategyConfig,
    ToolConfig,
    TracingConfig,
)


def test_llm_config_given_model_string_constructs():
    config = LLMConfig(model="openai/gpt-4o")

    assert config.model == "openai/gpt-4o"


def test_llm_config_given_no_sampling_params_defaults_to_none():
    config = LLMConfig(model="openai/gpt-4o")

    assert config.temperature is None
    assert config.top_p is None
    assert config.max_tokens is None


def test_llm_config_given_sampling_params_constructs():
    config = LLMConfig(model="openai/gpt-4o", temperature=0.2, top_p=0.9, max_tokens=512)

    assert config.temperature == 0.2
    assert config.top_p == 0.9
    assert config.max_tokens == 512


def test_llm_config_given_negative_temperature_raises_validation_error():
    with pytest.raises(ValidationError):
        LLMConfig(model="openai/gpt-4o", temperature=-0.1)


def test_llm_config_given_temperature_above_two_still_constructs():
    # No upper bound: provider-specific (2 for OpenAI, 1 for Anthropic), left for the
    # provider itself to reject — matches AgentRunRequest.temperature in api/schemas.py.
    config = LLMConfig(model="openai/gpt-4o", temperature=2.1)

    assert config.temperature == 2.1


def test_llm_config_given_negative_top_p_raises_validation_error():
    with pytest.raises(ValidationError):
        LLMConfig(model="openai/gpt-4o", top_p=-0.1)


def test_llm_config_given_top_p_above_one_raises_validation_error():
    with pytest.raises(ValidationError):
        LLMConfig(model="openai/gpt-4o", top_p=1.1)


def test_llm_config_given_zero_max_tokens_raises_validation_error():
    with pytest.raises(ValidationError):
        LLMConfig(model="openai/gpt-4o", max_tokens=0)


def test_llm_config_given_negative_max_tokens_raises_validation_error():
    with pytest.raises(ValidationError):
        LLMConfig(model="openai/gpt-4o", max_tokens=-1)


def test_llm_config_given_no_context_window_defaults_to_none():
    config = LLMConfig(model="openai/gpt-4o")

    assert config.context_window is None


def test_llm_config_given_context_window_constructs():
    config = LLMConfig(model="openai/gpt-4o", context_window=128000)

    assert config.context_window == 128000


def test_llm_config_given_context_window_zero_raises_validation_error():
    with pytest.raises(ValidationError):
        LLMConfig(model="openai/gpt-4o", context_window=0)


def test_llm_config_given_context_window_negative_raises_validation_error():
    with pytest.raises(ValidationError):
        LLMConfig(model="openai/gpt-4o", context_window=-1)


def test_logging_config_given_no_args_defaults_to_info():
    config = LoggingConfig()

    assert config.level == "INFO"


def test_logging_config_given_level_constructs():
    config = LoggingConfig(level="DEBUG")

    assert config.level == "DEBUG"


def test_logging_config_given_invalid_level_raises_validation_error():
    with pytest.raises(ValidationError):
        LoggingConfig(level="verbose")


def test_app_config_given_valid_json_constructs():
    raw = '{"llms": [{"model": "openai/gpt-4o"}], "logging": {"level": "DEBUG"}}'

    config = AppConfig.model_validate_json(raw)

    assert config.llms[0].model == "openai/gpt-4o"
    assert config.logging.level == "DEBUG"


def test_app_config_given_missing_logging_defaults_to_info():
    raw = '{"llms": [{"model": "openai/gpt-4o"}]}'

    config = AppConfig.model_validate_json(raw)

    assert config.logging.level == "INFO"


def test_agent_config_given_required_fields_constructs():
    config = AgentConfig(
        name="researcher",
        system_prompt="You are a research assistant.",
        model="openai/gpt-4o",
        strategy="react",
    )

    assert config.name == "researcher"
    assert config.system_prompt == "You are a research assistant."
    assert config.model == "openai/gpt-4o"
    assert config.strategy == "react"
    assert config.max_tool_iterations == 10
    assert config.temperature is None
    assert config.top_p is None
    assert config.max_tokens is None


def test_agent_config_given_sampling_params_constructs():
    config = AgentConfig(
        name="researcher",
        system_prompt="You are a research assistant.",
        model="openai/gpt-4o",
        strategy="react",
        temperature=0.1,
        top_p=0.8,
        max_tokens=256,
    )

    assert config.temperature == 0.1
    assert config.top_p == 0.8
    assert config.max_tokens == 256


def test_agent_config_given_top_p_above_one_raises_validation_error():
    # AgentConfig inherits SamplingDefaults' bounds the same way LLMConfig does — one
    # confirming test here, full bound coverage lives on the LLMConfig tests above.
    with pytest.raises(ValidationError):
        AgentConfig(
            name="researcher",
            system_prompt="You are a research assistant.",
            model="openai/gpt-4o",
            strategy="react",
            top_p=1.1,
        )


def test_app_config_given_no_agents_defaults_to_empty_list():
    raw = '{"llms": [{"model": "openai/gpt-4o"}]}'

    config = AppConfig.model_validate_json(raw)

    assert config.agents == []


def test_app_config_given_agents_constructs():
    raw = (
        '{"llms": [{"model": "openai/gpt-4o"}], '
        '"agents": [{"name": "researcher", "system_prompt": "You are helpful.", '
        '"model": "openai/gpt-4o", "strategy": "react"}]}'
    )

    config = AppConfig.model_validate_json(raw)

    assert config.agents[0].name == "researcher"
    assert config.agents[0].model == "openai/gpt-4o"


def test_tool_config_given_name_constructs():
    config = ToolConfig(name="get_current_time")

    assert config.name == "get_current_time"


def test_agent_config_given_no_tools_defaults_to_empty_list():
    config = AgentConfig(
        name="researcher",
        system_prompt="You are a research assistant.",
        model="openai/gpt-4o",
        strategy="react",
    )

    assert config.tools == []


def test_agent_config_given_tools_constructs():
    config = AgentConfig(
        name="researcher",
        system_prompt="You are a research assistant.",
        model="openai/gpt-4o",
        strategy="react",
        tools=["get_current_time"],
    )

    assert config.tools == ["get_current_time"]


def test_app_config_given_no_tools_defaults_to_empty_list():
    raw = '{"llms": [{"model": "openai/gpt-4o"}]}'

    config = AppConfig.model_validate_json(raw)

    assert config.tools == []


def test_agent_config_given_no_strategy_raises_validation_error():
    with pytest.raises(ValidationError):
        AgentConfig(
            name="researcher",
            system_prompt="You are a research assistant.",
            model="openai/gpt-4o",
        )


def test_agent_config_given_max_tool_iterations_override_constructs():
    config = AgentConfig(
        name="researcher",
        system_prompt="You are a research assistant.",
        model="openai/gpt-4o",
        strategy="react",
        max_tool_iterations=3,
    )

    assert config.max_tool_iterations == 3


def test_app_config_given_tools_constructs():
    raw = '{"llms": [{"model": "openai/gpt-4o"}], "tools": [{"name": "get_current_time"}]}'

    config = AppConfig.model_validate_json(raw)

    assert config.tools[0].name == "get_current_time"


def test_agent_config_given_no_max_input_chars_defaults_to_none():
    config = AgentConfig(
        name="researcher",
        system_prompt="You are a research assistant.",
        model="openai/gpt-4o",
        strategy="react",
    )

    assert config.max_input_chars is None


def test_agent_config_given_max_input_chars_constructs():
    config = AgentConfig(
        name="researcher",
        system_prompt="You are a research assistant.",
        model="openai/gpt-4o",
        strategy="react",
        max_input_chars=20000,
    )

    assert config.max_input_chars == 20000


def test_agent_config_given_max_input_chars_zero_raises_validation_error():
    with pytest.raises(ValidationError):
        AgentConfig(
            name="researcher",
            system_prompt="You are a research assistant.",
            model="openai/gpt-4o",
            strategy="react",
            max_input_chars=0,
        )


def test_agent_config_given_no_max_tool_result_chars_defaults_to_none():
    config = AgentConfig(
        name="researcher",
        system_prompt="You are a research assistant.",
        model="openai/gpt-4o",
        strategy="react",
    )

    assert config.max_tool_result_chars is None


def test_agent_config_given_max_tool_result_chars_constructs():
    config = AgentConfig(
        name="researcher",
        system_prompt="You are a research assistant.",
        model="openai/gpt-4o",
        strategy="react",
        max_tool_result_chars=5000,
    )

    assert config.max_tool_result_chars == 5000


def test_agent_config_given_max_tool_result_chars_zero_raises_validation_error():
    with pytest.raises(ValidationError):
        AgentConfig(
            name="researcher",
            system_prompt="You are a research assistant.",
            model="openai/gpt-4o",
            strategy="react",
            max_tool_result_chars=0,
        )


def test_agent_config_given_max_tool_result_chars_negative_raises_validation_error():
    with pytest.raises(ValidationError):
        AgentConfig(
            name="researcher",
            system_prompt="You are a research assistant.",
            model="openai/gpt-4o",
            strategy="react",
            max_tool_result_chars=-1,
        )


def test_compaction_config_given_required_field_constructs_with_defaults():
    config = CompactionConfig(model="anthropic/claude-3-5-haiku-20241022")

    assert config.model == "anthropic/claude-3-5-haiku-20241022"
    assert config.token_budget_pct == 0.8
    assert config.keep_recent_turns == 4
    assert "Summarize" in config.prompt


def test_compaction_config_given_overrides_constructs():
    config = CompactionConfig(
        model="anthropic/claude-3-5-haiku-20241022",
        token_budget_pct=0.5,
        keep_recent_turns=2,
        prompt="Custom instructions.",
    )

    assert config.token_budget_pct == 0.5
    assert config.keep_recent_turns == 2
    assert config.prompt == "Custom instructions."


def test_compaction_config_given_token_budget_pct_zero_raises_validation_error():
    with pytest.raises(ValidationError):
        CompactionConfig(model="anthropic/claude-3-5-haiku-20241022", token_budget_pct=0.0)


def test_compaction_config_given_token_budget_pct_above_one_raises_validation_error():
    with pytest.raises(ValidationError):
        CompactionConfig(model="anthropic/claude-3-5-haiku-20241022", token_budget_pct=1.1)


def test_compaction_config_given_negative_keep_recent_turns_raises_validation_error():
    with pytest.raises(ValidationError):
        CompactionConfig(model="anthropic/claude-3-5-haiku-20241022", keep_recent_turns=-1)


def test_app_config_given_no_compaction_defaults_to_none():
    raw = '{"llms": [{"model": "openai/gpt-4o"}]}'

    config = AppConfig.model_validate_json(raw)

    assert config.compaction is None


def test_app_config_given_compaction_constructs():
    raw = (
        '{"llms": [{"model": "openai/gpt-4o"}], '
        '"compaction": {"model": "openai/gpt-4o", "token_budget_pct": 0.7, '
        '"keep_recent_turns": 2}}'
    )

    config = AppConfig.model_validate_json(raw)

    assert config.compaction is not None
    assert config.compaction.model == "openai/gpt-4o"
    assert config.compaction.token_budget_pct == 0.7
    assert config.compaction.keep_recent_turns == 2


def test_agent_config_given_no_tool_call_and_total_char_caps_defaults_to_none():
    config = AgentConfig(
        name="researcher",
        system_prompt="You are a research assistant.",
        model="openai/gpt-4o",
        strategy="react",
    )

    assert (config.max_tool_calls_per_round, config.max_tool_results_total_chars) == (None, None)


def test_agent_config_given_tool_call_and_total_char_caps_constructs():
    config = AgentConfig(
        name="researcher",
        system_prompt="You are a research assistant.",
        model="openai/gpt-4o",
        strategy="react",
        max_tool_calls_per_round=3,
        max_tool_results_total_chars=200_000,
    )

    assert (config.max_tool_calls_per_round, config.max_tool_results_total_chars) == (3, 200_000)


def test_agent_config_given_max_tool_calls_per_round_zero_raises_validation_error():
    with pytest.raises(ValidationError):
        AgentConfig(
            name="researcher",
            system_prompt="You are a research assistant.",
            model="openai/gpt-4o",
            strategy="react",
            max_tool_calls_per_round=0,
        )


def test_agent_config_given_max_tool_results_total_chars_zero_raises_validation_error():
    with pytest.raises(ValidationError):
        AgentConfig(
            name="researcher",
            system_prompt="You are a research assistant.",
            model="openai/gpt-4o",
            strategy="react",
            max_tool_results_total_chars=0,
        )


def test_compaction_config_given_no_chunk_turns_defaults_to_four():
    assert CompactionConfig(model="anthropic/claude-3-5-haiku-20241022").chunk_turns == 4


def test_compaction_config_given_chunk_turns_constructs():
    config = CompactionConfig(model="anthropic/claude-3-5-haiku-20241022", chunk_turns=2)

    assert config.chunk_turns == 2


def test_compaction_config_given_zero_chunk_turns_raises_validation_error():
    with pytest.raises(ValidationError):
        CompactionConfig(model="anthropic/claude-3-5-haiku-20241022", chunk_turns=0)


def test_llm_config_given_no_retry_settings_defaults():
    config = LLMConfig(model="openai/gpt-4o")

    assert config.num_retries == 2
    assert config.timeout is None
    assert config.retry_base_delay == 1.0
    assert config.retry_max_delay == 30.0
    assert config.retry_multiplier == 2.0
    assert config.max_concurrent_requests is None


def test_llm_config_given_retry_settings_constructs():
    config = LLMConfig(
        model="openai/gpt-4o",
        num_retries=5,
        timeout=10.0,
        retry_base_delay=0.5,
        retry_max_delay=8.0,
        retry_multiplier=3.0,
        max_concurrent_requests=4,
    )

    assert config.num_retries == 5
    assert config.timeout == 10.0
    assert config.retry_base_delay == 0.5
    assert config.retry_max_delay == 8.0
    assert config.retry_multiplier == 3.0
    assert config.max_concurrent_requests == 4


def test_llm_config_given_negative_num_retries_raises_validation_error():
    with pytest.raises(ValidationError):
        LLMConfig(model="openai/gpt-4o", num_retries=-1)


def test_llm_config_given_zero_timeout_raises_validation_error():
    with pytest.raises(ValidationError):
        LLMConfig(model="openai/gpt-4o", timeout=0)


def test_llm_config_given_zero_retry_base_delay_raises_validation_error():
    with pytest.raises(ValidationError):
        LLMConfig(model="openai/gpt-4o", retry_base_delay=0)


def test_llm_config_given_zero_max_concurrent_requests_raises_validation_error():
    with pytest.raises(ValidationError):
        LLMConfig(model="openai/gpt-4o", max_concurrent_requests=0)


def test_llm_config_given_max_delay_below_base_delay_raises_validation_error():
    with pytest.raises(ValidationError):
        LLMConfig(model="openai/gpt-4o", retry_base_delay=10.0, retry_max_delay=1.0)


def test_llm_config_given_max_delay_equal_to_base_delay_constructs():
    config = LLMConfig(model="openai/gpt-4o", retry_base_delay=5.0, retry_max_delay=5.0)

    assert config.retry_max_delay == 5.0


def test_logging_config_given_no_args_defaults_format():
    config = LoggingConfig()

    assert config.format == "text"


def test_logging_config_given_json_format_constructs():
    config = LoggingConfig(format="json")

    assert config.format == "json"


def test_logging_config_given_invalid_format_raises_validation_error():
    with pytest.raises(ValidationError):
        LoggingConfig(format="xml")


def test_app_config_given_no_max_sessions_defaults_to_unbounded():
    config = AppConfig(llms=[LLMConfig(model="openai/gpt-4o")])

    assert config.max_sessions is None


def test_app_config_given_max_sessions_constructs():
    config = AppConfig(llms=[LLMConfig(model="openai/gpt-4o")], max_sessions=500)

    assert config.max_sessions == 500


def test_app_config_given_zero_max_sessions_raises_validation_error():
    with pytest.raises(ValidationError):
        AppConfig(llms=[LLMConfig(model="openai/gpt-4o")], max_sessions=0)


def test_app_config_given_no_max_concurrent_requests_defaults_to_unbounded():
    config = AppConfig(llms=[LLMConfig(model="openai/gpt-4o")])

    assert config.max_concurrent_requests is None


def test_app_config_given_max_concurrent_requests_constructs():
    config = AppConfig(llms=[LLMConfig(model="openai/gpt-4o")], max_concurrent_requests=20)

    assert config.max_concurrent_requests == 20


def test_app_config_given_zero_max_concurrent_requests_raises_validation_error():
    with pytest.raises(ValidationError):
        AppConfig(llms=[LLMConfig(model="openai/gpt-4o")], max_concurrent_requests=0)


def test_app_config_given_unknown_field_raises_validation_error():
    with pytest.raises(ValidationError):
        AppConfig.model_validate_json('{"llms": [{"model": "openai/gpt-4o"}], "max_sesions": 100}')


def test_agent_config_given_no_ceilings_defaults_to_unrestricted():
    config = AgentConfig(
        name="researcher",
        system_prompt="You are a research assistant.",
        model="openai/gpt-4o",
        strategy="react",
    )

    assert config.allowed_tools is None
    assert config.allowed_models is None
    assert config.allowed_strategies is None
    assert config.max_request_seconds is None


def test_agent_config_given_allowed_tools_superset_of_tools_constructs():
    config = AgentConfig(
        name="researcher",
        system_prompt="You are a research assistant.",
        model="openai/gpt-4o",
        strategy="react",
        tools=["search"],
        allowed_tools=["search", "write"],
    )

    assert config.allowed_tools == ["search", "write"]


def test_agent_config_given_tools_outside_allowed_tools_raises_validation_error():
    with pytest.raises(ValidationError):
        AgentConfig(
            name="researcher",
            system_prompt="You are a research assistant.",
            model="openai/gpt-4o",
            strategy="react",
            tools=["search"],
            allowed_tools=["write"],
        )


def test_agent_config_given_empty_allowed_tools_and_empty_tools_constructs():
    config = AgentConfig(
        name="researcher",
        system_prompt="You are a research assistant.",
        model="openai/gpt-4o",
        strategy="react",
        allowed_tools=[],
    )

    assert config.allowed_tools == []


def test_agent_config_given_model_matching_allowed_models_constructs():
    config = AgentConfig(
        name="researcher",
        system_prompt="You are a research assistant.",
        model="openai/gpt-4o",
        strategy="react",
        allowed_models=["openai/gpt-4o", "openai/gpt-4o-mini"],
    )

    assert config.allowed_models == ["openai/gpt-4o", "openai/gpt-4o-mini"]


def test_agent_config_given_model_outside_allowed_models_raises_validation_error():
    with pytest.raises(ValidationError):
        AgentConfig(
            name="researcher",
            system_prompt="You are a research assistant.",
            model="openai/gpt-4o",
            strategy="react",
            allowed_models=["anthropic/claude-sonnet-5"],
        )


def test_agent_config_given_strategy_matching_allowed_strategies_constructs():
    config = AgentConfig(
        name="researcher",
        system_prompt="You are a research assistant.",
        model="openai/gpt-4o",
        strategy="react",
        allowed_strategies=["react"],
    )

    assert config.allowed_strategies == ["react"]


def test_agent_config_given_strategy_outside_allowed_strategies_raises_validation_error():
    with pytest.raises(ValidationError):
        AgentConfig(
            name="researcher",
            system_prompt="You are a research assistant.",
            model="openai/gpt-4o",
            strategy="react",
            allowed_strategies=["langgraph"],
        )


def test_agent_config_given_max_request_seconds_constructs():
    config = AgentConfig(
        name="researcher",
        system_prompt="You are a research assistant.",
        model="openai/gpt-4o",
        strategy="react",
        max_request_seconds=30.0,
    )

    assert config.max_request_seconds == 30.0


def test_agent_config_given_zero_max_request_seconds_raises_validation_error():
    with pytest.raises(ValidationError):
        AgentConfig(
            name="researcher",
            system_prompt="You are a research assistant.",
            model="openai/gpt-4o",
            strategy="react",
            max_request_seconds=0,
        )


def test_llm_config_given_unknown_field_raises_validation_error():
    with pytest.raises(ValidationError):
        LLMConfig.model_validate({"model": "openai/gpt-4o", "tmeperature": 0.5})


def test_agent_config_given_unknown_field_raises_validation_error():
    with pytest.raises(ValidationError):
        AgentConfig.model_validate(
            {
                "name": "researcher",
                "system_prompt": "You are a research assistant.",
                "model": "openai/gpt-4o",
                "strategy": "react",
                "tempreature": 0.5,
            }
        )


def test_tool_config_given_unknown_field_raises_validation_error():
    with pytest.raises(ValidationError):
        ToolConfig.model_validate({"name": "get_current_time", "nmae": "typo"})


def test_strategy_config_given_unknown_field_raises_validation_error():
    with pytest.raises(ValidationError):
        StrategyConfig.model_validate({"name": "react", "nmae": "typo"})


def test_compaction_config_given_unknown_field_raises_validation_error():
    with pytest.raises(ValidationError):
        CompactionConfig.model_validate(
            {"model": "anthropic/claude-3-5-haiku-20241022", "tokne_budget_pct": 0.5}
        )


def test_logging_config_given_unknown_field_raises_validation_error():
    with pytest.raises(ValidationError):
        LoggingConfig.model_validate({"levle": "DEBUG"})


def test_guardrail_config_given_only_name_and_validator_id_uses_defaults():
    guardrail = GuardrailConfig(name="no-pii", validator_id="guardrails/detect_pii")

    assert guardrail.validator_params == {}
    assert guardrail.action == "block"


def test_guardrail_config_rejects_extra_field():
    with pytest.raises(ValidationError):
        GuardrailConfig(name="no-pii", validator_id="guardrails/detect_pii", bogus="x")


def test_agent_config_guardrail_lists_default_to_empty():
    agent = AgentConfig(
        name="researcher",
        system_prompt="You are helpful.",
        model="openai/gpt-4o",
        strategy="react",
    )

    assert agent.input_guardrails == []
    assert agent.tool_output_guardrails == []
    assert agent.output_guardrails == []


def test_app_config_guardrails_defaults_to_empty_list():
    config = AppConfig.model_validate({"llms": [{"model": "openai/gpt-4o"}]})

    assert config.guardrails == []


def test_tracing_config_given_no_args_defaults_to_disabled():
    config = TracingConfig()

    assert config.console is False
    assert config.endpoint is None
    assert config.capture_content is False


def test_tracing_config_given_endpoint_constructs():
    config = TracingConfig(endpoint="http://localhost:4317")

    assert config.endpoint == "http://localhost:4317"


def test_tracing_config_given_unknown_field_raises_validation_error():
    with pytest.raises(ValidationError):
        TracingConfig.model_validate({"console": True, "bogus": "x"})


def test_app_config_given_no_tracing_defaults_to_disabled():
    config = AppConfig.model_validate({"llms": [{"model": "openai/gpt-4o"}]})

    assert config.tracing.console is False
    assert config.tracing.endpoint is None
