"""Integration test for the CLI entrypoint's argument parsing and wiring."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI

from agent.api.__main__ import main


def _write_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "app_config.json"
    config_path.write_text(json.dumps({"llms": [{"model": "openai/gpt-4o"}]}))
    return config_path


def test_main_given_config_arg_starts_uvicorn_with_default_host_and_port(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    mock_run = MagicMock()
    monkeypatch.setattr("agent.api.__main__.uvicorn.run", mock_run)
    config_path = _write_config(tmp_path)

    main(["--config", str(config_path)])

    mock_run.assert_called_once()
    args, kwargs = mock_run.call_args
    assert isinstance(args[0], FastAPI)
    assert kwargs["host"] == "127.0.0.1"
    assert kwargs["port"] == 8000


def test_main_given_config_arg_passes_uvicorn_log_config_for_consistent_formatting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    mock_run = MagicMock()
    monkeypatch.setattr("agent.api.__main__.uvicorn.run", mock_run)
    config_path = _write_config(tmp_path)

    main(["--config", str(config_path)])

    _, kwargs = mock_run.call_args
    log_config = kwargs["log_config"]
    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        assert log_config["loggers"][logger_name]["handlers"] == []
        assert log_config["loggers"][logger_name]["propagate"] is True


def test_main_given_host_and_port_args_overrides_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    mock_run = MagicMock()
    monkeypatch.setattr("agent.api.__main__.uvicorn.run", mock_run)
    config_path = _write_config(tmp_path)

    main(["--config", str(config_path), "--host", "0.0.0.0", "--port", "9000"])

    _, kwargs = mock_run.call_args
    assert kwargs["host"] == "0.0.0.0"
    assert kwargs["port"] == 9000


def test_main_given_invalid_config_prints_fatal_message_and_exits_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    config_path = tmp_path / "app_config.json"
    config_path.write_text("not json")

    with pytest.raises(SystemExit) as exc_info:
        main(["--config", str(config_path)])

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Fatal:" in captured.err
