# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for choosing between the LLMRails and IORails engines."""

from typing import Any
from unittest.mock import MagicMock

import pytest
from nemoguardrails import Guardrails, RailsConfig
from nhx.guardrails.app.constants import NIM_CHAT, NIM_LLM
from nhx.guardrails.app.services.rails.engine import IORAILS_ENGINE_ENV, create_rails_engine, iorails_enabled

_SAFETY_PROMPT = {
    "task": "content_safety_check_input $model=content_safety",
    "content": "Is this safe? {{ user_input }}",
    "output_parser": "nemoguard_parse_prompt_safety",
    "max_tokens": 50,
}


def _config(engine: str = NIM_CHAT, rails: dict[str, Any] | None = None) -> RailsConfig:
    return RailsConfig.from_content(
        config={
            "models": [
                {
                    "type": "content_safety",
                    "engine": engine,
                    "model": "safety-judge",
                    "parameters": {"base_url": "http://models.invalid/v1"},
                }
            ],
            "rails": rails or {"input": {"flows": ["content safety check input $model=content_safety"]}},
            "prompts": [_SAFETY_PROMPT],
        }
    )


@pytest.fixture
def llmrails_class(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Stand-in for LLMRails: building the real one needs request-scoped credentials."""
    mock = MagicMock(name="LLMRails")
    monkeypatch.setattr("nhx.guardrails.app.services.rails.engine.LLMRails", mock)
    return mock


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1", True), ("TRUE", True), (" Yes ", True), ("", False), ("0", False), ("off", False)],
)
def test_iorails_switch_parsing(value: str, expected: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(IORAILS_ENGINE_ENV, value)

    assert iorails_enabled() is expected


def test_llmrails_is_the_default(monkeypatch: pytest.MonkeyPatch, llmrails_class: MagicMock) -> None:
    monkeypatch.delenv(IORAILS_ENGINE_ENV, raising=False)
    config = _config()

    assert create_rails_engine(config) is llmrails_class.return_value
    llmrails_class.assert_called_once_with(config)


def test_iorails_used_for_supported_config(monkeypatch: pytest.MonkeyPatch, llmrails_class: MagicMock) -> None:
    monkeypatch.setenv(IORAILS_ENGINE_ENV, "true")

    engine = create_rails_engine(_config())

    assert isinstance(engine, Guardrails)
    assert engine.use_iorails_engine
    llmrails_class.assert_not_called()


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(
            _config(
                rails={
                    "input": {"flows": ["content safety check input $model=content_safety"]},
                    "dialog": {"single_call": {"enabled": True}},
                }
            ),
            id="dialog-rails",
        ),
        pytest.param(_config(engine=NIM_LLM), id="text-completions-model"),
    ],
)
def test_unsupported_config_stays_on_llmrails(
    config: RailsConfig,
    monkeypatch: pytest.MonkeyPatch,
    llmrails_class: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv(IORAILS_ENGINE_ENV, "true")

    assert create_rails_engine(config) is llmrails_class.return_value
    assert "stays on LLMRails" in caplog.text
