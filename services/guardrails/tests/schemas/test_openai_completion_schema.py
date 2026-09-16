# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nhx.guardrails.api.schemas import (
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    LogProbs,
    UsageInfo,
)
from pydantic import ValidationError


def test_completion_response_choice():
    choice = CompletionResponseChoice(index=0, text="Hello, world!")
    assert choice.index == 0
    assert choice.text == "Hello, world!"
    assert choice.logprobs is None
    assert choice.finish_reason is None

    # Test with optional fields
    logprobs = LogProbs(token_logprobs=[-0.5], top_logprobs=[{"test": -0.5}])
    finish_reason = "stop"
    choice_with_options = CompletionResponseChoice(index=1, text="Test", logprobs=logprobs, finish_reason=finish_reason)
    assert choice_with_options.logprobs == logprobs
    assert choice_with_options.finish_reason == finish_reason


def test_completion_response_stream_choice_inheritance():
    # Inherits without modification, test by initializing with an optional field
    stream_choice = CompletionResponseStreamChoice(index=1, text="Stream choice")
    assert stream_choice.index == 1
    assert stream_choice.text == "Stream choice"


def test_completion_stream_response():
    choice = CompletionResponseStreamChoice(index=0, text="Streaming response")
    response = CompletionStreamResponse(model="gpt-3", choices=[choice])
    assert response.id.startswith("cmpl-")
    assert response.object == "text_completion"
    assert isinstance(response.created, int)
    assert response.choices[0].text == "Streaming response"
    assert response.usage is None


def test_completion_response():
    choice = CompletionResponseChoice(index=0, text="Regular response")
    usage = UsageInfo(prompt_tokens=10)
    response = CompletionResponse(model="gpt-3", choices=[choice], usage=usage)
    assert response.usage == usage


def test_completion_request():
    # String prompt
    request = CompletionRequest(model="gpt-3", prompt="Hello, world!")
    assert request.prompt == "Hello, world!"

    # List of integers prompt
    prompt_ints = [12345]
    request = CompletionRequest(model="gpt-3", prompt=prompt_ints)
    assert request.prompt == prompt_ints

    # Invalid prompt: empty string
    with pytest.raises(ValidationError):
        CompletionRequest(model="gpt-3", prompt="")

    # Invalid: missing prompt (required field)
    with pytest.raises(ValidationError) as exc_info:
        CompletionRequest(model="gpt-3")
    assert "prompt" in str(exc_info.value)


class TestCompletionRequestMovedFields:
    """best_of, echo, and suffix are only valid on the completions endpoint."""

    def test_best_of_accepted(self):
        request = CompletionRequest(model="gpt-3", prompt="hi", best_of=3)
        assert request.best_of == 3

    def test_echo_accepted(self):
        request = CompletionRequest(model="gpt-3", prompt="hi", echo=True)
        assert request.echo is True

    def test_suffix_accepted(self):
        request = CompletionRequest(model="gpt-3", prompt="hi", suffix=" world")
        assert request.suffix == " world"

    def test_not_on_base_request(self):
        from nhx.guardrails.api.schemas import BaseRequest

        base = BaseRequest(model="m")
        assert not hasattr(base, "best_of")
        assert not hasattr(base, "echo")
        assert not hasattr(base, "suffix")
