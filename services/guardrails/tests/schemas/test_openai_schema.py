# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nhx.guardrails.api.schemas import BaseRequest, CompletionRequest
from pydantic import ValidationError


def test_model_validation():
    # Valid model name
    request = BaseRequest(model="text-davinci-003")
    assert request.model == "text-davinci-003"

    # Missing model name
    with pytest.raises(ValidationError):
        BaseRequest()


def test_max_tokens_validation():
    # Default max_tokens
    request = BaseRequest(model="text-davinci-003")
    assert request.max_tokens is None

    # Invalid max_tokens (below 1)
    with pytest.raises(ValidationError):
        BaseRequest(model="text-davinci-003", max_tokens=0)

    # Valid max_tokens
    request = BaseRequest(model="text-davinci-003", max_tokens=10)
    assert request.max_tokens == 10


def test_temperature_validation():
    # Default temperature
    request = BaseRequest(model="text-davinci-003")
    assert request.temperature is None

    # Temperature out of bounds
    with pytest.raises(ValidationError):
        BaseRequest(model="text-davinci-003", temperature=-0.1)
    with pytest.raises(ValidationError):
        BaseRequest(model="text-davinci-003", temperature=2.1)

    # Boundary temperature values
    request = BaseRequest(model="text-davinci-003", temperature=0)
    assert request.temperature == 0
    request = BaseRequest(model="text-davinci-003", temperature=2)
    assert request.temperature == 2


def test_top_p_validation():
    request = BaseRequest(model="gpt3-turbo-instruct", top_p=1)
    assert request.top_p == 1
    assert isinstance(request.top_p, float)

    # top_p out of bound
    with pytest.raises(ValidationError):
        BaseRequest(model="gpt3-turbo-instruct", top_p=2.1)


def test_frequency_penalty_validation():
    request = BaseRequest(model="gpt3-turbo-instruct", frequency_penalty=1.2)
    assert request.frequency_penalty == 1.2

    with pytest.raises(ValidationError):
        BaseRequest(model="gpt3-turbo-instruct", frequency_penalty=2.2)


def test_presence_penalty_validation():
    request = BaseRequest(model="gpt3-turbo-instruct", presence_penalty=1.2)
    assert request.presence_penalty == 1.2

    with pytest.raises(ValidationError):
        BaseRequest(model="gpt3-turbo-instruct", presence_penalty=2.2)


def test_top_logprobs_validation():
    request = BaseRequest(model="gpt3-turbo-instruct", top_logprobs=1)
    assert request.top_logprobs == 1

    with pytest.raises(ValidationError):
        BaseRequest(model="gpt3-turbo-instruct", top_logprobs=32)

    with pytest.raises(ValidationError):
        BaseRequest(model="gpt3-turbo-instruct", top_logprobs=-3)


def test_optional_fields_defaults():
    request = BaseRequest(model="text-davinci-003")
    assert request.streaming is False
    assert request.top_p is None
    assert request.temperature is None
    assert request.max_tokens is None
    assert request.max_completion_tokens is None
    assert request.n is None
    assert request.stop is None
    assert request.frequency_penalty is None
    assert request.presence_penalty is None
    assert request.function_call is None
    assert request.seed is None
    assert request.logit_bias is None
    assert request.top_logprobs is None
    assert request.logprobs is None
    assert request.tool_choice is None
    assert request.user is None
    assert request.tools is None
    assert request.ignore_eos is None
    assert request.reasoning_effort is None
    assert request.stream_options is None


def test_completion_request_optional_fields_defaults():
    request = CompletionRequest(model="text-davinci-003", prompt="hello")
    assert request.best_of is None
    assert request.echo is None
    assert request.suffix is None
