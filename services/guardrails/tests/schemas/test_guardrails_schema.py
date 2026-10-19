# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
from nemoguardrails.rails.llm.options import (
    ActivatedRail,
    GenerationLog,
    GenerationResponse,
    GenerationStats,
    LLMCallInfo,
)
from nhx.guardrails.app.schemas.guardrails import (
    GuardrailsChatCompletionRequest,
    GuardrailsCompletionRequest,
)
from nhx.guardrails.app.schemas.utils.response_transformers import (
    create_guardrail_chat_completion_response_from_generation_response,
)
from nhx.guardrails.entities.values._private import (
    GenerationLog as NHXGenerationLog,
)
from nhx.guardrails.entities.values.common import (
    GuardrailsDataInput,
    GuardrailsDataOutput,
)

DEFAULT_CONFIG = os.getenv("DEFAULT_CONFIG_ID", "system/default")


def test_guardrails_request_config_id():
    guardrails_request = GuardrailsDataInput(config_id="config123")

    print(guardrails_request)
    assert guardrails_request.config_id == "config123"
    assert guardrails_request.config_ids == ["config123"]


def test_guardrails_request_config_ids():
    guardrails_request = GuardrailsDataInput(config_ids=["config123", "config456"])
    assert guardrails_request.config_ids == ["config123", "config456"]


def test_guardrails_request_default_config_id():
    guardrails_request = GuardrailsDataInput()
    assert guardrails_request.config_id == DEFAULT_CONFIG
    assert guardrails_request.config_ids == [DEFAULT_CONFIG]


def test_guardrails_request_only_config_ids():
    guardrails_request = GuardrailsDataInput(config_ids=["config123", "config456"])
    assert guardrails_request.config_ids == ["config123", "config456"]
    assert guardrails_request.config_id is None


def test_guardrails_request_config_as_config_id():
    with pytest.warns(UserWarning, match="No config_id or config_ids provided"):
        guardrails_request = GuardrailsDataInput(config="config1")
    assert guardrails_request.config_id == "config1"
    assert guardrails_request.config_ids == ["config1"]


def test_guardrails_request_config_and_config_ids():
    with pytest.warns(UserWarning, match="No config_id or config_ids provided"):
        with pytest.raises(ValueError):
            GuardrailsDataInput(config="config1", config_id="config2")


def test_guardrails_request_both_config_id_and_config_ids():
    with pytest.raises(ValueError):
        GuardrailsDataInput(config_id="config123", config_ids=["config456", "config789"])


def test_guardrails_request_ensure_config_ids():
    guardrails_request = GuardrailsDataInput(config_id="config123")
    assert guardrails_request.config_ids == ["config123"]


def test_guardrails_chat_completion_request():
    request = GuardrailsChatCompletionRequest(config_id="config123", messages="Hello, world!")
    assert request.config_id == "config123"
    assert request.messages == "Hello, world!"

    request = GuardrailsChatCompletionRequest(
        config_id="config123",
        messages=[{"user": "Hello, world!"}, {"bot": "Hi, user!"}],
    )
    assert request.config_id == "config123"
    assert request.messages == [{"user": "Hello, world!"}, {"bot": "Hi, user!"}]


def test_guardrails_completion_request():
    request = GuardrailsCompletionRequest(config_id="config123", prompt="Hello, world!")
    assert request.config_id == "config123"
    assert request.prompt == "Hello, world!"

    request = GuardrailsCompletionRequest(config_id="config123", prompt=["Hello, world!", "Hi, user!"])
    assert request.config_id == "config123"
    assert request.prompt == ["Hello, world!", "Hi, user!"]


def test_guardrails_data():
    data = GuardrailsDataOutput(
        llm_output={"output": "Hello, world!"},
        output_data={"data": "Hi, user!"},
        log=NHXGenerationLog(),
    )
    assert data.llm_output == {"output": "Hello, world!"}
    assert data.output_data == {"data": "Hi, user!"}


def test_nhx_ngm_generation_log():
    log1 = GenerationLog()
    log2 = NHXGenerationLog()

    assert log1.__dict__.keys() == log2.__dict__.keys()

    log1_fields = log1.model_fields
    log2_fields = log2.model_fields

    assert log1_fields.keys() == log2_fields.keys()


# Test function for the model_dump issue
def test_basemodel_log_conversion():
    """
    Test that BaseModel logs are properly converted to dictionaries.

    This test verifies the fix for the log field not being populated
    when it's a GenerationLog instance.
    """
    # Create a GenerationLog instance
    log_model = GenerationLog(
        activated_rails=[
            ActivatedRail(
                name="test_rail",
                type="input",
                action="modify",
            )
        ],
        stats=GenerationStats(total_time=0.5, llm_time=0.3),
        llm_calls=[
            LLMCallInfo(
                id="test_call_123",
                prompt="This is a test prompt",
                completion="This is a test completion",
                raw_response={"id": "xyz", "choices": [{"text": "sample"}]},
                llm_model_name="test-model",
                start_time=1686000000,
                end_time=1686000001,
                time_taken=1.0,
            )
        ],
        internal_events=[{"event_type": "test_event"}],
        colang_history="user: hello\nassistant: hi",
    )

    # Create a sample generation response with a BaseModel log
    generation_response = GenerationResponse(
        response=[{"role": "assistant", "content": "This is a test response"}],
        llm_output=None,
        log=log_model,  # Using GenerationLog as log
        output_data={"some_key": "some_value"},
    )

    # Config IDs for the test
    config_ids = ["test_config_1", "test_config_2"]

    # Before the fix, this would fail to include the log data
    # After the fix, the log should be properly converted to a dictionary
    log_options = {"activated_rails": True, "llm_calls": True, "internal_events": True, "colang_history": True}
    result = create_guardrail_chat_completion_response_from_generation_response(
        generation_response, config_ids, log_options
    )

    assert result.guardrails_data is not None
    assert result.guardrails_data.log is not None

    assert len(result.guardrails_data.log.activated_rails) == 1
    assert result.guardrails_data.log.activated_rails[0].name == "test_rail"
    assert len(result.guardrails_data.log.llm_calls) == 1
    assert result.guardrails_data.log.llm_calls[0].prompt == "This is a test prompt"
    assert result.guardrails_data.log.colang_history == "user: hello\nassistant: hi"


def test_dict_log_conversion():
    """
    Test that dictionary logs are properly preserved.

    This ensures the function continues to work with
    dictionary logs as before.
    """
    # Create a dictionary log that mimics the structure of GenerationLog
    log_dict = {
        "activated_rails": [
            {
                "name": "test_rail",
                "type": "input",
                "action": "modify",
            }
        ],
        "stats": {"total_time": 0.5, "llm_time": 0.3},
        "llm_calls": [
            {
                "id": "test_call_123",
                "prompt": "This is a test prompt",
                "completion": "This is a test completion",
                "raw_response": {"id": "xyz", "choices": [{"text": "sample"}]},
                "llm_model_name": "test-model",
                "start_time": 1686000000,
                "end_time": 1686000001,
                "time_taken": 1.0,
            }
        ],
        "internal_events": [{"event_type": "test_event"}],
        "colang_history": "user: hello\nassistant: hi",
    }

    # Create a sample generation response with a dictionary log
    generation_response = GenerationResponse(
        response=[{"role": "assistant", "content": "This is a test response"}],
        llm_output=None,
        log=log_dict,  # Using dict as log
        output_data={"some_key": "some_value"},
    )

    config_ids = ["test_config_1", "test_config_2"]

    log_options = {"activated_rails": True, "llm_calls": True, "internal_events": True, "colang_history": True}
    result = create_guardrail_chat_completion_response_from_generation_response(
        generation_response, config_ids, log_options
    )

    assert result.guardrails_data is not None
    assert result.guardrails_data.log is not None

    assert len(result.guardrails_data.log.activated_rails) == 1
    assert result.guardrails_data.log.activated_rails[0].name == "test_rail"
    assert len(result.guardrails_data.log.llm_calls) == 1
    assert result.guardrails_data.log.llm_calls[0].prompt == "This is a test prompt"
    assert result.guardrails_data.log.colang_history == "user: hello\nassistant: hi"
