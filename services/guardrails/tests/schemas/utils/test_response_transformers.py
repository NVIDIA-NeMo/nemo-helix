# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Dict, List

import pytest
from nemoguardrails.logging.explain import LLMCallInfo
from nemoguardrails.rails.llm.options import ActivatedRail, ExecutedAction, GenerationLog, GenerationResponse
from nhx.guardrails.app.schemas.utils.response_transformers import (
    _extract_status_from_response,
    create_guardrail_chat_completion_response_from_generation_response,
    create_guardrail_completion_response_from_generation_response,
)
from nhx.guardrails.entities.enums import StatusEnum
from pytest import LogCaptureFixture
from pytest_mock import MockerFixture

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_llm_call(
    model_name: str = "default/main-model",
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
    total_tokens: int = 30,
) -> LLMCallInfo:
    return LLMCallInfo(
        llm_model_name=model_name,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def build_generation_rail(llm_calls: list[LLMCallInfo], stop: bool = False) -> ActivatedRail:
    return ActivatedRail(
        type="generation",
        name="generate user intent",
        stop=stop,
        executed_actions=[
            ExecutedAction(
                action_name="generate_user_intent",
                llm_calls=llm_calls,
            )
        ],
    )


def build_input_rail(stop: bool) -> ActivatedRail:
    return ActivatedRail(
        type="input",
        name="content safety check input",
        stop=stop,
        executed_actions=[],
    )


def build_output_rail(name: str = "content safety check output", stop: bool = True) -> ActivatedRail:
    return ActivatedRail(
        type="output",
        name=name,
        stop=stop,
        executed_actions=[],
    )


def build_response(
    content: str,
    activated_rails: list[ActivatedRail],
    role: str = "assistant",
    llm_calls: list[LLMCallInfo] | None = None,
    internal_events: list[dict] | None = None,
    colang_history: str | None = None,
) -> GenerationResponse:
    """Build a GenerationResponse with List[dict] response format (for chat completions)."""
    return GenerationResponse(
        response=[{"role": role, "content": content}],
        llm_output=None,
        output_data=None,
        log=GenerationLog(
            activated_rails=activated_rails,
            llm_calls=llm_calls,
            internal_events=internal_events,
            colang_history=colang_history,
        ),
    )


def build_completion_response(
    content: str,
    activated_rails: list[ActivatedRail],
    llm_calls: list[LLMCallInfo] | None = None,
    internal_events: list[dict] | None = None,
    colang_history: str | None = None,
) -> GenerationResponse:
    """Build a GenerationResponse with str response format (for completions)."""
    return GenerationResponse(
        response=content,
        llm_output=None,
        output_data=None,
        log=GenerationLog(
            activated_rails=activated_rails,
            llm_calls=llm_calls,
            internal_events=internal_events,
            colang_history=colang_history,
        ),
    )


def get_rail(name: str, stop: bool) -> ActivatedRail:
    return ActivatedRail(type="input", name=name, stop=stop, decisions=[], executed_actions=[])


class TestExtractStatusFromResponse:
    @pytest.mark.parametrize(
        "messages",
        [
            [],
            [{"role": "assistant", "content": "hello"}],
        ],
    )
    def test_no_exception_no_activated_rails_returns_success_and_empty_map(
        self, mocker: MockerFixture, messages: List[Dict[str, str]]
    ):
        response = mocker.Mock(spec=GenerationResponse)
        response.response = messages
        response.log = None

        overall_status, activated_rails_map = _extract_status_from_response(response)

        assert overall_status == StatusEnum.SUCCESS
        assert activated_rails_map == {}

    def test_blocked_with_no_activated_rails_but_exception_response_present(
        self, mocker: MockerFixture, caplog: LogCaptureFixture
    ):
        response = mocker.Mock(spec=GenerationResponse)
        response.response = [{"role": "exception", "content": {}}]
        log = mocker.Mock()
        log.activated_rails = []
        response.log = log

        caplog.set_level(logging.WARNING)

        expected_warning_message = "Unexpected state. No activated rails received for an exception"
        overall_status, activated_rails_map = _extract_status_from_response(response)
        assert overall_status == StatusEnum.BLOCKED
        assert activated_rails_map == {}
        assert any(expected_warning_message in rec.message for rec in caplog.records)

    @pytest.mark.parametrize(
        "activated_rails, expected_status, expected_map",
        [
            # Mixed rails: some blocked, some successful
            (
                [get_rail(name="R1", stop=False), get_rail(name="R2", stop=True), get_rail(name="R3", stop=True)],
                StatusEnum.BLOCKED,
                {"R1": StatusEnum.SUCCESS, "R2": StatusEnum.BLOCKED, "R3": StatusEnum.BLOCKED},
            ),
            # Single blocked rail
            (
                [get_rail(name="R3", stop=True)],
                StatusEnum.BLOCKED,
                {"R3": StatusEnum.BLOCKED},
            ),
            # All rails successful (no exception in response means SUCCESS)
            (
                [get_rail(name="R1", stop=False), get_rail(name="R2", stop=False)],
                StatusEnum.SUCCESS,
                {"R1": StatusEnum.SUCCESS, "R2": StatusEnum.SUCCESS},
            ),
            # Multiple blocked rails
            (
                [get_rail(name="R1", stop=True), get_rail(name="R2", stop=True), get_rail(name="R3", stop=True)],
                StatusEnum.BLOCKED,
                {"R1": StatusEnum.BLOCKED, "R2": StatusEnum.BLOCKED, "R3": StatusEnum.BLOCKED},
            ),
        ],
    )
    def test_activated_rails_status_mapping(
        self,
        mocker: MockerFixture,
        activated_rails: List[ActivatedRail],
        expected_status: StatusEnum,
        expected_map: Dict[str, StatusEnum],
    ):
        response = mocker.Mock(spec=GenerationResponse)
        # Set exception in response based on whether any rail has stop=True
        has_blocked = any(rail.stop for rail in activated_rails)
        response.response = [{"role": "exception", "content": {}}] if has_blocked else []

        log = mocker.Mock()
        log.activated_rails = activated_rails
        response.log = log

        overall_status, activated_rails_map = _extract_status_from_response(response)

        assert overall_status == expected_status
        assert activated_rails_map == expected_map

    def test_exception_with_no_stoppers_returns_blocked_with_success_map(self, mocker: MockerFixture):
        response = mocker.Mock(spec=GenerationResponse)
        response.response = [{"role": "exception", "content": {}}]

        log = mocker.Mock()
        log.activated_rails = [
            get_rail(name="R1", stop=False),
            get_rail(name="R2", stop=False),
            get_rail(name="R3", stop=False),
        ]
        response.log = log

        overall_status, activated_rails_map = _extract_status_from_response(response)

        # Overall status is BLOCKED due to exception, but all rails show UNKNOWN status
        assert overall_status == StatusEnum.BLOCKED
        assert activated_rails_map == {
            "R1": StatusEnum.UNKNOWN,
            "R2": StatusEnum.UNKNOWN,
            "R3": StatusEnum.UNKNOWN,
        }


# ---------------------------------------------------------------------------
# Tests for create_guardrail_chat_completion_response_from_generation_response
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCreateGuardrailChatCompletionResponse:
    """Unit tests for create_guardrail_chat_completion_response_from_generation_response."""

    def test_safe_request(self):
        """All rails pass: finish_reason=stop, model and usage come from generation rail."""
        llm_call = build_llm_call(model_name="ws/main-model", prompt_tokens=41, completion_tokens=47, total_tokens=88)
        response = build_response(
            content="Hello! How can I help?",
            activated_rails=[
                build_input_rail(stop=False),
                build_generation_rail(llm_calls=[llm_call], stop=False),
                build_output_rail(stop=False),
            ],
        )

        result = create_guardrail_chat_completion_response_from_generation_response(response, config_ids=["cfg-1"])

        assert len(result.choices) == 1
        assert result.choices[0].finish_reason == "stop"
        assert result.choices[0].message["content"] == "Hello! How can I help?"
        assert result.model == "ws/main-model"
        assert result.usage.prompt_tokens == 41
        assert result.usage.completion_tokens == 47
        assert result.usage.total_tokens == 88
        assert result.guardrails_data.config_ids == ["cfg-1"]

    def test_input_blocked(self):
        """Input rail stops: finish_reason=content_filter, model='-', usage=zeros."""
        response = build_response(
            content="I'm sorry, I can't respond to that.",
            activated_rails=[
                build_input_rail(stop=True),
            ],
        )

        result = create_guardrail_chat_completion_response_from_generation_response(response)

        assert len(result.choices) == 1
        assert result.choices[0].finish_reason == "content_filter"
        assert result.model == "-"
        assert result.usage.prompt_tokens == 0
        assert result.usage.completion_tokens == 0
        assert result.usage.total_tokens == 0

    def test_input_passed_output_blocked(self):
        """Output rail blocks: finish_reason=content_filter, model+usage from generation rail."""
        llm_call = build_llm_call(model_name="ws/main-model", prompt_tokens=84, completion_tokens=171, total_tokens=255)
        response = build_response(
            content="I'm sorry, I can't respond to that.",
            activated_rails=[
                build_input_rail(stop=False),
                build_generation_rail(llm_calls=[llm_call], stop=False),
                build_output_rail(name="content safety check output", stop=True),
            ],
        )

        result = create_guardrail_chat_completion_response_from_generation_response(response)

        assert result.choices[0].finish_reason == "content_filter"
        assert result.model == "ws/main-model"
        assert result.usage.prompt_tokens == 84
        assert result.usage.completion_tokens == 171
        assert result.usage.total_tokens == 255

    def test_output_blocked(self):
        """Output rail blocks (no input rail ran): finish_reason=content_filter, model+usage from generation rail."""
        llm_call = build_llm_call(model_name="ws/main-model", prompt_tokens=49, completion_tokens=200, total_tokens=249)
        response = build_response(
            content="I'm sorry, the desired output triggered rule(s) designed to mitigate exploitation of import_networking.",
            activated_rails=[
                build_generation_rail(llm_calls=[llm_call], stop=False),
                build_output_rail(name="injection detection", stop=True),
            ],
        )

        result = create_guardrail_chat_completion_response_from_generation_response(response)

        assert result.choices[0].finish_reason == "content_filter"
        assert result.model == "ws/main-model"
        assert result.usage.prompt_tokens == 49
        assert result.usage.completion_tokens == 200
        assert result.usage.total_tokens == 249

    def test_no_log_returns_safe_defaults(self):
        """response.log=None: no crash, finish_reason=stop, model='-', usage=zeros."""
        response = GenerationResponse(
            response=[{"role": "assistant", "content": "Hello!"}],
            llm_output=None,
            output_data=None,
            log=None,
        )

        result = create_guardrail_chat_completion_response_from_generation_response(response)

        assert result.choices[0].finish_reason == "stop"
        assert result.model == "-"
        assert result.usage.total_tokens == 0

    def test_generation_rail_with_no_llm_calls(self):
        """Generation rail present but no LLM calls: model='-', usage=zeros."""
        response = build_response(
            content="Hello!",
            activated_rails=[
                build_generation_rail(llm_calls=[], stop=False),
            ],
        )

        result = create_guardrail_chat_completion_response_from_generation_response(response)

        assert result.model == "-"
        assert result.usage.total_tokens == 0

    def test_multiple_response_messages_produces_multiple_choices(self):
        """Multiple messages in response.response produces a choice per message with correct index."""
        llm_call = build_llm_call()
        response = GenerationResponse(
            response=[
                {"role": "assistant", "content": "First"},
                {"role": "assistant", "content": "Second"},
            ],
            llm_output=None,
            output_data=None,
            log=GenerationLog(activated_rails=[build_generation_rail(llm_calls=[llm_call])]),
        )

        result = create_guardrail_chat_completion_response_from_generation_response(response)

        assert len(result.choices) == 2
        assert result.choices[0].index == 0
        assert result.choices[0].message["content"] == "First"
        assert result.choices[1].index == 1
        assert result.choices[1].message["content"] == "Second"

    def test_usage_is_summed_across_multiple_llm_calls_in_generation_rail(self):
        """If a generation rail has multiple LLM calls, usage is summed and model is from the last call."""
        call_1 = build_llm_call(model_name="ws/model-a", prompt_tokens=10, completion_tokens=20, total_tokens=30)
        call_2 = build_llm_call(model_name="ws/model-a", prompt_tokens=5, completion_tokens=15, total_tokens=20)
        response = build_response(
            content="Hello!",
            activated_rails=[
                build_generation_rail(llm_calls=[call_1, call_2], stop=False),
            ],
        )

        result = create_guardrail_chat_completion_response_from_generation_response(response)

        assert result.model == "ws/model-a"
        assert result.usage.prompt_tokens == 15
        assert result.usage.completion_tokens == 35
        assert result.usage.total_tokens == 50

    def test_config_ids_passed_through_to_guardrails_data(self):
        """Verifies config_ids parameter is reflected in guardrails_data."""
        response = build_response(
            content="Hello!",
            activated_rails=[build_generation_rail(llm_calls=[build_llm_call()])],
        )

        result = create_guardrail_chat_completion_response_from_generation_response(
            response, config_ids=["ws/my-config"]
        )

        assert result.guardrails_data.config_ids == ["ws/my-config"]

    def test_log_omitted_when_no_log_options(self):
        """Verifies guardrails_data.log is None when no log_options are requested."""
        response = build_response(
            content="Hello!",
            activated_rails=[build_generation_rail(llm_calls=[build_llm_call()])],
        )

        result = create_guardrail_chat_completion_response_from_generation_response(response)

        assert result.guardrails_data.log is None

    def test_log_present_when_log_options_provided(self):
        """Verifies guardrails_data.log is populated when any log_options are requested."""
        llm_call = build_llm_call()
        rails = [build_generation_rail(llm_calls=[llm_call])]
        response = build_response(
            content="Hello!",
            activated_rails=rails,
            llm_calls=[llm_call],
        )

        result = create_guardrail_chat_completion_response_from_generation_response(
            response, log_options={"llm_calls": True}
        )

        log = result.guardrails_data.log
        assert log is not None

        # Assert the `llm_calls` field is populated
        assert log.llm_calls is not None
        assert len(log.llm_calls) == 1
        assert log.llm_calls[0].llm_model_name == llm_call.llm_model_name
        assert log.llm_calls[0].prompt_tokens == llm_call.prompt_tokens
        assert log.llm_calls[0].completion_tokens == llm_call.completion_tokens
        assert log.llm_calls[0].total_tokens == llm_call.total_tokens

        # Assert the fields that were not requested are not present
        assert log.activated_rails == []
        assert log.internal_events is None
        assert log.colang_history is None

    def test_activated_rails_excluded_from_log_when_not_requested(self):
        """Verifies activated_rails is not included in the log when not explicitly requested,
        while other requested fields are still present."""
        llm_call = build_llm_call()
        rails = [build_generation_rail(llm_calls=[llm_call])]
        response = build_response(
            content="Hello!",
            activated_rails=rails,
            llm_calls=[llm_call],
            internal_events=[{"type": "UserMessage", "content": "Hello!"}],
            colang_history="user Hello!\nassistant How can I help?",
        )

        result = create_guardrail_chat_completion_response_from_generation_response(
            response, log_options={"llm_calls": True, "internal_events": True, "colang_history": True}
        )

        log = result.guardrails_data.log
        assert log is not None

        # Assert the requested log fields are populated
        assert log.llm_calls is not None
        assert len(log.llm_calls) == 1
        assert log.llm_calls[0].llm_model_name == llm_call.llm_model_name
        assert log.llm_calls[0].prompt_tokens == llm_call.prompt_tokens
        assert log.llm_calls[0].completion_tokens == llm_call.completion_tokens
        assert log.llm_calls[0].total_tokens == llm_call.total_tokens
        assert log.internal_events == [{"type": "UserMessage", "content": "Hello!"}]
        assert log.colang_history == "user Hello!\nassistant How can I help?"

        # Assert the fields that were not requested are not present
        assert log.activated_rails == []

    def test_activated_rails_included_in_log_when_requested(self):
        """Verifies activated_rails appears in guardrails_data.log when log_options includes it."""
        llm_call = build_llm_call()
        rails = [build_generation_rail(llm_calls=[llm_call])]
        response = build_response(
            content="Hello!",
            activated_rails=rails,
        )

        result = create_guardrail_chat_completion_response_from_generation_response(
            response, log_options={"activated_rails": True}
        )

        log = result.guardrails_data.log
        assert log is not None

        # Assert the `activated_rails` field is populated
        assert len(log.activated_rails) == 1
        assert log.activated_rails[0].name == "generate user intent"
        assert log.activated_rails[0].type == "generation"

        # Assert the field that were not requested are not present
        assert log.llm_calls is None
        assert log.internal_events is None
        assert log.colang_history is None


# ---------------------------------------------------------------------------
# Tests for create_guardrail_completion_response_from_generation_response
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCreateGuardrailCompletionResponse:
    """Unit tests for create_guardrail_completion_response_from_generation_response."""

    def test_safe_request(self):
        """All rails pass: finish_reason=stop, model and usage come from generation rail."""
        llm_call = build_llm_call(model_name="ws/main-model", prompt_tokens=41, completion_tokens=47, total_tokens=88)
        response = build_completion_response(
            content="Hello! How can I help?",
            activated_rails=[
                build_input_rail(stop=False),
                build_generation_rail(llm_calls=[llm_call], stop=False),
                build_output_rail(stop=False),
            ],
        )

        result = create_guardrail_completion_response_from_generation_response(response, config_ids=["cfg-1"])

        assert len(result.choices) == 1
        assert result.choices[0].finish_reason == "stop"
        assert result.choices[0].text == "Hello! How can I help?"
        assert result.model == "ws/main-model"
        assert result.usage.prompt_tokens == 41
        assert result.usage.completion_tokens == 47
        assert result.usage.total_tokens == 88
        assert result.guardrails_data.config_ids == ["cfg-1"]

    def test_input_blocked(self):
        """Input rail stops: finish_reason=content_filter, model='-', usage=zeros."""
        response = build_completion_response(
            content="I'm sorry, I can't respond to that.",
            activated_rails=[
                build_input_rail(stop=True),
            ],
        )

        result = create_guardrail_completion_response_from_generation_response(response)

        assert len(result.choices) == 1
        assert result.choices[0].finish_reason == "content_filter"
        assert result.choices[0].text == "I'm sorry, I can't respond to that."
        assert result.model == "-"
        assert result.usage.prompt_tokens == 0
        assert result.usage.completion_tokens == 0
        assert result.usage.total_tokens == 0

    def test_input_passed_output_blocked(self):
        """Output rail blocks: finish_reason=content_filter, model+usage from generation rail."""
        llm_call = build_llm_call(model_name="ws/main-model", prompt_tokens=84, completion_tokens=171, total_tokens=255)
        response = build_completion_response(
            content="I'm sorry, I can't respond to that.",
            activated_rails=[
                build_input_rail(stop=False),
                build_generation_rail(llm_calls=[llm_call], stop=False),
                build_output_rail(name="content safety check output", stop=True),
            ],
        )

        result = create_guardrail_completion_response_from_generation_response(response)

        assert result.choices[0].finish_reason == "content_filter"
        assert result.choices[0].text == "I'm sorry, I can't respond to that."
        assert result.model == "ws/main-model"
        assert result.usage.prompt_tokens == 84
        assert result.usage.completion_tokens == 171
        assert result.usage.total_tokens == 255

    def test_output_blocked(self):
        """Output rail blocks (no input rail ran): finish_reason=content_filter, model+usage from generation rail."""
        llm_call = build_llm_call(model_name="ws/main-model", prompt_tokens=49, completion_tokens=200, total_tokens=249)
        response = build_completion_response(
            content="I'm sorry, the desired output triggered rule(s) designed to mitigate exploitation of import_networking.",
            activated_rails=[
                build_generation_rail(llm_calls=[llm_call], stop=False),
                build_output_rail(name="injection detection", stop=True),
            ],
        )

        result = create_guardrail_completion_response_from_generation_response(response)

        assert result.choices[0].finish_reason == "content_filter"
        assert result.model == "ws/main-model"
        assert result.usage.prompt_tokens == 49
        assert result.usage.completion_tokens == 200
        assert result.usage.total_tokens == 249

    def test_no_log_returns_safe_defaults(self):
        """response.log=None: no crash, finish_reason=stop, model='-', usage=zeros."""
        response = GenerationResponse(
            response="Hello!",
            llm_output=None,
            output_data=None,
            log=None,
        )

        result = create_guardrail_completion_response_from_generation_response(response)

        assert result.choices[0].finish_reason == "stop"
        assert result.choices[0].text == "Hello!"
        assert result.model == "-"
        assert result.usage.total_tokens == 0

    def test_generation_rail_with_no_llm_calls(self):
        """Generation rail present but no LLM calls: model='-', usage=zeros."""
        response = build_completion_response(
            content="Hello!",
            activated_rails=[
                build_generation_rail(llm_calls=[], stop=False),
            ],
        )

        result = create_guardrail_completion_response_from_generation_response(response)

        assert result.model == "-"
        assert result.usage.total_tokens == 0

    def test_usage_is_summed_across_multiple_llm_calls_in_generation_rail(self):
        """If a generation rail has multiple LLM calls, usage is summed and model is from the last call."""
        call_1 = build_llm_call(model_name="ws/model-a", prompt_tokens=10, completion_tokens=20, total_tokens=30)
        call_2 = build_llm_call(model_name="ws/model-a", prompt_tokens=5, completion_tokens=15, total_tokens=20)
        response = build_completion_response(
            content="Hello!",
            activated_rails=[
                build_generation_rail(llm_calls=[call_1, call_2], stop=False),
            ],
        )

        result = create_guardrail_completion_response_from_generation_response(response)

        assert result.model == "ws/model-a"
        assert result.usage.prompt_tokens == 15
        assert result.usage.completion_tokens == 35
        assert result.usage.total_tokens == 50

    def test_config_ids_passed_through_to_guardrails_data(self):
        """Verifies config_ids parameter is reflected in guardrails_data."""
        response = build_completion_response(
            content="Hello!",
            activated_rails=[build_generation_rail(llm_calls=[build_llm_call()])],
        )

        result = create_guardrail_completion_response_from_generation_response(response, config_ids=["ws/my-config"])

        assert result.guardrails_data.config_ids == ["ws/my-config"]

    def test_log_omitted_when_no_log_options(self):
        """Verifies guardrails_data.log is None when no log_options are requested."""
        response = build_completion_response(
            content="Hello!",
            activated_rails=[build_generation_rail(llm_calls=[build_llm_call()])],
        )

        result = create_guardrail_completion_response_from_generation_response(response)

        assert result.guardrails_data.log is None

    def test_log_present_when_log_options_provided(self):
        """Verifies guardrails_data.log is populated when any log_options are requested."""
        llm_call = build_llm_call()
        rails = [build_generation_rail(llm_calls=[llm_call])]
        response = build_completion_response(
            content="Hello!",
            activated_rails=rails,
            llm_calls=[llm_call],
        )

        result = create_guardrail_completion_response_from_generation_response(
            response, log_options={"llm_calls": True}
        )

        log = result.guardrails_data.log
        assert log is not None

        assert log.llm_calls is not None
        assert len(log.llm_calls) == 1
        assert log.llm_calls[0].llm_model_name == llm_call.llm_model_name
        assert log.llm_calls[0].prompt_tokens == llm_call.prompt_tokens
        assert log.llm_calls[0].completion_tokens == llm_call.completion_tokens
        assert log.llm_calls[0].total_tokens == llm_call.total_tokens

        assert log.activated_rails == []
        assert log.internal_events is None
        assert log.colang_history is None

    def test_activated_rails_excluded_from_log_when_not_requested(self):
        """Verifies activated_rails is not included in the log when not explicitly requested."""
        llm_call = build_llm_call()
        rails = [build_generation_rail(llm_calls=[llm_call])]
        response = build_completion_response(
            content="Hello!",
            activated_rails=rails,
            llm_calls=[llm_call],
            internal_events=[{"type": "UserMessage", "content": "Hello!"}],
            colang_history="user Hello!\nassistant How can I help?",
        )

        result = create_guardrail_completion_response_from_generation_response(
            response, log_options={"llm_calls": True, "internal_events": True, "colang_history": True}
        )

        log = result.guardrails_data.log
        assert log is not None

        assert log.llm_calls is not None
        assert len(log.llm_calls) == 1
        assert log.internal_events == [{"type": "UserMessage", "content": "Hello!"}]
        assert log.colang_history == "user Hello!\nassistant How can I help?"

        assert log.activated_rails == []

    def test_activated_rails_included_in_log_when_requested(self):
        """Verifies activated_rails appears in guardrails_data.log when log_options includes it."""
        llm_call = build_llm_call()
        rails = [build_generation_rail(llm_calls=[llm_call])]
        response = build_completion_response(
            content="Hello!",
            activated_rails=rails,
        )

        result = create_guardrail_completion_response_from_generation_response(
            response, log_options={"activated_rails": True}
        )

        log = result.guardrails_data.log
        assert log is not None

        assert len(log.activated_rails) == 1
        assert log.activated_rails[0].name == "generate user intent"
        assert log.activated_rails[0].type == "generation"

        assert log.llm_calls is None
        assert log.internal_events is None
        assert log.colang_history is None

    def test_object_field_is_text_completion(self):
        """Verifies the response object field is set to 'text_completion'."""
        response = build_completion_response(
            content="Hello!",
            activated_rails=[build_generation_rail(llm_calls=[build_llm_call()])],
        )

        result = create_guardrail_completion_response_from_generation_response(response)

        assert result.object == "text_completion"

    def test_list_response_fallback_produces_multiple_choices(self):
        """Verifies that List[dict] response format still works as a fallback."""
        llm_call = build_llm_call()
        response = GenerationResponse(
            response=[
                {"role": "assistant", "content": "First"},
                {"role": "assistant", "content": "Second"},
            ],
            llm_output=None,
            output_data=None,
            log=GenerationLog(activated_rails=[build_generation_rail(llm_calls=[llm_call])]),
        )

        result = create_guardrail_completion_response_from_generation_response(response)

        assert len(result.choices) == 2
        assert result.choices[0].index == 0
        assert result.choices[0].text == "First"
        assert result.choices[1].index == 1
        assert result.choices[1].text == "Second"

    def test_list_response_with_missing_content_uses_empty_string(self):
        """Verifies that a List[dict] response message without content field uses empty string."""
        llm_call = build_llm_call()
        response = GenerationResponse(
            response=[{"role": "assistant"}],
            llm_output=None,
            output_data=None,
            log=GenerationLog(activated_rails=[build_generation_rail(llm_calls=[llm_call])]),
        )

        result = create_guardrail_completion_response_from_generation_response(response)

        assert len(result.choices) == 1
        assert result.choices[0].text == ""
