# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, Mock, patch

import openai
import pytest
from nemo_evaluator_sdk.inference import (
    AddInferenceParameter,
    ClientInferenceError,
    InjectSystemMessage,
    LogHook,
    TransformReasoningOutput,
    deep_merge,
    make_inference_request,
    merge_default_headers,
    new_hooks,
    process_output,
    redact_request_for_logging,
    requests_log_var,
)
from nemo_evaluator_sdk.values.models import Model
from nemo_evaluator_sdk.values.params import RunConfigOnlineModel
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from pytest_mock import MockerFixture


class TestInferenceHelpers:
    def test_merge_default_headers_combines_model_and_request_headers(self):
        model = Model(
            url="https://judge.example.test/v1/chat/completions",
            name="judge-model",
            default_headers={"X-Trace-Id": "model", "X-Model": "judge"},
        )

        assert merge_default_headers(model, {"X-Trace-Id": "request", "X-Request": "abc"}) == {
            "X-Trace-Id": "request",
            "X-Model": "judge",
            "X-Request": "abc",
        }

    def test_merge_default_headers_returns_none_when_both_sources_are_empty(self):
        model = Model(
            url="https://judge.example.test/v1/chat/completions",
            name="judge-model",
        )

        assert merge_default_headers(model, None) is None

    def test_redact_request_for_logging_filters_only_auth_headers(self):
        assert redact_request_for_logging(
            {
                "model": "judge-model",
                "messages": [{"role": "user", "content": "hi"}],
                "extra_headers": {
                    "Authorization": "Bearer secret-token",
                    "X-Trace-Id": "trace-123",
                },
            }
        ) == {
            "model": "judge-model",
            "messages": [{"role": "user", "content": "hi"}],
            "extra_headers": {"X-Trace-Id": "trace-123"},
        }

    def test_redact_request_for_logging_omits_extra_headers_when_all_are_filtered(self):
        assert redact_request_for_logging(
            {
                "model": "judge-model",
                "messages": [{"role": "user", "content": "hi"}],
                "extra_headers": {
                    "Authorization": "Bearer secret-token",
                    "x-auth-token": "secret-token",
                },
            }
        ) == {
            "model": "judge-model",
            "messages": [{"role": "user", "content": "hi"}],
        }


def mock_chat_completion() -> ChatCompletion:
    return ChatCompletion(
        id="test-id",
        object="chat.completion",
        created=1234567890,
        model="test/model",
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content="test response"),
                finish_reason="stop",
            )
        ],
    )


def get_mock_args(mock_fn: AsyncMock) -> dict:
    call_args = mock_fn.call_args
    return call_args[1] if call_args[1] else call_args[0][0] if call_args[0] else {}


class TestMakeInferenceRequest:
    @staticmethod
    def _make_model(**kwargs) -> Model:
        return Model(
            url="https://judge.example.test/v1/chat/completions",
            name="judge-model",
            **kwargs,
        )

    @pytest.mark.asyncio
    async def test_uses_model_default_headers_as_request_extra_headers(self, mocker: MockerFixture):
        model = self._make_model(default_headers={"X-NHX-Principal-Id": "service:evaluator"})

        mock_chat = mocker.patch("nemo_evaluator_sdk.inference.AsyncOpenAI.chat")
        mock_chat.completions.create = AsyncMock(return_value=mock_chat_completion())

        await make_inference_request(model, {"messages": [{"role": "user", "content": "hi"}]})

        request_body = get_mock_args(mock_chat.completions.create)
        assert request_body["extra_headers"] == {"X-NHX-Principal-Id": "service:evaluator"}

    @pytest.mark.asyncio
    async def test_redacts_extra_headers_from_persisted_request_log(self, mocker: MockerFixture):
        model = self._make_model(default_headers={"X-Trace-Id": "trace-123"})

        mock_chat = mocker.patch("nemo_evaluator_sdk.inference.AsyncOpenAI.chat")
        mock_chat.completions.create = AsyncMock(return_value=mock_chat_completion())

        request_log: list[dict] = []
        requests_log_var.set(request_log)

        await make_inference_request(
            model,
            {"messages": [{"role": "user", "content": "hi"}]},
            default_headers={"Authorization": "Bearer secret-token"},
        )

        request_body = get_mock_args(mock_chat.completions.create)
        assert request_body["extra_headers"] == {
            "X-Trace-Id": "trace-123",
            "Authorization": "Bearer secret-token",
        }
        assert request_log[0]["request"]["extra_headers"] == {"X-Trace-Id": "trace-123"}

    @pytest.mark.asyncio
    async def test_filters_cookie_headers_from_persisted_request_log(self, mocker: MockerFixture):
        model = self._make_model()

        mock_chat = mocker.patch("nemo_evaluator_sdk.inference.AsyncOpenAI.chat")
        mock_chat.completions.create = AsyncMock(return_value=mock_chat_completion())

        request_log: list[dict] = []
        requests_log_var.set(request_log)

        await make_inference_request(
            model,
            {"messages": [{"role": "user", "content": "hi"}]},
            default_headers={"Cookie": "session=abc", "X-Trace-Id": "trace-123"},
        )

        request_body = get_mock_args(mock_chat.completions.create)
        assert request_body["extra_headers"] == {
            "Cookie": "session=abc",
            "X-Trace-Id": "trace-123",
        }
        assert request_log[0]["request"]["extra_headers"] == {"X-Trace-Id": "trace-123"}

    @pytest.mark.asyncio
    async def test_omits_logged_extra_headers_when_all_headers_are_auth_style(self, mocker: MockerFixture):
        model = self._make_model()

        mock_chat = mocker.patch("nemo_evaluator_sdk.inference.AsyncOpenAI.chat")
        mock_chat.completions.create = AsyncMock(return_value=mock_chat_completion())

        request_log: list[dict] = []
        requests_log_var.set(request_log)

        await make_inference_request(
            model,
            {"messages": [{"role": "user", "content": "hi"}]},
            default_headers={
                "Authorization": "Bearer secret-token",
                "x-auth-token": "secret-token",
                "Cookie": "session=abc",
                "Set-Cookie": "session=def",
            },
        )

        request_body = get_mock_args(mock_chat.completions.create)
        assert request_body["extra_headers"] == {
            "Authorization": "Bearer secret-token",
            "x-auth-token": "secret-token",
            "Cookie": "session=abc",
            "Set-Cookie": "session=def",
        }
        assert "extra_headers" not in request_log[0]["request"]

    @pytest.mark.asyncio
    async def test_per_call_headers_override_model_default_headers(self, mocker: MockerFixture):
        model = self._make_model(default_headers={"X-Trace-Id": "model", "X-Model": "judge"})

        mock_chat = mocker.patch("nemo_evaluator_sdk.inference.AsyncOpenAI.chat")
        mock_chat.completions.create = AsyncMock(return_value=mock_chat_completion())

        await make_inference_request(
            model,
            {"messages": [{"role": "user", "content": "hi"}]},
            default_headers={"X-Trace-Id": "request", "X-Request": "abc"},
        )

        request_body = get_mock_args(mock_chat.completions.create)
        assert request_body["extra_headers"] == {
            "X-Trace-Id": "request",
            "X-Model": "judge",
            "X-Request": "abc",
        }


def test_new_hooks_no_params():
    pre, post = new_hooks(None)
    assert len(pre) == 1, "at least log hook is initialized"
    assert isinstance(pre[0], LogHook), "at least log hook is initialized"
    assert len(post) == 1, "at least log hook is initialized"
    assert isinstance(post[0], LogHook), "log hook"
    assert pre[0] == post[0], "pre and post should have the same instance of log hook"


def test_new_hooks_offline_params():
    params = RunConfigOnlineModel()
    pre, post = new_hooks(params)

    assert len(pre) == 1, "at least log hook is initialized"
    assert isinstance(pre[0], LogHook), "at least log hook is initialized"
    assert len(post) == 1, "at least log hook is initialized"
    assert isinstance(post[0], LogHook), "log hook"
    assert pre[0] == post[0], "pre and post should have the same instance of log hook"


def test_add_inference_param():
    hook = AddInferenceParameter({"new": "value"})
    request = {"hello": "world"}
    request = hook.preprocess(request)
    assert request == {"hello": "world", "new": "value"}


def test_inject_system_message_chat_prefix():
    hook = InjectSystemMessage("'detailed thinking on'")
    request = {
        "messages": [
            {"role": "system", "content": "you are a helpful assistant"},
            {"role": "user", "content": "hello world"},
        ]
    }
    request = hook.preprocess(request)
    assert request["messages"][0] == {
        "role": "system",
        "content": "'detailed thinking on' you are a helpful assistant",
    }, "existing system message was not prefixed"
    assert request["messages"][1] == {"role": "user", "content": "hello world"}, "other messages should not be modified"


def test_inject_system_message_chat_prepend():
    hook = InjectSystemMessage("'detailed thinking on'")
    request = {
        "messages": [
            {"role": "user", "content": "hello world"},
        ]
    }
    request = hook.preprocess(request)
    assert request["messages"][0] == {
        "role": "system",
        "content": "'detailed thinking on'",
    }, "system messages was not prepended to list"
    assert request["messages"][1] == {"role": "user", "content": "hello world"}, "other messages should not be modified"


def test_inject_system_message_completions():
    hook = InjectSystemMessage("'detailed thinking on'")
    request = {"prompt": "hello world"}
    request = hook.preprocess(request)
    assert request["prompt"] == "'detailed thinking on' hello world"


def test_transform_reasoning_output():
    hook = TransformReasoningOutput("</think>")

    # Chat
    response = {
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "<think>reasoning content</think><think>second step\n</think>\nfinal output\n",
                },
            }
        ]
    }
    response = hook.postprocess(response)
    assert response["choices"][0]["message"] == {
        "role": "assistant",
        "content": "\nfinal output\n",
        "reasoning_content": "<think>reasoning content</think><think>second step\n</think>",
    }, "should only process last token with multiple end tokens are present"

    # Completions
    response = {"choices": [{"text": "<think>reasoning content</think><think>second step\n</think>\nfinal output\n"}]}
    response = hook.postprocess(response)
    assert response["choices"][0]["text"] == "\nfinal output\n"
    assert (
        response["choices"][0].get("reasoning_content")
        == "<think>reasoning content</think><think>second step\n</think>"
    )


def test_transform_reasoning_output_no_end_token():
    hook = TransformReasoningOutput("</think>")

    # Chat
    response = {
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "reasoning content without tags"}}]
    }
    response = hook.postprocess(response)
    assert response["choices"][0]["message"] == {
        "role": "assistant",
        "content": "reasoning content without tags",
    }, "no change if end token is not present"

    # Completions
    response = {"choices": [{"text": "reasoning content without tags"}]}
    response = hook.postprocess(response)
    assert response["choices"][0]["text"] == "reasoning content without tags"
    assert response["choices"][0].get("reasoning_content") is None


def test_transform_reasoning_output_only_end_token():
    hook = TransformReasoningOutput("</think>")

    # Chat
    response = {"choices": [{"index": 0, "message": {"role": "assistant", "content": "</think>"}}]}
    response = hook.postprocess(response)
    assert response["choices"][0]["message"] == {"role": "assistant", "content": "", "reasoning_content": "</think>"}

    # Completions
    response = {"choices": [{"text": "</think>"}]}
    response = hook.postprocess(response)
    assert response["choices"][0]["text"] == ""
    assert response["choices"][0]["reasoning_content"] == "</think>"


class TestProcessOutputNoneContent:
    """Test that process_output preserves non-text responses for downstream consumers."""

    def test_chat_content_none_is_returned(self):
        response = {"choices": [{"index": 0, "message": {"role": "assistant", "content": None}}]}
        assert process_output(response, hooks=[]) is None

    def test_completion_text_none_is_returned(self):
        response = {"choices": [{"text": None}]}
        assert process_output(response, hooks=[]) is None

    def test_chat_content_empty_string_returns_empty(self):
        """content="" is a valid (albeit empty) response, should be returned."""
        response = {"choices": [{"index": 0, "message": {"role": "assistant", "content": ""}}]}
        result = process_output(response, hooks=[])
        assert result == ""

    def test_completion_text_empty_string_returns_empty(self):
        """text="" is a valid (albeit empty) response, should be returned."""
        response = {"choices": [{"text": ""}]}
        result = process_output(response, hooks=[])
        assert result == ""


def test_process_output_raises_when_text_fields_are_missing():
    response = {"choices": [{"index": 0, "message": {"role": "assistant"}}]}
    with pytest.raises(ValueError, match="Invalid response format: No text found in the response"):
        process_output(response, hooks=[])


class TestMergeRequestParams:
    def test_add_inference_param_deep_merges_extra_body(self):
        hook = AddInferenceParameter(
            {
                "extra_body": {
                    "nvext": {
                        "max_thinking_tokens": 256,
                    }
                }
            }
        )
        request = {
            "extra_body": {
                "nvext": {
                    "guided_json": {"type": "object"},
                }
            }
        }

        request = hook.preprocess(request)

        assert request["extra_body"]["nvext"] == {
            "guided_json": {"type": "object"},
            "max_thinking_tokens": 256,
        }

    def test_merge_request_params_does_not_mutate_input(self):
        request = {"extra_body": {"nvext": {"max_thinking_tokens": 256}}}
        params = {"extra_body": {"nvext": {"guided_json": {"type": "object"}}}}

        merged = deep_merge(request, params)

        assert merged["extra_body"]["nvext"] == {
            "max_thinking_tokens": 256,
            "guided_json": {"type": "object"},
        }
        assert request == {"extra_body": {"nvext": {"max_thinking_tokens": 256}}}

    @pytest.mark.parametrize(
        ("base_request", "params", "expected"),
        [
            (
                {"extra_body": {"nvext": {"max_thinking_tokens": 256}}},
                {"extra_body": {"nvext": {"guided_json": {"type": "object"}}}},
                {"extra_body": {"nvext": {"max_thinking_tokens": 256, "guided_json": {"type": "object"}}}},
            ),
            (
                {"model": "test-model"},
                {"temperature": 0.2},
                {"model": "test-model", "temperature": 0.2},
            ),
            (
                {"temperature": 0.1, "top_p": 0.8},
                {"temperature": 0.2},
                {"temperature": 0.2, "top_p": 0.8},
            ),
            (
                {"extra_body": {"nvext": {"max_thinking_tokens": 256}}},
                {"extra_body": "disabled"},
                {"extra_body": "disabled"},
            ),
            (
                {"extra_body": "disabled"},
                {"extra_body": {"nvext": {"max_thinking_tokens": 256}}},
                {"extra_body": {"nvext": {"max_thinking_tokens": 256}}},
            ),
        ],
    )
    def test_merge_request_params(self, base_request, params, expected):
        merged = deep_merge(base_request, params)
        assert merged == expected


@pytest.mark.asyncio
async def test_make_inference_request_with_query_params():
    """
    Test that query parameters in the URL are properly extracted and passed to the OpenAI client
    as extra_query in the request body.
    """
    # Create a model with URL containing query parameters
    model = Model(
        url="https://prod.api.nvidia.com/llm/v1/azure/chat/completions?api-version=2024-12-01-pr",
        name="test/model",
    )

    # Mock response
    mock_chat_completion = ChatCompletion(
        id="test-id",
        object="chat.completion",
        created=1234567890,
        model="test/model",
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content="test response"),
                finish_reason="stop",
            )
        ],
    )

    with patch("nemo_evaluator_sdk.inference.AsyncOpenAI.chat") as mock_chat:
        mock_chat.completions.create = AsyncMock(return_value=mock_chat_completion)

        # Make the inference request
        request = {"messages": [{"role": "user", "content": "test"}]}
        await make_inference_request(model, request)

        # Verify the create method was called once
        assert mock_chat.completions.create.call_count == 1

        request_body = get_mock_args(mock_chat.completions.create)

        # Verify that extra_query was passed with the correct query parameters
        assert "extra_query" in request_body, "extra_query should be present in request body"
        assert request_body["extra_query"] == {"api-version": "2024-12-01-pr"}, (
            "extra_query should contain the query parameters from URL"
        )

        # Verify the base fields are also present
        assert request_body["model"] == "test/model"
        assert "messages" in request_body


@pytest.mark.asyncio
async def test_make_inference_request_without_query_params():
    """
    Test that when URL has no query parameters, extra_query is not added to the request body.
    """
    # Create a model with URL without query parameters
    model = Model(
        url="https://prod.api.nvidia.com/llm/v1/chat/completions",
        name="test/model",
    )

    # Mock response
    mock_chat_completion = ChatCompletion(
        id="test-id",
        object="chat.completion",
        created=1234567890,
        model="test/model",
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content="test response"),
                finish_reason="stop",
            )
        ],
    )

    with patch("nemo_evaluator_sdk.inference.AsyncOpenAI.chat") as mock_chat:
        mock_chat.completions.create = AsyncMock(return_value=mock_chat_completion)

        # Make the inference request
        request = {"messages": [{"role": "user", "content": "test"}]}
        await make_inference_request(model, request)

        # Verify the create method was called once
        assert mock_chat.completions.create.call_count == 1

        request_body = get_mock_args(mock_chat.completions.create)

        # Verify that extra_query was NOT added since there are no query parameters
        assert "extra_query" not in request_body, "extra_query should not be present when URL has no query parameters"

        # Verify the base fields are present
        assert request_body["model"] == "test/model"
        assert "messages" in request_body


@pytest.mark.asyncio
async def test_make_inference_request_bad_request_error_includes_context():
    """
    Test that BadRequestError (non-guided_json) includes base_url and model_id in the error message.
    """
    # Create a model
    model = Model(
        url="https://api.example.com/v1/chat/completions",
        name="test/model",
    )

    # Mock a BadRequestError with a proper response object
    mock_response = Mock()
    mock_response.status_code = 400
    mock_response.text = "Invalid request format"
    bad_request_error = openai.BadRequestError(
        message="Bad request",
        response=mock_response,
        body=None,
    )

    with patch("nemo_evaluator_sdk.inference.AsyncOpenAI.chat") as mock_chat:
        mock_chat.completions.create = AsyncMock(side_effect=bad_request_error)

        # Make the inference request and expect ClientInferenceError
        request = {"messages": [{"role": "user", "content": "test"}]}

        with pytest.raises(ClientInferenceError) as exc_info:
            await make_inference_request(model, request)

        # Verify the error message includes base_url and model_id
        error_message = str(exc_info.value)
        assert "base_url: https://api.example.com/v1" in error_message, "Error message should include base_url"
        assert "model_id: test/model" in error_message, "Error message should include model_id"
        assert "400" in error_message, "Error message should include status code"
        # Verify status_code is exposed as an attribute for callers that need to discriminate
        # (e.g. evaluator prechecks distinguishing 404 -> "no active inference deployment").
        assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_make_inference_request_api_status_error_includes_context():
    """
    Test that APIStatusError (like 404) includes base_url and model_id in the error message.
    """
    # Create a model with different URL to avoid any caching issues
    model = Model(
        url="https://api.notfound.com/v1/chat/completions",
        name="test/model-404",
    )

    # Mock a 404 error with a proper response object
    mock_response = Mock()
    mock_response.status_code = 404
    mock_response.text = "Model not found"
    api_status_error = openai.NotFoundError(
        message="Not found",
        response=mock_response,
        body=None,
    )

    with patch("nemo_evaluator_sdk.inference.AsyncOpenAI.chat") as mock_chat:
        mock_chat.completions.create = AsyncMock(side_effect=api_status_error)

        # Make the inference request and expect ClientInferenceError
        request = {"messages": [{"role": "user", "content": "test"}]}

        with pytest.raises(ClientInferenceError) as exc_info:
            await make_inference_request(model, request)

        # Verify the error message includes base_url and model_id (the main fix)
        error_message = str(exc_info.value)
        assert "base_url: https://api.notfound.com/v1" in error_message, "Error message should include base_url"
        assert "model_id: test/model-404" in error_message, "Error message should include model_id"
        # The status code should be present in the error message
        assert "error occurred" in error_message, "Error message should indicate an error occurred"
        # 404 status_code attribute lets evaluator prechecks render
        # "no active inference deployment" instead of the raw inference error.
        assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_make_inference_request_error_with_empty_details():
    """
    Test that when error details are empty, 'Details:' is not included in the error message.
    """
    # Create a model
    model = Model(
        url="https://api.empty.com/v1/chat/completions",
        name="test/model-empty",
    )

    # Mock a 404 error with empty response text
    mock_response = Mock()
    mock_response.status_code = 404
    mock_response.text = ""  # Empty details
    api_status_error = openai.NotFoundError(
        message="Not found",
        response=mock_response,
        body=None,
    )

    with patch("nemo_evaluator_sdk.inference.AsyncOpenAI.chat") as mock_chat:
        mock_chat.completions.create = AsyncMock(side_effect=api_status_error)

        # Make the inference request and expect ClientInferenceError
        request = {"messages": [{"role": "user", "content": "test"}]}

        with pytest.raises(ClientInferenceError) as exc_info:
            await make_inference_request(model, request)

        # Verify the error message does NOT end with "Details: " when details are empty
        error_message = str(exc_info.value)
        assert "base_url: https://api.empty.com/v1" in error_message, "Error message should include base_url"
        assert "model_id: test/model-empty" in error_message, "Error message should include model_id"
        assert not error_message.endswith("Details: "), (
            "Error message should not end with 'Details: ' when details are empty"
        )
        assert not error_message.endswith("Details:"), (
            "Error message should not end with 'Details:' when details are empty"
        )
