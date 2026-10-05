# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request
from nemoguardrails import LLMRails
from nhx.guardrails.app.handlers.completion import CompletionRequestHandler, ErrorData, process_chunk
from nhx.guardrails.entities.values._private import Model, RailsConfig
from nhx.guardrails.entities.values.chat import (
    GuardrailChatCompletionRequest,
    GuardrailChatCompletionResponse,
    GuardrailChatCompletionStreamResponse,
)
from nhx.guardrails.entities.values.completions import (
    GuardrailCompletionRequest,
)
from starlette.responses import StreamingResponse


@pytest.fixture
def mock_request():
    request = MagicMock(spec=Request)
    request.headers = {
        "X-Model-Authorization": "test_token",
        "x-custom-header": "custom_value",
    }
    request.state.request_id = "test_request_id"
    return request


@pytest.fixture
def mock_chat_request_body():
    return GuardrailChatCompletionRequest(model="test_model", prompt="test_prompt", options={}, state={}, messages=[])


@pytest.fixture
def mock_completion_request_body():
    return GuardrailCompletionRequest(model="test_model", prompt="test_prompt", options={}, state={})


@pytest.fixture
def handler(mock_request, mock_chat_request_body):
    handler = CompletionRequestHandler(
        rails_service=MagicMock(),
        request=mock_request,
        request_body=mock_chat_request_body,
        normal_response_model=GuardrailChatCompletionResponse,
        streaming_response_model=GuardrailChatCompletionStreamResponse,
        workspace="test-workspace",
    )
    handler.instantiate_llm_rails = AsyncMock()
    return handler


@pytest.mark.asyncio
async def test_handle_request_non_streaming(handler, mocker):
    handler.llm_rails = AsyncMock(spec=LLMRails)
    handler.llm_rails.config = AsyncMock(
        spec=RailsConfig, models=[Model(model="test_model", type="main", engine="nim")]
    )
    generate_result = MagicMock(
        llm_output={"choices": [{"message": {"role": "assistant", "content": "test_response"}}]},
        output_data={},
        log=None,
    )
    mock_run_generate = mocker.patch(
        "nhx.guardrails.app.handlers.completion.run_generate_async",
        new=AsyncMock(return_value=generate_result),
    )
    handler._post_process_response = MagicMock(return_value="processed_response")

    response = await handler.handle_request()

    assert response == "processed_response"
    mock_run_generate.assert_called_once()
    handler._post_process_response.assert_called_once()


@pytest.mark.asyncio
async def test_handle_request_streaming(handler):
    handler.llm_rails = AsyncMock(spec=LLMRails)
    handler.llm_rails.config = AsyncMock(
        spec=RailsConfig, models=[Model(model="test_model", type="main", engine="nim")]
    )
    handler.llm_rails.generate_async.return_value = MagicMock(
        llm_output={"choices": [{"message": {"role": "assistant", "content": "test_response"}}]},
        output_data={},
        log=None,
    )
    handler._post_process_response = MagicMock(return_value="processed_response")
    handler._handle_streaming = AsyncMock(return_value=StreamingResponse("streaming_response"))

    handler.body.stream = True
    response = await handler.handle_request()

    assert isinstance(response, StreamingResponse)
    handler._handle_streaming.assert_called_once()


# only valid error gets returned in data: {"error": {error details}}
# other chunks are returned as is
@pytest.mark.parametrize(
    "chunk, expected_output",
    [
        (
            '{"error":{"message":"Invalid input","type":"validation_error","param":"username","code":"1234"}}',
            '{"error":{"message":"Invalid input","type":"validation_error","param":"username","code":"1234"}}',
        ),
        ('{"error": "some_error"}', '{"error": "some_error"}'),
        ('{"key": "value"}', '{"key": "value"}'),
        ("invalid_json", "invalid_json"),
        ("", ""),
        ('{"nested": {"error": "some_error"}}', '{"nested": {"error": "some_error"}}'),
        ("76", "76"),
        ("true", "true"),
        ("false", "false"),
        ("null", "null"),
        ("[1, 2, 3]", "[1, 2, 3]"),
    ],
)
def test_process_chunk(chunk, expected_output):
    output = process_chunk(chunk)
    if isinstance(output, str):
        assert output == expected_output
        assert "username" not in chunk  # just to emphasize it is the right one
    else:
        # if the output is not a string, it should be an instance of ErrorData
        assert isinstance(output, ErrorData)
        assert "username" in chunk  # just to emphasize it is the right one

        assert output.model_dump_json() == chunk


@pytest.mark.asyncio
async def test_handle_streaming_with_exception(handler):
    """Test that exceptions during streaming that occur before a chunk is even generated are converted to error chunks."""

    # Mock LLMRails config
    handler.llm_rails = AsyncMock(spec=LLMRails)
    handler.llm_rails.config = AsyncMock(
        spec=RailsConfig, models=[Model(model="test_model", type="main", engine="nim")]
    )

    # Create a mock async generator that yields some chunks then raises an exception
    async def mock_stream_with_error():
        yield "First chunk"
        yield "Second chunk"
        raise Exception("Simulated streaming error")

    # Mock the _stream_async method to return our error-raising generator
    handler._stream_async = MagicMock(return_value=mock_stream_with_error())

    generator = await handler._handle_streaming(messages=[{"role": "user", "content": "test"}])

    # Collect all chunks from the generator
    chunks = []
    async for chunk in generator:
        chunks.append(chunk)

    # Verify we got chunks for the two successful yields
    assert len(chunks) >= 2, "Should have at least 2 normal chunks"

    # Verify the last chunk is an error chunk
    last_chunk = chunks[-1]
    assert "data: " in last_chunk

    # Extract and parse the error data
    error_json = last_chunk.replace("data: ", "").strip()
    import json

    error_data = json.loads(error_json)

    # Verify it's an error response
    assert "error" in error_data
    assert error_data["error"]["message"] == "Simulated streaming error"
    assert error_data["error"]["type"] == "Exception"
    assert error_data["error"]["code"] == "500"


@pytest.mark.asyncio
@patch("nhx.guardrails.app.handlers.completion.LLMRails")
@patch("nhx.guardrails.app.handlers.completion.set_main_model_into_context")
async def test_inline_config_sets_main_model_in_context(
    mock_set_main_model_into_context,
    _mock_llm_rails,
    mock_request,
):
    """
    Verify that inline configs set the main model into context.

    The main model is extracted at inference-time to determine the base URL for the model.
    This test ensures the model is set into context correctly when handling a /completion or /chat/completions request given an inline config.
    """
    inline_config = {
        "models": [{"type": "main", "engine": "nim", "model": "default/my-model"}],
    }
    request_body = GuardrailChatCompletionRequest(
        model="default/my-model",
        messages=[{"role": "user", "content": "test"}],
        guardrails={"config": inline_config},
    )

    handler = CompletionRequestHandler(
        rails_service=MagicMock(),
        request=mock_request,
        request_body=request_body,
        normal_response_model=GuardrailChatCompletionResponse,
        streaming_response_model=GuardrailChatCompletionStreamResponse,
        workspace="test-workspace",
    )
    handler._handle_non_streaming = AsyncMock(return_value=MagicMock())

    await handler.handle_request()

    mock_set_main_model_into_context.assert_called_once()
    called_model = mock_set_main_model_into_context.call_args[0][0]
    assert called_model.type == "main"
