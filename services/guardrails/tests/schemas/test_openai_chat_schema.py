# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nhx.guardrails.api.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    DeltaMessage,
    ImageURL,
    Role,
    UsageInfo,
)
from pydantic import ValidationError


@pytest.fixture
def chat_message():
    return ChatMessage(role=Role.user, content="Hello, world!")


@pytest.fixture
def usage_info():
    return UsageInfo(prompt_tokens=5, total_tokens=10, completion_tokens=15)


@pytest.fixture
def choices():
    return ChatCompletionResponseChoice(index=0, message={"content": "Hello, world!"})


# Test the ChatCompletionRequest class
def test_chat_completion_request(chat_message):
    # Test valid string message input
    request = ChatCompletionRequest(
        model="gpt-3.5",
        messages=[chat_message.model_dump(exclude_none=True)],
        stream=True,
    )
    # assert isinstance(request.messages[0], ChatMessage)
    assert request.messages[0].content == chat_message.content
    assert request.messages[0].role == Role.user
    assert request.streaming is True

    # Test valid list message input
    messages = [{"role": "user", "content": "Hello"}]
    request = ChatCompletionRequest(model="gpt-3.5", messages=messages)
    assert isinstance(request.messages, list)

    # Test invalid logprobs value
    # with pytest.raises(ValidationError):
    #     ChatCompletionRequest(model="gpt-3.5", messages="Test", logprobs=6)


# Test the ChatCompletionResponseChoice class
def test_chat_completion_response_choice():
    choice = ChatCompletionResponseChoice(index=1, message={"content": "hi"})
    assert choice.index == 1
    assert choice.message is not None
    assert choice.message.model_dump(exclude_none=True) == {"role": "assistant", "content": "hi"}
    assert choice.finish_reason is None


# Test the ChatCompletionResponse class
def test_chat_completion_response(usage_info, choices):
    response = ChatCompletionResponse(model="gpt-3.5", choices=[choices], usage=usage_info, message={"content": "hi"})
    assert response.id.startswith("chatcmpl-")
    assert isinstance(response.created, int)
    assert response.model == "gpt-3.5"


# Test the ChatCompletionResponseStreamChoice class
def test_chat_completion_response_stream_choice():
    choice = ChatCompletionResponseStreamChoice(index=1, delta=DeltaMessage(role="user", content="Hi"))
    assert choice.index == 1
    assert choice.delta.content == "Hi"


# Test the ChatCompletionStreamResponse class
def test_chat_completion_stream_response():
    choice = ChatCompletionResponseStreamChoice(index=1, delta=DeltaMessage(role="user", content="Hi"))
    stream_response = ChatCompletionStreamResponse(model="gpt-3.5", choices=[choice])
    assert stream_response.id.startswith("chatcmpl-")
    assert isinstance(stream_response.created, int)
    assert len(stream_response.choices) == 1


# Tests for ImageURL.validate_url
class TestImageURLValidator:
    def test_https_url_is_accepted(self):
        img = ImageURL(url="https://example.com/image.jpg")
        assert img.url == "https://example.com/image.jpg"

    def test_http_url_is_accepted(self):
        img = ImageURL(url="http://example.com/image.jpg")
        assert img.url == "http://example.com/image.jpg"

    def test_data_uri_is_accepted(self):
        img = ImageURL(url="data:image/jpeg;base64,/9j/4AAQSkZJRgAB")
        assert img.url.startswith("data:")

    def test_file_scheme_is_rejected(self):
        with pytest.raises(ValidationError, match="not supported"):
            ImageURL(url="file:///path/to/image.jpg")

    def test_unsupported_scheme_is_rejected(self):
        with pytest.raises(ValidationError, match="not supported"):
            ImageURL(url="fake://example.com/image.jpg")

    def test_empty_scheme_is_rejected(self):
        with pytest.raises(ValidationError, match="not supported"):
            ImageURL(url="just-a-plain-string")

    def test_http_url_without_hostname_is_rejected(self):
        with pytest.raises(ValidationError, match="valid hostname"):
            ImageURL(url="http://")

    def test_https_url_without_hostname_is_rejected(self):
        with pytest.raises(ValidationError, match="valid hostname"):
            ImageURL(url="https://")
