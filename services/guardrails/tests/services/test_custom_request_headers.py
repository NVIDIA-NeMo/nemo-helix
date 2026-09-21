# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for custom request headers handling."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage
from nmp.guardrails.app.handlers.checks import CheckRequestHandler
from nmp.guardrails.app.handlers.completion import CompletionRequestHandler
from nmp.guardrails.app.llms.chat.nim import ChatNIM
from nmp.guardrails.app.llms.completion.nim import NIM
from nmp.guardrails.app.utils.context_utils import (
    get_request_default_headers_from_context,
    set_request_default_headers_into_context,
)
from nmp.guardrails.entities.values.chat import GuardrailChatCompletionRequest
from nmp.guardrails.entities.values.check import GuardrailCheckRequest
from nmp.guardrails.entities.values.common import GuardrailsDataInput


def create_mock_config():
    """Create a mock Rails config for testing."""
    mock_config = MagicMock()
    mock_config.colang_version = "1.0"
    mock_config.passthrough = False
    mock_config.rails.dialog.single_call.enabled = False
    mock_config.flows = []
    mock_config.rails.input.flows = []
    mock_config.rails.output.flows = []
    mock_config.rails.retrieval.flows = []
    return mock_config


@pytest.fixture
def mock_openai_clients():
    """Fixture to mock OpenAI sync and async clients."""
    with patch("openai.OpenAI") as mock_openai_class, patch("openai.AsyncOpenAI") as mock_async_openai_class:
        # Sync client
        mock_openai_client = MagicMock()
        mock_openai_class.return_value = mock_openai_client
        mock_chat_completions = MagicMock()
        mock_openai_client.chat.completions.create = mock_chat_completions
        mock_openai_client.chat.completions.with_raw_response.create = mock_chat_completions

        # Async client
        mock_async_openai_client = MagicMock()
        mock_async_openai_class.return_value = mock_async_openai_client
        mock_async_chat_completions = AsyncMock()
        mock_async_openai_client.chat.completions.create = mock_async_chat_completions
        mock_async_openai_client.chat.completions.with_raw_response.create = mock_async_chat_completions

        yield {
            "sync_completions": mock_chat_completions,
            "async_completions": mock_async_chat_completions,
        }


@pytest.fixture
def mock_chat_dependencies():
    """Fixture to mock common dependencies for ChatNIM tests."""
    with (
        patch(
            "nmp.guardrails.app.utils.context_utils.get_request_default_headers_from_context",
            return_value=None,
        ) as mock_get_headers,
        patch(
            "nmp.guardrails.app.services.configs.registry.ConfigRegistry.get",
            return_value=create_mock_config(),
        ),
        patch(
            "nmp.guardrails.app.handlers.utils.get_model_config_object",
            return_value=MagicMock(model="test-model", engine="nim"),
        ),
        patch(
            "nmp.guardrails.app.llms.utils.get_x_model_auth_token_from_context",
            return_value="test-auth-token",
        ),
        patch(
            "nmp.guardrails.app.llms.utils.get_main_model_from_context",
            return_value=None,
        ),
    ):
        yield mock_get_headers


@pytest.fixture
def mock_llm_dependencies():
    """Fixture to mock dependencies for NIM LLM tests."""
    with (
        patch(
            "nmp.guardrails.app.utils.context_utils.get_request_default_headers_from_context",
            return_value=None,
        ) as mock_get_headers,
        patch(
            "nmp.guardrails.app.llms.utils.get_main_model_from_context",
            return_value=None,
        ),
        patch(
            "nmp.guardrails.app.llms.utils.get_x_model_auth_token_from_context",
            return_value="test-auth-token",
        ),
    ):
        yield mock_get_headers


class TestChatCustomHeaders:
    """Tests for ChatNIM custom headers handling."""

    def test_custom_headers_passed_to_openai_client(self, mock_openai_clients, mock_chat_dependencies):
        """Test that custom headers are passed to the OpenAI client."""
        custom_headers = {
            "X-Custom-Header1": "Value1",
            "X-Custom-Header2": "Value2",
            "x-custom-header3": "value3",
        }
        set_request_default_headers_into_context(custom_headers)
        mock_chat_dependencies.return_value = custom_headers

        chat_nim = ChatNIM(model="test-model")
        chat_nim.include_response_headers = False

        mock_openai_clients["sync_completions"].return_value = {
            "choices": [{"message": {"role": "assistant", "content": "Test response"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        messages = [HumanMessage(content="Hello, world!")]
        chat_nim._generate(messages=messages)

        mock_openai_clients["sync_completions"].assert_called_once()
        call_kwargs = mock_openai_clients["sync_completions"].call_args.kwargs
        assert "extra_headers" in call_kwargs
        assert call_kwargs["extra_headers"] == custom_headers

    def test_custom_headers_extracted_from_request(self, mock_openai_clients, mock_chat_dependencies):
        """Test that custom headers are extracted from request and set in context."""
        custom_headers = {
            "X-Custom-Header1": "Value1",
            "X-Custom-Header2": "Value2",
            "x-custom-header3": "value3",
            "X-Model-Authorization": "auth_token",
            "Content-Type": "application/json",
        }
        request = MagicMock()
        request.headers = custom_headers

        # Create a valid request body type
        request_body = GuardrailChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
        )

        request_handler = CompletionRequestHandler(
            rails_service=MagicMock(),
            request=request,
            request_body=request_body,
            normal_response_model=MagicMock(),
            streaming_response_model=MagicMock(),
            workspace="test-workspace",
        )

        extracted_headers = request_handler._get_custom_headers()
        expected_custom_headers = {
            "X-Custom-Header1": "Value1",
            "X-Custom-Header2": "Value2",
            "x-custom-header3": "value3",
        }

        assert extracted_headers == expected_custom_headers

        # set_custom_headers merges in service principal headers on top of custom headers.
        # No auth context is active in this test, so only X-NMP-Principal-Id is added
        # (X-NMP-Principal-On-Behalf-Of is omitted when there is no user in context).
        request_handler.set_custom_headers()
        context_headers = get_request_default_headers_from_context()
        assert context_headers == {
            **expected_custom_headers,
            "X-NMP-Principal-Id": "service:guardrails",
            "X-NMP-Actor-Aliases": "service:guardrails",
        }

    @pytest.mark.asyncio
    async def test_custom_headers_in_async_generate(self, mock_openai_clients, mock_chat_dependencies):
        """Test that custom headers are passed in async generate."""
        custom_headers = {
            "X-Custom-Header1": "Value1",
            "X-Custom-Header2": "Value2",
        }
        set_request_default_headers_into_context(custom_headers)
        mock_chat_dependencies.return_value = custom_headers

        chat_nim = ChatNIM(model="test-model")
        chat_nim.include_response_headers = False

        async def mock_create(*args, **kwargs):
            return {
                "choices": [{"message": {"role": "assistant", "content": "test response"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }

        mock_openai_clients["async_completions"].side_effect = mock_create

        messages = [HumanMessage(content="Hello, world!")]
        await chat_nim._agenerate(messages=messages)

        mock_openai_clients["async_completions"].assert_called_once()
        call_kwargs = mock_openai_clients["async_completions"].call_args.kwargs
        assert "extra_headers" in call_kwargs
        assert call_kwargs["extra_headers"] == custom_headers


class TestLLMCustomHeaders:
    """Tests for NIM LLM custom headers handling."""

    def test_custom_headers_in_sync_call(self, mock_llm_dependencies):
        """Test that custom headers are passed in sync LLM calls."""
        custom_headers = {
            "X-Custom-Header1": "Value1",
            "X-Custom-Header2": "Value2",
        }
        set_request_default_headers_into_context(custom_headers)
        mock_llm_dependencies.return_value = custom_headers

        with patch("httpx.Client.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.is_success = True
            mock_response.json.return_value = {"choices": [{"text": "Test response"}]}
            mock_response.headers = {}
            mock_post.return_value = mock_response

            llm = NIM(model="test-model")
            llm._call(prompt="Hello, world!")

            mock_post.assert_called_once()
            _, kwargs = mock_post.call_args
            headers = kwargs["headers"]
            assert "X-Custom-Header1" in headers
            assert headers["X-Custom-Header1"] == "Value1"
            assert "X-Custom-Header2" in headers
            assert headers["X-Custom-Header2"] == "Value2"

    @pytest.mark.asyncio
    async def test_custom_headers_in_async_call(self, mock_llm_dependencies):
        """Test that custom headers are passed in async LLM calls."""
        custom_headers = {
            "X-Custom-Header1": "Value1",
            "X-Custom-Header2": "Value2",
        }
        set_request_default_headers_into_context(custom_headers)
        mock_llm_dependencies.return_value = custom_headers

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.is_success = True
            mock_response.json.return_value = {"choices": [{"text": "Test response"}]}
            mock_response.headers = {}
            mock_post.return_value = mock_response

            llm = NIM(model="test-model")
            await llm._acall(prompt="Hello, world!")

            mock_post.assert_awaited_once()
            _, kwargs = mock_post.call_args
            headers = kwargs["headers"]
            assert "X-Custom-Header1" in headers
            assert headers["X-Custom-Header1"] == "Value1"
            assert "X-Custom-Header2" in headers
            assert headers["X-Custom-Header2"] == "Value2"


class TestCheckCustomHeaders:
    """Tests for CheckRequestHandler custom headers handling."""

    def test_check_set_custom_headers_injects_service_principal(self):
        """set_custom_headers should inject service principal headers into context."""
        custom_headers = {
            "X-Custom-Header1": "Value1",
            "Content-Type": "application/json",
        }
        request = MagicMock()
        request.headers = custom_headers

        request_body = GuardrailCheckRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="test-model",
            guardrails=GuardrailsDataInput(config_ids=["test-config"], config=None),
        )

        handler = CheckRequestHandler(
            rails_service=MagicMock(),
            request=request,
            request_body=request_body,
            response_model=MagicMock(),
            workspace="test-workspace",
        )

        handler.set_custom_headers()
        context_headers = get_request_default_headers_from_context()

        # All x-* headers are retained
        assert context_headers["X-Custom-Header1"] == "Value1"
        assert "Content-Type" not in context_headers

        # Service principal headers are injected
        assert context_headers["X-NMP-Principal-Id"] == "service:guardrails"

    def test_check_set_custom_headers_no_incoming_headers(self):
        """set_custom_headers should still inject service principal even with no custom headers."""
        request = MagicMock()
        request.headers = {}

        request_body = GuardrailCheckRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="test-model",
            guardrails=GuardrailsDataInput(config_ids=["test-config"], config=None),
        )

        handler = CheckRequestHandler(
            rails_service=MagicMock(),
            request=request,
            request_body=request_body,
            response_model=MagicMock(),
            workspace="test-workspace",
        )

        handler.set_custom_headers()
        context_headers = get_request_default_headers_from_context()

        assert context_headers["X-NMP-Principal-Id"] == "service:guardrails"
