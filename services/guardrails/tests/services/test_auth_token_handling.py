# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import MagicMock, patch
from urllib.parse import urlparse

import pytest
from nhx.guardrails.app.llms.chat.nim import ChatNIM
from nhx.guardrails.app.llms.completion.nim import NIM
from nhx.guardrails.app.utils.context_utils import (
    api_key_var,
    get_x_model_auth_token_from_context,
    set_x_model_auth_token_into_context,
)


class TestAuthTokenHandlingChat(unittest.TestCase):
    def setUp(self):
        # clear context var before each test
        api_key_var.set(None)
        # clear env vars
        self.patcher_env = patch.dict("os.environ", {}, clear=True)
        self.patcher_env.start()

        # patch openai.OpenAI as it is used in ChatNIM
        self.openai_patcher = patch("openai.OpenAI")
        self.mock_openai_client_class = self.openai_patcher.start()
        self.mock_openai_client = MagicMock()
        self.mock_openai_client_class.return_value = self.mock_openai_client

        # patch get_main_model_from_context to return None by default
        self.main_model_patcher = patch("nhx.guardrails.app.llms.utils.get_main_model_from_context")
        self.mock_get_main_model_from_context = self.main_model_patcher.start()
        self.mock_get_main_model_from_context.return_value = None

    def tearDown(self):
        self.patcher_env.stop()
        self.openai_patcher.stop()
        self.main_model_patcher.stop()

    def test_auth_token_from_context_var(self):
        # set auth token with contextvar
        test_api_key = "test_api_key_12345"
        set_x_model_auth_token_into_context(test_api_key)

        # instantiate ChatNIM
        chat_model = ChatNIM(model="test-model")

        # ensure OpenAI client was called with the correct api_key
        self.mock_openai_client_class.assert_called_with(
            api_key=test_api_key,
            base_url=chat_model._endpoint_url or chat_model.endpoint_url,
            max_retries=3,
        )

        # test get_x_model_auth_token() returns the correct key
        assert get_x_model_auth_token_from_context() == test_api_key

    def test_auth_token_missing_raises_exception(self):
        # ensure the context variable is not set
        api_key_var.set(None)

        with pytest.raises(Exception) as exc_info:
            ChatNIM(model="test-model")

        assert "Failed to find API key for test-model at URL https://integrate.api.nvidia.com/v1." in str(
            exc_info.value
        )

    def test_api_key_empty_uses_placeholder(self):
        # ensure the context variable is not set
        api_key_var.set(None)
        # setting endpoint to custom endpoint to trigger the 'EMPTY' API key scenario
        with patch.dict("os.environ", {"NIM_ENDPOINT_URL": "http://custom-endpoint.com"}):
            ChatNIM(model="test-model")

            # check the OpenAI client was called with api_key='EMPTY'
            self.mock_openai_client_class.assert_called_with(
                api_key="EMPTY",
                base_url="http://custom-endpoint.com",
                max_retries=3,
            )


class TestAuthTokenHandlingLLM:
    def setup_method(self, method):
        # clear context var before each test
        api_key_var.set(None)
        # clear env vars
        self.patcher_env = patch.dict("os.environ", {}, clear=True)
        self.patcher_env.start()

        # patch httpx.Client we use it in NIM
        self.httpx_client_patcher = patch("httpx.Client")
        self.mock_httpx_client_class = self.httpx_client_patcher.start()
        self.mock_httpx_client = self.mock_httpx_client_class.return_value

        # patch httpx.AsyncClient we used it in NIM
        self.httpx_async_client_patcher = patch("httpx.AsyncClient")
        self.mock_httpx_async_client_class = self.httpx_async_client_patcher.start()
        self.mock_httpx_async_client = self.mock_httpx_async_client_class.return_value

        # patch get_main_model_from_context to return None by default
        self.main_model_patcher = patch("nhx.guardrails.app.llms.utils.get_main_model_from_context")
        self.mock_get_main_model_from_context = self.main_model_patcher.start()
        self.mock_get_main_model_from_context.return_value = None

    def teardown_method(self, method):
        self.patcher_env.stop()
        self.httpx_client_patcher.stop()
        self.httpx_async_client_patcher.stop()
        self.main_model_patcher.stop()

    def test_auth_token_from_context_var(self):
        # set the API key via context variable (simulating X-Model-Authorization header)
        test_api_key = "test_api_key_12345"
        set_x_model_auth_token_into_context(test_api_key)

        llm = NIM(model="test-model")

        # Mock the response from httpx.Client.post
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.is_success = True
        mock_response.json.return_value = {"choices": [{"text": "test response"}]}
        self.mock_httpx_client_class.return_value.__enter__.return_value.post.return_value = mock_response

        llm._call(prompt="Hello, world!")

        headers_used = self.mock_httpx_client_class.return_value.__enter__.return_value.post.call_args[1]["headers"]
        expected_api_key = test_api_key
        assert headers_used["Authorization"] == f"Bearer {expected_api_key}"

    def test_auth_token_missing_uses_empty_placeholder(self):
        # ensure context var is not set
        api_key_var.set(None)

        llm = NIM(model="test-model")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.is_success = True
        mock_response.json.return_value = {"choices": [{"text": "test response"}]}

        self.mock_httpx_client_class.return_value.__enter__.return_value.post.return_value = mock_response

        llm._call(prompt="Hello, world!")

        headers_used = self.mock_httpx_client_class.return_value.__enter__.return_value.post.call_args[1]["headers"]
        assert headers_used["Authorization"] == "Bearer EMPTY"

    def test_auth_token_missing_raises_exception_on_unauthorized(self):
        # Ensure context var is not set
        api_key_var.set(None)

        llm = NIM(model="test-model")

        # Mock the response from httpx.Client.post
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.is_success = False
        mock_response._request = MagicMock()
        mock_response.json.return_value = {"error": "Unauthorized"}

        self.mock_httpx_client_class.return_value.__enter__.return_value.post.return_value = mock_response

        with pytest.raises(Exception) as exc_info:
            llm._call(prompt="Hello, world!")

        # Check the exact error message
        expected_message = (
            "Authentication failed. Please verify your API key is valid and configured to be used by this endpoint."
        )
        assert expected_message in str(exc_info.value)

    def test_api_key_empty_uses_placeholder_with_custom_endpoint(self):
        # ensure contextvar is not set
        api_key_var.set(None)

        # Set custom endpoint
        with patch.dict("os.environ", {"NIM_ENDPOINT_URL": "http://custom-endpoint.com"}):
            llm = NIM(model="test-model")

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.is_success = True
            mock_response.json.return_value = {"choices": [{"text": "test response"}]}
            self.mock_httpx_client_class.return_value.__enter__.return_value.post.return_value = mock_response

            llm._call(prompt="Hello, world!")

            # verify that the request was made with 'EMPTY' as the API key
            headers_used = self.mock_httpx_client_class.return_value.__enter__.return_value.post.call_args[1]["headers"]

            assert headers_used["Authorization"] == "Bearer EMPTY"

            # verify that the base URL is the custom endpoint that was set earlier
            url_used = self.mock_httpx_client_class.return_value.__enter__.return_value.post.call_args[1]["url"]

            parsed = urlparse(url_used)
            assert parsed.scheme == "http"
            assert parsed.hostname == "custom-endpoint.com"


if __name__ == "__main__":
    unittest.main()
