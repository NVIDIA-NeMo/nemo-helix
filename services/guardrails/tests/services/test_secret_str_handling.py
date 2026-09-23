# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import MagicMock, patch

import pytest
from nhx.guardrails.app.llms.chat.nim import ChatNIM
from nhx.guardrails.app.llms.completion.nim import NIM
from nhx.guardrails.app.utils.context_utils import (
    api_key_var,
    set_x_model_auth_token_into_context,
)
from pydantic import SecretStr


class TestSecretStrHandlingChat(unittest.TestCase):
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

    def test_secret_str_from_context_var(self):
        """Test that API key from X-Model-Authorization header is properly wrapped in SecretStr."""
        test_api_key = "test_api_key_12345"
        set_x_model_auth_token_into_context(test_api_key)

        chat_model = ChatNIM(model="test-model", client=None, async_client=None)
        assert isinstance(chat_model.api_key, SecretStr)
        assert chat_model.api_key.get_secret_value() == test_api_key

    def test_secret_str_from_instance(self):
        """Test that API key passed to instance raises ValueError."""
        instance_api_key = "instance_api_key_1234567890"

        with pytest.raises(ValueError) as exc_info:
            ChatNIM(model="test-model", api_key=instance_api_key, client=None, async_client=None)

        assert "API keys cannot be passed directly to ChatNIM" in str(exc_info.value)


class TestSecretStrHandlingLLM:
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

    def test_secret_str_from_context_var(self):
        """Test that API key from X-Model-Authorization header is properly wrapped in SecretStr."""
        test_api_key = "test_api_key_12345"
        set_x_model_auth_token_into_context(test_api_key)

        llm = NIM(model="test-model")
        assert isinstance(llm.api_key, SecretStr)
        assert llm.api_key.get_secret_value() == test_api_key

    def test_secret_str_from_instance(self):
        """Test that API key passed to instance raises ValueError."""
        instance_api_key = "instance_api_key_1234567890"

        with pytest.raises(ValueError) as exc_info:
            NIM(model="test-model", api_key=instance_api_key)

        assert "API keys cannot be passed directly to NIM" in str(exc_info.value)
