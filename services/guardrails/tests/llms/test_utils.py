# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for guardrails.llms.utils module."""

from unittest.mock import MagicMock

from nhx.guardrails.app.llms.utils import (
    DEFAULT_PROVIDER_NAME,
    get_main_model_api_key,
    get_provider_from_context,
)
from nhx.guardrails.app.utils.context_utils import (
    api_key_var,
    request_main_model_var,
    set_main_model_into_context,
    set_x_model_auth_token_into_context,
)


class TestGetMainModelApiKey:
    """Tests for get_main_model_api_key function."""

    def setup_method(self):
        """Reset context vars before each test."""
        api_key_var.set(None)
        request_main_model_var.set(None)

    def test_returns_auth_token_from_context(self):
        """Auth token from X-Model-Authorization header should be returned."""
        set_x_model_auth_token_into_context("header_api_key_123")

        result = get_main_model_api_key()

        assert result is not None
        assert result.get_secret_value() == "header_api_key_123"

    def test_returns_none_when_no_auth_token(self):
        """Should return None when no auth token is in context."""
        api_key_var.set(None)
        request_main_model_var.set(None)

        result = get_main_model_api_key()

        assert result is None


class TestGetProviderFromContext:
    """Tests for get_provider_from_context function."""

    def setup_method(self):
        """Reset context vars before each test."""
        request_main_model_var.set(None)

    def test_returns_default_provider_when_no_model_in_context(self):
        """Should return DEFAULT_PROVIDER_NAME when main_model is None."""
        request_main_model_var.set(None)

        result = get_provider_from_context()

        assert result == DEFAULT_PROVIDER_NAME

    def test_returns_nim_for_nimchat_engine(self):
        """Should normalize 'nimchat' to 'nim'."""
        mock_model = MagicMock()
        mock_model.engine = "nimchat"
        set_main_model_into_context(mock_model)

        result = get_provider_from_context()

        assert result == "nim"

    def test_returns_nim_for_nimllm_engine(self):
        """Should normalize 'nimllm' to 'nim'."""
        mock_model = MagicMock()
        mock_model.engine = "nimllm"
        set_main_model_into_context(mock_model)

        result = get_provider_from_context()

        assert result == "nim"

    def test_returns_nim_for_nim_engine(self):
        """Should return 'nim' for 'nim' engine."""
        mock_model = MagicMock()
        mock_model.engine = "nim"
        set_main_model_into_context(mock_model)

        result = get_provider_from_context()

        assert result == "nim"

    def test_returns_openai_for_openai_engine(self):
        """Should return 'openai' unchanged for non-nim providers."""
        mock_model = MagicMock()
        mock_model.engine = "openai"
        set_main_model_into_context(mock_model)

        result = get_provider_from_context()

        assert result == "openai"

    def test_default_provider_from_env_var(self):
        """Should use DEFAULT_LLM_PROVIDER env var for default."""
        request_main_model_var.set(None)

        # The DEFAULT_PROVIDER_NAME is set at module import time,
        # so we can only test its current value
        result = get_provider_from_context()

        assert result == DEFAULT_PROVIDER_NAME
