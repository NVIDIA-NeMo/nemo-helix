# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for mock_provider utils module."""

from unittest.mock import Mock

import pytest
from fastapi import Request
from nhx.common.config import Configuration
from nhx.core.inference_gateway.api.mock_provider.responses import MOCK_RESPONSE_HEADER
from nhx.core.inference_gateway.api.mock_provider.utils import is_mock_mode_enabled, is_mock_provider, is_mock_request
from nhx.core.inference_gateway.config import InferenceGatewayConfig


@pytest.fixture(autouse=True)
def clear_config_overrides():
    """Clear configuration overrides before and after each test."""
    Configuration.clear_overrides()
    yield
    Configuration.clear_overrides()


def _set_mock_prefix(prefix: str | None) -> None:
    """Helper to set the mock_provider_prefix config."""
    Configuration.set_overrides({InferenceGatewayConfig: InferenceGatewayConfig(mock_provider_prefix=prefix)})


class TestIsMockModeEnabled:
    """Tests for is_mock_mode_enabled function."""

    def test_returns_false_when_prefix_not_set(self):
        """Mock mode is disabled when mock_provider_prefix is None."""
        assert is_mock_mode_enabled() is False

    def test_returns_true_when_prefix_set(self):
        """Mock mode is enabled when mock_provider_prefix is set."""
        _set_mock_prefix("igw-mock-")
        assert is_mock_mode_enabled() is True

    def test_returns_true_for_any_prefix(self):
        """Mock mode is enabled for any non-None prefix value."""
        _set_mock_prefix("test-")
        assert is_mock_mode_enabled() is True


class TestIsMockProvider:
    """Tests for is_mock_provider function - prefix-based matching."""

    def test_returns_false_when_prefix_not_set(self):
        """No provider is mocked when mock_provider_prefix is None."""
        assert is_mock_provider("igw-mock-test") is False
        assert is_mock_provider("any-provider") is False

    def test_returns_true_when_name_starts_with_prefix(self):
        """Provider is mocked when name starts with configured prefix."""
        _set_mock_prefix("igw-mock-")

        assert is_mock_provider("igw-mock-judge") is True
        assert is_mock_provider("igw-mock-embeddings") is True
        assert is_mock_provider("igw-mock-") is True  # Exact match

    def test_returns_false_when_name_does_not_start_with_prefix(self):
        """Provider is NOT mocked when name doesn't start with prefix."""
        _set_mock_prefix("igw-mock-")

        assert is_mock_provider("real-llm-provider") is False
        assert is_mock_provider("nim-llm") is False
        assert is_mock_provider("my-igw-mock-provider") is False  # Prefix in middle

    def test_prefix_matching_is_case_sensitive(self):
        """Prefix matching is case-sensitive."""
        _set_mock_prefix("igw-mock-")

        assert is_mock_provider("IGW-MOCK-test") is False
        assert is_mock_provider("Igw-Mock-test") is False

    def test_custom_prefix(self):
        """Prefix matching works with custom prefix values."""
        _set_mock_prefix("test-mock-")

        assert is_mock_provider("test-mock-provider") is True
        assert is_mock_provider("igw-mock-provider") is False  # Different prefix

    def test_empty_string_prefix_matches_all(self):
        """Empty string prefix matches all provider names."""
        _set_mock_prefix("")

        assert is_mock_provider("any-provider") is True
        assert is_mock_provider("") is True


class TestIsMockRequest:
    """Tests for is_mock_request function - early return path detection."""

    @pytest.fixture
    def mock_request(self):
        """Create a mock FastAPI request."""
        request = Mock(spec=Request)
        request.headers = {}
        return request

    def test_returns_false_when_mode_disabled(self, mock_request):
        """Returns False when mock mode is disabled, even with header present."""
        mock_request.headers = {MOCK_RESPONSE_HEADER: '{"test": true}'}
        assert is_mock_request(mock_request) is False

    def test_returns_false_when_header_missing(self, mock_request):
        """Returns False when mock mode enabled but header is missing."""
        _set_mock_prefix("igw-mock-")
        assert is_mock_request(mock_request) is False

    def test_returns_true_when_mode_enabled_and_header_present(self, mock_request):
        """Returns True when mock mode enabled AND header is present."""
        _set_mock_prefix("igw-mock-")
        mock_request.headers = {MOCK_RESPONSE_HEADER: '{"test": true}'}
        assert is_mock_request(mock_request) is True

    def test_header_value_does_not_matter(self, mock_request):
        """Only checks for header presence, not its value."""
        _set_mock_prefix("igw-mock-")
        mock_request.headers = {MOCK_RESPONSE_HEADER: ""}  # Empty value
        assert is_mock_request(mock_request) is True
