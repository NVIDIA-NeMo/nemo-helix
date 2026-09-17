# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for nmp_testing client utilities."""

import pytest
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.client.client import AsyncNemoClient
from nemo_platform_plugin.workspaces.client import WorkspacesClient
from nmp.common.config import Configuration
from nmp.core.entities.service import EntitiesService
from nmp.core.inference_gateway.config import InferenceGatewayConfig
from nmp.core.inference_gateway.service import InferenceGatewayService
from nmp.testing import (
    ClientContext,
    MockProviderResponse,
    as_user,
    create_test_client,
    short_unique_name,
    unique_email,
)


@pytest.fixture(autouse=True)
def clear_config_overrides():
    """Clear configuration overrides before and after each test."""
    Configuration.clear_overrides()
    yield
    Configuration.clear_overrides()


# =============================================================================
# short_unique_name tests
# =============================================================================


def test_short_unique_name_basic():
    """Test that short_unique_name generates unique names with prefix."""
    name1 = short_unique_name("test")
    name2 = short_unique_name("test")

    assert name1.startswith("test-")
    assert name2.startswith("test-")
    assert name1 != name2  # Should be unique


def test_short_unique_name_respects_max_length():
    """Test that short_unique_name respects max_length constraint."""
    name = short_unique_name("verylongprefix", max_length=20)
    assert len(name) <= 20


def test_short_unique_name_truncates_long_prefix():
    """Test that long prefixes are truncated to fit max_length."""
    # With max_length=15 and 8-char suffix + hyphen, prefix can be max 6 chars
    name = short_unique_name("abcdefghij", max_length=15)
    assert len(name) == 15
    assert name.startswith("abcdef-")


# =============================================================================
# unique_email tests
# =============================================================================


def test_unique_email_basic():
    """Test that unique_email generates unique emails."""
    email1 = unique_email()
    email2 = unique_email()

    assert email1.endswith("@example.com")
    assert email2.endswith("@example.com")
    assert email1 != email2  # Should be unique


def test_unique_email_custom_prefix():
    """Test that unique_email respects custom prefix."""
    email = unique_email("admin")
    assert email.startswith("admin-")
    assert email.endswith("@example.com")


# =============================================================================
# MockProviderResponse tests
# =============================================================================


def test_mock_provider_response_defaults():
    """Test that MockProviderResponse defaults response_code to 200."""
    response = MockProviderResponse(response_body={"ok": True})
    assert response.response_code == 200
    assert response.response_body == {"ok": True}


# =============================================================================
# create_test_client tests
# =============================================================================


def test_create_test_client_returns_sdk_by_default():
    """Test that create_test_client returns NeMoPlatform SDK by default."""
    from nemo_platform import NeMoPlatform

    with create_test_client(EntitiesService) as client:
        assert isinstance(client, NeMoPlatform)


def test_create_test_client_returns_client_context():
    """Test that create_test_client with ClientContext returns ClientContext."""
    with create_test_client(EntitiesService, client_type=ClientContext) as ctx:
        assert isinstance(ctx, ClientContext)
        assert ctx.sdk is not None
        assert isinstance(ctx.async_client, AsyncNemoClient)
        assert ctx.test_client is not None


def test_create_test_client_creates_default_workspace():
    """Test that create_test_client creates default workspace."""
    with create_test_client(EntitiesService, client_type=ClientContext) as ctx:
        # Default workspace should exist
        workspace = client_from_platform(ctx.sdk, WorkspacesClient).get_workspace(name="default").data()
        assert workspace.name == "default"


# =============================================================================
# as_user tests
# =============================================================================


def test_as_user_returns_new_sdk():
    """Test that as_user returns a new SDK client."""
    with create_test_client(EntitiesService, client_type=ClientContext) as ctx:
        user_sdk = as_user(ctx.sdk, "test@example.com")
        # Should be a different SDK instance
        assert user_sdk is not ctx.sdk


# =============================================================================
# igw_mock_provider_mode tests
# =============================================================================


def test_mock_provider_mode_enabled_sets_prefix():
    """Test that igw_mock_provider_mode=True sets the prefix to 'igw-mock-'."""
    with create_test_client(
        InferenceGatewayService,
        client_type=ClientContext,
        igw_mock_provider_mode=True,
    ) as ctx:
        # Verify the config override is set correctly
        config = Configuration.get_service_config(InferenceGatewayConfig)
        assert config.mock_provider_prefix == "igw-mock-"
        assert ctx.sdk is not None


def test_mock_provider_mode_merges_with_existing_config():
    """Test that igw_mock_provider_mode merges with existing service_configs."""
    custom_refresh_interval = 999

    with create_test_client(
        InferenceGatewayService,
        client_type=ClientContext,
        igw_mock_provider_mode=True,
        service_configs={
            InferenceGatewayService: InferenceGatewayConfig(
                refresh_model_cache_interval_sec=custom_refresh_interval,
            ),
        },
    ) as ctx:
        # Verify both the custom config and mock_provider_prefix are set
        config = Configuration.get_service_config(InferenceGatewayConfig)
        assert config.mock_provider_prefix == "igw-mock-"
        assert config.refresh_model_cache_interval_sec == custom_refresh_interval
        assert ctx.sdk is not None


def test_mock_provider_mode_false_does_not_add_prefix():
    """Test that igw_mock_provider_mode=False doesn't add the igw-mock- prefix."""
    with create_test_client(
        InferenceGatewayService,
        client_type=ClientContext,
        igw_mock_provider_mode=False,
        service_configs={
            InferenceGatewayService: InferenceGatewayConfig(
                refresh_model_cache_interval_sec=123,
            ),
        },
    ) as ctx:
        # Verify that when igw_mock_provider_mode=False, the prefix is not set
        config = Configuration.get_service_config(InferenceGatewayConfig)
        # Our custom config is there
        assert config.refresh_model_cache_interval_sec == 123
        # But mock_provider_prefix should be None (the default)
        assert config.mock_provider_prefix is None
        assert ctx.sdk is not None


def test_config_overrides_cleared_after_context_exit():
    """Test that config overrides are properly cleared after context exits."""
    # First, enable mock provider mode
    with create_test_client(
        InferenceGatewayService,
        client_type=ClientContext,
        igw_mock_provider_mode=True,
    ):
        config = Configuration.get_service_config(InferenceGatewayConfig)
        assert config.mock_provider_prefix == "igw-mock-"

    # After context exit, overrides should be cleared
    assert InferenceGatewayConfig not in Configuration._overrides
