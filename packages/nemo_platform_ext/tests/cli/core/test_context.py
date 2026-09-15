# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

from nemo_platform_ext.cli.core.context import CLIContext
from nemo_platform_ext.config.models import NoAuthUser, OAuthUser


def test_context_instances_are_independent():
    """Test that CLIContext instances don't share state."""
    ctx1 = CLIContext(overrides={"base_url": "http://ctx1.example.com"})
    ctx2 = CLIContext(overrides={"base_url": "http://ctx2.example.com"})

    assert ctx1.overrides["base_url"] == "http://ctx1.example.com"
    assert ctx2.overrides["base_url"] == "http://ctx2.example.com"


def test_verbosity_default():
    """Test that verbosity defaults to 0."""
    ctx = CLIContext()
    assert ctx.verbosity == 0


def test_verbosity_can_be_set():
    """Test that verbosity can be set."""
    ctx = CLIContext(verbosity=1)
    assert ctx.verbosity == 1


def test_overrides_default_to_empty():
    """Test that overrides default to empty dict."""
    ctx = CLIContext()
    assert ctx.overrides == {}


def test_get_output_format_with_override():
    """Test get_output_format returns override when provided."""
    ctx = CLIContext()
    # Override should be returned directly without loading config
    result = ctx.get_output_format(override="yaml")
    assert result == "yaml"


def test_get_timestamp_format_with_override():
    """Test get_timestamp_format returns override when provided."""
    ctx = CLIContext()
    result = ctx.get_timestamp_format(override="relative")
    assert result == "relative"


def test_get_no_truncate_with_override():
    """Test get_no_truncate returns override when provided."""
    ctx = CLIContext()
    result = ctx.get_no_truncate(override=True)
    assert result is True


def test_get_no_truncate_default():
    """Test get_no_truncate returns False by default."""
    ctx = CLIContext()
    result = ctx.get_no_truncate(override=None)
    assert result is False


def test_get_client_uses_config_bootstrap_for_persisted_oauth_context_and_is_cached():
    """Test that get_client lets the SDK bootstrap config-backed OAuth auth and caches it."""
    ctx = CLIContext(overrides={"base_url": "http://test.example.com"})
    resolved_context = SimpleNamespace(
        cluster=SimpleNamespace(base_url="http://test.example.com"),
        context_name="dev",
        workspace="test-workspace",
        user=OAuthUser(name="dev-user", token="token-123", refresh_token="refresh-123"),
    )
    config_file = SimpleNamespace(contexts=[SimpleNamespace(name="dev")])

    with (
        patch("nemo_platform_ext.config.config.get_context", return_value=resolved_context),
        patch("nemo_platform_ext.config.config.Config.load") as mock_config_load,
        patch("nemo_platform_ext.config.config.Config.runtime_access_token_source_label", return_value=None),
        patch("nemo_platform.NeMoPlatform", autospec=True) as mock_client_cls,
    ):
        mock_config_load.return_value.get_config_file.return_value = config_file
        client = ctx.get_client()
        client2 = ctx.get_client()

    mock_client_cls.assert_called_once_with(
        base_url="http://test.example.com",
        context_name="dev",
        timeout=60.0,
        workspace="test-workspace",
    )
    assert client is mock_client_cls.return_value

    # Verify the client is cached
    assert client is client2


def test_get_client_preserves_direct_mode_for_synthetic_no_auth_context():
    """Test that synthesized default/no-auth contexts do not force SDK config bootstrap."""
    ctx = CLIContext(overrides={"base_url": "http://test.example.com"})
    resolved_context = SimpleNamespace(
        cluster=SimpleNamespace(base_url="http://test.example.com"),
        context_name="default",
        workspace="default",
        user=NoAuthUser(name="default-user"),
    )

    with (
        patch("nemo_platform_ext.config.config.get_context", return_value=resolved_context),
        patch("nemo_platform.NeMoPlatform", autospec=True) as mock_client_cls,
    ):
        ctx.get_client()

    mock_client_cls.assert_called_once_with(
        base_url="http://test.example.com",
        timeout=60.0,
        workspace="default",
    )


def test_get_client_passes_explicit_access_token_override():
    """Test that explicit access token overrides remain caller-managed static headers."""
    ctx = CLIContext(overrides={"base_url": "http://test.example.com", "access_token": "token-123"})
    resolved_context = SimpleNamespace(
        cluster=SimpleNamespace(base_url="http://test.example.com"),
        context_name="dev",
        workspace="test-workspace",
        user=OAuthUser(name="dev-user", token="token-123"),
    )

    with (
        patch("nemo_platform_ext.config.config.get_context", return_value=resolved_context),
        patch("nemo_platform.NeMoPlatform", autospec=True) as mock_client_cls,
    ):
        ctx.get_client()

    mock_client_cls.assert_called_once_with(
        base_url="http://test.example.com",
        default_headers={"Authorization": "Bearer token-123"},
        timeout=60.0,
        workspace="test-workspace",
    )


def test_get_async_client_uses_config_bootstrap_for_persisted_oauth_context_and_is_cached():
    """Test that get_async_client lets the SDK bootstrap config-backed OAuth auth and caches it."""
    ctx = CLIContext(overrides={"base_url": "http://test.example.com"})
    resolved_context = SimpleNamespace(
        cluster=SimpleNamespace(base_url="http://test.example.com"),
        context_name="dev",
        workspace="test-workspace",
        user=OAuthUser(name="dev-user", token="token-123", refresh_token="refresh-123"),
    )
    config_file = SimpleNamespace(contexts=[SimpleNamespace(name="dev")])

    with (
        patch("nemo_platform_ext.config.config.get_context", return_value=resolved_context),
        patch("nemo_platform_ext.config.config.Config.load") as mock_config_load,
        patch("nemo_platform_ext.config.config.Config.runtime_access_token_source_label", return_value=None),
        patch("nemo_platform.AsyncNeMoPlatform", autospec=True) as mock_client_cls,
    ):
        mock_config_load.return_value.get_config_file.return_value = config_file
        client = ctx.get_async_client()
        client2 = ctx.get_async_client()

    mock_client_cls.assert_called_once_with(
        base_url="http://test.example.com",
        context_name="dev",
        timeout=60.0,
        workspace="test-workspace",
    )
    assert client is mock_client_cls.return_value
    assert client is client2


def test_typed_client_shares_the_platform_clients_transport_and_auth():
    """typed_client derives a service client from the CLI's platform client, sharing its transport."""
    from nemo_platform_plugin.secrets.client import SecretsClient

    ctx = CLIContext(overrides={"base_url": "http://test.example.com", "access_token": "token-123"})
    resolved_context = SimpleNamespace(
        cluster=SimpleNamespace(base_url="http://test.example.com", certificate_authority=None),
        context_name="dev",
        workspace="test-workspace",
        user=OAuthUser(name="dev-user", token="token-123"),
    )

    with patch("nemo_platform_ext.config.config.get_context", return_value=resolved_context):
        secrets = ctx.typed_client(SecretsClient)
        platform = ctx.get_client()

    assert isinstance(secrets, SecretsClient)
    assert secrets._http is platform._client
    assert secrets.workspace == "test-workspace"
    assert secrets.default_headers == {"Authorization": "Bearer token-123"}


def test_get_workspace_returns_none_when_no_context_resolves():
    ctx = CLIContext(overrides={})

    with patch("nemo_platform_ext.config.config.get_context", side_effect=RuntimeError("no config")):
        assert ctx.get_workspace() is None
