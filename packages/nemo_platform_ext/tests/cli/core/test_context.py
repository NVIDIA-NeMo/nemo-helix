# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

from nemo_platform_ext.cli.core.context import CLIContext
from nemo_platform_ext.config.models import NoAuthUser, OAuthUser
from nemo_platform_plugin.client.client import NemoClient
from nemo_platform_plugin.client.constants import WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR
from nemo_platform_plugin.secrets.client import SecretsClient


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
    """A persisted OAuth context goes through the config/OIDC bootstrap and the client is cached."""
    ctx = CLIContext(overrides={"base_url": "http://test.example.com"})
    resolved_context = SimpleNamespace(
        cluster=SimpleNamespace(base_url="http://test.example.com", certificate_authority=None),
        context_name="dev",
        workspace="test-workspace",
        user=OAuthUser(name="dev-user", token="token-123", refresh_token="refresh-123"),
    )
    config_file = SimpleNamespace(contexts=[SimpleNamespace(name="dev")])

    with (
        patch("nemo_platform_ext.config.config.get_context", return_value=resolved_context),
        patch("nemo_platform_ext.config.config.Config.load") as mock_config_load,
        patch("nemo_platform_ext.config.config.Config.runtime_access_token_source_label", return_value=None),
        patch("nemo_platform_ext.client.bootstrap.build_nemo_client", autospec=True) as mock_build,
    ):
        mock_config_load.return_value.get_config_file.return_value = config_file
        client = ctx.get_client()
        client2 = ctx.get_client()

    mock_build.assert_called_once_with(
        base_url="http://test.example.com",
        context_name="dev",
        workspace="test-workspace",
        timeout=60.0,
    )
    assert client is mock_build.return_value
    assert client is client2


def test_get_client_preserves_direct_mode_for_synthetic_no_auth_context():
    """Synthesized default/no-auth contexts build a direct client with no config bootstrap."""
    ctx = CLIContext(overrides={"base_url": "http://test.example.com"})
    resolved_context = SimpleNamespace(
        cluster=SimpleNamespace(base_url="http://test.example.com", certificate_authority=None),
        context_name="default",
        workspace="default",
        user=NoAuthUser(name="default-user"),
    )

    with (
        patch("nemo_platform_ext.config.config.get_context", return_value=resolved_context),
        patch("nemo_platform_ext.client.bootstrap.build_nemo_client", autospec=True) as mock_build,
    ):
        client = ctx.get_client()

    mock_build.assert_not_called()
    assert isinstance(client, NemoClient)
    assert client.base_url == "http://test.example.com"
    assert client.workspace == "default"
    assert client.default_headers == {}
    assert client._auth is None


def test_get_client_passes_explicit_access_token_override():
    """Explicit access token overrides remain caller-managed static headers on a direct client."""
    ctx = CLIContext(overrides={"base_url": "http://test.example.com", "access_token": "token-123"})
    resolved_context = SimpleNamespace(
        cluster=SimpleNamespace(base_url="http://test.example.com", certificate_authority=None),
        context_name="dev",
        workspace="test-workspace",
        user=OAuthUser(name="dev-user", token="token-123"),
    )

    with (
        patch("nemo_platform_ext.config.config.get_context", return_value=resolved_context),
        patch("nemo_platform_ext.client.bootstrap.build_nemo_client", autospec=True) as mock_build,
    ):
        client = ctx.get_client()

    mock_build.assert_not_called()
    assert isinstance(client, NemoClient)
    assert client.workspace == "test-workspace"
    assert client.default_headers == {"Authorization": "Bearer token-123"}
    assert client._http.headers["Authorization"] == "Bearer token-123"


def test_get_client_uses_bootstrap_for_workload_identity(monkeypatch):
    """A workload identity token file forces the token-exchange bootstrap even without stored auth."""
    monkeypatch.setenv(WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR, "/var/run/secrets/token")
    ctx = CLIContext(overrides={"base_url": "http://test.example.com"})
    resolved_context = SimpleNamespace(
        cluster=SimpleNamespace(base_url="http://test.example.com", certificate_authority=None),
        context_name="default",
        workspace="default",
        user=NoAuthUser(name="default-user"),
    )

    with (
        patch("nemo_platform_ext.config.config.get_context", return_value=resolved_context),
        patch("nemo_platform_ext.client.bootstrap.build_nemo_client", autospec=True) as mock_build,
    ):
        ctx.get_client()

    mock_build.assert_called_once_with(
        base_url="http://test.example.com",
        context_name=None,
        workspace="default",
        timeout=60.0,
    )


def test_get_async_client_uses_config_bootstrap_for_persisted_oauth_context_and_is_cached():
    """The async client follows the same bootstrap rule and is cached."""
    ctx = CLIContext(overrides={"base_url": "http://test.example.com"})
    resolved_context = SimpleNamespace(
        cluster=SimpleNamespace(base_url="http://test.example.com", certificate_authority=None),
        context_name="dev",
        workspace="test-workspace",
        user=OAuthUser(name="dev-user", token="token-123", refresh_token="refresh-123"),
    )
    config_file = SimpleNamespace(contexts=[SimpleNamespace(name="dev")])

    with (
        patch("nemo_platform_ext.config.config.get_context", return_value=resolved_context),
        patch("nemo_platform_ext.config.config.Config.load") as mock_config_load,
        patch("nemo_platform_ext.config.config.Config.runtime_access_token_source_label", return_value=None),
        patch("nemo_platform_ext.client.bootstrap.build_async_nemo_client", autospec=True) as mock_build,
    ):
        mock_config_load.return_value.get_config_file.return_value = config_file
        client = ctx.get_async_client()
        client2 = ctx.get_async_client()

    mock_build.assert_called_once_with(
        base_url="http://test.example.com",
        context_name="dev",
        workspace="test-workspace",
        timeout=60.0,
    )
    assert client is mock_build.return_value
    assert client is client2


def test_typed_client_shares_transport_and_auth():
    """typed_client derives a service client that shares the base client's transport."""
    ctx = CLIContext(overrides={"base_url": "http://test.example.com", "access_token": "token-123"})
    resolved_context = SimpleNamespace(
        cluster=SimpleNamespace(base_url="http://test.example.com", certificate_authority=None),
        context_name="dev",
        workspace="test-workspace",
        user=OAuthUser(name="dev-user", token="token-123"),
    )

    with patch("nemo_platform_ext.config.config.get_context", return_value=resolved_context):
        secrets = ctx.typed_client(SecretsClient)

    assert isinstance(secrets, SecretsClient)
    assert secrets._http is ctx.get_client()._http
    assert secrets.workspace == "test-workspace"
    assert secrets.default_headers == {"Authorization": "Bearer token-123"}
