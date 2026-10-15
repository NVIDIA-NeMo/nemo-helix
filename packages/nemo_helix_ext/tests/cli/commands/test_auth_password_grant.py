# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from nemo_helix_ext.cli.app import app
from nemo_helix_plugin.client.oidc import AdvertisedOidcClient, NHXOIDCConfig
from typer.testing import CliRunner

from ..utils import assert_exit_code

runner = CliRunner()


def _mock_oidc_config() -> NHXOIDCConfig:
    return NHXOIDCConfig(
        auth_enabled=True,
        issuer="https://idp.example.com",
        clients=(
            AdvertisedOidcClient(
                name="public",
                client_id="test-client",
                client_authentication="public",
                default=True,
                token_endpoint="https://idp.example.com/token",
                device_authorization_endpoint="https://idp.example.com/device",
                scope_prefix="api://nhx",
            ),
        ),
    )


def test_login_password_grant_with_flags(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("NHX_BASE_URL", "https://cluster.example.com")
    monkeypatch.setattr(
        "nemo_helix_ext.cli.commands.auth.discover_nhx_config",
        lambda *_args, **_kwargs: _mock_oidc_config(),
    )
    monkeypatch.setattr(
        "nemo_helix_ext.auth.device_flow.authenticate_with_password_grant",
        lambda **_: SimpleNamespace(
            token_for_nhx="access-token",
            refresh_token="refresh-token",
            scope=None,
            expires_in=3600,
        ),
    )
    monkeypatch.setattr(
        "nemo_helix_ext.auth.login.decode_jwt_claims",
        lambda *_: {"email": "user@example.com", "scp": "platform:read"},
    )

    with patch("nemo_helix_ext.config.config.Config.write") as mock_write:
        result = runner.invoke(
            app,
            [
                "auth",
                "login",
                "--username",
                "user",
                "--password",
                "secret",
                "--scope",
                "platform:read",
            ],
        )

    assert_exit_code(result, 0)
    mock_write.assert_called_once()
    assert mock_write.call_args.args[0]["access_token"] == "access-token"
    assert mock_write.call_args.args[0]["refresh_token"] == "refresh-token"
    assert mock_write.call_args.args[0]["token_broker_url"] is None


def test_login_password_grant_with_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("NHX_BASE_URL", "https://cluster.example.com")
    monkeypatch.setenv("NHX_OIDC_USERNAME", "env-user")
    monkeypatch.setenv("NHX_OIDC_PASSWORD", "env-password")
    monkeypatch.setattr(
        "nemo_helix_ext.cli.commands.auth.discover_nhx_config",
        lambda *_args, **_kwargs: _mock_oidc_config(),
    )
    monkeypatch.setattr(
        "nemo_helix_ext.auth.device_flow.authenticate_with_password_grant",
        lambda **_: SimpleNamespace(token_for_nhx="access-token", refresh_token=None, scope=None, expires_in=3600),
    )
    monkeypatch.setattr(
        "nemo_helix_ext.auth.login.decode_jwt_claims",
        lambda *_: {"email": "user@example.com", "scp": ""},
    )

    with patch("nemo_helix_ext.config.config.Config.write") as mock_write:
        result = runner.invoke(app, ["auth", "login"])

    assert_exit_code(result, 0)
    mock_write.assert_called_once()
    assert mock_write.call_args.args[0]["access_token"] == "access-token"
    assert mock_write.call_args.args[0]["token_broker_url"] is None
