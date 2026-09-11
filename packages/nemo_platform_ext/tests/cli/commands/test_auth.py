# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest
import yaml
from nemo_platform_ext.auth.helpers import decode_jwt_claims, generate_unsigned_jwt
from nemo_platform_ext.cli.app import app
from nemo_platform_plugin.auth.access_keys.types import (
    AccessKeyCreateRequest,
    AccessKeyCreateResponse,
    AccessKeyListResponse,
    AccessKeyMetadataResponse,
    AccessKeyRevokeResponse,
    AccessKeyRotateRequest,
    AccessKeyRotateResponse,
    AccessKeyStatusChangeResponse,
    AccessKeyWorkspaceGrant,
)
from nemo_platform_plugin.client.constants import WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR
from nemo_platform_plugin.client.errors import NotFoundError
from typer.testing import CliRunner

from ..utils import assert_exit_code

runner = CliRunner()


# ---------------------------------------------------------------------------
# Shared mock helpers
# ---------------------------------------------------------------------------


def _mock_oidc_config() -> SimpleNamespace:
    return SimpleNamespace(
        auth_enabled=True,
        issuer="https://idp.example.com",
        client_id="test-client",
        token_endpoint="https://idp.example.com/token",
        device_authorization_endpoint="https://idp.example.com/device",
        default_scopes="openid profile email offline_access",
        scope_prefix="api://nmp",
    )


def _discover_auth_enabled(
    url: str, timeout: float = 10.0, *, certificate_authority: str | None = None
) -> SimpleNamespace:
    return SimpleNamespace(auth_enabled=True)


def _discover_auth_disabled(
    url: str, timeout: float = 10.0, *, certificate_authority: str | None = None
) -> SimpleNamespace:
    return SimpleNamespace(auth_enabled=False)


def _discover_oidc_config(
    url: str, timeout: float = 10.0, *, certificate_authority: str | None = None
) -> SimpleNamespace:
    return _mock_oidc_config()


def _discover_no_oidc(url: str, timeout: float = 10.0, *, certificate_authority: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        auth_enabled=True,
        issuer=None,
        client_id=None,
        token_endpoint=None,
        device_authorization_endpoint=None,
    )


def _decode_jwt_noop(token: str) -> dict:
    return {}


def _created_access_key(
    name: str | None = None,
    description: str | None = None,
) -> AccessKeyCreateResponse:
    return AccessKeyCreateResponse(
        jti="ak_example",
        name=name,
        token="signed.jwt.token",
        token_type="Bearer",
        principal="alice@example.com",
        created_at=datetime(2026, 7, 28, 12, 0, tzinfo=UTC),
        expires_at=None,
        description=description,
        status="ACTIVE",
        issuer="https://platform.example.com/apis/auth",
        audiences=["nemo-platform-access-key"],
    )


def _access_key_not_found_error(jti: str) -> NotFoundError:
    return NotFoundError(
        httpx.Response(
            404,
            json={"detail": f"Scoped Access Key {jti} was not found"},
            request=httpx.Request("DELETE", f"https://platform.example.com/apis/auth/v2/access-keys/{jti}"),
        )
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def oauth_config_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for env_key in (
        "NMP_ACCESS_TOKEN",
        "NEMO_WORKLOAD_TOKEN",
        "NEMO_WORKLOAD_TOKEN_FILE",
        WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR,
    ):
        monkeypatch.delenv(env_key, raising=False)

    config_data = {
        "current_context": "default",
        "clusters": [
            {"name": "default", "base_url": "https://default.example.com"},
            {"name": "foo", "base_url": "https://foo.example.com"},
        ],
        "users": [
            {
                "type": "oauth",
                "name": "default",
                "token": "default-token",
                "refresh_token": "default-refresh",
            },
            {
                "type": "oauth",
                "name": "foo",
                "token": "foo-token",
                "refresh_token": "foo-refresh",
            },
        ],
        "contexts": [
            {
                "name": "default",
                "cluster": "default",
                "user": "default",
                "workspace": "default-workspace",
            },
            {
                "name": "foo",
                "cluster": "foo",
                "user": "foo",
                "workspace": "foo-workspace",
            },
        ],
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config_data, f)
    monkeypatch.setenv("NMP_CONFIG_FILE", str(config_path))
    return config_path


# ---------------------------------------------------------------------------
# token
# ---------------------------------------------------------------------------


def test_auth_token_prints_raw_token(oauth_config_file: Path) -> None:
    result = runner.invoke(app, ["--context", "foo", "auth", "token"])

    assert_exit_code(result, 0)
    assert result.output == "foo-token\n"


def test_auth_token_decode_prints_claims_json(oauth_config_file: Path) -> None:
    token = generate_unsigned_jwt(
        principal_id="alice@example.com",
        email="alice@example.com",
        groups=["team-ml"],
        scopes=["openid", "email"],
        extra_claims={"iss": "https://idp.example.com"},
    )
    with open(oauth_config_file) as f:
        config_data = yaml.safe_load(f)
    for user in config_data["users"]:
        if user["name"] == "foo":
            user["token"] = token
    with open(oauth_config_file, "w") as f:
        yaml.safe_dump(config_data, f)

    result = runner.invoke(app, ["--context", "foo", "auth", "token", "--decode"])

    assert_exit_code(result, 0)
    claims = json.loads(result.output)
    assert claims["sub"] == "alice@example.com"
    assert claims["email"] == "alice@example.com"
    assert claims["groups"] == ["team-ml"]
    assert claims["scope"] == "openid email"
    assert claims["iss"] == "https://idp.example.com"


def test_auth_token_decode_rejects_malformed_token(oauth_config_file: Path) -> None:
    result = runner.invoke(app, ["--context", "foo", "auth", "token", "--decode"])

    assert_exit_code(result, 1)
    assert "Current access token is not a decodable JWT" in result.output


# ---------------------------------------------------------------------------
# logout
# ---------------------------------------------------------------------------


def test_auth_logout_writes_to_selected_context(oauth_config_file: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("nemo_platform_ext.cli.commands.auth.discover_nmp_config", _discover_auth_enabled)
    with patch("nemo_platform_ext.config.config.Config.write") as mock_write:
        result = runner.invoke(app, ["--context", "foo", "auth", "logout"])

    assert_exit_code(result, 0)
    assert mock_write.call_count == 1
    assert mock_write.call_args.kwargs["context_name"] == "foo"


def test_auth_logout_clears_selected_context_credentials(
    oauth_config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("nemo_platform_ext.cli.commands.auth.discover_nmp_config", _discover_auth_enabled)

    result = runner.invoke(app, ["--context", "foo", "auth", "logout"])

    assert_exit_code(result, 0)
    assert "Logged out successfully" in result.output
    assert "Context: foo" in result.output
    assert "Config file:" in result.output

    with open(oauth_config_file) as f:
        data = yaml.safe_load(f)

    default_user = next(user for user in data["users"] if user["name"] == "default")
    foo_user = next(user for user in data["users"] if user["name"] == "foo")

    assert default_user["type"] == "oauth"
    assert default_user["token"] == "default-token"
    assert default_user["refresh_token"] == "default-refresh"
    assert foo_user["type"] == "no-auth"
    assert "token" not in foo_user
    assert "refresh_token" not in foo_user


def test_auth_logout_warns_when_runtime_token_override_remains(
    oauth_config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("nemo_platform_ext.cli.commands.auth.discover_nmp_config", _discover_auth_enabled)
    monkeypatch.setenv(
        "NMP_ACCESS_TOKEN",
        generate_unsigned_jwt(
            principal_id="svc-nemo-ci",
            email="svc-nemo-ci@example.com",
            expires_in_seconds=900,
        ),
    )

    result = runner.invoke(app, ["--context", "foo", "auth", "logout"])

    assert_exit_code(result, 0)
    assert "Logged out successfully" in result.output
    assert "NMP_ACCESS_TOKEN environment override is still active" in result.output


def test_auth_logout_fails_if_credentials_remain(oauth_config_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_platform_ext.config.config import Config

    def fake_write(*args, **kwargs):
        return Config.load(config_path=oauth_config_file)

    monkeypatch.setattr("nemo_platform_ext.cli.commands.auth.discover_nmp_config", _discover_auth_enabled)
    monkeypatch.setattr("nemo_platform_ext.config.config.Config.write", fake_write)

    result = runner.invoke(app, ["--context", "foo", "auth", "logout"])

    assert_exit_code(result, 1)
    assert "Logout did not clear credentials" in result.output
    assert "context 'foo'" in result.output
    assert oauth_config_file.name in result.output


# ---------------------------------------------------------------------------
# refresh
# ---------------------------------------------------------------------------


def test_auth_refresh_updates_selected_context_only(oauth_config_file: Path, monkeypatch: pytest.MonkeyPatch):
    class FakeTokenProvider:
        def __init__(self, *args, **kwargs):
            self.tokens = SimpleNamespace(
                access_token="foo-refreshed-token",
                refresh_token="foo-refreshed-refresh",
            )

        def force_refresh(self) -> None:
            return None

    def discover_refresh_config(
        url: str, timeout: float = 10.0, *, certificate_authority: str | None = None
    ) -> SimpleNamespace:
        return SimpleNamespace(
            client_id="test-client-id",
            token_endpoint="https://idp.example.com/token",
            default_scopes="openid profile email",
            scope_prefix=None,
        )

    monkeypatch.setattr("nemo_platform_ext.cli.commands.auth.discover_nmp_config", discover_refresh_config)
    monkeypatch.setattr("nemo_platform_ext.cli.commands.auth.OIDCTokenProvider", FakeTokenProvider)

    result = runner.invoke(app, ["--context", "foo", "auth", "refresh"])

    assert_exit_code(result, 0)

    with open(oauth_config_file) as f:
        data = yaml.safe_load(f)

    default_user = next(user for user in data["users"] if user["name"] == "default")
    foo_user = next(user for user in data["users"] if user["name"] == "foo")

    assert default_user["token"] == "default-token"
    assert default_user["refresh_token"] == "default-refresh"
    assert foo_user["token"] == "foo-refreshed-token"
    assert foo_user["refresh_token"] == "foo-refreshed-refresh"


def test_auth_refresh_regenerates_unsigned_token(oauth_config_file: Path) -> None:
    original_token = generate_unsigned_jwt(
        principal_id="admin@example.com",
        email="admin@example.com",
        groups=["platform-admin"],
        scopes=["platform:read", "platform:write"],
        expires_in_seconds=900,
        issued_at=1700000000,
        issuer="https://quickstart.local",
        extra_claims={"custom": "value"},
    )

    with open(oauth_config_file) as f:
        data = yaml.safe_load(f)

    foo_user = next(user for user in data["users"] if user["name"] == "foo")
    foo_user["token"] = original_token
    foo_user["refresh_token"] = None

    with open(oauth_config_file, "w") as f:
        yaml.safe_dump(data, f)

    result = runner.invoke(app, ["--context", "foo", "auth", "refresh"])

    assert_exit_code(result, 0)
    assert "Unsigned token refreshed successfully" in result.output

    with open(oauth_config_file) as f:
        refreshed_data = yaml.safe_load(f)

    refreshed_user = next(user for user in refreshed_data["users"] if user["name"] == "foo")
    refreshed_claims = decode_jwt_claims(refreshed_user["token"])

    assert refreshed_claims["sub"] == "admin@example.com"
    assert refreshed_claims["email"] == "admin@example.com"
    assert refreshed_claims["groups"] == ["platform-admin"]
    assert refreshed_claims["scope"] == "platform:read platform:write"
    assert refreshed_claims["iss"] == "https://quickstart.local"
    assert refreshed_claims["custom"] == "value"
    assert refreshed_claims["exp"] - refreshed_claims["iat"] == 900
    assert refreshed_claims["iat"] > 1700000000
    assert refreshed_user.get("refresh_token") is None


def test_auth_access_keys_create_prints_token(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("NMP_BASE_URL", "https://cluster.example.com")

    fake_platform_client = MagicMock()
    fake_access_keys_client = MagicMock()
    fake_access_keys_client.create_access_key.return_value.data.return_value = _created_access_key()

    monkeypatch.setattr("nemo_platform_ext.cli.core.context.CLIContext.get_client", lambda self: fake_platform_client)
    monkeypatch.setattr(
        "nemo_platform_ext.cli.commands.auth.client_from_platform",
        lambda platform, client_cls: fake_access_keys_client,
    )

    result = runner.invoke(app, ["auth", "access-keys", "create"])

    assert_exit_code(result, 0)
    assert result.output.strip() == "signed.jwt.token"
    body = fake_access_keys_client.create_access_key.call_args.kwargs["body"]
    assert body == AccessKeyCreateRequest()
    assert "expires_in_seconds" not in body.model_fields_set


def test_auth_access_keys_create_sends_optional_metadata_and_expiration(monkeypatch: pytest.MonkeyPatch):
    fake_platform_client = MagicMock()
    fake_access_keys_client = MagicMock()
    fake_access_keys_client.create_access_key.return_value.data.return_value = _created_access_key(
        "short-lived", "CI automation"
    )
    monkeypatch.setattr("nemo_platform_ext.cli.core.context.CLIContext.get_client", lambda self: fake_platform_client)
    monkeypatch.setattr(
        "nemo_platform_ext.cli.commands.auth.client_from_platform",
        lambda platform, client_cls: fake_access_keys_client,
    )

    result = runner.invoke(
        app,
        [
            "auth",
            "access-keys",
            "create",
            "--name",
            "short-lived",
            "--description",
            "CI automation",
            "--expires-in",
            "3600",
        ],
    )

    assert_exit_code(result, 0)
    fake_access_keys_client.create_access_key.assert_called_once_with(
        body=AccessKeyCreateRequest(
            name="short-lived",
            description="CI automation",
            expires_in_seconds=3600,
        ),
    )


def test_auth_access_keys_create_sends_service_account(monkeypatch: pytest.MonkeyPatch):
    fake_platform_client = MagicMock()
    fake_access_keys_client = MagicMock()
    fake_access_keys_client.create_access_key.return_value.data.return_value = _created_access_key("otel")
    monkeypatch.setattr("nemo_platform_ext.cli.core.context.CLIContext.get_client", lambda self: fake_platform_client)
    monkeypatch.setattr(
        "nemo_platform_ext.cli.commands.auth.client_from_platform",
        lambda platform, client_cls: fake_access_keys_client,
    )

    result = runner.invoke(
        app,
        ["auth", "access-keys", "create", "--name", "otel", "--service-account", "otel-collector"],
    )

    assert_exit_code(result, 0)
    fake_access_keys_client.create_access_key.assert_called_once_with(
        body=AccessKeyCreateRequest(name="otel", service_account_id="otel-collector")
    )


def test_auth_access_keys_create_sends_scope_rotation_and_workspace_grants(
    monkeypatch: pytest.MonkeyPatch,
):
    fake_access_keys_client = MagicMock()
    fake_access_keys_client.create_access_key.return_value.data.return_value = _created_access_key("intake-worker")
    monkeypatch.setattr(
        "nemo_platform_ext.cli.core.context.CLIContext.get_client",
        lambda self: MagicMock(),
    )
    monkeypatch.setattr(
        "nemo_platform_ext.cli.commands.auth.client_from_platform",
        lambda platform, client_cls: fake_access_keys_client,
    )
    rotated_jti = "ak_00000000000000000000000000000000"

    result = runner.invoke(
        app,
        [
            "auth",
            "access-keys",
            "create",
            "--scope",
            "intake, entities, intake",
            "--rotate",
            rotated_jti,
            "--workspace",
            "team-a",
            "--workspace",
            "team-b:Viewer, Admin",
        ],
    )

    assert_exit_code(result, 0)
    fake_access_keys_client.create_access_key.assert_called_once_with(
        body=AccessKeyCreateRequest(
            scope=["intake", "entities"],
            rotates=rotated_jti,
            workspaces=[
                AccessKeyWorkspaceGrant(workspace="team-a", roles=["Editor"]),
                AccessKeyWorkspaceGrant(workspace="team-b", roles=["Viewer", "Admin"]),
            ],
        )
    )


@pytest.mark.parametrize("workspace_value", ["team-a:", "team-a:,,"])
def test_auth_access_keys_create_rejects_workspace_grant_with_empty_roles(
    monkeypatch: pytest.MonkeyPatch, workspace_value: str
):
    fake_access_keys_client = MagicMock()
    monkeypatch.setattr(
        "nemo_platform_ext.cli.core.context.CLIContext.get_client",
        lambda self: MagicMock(),
    )
    monkeypatch.setattr(
        "nemo_platform_ext.cli.commands.auth.client_from_platform",
        lambda platform, client_cls: fake_access_keys_client,
    )

    result = runner.invoke(
        app,
        ["auth", "access-keys", "create", "--workspace", workspace_value],
    )

    assert_exit_code(result, 2)
    assert "specify" in result.output and "role after ':'" in result.output
    fake_access_keys_client.create_access_key.assert_not_called()


def test_auth_access_keys_create_sends_explicit_null_expiration(monkeypatch: pytest.MonkeyPatch):
    fake_platform_client = MagicMock()
    fake_access_keys_client = MagicMock()
    fake_access_keys_client.create_access_key.return_value.data.return_value = _created_access_key("long-lived")
    monkeypatch.setattr("nemo_platform_ext.cli.core.context.CLIContext.get_client", lambda self: fake_platform_client)
    monkeypatch.setattr(
        "nemo_platform_ext.cli.commands.auth.client_from_platform",
        lambda platform, client_cls: fake_access_keys_client,
    )

    result = runner.invoke(app, ["auth", "access-keys", "create", "--name", "long-lived", "--expires-in", "none"])

    assert_exit_code(result, 0)
    body = fake_access_keys_client.create_access_key.call_args.kwargs["body"]
    assert body == AccessKeyCreateRequest(name="long-lived", expires_in_seconds=None)
    assert "expires_in_seconds" in body.model_fields_set


def test_auth_access_keys_create_rejects_invalid_expiration(monkeypatch: pytest.MonkeyPatch):
    fake_platform_client = MagicMock()
    fake_access_keys_client = MagicMock()
    monkeypatch.setattr("nemo_platform_ext.cli.core.context.CLIContext.get_client", lambda self: fake_platform_client)
    monkeypatch.setattr(
        "nemo_platform_ext.cli.commands.auth.client_from_platform",
        lambda platform, client_cls: fake_access_keys_client,
    )

    result = runner.invoke(app, ["auth", "access-keys", "create", "--expires-in", "zero"])

    assert_exit_code(result, 1)
    assert "--expires-in must be a positive integer number of seconds" in result.output
    assert "'none'." in result.output
    fake_access_keys_client.create_access_key.assert_not_called()


def test_auth_access_keys_create_reports_disabled_feature(monkeypatch: pytest.MonkeyPatch):
    from nemo_platform_plugin.client.errors import NemoHTTPError

    fake_platform_client = MagicMock()
    fake_access_keys_client = MagicMock()
    # Simulate the real 404 response the server sends when the feature is disabled,
    # to exercise the NemoHTTPError → AccessKeyFeatureDisabledError translation path.
    fake_access_keys_client.create_access_key.side_effect = NemoHTTPError(
        httpx.Response(
            404,
            json={"detail": "Scoped Access Keys are not enabled", "code": "access_keys_disabled"},
            request=httpx.Request("POST", "https://platform.example.com/apis/auth/v2/access-keys"),
        )
    )
    monkeypatch.setattr("nemo_platform_ext.cli.core.context.CLIContext.get_client", lambda self: fake_platform_client)
    monkeypatch.setattr(
        "nemo_platform_ext.cli.commands.auth.client_from_platform",
        lambda platform, client_cls: fake_access_keys_client,
    )

    result = runner.invoke(app, ["auth", "access-keys", "create"])

    assert result.exit_code == 1
    assert "Scoped Access Keys are not enabled" in result.output


def test_auth_access_keys_list_outputs_owned_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_access_keys_client = MagicMock()
    fake_access_keys_client.list_access_keys.return_value.data.return_value = AccessKeyListResponse(
        data=[
            AccessKeyMetadataResponse(
                jti="ak_example",
                name="ci-build",
                principal="alice@example.com",
                created_at=datetime(2026, 7, 28, 12, 0, tzinfo=UTC),
                last_used_at=datetime(2026, 7, 28, 12, 30, tzinfo=UTC),
                grace_period_expires_at=datetime(2026, 7, 28, 13, 0, tzinfo=UTC),
                description="CI automation",
                status="ROTATING",
                issuer="https://platform.example.com/apis/auth",
                audiences=["nemo-platform-access-key"],
            )
        ]
    )
    monkeypatch.setattr("nemo_platform_ext.cli.core.context.CLIContext.get_client", lambda self: MagicMock())
    monkeypatch.setattr(
        "nemo_platform_ext.cli.commands.auth.client_from_platform",
        lambda platform, client_cls: fake_access_keys_client,
    )

    result = runner.invoke(app, ["--output-format", "json", "auth", "access-keys", "list"])

    assert_exit_code(result, 0)
    assert '"jti": "ak_example"' in result.output
    assert '"name": "ci-build"' in result.output
    assert '"description": "CI automation"' in result.output
    assert '"status": "ROTATING"' in result.output
    assert '"last_used_at": "2026-07-28T12:30:00Z"' in result.output
    assert '"grace_period_expires_at": "2026-07-28T13:00:00Z"' in result.output
    fake_access_keys_client.list_access_keys.assert_called_once_with(query_params={"page": 1, "page_size": 100})


def test_auth_access_keys_list_table_includes_key_name(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_access_keys_client = MagicMock()
    fake_access_keys_client.list_access_keys.return_value.data.return_value = AccessKeyListResponse(
        data=[
            AccessKeyMetadataResponse(
                jti="ak_example",
                name="ci-build",
                principal="alice@example.com",
                created_at=datetime(2026, 7, 28, 12, 0, tzinfo=UTC),
                last_used_at=datetime(2026, 7, 28, 12, 30, tzinfo=UTC),
                grace_period_expires_at=datetime(2026, 7, 28, 13, 0, tzinfo=UTC),
                description="CI automation",
                status="ROTATING",
                issuer="https://platform.example.com/apis/auth",
                audiences=["nemo-platform-access-key"],
                scope=["intake"],
            )
        ]
    )
    monkeypatch.setattr("nemo_platform_ext.cli.core.context.CLIContext.get_client", lambda self: MagicMock())
    monkeypatch.setattr(
        "nemo_platform_ext.cli.commands.auth.client_from_platform",
        lambda platform, client_cls: fake_access_keys_client,
    )

    result = runner.invoke(app, ["auth", "access-keys", "list"])

    assert_exit_code(result, 0)
    assert "ak_example" in result.output
    assert "ci-build" in result.output
    assert "CI automation" in result.output
    assert "ROTATING" in result.output
    assert "last_used_at" in result.output
    assert "grace_period_expires_at" in result.output
    assert "2026-07-28T12:30:00Z" in result.output
    assert "2026-07-28T13:00:00Z" in result.output
    assert "intake" in result.output
    fake_access_keys_client.list_access_keys.assert_called_once_with(query_params={"page": 1, "page_size": 100})


def test_auth_access_keys_list_points_to_next_page(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_access_keys_client = MagicMock()
    fake_access_keys_client.list_access_keys.return_value.data.return_value = AccessKeyListResponse(
        data=[],
        has_more=True,
    )
    monkeypatch.setattr("nemo_platform_ext.cli.core.context.CLIContext.get_client", lambda _self: MagicMock())
    monkeypatch.setattr(
        "nemo_platform_ext.cli.commands.auth.client_from_platform",
        lambda _platform, _client_cls: fake_access_keys_client,
    )

    result = runner.invoke(
        app,
        ["--output-format", "json", "auth", "access-keys", "list", "--page", "3", "--page-size", "25"],
    )

    assert_exit_code(result, 0)
    assert result.stdout.strip() == "[]"
    assert "use --page 4" in result.stderr
    fake_access_keys_client.list_access_keys.assert_called_once_with(query_params={"page": 3, "page_size": 25})


def test_auth_access_keys_revoke_sends_jti(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_access_keys_client = MagicMock()
    fake_access_keys_client.revoke_access_key.return_value.data.return_value = AccessKeyRevokeResponse(
        jti="ak_example",
        revoked=True,
    )
    monkeypatch.setattr("nemo_platform_ext.cli.core.context.CLIContext.get_client", lambda self: MagicMock())
    monkeypatch.setattr(
        "nemo_platform_ext.cli.commands.auth.client_from_platform",
        lambda platform, client_cls: fake_access_keys_client,
    )

    result = runner.invoke(app, ["auth", "access-keys", "revoke", "ak_example"])

    assert_exit_code(result, 0)
    assert "Revoked Scoped Access Key ak_example." in result.output
    fake_access_keys_client.revoke_access_key.assert_called_once_with(jti="ak_example")


def test_auth_access_keys_revoke_reports_already_revoked(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_access_keys_client = MagicMock()
    fake_access_keys_client.revoke_access_key.return_value.data.return_value = AccessKeyRevokeResponse(
        jti="ak_example",
        revoked=False,
    )
    monkeypatch.setattr("nemo_platform_ext.cli.core.context.CLIContext.get_client", lambda _self: MagicMock())
    monkeypatch.setattr(
        "nemo_platform_ext.cli.commands.auth.client_from_platform",
        lambda _platform, _client_cls: fake_access_keys_client,
    )

    result = runner.invoke(app, ["auth", "access-keys", "revoke", "ak_example"])

    assert_exit_code(result, 0)
    assert "Scoped Access Key ak_example was already revoked." in result.output


def test_auth_access_keys_revoke_reports_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_access_keys_client = MagicMock()
    fake_access_keys_client.revoke_access_key.side_effect = _access_key_not_found_error("ak_unknown")
    monkeypatch.setattr("nemo_platform_ext.cli.core.context.CLIContext.get_client", lambda _self: MagicMock())
    monkeypatch.setattr(
        "nemo_platform_ext.cli.commands.auth.client_from_platform",
        lambda _platform, _client_cls: fake_access_keys_client,
    )

    result = runner.invoke(app, ["auth", "access-keys", "revoke", "ak_unknown"])

    assert_exit_code(result, 3)
    assert "Not found: (404) Scoped Access Key ak_unknown was not found" in result.output
    fake_access_keys_client.revoke_access_key.assert_called_once_with(jti="ak_unknown")


def test_auth_access_keys_suspend_and_unsuspend(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_access_keys_client = MagicMock()
    fake_access_keys_client.suspend_access_key.return_value.data.return_value = AccessKeyStatusChangeResponse(
        jti="ak_example", status="SUSPENDED", changed=True
    )
    fake_access_keys_client.unsuspend_access_key.return_value.data.return_value = AccessKeyStatusChangeResponse(
        jti="ak_example", status="ACTIVE", changed=True
    )
    monkeypatch.setattr("nemo_platform_ext.cli.core.context.CLIContext.get_client", lambda _self: MagicMock())
    monkeypatch.setattr(
        "nemo_platform_ext.cli.commands.auth.client_from_platform",
        lambda _platform, _client_cls: fake_access_keys_client,
    )

    suspend_result = runner.invoke(app, ["auth", "access-keys", "suspend", "ak_example"])
    unsuspend_result = runner.invoke(app, ["auth", "access-keys", "unsuspend", "ak_example"])

    assert_exit_code(suspend_result, 0)
    assert "Suspended Scoped Access Key ak_example." in suspend_result.output
    assert_exit_code(unsuspend_result, 0)
    assert "Unsuspended Scoped Access Key ak_example." in unsuspend_result.output
    fake_access_keys_client.suspend_access_key.assert_called_once_with(jti="ak_example")
    fake_access_keys_client.unsuspend_access_key.assert_called_once_with(jti="ak_example")


def test_auth_access_keys_unsuspend_noop_reports_expired_status(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_access_keys_client = MagicMock()
    fake_access_keys_client.unsuspend_access_key.return_value.data.return_value = AccessKeyStatusChangeResponse(
        jti="ak_example", status="EXPIRED", changed=False
    )
    monkeypatch.setattr("nemo_platform_ext.cli.core.context.CLIContext.get_client", lambda _self: MagicMock())
    monkeypatch.setattr(
        "nemo_platform_ext.cli.commands.auth.client_from_platform",
        lambda _platform, _client_cls: fake_access_keys_client,
    )

    result = runner.invoke(app, ["auth", "access-keys", "unsuspend", "ak_example"])

    assert_exit_code(result, 0)
    assert "Scoped Access Key ak_example was already expired." in result.output


def test_auth_access_keys_rotate_prints_new_token(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_access_keys_client = MagicMock()
    fake_access_keys_client.rotate_access_key.return_value.data.return_value = AccessKeyRotateResponse(
        new_key=AccessKeyCreateResponse(
            jti="ak_successor",
            name="ci-intake",
            token="signed.jwt.token",
            token_type="Bearer",
            principal="alice@example.com",
            created_at=datetime(2026, 7, 28, 12, 0, tzinfo=UTC),
            expires_at=None,
            description=None,
            status="ACTIVE",
            issuer="https://platform.example.com/apis/auth",
            audiences=["nemo-platform-access-key"],
        ),
        previous_jti="ak_example",
        previous_status="ROTATING",
        grace_period_seconds=3600,
    )
    monkeypatch.setattr("nemo_platform_ext.cli.core.context.CLIContext.get_client", lambda _self: MagicMock())
    monkeypatch.setattr(
        "nemo_platform_ext.cli.commands.auth.client_from_platform",
        lambda _platform, _client_cls: fake_access_keys_client,
    )

    result = runner.invoke(app, ["auth", "access-keys", "rotate", "ak_example"])

    assert_exit_code(result, 0)
    assert "signed.jwt.token" in result.output
    assert "Rotated Scoped Access Key ak_example; it remains usable for a grace period of 3600s." in result.output
    assert "no longer usable" not in result.output
    fake_access_keys_client.rotate_access_key.assert_called_once_with(jti="ak_example", body=AccessKeyRotateRequest())


def test_auth_access_keys_rotate_passes_explicit_grace_period_seconds(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_access_keys_client = MagicMock()
    fake_access_keys_client.rotate_access_key.return_value.data.return_value = AccessKeyRotateResponse(
        new_key=AccessKeyCreateResponse(
            jti="ak_successor",
            name="ci-intake",
            token="signed.jwt.token",
            token_type="Bearer",
            principal="alice@example.com",
            created_at=datetime(2026, 7, 28, 12, 0, tzinfo=UTC),
            expires_at=None,
            description=None,
            status="ACTIVE",
            issuer="https://platform.example.com/apis/auth",
            audiences=["nemo-platform-access-key"],
        ),
        previous_jti="ak_example",
        previous_status="ROTATING",
        grace_period_seconds=3600,
    )
    monkeypatch.setattr("nemo_platform_ext.cli.core.context.CLIContext.get_client", lambda _self: MagicMock())
    monkeypatch.setattr(
        "nemo_platform_ext.cli.commands.auth.client_from_platform",
        lambda _platform, _client_cls: fake_access_keys_client,
    )

    result = runner.invoke(app, ["auth", "access-keys", "rotate", "ak_example", "--grace-period-seconds", "3600"])

    assert_exit_code(result, 0)
    fake_access_keys_client.rotate_access_key.assert_called_once_with(
        jti="ak_example", body=AccessKeyRotateRequest(grace_period_seconds=3600)
    )


def test_auth_access_keys_rotate_reports_grace_period_expiration(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_access_keys_client = MagicMock()
    fake_access_keys_client.rotate_access_key.return_value.data.return_value = AccessKeyRotateResponse(
        new_key=AccessKeyCreateResponse(
            jti="ak_successor",
            name="ci-intake",
            token="signed.jwt.token",
            token_type="Bearer",
            principal="alice@example.com",
            created_at=datetime(2026, 7, 28, 12, 0, tzinfo=UTC),
            expires_at=None,
            description=None,
            status="ACTIVE",
            issuer="https://platform.example.com/apis/auth",
            audiences=["nemo-platform-access-key"],
        ),
        previous_jti="ak_example",
        previous_status="ROTATING",
        grace_period_seconds=3600,
        grace_period_expires_at=datetime(2026, 7, 28, 13, 0, tzinfo=UTC),
    )
    monkeypatch.setattr("nemo_platform_ext.cli.core.context.CLIContext.get_client", lambda _self: MagicMock())
    monkeypatch.setattr(
        "nemo_platform_ext.cli.commands.auth.client_from_platform",
        lambda _platform, _client_cls: fake_access_keys_client,
    )

    result = runner.invoke(app, ["auth", "access-keys", "rotate", "ak_example"])

    assert_exit_code(result, 0)
    assert (
        "Rotated Scoped Access Key ak_example; it remains usable until "
        "2026-07-28T13:00:00+00:00 (grace period 3600s)." in result.output
    )


def test_auth_access_keys_rotate_reports_revoked_status(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_access_keys_client = MagicMock()
    fake_access_keys_client.rotate_access_key.return_value.data.return_value = AccessKeyRotateResponse(
        new_key=AccessKeyCreateResponse(
            jti="ak_successor",
            name="ci-intake",
            token="signed.jwt.token",
            token_type="Bearer",
            principal="alice@example.com",
            created_at=datetime(2026, 7, 28, 12, 0, tzinfo=UTC),
            expires_at=None,
            description=None,
            status="ACTIVE",
            issuer="https://platform.example.com/apis/auth",
            audiences=["nemo-platform-access-key"],
        ),
        previous_jti="ak_example",
        previous_status="REVOKED",
        grace_period_seconds=3600,
    )
    monkeypatch.setattr("nemo_platform_ext.cli.core.context.CLIContext.get_client", lambda _self: MagicMock())
    monkeypatch.setattr(
        "nemo_platform_ext.cli.commands.auth.client_from_platform",
        lambda _platform, _client_cls: fake_access_keys_client,
    )

    result = runner.invoke(app, ["auth", "access-keys", "rotate", "ak_example"])

    assert_exit_code(result, 0)
    assert "Rotated Scoped Access Key ak_example; it is now REVOKED and is no longer usable." in result.output


def test_auth_access_keys_rotate_reports_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_access_keys_client = MagicMock()
    fake_access_keys_client.rotate_access_key.side_effect = _access_key_not_found_error("ak_unknown")
    monkeypatch.setattr("nemo_platform_ext.cli.core.context.CLIContext.get_client", lambda _self: MagicMock())
    monkeypatch.setattr(
        "nemo_platform_ext.cli.commands.auth.client_from_platform",
        lambda _platform, _client_cls: fake_access_keys_client,
    )

    result = runner.invoke(app, ["auth", "access-keys", "rotate", "ak_unknown"])

    assert_exit_code(result, 3)
    assert "Not found: (404) Scoped Access Key ak_unknown was not found" in result.output


def test_auth_access_keys_help_exposes_lifecycle_commands() -> None:
    result = runner.invoke(app, ["auth", "access-keys", "--help"])

    assert_exit_code(result, 0)
    assert "create" in result.output
    assert "list" in result.output
    assert "revoke" in result.output
    assert "suspend" in result.output
    assert "unsuspend" in result.output
    assert "rotate" in result.output

    create_help = runner.invoke(app, ["auth", "access-keys", "create", "--help"])
    assert_exit_code(create_help, 0)
    assert "Use 'none' to request no expiration" in " ".join(create_help.output.split())

    revoke_help = runner.invoke(app, ["auth", "access-keys", "revoke", "--help"])
    assert_exit_code(revoke_help, 0)
    assert "Stable ID of the Scoped Access Key" in " ".join(revoke_help.output.split())

    suspend_help = runner.invoke(app, ["auth", "access-keys", "suspend", "--help"])
    assert_exit_code(suspend_help, 0)
    assert "Unlike revocation, suspension is reversible until the key expires" in " ".join(suspend_help.output.split())


def test_auth_tokens_group_is_not_exposed() -> None:
    result = runner.invoke(app, ["auth", "tokens", "create"])

    assert result.exit_code != 0
    assert "No such command" in result.output
    assert "tokens" in result.output


def test_top_level_access_keys_group_is_not_exposed() -> None:
    result = runner.invoke(app, ["access-keys", "--help"])

    assert result.exit_code != 0
    assert "No such command" in result.output
    assert "access-keys" in result.output


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


def test_auth_status_shows_warning_for_unsigned_token(oauth_config_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nemo_platform_ext.cli.commands.auth.discover_nmp_config", _discover_auth_enabled)
    unsigned_token = generate_unsigned_jwt(
        principal_id="admin@example.com",
        email="admin@example.com",
        expires_in_seconds=900,
        issued_at=1700000000,
    )

    with open(oauth_config_file) as f:
        data = yaml.safe_load(f)

    foo_user = next(user for user in data["users"] if user["name"] == "foo")
    foo_user["token"] = unsigned_token
    foo_user["refresh_token"] = None

    with open(oauth_config_file, "w") as f:
        yaml.safe_dump(data, f)

    result = runner.invoke(app, ["--context", "foo", "auth", "status"])

    assert_exit_code(result, 0)
    assert "Unsigned JWT (alg=none)" in result.output
    assert "local/testing" in result.output


def test_runtime_token_source_label_ignores_workload_identity_token_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nemo_platform_ext.cli.commands.auth import _runtime_token_source_label

    token_file = tmp_path / "missing-workload-token.jwt"
    monkeypatch.delenv("NMP_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("NEMO_WORKLOAD_TOKEN", raising=False)
    monkeypatch.setenv("NEMO_WORKLOAD_TOKEN_FILE", str(token_file))
    monkeypatch.setenv(WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR, str(token_file))

    assert _runtime_token_source_label() is None


def test_runtime_token_source_label_returns_none_when_config_label_fails(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from nemo_platform_ext.cli.commands.auth import _runtime_token_source_label

    monkeypatch.setattr(
        "nemo_platform_ext.config.config.Config.runtime_access_token_source_label",
        lambda: (_ for _ in ()).throw(ValueError("invalid runtime token")),
    )

    with caplog.at_level(logging.DEBUG, logger="nemo_platform_ext.cli.commands.auth"):
        assert _runtime_token_source_label() is None

    assert "Failed to resolve runtime token override source label" in caplog.text
    assert "invalid runtime token" in caplog.text


def test_auth_status_shows_config_file_credential_source(
    oauth_config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("nemo_platform_ext.cli.commands.auth.discover_nmp_config", _discover_auth_enabled)

    result = runner.invoke(app, ["--context", "foo", "auth", "status"])

    assert_exit_code(result, 0)
    assert "Config File" in result.output
    assert oauth_config_file.name in result.output
    assert "Credential Source" in result.output
    assert "config file" in result.output


# ---------------------------------------------------------------------------
# login
# ---------------------------------------------------------------------------


def test_auth_login_with_base_url_updates_selected_context(oauth_config_file: Path, monkeypatch: pytest.MonkeyPatch):
    def password_grant(**kwargs) -> SimpleNamespace:
        return SimpleNamespace(token_for_nmp="foo-access-token", refresh_token="foo-refresh-token")

    monkeypatch.setattr("nemo_platform_ext.cli.commands.auth.discover_nmp_config", _discover_oidc_config)
    monkeypatch.setattr("nemo_platform_ext.auth.device_flow.authenticate_with_password_grant", password_grant)
    monkeypatch.setattr("nemo_platform_ext.cli.commands.auth.decode_jwt_claims", _decode_jwt_noop)

    result = runner.invoke(
        app,
        [
            "--context",
            "foo",
            "auth",
            "login",
            "--base-url",
            "https://foo-updated.example.com",
            "--username",
            "user",
            "--password",
            "secret",
        ],
    )

    assert_exit_code(result, 0)

    with open(oauth_config_file) as f:
        data = yaml.safe_load(f)

    default_cluster = next(cluster for cluster in data["clusters"] if cluster["name"] == "default")
    foo_cluster = next(cluster for cluster in data["clusters"] if cluster["name"] == "foo")
    foo_user = next(user for user in data["users"] if user["name"] == "foo")

    assert data["current_context"] == "foo"
    assert default_cluster["base_url"].rstrip("/") == "https://default.example.com"
    assert foo_cluster["base_url"].rstrip("/") == "https://foo-updated.example.com"
    assert foo_user["token"] == "foo-access-token"
    assert foo_user["refresh_token"] == "foo-refresh-token"


def test_auth_login_context_flag_updates_selected_context(oauth_config_file: Path, monkeypatch: pytest.MonkeyPatch):
    def password_grant(**kwargs) -> SimpleNamespace:
        return SimpleNamespace(token_for_nmp="foo-access-token", refresh_token="foo-refresh-token")

    monkeypatch.setattr("nemo_platform_ext.cli.commands.auth.discover_nmp_config", _discover_oidc_config)
    monkeypatch.setattr("nemo_platform_ext.auth.device_flow.authenticate_with_password_grant", password_grant)
    monkeypatch.setattr("nemo_platform_ext.cli.commands.auth.decode_jwt_claims", _decode_jwt_noop)

    result = runner.invoke(
        app,
        [
            "auth",
            "login",
            "--context",
            "foo",
            "--base-url",
            "https://foo-updated.example.com",
            "--username",
            "user",
            "--password",
            "secret",
        ],
    )

    assert_exit_code(result, 0)

    with open(oauth_config_file) as f:
        data = yaml.safe_load(f)

    default_cluster = next(cluster for cluster in data["clusters"] if cluster["name"] == "default")
    foo_cluster = next(cluster for cluster in data["clusters"] if cluster["name"] == "foo")
    foo_user = next(user for user in data["users"] if user["name"] == "foo")

    assert data["current_context"] == "foo"
    assert default_cluster["base_url"].rstrip("/") == "https://default.example.com"
    assert foo_cluster["base_url"].rstrip("/") == "https://foo-updated.example.com"
    assert foo_user["token"] == "foo-access-token"
    assert foo_user["refresh_token"] == "foo-refresh-token"


def test_auth_login_warns_when_env_access_token_will_override_saved_credentials(
    oauth_config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def password_grant(**kwargs) -> SimpleNamespace:
        return SimpleNamespace(token_for_nmp="foo-access-token", refresh_token="foo-refresh-token")

    monkeypatch.setenv(
        "NMP_ACCESS_TOKEN",
        generate_unsigned_jwt(
            principal_id="svc-nemo-ci",
            email="svc-nemo-ci@example.com",
            expires_in_seconds=900,
        ),
    )
    monkeypatch.setattr("nemo_platform_ext.cli.commands.auth.discover_nmp_config", _discover_oidc_config)
    monkeypatch.setattr("nemo_platform_ext.auth.device_flow.authenticate_with_password_grant", password_grant)
    monkeypatch.setattr("nemo_platform_ext.cli.commands.auth.decode_jwt_claims", _decode_jwt_noop)

    result = runner.invoke(
        app,
        [
            "auth",
            "login",
            "--context",
            "foo",
            "--base-url",
            "https://foo-updated.example.com",
            "--username",
            "user",
            "--password",
            "secret",
        ],
    )

    assert_exit_code(result, 0)
    assert "NMP_ACCESS_TOKEN environment override is active" in result.output
    assert "Unset the runtime token override" in result.output

    with open(oauth_config_file) as f:
        data = yaml.safe_load(f)

    foo_user = next(user for user in data["users"] if user["name"] == "foo")
    assert foo_user["token"] == "foo-access-token"
    assert foo_user["refresh_token"] == "foo-refresh-token"


def test_auth_login_uses_context_certificate_authority(
    oauth_config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with open(oauth_config_file) as f:
        config_data = yaml.safe_load(f)
    foo_cluster = next(cluster for cluster in config_data["clusters"] if cluster["name"] == "foo")
    foo_cluster["certificate_authority"] = "/tmp/foo-ca.pem"
    with open(oauth_config_file, "w") as f:
        yaml.safe_dump(config_data, f)

    discover_calls: list[tuple[str, str | None]] = []
    password_grant_calls: list[dict] = []

    def discover_config(
        url: str, timeout: float = 10.0, *, certificate_authority: str | None = None
    ) -> SimpleNamespace:
        discover_calls.append((url, certificate_authority))
        return _mock_oidc_config()

    def password_grant(**kwargs) -> SimpleNamespace:
        password_grant_calls.append(kwargs)
        return SimpleNamespace(token_for_nmp="foo-access-token", refresh_token="foo-refresh-token")

    monkeypatch.setattr("nemo_platform_ext.cli.commands.auth.discover_nmp_config", discover_config)
    monkeypatch.setattr("nemo_platform_ext.auth.device_flow.authenticate_with_password_grant", password_grant)
    monkeypatch.setattr("nemo_platform_ext.cli.commands.auth.decode_jwt_claims", _decode_jwt_noop)

    result = runner.invoke(
        app,
        [
            "--context",
            "foo",
            "auth",
            "login",
            "--username",
            "user",
            "--password",
            "secret",
        ],
    )

    assert_exit_code(result, 0)
    assert discover_calls == [("https://foo.example.com", "/tmp/foo-ca.pem")]
    assert password_grant_calls[0]["certificate_authority"] == "/tmp/foo-ca.pem"


def test_auth_login_with_base_url_creates_selected_context(oauth_config_file: Path, monkeypatch: pytest.MonkeyPatch):
    def password_grant(**kwargs) -> SimpleNamespace:
        return SimpleNamespace(token_for_nmp="dev-access-token", refresh_token="dev-refresh-token")

    monkeypatch.setattr("nemo_platform_ext.cli.commands.auth.discover_nmp_config", _discover_oidc_config)
    monkeypatch.setattr("nemo_platform_ext.auth.device_flow.authenticate_with_password_grant", password_grant)
    monkeypatch.setattr("nemo_platform_ext.cli.commands.auth.decode_jwt_claims", _decode_jwt_noop)

    result = runner.invoke(
        app,
        [
            "--context",
            "dev",
            "auth",
            "login",
            "--base-url",
            "https://dev.example.com",
            "--username",
            "user",
            "--password",
            "secret",
        ],
    )

    assert_exit_code(result, 0)

    with open(oauth_config_file) as f:
        data = yaml.safe_load(f)

    dev_context = next(context for context in data["contexts"] if context["name"] == "dev")
    dev_cluster = next(cluster for cluster in data["clusters"] if cluster["name"] == dev_context["cluster"])
    dev_user = next(user for user in data["users"] if user["name"] == dev_context["user"])

    assert dev_cluster["base_url"].rstrip("/") == "https://dev.example.com"
    assert dev_user["token"] == "dev-access-token"
    assert dev_user["refresh_token"] == "dev-refresh-token"


def test_auth_login_unsigned_token_writes_to_selected_context(
    oauth_config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("nemo_platform_ext.cli.commands.auth.discover_nmp_config", _discover_no_oidc)

    result = runner.invoke(
        app,
        [
            "--context",
            "foo",
            "auth",
            "login",
            "--unsigned-token",
            "--email",
            "admin@example.com",
            "--expires-in",
            "1000",
        ],
    )

    assert_exit_code(result, 0)
    with open(oauth_config_file) as f:
        data = yaml.safe_load(f)

    foo_user = next(user for user in data["users"] if user["name"] == "foo")
    assert foo_user["type"] == "oauth"
    assert data["current_context"] == "foo"
    claims = decode_jwt_claims(foo_user["token"])
    assert claims["sub"] == "admin@example.com"
    assert claims["email"] == "admin@example.com"
    assert claims["exp"] - claims["iat"] == 1000


def test_auth_login_unsigned_token_requires_email() -> None:
    result = runner.invoke(app, ["auth", "login", "--unsigned-token"])

    assert_exit_code(result, 1)
    assert "--email is required" in result.output


def test_auth_login_unsigned_options_require_unsigned_token() -> None:
    result = runner.invoke(app, ["auth", "login", "--email", "admin@example.com"])

    assert_exit_code(result, 1)
    assert "Unsigned token option(s) --email" in result.output
    assert "--unsigned-token" in result.output


def test_auth_login_unsigned_token_fails_when_oidc_enabled(
    oauth_config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def discover_full_oidc(
        url: str, timeout: float = 10.0, *, certificate_authority: str | None = None
    ) -> SimpleNamespace:
        return SimpleNamespace(
            auth_enabled=True,
            issuer="https://idp.example.com",
            client_id="nmp-client",
            token_endpoint="https://idp.example.com/token",
            device_authorization_endpoint="https://idp.example.com/device",
        )

    monkeypatch.setattr("nemo_platform_ext.cli.commands.auth.discover_nmp_config", discover_full_oidc)

    result = runner.invoke(
        app,
        [
            "--context",
            "foo",
            "auth",
            "login",
            "--unsigned-token",
            "--email",
            "admin@example.com",
        ],
    )

    assert_exit_code(result, 1)
    assert "Cluster has OIDC authentication configured" in result.output


def test_auth_login_unsigned_token_allows_partial_oidc_config(
    oauth_config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def discover_partial_oidc(
        url: str, timeout: float = 10.0, *, certificate_authority: str | None = None
    ) -> SimpleNamespace:
        return SimpleNamespace(
            auth_enabled=True,
            issuer="https://idp.example.com",
            client_id=None,
            token_endpoint=None,
            device_authorization_endpoint=None,
        )

    monkeypatch.setattr("nemo_platform_ext.cli.commands.auth.discover_nmp_config", discover_partial_oidc)

    result = runner.invoke(
        app,
        [
            "--context",
            "foo",
            "auth",
            "login",
            "--unsigned-token",
            "--email",
            "admin@example.com",
        ],
    )

    assert_exit_code(result, 0)


def test_auth_login_unsigned_token_uses_principal_id_when_provided(
    oauth_config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("nemo_platform_ext.cli.commands.auth.discover_nmp_config", _discover_no_oidc)

    result = runner.invoke(
        app,
        [
            "--context",
            "foo",
            "auth",
            "login",
            "--unsigned-token",
            "--principal-id",
            "principal-123",
            "--email",
            "admin@example.com",
        ],
    )

    assert_exit_code(result, 0)
    with open(oauth_config_file) as f:
        data = yaml.safe_load(f)

    foo_user = next(user for user in data["users"] if user["name"] == "foo")
    claims = decode_jwt_claims(foo_user["token"])
    assert claims["sub"] == "principal-123"
    assert claims["email"] == "admin@example.com"


# ---------------------------------------------------------------------------
# is_auth_disabled helper
# ---------------------------------------------------------------------------


@dataclass
class IsAuthDisabledCase:
    id: str
    auth_enabled: bool
    expected: bool


@pytest.mark.parametrize(
    "case",
    [
        IsAuthDisabledCase(id="disabled", auth_enabled=False, expected=True),
        IsAuthDisabledCase(id="enabled", auth_enabled=True, expected=False),
    ],
    ids=lambda c: c.id,
)
def test_is_auth_disabled(monkeypatch: pytest.MonkeyPatch, case: IsAuthDisabledCase) -> None:
    def mock_discover(url: str, timeout: float = 10.0, *, certificate_authority: str | None = None) -> SimpleNamespace:
        return SimpleNamespace(auth_enabled=case.auth_enabled)

    monkeypatch.setattr("nemo_platform_ext.cli.commands.auth.discover_nmp_config", mock_discover)
    from nemo_platform_ext.cli.commands.auth import is_auth_disabled

    assert is_auth_disabled("http://localhost:8080") is case.expected


def test_is_auth_disabled_raises_when_discovery_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx
    from nemo_platform_ext.auth.helpers import AuthError
    from nemo_platform_ext.cli.commands.auth import is_auth_disabled

    def raise_connect_error(
        url: str, timeout: float = 10.0, *, certificate_authority: str | None = None
    ) -> SimpleNamespace:
        raise httpx.ConnectError("Connection refused")

    monkeypatch.setattr("nemo_platform_ext.cli.commands.auth.discover_nmp_config", raise_connect_error)

    with pytest.raises(AuthError, match="Failed to discover auth configuration: Connection refused"):
        is_auth_disabled("http://localhost:8080")


# ---------------------------------------------------------------------------
# auth status when auth disabled
# ---------------------------------------------------------------------------


def test_auth_status_when_auth_disabled_shows_message(oauth_config_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nemo_platform_ext.cli.commands.auth.discover_nmp_config", _discover_auth_disabled)
    result = runner.invoke(app, ["--context", "foo", "auth", "status"])

    assert_exit_code(result, 0)
    assert "Authentication is disabled" in result.output


def test_auth_status_when_auth_disabled_does_not_show_token_details(
    oauth_config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("nemo_platform_ext.cli.commands.auth.discover_nmp_config", _discover_auth_disabled)
    result = runner.invoke(app, ["--context", "foo", "auth", "status"])

    assert_exit_code(result, 0)
    assert "foo-token" not in result.output
    assert "Refresh Token" not in result.output


def test_auth_status_when_cluster_unreachable_shows_local_state(
    oauth_config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import httpx

    def raise_connect_error(
        url: str, timeout: float = 10.0, *, certificate_authority: str | None = None
    ) -> SimpleNamespace:
        raise httpx.ConnectError("Connection refused")

    monkeypatch.setattr("nemo_platform_ext.cli.commands.auth.discover_nmp_config", raise_connect_error)
    result = runner.invoke(app, ["--context", "foo", "auth", "status"])
    output = " ".join(result.output.split())

    assert_exit_code(result, 0)
    assert "Failed to discover auth configuration:" in output
    assert "Connection refused" in output
    assert "Auth Discovery" in result.output
    assert "unavailable" in result.output
    assert "Auth Type" in result.output
    assert "oauth" in result.output
    assert "Credential Source" in result.output
    assert "config file" in result.output
    assert "Refresh Token" in result.output
    assert "foo-token" not in result.output


# ---------------------------------------------------------------------------
# auth logout when auth disabled
# ---------------------------------------------------------------------------


def test_auth_logout_when_auth_disabled_shows_message_and_skips_credential_clear(
    oauth_config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("nemo_platform_ext.cli.commands.auth.discover_nmp_config", _discover_auth_disabled)
    with patch("nemo_platform_ext.config.config.Config.write") as mock_write:
        result = runner.invoke(app, ["--context", "foo", "auth", "logout"])

    assert_exit_code(result, 0)
    assert "disabled" in result.output
    mock_write.assert_not_called()


def test_auth_logout_when_cluster_unreachable_clears_local_credentials(
    oauth_config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import httpx

    def raise_connect_error(
        url: str, timeout: float = 10.0, *, certificate_authority: str | None = None
    ) -> SimpleNamespace:
        raise httpx.ConnectError("Connection refused")

    monkeypatch.setattr("nemo_platform_ext.cli.commands.auth.discover_nmp_config", raise_connect_error)
    result = runner.invoke(app, ["--context", "foo", "auth", "logout"])
    output = " ".join(result.output.split())

    assert_exit_code(result, 0)
    assert "Failed to discover auth configuration: Connection refused" in output
    assert "continuing to clear local credentials" in output
    assert "Logged out successfully" in result.output

    with open(oauth_config_file) as f:
        data = yaml.safe_load(f)

    default_user = next(user for user in data["users"] if user["name"] == "default")
    foo_user = next(user for user in data["users"] if user["name"] == "foo")

    assert default_user["type"] == "oauth"
    assert default_user["token"] == "default-token"
    assert default_user["refresh_token"] == "default-refresh"
    assert foo_user["type"] == "no-auth"
    assert "token" not in foo_user
    assert "refresh_token" not in foo_user
