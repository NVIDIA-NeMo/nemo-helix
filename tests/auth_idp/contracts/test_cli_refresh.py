# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from nemo_platform_ext.auth.helpers import decode_jwt_claims, discover_nmp_config, generate_unsigned_jwt
from nemo_platform_ext.cli.app import app
from nemo_platform_ext.client.tls import NMP_CLIENT_SSL_CERT_FILE_ENVVAR
from nemo_platform_plugin.client.constants import WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR
from typer.testing import CliRunner

from tests.auth_idp.common import require_capability, runtime_tls_config
from tests.auth_idp.device_flow import authenticate_authentik_device_flow, with_url_origin
from tests.auth_idp.runtime_contract import JsonObject

pytestmark = [
    pytest.mark.auth_idp,
    pytest.mark.auth_idp_runtime,
    pytest.mark.e2e,
    pytest.mark.xdist_group("idp-live"),
]

CLI_REFRESH_CONTEXT_NAME = "auth-idp-refresh"


def _write_cli_config(
    config_path: Path,
    *,
    base_url: str,
    access_token: str,
    refresh_token: str,
) -> None:
    config = {
        "current_context": CLI_REFRESH_CONTEXT_NAME,
        "clusters": [
            {
                "name": "auth-idp",
                "base_url": base_url,
            }
        ],
        "users": [
            {
                "name": "interactive-user",
                "type": "oauth",
                "token": access_token,
                "refresh_token": refresh_token,
            }
        ],
        "contexts": [
            {
                "name": CLI_REFRESH_CONTEXT_NAME,
                "cluster": "auth-idp",
                "user": "interactive-user",
                "workspace": "default",
                "preferences": {
                    "output_format": "json",
                    "timestamp_format": "iso8601",
                    "truncate": True,
                    "color_output": False,
                },
            }
        ],
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _read_config_user(config_path: Path) -> JsonObject:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert isinstance(config, dict)
    users = config["users"]
    assert isinstance(users, list)
    user = users[0]
    assert isinstance(user, dict)
    return user


def test_cli_api_command_auto_refreshes_expired_device_flow_token(
    auth_idp_case,
    auth_idp_runtime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    require_capability(auth_idp_case, "device_flow")
    require_capability(auth_idp_case, "gateway_authn")

    oidc = discover_nmp_config(auth_idp_runtime.gateway_base_url)
    assert oidc.client_id
    assert oidc.device_authorization_endpoint
    assert oidc.token_endpoint
    assert "offline_access" in oidc.default_scopes.split()

    tls_config = runtime_tls_config(auth_idp_runtime)
    runtime_device_authorization_endpoint = with_url_origin(
        oidc.device_authorization_endpoint,
        auth_idp_runtime.gateway_base_url,
    )
    runtime_token_endpoint = with_url_origin(oidc.token_endpoint, auth_idp_runtime.gateway_base_url)
    token_response = authenticate_authentik_device_flow(
        gateway_base_url=auth_idp_runtime.gateway_base_url,
        device_authorization_endpoint=runtime_device_authorization_endpoint,
        token_endpoint=runtime_token_endpoint,
        client_id=oidc.client_id,
        scope=oidc.default_scopes,
        username=auth_idp_case.provider.interactive_user_username,
        password=auth_idp_case.provider.interactive_user_password,
        tls_config=tls_config,
    )
    refresh_token = token_response.get("refresh_token")
    assert isinstance(refresh_token, str)
    assert refresh_token

    expired_access_token = generate_unsigned_jwt(
        auth_idp_case.provider.interactive_user_username,
        email=auth_idp_case.provider.interactive_user_expected_email,
        groups=auth_idp_case.provider.workload_expected_groups,
        scopes=oidc.default_scopes.split(),
        expires_in_seconds=-120,
    )
    config_path = tmp_path / "config.yaml"
    _write_cli_config(
        config_path,
        base_url=auth_idp_runtime.gateway_base_url,
        access_token=expired_access_token,
        refresh_token=refresh_token,
    )

    monkeypatch.setenv("NMP_CONFIG_FILE", str(config_path))
    monkeypatch.delenv("NMP_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv(WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR, raising=False)
    runtime_oidc = replace(
        oidc,
        device_authorization_endpoint=runtime_device_authorization_endpoint,
        token_endpoint=runtime_token_endpoint,
    )
    monkeypatch.setattr("nemo_platform.client.bootstrap.discover_nmp_config", lambda *_args, **_kwargs: runtime_oidc)
    monkeypatch.setattr(
        "nemo_platform_ext.client.bootstrap.discover_nmp_config",
        lambda *_args, **_kwargs: runtime_oidc,
    )

    cli_env = {"NMP_CONFIG_FILE": str(config_path)}
    if "verify" in tls_config:
        cli_env[NMP_CLIENT_SSL_CERT_FILE_ENVVAR] = tls_config["verify"]

    result = CliRunner().invoke(
        app,
        [
            "--context",
            CLI_REFRESH_CONTEXT_NAME,
            "--no-auto-refresh",
            "--output-format",
            "json",
            "workspaces",
            "list",
        ],
        env=cli_env,
    )

    assert result.exit_code == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "Jwt is expired" not in result.output

    saved_user = _read_config_user(config_path)
    refreshed_access_token = saved_user["token"]
    assert isinstance(refreshed_access_token, str)
    assert refreshed_access_token != expired_access_token

    claims = decode_jwt_claims(refreshed_access_token)
    assert claims.get("exp", 0) > time.time()
    assert saved_user.get("refresh_token")
