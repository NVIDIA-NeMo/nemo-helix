# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import httpx
import pytest
from nemo_helix_ext.auth.helpers import select_advertised_client
from nemo_helix_ext.config.config import Config
from nemo_helix_ext.config.models import OAuthUser
from nemo_helix_plugin.client.oidc import OidcClientName

from tests.auth_idp.cli_flow import CLI_CONTEXT_NAME, login_with_real_cli
from tests.auth_idp.common import discover_runtime_nhx_config, require_capability, runtime_tls_config

pytestmark = [
    pytest.mark.auth_idp,
    pytest.mark.auth_idp_runtime,
    pytest.mark.e2e,
    pytest.mark.xdist_group("idp-live"),
]


@pytest.mark.parametrize("client_name", ["public", "confidential"])
def test_real_cli_login_authenticated_command_and_refresh(
    auth_idp_case,
    auth_idp_runtime,
    tmp_path: Path,
    client_name: OidcClientName,
) -> None:
    require_capability(auth_idp_case, "device_flow" if client_name == "public" else "confidential_oidc")
    config_path = tmp_path / f"{client_name}.yaml"
    session = login_with_real_cli(
        case=auth_idp_case,
        runtime=auth_idp_runtime,
        config_path=config_path,
        client_name=client_name,
    )
    session.run("--context", CLI_CONTEXT_NAME, "--output-format", "json", "workspaces", "list")

    before = _oauth_user(config_path)
    access_token = before.token.get_secret_value()
    assert before.refresh_token is not None
    refresh_token = before.refresh_token.get_secret_value()

    if client_name == "confidential":
        assert before.token_broker_url
        tls_config = runtime_tls_config(auth_idp_runtime)
        oidc = discover_runtime_nhx_config(auth_idp_runtime)
        public_client = select_advertised_client(oidc, "public")
        assert auth_idp_runtime.token_endpoint is not None
        direct_refresh = httpx.post(
            auth_idp_runtime.token_endpoint,
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": public_client.client_id,
            },
            timeout=30.0,
            **tls_config,
        )
        assert direct_refresh.is_error, "broker refresh handle was accepted by the provider token endpoint"

    session.run("--context", CLI_CONTEXT_NAME, "auth", "refresh")
    after = _oauth_user(config_path)
    assert after.token.get_secret_value() != access_token
    assert after.refresh_token is not None
    session.run("--context", CLI_CONTEXT_NAME, "--output-format", "json", "workspaces", "list")


def _oauth_user(config_path: Path) -> OAuthUser:
    user = Config.load(config_path=config_path).resolve().user
    assert isinstance(user, OAuthUser)
    return user
