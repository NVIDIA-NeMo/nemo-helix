# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import httpx
import pytest
from nemo_helix_ext.auth.helpers import discover_nhx_config
from nemo_helix_ext.client.tls import httpx_tls_config_from_env

from tests.auth_idp.authentik_live import AUTHENTIK_DOCKER_E2E_CONFIG

pytestmark = [
    pytest.mark.auth_idp,
    pytest.mark.auth_idp_docker,
    pytest.mark.e2e,
    AUTHENTIK_DOCKER_E2E_CONFIG,
    pytest.mark.xdist_group("idp-live"),
]


def test_authentik_discovery_exposes_gateway_reachable_device_flow(authentik_stack):
    oidc = discover_nhx_config(authentik_stack.gateway_base_url)

    assert oidc.auth_enabled is True
    assert oidc.client_id == "nemo-helix-cli"
    assert oidc.token_endpoint == f"{authentik_stack.gateway_base_url}/application/o/token/"
    assert oidc.device_authorization_endpoint == f"{authentik_stack.gateway_base_url}/application/o/device/"
    assert oidc.default_scopes == "openid email offline_access groups"

    response = httpx.post(
        oidc.device_authorization_endpoint,
        data={
            "client_id": oidc.client_id,
            "scope": oidc.default_scopes,
        },
        timeout=30.0,
        **httpx_tls_config_from_env(),
    )
    response.raise_for_status()
    body = response.json()

    assert body["verification_uri"] == f"{authentik_stack.gateway_base_url}/device"
    assert body["verification_uri_complete"].startswith(f"{authentik_stack.gateway_base_url}/device?code=")
    assert body["device_code"]
    assert body["user_code"]


def test_authentik_cli_provider_rejects_unseeded_human_app_password(authentik_stack):
    token_response = httpx.post(
        authentik_stack.token_endpoint,
        data={
            "grant_type": "password",
            "client_id": "nemo-helix-cli",
            "username": "nemo-user",
            "password": "nemo-user-token-secret-dev",
            "scope": "openid email offline_access groups",
        },
        timeout=30.0,
        **httpx_tls_config_from_env(),
    )

    assert token_response.status_code == 400
    assert token_response.json()["error"] == "invalid_grant"
