# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import httpx
import pytest
from nemo_helix_ext.auth.helpers import select_advertised_client

from tests.auth_idp.common import discover_runtime_nhx_config, require_capability, runtime_tls_config
from tests.auth_idp.runtime_contract import AuthIdpCase, AuthIdpRuntime

pytestmark = [pytest.mark.auth_idp, pytest.mark.auth_idp_runtime, pytest.mark.e2e, pytest.mark.xdist_group("idp-live")]


def _login_pair(case: AuthIdpCase, runtime: AuthIdpRuntime, scope: str) -> tuple[str, str]:
    require_capability(case, "device_flow")
    require_capability(case, "gateway_authn")
    public_client = select_advertised_client(discover_runtime_nhx_config(runtime), "public")
    assert public_client.bearer_token_source == "access_token"
    assert public_client.device_authorization_endpoint
    assert public_client.token_endpoint
    tokens = runtime.authenticate_device_flow(
        device_authorization_endpoint=public_client.device_authorization_endpoint,
        token_endpoint=public_client.token_endpoint,
        client_id=public_client.client_id,
        scope=scope,
        username=case.provider.interactive_user_username,
        password=case.provider.interactive_user_password,
        tls_config=runtime_tls_config(runtime),
    )
    access_token = tokens.get("access_token")
    id_token = tokens.get("id_token")
    assert isinstance(access_token, str) and access_token
    assert isinstance(id_token, str) and id_token
    return access_token, id_token


def test_access_token_profile_rejects_provider_id_bearer(
    auth_idp_case: AuthIdpCase, auth_idp_runtime: AuthIdpRuntime
) -> None:
    access_token, id_token = _login_pair(auth_idp_case, auth_idp_runtime, "openid email groups")
    gateway = auth_idp_runtime.gateway_base_url.rstrip("/")
    with httpx.Client(timeout=30.0, **runtime_tls_config(auth_idp_runtime)) as client:
        access_identity = client.get(
            f"{gateway}/apis/auth/authenticate", headers={"Authorization": f"Bearer {access_token}"}
        )
        id_identity = client.get(f"{gateway}/apis/auth/authenticate", headers={"Authorization": f"Bearer {id_token}"})
    assert access_identity.status_code == 200, access_identity.text
    assert id_identity.status_code == 401, id_identity.text
