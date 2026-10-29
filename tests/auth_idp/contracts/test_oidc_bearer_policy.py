# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import uuid

import httpx
import pytest
from nemo_helix_ext.auth.helpers import select_advertised_client
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.workspaces.client import WorkspacesClient

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


def test_provider_read_scope_still_denies_writes(auth_idp_case: AuthIdpCase, auth_idp_runtime: AuthIdpRuntime) -> None:
    if auth_idp_case.provider.name != "authentik":
        pytest.skip("The platform:read provider mapping is currently configured only for Authentik")
    require_capability(auth_idp_case, "workspace_rbac")
    baseline_access, _ = _login_pair(auth_idp_case, auth_idp_runtime, "openid email groups")
    read_access, _ = _login_pair(auth_idp_case, auth_idp_runtime, "openid email groups platform:read")
    gateway = auth_idp_runtime.gateway_base_url.rstrip("/")
    created: list[str] = []
    try:
        with httpx.Client(timeout=30.0, **runtime_tls_config(auth_idp_runtime)) as client:
            identity = client.get(
                f"{gateway}/apis/auth/authenticate", headers={"Authorization": f"Bearer {read_access}"}
            )
            assert identity.status_code == 200, identity.text
            assert "platform:read" in identity.json()["scopes"], "Provider fixture must actually grant platform:read"
            for token, expected in ((baseline_access, 201), (read_access, 403)):
                name = f"scope-policy-{uuid.uuid4().hex[:10]}"
                response = client.post(
                    f"{gateway}/apis/entities/v2/workspaces",
                    params={"wait_role_propagation": "true"},
                    headers={"Authorization": f"Bearer {token}"},
                    json={"name": name},
                )
                if response.status_code == 201:
                    created.append(name)
                assert response.status_code == expected, response.text
    finally:
        with NemoClient(
            base_url=gateway,
            auth=baseline_access,
            timeout=30.0,
            http_client=httpx.Client(**runtime_tls_config(auth_idp_runtime)),
            owns_http_client=True,
        ) as cleanup_client:
            workspaces_client = WorkspacesClient.from_client(cleanup_client)
            for name in reversed(created):
                workspaces_client.delete_workspace(name=name).data()
