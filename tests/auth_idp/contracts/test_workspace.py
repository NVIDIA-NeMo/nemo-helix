# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import httpx
import pytest
from nemo_helix_ext.auth.helpers import discover_nhx_config
from nemo_helix_plugin.client.errors import PermissionDeniedError
from nemo_helix_plugin.workspaces.client import WorkspacesClient

from tests.auth_idp.common import jwt_claims, require_capability, runtime_tls_config
from tests.auth_idp.device_flow import authenticate_authentik_device_flow, with_url_origin
from tests.auth_idp.helpers import grant_workspace_role
from tests.auth_idp.runtime_contract import AuthIdpCase, AuthIdpRuntime, TokenSet

pytestmark = [
    pytest.mark.auth_idp,
    pytest.mark.auth_idp_runtime,
    pytest.mark.e2e,
]

REQUEST_TIMEOUT_SECONDS = 10.0


def _retrieve_workspace_with_token(auth_idp_runtime: AuthIdpRuntime, token: str, workspace: str) -> httpx.Response:
    return httpx.get(
        f"{auth_idp_runtime.gateway_base_url}/apis/entities/v2/workspaces/{workspace}",
        headers={"Authorization": f"Bearer {token}"},
        timeout=REQUEST_TIMEOUT_SECONDS,
        **runtime_tls_config(auth_idp_runtime),
    )


def _workload_platform_token_for_workspace_rbac(
    auth_idp_case: AuthIdpCase,
    auth_idp_runtime: AuthIdpRuntime,
) -> TokenSet:
    require_capability(auth_idp_case, "workload_provider_token")
    require_capability(auth_idp_case, "workload_subject_token")
    require_capability(auth_idp_case, "workload_token_exchange")
    return auth_idp_runtime.workload_platform_token()


def _interactive_user_access_token(auth_idp_case: AuthIdpCase, auth_idp_runtime: AuthIdpRuntime) -> str:
    require_capability(auth_idp_case, "device_flow")

    oidc = discover_nhx_config(auth_idp_runtime.gateway_base_url)
    assert oidc.client_id
    assert oidc.device_authorization_endpoint
    assert oidc.token_endpoint
    tls_config = runtime_tls_config(auth_idp_runtime)
    token_response = authenticate_authentik_device_flow(
        gateway_base_url=auth_idp_runtime.gateway_base_url,
        device_authorization_endpoint=with_url_origin(
            oidc.device_authorization_endpoint,
            auth_idp_runtime.gateway_base_url,
        ),
        token_endpoint=with_url_origin(oidc.token_endpoint, auth_idp_runtime.gateway_base_url),
        client_id=oidc.client_id,
        scope=oidc.default_scopes,
        username=auth_idp_case.provider.interactive_user_username,
        password=auth_idp_case.provider.interactive_user_password,
        tls_config=tls_config,
    )
    access_token = token_response.get("access_token")
    assert isinstance(access_token, str)
    return access_token


def test_provider_workload_identity_is_denied_before_binding(
    auth_idp_case,
    auth_idp_runtime,
    auth_idp_workspace,
):
    require_capability(auth_idp_case, "workspace_rbac")

    with pytest.raises(PermissionDeniedError):
        WorkspacesClient.from_client(auth_idp_runtime.workload_provider_client()).get_workspace(name=auth_idp_workspace)


def test_provider_workload_identity_is_allowed_after_binding(
    auth_idp_case,
    auth_idp_runtime,
    auth_idp_workspace,
):
    require_capability(auth_idp_case, "workspace_rbac")

    e2e_setup_client = auth_idp_runtime.e2e_setup_client()
    for principal in auth_idp_runtime.workload_role_principals():
        grant_workspace_role(e2e_setup_client, workspace=auth_idp_workspace, principal=principal, roles=["Viewer"])

    retrieved = (
        WorkspacesClient.from_client(auth_idp_runtime.workload_provider_client())
        .get_workspace(name=auth_idp_workspace)
        .data()
    )
    assert retrieved.name == auth_idp_workspace


def test_provider_workload_identity_is_allowed_by_subject_alias_binding(
    auth_idp_case,
    auth_idp_runtime,
    auth_idp_workspace,
):
    require_capability(auth_idp_case, "workspace_rbac")

    workload_token = _workload_platform_token_for_workspace_rbac(auth_idp_case, auth_idp_runtime)
    subject = workload_token.claims.get("sub")
    assert isinstance(subject, str)

    e2e_setup_client = auth_idp_runtime.e2e_setup_client()
    grant_workspace_role(e2e_setup_client, workspace=auth_idp_workspace, principal=subject, roles=["Viewer"])

    response = _retrieve_workspace_with_token(auth_idp_runtime, workload_token.access_token, auth_idp_workspace)

    assert response.status_code == 200, response.text
    assert response.json()["name"] == auth_idp_workspace


def test_provider_interactive_user_is_allowed_by_email_alias_binding(
    auth_idp_case,
    auth_idp_runtime,
    auth_idp_workspace,
):
    require_capability(auth_idp_case, "workspace_rbac")

    access_token = _interactive_user_access_token(auth_idp_case, auth_idp_runtime)
    token_claims = jwt_claims(access_token)
    email = token_claims.get("email")
    assert isinstance(email, str)
    assert email == auth_idp_case.provider.interactive_user_expected_email

    e2e_setup_client = auth_idp_runtime.e2e_setup_client()
    grant_workspace_role(e2e_setup_client, workspace=auth_idp_workspace, principal=email, roles=["Viewer"])

    response = _retrieve_workspace_with_token(auth_idp_runtime, access_token, auth_idp_workspace)

    assert response.status_code == 200, response.text
    assert response.json()["name"] == auth_idp_workspace


def test_provider_workload_identity_returns_to_denied_after_revoke(
    auth_idp_case,
    auth_idp_runtime,
    auth_idp_workspace,
):
    require_capability(auth_idp_case, "workspace_rbac")

    e2e_setup_client = auth_idp_runtime.e2e_setup_client()
    for principal in auth_idp_runtime.workload_role_principals():
        grant_workspace_role(e2e_setup_client, workspace=auth_idp_workspace, principal=principal, roles=["Viewer"])
        WorkspacesClient.from_client(e2e_setup_client).delete_workspace_member(
            workspace=auth_idp_workspace,
            principal_id=principal,
            query_params={"wait_role_propagation": True},
        )

    with pytest.raises(PermissionDeniedError):
        WorkspacesClient.from_client(auth_idp_runtime.workload_provider_client()).get_workspace(name=auth_idp_workspace)
