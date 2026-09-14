# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
import uuid

import httpx
import pytest
from nemo_platform_ext.client.tls import HttpxTLSConfig
from nmp.testing import grant_workspace_role

from tests.auth_idp.common import jwt_claims, require_capability, runtime_tls_config

pytestmark = [
    pytest.mark.auth_idp,
    pytest.mark.auth_idp_runtime,
    pytest.mark.e2e,
    pytest.mark.xdist_group("idp-live"),
]

GATEWAY_REQUEST_TIMEOUT_SECONDS = 10.0
GATEWAY_TRANSIENT_RETRY_TIMEOUT_SECONDS = 20.0
GATEWAY_TRANSIENT_RETRY_SLEEP_SECONDS = 1.0
GATEWAY_TRANSIENT_STATUS_CODES = {502, 503, 504}


def _gateway_get_with_transient_retries(
    url: str,
    *,
    headers: dict[str, str],
    tls_config: HttpxTLSConfig,
) -> httpx.Response:
    deadline = time.monotonic() + GATEWAY_TRANSIENT_RETRY_TIMEOUT_SECONDS
    last_transient_response: httpx.Response | None = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            if last_transient_response is not None:
                return last_transient_response
            raise TimeoutError(f"gateway transient retry deadline reached before request: {url}")

        response = httpx.get(
            url,
            headers=headers,
            timeout=min(GATEWAY_REQUEST_TIMEOUT_SECONDS, remaining),
            **tls_config,
        )
        if response.status_code not in GATEWAY_TRANSIENT_STATUS_CODES:
            return response

        last_transient_response = response
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return response
        time.sleep(min(GATEWAY_TRANSIENT_RETRY_SLEEP_SECONDS, remaining))


def test_provider_gateway_rejects_unauthenticated_requests(auth_idp_case, auth_idp_runtime):
    require_capability(auth_idp_case, "gateway_authn")

    tls_config = runtime_tls_config(auth_idp_runtime)
    response = httpx.get(
        f"{auth_idp_runtime.gateway_base_url}/apis/entities/v2/workspaces",
        timeout=10.0,
        **tls_config,
    )

    assert response.status_code in {401, 403}


def test_provider_gateway_accepts_e2e_setup_token(auth_idp_case, auth_idp_runtime, auth_idp_workspace):
    require_capability(auth_idp_case, "gateway_authn")
    require_capability(auth_idp_case, "workspace_rbac")

    token = auth_idp_runtime.e2e_setup_token().access_token
    tls_config = runtime_tls_config(auth_idp_runtime)
    response = _gateway_get_with_transient_retries(
        f"{auth_idp_runtime.gateway_base_url}/apis/entities/v2/workspaces/{auth_idp_workspace}",
        headers={"Authorization": f"Bearer {token}"},
        tls_config=tls_config,
    )

    assert 200 <= response.status_code < 300, response.text
    assert response.json()["name"] == auth_idp_workspace


def test_provider_gateway_rejects_spoofed_principal_headers(auth_idp_case, auth_idp_runtime):
    require_capability(auth_idp_case, "spoofed_header_rejection")
    require_capability(auth_idp_case, "workload_provider_token")

    workload_provider_token = auth_idp_runtime.workload_provider_token()
    workspace_name = f"spoof-check-{uuid.uuid4().hex[:8]}"
    authenticated_principal_id = str(workload_provider_token.claims["sub"])
    headers = {
        "Authorization": f"Bearer {workload_provider_token.access_token}",
        "X-NMP-Principal-Id": "service:bootstrap",
        "X-NMP-Principal-Email": "attacker@example.com",
    }
    tls_config = runtime_tls_config(auth_idp_runtime)

    try:
        create_response = httpx.post(
            f"{auth_idp_runtime.gateway_base_url}/apis/entities/v2/workspaces",
            json={"name": workspace_name, "description": "Spoofed header check"},
            headers=headers,
            timeout=GATEWAY_REQUEST_TIMEOUT_SECONDS,
            **tls_config,
        )
        create_response.raise_for_status()
        assert create_response.json()["created_by"] == authenticated_principal_id

        members_response = httpx.get(
            f"{auth_idp_runtime.gateway_base_url}/apis/entities/v2/workspaces/{workspace_name}/members",
            headers=headers,
            timeout=GATEWAY_REQUEST_TIMEOUT_SECONDS,
            **tls_config,
        )
        members_response.raise_for_status()
        admin_member = next(member for member in members_response.json()["data"] if "Admin" in member["roles"])

        assert admin_member["granted_by"] == authenticated_principal_id
        assert admin_member["principal"] == authenticated_principal_id
        assert admin_member["principal"] not in {"service:bootstrap", "attacker@example.com"}
    finally:
        httpx.delete(
            f"{auth_idp_runtime.gateway_base_url}/apis/entities/v2/workspaces/{workspace_name}",
            headers=headers,
            timeout=GATEWAY_REQUEST_TIMEOUT_SECONDS,
            **tls_config,
        )


def test_provider_gateway_forwards_workload_groups(
    auth_idp_case,
    auth_idp_runtime,
    auth_idp_workspace,
):
    require_capability(auth_idp_case, "workspace_rbac")
    require_capability(auth_idp_case, "workload_token_exchange")

    workload_token = auth_idp_runtime.exchange_workload_token(auth_idp_runtime.workload_subject_token()).access_token
    claims = jwt_claims(workload_token)
    claim_groups = claims.get("groups")
    assert isinstance(claim_groups, (list, str))
    token_groups = (
        set(claim_groups)
        if isinstance(claim_groups, list)
        else {group.strip() for group in claim_groups.split(",") if group.strip()}
    )
    role_principals = auth_idp_runtime.workload_role_principals()
    assert set(role_principals).intersection(token_groups)

    e2e_setup_sdk = auth_idp_runtime.e2e_setup_sdk()
    for principal in role_principals:
        grant_workspace_role(e2e_setup_sdk, workspace=auth_idp_workspace, principal=principal, roles=["Viewer"])

    tls_config = runtime_tls_config(auth_idp_runtime)
    response = _gateway_get_with_transient_retries(
        f"{auth_idp_runtime.gateway_base_url}/apis/entities/v2/workspaces/{auth_idp_workspace}",
        headers={"Authorization": f"Bearer {workload_token}"},
        tls_config=tls_config,
    )

    assert response.status_code == 200, response.text
    assert response.json()["name"] == auth_idp_workspace
