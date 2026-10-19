# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import uuid

from fastapi.testclient import TestClient
from nemo_helix_ext.auth.helpers import generate_unsigned_jwt
from nhx.common.config import AuthConfig
from nhx.testing.client import create_test_client

SERVICE_PRINCIPAL = "service:integration-test"
WORKSPACES_PATH = "/apis/entities/v2/workspaces"
IAM_ROLE_BINDINGS_PATH = "/apis/auth/v2/iam/role-bindings"


def _machine_headers(*, principal_id: str, email: str, groups: list[str]) -> dict[str, str]:
    token = generate_unsigned_jwt(principal_id=principal_id, email=email, groups=groups)
    return {"Authorization": f"Bearer {token}"}


def test_external_machine_identity_group_binding_grants_workspace_access():
    workspace_id = f"machine-ws-{uuid.uuid4().hex[:8]}"
    group_name = f"machine-group-{uuid.uuid4().hex[:8]}"
    machine_principal_id = f"machine-{uuid.uuid4().hex[:8]}"
    assert not machine_principal_id.startswith("service:")
    machine_headers = _machine_headers(
        principal_id=machine_principal_id,
        email=f"{machine_principal_id}@example.com",
        groups=[group_name],
    )
    service_headers = {"X-NHX-Principal-Id": SERVICE_PRINCIPAL}

    with create_test_client(
        client_type=TestClient,
        auth_enabled=True,
        service_configs={
            AuthConfig: AuthConfig(
                enabled=True,
                allow_unsigned_jwt=True,
                policy_decision_point_provider="embedded",
                policy_decision_point_base_url="http://testserver",
                propagation_poll_interval_seconds=0.05,
            )
        },
    ) as client:
        response = client.post(
            WORKSPACES_PATH,
            json={"name": workspace_id, "description": "Workspace for external machine identity auth"},
            headers=service_headers,
        )
        assert response.status_code in (200, 201), f"Failed to create workspace: {response.text}"

        try:
            denied = client.get(
                f"{WORKSPACES_PATH}/{workspace_id}",
                headers=_machine_headers(
                    principal_id=machine_principal_id,
                    email=f"{machine_principal_id}@example.com",
                    groups=[],
                ),
            )
            assert denied.status_code == 403, (
                f"Machine identity without the bound group should be denied. Got {denied.status_code}: {denied.text}"
            )

            response = client.post(
                f"{IAM_ROLE_BINDINGS_PATH}?wait_role_propagation=true",
                json={"principal": group_name, "role": "Viewer", "workspace": workspace_id},
                headers=service_headers,
            )
            assert response.status_code in (200, 201), f"Failed to create role binding: {response.text}"

            allowed = client.get(f"{WORKSPACES_PATH}/{workspace_id}", headers=machine_headers)
            assert allowed.status_code == 200, (
                f"Machine identity with the bound group should be allowed. Got {allowed.status_code}: {allowed.text}"
            )

        finally:
            client.delete(f"{WORKSPACES_PATH}/{workspace_id}", headers=service_headers)
