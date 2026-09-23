# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for role binding propagation to the embedded policy engine.

These tests verify the complete authorization flow:

1. Create a workspace
2. Create a role binding for a user
3. Wait for policy refresh
4. Verify the user can access the workspace with appropriate permissions
"""

import uuid
from typing import Generator

import pytest
from fastapi.testclient import TestClient
from nemo_helix_ext.auth.helpers import generate_unsigned_jwt
from nhx.common.config import AuthConfig
from nhx.testing.client import create_test_client

# Service principal for authenticated requests
SERVICE_PRINCIPAL = "service:integration-test"

# Platform API paths (entities at /apis/entities, auth IAM at /apis/auth)
WORKSPACES_PATH = "/apis/entities/v2/workspaces"
IAM_ROLE_BINDINGS_PATH = "/apis/auth/v2/iam/role-bindings"

# test_client is class-scoped (expensive to boot): keep these tests on one xdist worker so
# they share it instead of each worker re-provisioning it from scratch.
pytestmark = pytest.mark.xdist_group("auth_role_binding_propagation")


@pytest.fixture(scope="class")
def test_client() -> Generator[TestClient, None, None]:
    """TestClient with fast policy refresh for propagation tests.

    Overrides the default test_client with much faster refresh settings
    so propagation tests don't have to wait long.
    """
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
        yield client


class TestRoleBindingPropagation:
    """Tests that verify role bindings propagate to the embedded policy engine."""

    def test_group_role_binding_grants_workspace_access_via_bearer_token(self, test_client: TestClient):
        """A bearer token group claim can satisfy a group-based role binding."""
        workspace_id = f"test-ws-{uuid.uuid4().hex[:8]}"
        group_name = f"group-{uuid.uuid4().hex[:8]}"
        member_email = f"member-{uuid.uuid4().hex[:8]}@example.com"
        service_headers = {"X-NHX-Principal-Id": SERVICE_PRINCIPAL}
        bearer_token = generate_unsigned_jwt(
            principal_id=f"subject-{uuid.uuid4().hex[:8]}",
            email=member_email,
            groups=[group_name],
        )
        user_headers = {"Authorization": f"Bearer {bearer_token}"}

        response = test_client.post(
            WORKSPACES_PATH,
            json={"name": workspace_id, "description": "Test workspace for group role binding propagation"},
            headers=service_headers,
        )
        assert response.status_code in (200, 201), f"Failed to create workspace: {response.text}"

        try:
            response = test_client.post(
                IAM_ROLE_BINDINGS_PATH,
                json={
                    "principal": group_name,
                    "role": "Viewer",
                    "workspace": workspace_id,
                },
                headers=service_headers,
            )
            assert response.status_code in (200, 201), f"Failed to create role binding: {response.text}"

            response = test_client.get(f"{WORKSPACES_PATH}/{workspace_id}", headers=user_headers)
            assert response.status_code == 200, (
                f"Bearer token group member should be able to read workspace. "
                f"Got {response.status_code}: {response.text}"
            )

            response = test_client.put(
                f"{WORKSPACES_PATH}/{workspace_id}",
                json={"description": "Updated by viewer group - should fail"},
                headers=user_headers,
            )
            assert response.status_code == 403, (
                f"Viewer group member should not be able to update workspace: {response.text}"
            )

        finally:
            test_client.delete(f"{WORKSPACES_PATH}/{workspace_id}", headers=service_headers)

    def test_group_role_binding_denies_workspace_access_when_group_missing(self, test_client: TestClient):
        """A bearer token without the bound group claim cannot use a group-based role binding."""
        workspace_id = f"test-ws-{uuid.uuid4().hex[:8]}"
        bound_group = f"group-{uuid.uuid4().hex[:8]}"
        other_group = f"group-{uuid.uuid4().hex[:8]}"
        member_email = f"member-{uuid.uuid4().hex[:8]}@example.com"
        service_headers = {"X-NHX-Principal-Id": SERVICE_PRINCIPAL}
        bearer_token = generate_unsigned_jwt(
            principal_id=f"subject-{uuid.uuid4().hex[:8]}",
            email=member_email,
            groups=[other_group],
        )
        user_headers = {"Authorization": f"Bearer {bearer_token}"}

        response = test_client.post(
            WORKSPACES_PATH,
            json={"name": workspace_id, "description": "Test workspace for missing group role binding denial"},
            headers=service_headers,
        )
        assert response.status_code in (200, 201), f"Failed to create workspace: {response.text}"

        try:
            response = test_client.post(
                IAM_ROLE_BINDINGS_PATH,
                json={
                    "principal": bound_group,
                    "role": "Viewer",
                    "workspace": workspace_id,
                },
                headers=service_headers,
            )
            assert response.status_code in (200, 201), f"Failed to create role binding: {response.text}"

            response = test_client.get(f"{WORKSPACES_PATH}/{workspace_id}", headers=user_headers)
            assert response.status_code == 403, (
                f"Bearer token without the bound group should be denied workspace access. "
                f"Got {response.status_code}: {response.text}"
            )

        finally:
            test_client.delete(f"{WORKSPACES_PATH}/{workspace_id}", headers=service_headers)

    def test_editor_can_access_workspace_after_role_binding(self, test_client: TestClient):
        """Test that a user with Editor role can access a workspace after the binding propagates.

        This test reproduces the issue where:

        - Role binding is created successfully
        - User can list workspaces (sees the workspace)
        - But user gets Forbidden when accessing the specific workspace
        """
        # Test data
        workspace_id = f"test-ws-{uuid.uuid4().hex[:8]}"
        editor_email = f"editor-{uuid.uuid4().hex[:8]}@example.com"
        service_headers = {"X-NHX-Principal-Id": SERVICE_PRINCIPAL}
        user_headers = {"X-NHX-Principal-Id": editor_email, "X-NHX-Principal-Email": editor_email}

        # 1. Create workspace (as service principal)
        response = test_client.post(
            WORKSPACES_PATH,
            json={"name": workspace_id, "description": "Test workspace for role binding propagation"},
            headers=service_headers,
        )
        assert response.status_code in (200, 201), f"Failed to create workspace: {response.text}"

        try:
            # 2. Create role binding for editor using IAM endpoint (auto-sets granted_by/granted_at)
            response = test_client.post(
                IAM_ROLE_BINDINGS_PATH,
                json={
                    "principal": editor_email,
                    "role": "Editor",
                    "workspace": workspace_id,
                },
                headers=service_headers,
            )
            assert response.status_code in (200, 201), f"Failed to create role binding: {response.text}"

            # 3. Verify user can list workspaces and see the new one
            response = test_client.get(WORKSPACES_PATH, headers=user_headers)
            assert response.status_code == 200, f"User cannot list workspaces: {response.text}"
            workspaces = response.json()
            workspace_names = [ws["name"] for ws in workspaces["data"]]
            assert workspace_id in workspace_names, f"Workspace {workspace_id} not in list: {workspace_names}"

            # 4. THE KEY TEST: Verify user can GET the specific workspace
            response = test_client.get(f"{WORKSPACES_PATH}/{workspace_id}", headers=user_headers)
            assert response.status_code == 200, (
                f"Editor should be able to GET workspace. "
                f"Got {response.status_code}: {response.text}. "
                f"This indicates role binding did not propagate to the policy engine."
            )
            workspace = response.json()
            assert workspace["name"] == workspace_id

        finally:
            # Cleanup: delete the workspace
            test_client.delete(f"{WORKSPACES_PATH}/{workspace_id}", headers=service_headers)

    def test_viewer_can_read_but_not_write_workspace(self, test_client: TestClient):
        """Test that a Viewer can read a workspace but not update it."""
        workspace_id = f"test-ws-{uuid.uuid4().hex[:8]}"
        viewer_email = f"viewer-{uuid.uuid4().hex[:8]}@example.com"
        service_headers = {"X-NHX-Principal-Id": SERVICE_PRINCIPAL}
        user_headers = {"X-NHX-Principal-Id": viewer_email, "X-NHX-Principal-Email": viewer_email}

        # Create workspace
        response = test_client.post(
            WORKSPACES_PATH,
            json={"name": workspace_id, "description": "Test workspace for viewer access"},
            headers=service_headers,
        )
        assert response.status_code in (200, 201), f"Failed to create workspace: {response.text}"

        try:
            # Create role binding with Viewer role using IAM endpoint
            response = test_client.post(
                IAM_ROLE_BINDINGS_PATH,
                json={
                    "principal": viewer_email,
                    "role": "Viewer",
                    "workspace": workspace_id,
                },
                headers=service_headers,
            )
            assert response.status_code in (200, 201), f"Failed to create role binding: {response.text}"

            # Viewer should be able to GET the workspace
            response = test_client.get(f"{WORKSPACES_PATH}/{workspace_id}", headers=user_headers)
            assert response.status_code == 200, f"Viewer should be able to read workspace: {response.text}"

            # Viewer should NOT be able to UPDATE the workspace
            response = test_client.put(
                f"{WORKSPACES_PATH}/{workspace_id}",
                json={"description": "Updated by viewer - should fail"},
                headers=user_headers,
            )
            assert response.status_code == 403, f"Viewer should not be able to update workspace: {response.text}"

        finally:
            test_client.delete(f"{WORKSPACES_PATH}/{workspace_id}", headers=service_headers)
