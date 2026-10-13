# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for authorization middleware.

These tests verify that the AuthorizationMiddleware correctly:
- Denies unauthenticated requests (401)
- Allows authenticated requests with proper roles
- Evaluates service principals through the PDP (except the PDP entrypoints themselves)
- Allows health endpoints without authentication
"""

import uuid

import pytest
from fastapi.testclient import TestClient

# Test principals for authenticated requests
TEST_USER_EMAIL = "test-user@example.com"
SERVICE_PRINCIPAL = "service:integration-test"

# Platform mounts entities at /apis/entities; list workspaces is GET /apis/entities/v2/workspaces
WORKSPACES_PATH = "/apis/entities/v2/workspaces"

# test_client is module-scoped (expensive to boot): keep these tests on one xdist worker so
# they share it instead of each worker re-provisioning it from scratch.
pytestmark = pytest.mark.xdist_group("auth_authorization_middleware")


class TestAuthorizationMiddleware:
    """Tests for authorization middleware integration."""

    def test_health_endpoint_allowed_without_auth(self, http_client: TestClient):
        """Health endpoints should always be accessible."""
        response = http_client.get("/health/live")
        assert response.status_code == 200

        response = http_client.get("/health/ready")
        # May be 200 or 503 depending on service state, but not 401/403
        assert response.status_code in (200, 503)

    def test_unauthenticated_request_denied(self, http_client: TestClient):
        """Requests without principal headers should be denied."""
        # Try to access a protected endpoint without auth headers
        response = http_client.get(WORKSPACES_PATH)
        assert response.status_code == 401, f"Expected 401, got {response.status_code}: {response.text}"

    def test_authenticated_user_allowed_to_list_workspaces(self, http_client: TestClient):
        """Authenticated users should be able to list workspaces."""
        headers = {
            "X-NHX-Principal-Id": TEST_USER_EMAIL,
            "X-NHX-Principal-Email": TEST_USER_EMAIL,
        }
        response = http_client.get(WORKSPACES_PATH, headers=headers)
        # Should be allowed (200) - listing workspaces is typically allowed
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    def test_service_principal_authorized_via_pdp(self, http_client: TestClient):
        """Service principals are authorized via PDP (policy allows service:*)."""
        headers = {
            "X-NHX-Principal-Id": SERVICE_PRINCIPAL,
        }
        response = http_client.get(WORKSPACES_PATH, headers=headers)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    def test_create_workspace_requires_auth(self, http_client: TestClient):
        """Creating a workspace should require authentication."""
        workspace_name = f"test-ws-{uuid.uuid4().hex[:8]}"
        workspace_data = {
            "name": workspace_name,
        }
        # Without auth headers
        response = http_client.post(WORKSPACES_PATH, json=workspace_data)
        assert response.status_code == 401, f"Expected 401, got {response.status_code}: {response.text}"

    def test_create_workspace_with_auth(self, http_client: TestClient):
        """Creating a workspace with proper auth should work."""
        workspace_name = f"test-ws-{uuid.uuid4().hex[:8]}"
        workspace_data = {
            "name": workspace_name,
        }
        headers = {
            "X-NHX-Principal-Id": TEST_USER_EMAIL,
            "X-NHX-Principal-Email": TEST_USER_EMAIL,
        }
        response = http_client.post(WORKSPACES_PATH, json=workspace_data, headers=headers)
        # Should be allowed with auth
        assert response.status_code in (200, 201, 409), (
            f"Expected 200/201/409, got {response.status_code}: {response.text}"
        )

    def test_authz_endpoint_available_for_service_principal(self, http_client: TestClient):
        """The PDP authz endpoint should be reachable by service principals."""
        auth_input = {
            "input": {
                "principal_id": TEST_USER_EMAIL,
                "method": "GET",
                "path": WORKSPACES_PATH,
            }
        }
        headers = {"X-NHX-Principal-Id": SERVICE_PRINCIPAL}
        response = http_client.post("/apis/auth/v2/authz/allow", json=auth_input, headers=headers)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

        result = response.json()
        assert "result" in result
        assert "allowed" in result["result"]

    def test_authz_invalid_entrypoint_returns_structured_detail(self, http_client: TestClient):
        headers = {"X-NHX-Principal-Id": SERVICE_PRINCIPAL}
        response = http_client.post(
            "/apis/auth/v2/authz/missing",
            json={"input": {}},
            headers=headers,
        )

        assert response.status_code == 400
        assert response.json() == {
            "detail": {
                "error": "Invalid entrypoint: missing",
                "valid_entrypoints": ["allow", "has_permissions", "has_role"],
            }
        }

    def test_authz_endpoint_blocked_without_service_principal(self, http_client: TestClient):
        """The PDP authz endpoint must not be accessible without a service principal."""
        auth_input = {
            "input": {
                "principal_id": TEST_USER_EMAIL,
                "method": "GET",
                "path": WORKSPACES_PATH,
            }
        }
        response = http_client.post("/apis/auth/v2/authz/allow", json=auth_input)
        assert response.status_code == 401, f"Expected 401, got {response.status_code}: {response.text}"

    def test_authz_endpoint_blocked_for_regular_user(self, http_client: TestClient):
        """The PDP authz endpoint must not be accessible to regular authenticated users."""
        auth_input = {
            "input": {
                "principal_id": TEST_USER_EMAIL,
                "method": "GET",
                "path": WORKSPACES_PATH,
            }
        }
        headers = {
            "X-NHX-Principal-Id": TEST_USER_EMAIL,
            "X-NHX-Principal-Email": TEST_USER_EMAIL,
        }
        response = http_client.post("/apis/auth/v2/authz/allow", json=auth_input, headers=headers)
        assert response.status_code == 403, f"Expected 403, got {response.status_code}: {response.text}"

    def test_iam_role_bindings_forbidden_for_user_principal(self, http_client: TestClient):
        """IAM role-bindings are internal; end users must not call them directly."""
        headers = {
            "X-NHX-Principal-Id": TEST_USER_EMAIL,
            "X-NHX-Principal-Email": TEST_USER_EMAIL,
        }
        response = http_client.get("/apis/auth/v2/iam/role-bindings", headers=headers)
        assert response.status_code == 403, f"Expected 403, got {response.status_code}: {response.text}"

    def test_nested_entities_forbidden_for_user_principal(self, http_client: TestClient):
        """Workspace-scoped entity APIs are internal; end users must not call them directly."""
        headers = {
            "X-NHX-Principal-Id": TEST_USER_EMAIL,
            "X-NHX-Principal-Email": TEST_USER_EMAIL,
        }
        response = http_client.get(
            "/apis/entities/v2/workspaces/default/entities/evaluation_config",
            headers=headers,
        )
        assert response.status_code == 403, f"Expected 403, got {response.status_code}: {response.text}"
