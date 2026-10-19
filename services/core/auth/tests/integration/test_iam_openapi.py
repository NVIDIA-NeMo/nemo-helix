# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for IAM endpoints in OpenAPI spec."""

import pytest
from fastapi.testclient import TestClient

SERVICE_PRINCIPAL = "service:integration-test"

# test_client is module-scoped (expensive to boot): keep these tests on one xdist worker so
# they share it instead of each worker re-provisioning it from scratch.
pytestmark = pytest.mark.xdist_group("auth_iam_openapi")


class TestIAMOpenAPI:
    """Tests for IAM endpoints in OpenAPI spec."""

    def test_iam_role_bindings_routes_in_openapi(self, http_client: TestClient):
        """Test that IAM role binding endpoints are documented in OpenAPI spec."""
        headers = {"X-NHX-Principal-Id": SERVICE_PRINCIPAL}
        response = http_client.get("/openapi.json", headers=headers)
        assert response.status_code == 200

        spec = response.json()
        paths = spec.get("paths", {})

        # Verify role binding endpoints are present (platform mounts auth at /apis/auth)
        assert "/apis/auth/v2/iam/role-bindings" in paths
        assert "get" in paths["/apis/auth/v2/iam/role-bindings"]
        assert "post" in paths["/apis/auth/v2/iam/role-bindings"]
        assert "/apis/auth/v2/iam/role-bindings/{name}" in paths
        assert "get" in paths["/apis/auth/v2/iam/role-bindings/{name}"]
        assert "delete" in paths["/apis/auth/v2/iam/role-bindings/{name}"]

    def test_iam_schemas_in_openapi(self, http_client: TestClient):
        """Test that IAM schemas are in OpenAPI spec."""
        headers = {"X-NHX-Principal-Id": SERVICE_PRINCIPAL}
        response = http_client.get("/openapi.json", headers=headers)
        assert response.status_code == 200

        spec = response.json()
        schemas = spec.get("components", {}).get("schemas", {})

        # Verify role binding schemas are present
        assert "RoleBinding" in schemas
        assert "RoleBindingInput" in schemas

    def test_authz_error_schema_models_only_structured_400_detail(self, http_client: TestClient):
        headers = {"X-NHX-Principal-Id": SERVICE_PRINCIPAL}
        response = http_client.get("/openapi.json", headers=headers)
        assert response.status_code == 200

        spec = response.json()
        operation = spec["paths"]["/apis/auth/v2/authz/{entrypoint}"]["post"]
        assert operation["responses"]["400"]["content"]["application/json"]["schema"] == {
            "$ref": "#/components/schemas/AuthzErrorResponse"
        }
        assert "content" not in operation["responses"]["500"]

        schemas = spec["components"]["schemas"]
        assert schemas["AuthzErrorResponse"]["properties"]["detail"] == {
            "$ref": "#/components/schemas/AuthzErrorDetail"
        }
        assert set(schemas["AuthzErrorDetail"]["required"]) == {"error"}
