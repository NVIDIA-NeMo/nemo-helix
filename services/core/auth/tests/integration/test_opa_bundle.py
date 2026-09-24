# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for OPA bundle endpoint."""

import gzip
import io
import json
import tarfile

import pytest
from fastapi.testclient import TestClient
from nhx.testing.client import TEST_ADMIN_EMAIL

# Seeded platform admin has iam.bundle.read via HelixAdmin role (see static-authz).
BUNDLE_ALLOWED_HEADERS = {
    "X-NHX-Principal-Id": TEST_ADMIN_EMAIL,
    "X-NHX-Principal-Email": TEST_ADMIN_EMAIL,
}

SERVICE_PRINCIPAL = "service:integration-test"
NON_ADMIN_USER = "test-user@example.com"

# test_client is module-scoped (expensive to boot): keep these tests on one xdist worker so
# they share it instead of each worker re-provisioning it from scratch.
pytestmark = pytest.mark.xdist_group("auth_opa_bundle")


class TestOPABundle:
    """Tests for OPA bundle endpoint."""

    def test_opa_bundle_download(self, http_client: TestClient):
        """Test downloading the OPA bundle."""
        headers = BUNDLE_ALLOWED_HEADERS
        response = http_client.get("/apis/auth/v2/iam/opa-bundle.tar.gz", headers=headers)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        assert response.headers.get("content-type") == "application/gzip"

        # Verify it's a valid gzip file containing a tar
        bundle_bytes = response.content
        assert len(bundle_bytes) > 0

        # Decompress and verify tar structure
        with gzip.GzipFile(fileobj=io.BytesIO(bundle_bytes)) as gz:
            tar_bytes = gz.read()

        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r") as tar:
            names = tar.getnames()

            # Verify expected files are present
            assert ".manifest" in names
            assert "data.json" in names

            # Verify at least one .rego policy file is present
            rego_files = [n for n in names if n.endswith(".rego")]
            assert len(rego_files) > 0, "Bundle should contain .rego policy files"

    def test_opa_bundle_contains_valid_data(self, http_client: TestClient):
        """Test that the OPA bundle contains valid authorization data."""
        headers = BUNDLE_ALLOWED_HEADERS
        response = http_client.get("/apis/auth/v2/iam/opa-bundle.tar.gz", headers=headers)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

        # Extract data.json from bundle
        with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as gz:
            tar_bytes = gz.read()

        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r") as tar:
            data_file = tar.extractfile("data.json")
            assert data_file is not None
            data = json.load(data_file)

        # Verify expected structure
        assert "authz" in data
        assert "roles" in data["authz"]
        assert "endpoints" in data["authz"]

        # Verify some expected roles exist
        roles = data["authz"]["roles"]
        assert "Viewer" in roles
        assert "Editor" in roles
        assert "Admin" in roles

        # Verify roles have permissions
        assert "permissions" in roles["Viewer"]
        assert len(roles["Viewer"]["permissions"]) > 0

    def test_opa_bundle_etag_support(self, http_client: TestClient):
        """Test that the OPA bundle supports E-Tag caching."""
        headers = BUNDLE_ALLOWED_HEADERS

        # First request - get the bundle and E-Tag
        response1 = http_client.get("/apis/auth/v2/iam/opa-bundle.tar.gz", headers=headers)
        assert response1.status_code == 200, f"Expected 200, got {response1.status_code}: {response1.text}"
        etag = response1.headers.get("ETag")
        assert etag is not None, "Response should include E-Tag header"

        # Second request with If-None-Match - should return 304
        headers_with_etag = {**headers, "If-None-Match": etag}
        response2 = http_client.get("/apis/auth/v2/iam/opa-bundle.tar.gz", headers=headers_with_etag)
        assert response2.status_code == 304, "Should return 304 Not Modified when E-Tag matches"

    def test_opa_bundle_cache_control(self, http_client: TestClient):
        """Test that the OPA bundle includes Cache-Control header."""
        headers = BUNDLE_ALLOWED_HEADERS
        response = http_client.get("/apis/auth/v2/iam/opa-bundle.tar.gz", headers=headers)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

        cache_control = response.headers.get("Cache-Control")
        assert cache_control is not None, "Response should include Cache-Control header"
        assert "max-age" in cache_control

    def test_opa_bundle_manifest(self, http_client: TestClient):
        """Test that the OPA bundle contains a valid manifest."""
        headers = BUNDLE_ALLOWED_HEADERS
        response = http_client.get("/apis/auth/v2/iam/opa-bundle.tar.gz", headers=headers)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

        with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as gz:
            tar_bytes = gz.read()

        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r") as tar:
            manifest_file = tar.extractfile(".manifest")
            assert manifest_file is not None
            manifest = json.load(manifest_file)

        assert "revision" in manifest
        assert "roots" in manifest
        assert "authz" in manifest["roots"]

    def test_opa_bundle_allowed_for_service_principal(self, http_client: TestClient):
        """Service principals are auto-authorized by middleware (no PDP); bundle download succeeds."""
        headers = {"X-NHX-Principal-Id": SERVICE_PRINCIPAL}
        response = http_client.get("/apis/auth/v2/iam/opa-bundle.tar.gz", headers=headers)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        assert len(response.content) > 0

    def test_opa_bundle_denied_without_iam_bundle_read(self, http_client: TestClient):
        """Principals without iam.bundle.read cannot download the bundle."""
        headers = {
            "X-NHX-Principal-Id": NON_ADMIN_USER,
            "X-NHX-Principal-Email": NON_ADMIN_USER,
        }
        response = http_client.get("/apis/auth/v2/iam/opa-bundle.tar.gz", headers=headers)
        assert response.status_code == 403, f"Expected 403, got {response.status_code}: {response.text}"
