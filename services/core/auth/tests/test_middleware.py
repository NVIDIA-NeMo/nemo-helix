# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for auth middleware."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nhx.common.auth import AuthorizationMiddleware, Principal
from nhx.common.config import AuthConfig


def make_mock_auth_config(enabled: bool = False):
    """Create a mock AuthConfig for testing."""
    config = AuthConfig()
    # Override the values for testing
    object.__setattr__(config, "enabled", enabled)
    object.__setattr__(config, "policy_decision_point_base_url", "http://opa:8181")
    return config


@pytest.fixture
def app_auth_disabled():
    """Create a FastAPI app with auth disabled."""
    with patch("nhx.common.auth.middleware.get_auth_config") as mock_get_config:
        mock_get_config.return_value = make_mock_auth_config(enabled=False)

        app = FastAPI()
        app.add_middleware(AuthorizationMiddleware)

        @app.get("/test")
        async def test_endpoint():
            return {"status": "ok"}

        @app.get("/v2/workspaces/{workspace_id}/test")
        async def workspace_endpoint(workspace_id: str):
            return {"workspace_id": workspace_id}

        yield app


def test_middleware_auth_disabled(app_auth_disabled):
    """Test that requests pass through when auth is disabled."""
    client = TestClient(app_auth_disabled)
    response = client.get("/test")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_middleware_workspace_endpoint_auth_disabled(app_auth_disabled):
    """Test workspace endpoint when auth is disabled."""
    client = TestClient(app_auth_disabled)
    response = client.get("/v2/workspaces/my-workspace/test")
    assert response.status_code == 200
    assert response.json() == {"workspace_id": "my-workspace"}


def test_middleware_internal_paths_use_standard_auth():
    """Paths under /internal/ are not auto-authorized; unauthenticated requests are denied."""
    with patch("nhx.common.auth.middleware.get_auth_config") as mock_get_config:
        mock_get_config.return_value = make_mock_auth_config(enabled=True)

        app = FastAPI()
        app.add_middleware(AuthorizationMiddleware)

        @app.get("/internal/test")
        async def internal_endpoint():
            return {"status": "ok"}

        client = TestClient(app, raise_server_exceptions=False)
        with patch("nhx.common.auth.client.AuthClient.authorize_request") as mock_authorize:
            mock_authorize.return_value = MagicMock(allowed=False)
            response = client.get("/internal/test")
        assert response.status_code == 401


def test_middleware_health_endpoints_skip_auth():
    """Test that health endpoints skip auth (paths in HEALTH_ENDPOINTS bypass)."""
    with patch("nhx.common.auth.middleware.get_auth_config") as mock_get_config:
        mock_get_config.return_value = make_mock_auth_config(enabled=True)

        app = FastAPI()
        app.add_middleware(AuthorizationMiddleware)

        @app.get("/health/live")
        async def health_live():
            return {"status": "live"}

        @app.get("/health/ready")
        async def health_ready():
            return {"status": "ready"}

        client = TestClient(app)

        response = client.get("/health/live")
        assert response.status_code == 200

        response = client.get("/health/ready")
        assert response.status_code == 200


def test_principal_from_headers():
    """Test creating principal from headers."""
    headers = {
        "x-nhx-principal-id": "user@example.com",
        "x-nhx-principal-email": "user@example.com",
        "x-nhx-principal-groups": "group1,group2",
    }

    principal = Principal.from_headers(headers)

    assert principal is not None
    assert principal.id == "user@example.com"
    assert principal.email == "user@example.com"
    assert principal.groups == ["group1", "group2"]


def test_principal_get_headers():
    """Test getting headers from principal."""
    principal = Principal(
        id="user@example.com",
        email="user@example.com",
        groups=["group1", "group2"],
        on_behalf_of=None,
    )

    headers = principal.get_headers()

    assert headers["X-NHX-Principal-Id"] == "user@example.com"
    assert headers["X-NHX-Principal-Email"] == "user@example.com"
    assert headers["X-NHX-Principal-Groups"] == "group1,group2"
