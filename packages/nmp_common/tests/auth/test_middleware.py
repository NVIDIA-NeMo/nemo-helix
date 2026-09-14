# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for authorization middleware."""

import asyncio
import time
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import jwt
import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from nmp.common.auth.access_key_lifecycle import ACCESS_KEY_LIFECYCLE_CIRCUIT_FAILURE_THRESHOLD
from nmp.common.auth.client import AuthClient
from nmp.common.auth.dependencies import get_auth_client
from nmp.common.auth.jwt import UnsignedJWTRejectedError
from nmp.common.auth.middleware import (
    BYPASS_PREFIXES,
    HEALTH_ENDPOINTS,
    PUBLIC_GET_PATHS,
    AuthorizationMiddleware,
)
from nmp.common.auth.models import Principal
from nmp.common.auth.token_claims import ActorClaims, TokenClaims
from nmp.common.auth.token_resolver import ResolvedBearerToken
from nmp.common.config import AuthConfig, Configuration, PlatformConfig
from nmp.common.config.base import OIDCConfig
from starlette.responses import Response


@pytest.fixture(autouse=True)
def _cleanup_config_overrides():
    """Clean up Configuration overrides set by create_test_app to prevent leaking to other tests."""
    yield
    Configuration.clear_override(AuthConfig)


@pytest.fixture
def oidc_config():
    """Create an OIDC config for testing."""
    return OIDCConfig(
        enabled=True,
        issuer="https://sso.example.com",
        client_id="test-client",
    )


@pytest.fixture
def auth_config_enabled(oidc_config):
    """Create an AuthConfig with auth and OIDC enabled."""
    return AuthConfig(
        enabled=True,
        policy_decision_point_base_url="http://localhost:8181",
        oidc=oidc_config,
    )


@pytest.fixture
def auth_config_disabled():
    """Create an AuthConfig with auth disabled."""
    return AuthConfig(
        enabled=False,
        policy_decision_point_base_url="http://localhost:8181",
    )


@pytest.fixture
def auth_config_oidc_disabled():
    """Create an AuthConfig with auth enabled but OIDC disabled."""
    return AuthConfig(
        enabled=True,
        policy_decision_point_base_url="http://localhost:8181",
        oidc=OIDCConfig(enabled=False),
    )


@pytest.fixture
def access_key_lifecycle_middleware(auth_config_oidc_disabled, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("NMP_AUTH_URL", raising=False)

    @asynccontextmanager
    async def make(handler, base_url: str = "https://platform.example.com"):
        config = auth_config_oidc_disabled.model_copy(
            update={"access_keys": auth_config_oidc_disabled.access_keys.model_copy(update={"enabled": True})}
        )
        Configuration.set_override(config)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            middleware = AuthorizationMiddleware(
                FastAPI(),
                service_name="test-service",
                http_client=http_client,
                access_key_lifecycle_http_client=http_client,
            )
            with patch(
                "nmp.common.sdk_factory.Configuration.get_platform_config",
                return_value=PlatformConfig(base_url=base_url, services=""),
            ):
                yield middleware

    return make


def create_test_app(auth_config: AuthConfig) -> FastAPI:
    """Create a test FastAPI app with auth middleware."""
    app = FastAPI()

    @app.get("/test")
    async def test_endpoint():
        return {"status": "ok"}

    @app.get("/health")
    async def health_endpoint():
        return {"status": "healthy"}

    @app.get("/apis/auth/discovery")
    async def discovery_endpoint():
        return {"auth_enabled": True}

    @app.get("/apis/files/v2/hf/{workspace}/{name}/resolve/{revision}/{path:path}")
    async def hf_download_endpoint(workspace: str, name: str, revision: str, path: str):
        return {"workspace": workspace, "name": name, "path": path}

    @app.post("/apis/files/v2/workspaces/{workspace}/filesets/{name}/otlp/v1/logs")
    async def files_otlp_logs_upload(workspace: str, name: str):
        return {"workspace": workspace, "name": name}

    @app.post("/apis/files/v2/workspaces/{workspace}/filesets/{name}/otlp/v1/logs/query")
    async def files_otlp_logs_query(workspace: str, name: str):
        return {"workspace": workspace, "name": name}

    Configuration.set_override(auth_config)
    app.add_middleware(AuthorizationMiddleware, service_name="test-service")

    return app


def create_test_app_with_platform_routes(auth_config: AuthConfig) -> FastAPI:
    """Test app including Entities and IAM routes used by internal-route tests."""
    app = create_test_app(auth_config)

    @app.get("/apis/entities/v2/workspaces")
    async def list_workspaces():
        return {"data": []}

    @app.post("/apis/entities/v2/workspaces")
    async def create_workspace():
        return {"data": {"name": "new-ws"}}

    @app.api_route("/apis/entities/v2/workspaces/{name}", methods=["GET", "PUT", "DELETE"])
    async def workspace_by_name(name: str):
        return {"name": name}

    @app.get("/apis/entities/v2/workspaces/{workspace}/entities/{entity_type}")
    async def nested_entities(workspace: str, entity_type: str):
        return {"workspace": workspace, "entity_type": entity_type}

    @app.get("/apis/auth/v2/iam/role-bindings")
    async def iam_list():
        return {"data": []}

    return app


class TestHealthEndpointsBypass:
    """Tests for health endpoints bypassing authentication."""

    def test_health_endpoints_in_bypass_list(self):
        """Verify that health endpoints are in the bypass list."""
        assert "/status" in HEALTH_ENDPOINTS
        assert "/health/live" in HEALTH_ENDPOINTS
        assert "/health/ready" in HEALTH_ENDPOINTS
        assert "/metrics" in HEALTH_ENDPOINTS
        assert "/apis/auth/discovery" in HEALTH_ENDPOINTS
        assert "/apis/auth/authenticate" in HEALTH_ENDPOINTS
        assert "/apis/auth/ext-authz" in HEALTH_ENDPOINTS
        assert "/apis/auth/ext-authz/" in BYPASS_PREFIXES
        assert "/apis/auth/authenticate/" not in BYPASS_PREFIXES

    def test_root_path_in_public_get_paths(self):
        assert "/" in PUBLIC_GET_PATHS

    @pytest.mark.parametrize("method", ["get", "head"])
    def test_root_bypasses_auth_for_safe_methods(self, auth_config_enabled, method):
        app = FastAPI()

        @app.api_route("/", methods=["GET", "HEAD"])
        async def root_handler():
            return {"status": "ok"}

        Configuration.set_override(auth_config_enabled)
        app.add_middleware(AuthorizationMiddleware, service_name="test-service")

        client = TestClient(app)
        with patch("nmp.common.auth.client.AuthClient.authorize_request") as mock_authorize:
            mock_authorize.return_value = MagicMock(allowed=False)
            response = getattr(client, method)("/")

        assert response.status_code == 200
        mock_authorize.assert_not_called()


class TestStudioPluginBypass:
    """Studio plugin manifest and bundles are public — the SPA fetches the manifest
    anonymously and loads bundles via dynamic import(), which cannot send Authorization."""

    def test_plugin_paths_in_bypass_lists(self):
        assert "/apis/plugins" in PUBLIC_GET_PATHS
        assert "/plugin-ui/" in BYPASS_PREFIXES

    def test_plugins_manifest_get_bypasses_auth(self, auth_config_enabled):
        app = FastAPI()

        @app.get("/apis/plugins")
        async def list_plugins():
            return []

        Configuration.set_override(auth_config_enabled)
        app.add_middleware(AuthorizationMiddleware, service_name="test-service")

        client = TestClient(app)
        with patch("nmp.common.auth.client.AuthClient.authorize_request") as mock_authorize:
            mock_authorize.return_value = MagicMock(allowed=False)
            response = client.get("/apis/plugins")

        assert response.status_code == 200
        mock_authorize.assert_not_called()

    def test_plugin_bundle_get_bypasses_auth(self, auth_config_enabled):
        app = FastAPI()

        @app.get("/plugin-ui/{plugin_name}/{filename}")
        async def serve_bundle(plugin_name: str, filename: str):
            return {"plugin": plugin_name, "filename": filename}

        Configuration.set_override(auth_config_enabled)
        app.add_middleware(AuthorizationMiddleware, service_name="test-service")

        client = TestClient(app)
        with patch("nmp.common.auth.client.AuthClient.authorize_request") as mock_authorize:
            mock_authorize.return_value = MagicMock(allowed=False)
            response = client.get("/plugin-ui/example/index.js")

        assert response.status_code == 200
        mock_authorize.assert_not_called()

    def test_other_paths_still_require_auth(self, auth_config_enabled):
        """The plugin bypasses must not open up other routes."""
        app = create_test_app(auth_config_enabled)
        client = TestClient(app, raise_server_exceptions=False)

        with patch("nmp.common.auth.client.AuthClient.authorize_request") as mock_authorize:
            mock_authorize.return_value = MagicMock(allowed=False)
            response = client.get("/test")

        assert response.status_code == 401


class TestBearerTokenAuth:
    """Tests for Bearer token authentication in middleware."""

    def test_bearer_token_oidc_not_configured(self, auth_config_oidc_disabled):
        """Test that Bearer token auth fails when OIDC is not configured."""
        app = create_test_app(auth_config_oidc_disabled)
        client = TestClient(app, raise_server_exceptions=False)

        # Mock PDP to allow auth check to pass for the test path
        with patch("nmp.common.auth.client.AuthClient.authorize_request") as mock_authorize:
            mock_authorize.return_value = MagicMock(allowed=False)

            response = client.get(
                "/test",
                headers={"Authorization": "Bearer some-token"},
            )

            # Should return 401 because OIDC is not configured
            assert response.status_code == 401
            assert "Bearer token authentication not configured" in response.json()["detail"]
            mock_authorize.assert_not_called()

    def test_bearer_token_unsigned_jwt_accepted_when_allowed(self):
        """Test that unsigned JWTs are accepted when allow_unsigned_jwt is true, even without OIDC."""
        config = AuthConfig(
            enabled=True,
            allow_unsigned_jwt=True,
            policy_decision_point_base_url="http://localhost:8181",
            oidc=OIDCConfig(enabled=False),
        )
        app = create_test_app(config)
        client = TestClient(app, raise_server_exceptions=False)

        valid_claims = TokenClaims(
            subject="admin@example.com",
            email="admin@example.com",
            groups=[],
            scopes=[],
            raw_claims={},
        )

        with patch("nmp.common.auth.jwt.JWTValidator.validate_token") as mock_validate:
            mock_validate.return_value = valid_claims

            with patch("nmp.common.auth.client.AuthClient.authorize_request") as mock_authorize:
                mock_authorize.return_value = MagicMock(allowed=True)

                response = client.get(
                    "/test",
                    headers={"Authorization": "Bearer unsigned-jwt-token"},
                )

                assert response.status_code == 200

    def test_bearer_token_invalid_token(self, auth_config_enabled):
        """Test that invalid Bearer tokens return 401."""
        app = create_test_app(auth_config_enabled)
        client = TestClient(app, raise_server_exceptions=False)

        with patch("nmp.common.auth.jwt.JWTValidator.validate_token") as mock_validate:
            mock_validate.return_value = None  # Invalid token

            response = client.get(
                "/test",
                headers={"Authorization": "Bearer invalid-token"},
            )

            assert response.status_code == 401
            assert "Invalid or expired token" in response.json()["detail"]

    @pytest.mark.parametrize(
        "auth_header",
        [
            "Bearer",
            "Bearer ",
            "Bearer token extra",
        ],
    )
    def test_malformed_bearer_token_returns_401(self, auth_config_enabled, auth_header):
        """Malformed Bearer headers fail auth instead of falling through as anonymous requests."""
        app = create_test_app(auth_config_enabled)
        client = TestClient(app, raise_server_exceptions=False)

        with patch("nmp.common.auth.client.AuthClient.authorize_request") as mock_authorize:
            response = client.get("/test", headers={"Authorization": auth_header})

        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid bearer token"
        mock_authorize.assert_not_called()

    def test_bearer_token_expired_unsigned_jwt_returns_401(self):
        """Expired unsigned JWTs are rejected when allow_unsigned_jwt is true."""
        config = AuthConfig(
            enabled=True,
            allow_unsigned_jwt=True,
            policy_decision_point_base_url="http://localhost:8181",
            oidc=OIDCConfig(enabled=False),
        )
        app = create_test_app(config)
        client = TestClient(app, raise_server_exceptions=False)
        now = int(time.time())
        token = jwt.encode(
            {
                "sub": "admin@example.com",
                "iat": now - 7200,
                "exp": now - 3600,
            },
            key="",
            algorithm="none",
        )

        with patch("nmp.common.auth.client.AuthClient.authorize_request") as mock_authorize:
            response = client.get(
                "/test",
                headers={"Authorization": f"Bearer {token}"},
            )

        assert response.status_code == 401
        assert "Invalid or expired token" in response.json()["detail"]
        mock_authorize.assert_not_called()

    def test_bearer_token_not_validated_when_auth_disabled(self):
        """Auth disabled allows requests even when local unsigned-JWT support is enabled."""
        config = AuthConfig(
            enabled=False,
            allow_unsigned_jwt=True,
            policy_decision_point_base_url="http://localhost:8181",
            oidc=OIDCConfig(enabled=False),
        )
        app = create_test_app(config)
        client = TestClient(app, raise_server_exceptions=False)

        with patch("nmp.common.auth.jwt.JWTValidator.validate_token") as mock_validate:
            mock_validate.return_value = None

            response = client.get(
                "/test",
                headers={"Authorization": "Bearer not-used"},
            )

            assert response.status_code == 200
            mock_validate.assert_not_called()

    def test_bearer_token_unsigned_token_rejected_message(self, auth_config_enabled):
        """Test that unsigned JWT rejection returns actionable 401 detail."""
        app = create_test_app(auth_config_enabled)
        client = TestClient(app, raise_server_exceptions=False)

        with patch("nmp.common.auth.jwt.JWTValidator.validate_token") as mock_validate:
            mock_validate.side_effect = UnsignedJWTRejectedError(
                "Unsigned JWTs are not accepted. Set auth.allow_unsigned_jwt=true for local development."
            )

            response = client.get(
                "/test",
                headers={"Authorization": "Bearer unsigned-token"},
            )

            assert response.status_code == 401
            assert "Unsigned JWTs are not accepted" in response.json()["detail"]

    def test_bearer_token_valid_token_auth_disabled(self, auth_config_disabled):
        """Test Bearer token with valid token when auth is disabled."""
        auth_config_disabled.oidc = OIDCConfig(
            enabled=True,
            issuer="https://sso.example.com",
            client_id="test-client",
        )
        app = create_test_app(auth_config_disabled)
        client = TestClient(app, raise_server_exceptions=False)

        valid_claims = TokenClaims(
            subject="user@example.com",
            email="user@example.com",
            groups=["users"],
            scopes=["openid"],
            raw_claims={},
        )

        with patch("nmp.common.auth.jwt.JWTValidator.validate_token") as mock_validate:
            mock_validate.return_value = valid_claims

            response = client.get(
                "/test",
                headers={"Authorization": "Bearer valid-token"},
            )

            # Should succeed because auth is disabled
            assert response.status_code == 200

    def test_bearer_token_valid_token_pdp_allows(self, auth_config_enabled):
        """Test Bearer token with valid token when PDP allows."""
        app = create_test_app(auth_config_enabled)
        client = TestClient(app, raise_server_exceptions=False)

        valid_claims = TokenClaims(
            subject="user@example.com",
            email="user@example.com",
            groups=["users"],
            scopes=["openid"],
            raw_claims={},
        )

        with patch("nmp.common.auth.jwt.JWTValidator.validate_token") as mock_validate:
            mock_validate.return_value = valid_claims

            with patch("nmp.common.auth.client.AuthClient.authorize_request") as mock_authorize:
                mock_authorize.return_value = MagicMock(allowed=True)

                response = client.get(
                    "/test",
                    headers={"Authorization": "Bearer valid-token"},
                )

                assert response.status_code == 200

    def test_bearer_token_valid_token_pdp_denies(self, auth_config_enabled):
        """Test Bearer token with valid token when PDP denies."""
        app = create_test_app(auth_config_enabled)
        client = TestClient(app, raise_server_exceptions=False)

        valid_claims = TokenClaims(
            subject="user@example.com",
            email="user@example.com",
            groups=["users"],
            scopes=["openid"],
            raw_claims={},
        )

        with patch("nmp.common.auth.jwt.JWTValidator.validate_token") as mock_validate:
            mock_validate.return_value = valid_claims

            with patch("nmp.common.auth.client.AuthClient.authorize_request") as mock_authorize:
                mock_authorize.return_value = MagicMock(allowed=False)

                response = client.get(
                    "/test",
                    headers={"Authorization": "Bearer valid-token"},
                )

                assert response.status_code == 403

    def test_bearer_token_scoped_access_key_accepted_without_oidc(self, auth_config_oidc_disabled):
        config = auth_config_oidc_disabled.model_copy(
            update={"access_keys": auth_config_oidc_disabled.access_keys.model_copy(update={"enabled": True})}
        )
        app = create_test_app(config)
        client = TestClient(app, raise_server_exceptions=False)

        valid_claims = TokenClaims(
            subject="alice@example.com",
            email="alice@example.com",
            groups=["team-ml"],
            scopes=[],
            raw_claims={"nmp_token_type": "access_key"},
        )
        resolved = ResolvedBearerToken(claims=valid_claims, token_kind="access_key")

        with patch("nmp.common.auth.access_keys.is_access_key_token_candidate", return_value=True):
            with patch.object(
                AuthorizationMiddleware,
                "_authenticate_access_key_lifecycle",
                new_callable=AsyncMock,
                return_value=resolved,
            ) as mock_lifecycle:
                with patch("nmp.common.auth.client.AuthClient.authorize_request") as mock_authorize:
                    mock_authorize.return_value = MagicMock(allowed=True)

                    response = client.get("/test", headers={"Authorization": "Bearer scoped-access-key"})

        assert response.status_code == 200
        mock_lifecycle.assert_called_once()
        mock_authorize.assert_called_once()

    def test_bearer_token_scoped_access_key_invalid_falls_back_to_oidc(self, auth_config_enabled):
        config = auth_config_enabled.model_copy(
            update={"access_keys": auth_config_enabled.access_keys.model_copy(update={"enabled": True})}
        )
        app = create_test_app(config)
        client = TestClient(app, raise_server_exceptions=False)

        oidc_claims = TokenClaims(
            subject="bob@example.com",
            email="bob@example.com",
            groups=[],
            scopes=[],
            raw_claims={},
        )

        # Non-access-key tokens skip validate_access_key_token entirely (is_access_key_token_candidate
        # returns False for OIDC tokens) and fall straight through to OIDC validation.
        with patch("nmp.common.auth.jwt.JWTValidator.validate_token") as mock_oidc_validate:
            mock_oidc_validate.return_value = oidc_claims
            with patch("nmp.common.auth.client.AuthClient.authorize_request") as mock_authorize:
                mock_authorize.return_value = MagicMock(allowed=True)

                response = client.get("/test", headers={"Authorization": "Bearer oidc-token"})

        assert response.status_code == 200
        mock_oidc_validate.assert_called_once()

    def test_scoped_access_key_middleware_mapping_is_skipped_when_access_keys_are_disabled(
        self,
        auth_config_oidc_disabled,
    ):
        app = create_test_app(auth_config_oidc_disabled)
        client = TestClient(app, raise_server_exceptions=False)

        with patch("nmp.common.auth.access_keys.validate_access_key_token") as mock_validate:
            response = client.get("/test", headers={"Authorization": "Bearer scoped-access-key"})

        assert response.status_code == 401
        assert response.json()["detail"] == "Bearer token authentication not configured"
        mock_validate.assert_not_called()

    @pytest.mark.asyncio
    async def test_access_key_lifecycle_sdk_uses_auth_service_discovery(
        self,
        auth_config_oidc_disabled,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.delenv("NMP_AUTH_URL", raising=False)
        Configuration.set_override(auth_config_oidc_disabled)
        platform_config = PlatformConfig(
            base_url="https://platform.example.com",
            service_discovery={"auth": "https://auth.internal:8080"},
            services="",
        )
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                200,
                json={
                    "principal": "alice@example.com",
                    "groups": [],
                    "scopes": [],
                    "jti": "ak_example",
                    "token_kind": "access_key",
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            middleware = AuthorizationMiddleware(
                FastAPI(),
                service_name="test-service",
                access_key_lifecycle_http_client=http_client,
            )
            with patch.object(Configuration, "get_platform_config", return_value=platform_config):
                response = await middleware._authenticate_access_key_lifecycle("scoped-access-key")

        assert isinstance(response, ResolvedBearerToken)
        assert requests[0].url == httpx.URL("https://auth.internal:8080/apis/auth/authenticate")

    @pytest.mark.asyncio
    async def test_access_key_lifecycle_callout_allows_active_token(self, access_key_lifecycle_middleware):
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                200,
                json={
                    "principal": "alice@example.com",
                    "email": "alice@example.com",
                    "groups": ["team-ml"],
                    "scopes": [],
                    "jti": "ak_example",
                    "token_kind": "access_key",
                },
            )

        async with access_key_lifecycle_middleware(handler) as middleware:
            response = await middleware._authenticate_access_key_lifecycle("scoped-access-key")

        assert isinstance(response, ResolvedBearerToken)
        assert response.claims.subject == "alice@example.com"
        assert requests[0].url == httpx.URL("https://platform.example.com/apis/auth/authenticate")
        assert requests[0].headers["authorization"] == "Bearer scoped-access-key"

    @pytest.mark.asyncio
    async def test_access_key_lifecycle_callout_does_not_use_pdp_transport(self, auth_config_oidc_disabled):
        config = auth_config_oidc_disabled.model_copy(
            update={"access_keys": auth_config_oidc_disabled.access_keys.model_copy(update={"enabled": True})}
        )
        Configuration.set_override(config)

        def reject_pdp_request(request: httpx.Request) -> httpx.Response:
            raise AssertionError(f"lifecycle request used PDP transport: {request.url}")

        def authenticate_access_key(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "principal": "alice@example.com",
                    "groups": [],
                    "scopes": [],
                    "jti": "ak_example",
                    "token_kind": "access_key",
                },
            )

        async with (
            httpx.AsyncClient(transport=httpx.MockTransport(reject_pdp_request)) as pdp_client,
            httpx.AsyncClient(transport=httpx.MockTransport(authenticate_access_key)) as lifecycle_client,
        ):
            middleware = AuthorizationMiddleware(
                FastAPI(),
                service_name="test-service",
                http_client=pdp_client,
                access_key_lifecycle_http_client=lifecycle_client,
            )
            with patch(
                "nmp.common.sdk_factory.Configuration.get_platform_config",
                return_value=PlatformConfig(base_url="unix:///tmp/nemo-platform.sock", services=""),
            ):
                response = await middleware._authenticate_access_key_lifecycle("scoped-access-key")

        assert isinstance(response, ResolvedBearerToken)
        assert response.claims.subject == "alice@example.com"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("callout_status", "expected_status", "expected_detail"),
        [
            (401, 401, "Invalid or expired token"),
            (500, 503, "Access-key lifecycle validation unavailable"),
        ],
    )
    async def test_access_key_lifecycle_callout_rejects_token_or_unexpected_status(
        self,
        access_key_lifecycle_middleware,
        callout_status,
        expected_status,
        expected_detail,
    ):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(callout_status)

        async with access_key_lifecycle_middleware(handler) as middleware:
            response = await middleware._authenticate_access_key_lifecycle("scoped-access-key")

        assert isinstance(response, Response)
        assert response.status_code == expected_status
        assert response.body == f'{{"detail":"{expected_detail}"}}'.encode()

    @pytest.mark.asyncio
    async def test_access_key_lifecycle_callout_timeout_returns_504(self, access_key_lifecycle_middleware):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("timed out", request=request)

        async with access_key_lifecycle_middleware(handler) as middleware:
            response = await middleware._authenticate_access_key_lifecycle("scoped-access-key")

        assert isinstance(response, Response)
        assert response.status_code == 504
        assert response.body == b'{"detail":"Access-key lifecycle validation timeout"}'

    @pytest.mark.asyncio
    async def test_access_key_lifecycle_callout_opens_circuit_after_repeated_failures(
        self,
        access_key_lifecycle_middleware,
    ):
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            raise httpx.ConnectError("down", request=request)

        async with access_key_lifecycle_middleware(handler) as middleware:
            response = None
            for _ in range(ACCESS_KEY_LIFECYCLE_CIRCUIT_FAILURE_THRESHOLD):
                response = await middleware._authenticate_access_key_lifecycle("scoped-access-key")

            assert isinstance(response, Response)
            assert response.status_code == 503
            assert "retry-after" in response.headers

            circuit_response = await middleware._authenticate_access_key_lifecycle("scoped-access-key")

        assert calls == ACCESS_KEY_LIFECYCLE_CIRCUIT_FAILURE_THRESHOLD
        assert circuit_response is not None
        assert circuit_response.status_code == 503
        assert "retry-after" in circuit_response.headers

    @pytest.mark.asyncio
    async def test_access_key_lifecycle_malformed_success_opens_circuit(
        self,
        access_key_lifecycle_middleware,
    ):
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(
                200,
                json={
                    "token_kind": "access_key",
                    "principal": "alice@example.com",
                },
            )

        async with access_key_lifecycle_middleware(handler) as middleware:
            response = None
            for _ in range(ACCESS_KEY_LIFECYCLE_CIRCUIT_FAILURE_THRESHOLD):
                response = await middleware._authenticate_access_key_lifecycle("scoped-access-key")

            assert isinstance(response, Response)
            assert response.status_code == 503
            assert "retry-after" in response.headers

            circuit_response = await middleware._authenticate_access_key_lifecycle("scoped-access-key")

        assert calls == ACCESS_KEY_LIFECYCLE_CIRCUIT_FAILURE_THRESHOLD
        assert circuit_response.status_code == 503
        assert "retry-after" in circuit_response.headers

    @pytest.mark.asyncio
    async def test_access_key_lifecycle_rejects_wrong_typed_sdk_response(
        self,
        access_key_lifecycle_middleware,
    ):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "principal": 123,
                    "groups": [],
                    "scopes": [],
                    "jti": "ak_example",
                    "token_kind": "access_key",
                },
            )

        async with access_key_lifecycle_middleware(handler) as middleware:
            response = await middleware._authenticate_access_key_lifecycle("scoped-access-key")

        assert isinstance(response, Response)
        assert response.status_code == 503
        assert response.body == b'{"detail":"Access-key lifecycle validation unavailable"}'

    def test_bearer_token_request_uses_shared_resolver(self, auth_config_enabled):
        app = create_test_app(auth_config_enabled)
        client = TestClient(app, raise_server_exceptions=False)
        claims = TokenClaims(
            subject="alice@example.com",
            email="alice@example.com",
            groups=["team-ml"],
            scopes=["models:read"],
            raw_claims={},
        )
        resolved = ResolvedBearerToken(claims=claims, token_kind="oidc_access_token")

        with patch(
            "nmp.common.auth.middleware.resolve_bearer_token",
            new=AsyncMock(return_value=resolved),
        ) as resolver:
            with patch("nmp.common.auth.client.AuthClient.authorize_request") as mock_authorize:
                mock_authorize.return_value = MagicMock(allowed=True)
                response = client.get("/test", headers={"Authorization": "Bearer oidc-token"})

        assert response.status_code == 200
        resolver.assert_awaited_once()
        mock_authorize.assert_called_once()

    def test_access_key_bearer_uses_authenticate_callout_without_local_resolver(self, auth_config_oidc_disabled):
        app = FastAPI()

        @app.get("/whoami")
        async def whoami(auth_client: AuthClient = Depends(get_auth_client)):
            return {
                "principal": auth_client.principal.id,
                "email": auth_client.principal.email,
                "groups": auth_client.principal.groups,
            }

        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                200,
                json={
                    "principal": "alice@example.com",
                    "email": "alice@example.com",
                    "groups": ["team-ml"],
                    "scopes": ["models:read"],
                    "jti": "ak_example",
                    "token_kind": "access_key",
                },
            )

        config = auth_config_oidc_disabled.model_copy(
            update={"access_keys": auth_config_oidc_disabled.access_keys.model_copy(update={"enabled": True})}
        )
        Configuration.set_override(config)
        http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        app.add_middleware(
            AuthorizationMiddleware,
            service_name="test-service",
            http_client=http_client,
            access_key_lifecycle_http_client=http_client,
        )
        client = TestClient(app, raise_server_exceptions=False)
        token = jwt.encode(
            {
                "sub": "alice@example.com",
                "iat": int(time.time()),
                "nbf": int(time.time()),
                "jti": "ak_" + "a" * 32,
                "nmp_token_type": "access_key",
            },
            key="",
            algorithm="none",
        )

        try:
            with (
                patch(
                    "nmp.common.sdk_factory.Configuration.get_platform_config",
                    return_value=PlatformConfig(base_url="https://platform.example.com", services=""),
                ),
                patch("nmp.common.auth.middleware.resolve_bearer_token", new=AsyncMock()) as resolver,
                patch.object(AuthClient, "authorize_request", autospec=True) as mock_authorize,
            ):
                mock_authorize.return_value = MagicMock(allowed=True)
                response = client.get("/whoami", headers={"Authorization": f"Bearer {token}"})

            assert response.status_code == 200
            assert response.json() == {
                "principal": "alice@example.com",
                "email": "alice@example.com",
                "groups": ["team-ml"],
            }
            assert requests[0].url == httpx.URL("https://platform.example.com/apis/auth/authenticate")
            assert requests[0].headers["authorization"] == f"Bearer {token}"
            resolver.assert_not_awaited()
            mock_authorize.assert_called_once()
        finally:
            asyncio.run(http_client.aclose())

    def test_bearer_token_sets_auth_client_context_for_service_handler(self, auth_config_enabled):
        app = FastAPI()

        @app.get("/whoami")
        async def whoami(auth_client: AuthClient = Depends(get_auth_client)):
            principal = auth_client.principal
            effective_principal = principal.effective_principal
            return {
                "principal": principal.id,
                "email": principal.email,
                "groups": principal.groups,
                "on_behalf_of": principal.on_behalf_of,
                "on_behalf_of_email": principal.on_behalf_of_email,
                "on_behalf_of_groups": principal.on_behalf_of_groups,
                "effective_principal": effective_principal.id,
                "effective_email": effective_principal.email,
                "effective_groups": effective_principal.groups,
            }

        Configuration.set_override(auth_config_enabled)
        app.add_middleware(AuthorizationMiddleware, service_name="test-service")
        client = TestClient(app, raise_server_exceptions=False)
        claims = TokenClaims(
            subject="alice@example.com",
            email="alice@example.com",
            groups=["team-ml", "team-ai"],
            scopes=["models:read"],
            raw_claims={},
        )
        resolved = ResolvedBearerToken(claims=claims, token_kind="access_key")

        with (
            patch(
                "nmp.common.auth.middleware.resolve_bearer_token",
                new=AsyncMock(return_value=resolved),
            ) as resolver,
            patch.object(AuthClient, "authorize_request", autospec=True) as mock_authorize,
        ):
            mock_authorize.return_value = MagicMock(allowed=True)
            response = client.get("/whoami", headers={"Authorization": "Bearer scoped-access-key"})

        assert response.status_code == 200
        assert response.json() == {
            "principal": "alice@example.com",
            "email": "alice@example.com",
            "groups": ["team-ml", "team-ai"],
            "on_behalf_of": None,
            "on_behalf_of_email": None,
            "on_behalf_of_groups": None,
            "effective_principal": "alice@example.com",
            "effective_email": "alice@example.com",
            "effective_groups": ["team-ml", "team-ai"],
        }
        resolver.assert_awaited_once()
        mock_authorize.assert_called_once()
        assert mock_authorize.call_args.kwargs["scopes"] == ["models:read"]

    def test_bearer_token_with_actor_sets_delegated_auth_client_context(self, auth_config_enabled):
        app = FastAPI()

        @app.get("/whoami")
        async def whoami(auth_client: AuthClient = Depends(get_auth_client)):
            principal = auth_client.principal
            effective_principal = principal.effective_principal
            return {
                "principal": principal.id,
                "email": principal.email,
                "groups": principal.groups,
                "on_behalf_of": principal.on_behalf_of,
                "on_behalf_of_email": principal.on_behalf_of_email,
                "on_behalf_of_groups": principal.on_behalf_of_groups,
                "effective_principal": effective_principal.id,
                "effective_email": effective_principal.email,
                "effective_groups": effective_principal.groups,
            }

        Configuration.set_override(auth_config_enabled)
        app.add_middleware(AuthorizationMiddleware, service_name="test-service")
        client = TestClient(app, raise_server_exceptions=False)
        claims = TokenClaims(
            subject="creator@example.com",
            email="creator@example.com",
            groups=["workspace-editors"],
            scopes=["models:read"],
            raw_claims={},
            actor=ActorClaims(
                subject="system:serviceaccount:nemo-runs:job-runner",
                groups=["system:serviceaccounts"],
            ),
        )
        resolved = ResolvedBearerToken(claims=claims, token_kind="workload_access_token")

        with patch(
            "nmp.common.auth.middleware.resolve_bearer_token",
            new=AsyncMock(return_value=resolved),
        ) as resolver:
            with patch.object(AuthClient, "authorize_request", autospec=True) as mock_authorize:
                mock_authorize.return_value = MagicMock(allowed=True)
                response = client.get("/whoami", headers={"Authorization": "Bearer workload-token"})

        assert response.status_code == 200
        assert response.json() == {
            "principal": "system:serviceaccount:nemo-runs:job-runner",
            "email": None,
            "groups": ["system:serviceaccounts"],
            "on_behalf_of": "creator@example.com",
            "on_behalf_of_email": "creator@example.com",
            "on_behalf_of_groups": ["workspace-editors"],
            "effective_principal": "creator@example.com",
            "effective_email": "creator@example.com",
            "effective_groups": ["workspace-editors"],
        }
        resolver.assert_awaited_once()
        mock_authorize.assert_called_once()
        assert mock_authorize.call_args.kwargs["scopes"] == ["models:read"]

    def test_oidc_bearer_token_with_actor_uses_direct_auth_client_context(self, auth_config_enabled):
        app = FastAPI()

        @app.get("/whoami")
        async def whoami(auth_client: AuthClient = Depends(get_auth_client)):
            principal = auth_client.principal
            effective_principal = principal.effective_principal
            return {
                "principal": principal.id,
                "email": principal.email,
                "groups": principal.groups,
                "on_behalf_of": principal.on_behalf_of,
                "on_behalf_of_email": principal.on_behalf_of_email,
                "on_behalf_of_groups": principal.on_behalf_of_groups,
                "effective_principal": effective_principal.id,
                "effective_email": effective_principal.email,
                "effective_groups": effective_principal.groups,
            }

        Configuration.set_override(auth_config_enabled)
        app.add_middleware(AuthorizationMiddleware, service_name="test-service")
        client = TestClient(app, raise_server_exceptions=False)
        claims = TokenClaims(
            subject="creator@example.com",
            email="creator@example.com",
            groups=["workspace-editors"],
            scopes=["models:read"],
            raw_claims={"act": {"sub": "system:serviceaccount:nemo-runs:job-runner"}},
            actor=ActorClaims(
                subject="system:serviceaccount:nemo-runs:job-runner",
                groups=["system:serviceaccounts"],
            ),
        )
        resolved = ResolvedBearerToken(claims=claims, token_kind="oidc_access_token")

        with patch(
            "nmp.common.auth.middleware.resolve_bearer_token",
            new=AsyncMock(return_value=resolved),
        ) as resolver:
            with patch.object(AuthClient, "authorize_request", autospec=True) as mock_authorize:
                mock_authorize.return_value = MagicMock(allowed=True)
                response = client.get("/whoami", headers={"Authorization": "Bearer oidc-token"})

        assert response.status_code == 200
        assert response.json() == {
            "principal": "creator@example.com",
            "email": "creator@example.com",
            "groups": ["workspace-editors"],
            "on_behalf_of": None,
            "on_behalf_of_email": None,
            "on_behalf_of_groups": None,
            "effective_principal": "creator@example.com",
            "effective_email": "creator@example.com",
            "effective_groups": ["workspace-editors"],
        }
        resolver.assert_awaited_once()
        mock_authorize.assert_called_once()
        assert mock_authorize.call_args.kwargs["scopes"] == ["models:read"]

    def test_auth_jwks_path_bypasses_auth(self, auth_config_enabled):
        app = FastAPI()

        @app.get("/apis/auth/jwks")
        async def jwks():
            return {"keys": []}

        Configuration.set_override(auth_config_enabled)
        app.add_middleware(AuthorizationMiddleware, service_name="test-service")

        client = TestClient(app, raise_server_exceptions=False)
        with patch("nmp.common.auth.client.AuthClient.authorize_request") as mock_authorize:
            response = client.get("/apis/auth/jwks")

        assert response.status_code == 200
        mock_authorize.assert_not_called()

    def test_access_key_specific_jwks_path_is_not_a_health_bypass(self):
        assert "/apis/auth/v2/access-keys/jwks" not in HEALTH_ENDPOINTS

    def test_authenticate_path_bypasses_auth(self, auth_config_enabled):
        app = FastAPI()

        @app.post("/apis/auth/authenticate")
        async def authenticate():
            return {"ok": True}

        Configuration.set_override(auth_config_enabled)
        app.add_middleware(AuthorizationMiddleware, service_name="test-service")

        client = TestClient(app, raise_server_exceptions=False)
        with patch("nmp.common.auth.client.AuthClient.authorize_request") as mock_authorize:
            response = client.post("/apis/auth/authenticate")

        assert response.status_code == 200
        mock_authorize.assert_not_called()

    def test_ext_authz_path_bypasses_auth(self, auth_config_enabled):
        app = FastAPI()

        @app.post("/apis/auth/ext-authz")
        async def ext_authz():
            return {"ok": True}

        Configuration.set_override(auth_config_enabled)
        app.add_middleware(AuthorizationMiddleware, service_name="test-service")

        client = TestClient(app, raise_server_exceptions=False)
        with patch("nmp.common.auth.client.AuthClient.authorize_request") as mock_authorize:
            response = client.post("/apis/auth/ext-authz")

        assert response.status_code == 200
        mock_authorize.assert_not_called()

    def test_ext_authz_prefixed_callout_path_bypasses_auth(self, auth_config_enabled):
        app = FastAPI()

        @app.delete("/apis/auth/ext-authz/apis/entities/v2/workspaces/default")
        async def ext_authz_prefixed():
            return {"ok": True}

        Configuration.set_override(auth_config_enabled)
        app.add_middleware(AuthorizationMiddleware, service_name="test-service")

        client = TestClient(app, raise_server_exceptions=False)
        with patch("nmp.common.auth.client.AuthClient.authorize_request") as mock_authorize:
            response = client.delete("/apis/auth/ext-authz/apis/entities/v2/workspaces/default")

        assert response.status_code == 200
        mock_authorize.assert_not_called()

    def test_authenticate_prefixed_path_is_not_a_callout_bypass(self, auth_config_enabled):
        app = FastAPI()

        @app.delete("/apis/auth/authenticate/apis/entities/v2/workspaces/default")
        async def authenticate_prefixed():
            return {"ok": True}

        Configuration.set_override(auth_config_enabled)
        app.add_middleware(AuthorizationMiddleware, service_name="test-service")

        client = TestClient(app, raise_server_exceptions=False)
        with patch("nmp.common.auth.client.AuthClient.authorize_request") as mock_authorize:
            mock_authorize.return_value = MagicMock(allowed=True)
            response = client.delete("/apis/auth/authenticate/apis/entities/v2/workspaces/default")

        assert response.status_code == 200
        mock_authorize.assert_called_once()


class TestPrincipalHeadersAuth:
    """Tests for X-NMP-Principal-* header authentication."""

    def test_principal_headers_with_principal_id(self, auth_config_enabled):
        """Principal is taken from X-NMP-Principal-Id when auth is enabled."""
        app = create_test_app(auth_config_enabled)
        client = TestClient(app, raise_server_exceptions=False)

        with patch("nmp.common.auth.client.AuthClient.authorize_request") as mock_authorize:
            mock_authorize.return_value = MagicMock(allowed=True)

            response = client.get(
                "/test",
                headers={"X-NMP-Principal-Id": "user@example.com"},
            )

            assert response.status_code == 200

    def test_principal_headers_auth_disabled(self, auth_config_disabled):
        """X-NMP-Principal-* headers still set principal when auth is disabled."""
        app = create_test_app(auth_config_disabled)
        client = TestClient(app, raise_server_exceptions=False)

        response = client.get(
            "/test",
            headers={"X-NMP-Principal-Id": "user@example.com"},
        )

        # Should succeed because auth is disabled
        assert response.status_code == 200


class TestServicePrincipalAuth:
    """Tests for service principal authentication."""

    def test_service_principal_uses_pdp(self, auth_config_enabled):
        """Service principals are authorized via PDP (same path as X-NMP-Principal-* users)."""
        app = create_test_app(auth_config_enabled)
        client = TestClient(app, raise_server_exceptions=False)

        with patch("nmp.common.auth.client.AuthClient.authorize_request") as mock_authorize:
            mock_authorize.return_value = MagicMock(allowed=True)

            response = client.get(
                "/test",
                headers={"X-NMP-Principal-Id": "service:my-service"},
            )

            assert response.status_code == 200
            mock_authorize.assert_called_once()


class TestCompatibilityAuth:
    """Tests for compatibility auth paths used by non-standard clients."""

    @pytest.mark.parametrize(
        "token,expected_status",
        [
            ("service:nim", 200),
            ("service:customizer", 200),
            ("invalid-token", 401),
        ],
    )
    def test_hf_endpoint_bearer_token(self, auth_config_enabled, token, expected_status):
        """HF endpoints accept service:* Bearer tokens, reject invalid tokens."""
        app = create_test_app(auth_config_enabled)
        client = TestClient(app, raise_server_exceptions=False)

        with patch("nmp.common.auth.client.AuthClient.authorize_request") as mock_authorize:
            mock_authorize.return_value = MagicMock(allowed=expected_status == 200)

            response = client.get(
                "/apis/files/v2/hf/my-workspace/my-fileset/resolve/main/model.bin",
                headers={"Authorization": f"Bearer {token}"},
            )

            assert response.status_code == expected_status
            if expected_status == 200:
                mock_authorize.assert_called_once()

    def test_hf_endpoint_authorizes_as_bearer_service_principal(self, auth_config_enabled):
        """The PDP receives the service principal synthesized from the HF Bearer token."""
        app = create_test_app(auth_config_enabled)
        client = TestClient(app, raise_server_exceptions=False)

        with patch.object(AuthClient, "authorize_request", autospec=True) as mock_authorize:
            mock_authorize.return_value = MagicMock(allowed=True)

            response = client.get(
                "/apis/files/v2/hf/my-workspace/my-fileset/resolve/main/model.bin",
                headers={"Authorization": "Bearer service:models"},
            )

        assert response.status_code == 200
        auth_client = mock_authorize.call_args.args[0]
        assert auth_client.principal.id == "service:models"

    def test_files_otlp_logs_upload_accepts_service_principal_header(self, auth_config_enabled):
        """Job log uploads accept the launcher fallback service principal header."""
        app = create_test_app(auth_config_enabled)
        client = TestClient(app, raise_server_exceptions=False)

        with patch.object(AuthClient, "authorize_request", autospec=True) as mock_authorize:
            mock_authorize.return_value = MagicMock(allowed=True)

            response = client.post(
                "/apis/files/v2/workspaces/my-workspace/filesets/job-fileset-1/otlp/v1/logs",
                headers={"X-NMP-Principal-Id": "service:jobs"},
            )

        assert response.status_code == 200
        auth_client = mock_authorize.call_args.args[0]
        assert auth_client.principal.id == "service:jobs"

    def test_files_otlp_logs_upload_does_not_accept_service_bearer_token(self, auth_config_enabled):
        """Job log uploads use X-NMP service headers, not raw service bearer tokens."""
        app = create_test_app(auth_config_enabled)
        client = TestClient(app, raise_server_exceptions=False)

        with patch("nmp.common.auth.jwt.JWTValidator.validate_token") as mock_validate:
            mock_validate.return_value = None
            with patch("nmp.common.auth.client.AuthClient.authorize_request") as mock_authorize:
                response = client.post(
                    "/apis/files/v2/workspaces/my-workspace/filesets/job-fileset-1/otlp/v1/logs",
                    headers={"Authorization": "Bearer service:jobs"},
                )

        assert response.status_code == 401
        mock_validate.assert_called_once()
        mock_authorize.assert_not_called()

    def test_regular_endpoint_does_not_accept_service_bearer_token(self, auth_config_enabled):
        """Raw service bearer tokens are not accepted on arbitrary routes."""
        app = create_test_app(auth_config_enabled)
        client = TestClient(app, raise_server_exceptions=False)

        with patch("nmp.common.auth.jwt.JWTValidator.validate_token") as mock_validate:
            mock_validate.return_value = None
            with patch("nmp.common.auth.client.AuthClient.authorize_request") as mock_authorize:
                response = client.get(
                    "/test",
                    headers={"Authorization": "Bearer service:jobs"},
                )

        assert response.status_code == 401
        mock_validate.assert_called_once()
        mock_authorize.assert_not_called()


class TestInternalServiceOnlyRoutes:
    """IAM role-bindings and nested Entities APIs require service principals."""

    def test_iam_role_bindings_forbidden_for_user_principal(self, auth_config_enabled):
        app = create_test_app_with_platform_routes(auth_config_enabled)
        client = TestClient(app, raise_server_exceptions=False)
        with patch("nmp.common.auth.client.AuthClient.authorize_request") as mock_authorize:
            mock_authorize.return_value = MagicMock(allowed=False)
            response = client.get(
                "/apis/auth/v2/iam/role-bindings",
                headers={"X-NMP-Principal-Id": "user@example.com"},
            )
        assert response.status_code == 403

    def test_iam_role_bindings_allowed_for_service_principal(self, auth_config_enabled):
        app = create_test_app_with_platform_routes(auth_config_enabled)
        client = TestClient(app, raise_server_exceptions=False)
        with patch("nmp.common.auth.client.AuthClient.authorize_request") as mock_authorize:
            mock_authorize.return_value = MagicMock(allowed=True)
            response = client.get(
                "/apis/auth/v2/iam/role-bindings",
                headers={"X-NMP-Principal-Id": "service:integration-test"},
            )
        assert response.status_code == 200

    def test_nested_entities_forbidden_for_user_principal(self, auth_config_enabled):
        app = create_test_app_with_platform_routes(auth_config_enabled)
        client = TestClient(app, raise_server_exceptions=False)
        with patch("nmp.common.auth.client.AuthClient.authorize_request") as mock_authorize:
            mock_authorize.return_value = MagicMock(allowed=False)
            response = client.get(
                "/apis/entities/v2/workspaces/ws1/entities/evaluation_config",
                headers={"X-NMP-Principal-Id": "user@example.com"},
            )
        assert response.status_code == 403

    def test_workspace_list_allowed_for_user_principal(self, auth_config_enabled):
        app = create_test_app_with_platform_routes(auth_config_enabled)
        client = TestClient(app, raise_server_exceptions=False)
        with patch("nmp.common.auth.client.AuthClient.authorize_request") as mock_authorize:
            mock_authorize.return_value = MagicMock(allowed=True)
            response = client.get(
                "/apis/entities/v2/workspaces",
                headers={"X-NMP-Principal-Id": "user@example.com"},
            )
        assert response.status_code == 200

    def test_internal_routes_skipped_when_auth_disabled(self, auth_config_disabled):
        app = create_test_app_with_platform_routes(auth_config_disabled)
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get(
            "/apis/auth/v2/iam/role-bindings",
            headers={"X-NMP-Principal-Id": "user@example.com"},
        )
        assert response.status_code == 200


class TestServicePrincipalDelegationMiddleware:
    """Service + on-behalf-of headers: AuthClient principal exposes effective delegate claims."""

    def test_principal_parsed_for_delegation_has_effective_delegate_attributes(self, auth_config_enabled):
        app = create_test_app(auth_config_enabled)
        client = TestClient(app, raise_server_exceptions=False)
        principal_seen: dict = {}

        original_from_headers = Principal.from_headers

        def capturing_from_headers(headers):
            p = original_from_headers(headers)
            principal_seen["p"] = p
            return p

        with (
            patch.object(Principal, "from_headers", side_effect=capturing_from_headers),
            patch.object(AuthClient, "authorize_request", new_callable=AsyncMock) as mock_authorize,
        ):
            mock_authorize.return_value = MagicMock(allowed=True)
            response = client.get(
                "/test",
                headers={
                    "x-nmp-principal-id": "service:worker",
                    "x-nmp-principal-on-behalf-of": "user@example.com",
                    "x-nmp-principal-on-behalf-of-groups": "ws-editors",
                    "x-nmp-principal-on-behalf-of-email": "user@example.com",
                },
            )

        assert response.status_code == 200
        p = principal_seen["p"]
        assert p is not None
        assert p.id == "service:worker"
        assert p.on_behalf_of == "user@example.com"
        assert p.effective_groups == ["ws-editors"]
        assert p.effective_email == "user@example.com"

    def test_service_delegation_returns_403_when_pdp_denies(self, auth_config_enabled):
        app = create_test_app(auth_config_enabled)
        client = TestClient(app, raise_server_exceptions=False)

        mock_authorize = AsyncMock(return_value=MagicMock(allowed=False))

        with patch.object(AuthClient, "authorize_request", mock_authorize):
            response = client.get(
                "/test",
                headers={
                    "x-nmp-principal-id": "service:worker",
                    "x-nmp-principal-on-behalf-of": "user@example.com",
                    "x-nmp-principal-on-behalf-of-groups": "no-access",
                },
            )

        assert response.status_code == 403
