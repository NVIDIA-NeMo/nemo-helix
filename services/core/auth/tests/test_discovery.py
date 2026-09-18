# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for discovery endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from nmp.common.config import AuthConfig, Configuration
from nmp.common.config.base import OIDCConfig
from nmp.core.auth.api.v2.discovery.endpoints import (
    AuthDiscoveryResponse,
    OIDCDiscoveryResponse,
    _clear_idp_discovery_cache,
    _fetch_idp_discovery,
    get_auth_discovery,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear the module-level IdP discovery cache between tests."""
    _clear_idp_discovery_cache()
    yield
    _clear_idp_discovery_cache()


@pytest.fixture
def oidc_config():
    """Create an OIDC config for testing."""
    return OIDCConfig(
        enabled=True,
        issuer="https://sso.example.com",
        client_id="test-client",
        authorization_endpoint="https://sso.example.com/authorize",
        token_endpoint="https://sso.example.com/token",
        device_authorization_endpoint="https://sso.example.com/device/code",
        userinfo_endpoint="https://sso.example.com/userinfo",
        workload_token_exchange_enabled=True,
        workload_client_id="test-workload-client",
        workload_token_endpoint="https://workload-idp.example.com/token",
        workload_audience="nemo-platform",
        workload_scope="openid email groups",
    )


@pytest.fixture
def auth_config_oidc_enabled(oidc_config):
    """Create an AuthConfig with auth and OIDC enabled."""
    return AuthConfig(
        enabled=True,
        policy_decision_point_base_url="http://localhost:8181",
        oidc=oidc_config,
    )


@pytest.fixture
def auth_config_oidc_disabled():
    """Create an AuthConfig with OIDC disabled."""
    return AuthConfig(
        enabled=True,
        policy_decision_point_base_url="http://localhost:8181",
        oidc=OIDCConfig(enabled=False),
    )


@pytest.fixture
def auth_config_disabled():
    """Create an AuthConfig with auth disabled."""
    return AuthConfig(
        enabled=False,
        policy_decision_point_base_url="http://localhost:8181",
        oidc=OIDCConfig(enabled=False),
    )


class TestOIDCDiscoveryResponse:
    """Tests for OIDCDiscoveryResponse model."""

    def test_oidc_discovery_response_creation(self):
        """Test creating an OIDCDiscoveryResponse instance."""
        response = OIDCDiscoveryResponse(
            issuer="https://sso.example.com",
            authorization_endpoint="https://sso.example.com/authorize",
            token_endpoint="https://sso.example.com/token",
            device_authorization_endpoint="https://sso.example.com/device/code",
            userinfo_endpoint="https://sso.example.com/userinfo",
            client_id="test-client",
        )

        assert response.issuer == "https://sso.example.com"
        assert response.authorization_endpoint == "https://sso.example.com/authorize"
        assert response.token_endpoint == "https://sso.example.com/token"
        assert response.device_authorization_endpoint == "https://sso.example.com/device/code"
        assert response.userinfo_endpoint == "https://sso.example.com/userinfo"
        assert response.client_id == "test-client"

    def test_oidc_discovery_response_optional_fields(self):
        """Test OIDCDiscoveryResponse with optional fields."""
        response = OIDCDiscoveryResponse(
            issuer="https://sso.example.com",
            client_id="test-client",
        )

        assert response.issuer == "https://sso.example.com"
        assert response.authorization_endpoint is None
        assert response.token_endpoint is None
        assert response.device_authorization_endpoint is None
        assert response.userinfo_endpoint is None

    def test_oidc_discovery_response_includes_workload_exchange_fields(self):
        """Test OIDCDiscoveryResponse includes workload identity token exchange fields."""
        response = OIDCDiscoveryResponse(
            issuer="https://sso.example.com",
            client_id="test-client",
            token_endpoint="https://sso.example.com/token",
            workload_token_exchange_enabled=True,
            workload_client_id="test-workload-client",
            workload_token_endpoint="https://workload-idp.example.com/token",
            workload_audience="nemo-platform",
            workload_scope="openid email groups",
        )

        assert response.workload_token_exchange_enabled is True
        assert response.workload_client_id == "test-workload-client"
        assert response.workload_token_endpoint == "https://workload-idp.example.com/token"
        assert response.workload_audience == "nemo-platform"
        assert response.workload_scope == "openid email groups"


class TestAuthDiscoveryResponse:
    """Tests for AuthDiscoveryResponse model."""

    def test_auth_discovery_response_with_oidc(self):
        """Test AuthDiscoveryResponse with OIDC config."""
        oidc = OIDCDiscoveryResponse(
            issuer="https://sso.example.com",
            client_id="test-client",
        )

        response = AuthDiscoveryResponse(
            auth_enabled=True,
            oidc=oidc,
        )

        assert response.auth_enabled is True
        assert response.oidc is not None
        assert response.oidc.issuer == "https://sso.example.com"

    def test_auth_discovery_response_without_oidc(self):
        """Test AuthDiscoveryResponse without OIDC config."""
        response = AuthDiscoveryResponse(
            auth_enabled=False,
            oidc=None,
        )

        assert response.auth_enabled is False
        assert response.oidc is None


class TestGetAuthDiscovery:
    """Tests for get_auth_discovery endpoint."""

    @pytest.mark.asyncio
    async def test_auth_enabled_oidc_enabled_with_configured_endpoints(self, auth_config_oidc_enabled):
        """Test response when auth and OIDC are enabled with configured endpoints."""
        Configuration.set_override(auth_config_oidc_enabled)

        try:
            result = await get_auth_discovery()

            assert result.auth_enabled is True
            assert result.oidc is not None
            assert result.oidc.issuer == "https://sso.example.com"
            assert result.oidc.client_id == "test-client"
            assert result.oidc.authorization_endpoint == "https://sso.example.com/authorize"
            assert result.oidc.token_endpoint == "https://sso.example.com/token"
            assert result.oidc.device_authorization_endpoint == "https://sso.example.com/device/code"
            assert result.oidc.userinfo_endpoint == "https://sso.example.com/userinfo"
            assert result.oidc.workload_token_exchange_enabled is True
            assert result.oidc.workload_client_id == "test-workload-client"
            assert result.oidc.workload_token_endpoint == "https://workload-idp.example.com/token"
            assert result.oidc.workload_audience == "nemo-platform"
            assert result.oidc.workload_scope == "openid email groups"
        finally:
            Configuration.clear_overrides()

    @pytest.mark.asyncio
    async def test_auth_enabled_oidc_disabled(self, auth_config_oidc_disabled):
        """Test response when auth is enabled but OIDC is disabled."""
        Configuration.set_override(auth_config_oidc_disabled)

        try:
            result = await get_auth_discovery()

            assert result.auth_enabled is True
            assert result.oidc is None
        finally:
            Configuration.clear_overrides()

    @pytest.mark.asyncio
    async def test_workload_token_endpoint_defaults_to_platform_auth_endpoint(self):
        """Test workload exchange endpoint defaults to the NeMo auth service."""
        oidc_config = OIDCConfig(
            enabled=True,
            issuer="https://sso.example.com",
            client_id="test-client",
            workload_token_exchange_enabled=True,
            workload_client_id="test-workload-client",
        )
        auth_config = AuthConfig(
            enabled=True,
            policy_decision_point_base_url="http://localhost:8181",
            oidc=oidc_config,
        )
        Configuration.set_override(auth_config)

        try:
            result = await get_auth_discovery()

            assert result.oidc is not None
            assert result.oidc.workload_token_endpoint == "http://localhost:8080/apis/auth/token"
        finally:
            Configuration.clear_overrides()

    @pytest.mark.asyncio
    async def test_auth_disabled(self, auth_config_disabled):
        """Test response when auth is disabled."""
        Configuration.set_override(auth_config_disabled)

        try:
            result = await get_auth_discovery()

            assert result.auth_enabled is False
            assert result.oidc is None
        finally:
            Configuration.clear_overrides()

    @pytest.mark.asyncio
    async def test_oidc_discovery_fetches_endpoints(self):
        """Test that OIDC discovery fetches endpoints when not configured."""
        oidc_config = OIDCConfig(
            enabled=True,
            issuer="https://sso.example.com",
            client_id="test-client",
            # No endpoints configured - should be fetched from discovery
        )

        auth_config = AuthConfig(
            enabled=True,
            policy_decision_point_base_url="http://localhost:8181",
            oidc=oidc_config,
        )

        Configuration.set_override(auth_config)

        discovery_doc = {
            "authorization_endpoint": "https://sso.example.com/auth",
            "token_endpoint": "https://sso.example.com/token",
            "device_authorization_endpoint": "https://sso.example.com/device",
            "userinfo_endpoint": "https://sso.example.com/userinfo",
        }

        try:
            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_response = MagicMock()
                mock_response.is_success = True
                mock_response.json.return_value = discovery_doc
                mock_client.get.return_value = mock_response
                mock_client.__aenter__.return_value = mock_client
                mock_client.__aexit__.return_value = None
                mock_client_class.return_value = mock_client

                result = await get_auth_discovery()

                assert result.oidc is not None
                assert result.oidc.authorization_endpoint == "https://sso.example.com/auth"
                assert result.oidc.token_endpoint == "https://sso.example.com/token"
                assert result.oidc.device_authorization_endpoint == "https://sso.example.com/device"
                assert result.oidc.userinfo_endpoint == "https://sso.example.com/userinfo"
        finally:
            Configuration.clear_overrides()

    @pytest.mark.asyncio
    async def test_oidc_discovery_failure_uses_configured_values(self, auth_config_oidc_enabled):
        """Test that discovery failure falls back to configured values."""
        Configuration.set_override(auth_config_oidc_enabled)

        try:
            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.get.side_effect = httpx.HTTPError("Connection failed")
                mock_client.__aenter__.return_value = mock_client
                mock_client.__aexit__.return_value = None
                mock_client_class.return_value = mock_client

                result = await get_auth_discovery()

                # Should still return configured endpoints
                assert result.oidc is not None
                assert result.oidc.authorization_endpoint == "https://sso.example.com/authorize"
                assert result.oidc.token_endpoint == "https://sso.example.com/token"
        finally:
            Configuration.clear_overrides()

    @pytest.mark.asyncio
    async def test_configured_endpoints_take_priority_over_discovery(self):
        """Test that configured endpoints take priority over discovered ones."""
        oidc_config = OIDCConfig(
            enabled=True,
            issuer="https://sso.example.com",
            client_id="test-client",
            authorization_endpoint="https://custom.example.com/authorize",  # Configured
            # token_endpoint not configured - should use discovery
        )

        auth_config = AuthConfig(
            enabled=True,
            policy_decision_point_base_url="http://localhost:8181",
            oidc=oidc_config,
        )

        Configuration.set_override(auth_config)

        discovery_doc = {
            "authorization_endpoint": "https://sso.example.com/auth",  # Should be ignored
            "token_endpoint": "https://sso.example.com/token",  # Should be used
        }

        try:
            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_response = MagicMock()
                mock_response.is_success = True
                mock_response.json.return_value = discovery_doc
                mock_client.get.return_value = mock_response
                mock_client.__aenter__.return_value = mock_client
                mock_client.__aexit__.return_value = None
                mock_client_class.return_value = mock_client

                result = await get_auth_discovery()

                assert result.oidc is not None
                # Configured value takes priority
                assert result.oidc.authorization_endpoint == "https://custom.example.com/authorize"
                # Discovery value is used for unconfigured endpoint
                assert result.oidc.token_endpoint == "https://sso.example.com/token"
        finally:
            Configuration.clear_overrides()

    @pytest.mark.asyncio
    async def test_configured_userinfo_endpoint_used_when_discovery_omits_it(self):
        """A configured userinfo_endpoint override must surface even if the IdP's discovery
        document doesn't advertise one (matches the other endpoint overrides)."""
        oidc_config = OIDCConfig(
            enabled=True,
            issuer="https://sso.example.com",
            client_id="test-client",
            userinfo_endpoint="https://custom.example.com/userinfo",
        )

        auth_config = AuthConfig(
            enabled=True,
            policy_decision_point_base_url="http://localhost:8181",
            oidc=oidc_config,
        )

        Configuration.set_override(auth_config)

        discovery_doc = {
            "authorization_endpoint": "https://sso.example.com/auth",
            "token_endpoint": "https://sso.example.com/token",
            # userinfo_endpoint intentionally omitted from discovery
        }

        try:
            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_response = MagicMock()
                mock_response.is_success = True
                mock_response.json.return_value = discovery_doc
                mock_client.get.return_value = mock_response
                mock_client.__aenter__.return_value = mock_client
                mock_client.__aexit__.return_value = None
                mock_client_class.return_value = mock_client

                result = await get_auth_discovery()

                assert result.oidc is not None
                assert result.oidc.userinfo_endpoint == "https://custom.example.com/userinfo"
        finally:
            Configuration.clear_overrides()


class TestIdpDiscoveryCache:
    """Tests for IdP discovery document caching."""

    @pytest.mark.asyncio
    async def test_cache_hit_avoids_http_call(self):
        """Test that a second call within TTL returns cached data."""
        discovery_doc = {
            "authorization_endpoint": "https://sso.example.com/auth",
            "token_endpoint": "https://sso.example.com/token",
        }

        with patch("nmp.core.auth.api.v2.discovery.endpoints.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.is_success = True
            mock_response.json.return_value = discovery_doc
            mock_client.get.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            result1 = await _fetch_idp_discovery("https://sso.example.com", cache_ttl=300)
            result2 = await _fetch_idp_discovery("https://sso.example.com", cache_ttl=300)

            assert result1 == discovery_doc
            assert result2 == discovery_doc
            assert mock_client.get.call_count == 1

    @pytest.mark.asyncio
    async def test_cache_miss_after_ttl_expiry(self):
        """Test that expired cache triggers a new HTTP call."""
        import nmp.core.auth.api.v2.discovery.endpoints as mod

        discovery_v1 = {"token_endpoint": "https://sso.example.com/token-v1"}
        discovery_v2 = {"token_endpoint": "https://sso.example.com/token-v2"}

        with patch("nmp.core.auth.api.v2.discovery.endpoints.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.is_success = True
            mock_response.json.return_value = discovery_v1
            mock_client.get.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            result1 = await _fetch_idp_discovery("https://sso.example.com", cache_ttl=300)
            assert result1 == discovery_v1

            # Simulate TTL expiry
            mod._idp_discovery_cache_time -= 600

            mock_response.json.return_value = discovery_v2
            result2 = await _fetch_idp_discovery("https://sso.example.com", cache_ttl=300)

            assert result2 == discovery_v2
            assert mock_client.get.call_count == 2

    @pytest.mark.asyncio
    async def test_fetch_failure_returns_stale_cache(self):
        """Test graceful degradation: stale cache is served on fetch failure."""
        import nmp.core.auth.api.v2.discovery.endpoints as mod

        discovery_doc = {"token_endpoint": "https://sso.example.com/token"}

        with patch("nmp.core.auth.api.v2.discovery.endpoints.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.is_success = True
            mock_response.json.return_value = discovery_doc
            mock_client.get.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            # Populate the cache
            result1 = await _fetch_idp_discovery("https://sso.example.com", cache_ttl=300)
            assert result1 == discovery_doc

            # Expire the cache and make the next fetch fail
            mod._idp_discovery_cache_time -= 600
            mock_client.get.side_effect = httpx.HTTPError("Connection refused")

            result2 = await _fetch_idp_discovery("https://sso.example.com", cache_ttl=300)

            # Should return stale cached data
            assert result2 == discovery_doc

    @pytest.mark.asyncio
    async def test_fetch_failure_without_cache_returns_empty(self):
        """Test that fetch failure with no prior cache returns empty dict."""
        with patch("nmp.core.auth.api.v2.discovery.endpoints.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get.side_effect = httpx.HTTPError("Connection refused")
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            result = await _fetch_idp_discovery("https://sso.example.com", cache_ttl=300)
            assert result == {}

    @pytest.mark.asyncio
    async def test_cache_disabled_with_zero_ttl(self):
        """Test that setting TTL to 0 disables caching."""
        discovery_doc = {"token_endpoint": "https://sso.example.com/token"}

        with patch("nmp.core.auth.api.v2.discovery.endpoints.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.is_success = True
            mock_response.json.return_value = discovery_doc
            mock_client.get.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            await _fetch_idp_discovery("https://sso.example.com", cache_ttl=0)
            await _fetch_idp_discovery("https://sso.example.com", cache_ttl=0)

            # Both calls should hit the network
            assert mock_client.get.call_count == 2
