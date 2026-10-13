# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for auth discovery."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from nemo_helix_plugin.config import Runtime
from nhx.common.auth.discovery import AuthDiscoveryResponse, PublicOidcAdvertisedClient
from nhx.common.config import AuthConfig, Configuration, HelixConfig
from nhx.common.config.base import (
    OIDCConfidentialClientConfig,
    OIDCConfig,
    OIDCPublicClientConfig,
    OIDCServerSessionsConfig,
    OIDCWorkloadConfig,
    TokenSigningConfig,
)
from nhx.core.auth.api.v2.discovery.endpoints import (
    _clear_idp_discovery_cache,
    _fetch_idp_discovery,
    get_auth_discovery,
)


@pytest.fixture(autouse=True)
def _clear_state():
    _clear_idp_discovery_cache()
    Configuration.clear_overrides()
    yield
    _clear_idp_discovery_cache()
    Configuration.clear_overrides()


def _public_client(**overrides: object) -> OIDCPublicClientConfig:
    values: dict[str, object] = {
        "client_id": "test-public",
        "authorization_endpoint": "https://sso.example.com/authorize",
        "token_endpoint": "https://sso.example.com/token",
        "device_authorization_endpoint": "https://sso.example.com/device/code",
        "bearer_token_source": "id_token",
        "default_scopes": "openid email offline_access",
        "device_authorization_requires_device_id": True,
        "device_authorization_display_name": "NeMo Helix CLI",
        "device_token_request_includes_scope": False,
    }
    values.update(overrides)
    return OIDCPublicClientConfig.model_validate(values)


def _confidential_client(**overrides: object) -> OIDCConfidentialClientConfig:
    values: dict[str, object] = {
        "client_id": "test-confidential",
        "client_secret_env_var": "NHX_OIDC_CLIENT_SECRET",
        "login_redirect_uri": "https://127.0.0.1:18082/apis/auth/v2/login/callback",
        "authorization_endpoint": "https://sso.example.com/authorize",
        "token_endpoint": "http://idp.internal/token",
        "default_scopes": "openid email offline_access",
    }
    values.update(overrides)
    return OIDCConfidentialClientConfig.model_validate(values)


def _set_config(oidc: OIDCConfig, *, advertised_base_url: str | None = "https://127.0.0.1:18082") -> None:
    auth = AuthConfig(
        enabled=True,
        policy_decision_point_base_url="http://localhost:8181",
        token_signing=TokenSigningConfig(private_key_file="/var/run/secrets/nemo-helix/private.pem"),
        oidc=oidc,
    )
    platform = HelixConfig(
        runtime=Runtime.NONE,
        base_url="http://nemo-api:8080",
        advertised_base_url=advertised_base_url,
    )
    Configuration.set_overrides({AuthConfig: auth, HelixConfig: platform})


@pytest.mark.asyncio
async def test_discovery_advertises_self_contained_public_and_confidential_clients() -> None:
    _set_config(
        OIDCConfig(
            enabled=True,
            issuer="https://sso.example.com",
            public_client=_public_client(),
            confidential_client=_confidential_client(),
            server_sessions=OIDCServerSessionsConfig(encryption_key_env_var="NHX_AUTH_SESSION_ENCRYPTION_KEY"),
        )
    )

    result = await get_auth_discovery()

    assert result.auth_enabled is True
    assert result.oidc is not None
    clients = {client.name: client.model_dump() for client in result.oidc.clients}
    assert clients["public"] == {
        "name": "public",
        "client_id": "test-public",
        "client_authentication": "public",
        "default": True,
        "server_side_sessions": False,
        "default_scopes": "openid email offline_access",
        "bearer_token_source": "id_token",
        "scope_prefix": None,
        "authorization_endpoint": "https://sso.example.com/authorize",
        "token_endpoint": "https://sso.example.com/token",
        "device_authorization_endpoint": "https://sso.example.com/device/code",
        "device_authorization_requires_device_id": True,
        "device_authorization_display_name": "NeMo Helix CLI",
        "device_token_request_includes_scope": False,
        "authorization_start_endpoint": None,
        "broker_token_endpoint": None,
    }
    assert clients["confidential"] == {
        "name": "confidential",
        "client_id": "test-confidential",
        "client_authentication": "client_secret_basic",
        "default": False,
        "default_scopes": "openid email offline_access",
        "bearer_token_source": "access_token",
        "scope_prefix": None,
        "authorization_start_endpoint": "https://127.0.0.1:18082/apis/auth/v2/authorize",
        "broker_token_endpoint": "https://127.0.0.1:18082/apis/auth/v2/token",
    }
    response_fields = result.oidc.model_dump().keys()
    assert "client_id" not in response_fields
    assert "client_authentication" not in response_fields
    assert "public_client_id" not in response_fields
    assert "token_authorization_url" not in response_fields
    serialized = result.model_dump_json()
    assert "NHX_OIDC_CLIENT_SECRET" not in serialized
    assert "NHX_AUTH_SESSION_ENCRYPTION_KEY" not in serialized
    assert "http://idp.internal/token" not in serialized


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("public", "confidential", "expected_name"),
    [
        (_public_client(), None, "public"),
        (None, _confidential_client(), "confidential"),
    ],
)
async def test_single_interactive_profile_is_default(
    public: OIDCPublicClientConfig | None,
    confidential: OIDCConfidentialClientConfig | None,
    expected_name: str,
) -> None:
    _set_config(
        OIDCConfig(
            enabled=True,
            issuer="https://sso.example.com",
            public_client=public,
            confidential_client=confidential,
            server_sessions=OIDCServerSessionsConfig(encryption_key_env_var="NHX_AUTH_SESSION_ENCRYPTION_KEY"),
        )
    )

    result = await get_auth_discovery()

    assert result.oidc is not None
    assert [(client.name, client.default) for client in result.oidc.clients] == [(expected_name, True)]


@pytest.mark.asyncio
async def test_public_endpoints_fall_back_to_provider_discovery() -> None:
    _set_config(
        OIDCConfig(
            enabled=True,
            issuer="https://sso.example.com",
            public_client=_public_client(
                authorization_endpoint=None,
                token_endpoint=None,
                device_authorization_endpoint=None,
            ),
        )
    )
    discovery_doc = {
        "authorization_endpoint": "https://discovered.example/authorize",
        "token_endpoint": "https://discovered.example/token",
        "device_authorization_endpoint": "https://discovered.example/device",
        "userinfo_endpoint": "https://discovered.example/userinfo",
    }

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_response = MagicMock(is_success=True)
        mock_response.json.return_value = discovery_doc
        mock_client.get.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client
        result = await get_auth_discovery()

    assert result.oidc is not None
    public = result.oidc.clients[0]
    assert isinstance(public, PublicOidcAdvertisedClient)
    assert public.authorization_endpoint == "https://discovered.example/authorize"
    assert public.token_endpoint == "https://discovered.example/token"
    assert public.device_authorization_endpoint == "https://discovered.example/device"
    assert result.oidc.userinfo_endpoint == "https://discovered.example/userinfo"


@pytest.mark.asyncio
async def test_brokered_public_client_advertises_helix_endpoints() -> None:
    _set_config(
        OIDCConfig(
            enabled=True,
            issuer="https://sso.example.com",
            public_client=_public_client(server_side_sessions=True),
            server_sessions=OIDCServerSessionsConfig(encryption_key_env_var="NHX_AUTH_SESSION_ENCRYPTION_KEY"),
        )
    )

    result = await get_auth_discovery()

    assert result.oidc is not None
    public = result.oidc.clients[0]
    assert isinstance(public, PublicOidcAdvertisedClient)
    assert public.server_side_sessions is True
    assert public.authorization_start_endpoint == "https://127.0.0.1:18082/apis/auth/v2/authorize?client=public"
    assert public.broker_token_endpoint == "https://127.0.0.1:18082/apis/auth/v2/token?client=public"
    assert public.authorization_endpoint is None
    assert public.token_endpoint is None
    assert public.device_authorization_endpoint is None


@pytest.mark.asyncio
async def test_workload_capability_is_sourced_from_workload_profile() -> None:
    _set_config(
        OIDCConfig(
            enabled=True,
            issuer="https://sso.example.com",
            workload=OIDCWorkloadConfig(
                client_id="test-workload",
                audience="nemo-helix",
                scope="openid email groups",
            ),
        )
    )

    result = await get_auth_discovery()

    assert result.oidc is not None
    assert result.oidc.workload_token_exchange_enabled is True
    assert result.oidc.workload_client_id == "test-workload"
    assert result.oidc.workload_token_endpoint == "https://127.0.0.1:18082/apis/auth/token"
    assert result.oidc.workload_audience == "nemo-helix"
    assert result.oidc.workload_scope == "openid email groups"


@pytest.mark.asyncio
async def test_auth_disabled_has_no_oidc_document() -> None:
    Configuration.set_override(AuthConfig(enabled=False, oidc=OIDCConfig(enabled=False)))

    assert await get_auth_discovery() == AuthDiscoveryResponse(auth_enabled=False, oidc=None)


class TestIdpDiscoveryCache:
    @pytest.mark.asyncio
    async def test_cache_hit_avoids_http_call(self) -> None:
        discovery_doc = {"token_endpoint": "https://sso.example.com/token"}
        with patch("nhx.core.auth.api.v2.discovery.endpoints.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock(is_success=True)
            mock_response.json.return_value = discovery_doc
            mock_client.get.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            assert await _fetch_idp_discovery("https://sso.example.com", 300) == discovery_doc
            assert await _fetch_idp_discovery("https://sso.example.com", 300) == discovery_doc
            assert mock_client.get.call_count == 1

    @pytest.mark.asyncio
    async def test_fetch_failure_returns_stale_cache(self) -> None:
        import nhx.core.auth.api.v2.discovery.endpoints as endpoints

        discovery_doc = {"token_endpoint": "https://sso.example.com/token"}
        with patch("nhx.core.auth.api.v2.discovery.endpoints.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock(is_success=True)
            mock_response.json.return_value = discovery_doc
            mock_client.get.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client
            await _fetch_idp_discovery("https://sso.example.com", 300)
            endpoints._idp_discovery_cache_time -= 600
            mock_client.get.side_effect = httpx.HTTPError("Connection refused")

            assert await _fetch_idp_discovery("https://sso.example.com", 300) == discovery_doc
