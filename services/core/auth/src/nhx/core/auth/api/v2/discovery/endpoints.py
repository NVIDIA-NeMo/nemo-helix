# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discovery endpoints for NeMo Helix configuration."""

import logging
import time

import httpx
from fastapi import APIRouter
from nhx.common.auth.discovery import (
    AuthDiscoveryResponse,
    ConfidentialOidcAdvertisedClient,
    OidcAdvertisedClient,
    OIDCDiscoveryResponse,
    PublicOidcAdvertisedClient,
)
from nhx.common.config import get_auth_config, get_platform_config
from nhx.common.config.base import OIDCConfig
from nhx.core.auth.api.v2.workload_token_exchange import workload_token_endpoint_url

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Discovery"])

# Module-level cache for IdP discovery document responses.
# There is only one key per issuer so a simple variable + timestamp suffices.
_idp_discovery_cache: dict[str, object] | None = None
_idp_discovery_cache_time: float = 0.0


async def _fetch_idp_discovery(issuer: str, cache_ttl: int) -> dict[str, object]:
    """Fetch IdP discovery document with caching and graceful degradation.

    Caches the IdP's .well-known/openid-configuration response in memory
    with a configurable TTL. On fetch failure, returns stale cached data
    if available (graceful degradation).

    Args:
        issuer: The OIDC issuer URL.
        cache_ttl: Cache time-to-live in seconds.  0 disables caching.

    Returns:
        The discovery document dict, or an empty dict on failure with no cache.
    """
    global _idp_discovery_cache, _idp_discovery_cache_time  # noqa: PLW0603

    now = time.monotonic()
    if _idp_discovery_cache is not None and cache_ttl > 0 and (now - _idp_discovery_cache_time) < cache_ttl:
        return _idp_discovery_cache

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{issuer.rstrip('/')}/.well-known/openid-configuration",
                timeout=5.0,
            )
            if response.is_success:
                _idp_discovery_cache = response.json()
                _idp_discovery_cache_time = now
                return _idp_discovery_cache
    except Exception as e:
        logger.warning(f"Failed to fetch OIDC discovery: {e}")

    # Graceful degradation: return stale cache if available
    if _idp_discovery_cache is not None:
        logger.info("Returning stale OIDC discovery cache after fetch failure")
        return _idp_discovery_cache

    return {}


def _clear_idp_discovery_cache() -> None:
    """Reset the module-level IdP discovery cache (for testing)."""
    global _idp_discovery_cache, _idp_discovery_cache_time  # noqa: PLW0603
    _idp_discovery_cache = None
    _idp_discovery_cache_time = 0.0


@router.get(
    "/discovery",
    response_model=AuthDiscoveryResponse,
    summary="Discover auth configuration",
    description="""
Return authentication configuration for CLI/SDK discovery.

This endpoint is unauthenticated and returns the information clients
need to authenticate with this NeMo Helix deployment.

**Response fields:**

- `auth_enabled`: Whether authentication is enabled on this cluster
- `oidc`: OIDC configuration (only present when OIDC is enabled)
  - `issuer`: The OIDC issuer URL
  - `userinfo_endpoint`: UserInfo endpoint
  - `clients`: Complete public and confidential user-login client contracts. Exactly one is the default.
  - `workload_token_exchange_enabled`: Whether SDK workload identity token exchange is enabled
  - `workload_client_id`: OAuth client ID to use for workload identity token exchange
  - `workload_token_endpoint`: Token endpoint to use only for workload identity token exchange
  - `workload_audience`: RFC 8693 audience for exchanged workload tokens
  - `workload_scope`: OAuth scopes for exchanged workload tokens
""",
)
async def get_auth_discovery_endpoint() -> AuthDiscoveryResponse:
    """FastAPI route wrapper for auth configuration discovery."""
    return await get_auth_discovery()


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _advertised_clients(oidc: OIDCConfig, discovery: dict[str, object]) -> list[OidcAdvertisedClient]:
    """Return complete user-login client contracts for this deployment."""
    clients: list[OidcAdvertisedClient] = []
    public = oidc.public_client
    confidential = oidc.confidential_client
    advertised_base_url = get_platform_config().effective_advertised_base_url.rstrip("/")
    if public is not None:
        if public.server_side_sessions:
            clients.append(
                PublicOidcAdvertisedClient(
                    client_id=public.client_id,
                    default=True,
                    server_side_sessions=True,
                    bearer_token_source=public.bearer_token_source,
                    default_scopes=public.default_scopes,
                    scope_prefix=public.scope_prefix,
                    authorization_start_endpoint=f"{advertised_base_url}/apis/auth/v2/authorize?client=public",
                    broker_token_endpoint=f"{advertised_base_url}/apis/auth/v2/token?client=public",
                )
            )
        else:
            clients.append(
                PublicOidcAdvertisedClient(
                    client_id=public.client_id,
                    default=True,
                    server_side_sessions=False,
                    bearer_token_source=public.bearer_token_source,
                    default_scopes=public.default_scopes,
                    scope_prefix=public.scope_prefix,
                    authorization_endpoint=public.authorization_endpoint
                    or _optional_string(discovery.get("authorization_endpoint")),
                    token_endpoint=public.token_endpoint or _optional_string(discovery.get("token_endpoint")),
                    device_authorization_endpoint=public.device_authorization_endpoint
                    or _optional_string(discovery.get("device_authorization_endpoint")),
                    device_authorization_requires_device_id=public.device_authorization_requires_device_id,
                    device_authorization_display_name=public.device_authorization_display_name,
                    device_token_request_includes_scope=public.device_token_request_includes_scope,
                )
            )
    if confidential is not None:
        clients.append(
            ConfidentialOidcAdvertisedClient(
                client_id=confidential.client_id,
                default=public is None,
                authorization_start_endpoint=f"{advertised_base_url}/apis/auth/v2/authorize",
                broker_token_endpoint=f"{advertised_base_url}/apis/auth/v2/token",
                bearer_token_source=confidential.bearer_token_source,
                default_scopes=confidential.default_scopes,
                scope_prefix=confidential.scope_prefix,
            )
        )
    return clients


async def get_auth_discovery() -> AuthDiscoveryResponse:
    """Return auth configuration for CLI/SDK discovery.

    This endpoint is unauthenticated and returns the information
    clients need to authenticate with this NeMo Helix deployment.
    """
    config = get_auth_config()

    oidc = None
    if config.oidc.enabled and config.oidc.issuer:
        discovery = await _fetch_idp_discovery(config.oidc.issuer, config.oidc.discovery_cache_ttl)

        clients = _advertised_clients(config.oidc, discovery)
        workload = config.oidc.workload

        oidc = OIDCDiscoveryResponse(
            issuer=config.oidc.issuer,
            userinfo_endpoint=config.oidc.userinfo_endpoint or _optional_string(discovery.get("userinfo_endpoint")),
            clients=clients,
            workload_token_exchange_enabled=workload is not None,
            workload_client_id=workload.client_id if workload is not None else None,
            workload_token_endpoint=(workload.token_endpoint or workload_token_endpoint_url())
            if workload is not None
            else None,
            workload_audience=workload.audience if workload is not None else None,
            workload_scope=workload.scope if workload is not None else None,
        )

    return AuthDiscoveryResponse(
        auth_enabled=config.enabled,
        oidc=oidc,
    )
