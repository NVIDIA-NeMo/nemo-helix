# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discovery endpoints for NeMo Platform configuration."""

import logging
import time

import httpx
from fastapi import APIRouter, Request
from nmp.common.config import get_auth_config
from nmp.core.auth.api.v2.workload_token_exchange import workload_token_endpoint_url
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Discovery"])

# Module-level cache for IdP discovery document responses.
# There is only one key per issuer so a simple variable + timestamp suffices.
_idp_discovery_cache: dict | None = None
_idp_discovery_cache_time: float = 0.0


class OIDCDiscoveryResponse(BaseModel):
    """OIDC discovery response for CLI/SDK."""

    issuer: str
    authorization_endpoint: str | None = None
    token_endpoint: str | None = None
    device_authorization_endpoint: str | None = None
    userinfo_endpoint: str | None = None
    client_id: str
    default_scopes: str = "openid profile email offline_access"
    scope_prefix: str | None = None
    workload_token_exchange_enabled: bool = False
    workload_client_id: str | None = None
    workload_token_endpoint: str | None = None
    workload_audience: str | None = None
    workload_scope: str | None = None


class AuthDiscoveryResponse(BaseModel):
    """Auth discovery response for CLI/SDK."""

    auth_enabled: bool
    oidc: OIDCDiscoveryResponse | None = None


async def _fetch_idp_discovery(issuer: str, cache_ttl: int) -> dict:
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
need to authenticate with this NeMo Platform deployment.

**Response fields:**

- `auth_enabled`: Whether authentication is enabled on this cluster
- `oidc`: OIDC configuration (only present when OIDC is enabled)
  - `issuer`: The OIDC issuer URL
  - `authorization_endpoint`: Authorization endpoint for browser-based flows
  - `token_endpoint`: Token exchange endpoint
  - `device_authorization_endpoint`: Device flow authorization endpoint (for CLI)
  - `userinfo_endpoint`: UserInfo endpoint
  - `client_id`: OAuth client ID to use
  - `default_scopes`: OAuth scopes to request during authentication
  - `scope_prefix`: Prefix to prepend to custom scopes (those with ':' or '.default')
  - `workload_token_exchange_enabled`: Whether SDK workload identity token exchange is enabled
  - `workload_client_id`: OAuth client ID to use for workload identity token exchange
  - `workload_token_endpoint`: Token endpoint to use only for workload identity token exchange
  - `workload_audience`: RFC 8693 audience for exchanged workload tokens
  - `workload_scope`: OAuth scopes for exchanged workload tokens
""",
)
async def get_auth_discovery_endpoint(request: Request) -> AuthDiscoveryResponse:
    """FastAPI route wrapper for auth configuration discovery."""
    return await get_auth_discovery(request)


async def get_auth_discovery(request: Request | None = None) -> AuthDiscoveryResponse:
    """Return auth configuration for CLI/SDK discovery.

    This endpoint is unauthenticated and returns the information
    clients need to authenticate with this NeMo Platform deployment.
    """
    config = get_auth_config()

    oidc = None
    if config.oidc.enabled and config.oidc.issuer:
        discovery = await _fetch_idp_discovery(config.oidc.issuer, config.oidc.discovery_cache_ttl)

        oidc = OIDCDiscoveryResponse(
            issuer=config.oidc.issuer,
            authorization_endpoint=config.oidc.authorization_endpoint or discovery.get("authorization_endpoint"),
            token_endpoint=config.oidc.token_endpoint or discovery.get("token_endpoint"),
            device_authorization_endpoint=(
                config.oidc.device_authorization_endpoint or discovery.get("device_authorization_endpoint")
            ),
            userinfo_endpoint=config.oidc.userinfo_endpoint or discovery.get("userinfo_endpoint"),
            client_id=config.oidc.client_id,
            default_scopes=config.oidc.default_scopes,
            scope_prefix=config.oidc.scope_prefix,
            workload_token_exchange_enabled=config.oidc.workload_token_exchange_enabled,
            workload_client_id=config.oidc.workload_client_id,
            workload_token_endpoint=(
                config.oidc.workload_token_endpoint
                or (workload_token_endpoint_url(request) if config.oidc.workload_token_exchange_enabled else None)
            ),
            workload_audience=config.oidc.workload_audience,
            workload_scope=config.oidc.workload_scope,
        )

    return AuthDiscoveryResponse(
        auth_enabled=config.enabled,
        oidc=oidc,
    )
