# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JWT validation for native OIDC authentication."""

import logging
import time

import httpx
import jwt
from jwt.types import Options
from nmp.common import http_clients
from nmp.common.config import AuthConfig

from . import token_claims
from .json_payload import JsonObject, JsonObjectDeserializationError, JsonObjectDeserializer
from .jwks import AsyncJWKSClient

logger = logging.getLogger(__name__)

# Cache TTLs for JWTValidator internals.
# JWKS keys are refreshed after this period even for known key IDs,
# ensuring revoked keys are eventually dropped.
_JWKS_CACHE_LIFESPAN = 3600  # 1 hour

# Discovery document is re-fetched after this period so changes to the
# IdP JWKS URI or token endpoint are picked up without a restart.
_DISCOVERY_CACHE_TTL = 3600  # 1 hour

__all__ = [
    "JWTValidator",
    "UnsignedJWTRejectedError",
]


class UnsignedJWTRejectedError(Exception):
    """Raised when an unsigned JWT is rejected by configuration."""


class JWTValidator:
    """Validates JWT tokens against an OIDC issuer."""

    def __init__(self, config: AuthConfig):
        self.config = config
        self._claims_extractor = token_claims.TokenClaimsExtractor(config)
        self._json_deserializer = JsonObjectDeserializer()
        self._jwks_client: AsyncJWKSClient | None = None
        self._discovery_cache: JsonObject | None = None
        self._discovery_cache_time: float = 0.0

    async def _discover_oidc_config(self) -> JsonObject:
        """Fetch OIDC discovery document from issuer.

        Results are cached with a TTL. After expiry the document is
        re-fetched so that changes to the IdP JWKS URI or endpoints
        are eventually picked up without a process restart.
        """
        now = time.monotonic()
        if self._discovery_cache and (now - self._discovery_cache_time) < _DISCOVERY_CACHE_TTL:
            return self._discovery_cache

        discovery_url = f"{self.config.oidc.issuer.rstrip('/')}/.well-known/openid-configuration"
        async with httpx.AsyncClient() as client:
            response = await client.get(discovery_url, timeout=10.0)
            response.raise_for_status()
            try:
                self._discovery_cache = self._json_deserializer.deserialize(response.content)
            except JsonObjectDeserializationError as exc:
                raise jwt.InvalidTokenError("OIDC discovery was not a JSON object") from exc
            self._discovery_cache_time = now
            return self._discovery_cache

    async def jwks_uri(self) -> str:
        """Return the configured or discovered OIDC JWKS URI."""
        if self.config.oidc.jwks_uri:
            return self.config.oidc.jwks_uri
        discovery = await self._discover_oidc_config()
        jwks_uri = discovery.get("jwks_uri")
        if not isinstance(jwks_uri, str) or not jwks_uri:
            raise jwt.InvalidTokenError("OIDC discovery did not include jwks_uri")
        return jwks_uri

    async def jwks(self) -> JsonObject:
        """Return the cached-or-fetched OIDC JWKS document."""
        jwks_client = await self._get_jwks_client()
        return await jwks_client.get_jwks()

    async def _introspection_endpoint(self) -> str | None:
        """Return the configured or discovered RFC 7662 introspection endpoint, if any."""
        if self.config.oidc.introspection_endpoint:
            return self.config.oidc.introspection_endpoint
        discovery = await self._discover_oidc_config()
        endpoint = discovery.get("introspection_endpoint")
        return endpoint if isinstance(endpoint, str) and endpoint else None

    async def _introspect_token(self, token: str) -> JsonObject | None:
        """Resolve an opaque access token to claims via RFC 7662 introspection.

        Returns None when introspection is unavailable or reports the token as inactive.
        """
        introspection_endpoint = await self._introspection_endpoint()
        if introspection_endpoint is None:
            logger.warning("Cannot introspect opaque token: no introspection_endpoint configured or discoverable")
            return None

        client = http_clients.shared_async_http_client()
        client_secret = self.config.oidc.introspection_client_secret
        auth = (self.config.oidc.client_id, client_secret) if client_secret else None
        response = await client.post(
            introspection_endpoint,
            data={"token": token, "token_type_hint": "access_token"},
            auth=auth if auth is not None else httpx.USE_CLIENT_DEFAULT,
            timeout=10.0,
        )
        response.raise_for_status()
        try:
            claims = self._json_deserializer.deserialize(response.content)
        except JsonObjectDeserializationError as exc:
            raise jwt.InvalidTokenError("Introspection response was not a JSON object") from exc

        if claims.get("active") is not True:
            logger.warning("Introspection reported the token as inactive")
            return None

        audience = self.config.oidc.audience
        if audience and not self._introspected_audience_matches(claims.get("aud"), audience):
            logger.warning(f"Introspected token audience does not include {audience!r}")
            return None

        return claims

    @staticmethod
    def _introspected_audience_matches(claim_audience: object, expected: str) -> bool:
        """Check an RFC 7662 aud claim (string or list) contains the expected audience."""
        if isinstance(claim_audience, str):
            return claim_audience == expected
        if isinstance(claim_audience, list):
            return expected in claim_audience
        return False

    async def _userinfo_endpoint(self) -> str | None:
        """Return the configured or discovered OIDC UserInfo endpoint, if any."""
        if self.config.oidc.userinfo_endpoint:
            return self.config.oidc.userinfo_endpoint
        discovery = await self._discover_oidc_config()
        endpoint = discovery.get("userinfo_endpoint")
        return endpoint if isinstance(endpoint, str) and endpoint else None

    async def _resolve_via_userinfo(self, token: str) -> JsonObject | None:
        """Resolve an opaque access token to claims via the OIDC UserInfo endpoint.

        Unlike RFC 7662 introspection, the UserInfo endpoint (OIDC Core 5.3) is
        authenticated with the access token itself (no separate client credential)
        and its response has no active/aud/scope claims: a successful response is
        itself proof the token is valid for the caller, but audience and the
        token's original OAuth scope grant cannot be recovered or enforced for
        tokens resolved this way. Callers must not invoke this when oidc.audience
        is configured — see validate_token, which skips this path in that case.

        The scope gap matters: TokenClaims.scopes ends up empty for every token
        resolved here, and the platform's scope check
        (services/core/auth/src/nmp/core/auth/app/policies/scopes.rego,
        scope_check_passed) treats an empty scope list as "no platform scopes
        provided" and skips scope enforcement entirely, falling back to
        RBAC/permissions only. A caller holding an access token that the IdP
        scoped down (e.g. to a single read-only scope) is therefore not
        restricted to that scope once resolved through this path — only their
        role/group-based permissions apply. Enable this only for IdPs, like
        Starfleet, that don't hand out narrower per-token scopes in practice.

        Returns None when the UserInfo endpoint is unavailable or rejects the token.
        """
        userinfo_endpoint = await self._userinfo_endpoint()
        if userinfo_endpoint is None:
            logger.warning("Cannot resolve opaque token: no userinfo_endpoint configured or discoverable")
            return None

        client = http_clients.shared_async_http_client()
        try:
            response = await client.get(
                userinfo_endpoint,
                headers={"Authorization": f"Bearer {token}"},
                timeout=10.0,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 401:
                logger.warning("UserInfo endpoint rejected the token")
                return None
            raise

        logger.warning(
            "Resolved opaque token via UserInfo endpoint; the token's original OAuth scope "
            "grant cannot be recovered, so scope enforcement is skipped for this request "
            "(RBAC/permissions still apply)"
        )
        try:
            return self._json_deserializer.deserialize(response.content)
        except JsonObjectDeserializationError as exc:
            raise jwt.InvalidTokenError("UserInfo response was not a JSON object") from exc

    async def _get_jwks_client(self) -> AsyncJWKSClient:
        """Get or create JWKS client for token validation.

        The client is initialized with a lifespan so that cached keys
        are periodically refreshed. This ensures that keys revoked by
        the IdP are eventually dropped even if their key ID was
        previously seen.
        """
        if self._jwks_client:
            return self._jwks_client

        jwks_uri = await self.jwks_uri()
        self._jwks_client = AsyncJWKSClient(jwks_uri, lifespan=_JWKS_CACHE_LIFESPAN)
        return self._jwks_client

    async def validate_token(self, token: str) -> token_claims.TokenClaims | None:
        """Validate a JWT token and extract claims.

        Args:
            token: The JWT token string to validate.

        Returns:
            TokenClaims if valid, None if invalid or validation fails.
        """
        try:
            token_alg = ""
            try:
                token_header = jwt.get_unverified_header(token)
                token_alg = str(token_header.get("alg", "")).lower()
            except jwt.PyJWTError:
                token_alg = ""
                if self.config.oidc.resolve_opaque_tokens_via_userinfo:
                    if self.config.oidc.audience:
                        logger.warning(
                            "Skipping UserInfo resolution because oidc.audience is configured and "
                            "UserInfo responses cannot be checked against it; falling back to "
                            "introspection if enabled"
                        )
                    else:
                        claims = await self._resolve_via_userinfo(token)
                        if claims is None:
                            return None
                        return self._claims_extractor.extract(claims)
                if self.config.oidc.introspect_opaque_tokens:
                    claims = await self._introspect_token(token)
                    if claims is None:
                        return None
                    return self._claims_extractor.extract(claims)

            if token_alg == "none":
                if not self.config.allow_unsigned_jwt:
                    logger.warning("Unsigned JWT rejected: auth.allow_unsigned_jwt is disabled")
                    raise UnsignedJWTRejectedError(
                        "Unsigned JWTs are not accepted. Set auth.allow_unsigned_jwt=true for local development."
                    )

                claims = jwt.decode(
                    token,
                    algorithms=["none"],
                    options={
                        "verify_signature": False,
                        "verify_exp": True,
                        "verify_iat": True,
                        "verify_nbf": True,
                        "verify_aud": False,
                        "verify_iss": False,
                        "require": ["sub", "exp", "iat"],
                    },
                )
                return self._claims_extractor.extract(claims)

            jwks_client = await self._get_jwks_client()
            signing_key = await jwks_client.get_signing_key_from_jwt(token)

            # Only validate audience when explicitly configured.
            # When audience is not set, skip the check so tokens from any
            # audience are accepted (the issuer + signature checks are still
            # enforced).
            audience = [self.config.oidc.audience] if self.config.oidc.audience else None

            # Build list of allowed issuers
            allowed_issuers = [self.config.oidc.issuer] + self.config.oidc.additional_issuers

            # Decode and validate token (validate issuer manually to support multiple)
            decode_options: Options = {"require": ["exp", "iat", "sub"]}
            if audience is None:
                decode_options["verify_aud"] = False
            claims = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256", "ES256"],
                audience=audience,
                options=decode_options,
            )

            # Validate issuer manually (PyJWT only supports single issuer)
            token_issuer = claims.get("iss", "")
            if token_issuer not in allowed_issuers:
                logger.warning(f"Invalid token issuer: {token_issuer} not in {allowed_issuers}")
                return None
            return self._claims_extractor.extract(claims)

        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except UnsignedJWTRejectedError:
            raise
        except jwt.InvalidAudienceError:
            logger.warning("Invalid token audience")
            return None
        except jwt.InvalidIssuerError:
            logger.warning("Invalid token issuer")
            return None
        except jwt.PyJWTError as e:
            logger.warning(f"Token validation failed: {e}")
            return None
        except httpx.HTTPError as e:
            logger.error(f"Failed to fetch JWKS: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during token validation: {e}")
            return None
