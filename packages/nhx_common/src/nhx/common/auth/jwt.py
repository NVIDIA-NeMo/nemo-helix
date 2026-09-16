# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JWT validation for native OIDC authentication."""

import logging
import time

import httpx
import jwt
from jwt.types import Options
from nhx.common.config import AuthConfig

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
