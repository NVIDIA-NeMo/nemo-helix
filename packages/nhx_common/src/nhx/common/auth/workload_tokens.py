# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx
import jwt
from nemo_helix_plugin.client.auth import ServicePrincipalTokenProvider
from nhx.common.auth.jwks import AsyncJWKSClient
from nhx.common.config import AuthConfig, get_platform_config
from starlette.requests import Request

from .models import Principal
from .signing_keys import RSASigningKey, RSASigningKeyCache
from .token_claims import ActorClaims, TokenClaims, groups_from_claim, scopes_from_claim
from .token_resolver import ResolvedBearerToken

WORKLOAD_TOKEN_PATH = "/apis/auth/token"
WORKLOAD_JWKS_PATH = "/apis/auth/jwks"
DEFAULT_WORKLOAD_AUDIENCE = "nemo-helix"
DEFAULT_WORKLOAD_SCOPE = "openid email groups"

_MISSING_WORKLOAD_TOKEN_KEY_ID_MESSAGE = (
    "auth.oidc.workload_token_key_id or auth.token_signing.key_id must be configured for workload token exchange"
)
_WORKLOAD_SIGNING_KEY_CACHE = RSASigningKeyCache()
_WORKLOAD_SUBJECT_JWKS_CLIENTS: dict[tuple[str, int], AsyncJWKSClient] = {}


@dataclass(frozen=True)
class _SigningKeyLoadRequest:
    kid: str
    private_key_file: str | None
    missing_private_key_message: str
    invalid_private_key_message: str


@dataclass(frozen=True)
class ServiceWorkloadAccessTokenProvider(ServicePrincipalTokenProvider):
    config: AuthConfig
    service_name: str
    on_behalf_of: str | Principal | None = None

    @property
    def service_principal_id(self) -> str:
        return f"service:{self.service_name}"

    @property
    def delegates_principal_identity(self) -> bool:
        return self.on_behalf_of is not None

    def get_access_token(self) -> str:
        return issue_service_workload_access_token(
            self.config,
            service_name=self.service_name,
            on_behalf_of=self.on_behalf_of,
        )

    async def get_access_token_async(self) -> str:
        return await issue_service_workload_access_token_async(
            self.config,
            service_name=self.service_name,
            on_behalf_of=self.on_behalf_of,
        )


def _platform_base_url_from_request(request: Request | None) -> str:
    if request is not None:
        return str(request.base_url).rstrip("/")
    return get_platform_config().base_url.rstrip("/")


def workload_token_endpoint_url(request: Request | None = None) -> str:
    """Return the externally reachable workload token exchange endpoint URL."""
    return f"{_platform_base_url_from_request(request)}{WORKLOAD_TOKEN_PATH}"


def workload_jwks_url(request: Request | None = None) -> str:
    """Return the externally reachable workload exchange JWKS URL."""
    return f"{_platform_base_url_from_request(request)}{WORKLOAD_JWKS_PATH}"


def workload_token_issuer(config: AuthConfig, request: Request | None) -> str:
    return (
        config.oidc.workload_token_issuer
        or config.token_signing.issuer
        or f"{_platform_base_url_from_request(request)}/apis/auth"
    )


def workload_token_issuers(config: AuthConfig, request: Request | None) -> set[str]:
    if config.oidc.workload_token_issuer or config.token_signing.issuer:
        return {workload_token_issuer(config, request)}
    return {workload_token_issuer(config, request), workload_token_issuer(config, None)}


def _workload_token_key_id(config: AuthConfig) -> str:
    return config.oidc.workload_token_key_id or config.token_signing.key_id


def _workload_private_key_file(config: AuthConfig) -> str | None:
    return config.oidc.workload_token_private_key_file or config.token_signing.private_key_file


def _workload_signing_key_load_request(config: AuthConfig) -> _SigningKeyLoadRequest:
    kid = _workload_token_key_id(config)
    if not kid:
        raise RuntimeError(_MISSING_WORKLOAD_TOKEN_KEY_ID_MESSAGE)

    return _SigningKeyLoadRequest(
        kid=kid,
        private_key_file=_workload_private_key_file(config),
        missing_private_key_message="auth.token_signing.private_key_file must be configured for workload token exchange",
        invalid_private_key_message="workload token private key must be an RSA private key",
    )


def workload_signing_key(config: AuthConfig) -> RSASigningKey:
    load_request = _workload_signing_key_load_request(config)
    return _WORKLOAD_SIGNING_KEY_CACHE.get_from_file(
        kid=load_request.kid,
        private_key_file=load_request.private_key_file,
        missing_private_key_message=load_request.missing_private_key_message,
        invalid_private_key_message=load_request.invalid_private_key_message,
    )


async def workload_signing_key_async(config: AuthConfig) -> RSASigningKey:
    load_request = _workload_signing_key_load_request(config)
    return await _WORKLOAD_SIGNING_KEY_CACHE.get_from_file_async(
        kid=load_request.kid,
        private_key_file=load_request.private_key_file,
        missing_private_key_message=load_request.missing_private_key_message,
        invalid_private_key_message=load_request.invalid_private_key_message,
    )


def workload_public_jwk(config: AuthConfig) -> dict[str, Any]:
    load_request = _workload_signing_key_load_request(config)
    return _WORKLOAD_SIGNING_KEY_CACHE.public_jwk_from_file(
        kid=load_request.kid,
        private_key_file=load_request.private_key_file,
        missing_private_key_message=load_request.missing_private_key_message,
        invalid_private_key_message=load_request.invalid_private_key_message,
    )


async def workload_public_jwk_async(config: AuthConfig) -> dict[str, Any]:
    load_request = _workload_signing_key_load_request(config)
    return await _WORKLOAD_SIGNING_KEY_CACHE.public_jwk_from_file_async(
        kid=load_request.kid,
        private_key_file=load_request.private_key_file,
        missing_private_key_message=load_request.missing_private_key_message,
        invalid_private_key_message=load_request.invalid_private_key_message,
    )


def _default_workload_audience(config: AuthConfig) -> str:
    return config.oidc.workload_audience or config.oidc.audience or DEFAULT_WORKLOAD_AUDIENCE


def allowed_audiences(config: AuthConfig) -> set[str]:
    return {_default_workload_audience(config), *config.oidc.workload_allowed_audiences}


def _workload_subject_audience(config: AuthConfig) -> str:
    return config.oidc.workload_client_id or config.oidc.client_id or DEFAULT_WORKLOAD_AUDIENCE


def _workload_subject_jwks_client(config: AuthConfig) -> AsyncJWKSClient | None:
    jwks_uri = config.oidc.workload_subject_jwks_uri
    if not jwks_uri:
        return None

    cache_ttl = config.oidc.workload_subject_jwks_cache_ttl_seconds
    key = (jwks_uri, cache_ttl)
    client = _WORKLOAD_SUBJECT_JWKS_CLIENTS.get(key)
    if client is None:
        client = AsyncJWKSClient(jwks_uri, lifespan=cache_ttl)
        _WORKLOAD_SUBJECT_JWKS_CLIENTS[key] = client
    return client


def _split_scope(scope: Any) -> list[str]:
    if not isinstance(scope, str):
        return []
    return [part for part in scope.split() if part]


def granted_workload_scope(config: AuthConfig, requested_scope: Any) -> str | None:
    allowed_scopes = _split_scope(config.oidc.workload_scope or DEFAULT_WORKLOAD_SCOPE)
    requested_scopes = _split_scope(requested_scope) or allowed_scopes
    allowed = set(allowed_scopes)
    granted = [scope for scope in requested_scopes if scope in allowed]
    return " ".join(granted) or None


def _principal_claims(principal: Principal) -> dict[str, Any]:
    claims: dict[str, Any] = {"sub": principal.id}
    if principal.email:
        claims["email"] = principal.email
    if principal.groups:
        claims["groups"] = ",".join(principal.groups)
    return claims


def _service_token_claims(service_name: str, on_behalf_of: str | Principal | None) -> dict[str, Any]:
    service_subject = f"service:{service_name}"
    if on_behalf_of is None:
        return {"sub": service_subject}

    if isinstance(on_behalf_of, Principal):
        subject_claims = _principal_claims(on_behalf_of.effective_principal)
    else:
        subject_claims = {"sub": on_behalf_of}
    subject_claims["act"] = {"sub": service_subject}
    return subject_claims


def _encode_access_token(
    config: AuthConfig,
    signing_key: RSASigningKey,
    *,
    request: Request | None,
    subject_claims: dict[str, Any],
    scope: str | None,
) -> str:
    now = int(time.time())
    claims: dict[str, Any] = {
        "iss": workload_token_issuer(config, request),
        **subject_claims,
        "aud": _default_workload_audience(config),
        "iat": now,
        "nbf": now,
        "exp": now + config.oidc.workload_token_ttl_seconds,
    }
    if scope:
        claims["scope"] = scope
    return jwt.encode(
        claims,
        signing_key.private_key,
        algorithm="RS256",
        headers={"kid": signing_key.kid},
    )


def issue_service_workload_access_token(
    config: AuthConfig,
    *,
    service_name: str,
    on_behalf_of: str | Principal | None = None,
    request: Request | None = None,
) -> str:
    signing_key = workload_signing_key(config)
    return _encode_access_token(
        config,
        signing_key,
        request=request,
        subject_claims=_service_token_claims(service_name, on_behalf_of),
        scope=granted_workload_scope(config, None),
    )


async def issue_service_workload_access_token_async(
    config: AuthConfig,
    *,
    service_name: str,
    on_behalf_of: str | Principal | None = None,
    request: Request | None = None,
) -> str:
    signing_key = await workload_signing_key_async(config)
    return _encode_access_token(
        config,
        signing_key,
        request=request,
        subject_claims=_service_token_claims(service_name, on_behalf_of),
        scope=granted_workload_scope(config, None),
    )


def _actor_from_claims(claims: dict[str, object]) -> ActorClaims | None:
    actor_claims = claims.get("act")
    if not isinstance(actor_claims, dict):
        return None

    actor_subject = actor_claims.get("sub")
    if not isinstance(actor_subject, str):
        return None

    actor_subject = actor_subject.strip()
    if not actor_subject:
        return None

    return ActorClaims(
        subject=actor_subject,
        groups=groups_from_claim(actor_claims.get("groups", [])),
    )


async def validate_workload_access_token(
    config: AuthConfig,
    request: Request | None,
    token: str,
) -> TokenClaims | None:
    if not config.oidc.workload_token_exchange_enabled:
        return None

    signing_key = await workload_signing_key_async(config)
    public_key = signing_key.private_key.public_key()

    try:
        claims = jwt.decode(
            token,
            public_key,
            algorithms=["RS256"],
            audience=list(allowed_audiences(config)),
            options={"require": ["sub", "iat", "nbf", "exp"]},
            leeway=30,
        )
        issuer = claims.get("iss")
        if not isinstance(issuer, str) or issuer not in workload_token_issuers(config, request):
            return None
        subject = claims.get("sub")
        if not isinstance(subject, str) or not subject:
            return None
        return TokenClaims(
            subject=subject,
            email=claims.get("email") if isinstance(claims.get("email"), str) else None,
            groups=groups_from_claim(claims.get("groups", [])),
            scopes=scopes_from_claim(claims.get("scope") or claims.get("scp")),
            raw_claims=claims,
            actor=_actor_from_claims(claims),
        )
    except (httpx.HTTPError, jwt.PyJWTError):
        return None


async def resolve_workload_access_token(
    config: AuthConfig,
    request: Request | None,
    token: str,
) -> ResolvedBearerToken | None:
    claims = await validate_workload_access_token(config, request, token)
    if claims is None:
        return None
    return ResolvedBearerToken(claims=claims, token_kind="workload_access_token")


async def validate_workload_subject_token(
    config: AuthConfig,
    token: str,
) -> TokenClaims | None:
    if not config.oidc.workload_token_exchange_enabled:
        return None
    if not config.oidc.workload_subject_issuers:
        return None

    jwks_client = _workload_subject_jwks_client(config)
    if jwks_client is None:
        return None

    try:
        signing_key = await jwks_client.get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256", "ES256"],
            audience=_workload_subject_audience(config),
            leeway=30,
            options={"require": ["exp"]},
        )
        issuer = claims.get("iss")
        if issuer not in config.oidc.workload_subject_issuers:
            return None
        subject = claims.get(config.oidc.subject_claim, claims.get("sub"))
        if not isinstance(subject, str) or not subject:
            return None
        email = claims.get(config.oidc.email_claim)
        return TokenClaims(
            subject=subject,
            email=email if isinstance(email, str) else None,
            groups=groups_from_claim(claims.get(config.oidc.groups_claim, claims.get("groups", []))),
            scopes=scopes_from_claim(claims.get("scope") or claims.get("scp")),
            raw_claims=claims,
        )
    except jwt.PyJWTError:
        return None


async def resolve_workload_subject_token(
    config: AuthConfig,
    token: str,
) -> ResolvedBearerToken | None:
    claims = await validate_workload_subject_token(config, token)
    if claims is None:
        return None
    return ResolvedBearerToken(claims=claims, token_kind="workload_subject_token")
