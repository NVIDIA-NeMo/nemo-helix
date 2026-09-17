# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Annotated, Any

import httpx
import jwt
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from nemo_platform_plugin.auth.types import (
    AuthenticateErrorResponse as AuthenticateErrorResponse,
)
from nemo_platform_plugin.auth.types import (
    AuthenticateResponse as AuthenticateResponse,
)
from nmp.common.auth.bearer import MalformedBearerTokenError, parse_bearer_authorization_header
from nmp.common.auth.jwt import JWTValidator
from nmp.common.auth.token_claims import ActorClaims, TokenClaims, groups_from_claim, scopes_from_claim
from nmp.common.auth.token_resolver import ResolvedBearerToken, resolve_bearer_token
from nmp.common.config import AuthConfig, get_auth_config
from nmp.core.auth.api.v2.workload_token_exchange import (
    WorkloadTokenExchangeService,
    allowed_audiences,
    get_workload_token_exchange_service,
    workload_jwks_url,
    workload_token_issuer,
)
from nmp.core.auth.app.access_keys import AccessKeyRegistry, get_access_key_registry

router = APIRouter(tags=["Authentication"])
logger = logging.getLogger(__name__)


_AUTHENTICATE_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    401: {
        "description": "Missing or invalid bearer token",
        "model": AuthenticateErrorResponse,
    },
    500: {
        "description": "Bearer token authentication is misconfigured",
        "model": AuthenticateErrorResponse,
    },
}


@dataclass(frozen=True)
class AuthenticateDependencies:
    workload_token_exchange_service: WorkloadTokenExchangeService
    access_key_registry: AccessKeyRegistry


def get_authenticate_dependencies(
    workload_token_exchange_service: WorkloadTokenExchangeService = Depends(get_workload_token_exchange_service),
    access_key_registry: AccessKeyRegistry = Depends(get_access_key_registry),
) -> AuthenticateDependencies:
    return AuthenticateDependencies(
        workload_token_exchange_service=workload_token_exchange_service,
        access_key_registry=access_key_registry,
    )


AuthenticateDependency = Annotated[AuthenticateDependencies, Depends(get_authenticate_dependencies)]


class TokenIssuerBoundaryMisconfigurationError(Exception):
    """Raised when IdP and NeMo-issued token validation paths overlap."""


def _bearer_token_from_request(request: Request) -> str:
    try:
        token = parse_bearer_authorization_header(request.headers.get("authorization"))
    except MalformedBearerTokenError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid bearer token") from exc
    if token is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")
    return token


def _scopes_from_claims(claims: dict[str, object]) -> list[str]:
    return scopes_from_claim(claims.get("scope") or claims.get("scp"))


def _normalize_url(value: str) -> str:
    return value.rstrip("/")


def _normalize_jwk_public_material(jwk: dict[Any, Any]) -> tuple[tuple[str, str], ...] | None:
    match jwk.get("kty"):
        case "RSA":
            keys = ("kty", "n", "e")
        case "EC":
            keys = ("kty", "crv", "x", "y")
        case "OKP":
            keys = ("kty", "crv", "x")
        case _:
            return None

    material: list[tuple[str, str]] = []
    for key in keys:
        value = jwk.get(key)
        if not isinstance(value, str) or not value:
            return None
        material.append((key, value))
    return tuple(material)


async def _validate_token_issuer_boundary(
    config: AuthConfig,
    request: Request,
    workload_token_exchange_service: WorkloadTokenExchangeService,
    jwt_validator: JWTValidator,
) -> None:
    if not (config.oidc.enabled and config.oidc.workload_token_exchange_enabled):
        return

    workload_issuer = _normalize_url(workload_token_issuer(config, request))
    idp_issuers = {_normalize_url(issuer) for issuer in [config.oidc.issuer, *config.oidc.additional_issuers] if issuer}
    if workload_issuer in idp_issuers:
        raise TokenIssuerBoundaryMisconfigurationError("OIDC issuer must not match NeMo workload token issuer")

    try:
        idp_jwks_uri = _normalize_url(await jwt_validator.jwks_uri())
    except (httpx.HTTPError, jwt.PyJWTError):
        return

    nemo_jwks_uri = _normalize_url(workload_jwks_url(request))
    if idp_jwks_uri == nemo_jwks_uri:
        raise TokenIssuerBoundaryMisconfigurationError("OIDC JWKS URI must not match NeMo workload JWKS URI")

    try:
        signing_key = await workload_token_exchange_service.public_jwk_async(config)
    except (OSError, RuntimeError, ValueError):
        return

    nemo_material = _normalize_jwk_public_material(signing_key)
    if nemo_material is None:
        return

    try:
        idp_jwks = await jwt_validator.jwks()
    except (httpx.HTTPError, jwt.PyJWTError):
        return

    idp_keys = idp_jwks.get("keys", [])
    if not isinstance(idp_keys, list):
        return

    for jwk in idp_keys:
        if isinstance(jwk, dict) and _normalize_jwk_public_material(jwk) == nemo_material:
            raise TokenIssuerBoundaryMisconfigurationError(
                "OIDC JWKS must not contain the NeMo workload token signing key"
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


def _response_from_resolved(resolved: ResolvedBearerToken) -> AuthenticateResponse:
    principal = resolved.principal
    jti = resolved.claims.raw_claims.get("jti")
    return AuthenticateResponse(
        principal=principal.id,
        email=principal.email,
        groups=principal.groups,
        scopes=resolved.scopes,
        jti=jti if isinstance(jti, str) and jti else None,
        token_kind=resolved.token_kind,
        on_behalf_of=principal.on_behalf_of,
        on_behalf_of_email=principal.on_behalf_of_email,
        on_behalf_of_groups=list(principal.on_behalf_of_groups or []),
    )


async def _validate_workload_access_token(
    config: AuthConfig,
    request: Request,
    token: str,
    workload_token_exchange_service: WorkloadTokenExchangeService,
) -> TokenClaims | None:
    if not config.oidc.workload_token_exchange_enabled:
        return None

    try:
        signing_key = await workload_token_exchange_service.workload_signing_key_async(config)
        public_key = signing_key.private_key.public_key()
    except Exception as exc:
        logger.exception("Failed to load workload access token signing key")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Workload token authentication is misconfigured",
        ) from exc

    try:
        claims = jwt.decode(
            token,
            public_key,
            algorithms=["RS256"],
            audience=list(allowed_audiences(config)),
            issuer=workload_token_issuer(config, request),
            options={"require": ["sub", "iat", "nbf", "exp"]},
            leeway=30,
        )
        subject = claims.get("sub")
        if not isinstance(subject, str) or not subject:
            return None
        return TokenClaims(
            subject=subject,
            email=claims.get("email") if isinstance(claims.get("email"), str) else None,
            groups=groups_from_claim(claims.get("groups", [])),
            scopes=_scopes_from_claims(claims),
            raw_claims=claims,
            actor=_actor_from_claims(claims),
        )
    except jwt.PyJWTError:
        return None


async def _resolve_workload_access_token(
    config: AuthConfig,
    request: Request,
    workload_token_exchange_service: WorkloadTokenExchangeService,
    token: str,
) -> ResolvedBearerToken | None:
    claims = await _validate_workload_access_token(config, request, token, workload_token_exchange_service)
    if claims is None:
        return None
    return ResolvedBearerToken(claims=claims, token_kind="workload_access_token")


async def _resolve_workload_subject_token(
    config: AuthConfig,
    workload_token_exchange_service: WorkloadTokenExchangeService,
    token: str,
) -> ResolvedBearerToken | None:
    if not config.oidc.workload_token_exchange_enabled or not config.oidc.workload_subject_jwks_uri:
        return None

    try:
        claims = await workload_token_exchange_service.decode_jwt_subject_token(config, token)
    except jwt.PyJWTError:
        return None

    subject = claims.get(config.oidc.subject_claim, claims.get("sub"))
    if not isinstance(subject, str) or not subject:
        return None

    email = claims.get(config.oidc.email_claim)
    token_claims = TokenClaims(
        subject=subject,
        email=email if isinstance(email, str) else None,
        groups=groups_from_claim(claims.get(config.oidc.groups_claim, claims.get("groups", []))),
        scopes=_scopes_from_claims(claims),
        raw_claims=claims,
    )
    return ResolvedBearerToken(claims=token_claims, token_kind="workload_subject_token")


async def _resolve_authenticated_bearer_token(
    request: Request,
    workload_token_exchange_service: WorkloadTokenExchangeService,
    access_key_registry: AccessKeyRegistry,
) -> ResolvedBearerToken:
    token = _bearer_token_from_request(request)
    config = get_auth_config()
    jwt_validator = JWTValidator(config)

    async def resolve_workload_access(candidate: str) -> ResolvedBearerToken | None:
        return await _resolve_workload_access_token(config, request, workload_token_exchange_service, candidate)

    async def resolve_workload_subject(candidate: str) -> ResolvedBearerToken | None:
        return await _resolve_workload_subject_token(config, workload_token_exchange_service, candidate)

    try:
        await _validate_token_issuer_boundary(config, request, workload_token_exchange_service, jwt_validator)
    except TokenIssuerBoundaryMisconfigurationError as exc:
        logger.error("Authentication token issuer boundary is misconfigured: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication token issuers are misconfigured",
        ) from exc

    resolved = await resolve_bearer_token(
        config,
        token,
        jwt_validator=jwt_validator,
        extra_resolvers=[resolve_workload_access, resolve_workload_subject],
    )
    if resolved is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid bearer token")

    if resolved.token_kind == "access_key":
        jti = resolved.claims.raw_claims.get("jti")
        if not isinstance(jti, str) or not jti:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid bearer token")
        if not await access_key_registry.is_active(jti, resolved.claims.subject, claims=resolved.claims):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid bearer token")

    return resolved


async def _authenticate_bearer_token(
    request: Request,
    workload_token_exchange_service: WorkloadTokenExchangeService,
    access_key_registry: AccessKeyRegistry,
) -> AuthenticateResponse:
    resolved = await _resolve_authenticated_bearer_token(
        request,
        workload_token_exchange_service,
        access_key_registry,
    )
    return _response_from_resolved(resolved)


async def _ext_authz_bearer_token(
    request: Request,
    workload_token_exchange_service: WorkloadTokenExchangeService,
    access_key_registry: AccessKeyRegistry,
) -> Response:
    resolved = await _resolve_authenticated_bearer_token(
        request,
        workload_token_exchange_service,
        access_key_registry,
    )
    return Response(status_code=status.HTTP_200_OK, headers=resolved.principal_headers())


@router.get(
    "/authenticate",
    response_model=AuthenticateResponse,
    operation_id="get_authenticate_bearer_token",
    responses=_AUTHENTICATE_ERROR_RESPONSES,
)
async def authenticate_bearer_token_get(
    request: Request,
    dependencies: AuthenticateDependency,
) -> AuthenticateResponse:
    return await _authenticate_bearer_token(
        request,
        dependencies.workload_token_exchange_service,
        dependencies.access_key_registry,
    )


@router.post(
    "/authenticate",
    response_model=AuthenticateResponse,
    operation_id="post_authenticate_bearer_token",
    responses=_AUTHENTICATE_ERROR_RESPONSES,
)
async def authenticate_bearer_token_post(
    request: Request,
    dependencies: AuthenticateDependency,
) -> AuthenticateResponse:
    return await _authenticate_bearer_token(
        request,
        dependencies.workload_token_exchange_service,
        dependencies.access_key_registry,
    )


@router.api_route(
    "/ext-authz",
    methods=["DELETE", "GET", "PATCH", "POST", "PUT", "OPTIONS"],
    include_in_schema=False,
)
async def ext_authz_bearer_token(
    request: Request,
    dependencies: AuthenticateDependency,
) -> Response:
    return await _ext_authz_bearer_token(
        request,
        dependencies.workload_token_exchange_service,
        dependencies.access_key_registry,
    )


@router.api_route(
    "/ext-authz/{original_path:path}",
    methods=["DELETE", "GET", "PATCH", "POST", "PUT", "OPTIONS"],
    include_in_schema=False,
)
async def ext_authz_bearer_token_prefixed(
    request: Request,
    original_path: str,
    dependencies: AuthenticateDependency,
) -> Response:
    _ = original_path
    return await _ext_authz_bearer_token(
        request,
        dependencies.workload_token_exchange_service,
        dependencies.access_key_registry,
    )
