# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Any

import httpx
import jwt
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from nemo_helix_plugin.auth.types import (
    AuthenticateErrorResponse as AuthenticateErrorResponse,
)
from nemo_helix_plugin.auth.types import (
    AuthenticateResponse as AuthenticateResponse,
)
from nhx.common.auth.bearer import MalformedBearerTokenError
from nhx.common.auth.jwt import JWTValidator
from nhx.common.auth.middleware import request_bearer_token
from nhx.common.auth.token_claims import ActorClaims, TokenClaims, groups_from_claim, scopes_from_claim
from nhx.common.auth.token_resolver import ResolvedBearerToken, resolve_bearer_token
from nhx.common.config import AuthConfig, get_auth_config
from nhx.core.auth.api.v2.workload_token_exchange import (
    WorkloadTokenExchangeService,
    allowed_audiences,
    get_workload_token_exchange_service,
    workload_jwks_url,
    workload_token_issuer,
)
from nhx.core.auth.app.access_keys import AccessKeyRegistry, get_access_key_registry
from nhx.core.auth.oidc_broker.routes import get_credential_store
from nhx.core.auth.oidc_broker.web_session import (
    WEB_SESSION_COOKIE,
    WEB_SESSION_CSRF_HEADER,
    WEB_SESSION_CSRF_VALUE,
    mint_web_session_access_token,
    resolve_web_session,
    web_sessions_enabled,
)
from nhx.core.entities.app.repository import AccountCredentialStore

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


class _Verdict(Enum):
    """The outcome of the last issuer-boundary check."""

    CONSISTENT = "consistent"  # Checked: IdP and NeMo-issued tokens cannot be confused.
    UNVERIFIED = "unverified"  # The IdP or the signing key could not be read; allowed, as before.
    MISCONFIGURED = "misconfigured"  # They overlap; authentication fails closed.


# How long each verdict is reused before the boundary is checked again; rechecks run in the
# background, so no request waits on one. A consistent verdict lasts as long as the validator's
# cached JWKS, so a key the IdP adds later is still caught. The other two are rechecked sooner, so a
# recovered IdP or a fixed misconfiguration is noticed within about 30s. That figure is a judgement
# call rather than a derived one: shorter would contact a failing IdP more often (each recheck can
# take up to two 10s fetch timeouts, and only one runs at a time), longer would delay recovery.
_REUSE_VERDICT_SECONDS: dict[_Verdict, float] = {
    _Verdict.CONSISTENT: 3600.0,
    _Verdict.UNVERIFIED: 30.0,
    _Verdict.MISCONFIGURED: 30.0,
}


class _IssuerBoundary:
    """One auth config's shared JWT validator and its reused issuer-boundary verdict.

    The ext-authz callout runs on every gateway request under Envoy's timeout, so it must not pay
    for IdP discovery and JWKS fetches each time. Sharing the validator keeps its discovery and
    JWKS caches; reusing the verdict keeps the boundary check from re-fetching between them.
    """

    def __init__(self, config: AuthConfig) -> None:
        self.config = config
        self.jwt_validator = JWTValidator(config)
        self._first_check_lock = asyncio.Lock()
        self._verdict: _Verdict | None = None
        # Why authentication fails closed while the verdict is MISCONFIGURED; None otherwise.
        self._misconfiguration_reason: str | None = None
        self._reuse_verdict_until = 0.0
        # The background recheck in progress, kept so it is not garbage-collected mid-flight.
        self._recheck_task: asyncio.Task[None] | None = None

    async def check(self, workload_token_exchange_service: WorkloadTokenExchangeService) -> None:
        """Raise if IdP and NeMo-issued tokens could be confused, reusing a recent verdict."""
        if self._verdict is None:
            # With no verdict yet, the first check decides before any request is answered, so a
            # misconfiguration fails closed from the first request. Concurrent first requests
            # wait for that one check instead of each fetching from the IdP.
            async with self._first_check_lock:
                if self._verdict is None:
                    await self._recheck(workload_token_exchange_service)
        elif time.monotonic() >= self._reuse_verdict_until and self._recheck_task is None:
            # Keep answering from the expired verdict while one task rechecks, so a slow IdP never
            # holds up requests. A MISCONFIGURED verdict keeps failing closed until it is cleared.
            self._recheck_task = asyncio.create_task(self._recheck_in_background(workload_token_exchange_service))

        if self._misconfiguration_reason is not None:
            # A new exception each time: re-raising one instance would grow its traceback, and keep
            # every failed request's frames alive, for as long as the verdict is reused.
            raise TokenIssuerBoundaryMisconfigurationError(self._misconfiguration_reason)

    async def _recheck(self, workload_token_exchange_service: WorkloadTokenExchangeService) -> None:
        # A recheck reads the IdP as it is now. The validator caches discovery and JWKS for an
        # hour, so reusing it would replay what the IdP served last time.
        jwt_validator = self.jwt_validator if self._verdict is None else JWTValidator(self.config)
        try:
            finished = await _validate_token_issuer_boundary(
                self.config, workload_token_exchange_service, jwt_validator
            )
        except TokenIssuerBoundaryMisconfigurationError as exc:
            verdict, reason = _Verdict.MISCONFIGURED, str(exc)
        else:
            verdict, reason = (_Verdict.CONSISTENT if finished else _Verdict.UNVERIFIED), None
        self.jwt_validator = jwt_validator
        self._verdict, self._misconfiguration_reason = verdict, reason
        self._reuse_verdict_until = time.monotonic() + _REUSE_VERDICT_SECONDS[verdict]

    async def _recheck_in_background(self, workload_token_exchange_service: WorkloadTokenExchangeService) -> None:
        try:
            await self._recheck(workload_token_exchange_service)
        except Exception:
            # Keep the current verdict and try again after the short window rather than on the
            # next request.
            logger.exception("Issuer-boundary recheck failed; keeping the current verdict")
            self._reuse_verdict_until = time.monotonic() + _REUSE_VERDICT_SECONDS[_Verdict.UNVERIFIED]
        finally:
            self._recheck_task = None


_ISSUER_BOUNDARY_STATE_KEY = "issuer_boundary"


async def get_issuer_boundary(request: Request) -> _IssuerBoundary:
    """Return this app's shared validator state, starting afresh when the auth config changes.

    ``async`` so FastAPI runs it on the event loop; as a sync dependency it would run in a worker
    thread, and concurrent first requests could each build their own state.
    """
    config = get_auth_config()
    issuer_boundary = getattr(request.app.state, _ISSUER_BOUNDARY_STATE_KEY, None)
    if isinstance(issuer_boundary, _IssuerBoundary) and issuer_boundary.config is config:
        return issuer_boundary

    issuer_boundary = _IssuerBoundary(config)
    setattr(request.app.state, _ISSUER_BOUNDARY_STATE_KEY, issuer_boundary)
    return issuer_boundary


@dataclass(frozen=True)
class AuthenticateDependencies:
    workload_token_exchange_service: WorkloadTokenExchangeService
    access_key_registry: AccessKeyRegistry
    issuer_boundary: _IssuerBoundary


def get_authenticate_dependencies(
    workload_token_exchange_service: WorkloadTokenExchangeService = Depends(get_workload_token_exchange_service),
    access_key_registry: AccessKeyRegistry = Depends(get_access_key_registry),
    issuer_boundary: _IssuerBoundary = Depends(get_issuer_boundary),
) -> AuthenticateDependencies:
    return AuthenticateDependencies(
        workload_token_exchange_service=workload_token_exchange_service,
        access_key_registry=access_key_registry,
        issuer_boundary=issuer_boundary,
    )


AuthenticateDependency = Annotated[AuthenticateDependencies, Depends(get_authenticate_dependencies)]
CredentialStoreDependency = Annotated[AccountCredentialStore, Depends(get_credential_store)]


class TokenIssuerBoundaryMisconfigurationError(Exception):
    """Raised when IdP and NeMo-issued token validation paths overlap."""


def _bearer_token_from_request(request: Request, original_path: str | None = None) -> str:
    """The request's bearer token; for an ext-authz callout, read as the original request would carry it."""
    try:
        token = request_bearer_token(
            "/" + original_path if original_path is not None else request.url.path, request.headers
        )
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
    workload_token_exchange_service: WorkloadTokenExchangeService,
    jwt_validator: JWTValidator,
) -> bool:
    """Raise if IdP and NeMo-issued tokens could be confused; return whether the check finished.

    ``False`` means the IdP or the signing key could not be read, so the check was skipped.
    """
    if not (config.oidc.enabled and config.oidc.workload is not None):
        return True

    workload_issuer = _normalize_url(workload_token_issuer(config))
    idp_issuers = {_normalize_url(issuer) for issuer in [config.oidc.issuer, *config.oidc.additional_issuers] if issuer}
    if workload_issuer in idp_issuers:
        raise TokenIssuerBoundaryMisconfigurationError("OIDC issuer must not match NeMo workload token issuer")

    try:
        idp_jwks_uri = _normalize_url(await jwt_validator.jwks_uri())
    except (httpx.HTTPError, jwt.PyJWTError):
        return False

    nemo_jwks_uri = _normalize_url(workload_jwks_url())
    if idp_jwks_uri == nemo_jwks_uri:
        raise TokenIssuerBoundaryMisconfigurationError("OIDC JWKS URI must not match NeMo workload JWKS URI")

    try:
        signing_key = await workload_token_exchange_service.public_jwk_async(config)
    except (OSError, RuntimeError, ValueError):
        return False

    nemo_material = _normalize_jwk_public_material(signing_key)
    if nemo_material is None:
        return True

    try:
        idp_jwks = await jwt_validator.jwks()
    except (httpx.HTTPError, jwt.PyJWTError):
        return False

    idp_keys = idp_jwks.get("keys", [])
    if not isinstance(idp_keys, list):
        return True

    for jwk in idp_keys:
        if isinstance(jwk, dict) and _normalize_jwk_public_material(jwk) == nemo_material:
            raise TokenIssuerBoundaryMisconfigurationError(
                "OIDC JWKS must not contain the NeMo workload token signing key"
            )
    return True


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
    token: str,
    workload_token_exchange_service: WorkloadTokenExchangeService,
) -> TokenClaims | None:
    if config.oidc.workload is None:
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
            issuer=workload_token_issuer(config),
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
    workload_token_exchange_service: WorkloadTokenExchangeService,
    token: str,
) -> ResolvedBearerToken | None:
    claims = await _validate_workload_access_token(config, token, workload_token_exchange_service)
    if claims is None:
        return None
    return ResolvedBearerToken(claims=claims, token_kind="workload_access_token")


async def _resolve_workload_subject_token(
    config: AuthConfig,
    workload_token_exchange_service: WorkloadTokenExchangeService,
    token: str,
) -> ResolvedBearerToken | None:
    workload = config.oidc.workload
    if workload is None or not workload.subject_jwks_uri:
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
    issuer_boundary: _IssuerBoundary,
    original_path: str | None = None,
) -> ResolvedBearerToken:
    token = _bearer_token_from_request(request, original_path)
    config = issuer_boundary.config
    jwt_validator = issuer_boundary.jwt_validator

    async def resolve_workload_access(candidate: str) -> ResolvedBearerToken | None:
        return await _resolve_workload_access_token(config, workload_token_exchange_service, candidate)

    async def resolve_workload_subject(candidate: str) -> ResolvedBearerToken | None:
        return await _resolve_workload_subject_token(config, workload_token_exchange_service, candidate)

    try:
        await issuer_boundary.check(workload_token_exchange_service)
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
    issuer_boundary: _IssuerBoundary,
) -> AuthenticateResponse:
    resolved = await _resolve_authenticated_bearer_token(
        request,
        workload_token_exchange_service,
        access_key_registry,
        issuer_boundary,
    )
    return _response_from_resolved(resolved)


async def _ext_authz_bearer_token(
    request: Request,
    workload_token_exchange_service: WorkloadTokenExchangeService,
    access_key_registry: AccessKeyRegistry,
    issuer_boundary: _IssuerBoundary,
    credential_store: AccountCredentialStore,
    original_path: str | None = None,
) -> Response:
    if request.headers.get("authorization") is None and request.cookies.get(WEB_SESSION_COOKIE):
        if not web_sessions_enabled(get_auth_config()):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid session")
        principal = await resolve_web_session(request, credential_store)
        if principal is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid session")
        if request.method in {"POST", "PUT", "PATCH", "DELETE"} and (
            request.headers.get(WEB_SESSION_CSRF_HEADER) != WEB_SESSION_CSRF_VALUE
        ):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Missing or invalid X-Source")
        if get_auth_config().oidc.workload is not None:
            access_token = await mint_web_session_access_token(
                get_auth_config(),
                workload_token_exchange_service,
                principal,
            )
            return Response(
                status_code=status.HTTP_200_OK,
                headers={"Authorization": f"Bearer {access_token}"},
            )
        return Response(status_code=status.HTTP_200_OK, headers=principal.get_headers())

    resolved = await _resolve_authenticated_bearer_token(
        request,
        workload_token_exchange_service,
        access_key_registry,
        issuer_boundary,
        original_path,
    )
    if get_auth_config().oidc.workload is not None:
        return Response(status_code=status.HTTP_200_OK)
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
        dependencies.issuer_boundary,
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
        dependencies.issuer_boundary,
    )


@router.api_route(
    "/ext-authz",
    methods=["DELETE", "GET", "PATCH", "POST", "PUT", "OPTIONS"],
    include_in_schema=False,
)
async def ext_authz_bearer_token(
    request: Request,
    dependencies: AuthenticateDependency,
    credential_store: CredentialStoreDependency,
) -> Response:
    return await _ext_authz_bearer_token(
        request,
        dependencies.workload_token_exchange_service,
        dependencies.access_key_registry,
        dependencies.issuer_boundary,
        credential_store,
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
    credential_store: CredentialStoreDependency,
) -> Response:
    return await _ext_authz_bearer_token(
        request,
        dependencies.workload_token_exchange_service,
        dependencies.access_key_registry,
        dependencies.issuer_boundary,
        credential_store,
        original_path,
    )
