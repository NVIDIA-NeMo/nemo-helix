# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Workload identity token exchange endpoints."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import jwt
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from nhx.common.auth.access_keys import public_jwk_from_private_key_pem_async
from nhx.common.auth.signing_keys import RSASigningKey, RSASigningKeyCache
from nhx.common.auth.token_claims import groups_from_claim
from nhx.common.auth.workload_delegations import (
    DOCKER_OPAQUE_WORKLOAD_PROOF_TOKEN_TYPE,
    KUBERNETES_POD_UID_REFERENCE_NAME,
    InvalidWorkloadProofTokenError,
    WorkloadDelegationEntity,
    WorkloadDelegationStore,
    parse_opaque_docker_proof_token,
    reference_delegation_name,
    verify_opaque_docker_proof_token_hash,
)
from nhx.common.config import AuthConfig, get_auth_config, get_platform_config
from nhx.common.entities import EntityClient
from nhx.common.service.dependencies import get_entity_client
from opentelemetry import metrics
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)
meter = metrics.get_meter(__name__)

TOKEN_EXCHANGE_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:token-exchange"
JWT_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:jwt"
ACCESS_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:access_token"

WORKLOAD_TOKEN_PATH = "/apis/auth/token"
WORKLOAD_JWKS_PATH = "/apis/auth/jwks"
DEFAULT_WORKLOAD_AUDIENCE = "nemo-helix"
DEFAULT_WORKLOAD_SCOPE = "openid email groups"

_delegation_lookup_retry_exhausted_total = meter.create_counter(
    name="nhx.auth.workload_delegation.lookup_retry_exhausted.total",
    description="Number of workload delegation lookups that exhausted their retry budget",
)

_TOKEN_EXCHANGE_FORM_REQUEST_BODY: dict[str, Any] = {
    "required": True,
    "content": {
        "application/x-www-form-urlencoded": {
            "schema": {
                "type": "object",
                "required": ["grant_type", "client_id", "subject_token", "subject_token_type"],
                "properties": {
                    "grant_type": {
                        "type": "string",
                        "description": "OAuth 2.0 token exchange grant type.",
                        "enum": [TOKEN_EXCHANGE_GRANT_TYPE],
                    },
                    "client_id": {
                        "type": "string",
                        "description": "Workload token exchange OAuth client ID.",
                    },
                    "subject_token": {
                        "type": "string",
                        "description": "JWT subject token to exchange.",
                    },
                    "subject_token_type": {
                        "type": "string",
                        "description": "Token type identifier for the subject token.",
                        "enum": [JWT_TOKEN_TYPE, DOCKER_OPAQUE_WORKLOAD_PROOF_TOKEN_TYPE],
                    },
                    "requested_token_type": {
                        "type": "string",
                        "description": "Requested token type identifier for the issued token.",
                        "default": ACCESS_TOKEN_TYPE,
                        "enum": [ACCESS_TOKEN_TYPE],
                    },
                    "audience": {
                        "type": "string",
                        "description": "Requested audience for the issued access token.",
                    },
                    "resource": {
                        "type": "string",
                        "description": "Unsupported for the jobs workload token exchange profile.",
                        "not": {},
                    },
                    "actor_token": {
                        "type": "string",
                        "description": "Unsupported for the jobs workload token exchange profile.",
                        "not": {},
                    },
                    "actor_token_type": {
                        "type": "string",
                        "description": "Unsupported for the jobs workload token exchange profile.",
                        "not": {},
                    },
                    "scope": {
                        "type": "string",
                        "description": "Space-separated scopes requested for the issued access token.",
                    },
                },
            }
        }
    },
}

router = APIRouter(tags=["Workload Identity"])


@dataclass(frozen=True)
class _SubjectJWKSCacheEntry:
    jwks: dict[str, Any]
    fetched_at: float


@dataclass(frozen=True)
class VerifiedWorkloadReference:
    name: str
    value: str


@dataclass(frozen=True)
class DecodedSubjectToken:
    claims: dict[str, Any]
    bound_reference: VerifiedWorkloadReference | None = None


@dataclass(frozen=True)
class _SubjectTokenDecoder:
    name: str
    decode: Callable[[], Awaitable[DecodedSubjectToken]]


@dataclass(frozen=True)
class VerifiedSubjectToken:
    subject: str
    groups: list[str]
    email: str | None = None
    delegation_name: str | None = None
    bound_reference: VerifiedWorkloadReference | None = None
    is_docker_proof: bool = False
    is_opaque_docker_proof: bool = False
    opaque_secret: bytes | None = None


@dataclass(frozen=True)
class _SigningKeyLoadRequest:
    kid: str
    private_key_file: str | None
    missing_private_key_message: str
    invalid_private_key_message: str


class _InvalidGrantError(Exception):
    """Raised when a validated subject token is not authorized for delegation."""


class WorkloadTokenExchangeResponse(BaseModel):
    """RFC 8693 token exchange response for workload identity access tokens."""

    access_token: str = Field(description="JWT access token minted for the workload identity.")
    issued_token_type: str = Field(description="Token type identifier for the issued token.")
    token_type: str = Field(description="OAuth token type used in Authorization headers.")
    expires_in: int = Field(description="Lifetime of the access token in seconds.")
    scope: str | None = Field(default=None, description="Space-separated scopes granted to the access token.")


class WorkloadTokenExchangeErrorResponse(BaseModel):
    """RFC 8693 token exchange error response."""

    error: str = Field(
        description=(
            "OAuth 2.0 or RFC 8693 token exchange error code, such as invalid_client, "
            "invalid_request, invalid_scope, or invalid_target."
        ),
    )
    error_description: str | None = Field(
        default=None,
        description="Human-readable ASCII text providing additional information about the error.",
    )
    error_uri: str | None = Field(
        default=None,
        description="URI identifying a human-readable web page with information about the error.",
    )


class JsonWebKey(BaseModel):
    """JSON Web Key object."""

    model_config = ConfigDict(extra="allow")


class JsonWebKeySetResponse(BaseModel):
    """JSON Web Key Set document."""

    keys: list[JsonWebKey] = Field(description="Public signing keys in the JWKS document.")


class ValidationError(BaseModel):
    """FastAPI validation error item."""

    loc: list[str | int] = Field(description="Location of the validation error.")
    msg: str = Field(description="Validation error message.")
    type: str = Field(description="Validation error type.")


class HTTPValidationError(BaseModel):
    """FastAPI request validation error response."""

    detail: list[ValidationError] | None = Field(default=None, description="Validation error details.")


_WORKLOAD_TOKEN_EXCHANGE_SERVICE_STATE_KEY = "workload_token_exchange_service"
_MISSING_WORKLOAD_TOKEN_KEY_ID_MESSAGE = (
    "auth.oidc.workload_token_key_id or auth.token_signing.key_id must be configured for workload token exchange"
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


def _subject_token_key_id(subject_token: str) -> str:
    key_id = jwt.get_unverified_header(subject_token).get("kid")
    if not key_id:
        raise jwt.InvalidTokenError("subject token did not include a signing key id")
    return str(key_id)


def _find_subject_signing_key(key_id: str, jwks: dict[str, Any]) -> Any | None:
    jwk_set = jwt.PyJWKSet.from_dict(jwks)
    signing_keys = [jwk for jwk in jwk_set.keys if jwk.public_key_use in ("sig", None) and jwk.key_id]
    for jwk in signing_keys:
        if jwk.key_id == key_id:
            return jwk.key
    return None


def _validate_subject_jwks(jwks: dict[str, Any]) -> None:
    try:
        jwt.PyJWKSet.from_dict(jwks)
    except (jwt.PyJWTError, TypeError, ValueError, KeyError) as exc:
        raise jwt.InvalidTokenError("Subject token JWKS response was not a valid JWKS") from exc


class WorkloadTokenExchangeService:
    """Stateful helpers for workload token exchange endpoints."""

    def __init__(
        self,
        signing_key_cache: RSASigningKeyCache | None = None,
        *,
        delegation_lookup_retry_timeout_seconds: float = 5.0,
        delegation_lookup_retry_interval_seconds: float = 0.1,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._signing_key_cache = signing_key_cache or RSASigningKeyCache()
        self._subject_jwks_cache: dict[str, _SubjectJWKSCacheEntry] = {}
        self._delegation_lookup_retry_timeout_seconds = delegation_lookup_retry_timeout_seconds
        self._delegation_lookup_retry_interval_seconds = delegation_lookup_retry_interval_seconds
        self._sleep = sleep

    def workload_signing_key(self, config: AuthConfig) -> RSASigningKey:
        load_request = _workload_signing_key_load_request(config)
        return self._signing_key_cache.get_from_file(
            kid=load_request.kid,
            private_key_file=load_request.private_key_file,
            missing_private_key_message=load_request.missing_private_key_message,
            invalid_private_key_message=load_request.invalid_private_key_message,
        )

    async def workload_signing_key_async(self, config: AuthConfig) -> RSASigningKey:
        load_request = _workload_signing_key_load_request(config)
        return await self._signing_key_cache.get_from_file_async(
            kid=load_request.kid,
            private_key_file=load_request.private_key_file,
            missing_private_key_message=load_request.missing_private_key_message,
            invalid_private_key_message=load_request.invalid_private_key_message,
        )

    def public_jwk(self, config: AuthConfig) -> dict[str, Any]:
        load_request = _workload_signing_key_load_request(config)
        return self._signing_key_cache.public_jwk_from_file(
            kid=load_request.kid,
            private_key_file=load_request.private_key_file,
            missing_private_key_message=load_request.missing_private_key_message,
            invalid_private_key_message=load_request.invalid_private_key_message,
        )

    async def public_jwk_async(self, config: AuthConfig) -> dict[str, Any]:
        load_request = _workload_signing_key_load_request(config)
        return await self._signing_key_cache.public_jwk_from_file_async(
            kid=load_request.kid,
            private_key_file=load_request.private_key_file,
            missing_private_key_message=load_request.missing_private_key_message,
            invalid_private_key_message=load_request.invalid_private_key_message,
        )

    async def fetch_subject_jwks(self, config: AuthConfig, *, refresh: bool = False) -> dict[str, Any]:
        jwks_uri = config.oidc.workload_subject_jwks_uri
        if not jwks_uri:
            raise jwt.InvalidTokenError("JWT subject token validation is disabled")

        now = time.monotonic()
        cache_ttl = config.oidc.workload_subject_jwks_cache_ttl_seconds
        cached = self._subject_jwks_cache.get(jwks_uri)
        if cached is not None and cache_ttl > 0 and not refresh and (now - cached.fetched_at) < cache_ttl:
            return cached.jwks

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(jwks_uri)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise jwt.InvalidTokenError(f"Subject token JWKS request failed: {exc}") from exc

        jwks = response.json()
        if not isinstance(jwks, dict):
            raise jwt.InvalidTokenError("Subject token JWKS response was not an object")
        _validate_subject_jwks(jwks)
        if cache_ttl > 0:
            self._subject_jwks_cache[jwks_uri] = _SubjectJWKSCacheEntry(jwks=jwks, fetched_at=time.monotonic())
        return jwks

    async def get_subject_signing_key(self, config: AuthConfig, subject_token: str) -> Any:
        key_id = _subject_token_key_id(subject_token)
        jwks = await self.fetch_subject_jwks(config)
        signing_key = _find_subject_signing_key(key_id, jwks)
        if signing_key is not None:
            return signing_key

        jwks = await self.fetch_subject_jwks(config, refresh=True)
        signing_key = _find_subject_signing_key(key_id, jwks)
        if signing_key is not None:
            return signing_key

        raise jwt.InvalidTokenError(f'Unable to find a signing key that matches: "{key_id}"')

    async def decode_jwt_subject_token(self, config: AuthConfig, subject_token: str) -> dict[str, Any]:
        signing_key = await self.get_subject_signing_key(config, subject_token)
        claims = jwt.decode(
            subject_token,
            signing_key,
            algorithms=["RS256"],
            audience=_workload_subject_audience(config),
            leeway=30,
            options={"require": ["exp"]},
        )
        issuer = claims.get("iss")
        if issuer not in _allowed_subject_issuers(config):
            raise jwt.InvalidIssuerError(f"unexpected subject token issuer: {issuer!r}")
        return claims

    async def decode_subject_token(self, config: AuthConfig, subject_token: str, audience: str) -> DecodedSubjectToken:
        async def decode_configured_jwt_subject_token() -> DecodedSubjectToken:
            return DecodedSubjectToken(claims=await self.decode_jwt_subject_token(config, subject_token))

        errors: list[str] = []
        for decoder in (
            _SubjectTokenDecoder("JWT subject token", decode_configured_jwt_subject_token),
            _SubjectTokenDecoder(
                "Kubernetes TokenReview subject token",
                lambda: _decode_kubernetes_subject_token(config, subject_token, audience),
            ),
        ):
            try:
                return await decoder.decode()
            except jwt.InvalidTokenError as exc:
                message = str(exc) or exc.__class__.__name__
                errors.append(f"{decoder.name}: {message}")
        raise jwt.InvalidTokenError("; ".join(errors))

    async def get_delegation_with_retry(
        self,
        store: WorkloadDelegationStore,
        name: str,
        *,
        retry: bool,
    ) -> WorkloadDelegationEntity | None:
        """Fetch a delegation row by name with a bounded retry for Kubernetes startup races."""
        if not retry:
            return await store.get(name)

        retry_attempts = (
            max(
                0,
                math.ceil(
                    self._delegation_lookup_retry_timeout_seconds / self._delegation_lookup_retry_interval_seconds
                ),
            )
            if self._delegation_lookup_retry_timeout_seconds > 0 and self._delegation_lookup_retry_interval_seconds > 0
            else 0
        )
        while True:
            entity = await store.get(name)
            if entity is not None:
                return entity
            if retry_attempts <= 0:
                _delegation_lookup_retry_exhausted_total.add(1)
                return None
            retry_attempts -= 1
            await self._sleep(self._delegation_lookup_retry_interval_seconds)


def get_workload_token_exchange_service(request: Request) -> WorkloadTokenExchangeService:
    service = getattr(request.app.state, _WORKLOAD_TOKEN_EXCHANGE_SERVICE_STATE_KEY, None)
    if isinstance(service, WorkloadTokenExchangeService):
        return service

    service = WorkloadTokenExchangeService()
    setattr(request.app.state, _WORKLOAD_TOKEN_EXCHANGE_SERVICE_STATE_KEY, service)
    return service


def _dedupe_public_jwks(keys: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for key in keys:
        identity = json.dumps(key, sort_keys=True)
        if identity in seen:
            continue
        seen.add(identity)
        deduped.append(key)
    return deduped


async def auth_jwks_response(
    config: AuthConfig,
    workload_token_exchange_service: WorkloadTokenExchangeService,
) -> JsonWebKeySetResponse:
    keys: list[dict[str, Any]] = []
    if config.oidc.workload_token_exchange_enabled:
        keys.append(await workload_token_exchange_service.public_jwk_async(config))
    if config.access_keys.enabled:
        keys.append(await public_jwk_from_private_key_pem_async(config))
    return JsonWebKeySetResponse(keys=[JsonWebKey(**key) for key in _dedupe_public_jwks(keys)])


def _oauth_error(status_code: int, error: str, description: str) -> JSONResponse:
    # Keep descriptions passed to clients fixed and non-sensitive. Detailed
    # validation errors are logged at call sites instead of returned here.
    return JSONResponse(
        status_code=status_code,
        content={
            "error": error,
            "error_description": description,
        },
    )


def _workload_client_id(config: AuthConfig) -> str:
    return config.oidc.workload_client_id or config.oidc.client_id


def _workload_subject_audience(config: AuthConfig) -> str:
    return _workload_client_id(config) or DEFAULT_WORKLOAD_AUDIENCE


def _default_workload_audience(config: AuthConfig) -> str:
    return config.oidc.workload_audience or config.oidc.audience or DEFAULT_WORKLOAD_AUDIENCE


def allowed_audiences(config: AuthConfig) -> set[str]:
    return {_default_workload_audience(config), *config.oidc.workload_allowed_audiences}


def _validated_audience(config: AuthConfig, requested_audience: Any) -> str:
    audience = str(requested_audience or _default_workload_audience(config))
    if audience not in allowed_audiences(config):
        raise jwt.InvalidAudienceError(f"unexpected requested audience: {audience!r}")
    return audience


def _epoch_seconds(value: datetime) -> int:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return int(value.timestamp())


def _split_scope(scope: Any) -> list[str]:
    if not isinstance(scope, str):
        return []
    return [part for part in scope.split() if part]


def _granted_workload_scope(config: AuthConfig, requested_scope: Any) -> str | None:
    allowed_scopes = _split_scope(config.oidc.workload_scope or DEFAULT_WORKLOAD_SCOPE)
    requested_scopes = _split_scope(requested_scope) or allowed_scopes
    allowed = set(allowed_scopes)
    granted = [scope for scope in requested_scopes if scope in allowed]
    return " ".join(granted) or None


def _groups_claim_for_gateway_header(groups: Any) -> str | None:
    if isinstance(groups, str):
        return groups
    if isinstance(groups, list):
        return ",".join(str(group).strip() for group in groups if str(group).strip())
    return None


def _allowed_subject_issuers(config: AuthConfig) -> set[str]:
    return {issuer for issuer in config.oidc.workload_subject_issuers if issuer}


def _kubernetes_reviewer_credentials() -> tuple[str, str]:
    service_account_dir = Path("/var/run/secrets/kubernetes.io/serviceaccount")
    reviewer_token = (service_account_dir / "token").read_text(encoding="utf-8").strip()
    ca_path = service_account_dir / "ca.crt"
    return reviewer_token, str(ca_path)


async def _decode_kubernetes_subject_token(
    config: AuthConfig,
    subject_token: str,
    audience: str,
) -> DecodedSubjectToken:
    if not config.oidc.workload_kubernetes_token_review_enabled:
        raise jwt.InvalidTokenError("Kubernetes TokenReview subject token validation is disabled")

    import os

    host = os.environ.get("KUBERNETES_SERVICE_HOST")
    port = os.environ.get("KUBERNETES_SERVICE_PORT", "443")
    if not host:
        raise jwt.InvalidTokenError("Kubernetes service environment is unavailable")

    reviewer_token, ca_path = _kubernetes_reviewer_credentials()
    token_review_url = f"https://{host}:{port}/apis/authentication.k8s.io/v1/tokenreviews"
    try:
        async with httpx.AsyncClient(timeout=10.0, verify=ca_path) as client:
            response = await client.post(
                token_review_url,
                json={
                    "apiVersion": "authentication.k8s.io/v1",
                    "kind": "TokenReview",
                    "spec": {
                        "token": subject_token,
                        "audiences": [audience],
                    },
                },
                headers={
                    "Authorization": f"Bearer {reviewer_token}",
                    "Content-Type": "application/json",
                },
            )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise jwt.InvalidTokenError(f"Kubernetes TokenReview request failed: {exc}") from exc

    token_review = response.json()
    if not isinstance(token_review, dict):
        raise jwt.InvalidTokenError("Kubernetes TokenReview response was not an object")

    status = token_review.get("status", {})
    if not isinstance(status, dict):
        raise jwt.InvalidTokenError("Kubernetes TokenReview response did not include a status object")
    if not status.get("authenticated"):
        raise jwt.InvalidTokenError(status.get("error") or "Kubernetes TokenReview rejected subject token")

    user = status.get("user", {})
    if not isinstance(user, dict):
        raise jwt.InvalidTokenError("Kubernetes TokenReview response did not include a user object")
    subject = user.get("username")
    if not subject:
        raise jwt.InvalidTokenError("Kubernetes TokenReview response did not include a username")

    claims: dict[str, Any] = {
        "sub": subject,
        "groups": user.get("groups", []),
    }
    bound_reference = None
    extra = user.get("extra")
    if isinstance(extra, dict) and KUBERNETES_POD_UID_REFERENCE_NAME in extra:
        pod_uids = extra.get(KUBERNETES_POD_UID_REFERENCE_NAME)
        if not isinstance(pod_uids, list) or len(pod_uids) != 1 or not isinstance(pod_uids[0], str) or not pod_uids[0]:
            raise _InvalidGrantError("Kubernetes TokenReview did not include exactly one pod UID reference")
        bound_reference = VerifiedWorkloadReference(name=KUBERNETES_POD_UID_REFERENCE_NAME, value=pod_uids[0])

    return DecodedSubjectToken(claims=claims, bound_reference=bound_reference)


def _subject_groups_from_claims(claims: dict[str, Any]) -> list[str]:
    return groups_from_claim(claims.get("groups", []))


async def _normalize_verified_subject_token(
    config: AuthConfig,
    workload_token_exchange_service: WorkloadTokenExchangeService,
    *,
    subject_token: str,
    subject_token_type: str,
) -> VerifiedSubjectToken:
    if subject_token_type == DOCKER_OPAQUE_WORKLOAD_PROOF_TOKEN_TYPE:
        try:
            parsed = parse_opaque_docker_proof_token(subject_token)
        except InvalidWorkloadProofTokenError as exc:
            raise _InvalidGrantError("Docker opaque workload proof token is malformed") from exc
        return VerifiedSubjectToken(
            subject=parsed.delegation_name,
            groups=[],
            delegation_name=parsed.delegation_name,
            is_docker_proof=True,
            is_opaque_docker_proof=True,
            opaque_secret=parsed.secret,
        )

    decoded = await workload_token_exchange_service.decode_subject_token(
        config,
        subject_token,
        _workload_subject_audience(config),
    )
    claims = decoded.claims
    subject = claims.get("sub")
    if not isinstance(subject, str) or not subject:
        raise jwt.InvalidTokenError("Subject token did not include a subject")

    return VerifiedSubjectToken(
        subject=subject,
        groups=_subject_groups_from_claims(claims),
        email=claims.get("email") if isinstance(claims.get("email"), str) else None,
        bound_reference=decoded.bound_reference,
    )


def _delegation_lookup_name(verified_subject: VerifiedSubjectToken, audience: str) -> str | None:
    if verified_subject.is_docker_proof:
        return verified_subject.delegation_name
    if verified_subject.bound_reference is None:
        return None
    return reference_delegation_name(
        workload_audience=audience,
        workload_subject=verified_subject.subject,
        bound_reference_name=verified_subject.bound_reference.name,
        bound_reference_value=verified_subject.bound_reference.value,
    )


def _validate_delegation_for_exchange(
    delegation: WorkloadDelegationEntity,
    verified_subject: VerifiedSubjectToken,
    *,
    audience: str,
) -> None:
    if not delegation.is_active(now=datetime.now(timezone.utc)):
        raise _InvalidGrantError("Workload delegation is expired or revoked")
    if delegation.workload_audience != audience:
        raise _InvalidGrantError("Workload delegation audience does not match")
    if delegation.workload_subject != verified_subject.subject:
        raise _InvalidGrantError("Workload delegation subject does not match")

    stored_reference = None
    if delegation.bound_reference_name or delegation.bound_reference_value:
        if not delegation.bound_reference_name or not delegation.bound_reference_value:
            raise _InvalidGrantError("Workload delegation bound reference is incomplete")
        stored_reference = VerifiedWorkloadReference(
            name=delegation.bound_reference_name,
            value=delegation.bound_reference_value,
        )

    if stored_reference != verified_subject.bound_reference:
        raise _InvalidGrantError("Workload delegation bound reference does not match")
    if verified_subject.is_docker_proof and stored_reference is not None:
        raise _InvalidGrantError("Docker workload delegation cannot use a bound reference")

    if verified_subject.is_opaque_docker_proof:
        if verified_subject.opaque_secret is None or not delegation.opaque_subject_token_hash:
            raise _InvalidGrantError("Docker opaque workload delegation is missing proof-token state")
        if not verify_opaque_docker_proof_token_hash(
            verified_subject.opaque_secret,
            delegation.opaque_subject_token_hash,
        ):
            raise _InvalidGrantError("Docker opaque workload proof-token hash does not match")
    elif delegation.opaque_subject_token_hash:
        raise _InvalidGrantError("Workload delegation requires an opaque proof token")


def _build_workload_only_claims(verified_subject: VerifiedSubjectToken) -> dict[str, Any]:
    claims: dict[str, Any] = {"sub": verified_subject.subject}
    if verified_subject.email:
        claims["email"] = verified_subject.email
    if verified_subject.groups:
        claims["groups"] = ",".join(verified_subject.groups)
    return claims


def _build_delegated_claims(
    delegation: WorkloadDelegationEntity,
    verified_subject: VerifiedSubjectToken,
) -> dict[str, Any]:
    delegated_principal = delegation.auth_context.to_principal().effective_principal
    claims: dict[str, Any] = {"sub": delegated_principal.id}
    if delegated_principal.email:
        claims["email"] = delegated_principal.email
    if delegated_principal.groups:
        claims["groups"] = ",".join(delegated_principal.groups)

    actor_claims: dict[str, Any] = {"sub": verified_subject.subject}
    if verified_subject.groups:
        actor_claims["groups"] = ",".join(verified_subject.groups)
    claims["act"] = actor_claims
    return claims


@router.get(
    "/jwks",
    summary="Workload identity token exchange JWKS",
    description="Return the public signing key for workload identity access tokens minted by the NeMo auth service.",
    response_model=JsonWebKeySetResponse,
)
async def jwks(
    workload_token_exchange_service: WorkloadTokenExchangeService = Depends(get_workload_token_exchange_service),
) -> JsonWebKeySetResponse:
    """Return workload identity exchange signing keys."""
    return await auth_jwks_response(get_auth_config(), workload_token_exchange_service)


@router.post(
    "/token",
    summary="Exchange a workload identity subject token",
    description="Exchange a configured workload identity subject token for a NeMo Helix access token.",
    response_model=WorkloadTokenExchangeResponse,
    responses={
        400: {
            "description": "RFC 8693 token exchange error",
            "model": WorkloadTokenExchangeErrorResponse,
        },
        401: {
            "description": "OAuth 2.0 invalid_client error",
            "model": WorkloadTokenExchangeErrorResponse,
        },
    },
    openapi_extra={"requestBody": _TOKEN_EXCHANGE_FORM_REQUEST_BODY},
)
async def token_exchange(
    request: Request,
    workload_token_exchange_service: WorkloadTokenExchangeService = Depends(get_workload_token_exchange_service),
    entity_client: EntityClient | None = Depends(get_entity_client),
) -> WorkloadTokenExchangeResponse | JSONResponse:
    """Exchange an RFC 8693 workload identity subject token for a NeMo access token."""
    config = get_auth_config()
    if not config.oidc.workload_token_exchange_enabled:
        return _oauth_error(400, "invalid_request", "Workload token exchange is not enabled")

    form = await request.form()
    client_id = _workload_client_id(config)

    if form.get("grant_type") != TOKEN_EXCHANGE_GRANT_TYPE:
        return _oauth_error(400, "unsupported_grant_type", "Only RFC 8693 token exchange is supported")
    if form.get("client_id") != client_id:
        return _oauth_error(401, "invalid_client", "Unknown workload token exchange client")
    subject_token_type = str(form.get("subject_token_type") or "")
    if subject_token_type not in {JWT_TOKEN_TYPE, DOCKER_OPAQUE_WORKLOAD_PROOF_TOKEN_TYPE}:
        return _oauth_error(400, "invalid_request", "subject_token_type must be a supported workload token type")
    if form.get("requested_token_type", ACCESS_TOKEN_TYPE) != ACCESS_TOKEN_TYPE:
        return _oauth_error(400, "invalid_request", "requested_token_type must be access_token")
    if "actor_token" in form or "actor_token_type" in form:
        return _oauth_error(
            400,
            "invalid_request",
            "actor_token is not supported for this workload token exchange profile",
        )

    if "resource" in form:
        return _oauth_error(400, "invalid_target", "resource is not supported for workload token exchange")

    audience_values = [value for value in form.getlist("audience") if str(value or "").strip()]
    if len(audience_values) > 1:
        return _oauth_error(400, "invalid_target", "Only one audience is supported for workload token exchange")
    requested_audience = audience_values[0] if audience_values else None

    subject_token = form.get("subject_token")
    if not subject_token:
        return _oauth_error(400, "invalid_request", "subject_token is required")

    try:
        audience = _validated_audience(config, requested_audience)
    except jwt.InvalidAudienceError as exc:
        logger.info("Requested token audience validation failed: %s", exc)
        return _oauth_error(400, "invalid_target", "Requested audience is not allowed")

    try:
        verified_subject = await _normalize_verified_subject_token(
            config,
            workload_token_exchange_service,
            subject_token=str(subject_token),
            subject_token_type=subject_token_type,
        )
    except jwt.InvalidTokenError as exc:
        logger.info("Subject token validation failed: %s", exc)
        # RFC 8693 clients only need a stable invalid_request response. Avoid
        # returning decoder details that may include infrastructure internals.
        return _oauth_error(400, "invalid_request", "Could not validate subject token")
    except _InvalidGrantError as exc:
        logger.info("Subject token was not eligible for workload delegation: %s", exc)
        return _oauth_error(400, "invalid_request", "Subject token is not authorized for workload delegation")

    try:
        lookup_name = _delegation_lookup_name(verified_subject, audience)
        delegation = None
        if lookup_name is not None:
            if entity_client is None:
                raise _InvalidGrantError("Entity client is unavailable for workload delegation lookup")
            delegation_store = WorkloadDelegationStore(entity_client)
            delegation = await workload_token_exchange_service.get_delegation_with_retry(
                delegation_store,
                lookup_name,
                retry=verified_subject.bound_reference is not None,
            )
            if delegation is None:
                raise _InvalidGrantError("Workload delegation is not ready")
            _validate_delegation_for_exchange(delegation, verified_subject, audience=audience)
    except _InvalidGrantError as exc:
        logger.info("Workload delegation lookup failed: %s", exc)
        return _oauth_error(400, "invalid_request", "Subject token is not authorized for workload delegation")

    now = int(time.time())
    scope = _granted_workload_scope(config, form.get("scope"))
    subject_claims = (
        _build_delegated_claims(delegation, verified_subject)
        if delegation is not None
        else _build_workload_only_claims(verified_subject)
    )
    configured_exp = now + config.oidc.workload_token_ttl_seconds
    token_exp = min(configured_exp, _epoch_seconds(delegation.expires_at)) if delegation is not None else configured_exp
    expires_in = max(0, token_exp - now)
    exchanged_claims: dict[str, Any] = {
        "iss": workload_token_issuer(config, request),
        **subject_claims,
        "aud": audience,
        "iat": now,
        "nbf": now,
        "exp": token_exp,
    }
    if scope:
        exchanged_claims["scope"] = scope

    signing_key = await workload_token_exchange_service.workload_signing_key_async(config)
    access_token = jwt.encode(
        exchanged_claims,
        signing_key.private_key,
        algorithm="RS256",
        headers={"kid": signing_key.kid},
    )
    return WorkloadTokenExchangeResponse(
        access_token=access_token,
        issued_token_type=ACCESS_TOKEN_TYPE,
        token_type="Bearer",
        expires_in=expires_in,
        scope=scope,
    )
