# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Server-side OIDC login broker for browser and CLI user authentication.

The auth service completes brokered public and confidential OAuth flows. The
browser receives an HTTP-only session cookie. The CLI receives a provider access
token and an opaque NeMo refresh handle. Provider refresh tokens and
confidential-client secrets never leave this service.
"""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Literal
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import RedirectResponse
from nhx.common.auth.token_resolver import resolve_bearer_token
from nhx.common.config import get_auth_config, get_platform_config
from nhx.common.config.base import OIDCConfidentialClientConfig, OIDCPublicClientConfig
from nhx.core.auth.app.account_resolution import get_account_session_maker
from nhx.core.auth.oidc_broker.crypto import (
    client_secret_basic_header,
    decrypt_secret,
    encrypt_secret,
    pkce_challenge,
    read_env,
)
from nhx.core.auth.oidc_broker.store import OidcLoginStore
from nhx.core.auth.oidc_broker.web_session import (
    WEB_SESSION_COOKIE,
    WEB_SESSION_CSRF_HEADER,
    WEB_SESSION_CSRF_VALUE,
    resolve_web_session,
)
from nhx.core.entities.app.repository import (
    AccountCredentialRecord,
    AccountCredentialStore,
    AccountIdentityStore,
)
from nhx.core.entities.app.repository.account_identity import AccountIdentityRecord
from pydantic import BaseModel, ConfigDict, Field
from starlette.datastructures import URL

router = APIRouter(tags=["Authentication"])

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
_LOGIN_TTL_SECONDS = 600
_TOKEN_AUTHORIZATION_CODE_TTL_SECONDS = 120
_WEB_SESSION_TTL_SECONDS = 60 * 60 * 8
_BROKER_REFRESH_TTL_SECONDS = 60 * 60 * 24 * 30
_DEFAULT_TOKEN_EXPIRES_IN_SECONDS = 3600


class TokenAuthorizationStart(BaseModel):
    redirect_uri: str
    code_challenge: str = Field(min_length=43, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    state: str = Field(min_length=16, max_length=512)


class AuthorizationCodeBrokerTokenRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    grant_type: Literal["authorization_code"]
    code: str
    code_verifier: str = Field(min_length=43, max_length=128, pattern=r"^[A-Za-z0-9._~-]+$")


class RefreshTokenBrokerTokenRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    grant_type: Literal["refresh_token"]
    refresh_token: str


BrokerTokenRequest = Annotated[
    AuthorizationCodeBrokerTokenRequest | RefreshTokenBrokerTokenRequest,
    Field(discriminator="grant_type"),
]


class WebSessionResponse(BaseModel):
    id: str
    email: str | None = None
    groups: list[str]
    account_id: str
    authz_aliases: list[str]


async def get_login_store() -> OidcLoginStore:
    key = _session_encryption_key()
    try:
        session_maker = await get_account_session_maker()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="OIDC server-side login is not available") from exc
    return OidcLoginStore(session_maker, key)


async def get_credential_store() -> AccountCredentialStore:
    try:
        return AccountCredentialStore(await get_account_session_maker())
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="OIDC server-side login is not available") from exc


async def get_identity_store() -> AccountIdentityStore:
    try:
        return AccountIdentityStore(await get_account_session_maker())
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="OIDC server-side login is not available") from exc


LoginStore = Annotated[OidcLoginStore, Depends(get_login_store)]
CredentialStore = Annotated[AccountCredentialStore, Depends(get_credential_store)]
IdentityStore = Annotated[AccountIdentityStore, Depends(get_identity_store)]
BrokerClientName = Literal["public", "confidential"]
BearerTokenSource = Literal["access_token", "id_token"]


@dataclass(frozen=True)
class BrokerClient:
    name: BrokerClientName
    client_id: str
    authorization_endpoint: str
    token_endpoint: str
    redirect_uri: str
    default_scopes: str
    bearer_token_source: BearerTokenSource
    client_secret_env_var: str | None = None


def _require_confidential() -> OIDCConfidentialClientConfig:
    confidential = get_auth_config().oidc.confidential_client
    if confidential is None:
        raise HTTPException(status_code=404, detail="Confidential OIDC login is not configured")
    if not confidential.token_endpoint or not confidential.authorization_endpoint:
        raise HTTPException(status_code=503, detail="Confidential OIDC login is not available")
    return confidential


def _require_public() -> OIDCPublicClientConfig:
    public = get_auth_config().oidc.public_client
    if public is None or not public.server_side_sessions:
        raise HTTPException(status_code=404, detail="Public OIDC server-side login is not configured")
    if not public.token_endpoint or not public.authorization_endpoint:
        raise HTTPException(status_code=503, detail="Public OIDC server-side login is not available")
    return public


def _broker_client(name: BrokerClientName) -> BrokerClient:
    if name == "public":
        public = _require_public()
        advertised_base_url = get_platform_config().effective_advertised_base_url.rstrip("/")
        return BrokerClient(
            name="public",
            client_id=public.client_id,
            authorization_endpoint=_required_endpoint(public.authorization_endpoint),
            token_endpoint=_required_endpoint(public.token_endpoint),
            redirect_uri=f"{advertised_base_url}/apis/auth/v2/login/callback",
            default_scopes=public.default_scopes,
            bearer_token_source=public.bearer_token_source,
        )

    confidential = _require_confidential()
    return BrokerClient(
        name="confidential",
        client_id=confidential.client_id,
        authorization_endpoint=_required_endpoint(confidential.authorization_endpoint),
        token_endpoint=_required_endpoint(confidential.token_endpoint),
        redirect_uri=confidential.login_redirect_uri,
        default_scopes=confidential.default_scopes,
        bearer_token_source=confidential.bearer_token_source,
        client_secret_env_var=confidential.client_secret_env_var,
    )


def _required_env(name: str | None) -> str:
    if not name:
        raise HTTPException(status_code=503, detail="OIDC server-side login is not available")
    try:
        return read_env(name)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail="OIDC server-side login is not available") from exc


def _required_endpoint(value: str | None) -> str:
    if value is None:
        raise HTTPException(status_code=503, detail="OIDC server-side login is not available")
    return value


def _session_encryption_key() -> str:
    sessions = get_auth_config().oidc.server_sessions
    if sessions is None:
        raise HTTPException(status_code=503, detail="OIDC server-side login is not available")
    return _required_env(sessions.encryption_key_env_var)


def _encrypted_tokens(tokens: dict[str, object]) -> str:
    serializable = {key: value for key, value in tokens.items() if isinstance(value, (str, int, float, bool))}
    return encrypt_secret(json.dumps(serializable), _session_encryption_key())


def _decrypted_tokens(record: AccountCredentialRecord) -> dict[str, object]:
    if record.encrypted_payload is None:
        raise HTTPException(status_code=401, detail="Credential payload is unavailable")
    try:
        payload = json.loads(decrypt_secret(record.encrypted_payload, _session_encryption_key()))
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=401, detail="Credential payload is unavailable") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=401, detail="Credential payload is unavailable")
    return payload


async def _resolve_identity(
    tokens: dict[str, object], identity_store: AccountIdentityStore
) -> tuple[AccountIdentityRecord, dict[str, Any]]:
    access_token = tokens.get("access_token")
    if not isinstance(access_token, str):
        raise HTTPException(status_code=502, detail="Identity provider token response was incomplete")
    resolved = await resolve_bearer_token(get_auth_config(), access_token, skip_access_key_check=True)
    if resolved is None or resolved.token_kind != "oidc_access_token":
        raise HTTPException(status_code=401, detail="Identity provider returned an invalid access token")

    claims = resolved.claims
    raw_claims = dict(claims.raw_claims)
    display_name = raw_claims.get("name")
    identity = await identity_store.resolve_or_materialize(
        issuer=get_auth_config().oidc.issuer,
        subject=claims.subject,
        subject_claim=get_auth_config().oidc.subject_claim,
        account_type="user",
        display_name=display_name if isinstance(display_name, str) else claims.email or claims.subject,
        primary_email=claims.email,
        claims_snapshot=raw_claims,
        linked_via="oidc_broker_login",
    )
    aliases = [claims.subject]
    if claims.email and claims.email not in aliases:
        aliases.append(claims.email)
    return identity, {
        "principal_id": claims.subject,
        "email": claims.email,
        "groups": list(claims.groups),
        "authz_aliases": aliases,
    }


async def _finish_token_authorization(
    payload: dict[str, str],
    code: str,
    credential_store: AccountCredentialStore,
    identity_store: AccountIdentityStore,
) -> RedirectResponse:
    client = _broker_client(_broker_client_name(payload["oidc_client"]))
    tokens = await _exchange_code(client, code, payload["verifier"])
    identity, metadata = await _resolve_identity(tokens, identity_store)
    one_time = secrets.token_urlsafe(32)
    await credential_store.create(
        credential_type="token_authorization_code",
        handle=one_time,
        owner_account_id=identity.account_id,
        account_identity_id=identity.identity_id,
        expires_at=datetime.now(timezone.utc) + timedelta(seconds=_TOKEN_AUTHORIZATION_CODE_TTL_SECONDS),
        public_metadata={
            **metadata,
            "code_challenge": payload["cli_code_challenge"],
            "oidc_client": client.name,
        },
        encrypted_payload=_encrypted_tokens(tokens),
    )
    redirect_url = URL(payload["redirect_uri"]).include_query_params(code=one_time, state=payload["cli_state"])
    return RedirectResponse(str(redirect_url), status_code=302)


def _finish_token_authorization_error(
    payload: dict[str, str],
    *,
    error: str,
    error_description: str | None,
) -> RedirectResponse:
    params = {"error": error, "state": payload["cli_state"]}
    if error_description:
        params["error_description"] = error_description
    redirect_url = URL(payload["redirect_uri"]).include_query_params(**params)
    return RedirectResponse(str(redirect_url), status_code=302)


def _authorization_redirect_url(*, client: BrokerClient, state: str, verifier: str) -> str:
    return str(
        URL(client.authorization_endpoint).include_query_params(
            response_type="code",
            client_id=client.client_id,
            redirect_uri=client.redirect_uri,
            state=state,
            code_challenge=pkce_challenge(verifier),
            code_challenge_method="S256",
            scope=client.default_scopes,
        )
    )


async def _exchange_code(client: BrokerClient, code: str, verifier: str) -> dict[str, object]:
    return await _post_token(
        client,
        {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": client.redirect_uri,
            "code_verifier": verifier,
        },
    )


async def _post_token(client_config: BrokerClient, form: dict[str, str]) -> dict[str, object]:
    headers: dict[str, str] = {}
    if client_config.client_secret_env_var is None:
        form = {**form, "client_id": client_config.client_id}
    else:
        secret = _required_env(client_config.client_secret_env_var)
        headers["Authorization"] = client_secret_basic_header(client_config.client_id, secret)
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(client_config.token_endpoint, data=form, headers=headers)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="Identity provider token request failed") from exc
    if not response.is_success:
        raise HTTPException(status_code=502, detail="Identity provider token request failed")
    try:
        payload = response.json()
    except ValueError as exc:
        raise HTTPException(status_code=502, detail="Identity provider token response was invalid") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=502, detail="Identity provider token response was invalid")
    return payload


def _loopback_redirect(redirect_uri: str) -> None:
    parsed = urlparse(redirect_uri)
    try:
        port = parsed.port
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="redirect_uri must be an http loopback address") from exc
    if (
        parsed.scheme != "http"
        or parsed.hostname not in _LOOPBACK_HOSTS
        or port is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise HTTPException(status_code=400, detail="redirect_uri must be an http loopback address")


def _return_path(value: str) -> str:
    parsed = urlparse(value)
    if not value.startswith("/") or value.startswith("//") or "\\" in value or parsed.scheme or parsed.netloc:
        raise HTTPException(status_code=400, detail="return_to must be a relative path")
    return value


def _broker_client_name(value: object) -> BrokerClientName:
    if value == "public":
        return "public"
    if value == "confidential":
        return "confidential"
    raise HTTPException(status_code=400, detail="OIDC broker client was not found")


def _credential_client(record: AccountCredentialRecord) -> BrokerClient:
    return _broker_client(_broker_client_name(record.public_metadata.get("oidc_client")))


_REDIRECT_RESPONSES: dict[int | str, dict[str, Any]] = {
    302: {
        "description": "Redirect to the identity provider or the CLI loopback callback",
        "headers": {"Location": {"description": "Redirect target", "schema": {"type": "string"}}},
    }
}


@router.get(
    "/v2/login",
    status_code=302,
    response_class=RedirectResponse,
    responses=_REDIRECT_RESPONSES,
)
async def start_web_session_login(
    store: LoginStore,
    return_to: str = "/",
    client: BrokerClientName = "confidential",
) -> RedirectResponse:
    selected_client = _broker_client(client)
    verifier = secrets.token_urlsafe(48)
    state = secrets.token_urlsafe(24)
    await store.put(
        "web_session_txn",
        state,
        {
            "verifier": verifier,
            "return_to": _return_path(return_to),
            "oidc_client": selected_client.name,
        },
        ttl_seconds=_LOGIN_TTL_SECONDS,
    )
    return RedirectResponse(
        _authorization_redirect_url(client=selected_client, state=state, verifier=verifier),
        status_code=302,
    )


@router.get(
    "/v2/login/callback",
    status_code=302,
    response_class=RedirectResponse,
    responses=_REDIRECT_RESPONSES,
)
async def token_authorization_callback(
    state: str,
    store: LoginStore,
    credential_store: CredentialStore,
    identity_store: IdentityStore,
    code: str | None = None,
    error: str | None = None,
    error_description: str | None = None,
) -> RedirectResponse:
    record = await store.pop("token_authorization_txn", state)
    if record is not None:
        if error is not None:
            return _finish_token_authorization_error(
                record.payload,
                error=error,
                error_description=error_description,
            )
        if code is None:
            raise HTTPException(status_code=400, detail="Login callback did not include an authorization code")
        return await _finish_token_authorization(record.payload, code, credential_store, identity_store)

    record = await store.pop("web_session_txn", state)
    if record is None:
        raise HTTPException(status_code=400, detail="Login transaction was not found")
    if error is not None:
        raise HTTPException(status_code=400, detail="Login callback returned an error")
    if code is None:
        raise HTTPException(status_code=400, detail="Login callback did not include an authorization code")
    selected_client = _broker_client(_broker_client_name(record.payload["oidc_client"]))
    tokens = await _exchange_code(selected_client, code, record.payload["verifier"])
    identity, metadata = await _resolve_identity(tokens, identity_store)
    session_id = secrets.token_urlsafe(32)
    await credential_store.create(
        credential_type="web_session",
        handle=session_id,
        owner_account_id=identity.account_id,
        account_identity_id=identity.identity_id,
        expires_at=datetime.now(timezone.utc) + timedelta(seconds=_WEB_SESSION_TTL_SECONDS),
        public_metadata={**metadata, "oidc_client": selected_client.name},
        encrypted_payload=_encrypted_tokens(tokens),
    )
    response = RedirectResponse(record.payload["return_to"], status_code=302)
    response.set_cookie(
        WEB_SESSION_COOKIE,
        session_id,
        max_age=_WEB_SESSION_TTL_SECONDS,
        httponly=True,
        secure=True,
        samesite="lax",
        path="/",
    )
    return response


@router.post("/v2/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(request: Request, credential_store: CredentialStore) -> Response:
    if request.headers.get(WEB_SESSION_CSRF_HEADER) != WEB_SESSION_CSRF_VALUE:
        raise HTTPException(status_code=403, detail="Missing or invalid X-Source")
    session_id = request.cookies.get(WEB_SESSION_COOKIE)
    if session_id:
        await credential_store.revoke("web_session", session_id)
    response = Response(status_code=status.HTTP_204_NO_CONTENT)
    response.delete_cookie(WEB_SESSION_COOKIE, path="/", secure=True, httponly=True, samesite="lax")
    return response


@router.get("/v2/session", response_model=WebSessionResponse)
async def web_session(request: Request, credential_store: CredentialStore) -> WebSessionResponse:
    principal = await resolve_web_session(request, credential_store)
    if principal is None or principal.account_id is None:
        raise HTTPException(status_code=401, detail="Missing session")
    return WebSessionResponse(
        id=principal.id,
        email=principal.email,
        groups=principal.groups,
        account_id=principal.account_id,
        authz_aliases=principal.authz_aliases,
    )


@router.post("/v2/authorize")
async def start_token_authorization(
    body: TokenAuthorizationStart,
    store: LoginStore,
    client: BrokerClientName = "confidential",
) -> dict[str, str]:
    selected_client = _broker_client(client)
    _loopback_redirect(body.redirect_uri)
    verifier = secrets.token_urlsafe(48)
    transaction_id = secrets.token_urlsafe(24)
    await store.put(
        "token_authorization_txn",
        transaction_id,
        {
            "verifier": verifier,
            "redirect_uri": body.redirect_uri,
            "cli_code_challenge": body.code_challenge,
            "cli_state": body.state,
            "oidc_client": selected_client.name,
        },
        ttl_seconds=_LOGIN_TTL_SECONDS,
    )
    advertised_base_url = get_platform_config().effective_advertised_base_url.rstrip("/")
    authorization_url = f"{advertised_base_url}/apis/auth/v2/authorize/{transaction_id}"
    return {"authorization_url": authorization_url, "transaction_id": transaction_id}


@router.get(
    "/v2/authorize/{transaction_id}",
    status_code=302,
    response_class=RedirectResponse,
    responses=_REDIRECT_RESPONSES,
)
async def continue_token_authorization(transaction_id: str, store: LoginStore) -> RedirectResponse:
    record = await store.get("token_authorization_txn", transaction_id)
    if record is None:
        raise HTTPException(status_code=400, detail="Login transaction was not found")
    return RedirectResponse(
        _authorization_redirect_url(
            client=_broker_client(_broker_client_name(record.payload["oidc_client"])),
            state=transaction_id,
            verifier=record.payload["verifier"],
        ),
        status_code=302,
    )


@router.post("/v2/token")
async def broker_token(body: BrokerTokenRequest, credential_store: CredentialStore) -> dict[str, str | int]:
    if body.grant_type == "authorization_code":
        pending_record = await credential_store.get_active("token_authorization_code", body.code)
        if pending_record is None:
            raise HTTPException(status_code=400, detail="Login code was not found")
        code_challenge = pending_record.public_metadata.get("code_challenge")
        if not isinstance(code_challenge, str) or not secrets.compare_digest(
            pkce_challenge(body.code_verifier), code_challenge
        ):
            raise HTTPException(status_code=400, detail="Code verifier did not match")
        record = await credential_store.consume_active("token_authorization_code", body.code)
        if record is None:
            raise HTTPException(status_code=400, detail="Login code was not found")
        client = _credential_client(record)
        tokens = _decrypted_tokens(record)
        provider_refresh = tokens.get("refresh_token")
        refresh_handle: str | None = None
        if isinstance(provider_refresh, str):
            refresh_handle = secrets.token_urlsafe(32)
            await credential_store.create(
                credential_type="broker_refresh",
                handle=refresh_handle,
                owner_account_id=record.owner_account_id,
                subject_account_id=record.subject_account_id,
                account_identity_id=record.account_identity_id,
                expires_at=datetime.now(timezone.utc) + timedelta(seconds=_BROKER_REFRESH_TTL_SECONDS),
                public_metadata=record.public_metadata,
                encrypted_payload=_encrypted_tokens(tokens),
            )
        return _token_response(tokens, refresh_handle=refresh_handle, bearer_token_source=client.bearer_token_source)

    record = await credential_store.get_active("broker_refresh", body.refresh_token)
    if record is None:
        raise HTTPException(status_code=400, detail="Refresh token was not found")
    client = _credential_client(record)
    stored = _decrypted_tokens(record)
    provider_refresh = stored.get("refresh_token")
    if not isinstance(provider_refresh, str):
        raise HTTPException(status_code=400, detail="Refresh token was not found")
    refreshed = await _post_token(
        client,
        {"grant_type": "refresh_token", "refresh_token": provider_refresh},
    )
    if not isinstance(refreshed.get("refresh_token"), str):
        refreshed["refresh_token"] = provider_refresh
    refresh_handle = secrets.token_urlsafe(32)
    rotated = await credential_store.rotate_active(
        record,
        new_handle=refresh_handle,
        expires_at=datetime.now(timezone.utc) + timedelta(seconds=_BROKER_REFRESH_TTL_SECONDS),
        public_metadata=record.public_metadata,
        encrypted_payload=_encrypted_tokens(refreshed),
    )
    if rotated is None:
        raise HTTPException(status_code=400, detail="Refresh token was not found")
    return _token_response(refreshed, refresh_handle=refresh_handle, bearer_token_source=client.bearer_token_source)


def _token_response(
    payload: dict[str, object],
    *,
    refresh_handle: str | None,
    bearer_token_source: BearerTokenSource,
) -> dict[str, str | int]:
    bearer_token = payload.get(bearer_token_source)
    if not isinstance(bearer_token, str):
        raise HTTPException(
            status_code=502,
            detail=f"Identity provider token response did not include {bearer_token_source}",
        )
    expires_in = payload.get("expires_in")
    response: dict[str, str | int] = {
        "access_token": bearer_token,
        "token_type": "Bearer",
        "expires_in": expires_in if isinstance(expires_in, int) else _DEFAULT_TOKEN_EXPIRES_IN_SECONDS,
    }
    if refresh_handle is not None:
        response["refresh_token"] = refresh_handle
    return response
