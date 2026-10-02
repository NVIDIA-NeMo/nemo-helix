# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Confidential OIDC login broker.

The auth service is the OAuth client. Studio receives an HTTP-only session
cookie. The CLI receives a provider access token and an opaque NeMo refresh
handle. The client secret and provider refresh token never leave this service.
"""

from __future__ import annotations

import json
import secrets
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any
from urllib.parse import urlencode, urlparse

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import RedirectResponse
from nhx.common.auth.token_resolver import resolve_bearer_token
from nhx.common.config import get_auth_config
from nhx.core.auth.app.account_resolution import get_account_session_maker
from nhx.core.auth.oidc_broker.crypto import (
    client_secret_basic_header,
    decrypt_secret,
    encrypt_secret,
    pkce_challenge,
    read_env,
)
from nhx.core.auth.oidc_broker.store import OidcLoginStore
from nhx.core.entities.app.repository import (
    AccountCredentialRecord,
    AccountCredentialStore,
    AccountIdentityStore,
)
from nhx.core.entities.app.repository.account_identity import AccountIdentityRecord
from pydantic import BaseModel

router = APIRouter(tags=["Authentication"])

SESSION_COOKIE = "nhx_session"
CSRF_HEADER = "x-nhx-requested-by"
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
_LOGIN_TTL_SECONDS = 600
_CLI_CODE_TTL_SECONDS = 120
_SESSION_TTL_SECONDS = 60 * 60 * 8
_CLI_REFRESH_TTL_SECONDS = 60 * 60 * 24 * 30


class CliLoginStart(BaseModel):
    redirect_uri: str


class CliTokenRequest(BaseModel):
    grant_type: str
    code: str | None = None
    refresh_token: str | None = None


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
        raise HTTPException(status_code=503, detail="Confidential OIDC login is not available") from exc
    return OidcLoginStore(session_maker, key)


async def get_credential_store() -> AccountCredentialStore:
    try:
        return AccountCredentialStore(await get_account_session_maker())
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="Confidential OIDC login is not available") from exc


async def get_identity_store() -> AccountIdentityStore:
    try:
        return AccountIdentityStore(await get_account_session_maker())
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="Confidential OIDC login is not available") from exc


LoginStore = Annotated[OidcLoginStore, Depends(get_login_store)]
CredentialStore = Annotated[AccountCredentialStore, Depends(get_credential_store)]
IdentityStore = Annotated[AccountIdentityStore, Depends(get_identity_store)]


def _require_confidential() -> None:
    oidc = get_auth_config().oidc
    if oidc.token_endpoint_auth_method != "client_secret_basic":
        raise HTTPException(status_code=404, detail="Confidential OIDC login is not configured")
    if not oidc.client_secret_env_var or not oidc.token_endpoint or not oidc.authorization_endpoint:
        raise HTTPException(status_code=503, detail="Confidential OIDC login is not available")


def _required_env(name: str | None) -> str:
    if not name:
        raise HTTPException(status_code=503, detail="Confidential OIDC login is not available")
    try:
        return read_env(name)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail="Confidential OIDC login is not available") from exc


def _client_secret() -> str:
    return _required_env(get_auth_config().oidc.client_secret_env_var)


def _session_encryption_key() -> str:
    return _required_env(get_auth_config().oidc.session_encryption_key_env_var)


def _redirect_uri(request: Request) -> str:
    configured = get_auth_config().oidc.login_redirect_uri
    if configured:
        return configured
    return str(request.base_url).rstrip("/") + "/apis/auth/v2/login/callback"


def _return_path(value: str) -> str:
    parsed = urlparse(value)
    if not value.startswith("/") or value.startswith("//") or "\\" in value or parsed.scheme or parsed.netloc:
        raise HTTPException(status_code=400, detail="return_to must be a relative path")
    return value


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
        linked_via="confidential_oidc_login",
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


async def _finish_cli_login(
    payload: dict[str, str],
    code: str,
    credential_store: AccountCredentialStore,
    identity_store: AccountIdentityStore,
) -> RedirectResponse:
    tokens = await _exchange_code(code, payload["verifier"], payload["idp_redirect_uri"])
    identity, metadata = await _resolve_identity(tokens, identity_store)
    one_time = secrets.token_urlsafe(32)
    await credential_store.create(
        credential_type="cli_login_code",
        handle=one_time,
        owner_account_id=identity.account_id,
        account_identity_id=identity.identity_id,
        expires_at=datetime.now(timezone.utc) + timedelta(seconds=_CLI_CODE_TTL_SECONDS),
        public_metadata=metadata,
        encrypted_payload=_encrypted_tokens(tokens),
    )
    separator = "&" if "?" in payload["redirect_uri"] else "?"
    return RedirectResponse(f"{payload['redirect_uri']}{separator}code={one_time}", status_code=302)


async def _exchange_code(code: str, verifier: str, redirect_uri: str) -> dict[str, object]:
    return await _post_token(
        {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "code_verifier": verifier,
        }
    )


async def _post_token(form: dict[str, str]) -> dict[str, object]:
    oidc = get_auth_config().oidc
    if not oidc.token_endpoint:
        raise HTTPException(status_code=503, detail="Confidential OIDC login is not available")
    headers = {"Authorization": client_secret_basic_header(oidc.client_id, _client_secret())}
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(oidc.token_endpoint, data=form, headers=headers)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="Identity provider token request failed") from exc
    if response.status_code != 200:
        raise HTTPException(status_code=502, detail="Identity provider token request failed")
    try:
        payload = response.json()
    except ValueError as exc:
        raise HTTPException(status_code=502, detail="Identity provider token response was invalid") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("access_token"), str):
        raise HTTPException(status_code=502, detail="Identity provider token response was incomplete")
    return payload


def _loopback_redirect(redirect_uri: str) -> None:
    parsed = urlparse(redirect_uri)
    try:
        port = parsed.port
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="CLI redirect_uri must be an http loopback address") from exc
    if (
        parsed.scheme != "http"
        or parsed.hostname not in _LOOPBACK_HOSTS
        or port is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise HTTPException(status_code=400, detail="CLI redirect_uri must be an http loopback address")


@router.get("/v2/login")
async def start_login(request: Request, store: LoginStore, return_to: str = "/") -> RedirectResponse:
    _require_confidential()
    verifier = secrets.token_urlsafe(48)
    state = secrets.token_urlsafe(24)
    await store.put(
        "web_txn",
        state,
        {"verifier": verifier, "return_to": _return_path(return_to), "redirect_uri": _redirect_uri(request)},
        ttl_seconds=_LOGIN_TTL_SECONDS,
    )
    query = urlencode(
        {
            "response_type": "code",
            "client_id": get_auth_config().oidc.client_id,
            "redirect_uri": _redirect_uri(request),
            "state": state,
            "code_challenge": pkce_challenge(verifier),
            "code_challenge_method": "S256",
            "scope": get_auth_config().oidc.default_scopes,
        }
    )
    return RedirectResponse(f"{get_auth_config().oidc.authorization_endpoint}?{query}", status_code=302)


@router.get("/v2/login/callback")
async def login_callback(
    request: Request,
    code: str,
    state: str,
    store: LoginStore,
    credential_store: CredentialStore,
    identity_store: IdentityStore,
) -> RedirectResponse:
    _require_confidential()
    cli_record = await store.pop("cli_txn", state)
    if cli_record is not None:
        return await _finish_cli_login(cli_record.payload, code, credential_store, identity_store)
    record = await store.pop("web_txn", state)
    if record is None:
        raise HTTPException(status_code=400, detail="Login transaction was not found")

    tokens = await _exchange_code(code, record.payload["verifier"], record.payload["redirect_uri"])
    identity, metadata = await _resolve_identity(tokens, identity_store)
    session_id = secrets.token_urlsafe(32)
    await credential_store.create(
        credential_type="web_session",
        handle=session_id,
        owner_account_id=identity.account_id,
        account_identity_id=identity.identity_id,
        expires_at=datetime.now(timezone.utc) + timedelta(seconds=_SESSION_TTL_SECONDS),
        public_metadata=metadata,
        encrypted_payload=_encrypted_tokens(tokens),
    )
    response = RedirectResponse(record.payload.get("return_to") or "/", status_code=302)
    response.set_cookie(
        SESSION_COOKIE,
        session_id,
        max_age=_SESSION_TTL_SECONDS,
        httponly=True,
        secure=True,
        samesite="lax",
        path="/",
    )
    return response


@router.post("/v2/logout", status_code=204)
async def logout(request: Request, credential_store: CredentialStore) -> Response:
    if request.headers.get(CSRF_HEADER) != "1":
        raise HTTPException(status_code=403, detail="Missing X-NHX-Requested-By")
    session_id = request.cookies.get(SESSION_COOKIE)
    if session_id:
        await credential_store.revoke("web_session", session_id)
    response = Response(status_code=204)
    response.delete_cookie(SESSION_COOKIE, path="/", secure=True, httponly=True, samesite="lax")
    return response


@router.get("/v2/session")
async def session(request: Request, credential_store: CredentialStore) -> WebSessionResponse:
    session_id = request.cookies.get(SESSION_COOKIE)
    if not session_id:
        raise HTTPException(status_code=401, detail="Missing session")
    record = await credential_store.get_active("web_session", session_id)
    if record is None:
        raise HTTPException(status_code=401, detail="Missing session")
    principal_id = record.public_metadata.get("principal_id")
    email = record.public_metadata.get("email")
    groups = record.public_metadata.get("groups")
    aliases = record.public_metadata.get("authz_aliases")
    if (
        not isinstance(principal_id, str)
        or not principal_id
        or (email is not None and not isinstance(email, str))
        or not isinstance(groups, list)
        or not all(isinstance(group, str) for group in groups)
        or not isinstance(aliases, list)
        or not all(isinstance(alias, str) for alias in aliases)
    ):
        raise HTTPException(status_code=401, detail="Missing session")
    return WebSessionResponse(
        id=principal_id,
        email=email,
        groups=groups,
        account_id=record.subject_account_id,
        authz_aliases=aliases,
    )


@router.post("/v2/cli/login")
async def start_cli_login(body: CliLoginStart, request: Request, store: LoginStore) -> dict[str, str]:
    _require_confidential()
    _loopback_redirect(body.redirect_uri)
    verifier = secrets.token_urlsafe(48)
    transaction_id = secrets.token_urlsafe(24)
    await store.put(
        "cli_txn",
        transaction_id,
        {
            "verifier": verifier,
            "redirect_uri": body.redirect_uri,
            "idp_redirect_uri": _redirect_uri(request),
        },
        ttl_seconds=_LOGIN_TTL_SECONDS,
    )
    login_url = str(request.base_url).rstrip("/") + f"/apis/auth/v2/cli/login/{transaction_id}"
    return {"login_url": login_url, "transaction_id": transaction_id}


@router.get("/v2/cli/login/{transaction_id}")
async def continue_cli_login(transaction_id: str, store: LoginStore) -> RedirectResponse:
    _require_confidential()
    record = await store.get("cli_txn", transaction_id)
    if record is None:
        raise HTTPException(status_code=400, detail="Login transaction was not found")
    query = urlencode(
        {
            "response_type": "code",
            "client_id": get_auth_config().oidc.client_id,
            "redirect_uri": record.payload["idp_redirect_uri"],
            "state": transaction_id,
            "code_challenge": pkce_challenge(record.payload["verifier"]),
            "code_challenge_method": "S256",
            "scope": get_auth_config().oidc.default_scopes,
        }
    )
    return RedirectResponse(f"{get_auth_config().oidc.authorization_endpoint}?{query}", status_code=302)


@router.post("/v2/cli/token")
async def cli_token(body: CliTokenRequest, credential_store: CredentialStore) -> dict[str, str | int]:
    _require_confidential()
    if body.grant_type == "authorization_code":
        if not body.code:
            raise HTTPException(status_code=400, detail="code is required")
        record = await credential_store.consume_active("cli_login_code", body.code)
        if record is None:
            raise HTTPException(status_code=400, detail="Login code was not found")
        tokens = _decrypted_tokens(record)
        provider_refresh = tokens.get("refresh_token")
        refresh_handle: str | None = None
        if isinstance(provider_refresh, str):
            refresh_handle = secrets.token_urlsafe(32)
            await credential_store.create(
                credential_type="cli_refresh",
                handle=refresh_handle,
                owner_account_id=record.owner_account_id,
                subject_account_id=record.subject_account_id,
                account_identity_id=record.account_identity_id,
                expires_at=datetime.now(timezone.utc) + timedelta(seconds=_CLI_REFRESH_TTL_SECONDS),
                public_metadata=record.public_metadata,
                encrypted_payload=_encrypted_tokens(tokens),
            )
        return _token_response(tokens, refresh_handle=refresh_handle)

    if body.grant_type == "refresh_token":
        if not body.refresh_token:
            raise HTTPException(status_code=400, detail="refresh_token is required")
        record = await credential_store.get_active("cli_refresh", body.refresh_token)
        if record is None:
            raise HTTPException(status_code=400, detail="Refresh token was not found")
        stored = _decrypted_tokens(record)
        provider_refresh = stored.get("refresh_token")
        if not isinstance(provider_refresh, str):
            raise HTTPException(status_code=400, detail="Refresh token was not found")
        refreshed = await _post_token({"grant_type": "refresh_token", "refresh_token": provider_refresh})
        if not isinstance(refreshed.get("refresh_token"), str):
            refreshed["refresh_token"] = provider_refresh
        await credential_store.update(
            replace(
                record,
                encrypted_payload=_encrypted_tokens(refreshed),
                last_used_at=datetime.now(timezone.utc),
            )
        )
        return _token_response(refreshed, refresh_handle=body.refresh_token)
    raise HTTPException(status_code=400, detail="Unsupported grant_type")


def _token_response(payload: dict[str, object], *, refresh_handle: str | None) -> dict[str, str | int]:
    access_token = payload.get("access_token")
    if not isinstance(access_token, str):
        raise HTTPException(status_code=502, detail="Identity provider token response was incomplete")
    expires_in = payload.get("expires_in")
    response: dict[str, str | int] = {
        "access_token": access_token,
        "token_type": "Bearer",
        "expires_in": expires_in if isinstance(expires_in, int) else 3600,
    }
    if refresh_handle is not None:
        response["refresh_token"] = refresh_handle
    return response
