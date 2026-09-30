# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Server-side OIDC authorization-code flow for confidential Studio clients.

This is a deliberately small BFF-style spike.  The browser never receives the
client secret; Studio redirects to these endpoints, the backend exchanges the
code using ``client_secret_basic``, then the SPA asks the backend for the current
bearer token.  A production version should replace the in-memory session store
with shared encrypted/session storage before enabling multi-replica Studio.
"""

from __future__ import annotations

import base64
import hashlib
import os
import secrets
import time
from typing import Any
from urllib.parse import quote, urlencode

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse, RedirectResponse, Response
from nmp.common.http_clients import shared_async_http_client
from nmp.studio.config import StudioConfig
from pydantic import BaseModel

STATE_COOKIE = "nhx_studio_oidc_state"
VERIFIER_COOKIE = "nhx_studio_oidc_verifier"
SESSION_COOKIE = "nhx_studio_oidc_session"
DEFAULT_SCOPE = "openid profile email"
SESSION_COOKIE_MAX_AGE_SECONDS = 60 * 60 * 8
COOKIE_PATH = "/studio"


class OidcSession(BaseModel):
    """Tokens for one browser session."""

    access_token: str | None = None
    id_token: str | None = None
    refresh_token: str | None = None
    expires_at: int | None = None
    token_type: str = "Bearer"


_SESSIONS: dict[str, OidcSession] = {}


def build_confidential_oidc_router(config: StudioConfig) -> APIRouter:
    """Build routes for the confidential-client Studio OIDC flow."""
    router = APIRouter(include_in_schema=False)

    @router.get("/studio/auth/confidential/login")
    async def login(request: Request) -> RedirectResponse:
        settings = _settings(config)
        state = secrets.token_urlsafe(32)
        verifier = secrets.token_urlsafe(64)
        params = {
            "response_type": "code",
            "client_id": settings.client_id,
            "redirect_uri": _redirect_uri(request, config),
            "scope": settings.scope,
            "state": state,
            "code_challenge": _code_challenge(verifier),
            "code_challenge_method": "S256",
        }
        response = RedirectResponse(f"{settings.authorization_endpoint}?{urlencode(params)}")
        _set_studio_cookie(response, STATE_COOKIE, state, max_age=300)
        _set_studio_cookie(response, VERIFIER_COOKIE, verifier, max_age=300)
        return response

    @router.get("/studio/auth/confidential/callback")
    async def callback(request: Request, code: str | None = None, state: str | None = None) -> RedirectResponse:
        if not code or not state:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing OIDC callback parameters")
        if state != request.cookies.get(STATE_COOKIE):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid OIDC state")
        verifier = request.cookies.get(VERIFIER_COOKIE)
        if not verifier:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing OIDC code verifier")

        token_response = await _exchange_code(
            settings=_settings(config),
            code=code,
            code_verifier=verifier,
            redirect_uri=_redirect_uri(request, config),
        )
        session_id = secrets.token_urlsafe(32)
        _SESSIONS[session_id] = _session_from_token_response(token_response)

        response = RedirectResponse(_studio_root(request, config))
        response.delete_cookie(STATE_COOKIE, path=COOKIE_PATH)
        response.delete_cookie(VERIFIER_COOKIE, path=COOKIE_PATH)
        _set_studio_cookie(
            response,
            session_id_cookie_name(config),
            session_id,
            max_age=SESSION_COOKIE_MAX_AGE_SECONDS,
        )
        return response

    @router.get("/studio/auth/confidential/token")
    async def token(request: Request) -> JSONResponse:
        session = _session_from_request(request, config)
        bearer = _select_bearer_token(session, config)
        if not bearer:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="No Studio OIDC session")
        return JSONResponse(
            {
                "access_token": bearer,
                "token_type": session.token_type,
                "expires_at": session.expires_at,
            },
            headers={"Cache-Control": "no-store"},
        )

    @router.post("/studio/auth/confidential/logout")
    async def logout(request: Request) -> Response:
        session_id = request.cookies.get(session_id_cookie_name(config))
        if session_id:
            _SESSIONS.pop(session_id, None)
        response = Response(status_code=status.HTTP_204_NO_CONTENT)
        response.delete_cookie(session_id_cookie_name(config), path=COOKIE_PATH)
        return response

    return router


class _Settings(BaseModel):
    authorization_endpoint: str
    token_endpoint: str
    client_id: str
    client_secret: str
    scope: str


def _settings(config: StudioConfig) -> _Settings:
    secret_env_var = config.confidential_oidc.client_secret_env_var
    client_secret = os.environ.get(secret_env_var, "") if secret_env_var else ""
    values = _Settings(
        authorization_endpoint=(
            config.confidential_oidc.authorization_endpoint
            or config._resolve_config_path("auth.oidc.authorization_endpoint")
            or ""
        ),
        token_endpoint=config.confidential_oidc.token_endpoint
        or config._resolve_config_path("auth.oidc.token_endpoint")
        or "",
        client_id=config.confidential_oidc.client_id or config._resolve_config_path("auth.oidc.client_id") or "",
        client_secret=client_secret,
        scope=config.confidential_oidc.scope
        or config._resolve_config_path("auth.oidc.default_scopes")
        or DEFAULT_SCOPE,
    )
    missing = [name for name, value in values.model_dump().items() if not value]
    if missing:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Studio confidential OIDC is missing configuration: {', '.join(missing)}",
        )
    return values


def _studio_root(request: Request, config: StudioConfig) -> str:
    if config.confidential_oidc.post_login_redirect_path:
        return config.confidential_oidc.post_login_redirect_path
    base = str(request.base_url).rstrip("/")
    return f"{base}/studio/"


def _redirect_uri(request: Request, config: StudioConfig) -> str:
    if config.confidential_oidc.redirect_uri:
        return config.confidential_oidc.redirect_uri
    base = str(request.base_url).rstrip("/")
    return f"{base}/studio/auth/confidential/callback"


def session_id_cookie_name(config: StudioConfig) -> str:
    return config.confidential_oidc.session_cookie_name or SESSION_COOKIE


def _set_studio_cookie(response: Response, key: str, value: str, *, max_age: int) -> None:
    response.set_cookie(
        key,
        value,
        max_age=max_age,
        httponly=True,
        samesite="lax",
        secure=True,
        path=COOKIE_PATH,
    )


def _code_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _basic_auth(client_id: str, client_secret: str) -> str:
    raw = f"{quote(client_id, safe='')}:{quote(client_secret, safe='')}".encode("utf-8")
    return "Basic " + base64.b64encode(raw).decode("ascii")


async def _exchange_code(
    *,
    settings: _Settings,
    code: str,
    code_verifier: str,
    redirect_uri: str,
) -> dict[str, Any]:
    response = await shared_async_http_client().post(
        settings.token_endpoint,
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "code_verifier": code_verifier,
        },
        headers={
            "Authorization": _basic_auth(settings.client_id, settings.client_secret),
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
    )
    if response.status_code >= 400:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="OIDC token exchange failed")
    data = response.json()
    if not isinstance(data, dict):
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="OIDC token endpoint returned invalid JSON")
    return data


def _session_from_token_response(data: dict[str, Any]) -> OidcSession:
    expires_in = data.get("expires_in")
    expires_at = int(time.time()) + int(expires_in) if isinstance(expires_in, int | float) else None
    token_type = data.get("token_type")
    return OidcSession(
        access_token=data.get("access_token") if isinstance(data.get("access_token"), str) else None,
        id_token=data.get("id_token") if isinstance(data.get("id_token"), str) else None,
        refresh_token=data.get("refresh_token") if isinstance(data.get("refresh_token"), str) else None,
        expires_at=expires_at,
        token_type=token_type if isinstance(token_type, str) else "Bearer",
    )


def _session_from_request(request: Request, config: StudioConfig) -> OidcSession:
    session_id = request.cookies.get(session_id_cookie_name(config))
    if not session_id or session_id not in _SESSIONS:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="No Studio OIDC session")
    return _SESSIONS[session_id]


def _select_bearer_token(session: OidcSession, config: StudioConfig) -> str | None:
    source = config.confidential_oidc.bearer_token_source or config._resolve_config_path(
        "auth.oidc.bearer_token_source"
    )
    if source == "id_token":
        return session.id_token
    return session.access_token
