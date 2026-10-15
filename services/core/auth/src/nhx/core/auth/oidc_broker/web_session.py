# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Browser session resolution for the OIDC broker and gateway auth callout."""

from __future__ import annotations

import time

import jwt
from fastapi import Request
from nhx.common.auth.models import Principal
from nhx.common.config import AuthConfig
from nhx.core.auth.api.v2.workload_token_exchange import (
    DEFAULT_WORKLOAD_AUDIENCE,
    WorkloadTokenExchangeService,
    workload_token_issuer,
)
from nhx.core.entities.app.repository import AccountCredentialRecord, AccountCredentialStore

WEB_SESSION_COOKIE = "nhx_session"
WEB_SESSION_CSRF_HEADER = "x-source"
WEB_SESSION_CSRF_VALUE = "NeMo Studio"
_WEB_SESSION_ACCESS_TOKEN_TTL_SECONDS = 60


def web_sessions_enabled(config: AuthConfig) -> bool:
    public_client = config.oidc.public_client
    return config.oidc.confidential_client is not None or (
        public_client is not None and public_client.server_side_sessions
    )


def _principal_from_record(record: AccountCredentialRecord) -> Principal | None:
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
        return None
    return Principal(
        id=principal_id,
        email=email,
        groups=groups,
        account_id=record.subject_account_id,
        authz_aliases=aliases,
    )


async def resolve_web_session(
    request: Request,
    credential_store: AccountCredentialStore,
) -> Principal | None:
    session_id = request.cookies.get(WEB_SESSION_COOKIE)
    if not session_id:
        return None
    record = await credential_store.get_active("web_session", session_id)
    if record is None:
        return None
    return _principal_from_record(record)


async def mint_web_session_access_token(
    config: AuthConfig,
    workload_token_exchange_service: WorkloadTokenExchangeService,
    principal: Principal,
) -> str:
    workload = config.oidc.workload
    if workload is None:
        raise RuntimeError("Workload token exchange is not configured")

    now = int(time.time())
    claims: dict[str, object] = {
        "iss": workload_token_issuer(config),
        "sub": principal.id,
        "aud": workload.audience or config.oidc.audience or DEFAULT_WORKLOAD_AUDIENCE,
        "iat": now,
        "nbf": now,
        "exp": now + min(workload.token_ttl_seconds, _WEB_SESSION_ACCESS_TOKEN_TTL_SECONDS),
        "nhx_actor_account_id": principal.account_id,
        "nhx_actor_aliases": principal.authz_aliases,
    }
    if principal.email:
        claims["email"] = principal.email
    if principal.groups:
        claims["groups"] = principal.groups

    signing_key = await workload_token_exchange_service.workload_signing_key_async(config)
    return jwt.encode(
        claims,
        signing_key.private_key,
        algorithm="RS256",
        headers={"kid": signing_key.kid},
    )
