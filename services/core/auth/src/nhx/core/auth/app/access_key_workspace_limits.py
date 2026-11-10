# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Workspace limits of Scoped Access Keys, resolved for a PDP input.

Services forward the key ID (``access_key_jti`` for the caller, ``subject_access_key_jti`` for the
on-behalf-of principal). The policy gets ``access_key_workspaces``: the names of the active workspaces
the request may touch, or ``null`` when no key limit applies.

    caller key limited to [team-a, team-b], no delegation  -> ["team-a", "team-b"]
    service acting for a key limited to [team-a]           -> ["team-a"]
    key's only workspace deleted, or key revoked           -> []   (denies everything)
    service account ever issued a limited key, no key ID   -> []   (delegated work, or the ID was lost)
    same account with one of its unlimited keys            -> []

A service account ever issued a limited key stays limited, because delegated work keeps no key ID.
``wait_role`` sets ``role_binding_probe`` to ask about role bindings alone.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from nhx.common.auth.principal_identifier import (
    SERVICE_ACCOUNT_PRINCIPAL_PREFIX,
    InvalidPrincipalIdentifier,
    parse_principal_identifier,
)
from nhx.core.auth.app.account_resolution import get_account_session_maker
from nhx.core.entities.app.repository import AccessKeyWorkspaceStore, AccountCredentialStore
from nhx.core.entities.app.repository.access_key_workspace import ACCESS_KEY_CREDENTIAL_TYPE
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

ACCESS_KEY_WORKSPACES_FIELD = "access_key_workspaces"
# Set by AuthClient.wait_role, which polls role bindings for a principal rather than authorizing a request.
ROLE_BINDING_PROBE_FIELD = "role_binding_probe"
# (principal field, key ID field) for the caller and for the on-behalf-of principal.
_IDENTITY_FIELDS = (
    ("principal_id", "access_key_jti"),
    ("on_behalf_of_principal_id", "subject_access_key_jti"),
)


class AccessKeyWorkspaceLimitUnavailableError(RuntimeError):
    """Raised when a key's workspace limit can't be read, so the request must fail closed."""


def _string_field(auth_input: Mapping[str, Any], key: str) -> str | None:
    value = auth_input.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _identities(auth_input: Mapping[str, Any]) -> list[tuple[str, str | None, bool]]:
    """(principal ID, key ID, is service account) for identities that may carry a key limit."""
    identities: list[tuple[str, str | None, bool]] = []
    for principal_field, jti_field in _IDENTITY_FIELDS:
        principal_id = _string_field(auth_input, principal_field)
        if principal_id is None:
            continue
        try:
            parsed = parse_principal_identifier(principal_id, validate=False)
        except InvalidPrincipalIdentifier:
            parsed = None
        # Keys can't be issued to internal services, so a service never carries a key limit.
        if parsed is not None and parsed.is_service_principal():
            continue
        is_service_account = parsed is not None and parsed.is_service_account()
        jti = _string_field(auth_input, jti_field)
        if jti is None and not is_service_account:
            continue
        # has_role sends the on-behalf-of principal and key ID in both slots; look them up once.
        if (principal_id, jti, is_service_account) not in identities:
            identities.append((principal_id, jti, is_service_account))
    return identities


def _is_role_binding_probe(entrypoint: str, auth_input: Mapping[str, Any]) -> bool:
    # Only internal services can call the PDP, so an external caller can't set the flag. It is honoured only for a
    # role check that carries no key ID, which is all wait_role sends.
    return (
        entrypoint == "has_role"
        and auth_input.get(ROLE_BINDING_PROBE_FIELD) is True
        and not any(_string_field(auth_input, jti_field) for _, jti_field in _IDENTITY_FIELDS)
    )


async def resolve_access_key_workspaces(entrypoint: str, auth_input: Mapping[str, Any]) -> list[str] | None:
    """Return the workspace names the request is limited to, or None when no key limit applies."""
    if _is_role_binding_probe(entrypoint, auth_input):
        return None
    identities = _identities(auth_input)
    if not identities:
        return None
    try:
        session_maker = await get_account_session_maker()
        workspace_store = AccessKeyWorkspaceStore(session_maker)
        credential_store = AccountCredentialStore(session_maker)
        limits: list[set[str]] = []
        for principal_id, jti, is_service_account in identities:
            limit = await _identity_limit(principal_id, jti, is_service_account, workspace_store, credential_store)
            if limit is not None:
                limits.append(limit)
    except (SQLAlchemyError, RuntimeError) as exc:
        logger.error("Could not read access key workspace limits: %s", exc)
        raise AccessKeyWorkspaceLimitUnavailableError("Access key workspace limits are unavailable") from exc

    if not limits:
        return None
    return sorted(set.intersection(*limits))


async def _identity_limit(
    principal_id: str,
    jti: str | None,
    is_service_account: bool,
    workspace_store: AccessKeyWorkspaceStore,
    credential_store: AccountCredentialStore,
) -> set[str] | None:
    if jti is None or not (workspace_ids := await workspace_store.workspace_ids_for_key(jti)):
        # No key ID (only service accounts get here, see _identities) or an unlimited one: neither lifts a limited
        # service account's denial, whoever owns that key.
        return await _limited_account_limit(principal_id, workspace_store) if is_service_account else None
    # Only service-account keys can be limited (see access_key_workspace_ids); anything else is a mismatch.
    if not is_service_account:
        return set()
    credential = await credential_store.get_active(ACCESS_KEY_CREDENTIAL_TYPE, jti)
    if credential is None:
        return set()
    metadata = credential.public_metadata
    key_principal = metadata.get("subject_principal") or metadata.get("owner_principal")
    if key_principal != principal_id:
        return set()
    return set(await workspace_store.active_workspace_names(workspace_ids))


async def _limited_account_limit(principal_id: str, workspace_store: AccessKeyWorkspaceStore) -> set[str] | None:
    """Deny everything for a service account that was ever issued a limited key; no limit otherwise."""
    if await workspace_store.service_account_is_limited(principal_id.removeprefix(SERVICE_ACCOUNT_PRINCIPAL_PREFIX)):
        return set()
    return None
