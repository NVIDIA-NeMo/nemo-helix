# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Caller account resolution and preference defaults for the /me APIs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException, status
from nemo_helix_plugin.auth.profiles.types import AuthMethod, CallerKind, PreferenceDefaults
from nhx.common.auth import AuthClient
from nhx.core.auth.app.account_resolution import AccountResolutionError, AccountResolver
from nhx.core.auth.config import AuthServiceConfig
from nhx.core.entities.app.repository import (
    AccountIdentityStore,
    AccountPreferenceStore,
    PreferenceRecord,
    merge_patch,
)
from nhx.core.entities.app.repository.sqlalchemy.models import DBWorkspace
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

# Single-user installs with auth disabled and no principal headers share one stable account.
LOCAL_ISSUER = "nemo:local"
LOCAL_SUBJECT = "local-user"
LOCAL_DISPLAY_NAME = "Local user"
# Admin-managed preference defaults are stored as preferences of this system account.
DEFAULTS_ISSUER = "nemo:system"
DEFAULTS_SUBJECT = "preference-defaults"


@dataclass(frozen=True)
class Caller:
    account_id: str
    principal_id: str
    email: str | None
    groups: list[str]
    caller_kind: CallerKind
    auth_method: AuthMethod
    delegated: bool

    @property
    def can_edit(self) -> bool:
        """Delegated agents may read the user's profile and preferences but never change them."""
        return not self.delegated and self.caller_kind == "principal"


def auth_method(auth_client: AuthClient) -> AuthMethod:
    if not auth_client.auth_enabled:
        return "disabled"
    resolved = auth_client.resolved_bearer_token
    if resolved is None:
        return "trusted_header"
    if resolved.token_kind == "access_key":
        return "access_key"
    if resolved.token_kind == "oidc_access_token":
        return "oidc_bearer"
    # The gateway turns a browser session cookie into a short-lived workload token bound to the session's account.
    if resolved.token_kind == "workload_access_token" and "nhx_actor_account_id" in resolved.claims.raw_claims:
        return "web_session"
    return "workload_token"


async def resolve_caller(
    auth_client: AuthClient,
    config: AuthServiceConfig,
    session_maker: async_sessionmaker[AsyncSession],
) -> Caller:
    """Resolve the effective principal (the on-behalf-of user when delegated) to its stable account."""
    principal = auth_client.principal
    effective = principal.effective_principal
    account_id = effective.account_id
    if account_id is None:
        resolution: Mapping[str, Any] = auth_client.identity_resolution or {}
        descriptor = resolution.get("on_behalf_of" if principal.is_delegated else "principal")
        try:
            if isinstance(descriptor, Mapping):
                account_id, _, _ = await AccountResolver(config).resolve_descriptor(descriptor)
            elif not auth_client.auth_enabled and not principal.id:
                account_id = await _local_account_id(session_maker)
        except AccountResolutionError as exc:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Caller has no usable account") from exc
    if account_id is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    return Caller(
        account_id=account_id,
        principal_id=effective.id,
        email=effective.email,
        groups=effective.groups,
        caller_kind="service_principal" if effective.is_service_identity() else "principal",
        auth_method=auth_method(auth_client),
        delegated=principal.is_delegated,
    )


def require_editable(caller: Caller) -> None:
    if caller.caller_kind != "principal":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Service accounts have no profile")
    if caller.delegated:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Delegated callers cannot change the user's profile"
        )


async def require_readable_workspace(
    auth_client: AuthClient, session_maker: async_sessionmaker[AsyncSession], workspace: str
) -> None:
    """404 for missing and inaccessible workspaces alike, so names cannot be probed."""
    async with session_maker() as session:
        exists = await session.scalar(
            select(DBWorkspace.name).where(DBWorkspace.name == workspace, DBWorkspace.deletion_stage.is_(None))
        )
    if exists is None or not await auth_client.has_permissions(workspace, ["workspaces.read"]):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workspace not found")


async def _system_account_id(
    session_maker: async_sessionmaker[AsyncSession], *, issuer: str, subject: str, **kwargs: Any
) -> str:
    record = await AccountIdentityStore(session_maker).resolve_or_materialize(
        issuer=issuer, subject=subject, subject_claim="system", linked_via="system", **kwargs
    )
    return record.account_id


async def _local_account_id(session_maker: async_sessionmaker[AsyncSession]) -> str:
    return await _system_account_id(
        session_maker,
        issuer=LOCAL_ISSUER,
        subject=LOCAL_SUBJECT,
        account_type="user",
        display_name=LOCAL_DISPLAY_NAME,
    )


class PreferenceDefaultsStore:
    """Config-managed defaults (read-only) layered over admin-managed defaults stored in the database."""

    def __init__(self, config: AuthServiceConfig, session_maker: async_sessionmaker[AsyncSession]) -> None:
        self.config = config
        self.session_maker = session_maker
        self.preferences = AccountPreferenceStore(session_maker)

    def is_config_managed(self, namespace: str) -> bool:
        return namespace in self.config.preference_defaults

    async def account_id(self) -> str:
        return await _system_account_id(
            self.session_maker, issuer=DEFAULTS_ISSUER, subject=DEFAULTS_SUBJECT, account_type="service"
        )

    async def get(self, namespace: str) -> PreferenceDefaults | None:
        if self.is_config_managed(namespace):
            return PreferenceDefaults(
                namespace=namespace, value=self.config.preference_defaults[namespace], managed_by="config", db_version=0
            )
        record = await self.preferences.get(await self.account_id(), namespace, None)
        return _api_defaults(record) if record else None

    async def list_defaults(self, namespace_prefix: str | None = None) -> list[PreferenceDefaults]:
        stored = await self.preferences.list_documents(await self.account_id(), namespace_prefix=namespace_prefix)
        by_namespace = {record.namespace: _api_defaults(record) for record in stored}
        for namespace, value in self.config.preference_defaults.items():
            if (
                namespace_prefix is None
                or namespace == namespace_prefix
                or namespace.startswith(f"{namespace_prefix}.")
            ):
                by_namespace[namespace] = PreferenceDefaults(
                    namespace=namespace, value=value, managed_by="config", db_version=0
                )
        return [by_namespace[namespace] for namespace in sorted(by_namespace)]


def _api_defaults(record: PreferenceRecord) -> PreferenceDefaults:
    return PreferenceDefaults(
        namespace=record.namespace, value=record.value, managed_by="api", db_version=record.db_version
    )


def effective_value(defaults: dict[str, Any], value: dict[str, Any]) -> dict[str, Any]:
    """User values win key by key; a key the user removed falls back to the default."""
    return merge_patch(defaults, value)
