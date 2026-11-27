# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Workspace limits of workspace-limited Scoped Access Keys."""

from __future__ import annotations

from collections.abc import Collection
from typing import Final

from nhx.core.entities.app.repository.account_credential import AccountCredentialType, credential_lookup_hash
from nhx.core.entities.app.repository.sqlalchemy.models import (
    DBAccessKeyWorkspace,
    DBAccountCredential,
    DBAccountIdentity,
    DBWorkspace,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

ACCESS_KEY_CREDENTIAL_TYPE: Final[AccountCredentialType] = "access_key"
SERVICE_ACCOUNT_IDENTITY_ISSUER = "nemo:service-account"


class AccessKeyWorkspaceStore:
    """Read the ``access_key_workspaces`` rows that limit a key to a set of workspace IDs."""

    def __init__(self, session_maker: async_sessionmaker[AsyncSession]) -> None:
        self.session_maker = session_maker

    async def workspace_ids_for_key(self, jti: str) -> list[str]:
        """The workspace IDs a key is limited to; empty when the key has no limit."""
        async with self.session_maker() as session:
            rows = await session.scalars(
                select(DBAccessKeyWorkspace.workspace_id)
                .join(DBAccountCredential, DBAccountCredential.id == DBAccessKeyWorkspace.credential_id)
                .where(
                    DBAccountCredential.credential_type == ACCESS_KEY_CREDENTIAL_TYPE,
                    DBAccountCredential.lookup_hash == credential_lookup_hash(jti),
                )
                .order_by(DBAccessKeyWorkspace.workspace_id)
            )
            return list(rows)

    async def service_account_is_limited(self, service_account_id: str) -> bool:
        """Whether a service account was ever issued a workspace-limited key.

        Revoked and expired keys count too: work a limited key delegated (jobs, deployments) carries no key ID,
        so it must stay denied after the key itself is gone.
        """
        async with self.session_maker() as session:
            found = await session.scalar(
                select(DBAccessKeyWorkspace.credential_id)
                .join(DBAccountCredential, DBAccountCredential.id == DBAccessKeyWorkspace.credential_id)
                .join(DBAccountIdentity, DBAccountIdentity.account_id == DBAccountCredential.subject_account_id)
                .where(
                    DBAccountIdentity.issuer == SERVICE_ACCOUNT_IDENTITY_ISSUER,
                    DBAccountIdentity.subject == service_account_id,
                    DBAccountCredential.credential_type == ACCESS_KEY_CREDENTIAL_TYPE,
                )
                .limit(1)
            )
            return found is not None

    async def active_workspace_names(self, workspace_ids: Collection[str]) -> list[str]:
        """Current names of the given workspaces, skipping deleted ones and ones marked for deletion."""
        if not workspace_ids:
            return []
        async with self.session_maker() as session:
            rows = await session.scalars(
                select(DBWorkspace.name)
                .where(DBWorkspace.id.in_(list(workspace_ids)), DBWorkspace.deletion_stage.is_(None))
                .order_by(DBWorkspace.name)
            )
            return list(rows)
