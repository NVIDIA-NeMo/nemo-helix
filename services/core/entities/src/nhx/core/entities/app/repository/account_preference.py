# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-account preference documents: one opaque JSON object per (account, namespace, workspace)."""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from nhx.core.entities.app.repository.sqlalchemy.models import DBAccountPreference
from nhx.core.entities.utils.identifiers import generate_entity_id
from sqlalchemy import func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm.exc import StaleDataError

PREFERENCE_MAX_BYTES = 64 * 1024
PREFERENCE_MAX_DOCUMENTS = 200
_WRITE_ATTEMPTS = 5


class PreferenceTooLargeError(ValueError):
    """The serialized document exceeds PREFERENCE_MAX_BYTES."""


class PreferenceLimitError(ValueError):
    """The account already holds PREFERENCE_MAX_DOCUMENTS documents."""


class PreferenceVersionConflictError(RuntimeError):
    """If-Match did not match the stored version, or concurrent writers kept colliding."""


@dataclass(frozen=True)
class PreferenceRecord:
    namespace: str
    workspace: str | None
    value: dict[str, Any]
    db_version: int
    updated_at: datetime | None


def merge_patch(target: Any, patch: Any) -> Any:
    """Apply an RFC 7396 JSON merge patch."""
    if not isinstance(patch, dict):
        return patch
    result = dict(target) if isinstance(target, dict) else {}
    for key, value in patch.items():
        if value is None:
            result.pop(key, None)
        else:
            result[key] = merge_patch(result.get(key), value)
    return result


def _record(row: DBAccountPreference) -> PreferenceRecord:
    return PreferenceRecord(
        namespace=row.namespace,
        workspace=row.workspace,
        value=row.value,
        db_version=row.db_version,
        updated_at=row.updated_at,
    )


def _check_size(value: dict[str, Any]) -> None:
    if len(json.dumps(value, separators=(",", ":")).encode()) > PREFERENCE_MAX_BYTES:
        raise PreferenceTooLargeError(f"Preference documents are limited to {PREFERENCE_MAX_BYTES} bytes")


class AccountPreferenceStore:
    """CRUD for account preference documents.

    ``if_match`` is the expected ``db_version``; 0 means the document must not exist yet. Without it,
    writes retry on concurrent modification (last writer wins per key for merge patches).
    """

    def __init__(self, session_maker: async_sessionmaker[AsyncSession]) -> None:
        self.session_maker = session_maker

    async def get(self, account_id: str, namespace: str, workspace: str | None) -> PreferenceRecord | None:
        async with self.session_maker() as session:
            row = await self._get_row(session, account_id, namespace, workspace)
        return _record(row) if row else None

    async def list_documents(
        self, account_id: str, *, namespace_prefix: str | None = None, workspace: str | None = None
    ) -> list[PreferenceRecord]:
        """Documents in one scope (global when ``workspace`` is None), optionally under a namespace prefix."""
        stmt = select(DBAccountPreference).where(
            DBAccountPreference.account_id == account_id, _workspace_clause(workspace)
        )
        if namespace_prefix:
            stmt = stmt.where(
                or_(
                    DBAccountPreference.namespace == namespace_prefix,
                    DBAccountPreference.namespace.startswith(f"{namespace_prefix}.", autoescape=True),
                )
            )
        async with self.session_maker() as session:
            rows = (await session.scalars(stmt.order_by(DBAccountPreference.namespace))).all()
        return [_record(row) for row in rows]

    async def put(
        self,
        account_id: str,
        namespace: str,
        workspace: str | None,
        value: dict[str, Any],
        *,
        if_match: int | None = None,
    ) -> PreferenceRecord:
        # Applied as a patch onto nothing so nulls are dropped: a stored null would hide the default.
        return await self._write(account_id, namespace, workspace, lambda _current: merge_patch({}, value), if_match)

    async def patch(
        self,
        account_id: str,
        namespace: str,
        workspace: str | None,
        patch: dict[str, Any],
        *,
        if_match: int | None = None,
    ) -> PreferenceRecord:
        return await self._write(
            account_id, namespace, workspace, lambda current: merge_patch(current, patch), if_match
        )

    async def delete(
        self, account_id: str, namespace: str, workspace: str | None, *, if_match: int | None = None
    ) -> bool:
        try:
            async with self.session_maker() as session, session.begin():
                row = await self._get_row(session, account_id, namespace, workspace)
                if if_match is not None and if_match != (row.db_version if row else 0):
                    raise PreferenceVersionConflictError("Preference version does not match If-Match")
                if row is None:
                    return False
                await session.delete(row)
        except StaleDataError as exc:
            raise PreferenceVersionConflictError("Preference was modified concurrently") from exc
        return True

    async def _write(
        self,
        account_id: str,
        namespace: str,
        workspace: str | None,
        compute: Callable[[dict[str, Any]], dict[str, Any]],
        if_match: int | None,
    ) -> PreferenceRecord:
        for attempt in range(_WRITE_ATTEMPTS):
            if attempt:
                await asyncio.sleep(random.uniform(0, 0.01 * attempt))
            try:
                async with self.session_maker() as session, session.begin():
                    row = await self._get_row(session, account_id, namespace, workspace)
                    if if_match is not None and if_match != (row.db_version if row else 0):
                        raise PreferenceVersionConflictError("Preference version does not match If-Match")
                    value = compute(row.value if row else {})
                    _check_size(value)
                    if row is None:
                        count = await session.scalar(
                            select(func.count())
                            .select_from(DBAccountPreference)
                            .where(DBAccountPreference.account_id == account_id)
                        )
                        if (count or 0) >= PREFERENCE_MAX_DOCUMENTS:
                            raise PreferenceLimitError(
                                f"Accounts are limited to {PREFERENCE_MAX_DOCUMENTS} preference documents"
                            )
                        row = DBAccountPreference(
                            id=generate_entity_id("account_preference"),
                            account_id=account_id,
                            namespace=namespace,
                            workspace=workspace,
                            value=value,
                        )
                        session.add(row)
                    else:
                        row.value = value
                    await session.flush()
                    return _record(row)
            except (IntegrityError, StaleDataError) as exc:
                if if_match is not None:
                    raise PreferenceVersionConflictError("Preference was modified concurrently") from exc
        raise PreferenceVersionConflictError("Preference was modified concurrently; retry")

    @staticmethod
    async def _get_row(
        session: AsyncSession, account_id: str, namespace: str, workspace: str | None
    ) -> DBAccountPreference | None:
        return await session.scalar(
            select(DBAccountPreference).where(
                DBAccountPreference.account_id == account_id,
                DBAccountPreference.namespace == namespace,
                _workspace_clause(workspace),
            )
        )


def _workspace_clause(workspace: str | None):
    if workspace is None:
        return DBAccountPreference.workspace.is_(None)
    return DBAccountPreference.workspace == workspace
