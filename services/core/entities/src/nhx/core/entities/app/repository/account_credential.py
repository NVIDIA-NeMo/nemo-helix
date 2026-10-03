# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct storage for account-bound credentials."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any

from nhx.core.entities.app.repository.sqlalchemy.models import DBAccount, DBAccountCredential, DBAccountIdentity
from nhx.core.entities.utils.identifiers import generate_entity_id
from sqlalchemy import and_, delete, func, or_, select, update
from sqlalchemy.engine import CursorResult, Result
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import aliased


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _aware(value: datetime | None) -> datetime | None:
    if value is None or value.tzinfo is not None:
        return value
    return value.replace(tzinfo=timezone.utc)


def _naive_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def credential_lookup_hash(handle: str) -> str:
    """Hash a client-presented credential handle for indexed lookup."""
    return hashlib.sha256(handle.encode("utf-8")).hexdigest()


def _rowcount(result: Result[Any]) -> int:
    if not isinstance(result, CursorResult):
        raise RuntimeError("Credential mutation did not return a row count")
    return result.rowcount


class AccountCredentialConflictError(RuntimeError):
    """Raised when a credential create or versioned update conflicts."""


@dataclass(frozen=True)
class AccountCredentialRecord:
    id: str
    owner_account_id: str
    subject_account_id: str
    account_identity_id: str | None
    credential_type: str
    lookup_hash: str
    status: str
    issued_at: datetime
    expires_at: datetime | None
    last_used_at: datetime | None
    revoked_at: datetime | None
    revoked_by: str | None
    public_metadata: dict[str, Any]
    encrypted_payload: str | None
    db_version: int


class AccountCredentialStore:
    """Persist credentials and enforce their account and identity lifecycle."""

    def __init__(self, session_maker: async_sessionmaker[AsyncSession]) -> None:
        self.session_maker = session_maker

    async def create(
        self,
        *,
        credential_type: str,
        handle: str,
        owner_account_id: str,
        subject_account_id: str | None = None,
        account_identity_id: str | None = None,
        status: str = "ACTIVE",
        issued_at: datetime | None = None,
        expires_at: datetime | None = None,
        public_metadata: dict[str, Any] | None = None,
        encrypted_payload: str | None = None,
    ) -> AccountCredentialRecord:
        row = DBAccountCredential(
            id=generate_entity_id("account_credential"),
            owner_account_id=owner_account_id,
            subject_account_id=subject_account_id or owner_account_id,
            account_identity_id=account_identity_id,
            credential_type=credential_type,
            lookup_hash=credential_lookup_hash(handle),
            status=status,
            issued_at=_naive_utc(issued_at or datetime.now(timezone.utc)),
            expires_at=_naive_utc(expires_at) if expires_at is not None else None,
            public_metadata=public_metadata or {},
            encrypted_payload=encrypted_payload,
        )
        try:
            async with self.session_maker() as session:
                session.add(row)
                await session.commit()
        except IntegrityError as exc:
            raise AccountCredentialConflictError(f"Credential already exists for type={credential_type!r}") from exc
        return self._record(row)

    async def get(self, credential_type: str, handle: str) -> AccountCredentialRecord | None:
        async with self.session_maker() as session:
            row = await session.scalar(
                select(DBAccountCredential).where(
                    DBAccountCredential.credential_type == credential_type,
                    DBAccountCredential.lookup_hash == credential_lookup_hash(handle),
                )
            )
        return self._record(row) if row is not None else None

    async def get_active(self, credential_type: str, handle: str) -> AccountCredentialRecord | None:
        owner = aliased(DBAccount)
        subject = aliased(DBAccount)
        identity = aliased(DBAccountIdentity)
        now = _utcnow()
        async with self.session_maker() as session:
            row = await session.scalar(
                select(DBAccountCredential)
                .join(owner, owner.id == DBAccountCredential.owner_account_id)
                .join(subject, subject.id == DBAccountCredential.subject_account_id)
                .outerjoin(
                    identity,
                    and_(
                        identity.id == DBAccountCredential.account_identity_id,
                        identity.account_id == DBAccountCredential.subject_account_id,
                    ),
                )
                .where(
                    DBAccountCredential.credential_type == credential_type,
                    DBAccountCredential.lookup_hash == credential_lookup_hash(handle),
                    DBAccountCredential.status.in_(("ACTIVE", "ROTATING")),
                    or_(DBAccountCredential.expires_at.is_(None), DBAccountCredential.expires_at > now),
                    owner.status == "active",
                    subject.status == "active",
                    or_(DBAccountCredential.account_identity_id.is_(None), identity.status == "active"),
                )
            )
        return self._record(row) if row is not None else None

    async def consume_active(self, credential_type: str, handle: str) -> AccountCredentialRecord | None:
        record = await self.get_active(credential_type, handle)
        if record is None:
            return None
        async with self.session_maker() as session:
            result = await session.execute(
                delete(DBAccountCredential).where(
                    DBAccountCredential.id == record.id,
                    DBAccountCredential.db_version == record.db_version,
                )
            )
            if _rowcount(result) != 1:
                await session.rollback()
                return None
            await session.commit()
        return record

    async def update(self, record: AccountCredentialRecord) -> AccountCredentialRecord:
        values = {
            "status": record.status,
            "expires_at": _naive_utc(record.expires_at) if record.expires_at is not None else None,
            "last_used_at": _naive_utc(record.last_used_at) if record.last_used_at is not None else None,
            "revoked_at": _naive_utc(record.revoked_at) if record.revoked_at is not None else None,
            "revoked_by": record.revoked_by,
            "public_metadata": record.public_metadata,
            "encrypted_payload": record.encrypted_payload,
            "db_version": record.db_version + 1,
        }
        async with self.session_maker() as session:
            result = await session.execute(
                update(DBAccountCredential)
                .where(
                    DBAccountCredential.id == record.id,
                    DBAccountCredential.db_version == record.db_version,
                )
                .values(**values)
            )
            if _rowcount(result) != 1:
                await session.rollback()
                raise AccountCredentialConflictError(f"Credential {record.id!r} changed concurrently")
            await session.commit()
        return replace(record, db_version=record.db_version + 1)

    async def revoke(self, credential_type: str, handle: str, *, revoked_by: str | None = None) -> bool:
        record = await self.get(credential_type, handle)
        if record is None or record.status == "REVOKED":
            return False
        await self.update(
            replace(record, status="REVOKED", revoked_at=datetime.now(timezone.utc), revoked_by=revoked_by)
        )
        return True

    async def delete(self, credential_type: str, handle: str) -> bool:
        async with self.session_maker() as session:
            result = await session.execute(
                delete(DBAccountCredential).where(
                    DBAccountCredential.credential_type == credential_type,
                    DBAccountCredential.lookup_hash == credential_lookup_hash(handle),
                )
            )
            await session.commit()
        return _rowcount(result) == 1

    async def list_for_owner(
        self,
        credential_type: str,
        owner_account_id: str,
        *,
        include_service_subjects: bool,
        offset: int,
        limit: int,
    ) -> tuple[list[AccountCredentialRecord], bool]:
        subject = aliased(DBAccount)
        ownership = DBAccountCredential.owner_account_id == owner_account_id
        query = select(DBAccountCredential).join(subject, subject.id == DBAccountCredential.subject_account_id)
        if include_service_subjects:
            ownership = or_(ownership, subject.type == "service")
        query = (
            query.where(DBAccountCredential.credential_type == credential_type, ownership)
            .order_by(DBAccountCredential.issued_at.desc())
            .offset(offset)
            .limit(limit + 1)
        )
        async with self.session_maker() as session:
            rows = list((await session.scalars(query)).all())
        return [self._record(row) for row in rows[:limit]], len(rows) > limit

    async def revoke_for_account(self, account_id: str, *, revoked_by: str | None = None) -> int:
        now = _utcnow()
        async with self.session_maker() as session:
            result = await session.execute(
                update(DBAccountCredential)
                .where(
                    or_(
                        DBAccountCredential.owner_account_id == account_id,
                        DBAccountCredential.subject_account_id == account_id,
                    ),
                    DBAccountCredential.status.in_(("ACTIVE", "SUSPENDED", "ROTATING")),
                )
                .values(status="REVOKED", revoked_at=now, revoked_by=revoked_by)
            )
            await session.commit()
        return _rowcount(result)

    async def count(self, credential_type: str) -> int:
        async with self.session_maker() as session:
            return int(
                await session.scalar(
                    select(func.count())
                    .select_from(DBAccountCredential)
                    .where(DBAccountCredential.credential_type == credential_type)
                )
                or 0
            )

    @staticmethod
    def _record(row: DBAccountCredential) -> AccountCredentialRecord:
        return AccountCredentialRecord(
            id=row.id,
            owner_account_id=row.owner_account_id,
            subject_account_id=row.subject_account_id,
            account_identity_id=row.account_identity_id,
            credential_type=row.credential_type,
            lookup_hash=row.lookup_hash,
            status=row.status,
            issued_at=_aware(row.issued_at) or datetime.now(timezone.utc),
            expires_at=_aware(row.expires_at),
            last_used_at=_aware(row.last_used_at),
            revoked_at=_aware(row.revoked_at),
            revoked_by=row.revoked_by,
            public_metadata=dict(row.public_metadata),
            encrypted_payload=row.encrypted_payload,
            db_version=row.db_version,
        )
