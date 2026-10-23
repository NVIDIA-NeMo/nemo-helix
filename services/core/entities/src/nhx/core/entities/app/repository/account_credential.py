# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct storage for account-bound credentials."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Literal

from nemo_helix_plugin.filter_ops import ElemMatchCondition
from nhx.common.api.filter import FilterOperator
from nhx.common.datetime_utils import ensure_utc
from nhx.core.entities.app.repository.sqlalchemy.filter import SQLAlchemyFilterRepository
from nhx.core.entities.app.repository.sqlalchemy.models import DBAccount, DBAccountCredential, DBAccountIdentity
from nhx.core.entities.utils.identifiers import generate_entity_id
from sqlalchemy import and_, delete, func, or_, select, update
from sqlalchemy.engine import CursorResult, Result
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import aliased

AccountCredentialType = Literal["access_key", "broker_refresh", "token_authorization_code", "web_session"]
AccountCredentialStatus = Literal["ACTIVE", "REVOKED", "SUSPENDED", "ROTATING"]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


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
    credential_type: AccountCredentialType
    lookup_hash: str
    status: AccountCredentialStatus
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
        credential_type: AccountCredentialType,
        handle: str,
        owner_account_id: str,
        subject_account_id: str | None = None,
        account_identity_id: str | None = None,
        status: AccountCredentialStatus = "ACTIVE",
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

    async def get(self, credential_type: AccountCredentialType, handle: str) -> AccountCredentialRecord | None:
        async with self.session_maker() as session:
            row = await session.scalar(
                select(DBAccountCredential).where(
                    DBAccountCredential.credential_type == credential_type,
                    DBAccountCredential.lookup_hash == credential_lookup_hash(handle),
                )
            )
        return self._record(row) if row is not None else None

    async def get_active(self, credential_type: AccountCredentialType, handle: str) -> AccountCredentialRecord | None:
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

    async def consume_active(
        self, credential_type: AccountCredentialType, handle: str
    ) -> AccountCredentialRecord | None:
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

    async def rotate_active(
        self,
        record: AccountCredentialRecord,
        *,
        new_handle: str,
        expires_at: datetime | None,
        public_metadata: dict[str, Any],
        encrypted_payload: str,
    ) -> AccountCredentialRecord | None:
        """Replace an active credential with a newly issued handle in one transaction."""
        row = DBAccountCredential(
            id=generate_entity_id("account_credential"),
            owner_account_id=record.owner_account_id,
            subject_account_id=record.subject_account_id,
            account_identity_id=record.account_identity_id,
            credential_type=record.credential_type,
            lookup_hash=credential_lookup_hash(new_handle),
            issued_at=_utcnow(),
            expires_at=_naive_utc(expires_at) if expires_at is not None else None,
            public_metadata=public_metadata,
            encrypted_payload=encrypted_payload,
        )
        try:
            async with self.session_maker() as session:
                result = await session.execute(
                    delete(DBAccountCredential).where(
                        DBAccountCredential.id == record.id,
                        DBAccountCredential.db_version == record.db_version,
                        DBAccountCredential.status.in_(("ACTIVE", "ROTATING")),
                    )
                )
                if _rowcount(result) != 1:
                    await session.rollback()
                    return None
                session.add(row)
                await session.commit()
        except IntegrityError as exc:
            raise AccountCredentialConflictError(
                f"Credential already exists for type={record.credential_type!r}"
            ) from exc
        return self._record(row)

    async def revoke(
        self, credential_type: AccountCredentialType, handle: str, *, revoked_by: str | None = None
    ) -> bool:
        record = await self.get(credential_type, handle)
        if record is None or record.status == "REVOKED":
            return False
        await self.update(
            replace(record, status="REVOKED", revoked_at=datetime.now(timezone.utc), revoked_by=revoked_by)
        )
        return True

    async def delete(self, credential_type: AccountCredentialType, handle: str) -> bool:
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
        credential_type: AccountCredentialType,
        owner_account_id: str,
        *,
        include_service_subjects: bool,
        service_subject_metadata_contains: tuple[str, Sequence[str]] | None = None,
        offset: int,
        limit: int,
    ) -> tuple[list[AccountCredentialRecord], bool]:
        """List credentials owned by ``owner_account_id``, newest first, and whether more rows follow.

        ``include_service_subjects`` adds every credential whose subject is a service account.
        Otherwise ``service_subject_metadata_contains=(key, values)`` adds just the service-account
        credentials whose ``public_metadata[key]`` array contains any of ``values``.
        """
        subject = aliased(DBAccount)
        ownership = DBAccountCredential.owner_account_id == owner_account_id
        query = select(DBAccountCredential).join(subject, subject.id == DBAccountCredential.subject_account_id)
        async with self.session_maker() as session:
            if include_service_subjects:
                ownership = or_(ownership, subject.type == "service")
            elif service_subject_metadata_contains is not None and service_subject_metadata_contains[1]:
                key, values = service_subject_metadata_contains
                filters = SQLAlchemyFilterRepository(
                    DBAccountCredential,
                    dialect_name=session.get_bind().dialect.name,
                    json_path_columns=frozenset({"public_metadata"}),
                )
                ownership = or_(
                    ownership,
                    and_(
                        subject.type == "service",
                        # One $elemMatch/$in, so a long list of values stays a single array scan.
                        filters.elem_match(
                            f"public_metadata.{key}", [ElemMatchCondition(None, FilterOperator.IN, list(values))]
                        ),
                    ),
                )
            query = (
                query.where(DBAccountCredential.credential_type == credential_type, ownership)
                .order_by(DBAccountCredential.issued_at.desc())
                .offset(offset)
                # Fetch one extra row as a lookahead so callers can set `has_more`
                # without running a separate count query.
                .limit(limit + 1)
            )
            rows = list((await session.scalars(query)).all())
        return [self._record(row) for row in rows[:limit]], len(rows) > limit

    async def list_for_subject(
        self,
        credential_type: AccountCredentialType,
        subject_account_id: str,
        *,
        after_id: str | None = None,
        limit: int,
    ) -> tuple[list[AccountCredentialRecord], bool]:
        """List credentials of any status whose subject is ``subject_account_id``, ordered by ID.

        Returns the page and whether more rows follow. Pass the last returned record's ID as
        ``after_id`` to fetch the next page. Callers use this to scan every row for a security
        check, where a skipped row would fail open: unlike offset paging, a row deleted while the
        scan runs can't shift a later row onto a page already read.
        """
        query = select(DBAccountCredential).where(
            DBAccountCredential.credential_type == credential_type,
            DBAccountCredential.subject_account_id == subject_account_id,
        )
        if after_id is not None:
            query = query.where(DBAccountCredential.id > after_id)
        query = query.order_by(DBAccountCredential.id).limit(limit + 1)
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
                .values(db_version=DBAccountCredential.db_version + 1)
            )
            await session.commit()
        return _rowcount(result)

    async def count(self, credential_type: AccountCredentialType) -> int:
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
            credential_type=_credential_type(row.credential_type),
            lookup_hash=row.lookup_hash,
            status=_credential_status(row.status),
            issued_at=ensure_utc(row.issued_at) or datetime.now(timezone.utc),
            expires_at=ensure_utc(row.expires_at),
            last_used_at=ensure_utc(row.last_used_at),
            revoked_at=ensure_utc(row.revoked_at),
            revoked_by=row.revoked_by,
            public_metadata=dict(row.public_metadata),
            encrypted_payload=row.encrypted_payload,
            db_version=row.db_version,
        )


def _credential_type(value: str) -> AccountCredentialType:
    if value == "access_key":
        return "access_key"
    if value == "broker_refresh":
        return "broker_refresh"
    if value == "token_authorization_code":
        return "token_authorization_code"
    if value == "web_session":
        return "web_session"
    raise ValueError(f"Stored account credential has an unsupported type: {value!r}")


def _credential_status(value: str) -> AccountCredentialStatus:
    if value == "ACTIVE":
        return "ACTIVE"
    if value == "REVOKED":
        return "REVOKED"
    if value == "SUSPENDED":
        return "SUSPENDED"
    if value == "ROTATING":
        return "ROTATING"
    raise ValueError(f"Stored account credential has an unsupported status: {value!r}")
