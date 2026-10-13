# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Encrypted persistence for brokered OIDC login transactions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from nhx.core.auth.oidc_broker.crypto import decrypt_secret, encrypt_secret
from nhx.core.entities.app.repository.sqlalchemy.base import Base
from sqlalchemy import DateTime, Index, String, Text, delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import Mapped, mapped_column


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class DBAuthLoginTransaction(Base):
    """One encrypted, short-lived auth transaction created before identity is known."""

    __tablename__ = "auth_login_transactions"

    record_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    payload: Mapped[str] = mapped_column(Text, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    __table_args__ = (Index("idx_auth_login_transactions_expires", "expires_at"),)


@dataclass
class LoginRecord:
    """Decrypted login record."""

    kind: str
    payload: dict[str, str]
    expires_at: datetime


class OidcLoginStore:
    """Read and write encrypted pre-authentication OIDC transactions."""

    def __init__(self, session_maker: async_sessionmaker[AsyncSession], encryption_key: str) -> None:
        self._session_maker = session_maker
        self._encryption_key = encryption_key

    async def put(self, kind: str, record_id: str, payload: dict[str, str], ttl_seconds: int) -> None:
        now = _utcnow()
        expires_at = now + timedelta(seconds=ttl_seconds)
        stored = DBAuthLoginTransaction(
            record_hash=_hash_id(record_id),
            kind=kind,
            payload=encrypt_secret(json.dumps(payload), self._encryption_key),
            expires_at=expires_at,
        )
        async with self._session_maker() as session:
            await _purge_expired(session, now)
            await session.merge(stored)
            await session.commit()

    async def pop(self, kind: str, record_id: str) -> LoginRecord | None:
        async with self._session_maker() as session:
            result = await session.execute(
                delete(DBAuthLoginTransaction)
                .where(
                    DBAuthLoginTransaction.record_hash == _hash_id(record_id),
                    DBAuthLoginTransaction.kind == kind,
                )
                .returning(
                    DBAuthLoginTransaction.kind,
                    DBAuthLoginTransaction.payload,
                    DBAuthLoginTransaction.expires_at,
                )
            )
            await session.commit()
        row = result.first()
        if row is None or row.expires_at <= _utcnow():
            return None
        payload = json.loads(decrypt_secret(row.payload, self._encryption_key))
        if not isinstance(payload, dict):
            return None
        return LoginRecord(
            kind=row.kind,
            payload={str(key): str(value) for key, value in payload.items()},
            expires_at=row.expires_at,
        )

    async def get(self, kind: str, record_id: str) -> LoginRecord | None:
        async with self._session_maker() as session:
            row = await session.scalar(
                select(DBAuthLoginTransaction).where(DBAuthLoginTransaction.record_hash == _hash_id(record_id))
            )
        if row is None or row.kind != kind or row.expires_at <= _utcnow():
            return None
        payload = json.loads(decrypt_secret(row.payload, self._encryption_key))
        if not isinstance(payload, dict):
            return None
        return LoginRecord(
            kind=row.kind, payload={str(key): str(value) for key, value in payload.items()}, expires_at=row.expires_at
        )


def _hash_id(record_id: str) -> str:
    return hashlib.sha256(record_id.encode("utf-8")).hexdigest()


async def _purge_expired(session: AsyncSession, now: datetime) -> None:
    await session.execute(delete(DBAuthLoginTransaction).where(DBAuthLoginTransaction.expires_at <= now))
