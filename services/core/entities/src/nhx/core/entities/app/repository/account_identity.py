# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct storage for stable account identity resolution."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Literal
from weakref import WeakKeyDictionary

from nhx.core.entities.app.repository.sqlalchemy.models import DBAccount, DBAccountIdentity
from nhx.core.entities.cache import TTLCache
from nhx.core.entities.config import EntitiesConfig
from nhx.core.entities.utils.identifiers import generate_entity_id
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

AccountType = Literal["user", "service"]

# Stores are built per request, so the cache is keyed by session maker. Each event loop owns one session
# maker and engine, so a cache never serves identities read from a different database.
_identity_caches: WeakKeyDictionary[async_sessionmaker[AsyncSession], TTLCache] = WeakKeyDictionary()
_identity_caches_lock = threading.Lock()


def _identity_cache(session_maker: async_sessionmaker[AsyncSession]) -> TTLCache | None:
    """Return the cache for ``session_maker``, or ``None`` if caching is disabled.

    TTL and size are fixed when a session maker's cache is created; only the enabled flag is re-read per call.
    """
    config = EntitiesConfig.get()
    if not config.account_identity_cache_enabled:
        return None
    with _identity_caches_lock:
        cache = _identity_caches.get(session_maker)
        if cache is None:
            cache = TTLCache(
                maxsize=config.account_identity_cache_max_size,
                ttl=config.account_identity_cache_ttl_sec,
            )
            _identity_caches[session_maker] = cache
        return cache


def _remember(
    cache: TTLCache | None, issuer: str, subject: str, record: AccountIdentityRecord
) -> AccountIdentityRecord:
    if cache is not None:
        cache[(issuer, subject)] = record
    return record


def evict_account_identity(session_maker: async_sessionmaker[AsyncSession], *, issuer: str, subject: str) -> None:
    """Drop one cached identity so its next resolution reads the database.

    Call after deactivating or relinking the identity. This only clears the cache behind ``session_maker``;
    other event loops and replicas keep serving the old record until it expires.
    """
    with _identity_caches_lock:
        cache = _identity_caches.get(session_maker)
    if cache is not None:
        cache.pop((issuer, subject), None)


class AccountIdentityUnavailableError(RuntimeError):
    """Raised when an account or identity exists but is not active."""


class AccountIdentityConflictError(RuntimeError):
    """Raised when a concurrent write conflicts and no active identity can be reread."""


@dataclass(frozen=True)
class AccountIdentityRecord:
    """Resolved stable account identity."""

    account_id: str
    account_type: AccountType
    identity_id: str
    issuer: str
    subject: str


class AccountIdentityStore:
    """Resolve or materialize account identities against the entities database."""

    def __init__(self, session_maker: async_sessionmaker[AsyncSession]) -> None:
        self.session_maker = session_maker

    async def get_active_identity(self, *, issuer: str, subject: str) -> AccountIdentityRecord | None:
        """Return the active identity for ``issuer``/``subject`` if present."""
        async with self.session_maker() as session:
            return await self._get_active_identity(session, issuer=issuer, subject=subject)

    async def resolve_or_materialize(
        self,
        *,
        issuer: str,
        subject: str,
        subject_claim: str,
        account_type: AccountType,
        display_name: str | None = None,
        primary_email: str | None = None,
        link_key: str | None = None,
        link_key_claim: str | None = None,
        claims_snapshot: dict[str, Any] | None = None,
        linked_via: str = "resolver_materialization",
        linked_by: str | None = None,
    ) -> AccountIdentityRecord:
        """Resolve an identity, materializing account and identity rows on miss."""
        cache = _identity_cache(self.session_maker)
        if cache is not None:
            cached = cache.get((issuer, subject))
            if cached is not None:
                return cached

        existing = await self.get_active_identity(issuer=issuer, subject=subject)
        if existing is not None:
            return _remember(cache, issuer, subject, existing)

        account_id = generate_entity_id("account")
        identity_id = generate_entity_id("account_identity")
        try:
            async with self.session_maker() as session:
                async with session.begin():
                    account = await self._get_active_account_by_link_key(session, link_key=link_key)
                    if account is None:
                        account = DBAccount(
                            id=account_id,
                            type=account_type,
                            link_key=link_key,
                            display_name=display_name,
                            primary_email=primary_email,
                            status="active",
                        )
                        session.add(account)
                    identity = DBAccountIdentity(
                        id=identity_id,
                        account_id=account.id,
                        issuer=issuer,
                        subject=subject,
                        subject_claim=subject_claim,
                        status="active",
                        linked_via=linked_via,
                        linked_by=linked_by,
                        last_seen_at=None,
                        link_key_claim_at_link_time=link_key_claim,
                        link_key_value_at_link_time=link_key,
                        claims_snapshot=claims_snapshot or {},
                    )
                    session.add(identity)
        except IntegrityError as exc:
            existing_after_conflict = await self.get_active_identity(issuer=issuer, subject=subject)
            if existing_after_conflict is not None:
                return _remember(cache, issuer, subject, existing_after_conflict)
            if link_key is not None:
                linked_after_conflict = await self._link_identity_to_active_account_by_link_key(
                    issuer=issuer,
                    subject=subject,
                    subject_claim=subject_claim,
                    link_key=link_key,
                    link_key_claim=link_key_claim,
                    claims_snapshot=claims_snapshot,
                    linked_via=linked_via,
                    linked_by=linked_by,
                )
                if linked_after_conflict is not None:
                    return _remember(cache, issuer, subject, linked_after_conflict)
            raise AccountIdentityConflictError(
                f"Account identity conflict for issuer={issuer!r}, subject={subject!r}"
            ) from exc

        created = await self.get_active_identity(issuer=issuer, subject=subject)
        if created is None:
            raise AccountIdentityConflictError(
                f"Account identity was not readable after materialization for issuer={issuer!r}, subject={subject!r}"
            )
        return _remember(cache, issuer, subject, created)

    async def _link_identity_to_active_account_by_link_key(
        self,
        *,
        issuer: str,
        subject: str,
        subject_claim: str,
        link_key: str,
        link_key_claim: str | None,
        claims_snapshot: dict[str, Any] | None,
        linked_via: str,
        linked_by: str | None,
    ) -> AccountIdentityRecord | None:
        try:
            async with self.session_maker() as session:
                async with session.begin():
                    account = await self._get_active_account_by_link_key(session, link_key=link_key)
                    if account is None:
                        return None
                    session.add(
                        DBAccountIdentity(
                            id=generate_entity_id("account_identity"),
                            account_id=account.id,
                            issuer=issuer,
                            subject=subject,
                            subject_claim=subject_claim,
                            status="active",
                            linked_via=linked_via,
                            linked_by=linked_by,
                            last_seen_at=None,
                            link_key_claim_at_link_time=link_key_claim,
                            link_key_value_at_link_time=link_key,
                            claims_snapshot=claims_snapshot or {},
                        )
                    )
        except IntegrityError as exc:
            existing_after_conflict = await self.get_active_identity(issuer=issuer, subject=subject)
            if existing_after_conflict is not None:
                return existing_after_conflict
            raise AccountIdentityConflictError(
                f"Account identity conflict for issuer={issuer!r}, subject={subject!r}"
            ) from exc

        return await self.get_active_identity(issuer=issuer, subject=subject)

    async def _get_active_identity(
        self,
        session: AsyncSession,
        *,
        issuer: str,
        subject: str,
    ) -> AccountIdentityRecord | None:
        stmt = (
            select(DBAccountIdentity, DBAccount)
            .join(DBAccount, DBAccount.id == DBAccountIdentity.account_id)
            .where(DBAccountIdentity.issuer == issuer, DBAccountIdentity.subject == subject)
        )
        row = (await session.execute(stmt)).one_or_none()
        if row is None:
            return None

        identity, account = row
        if identity.status != "active" or account.status != "active":
            raise AccountIdentityUnavailableError(
                f"Account identity is not active for issuer={issuer!r}, subject={subject!r}"
            )
        return AccountIdentityRecord(
            account_id=account.id,
            account_type=account.type,
            identity_id=identity.id,
            issuer=identity.issuer,
            subject=identity.subject,
        )

    async def _get_active_account_by_link_key(self, session: AsyncSession, *, link_key: str | None) -> DBAccount | None:
        if link_key is None:
            return None
        stmt = select(DBAccount).where(DBAccount.link_key == link_key, DBAccount.status == "active")
        return (await session.execute(stmt)).scalar_one_or_none()
