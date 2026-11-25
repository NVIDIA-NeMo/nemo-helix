# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import Callable, Iterator

import pytest
from nhx.common.config import Configuration
from nhx.core.entities.app.repository.account_identity import (
    AccountIdentityStore,
    AccountIdentityUnavailableError,
    _identity_cache,
    evict_account_identity,
)
from nhx.core.entities.app.repository.sqlalchemy.models import DBAccount, DBAccountIdentity
from nhx.core.entities.config import EntitiesConfig
from pydantic import ValidationError
from sqlalchemy import event, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


@pytest.mark.asyncio
async def test_resolve_or_materialize_creates_account_and_identity(session_maker):
    store = AccountIdentityStore(session_maker)

    record = await store.resolve_or_materialize(
        issuer="https://idp.example.com",
        subject="sub-123",
        subject_claim="sub",
        account_type="user",
        display_name="Alice",
        primary_email="alice@example.com",
        claims_snapshot={"sub": "sub-123", "email": "alice@example.com"},
    )

    assert record.account_id.startswith("account-")
    assert record.account_type == "user"
    async with session_maker() as session:
        account_count = await session.scalar(select(func.count()).select_from(DBAccount))
        identity_count = await session.scalar(select(func.count()).select_from(DBAccountIdentity))
    assert account_count == 1
    assert identity_count == 1


@pytest.mark.asyncio
async def test_resolve_or_materialize_reuses_existing_identity(session_maker):
    store = AccountIdentityStore(session_maker)

    first = await store.resolve_or_materialize(
        issuer="https://idp.example.com",
        subject="sub-123",
        subject_claim="sub",
        account_type="user",
    )
    second = await store.resolve_or_materialize(
        issuer="https://idp.example.com",
        subject="sub-123",
        subject_claim="sub",
        account_type="user",
    )

    assert second.account_id == first.account_id
    async with session_maker() as session:
        account_count = await session.scalar(select(func.count()).select_from(DBAccount))
        identity_count = await session.scalar(select(func.count()).select_from(DBAccountIdentity))
    assert account_count == 1
    assert identity_count == 1


@pytest.mark.asyncio
async def test_resolve_or_materialize_handles_concurrent_identity_creation(session_maker):
    store = AccountIdentityStore(session_maker)

    async def resolve_once():
        return await store.resolve_or_materialize(
            issuer="https://idp.example.com",
            subject="sub-race",
            subject_claim="sub",
            account_type="user",
        )

    records = await asyncio.gather(*(resolve_once() for _ in range(8)))

    assert len({record.account_id for record in records}) == 1
    async with session_maker() as session:
        account_count = await session.scalar(select(func.count()).select_from(DBAccount))
        identity_count = await session.scalar(select(func.count()).select_from(DBAccountIdentity))
    assert account_count == 1
    assert identity_count == 1


@pytest.mark.asyncio
async def test_resolve_or_materialize_does_not_auto_link_by_email(session_maker):
    store = AccountIdentityStore(session_maker)

    first = await store.resolve_or_materialize(
        issuer="https://idp-a.example.com",
        subject="sub-a",
        subject_claim="sub",
        account_type="user",
        primary_email="same@example.com",
    )
    second = await store.resolve_or_materialize(
        issuer="https://idp-b.example.com",
        subject="sub-b",
        subject_claim="sub",
        account_type="user",
        primary_email="same@example.com",
    )

    assert first.account_id != second.account_id
    async with session_maker() as session:
        account_count = await session.scalar(select(func.count()).select_from(DBAccount))
        identity_count = await session.scalar(select(func.count()).select_from(DBAccountIdentity))
    assert account_count == 2
    assert identity_count == 2


@pytest.mark.asyncio
async def test_resolve_or_materialize_reuses_account_with_same_link_key(session_maker):
    store = AccountIdentityStore(session_maker)

    first = await store.resolve_or_materialize(
        issuer="https://idp-a.example.com",
        subject="sub-a",
        subject_claim="sub",
        account_type="user",
        link_key="email:alice@example.com",
        link_key_claim="email",
    )
    second = await store.resolve_or_materialize(
        issuer="https://idp-b.example.com",
        subject="sub-b",
        subject_claim="sub",
        account_type="user",
        link_key="email:alice@example.com",
        link_key_claim="email",
    )

    assert second.account_id == first.account_id
    async with session_maker() as session:
        account_count = await session.scalar(select(func.count()).select_from(DBAccount))
        identity_count = await session.scalar(select(func.count()).select_from(DBAccountIdentity))
    assert account_count == 1
    assert identity_count == 2


ISSUER = "https://idp.example.com"


def _count_statements(session_maker: async_sessionmaker[AsyncSession]) -> list[str]:
    """Record every SQL statement the session maker's engine runs from now on."""
    statements: list[str] = []
    engine = session_maker.kw["bind"]
    event.listen(engine.sync_engine, "before_cursor_execute", lambda *args: statements.append(args[2]))
    return statements


async def _resolve(store: AccountIdentityStore, subject: str = "sub-cache"):
    return await store.resolve_or_materialize(issuer=ISSUER, subject=subject, subject_claim="sub", account_type="user")


@pytest.fixture
def cache_config() -> Iterator[Callable[..., None]]:
    """Override the entities config for one test."""

    def configure(**fields: object) -> None:
        Configuration.set_override(EntitiesConfig(**fields))

    yield configure
    Configuration.clear_override(EntitiesConfig)


@pytest.mark.asyncio
async def test_cached_identity_resolves_without_a_database_query(session_maker, cache_config):
    cache_config(account_identity_cache_enabled=True)
    store = AccountIdentityStore(session_maker)
    first = await _resolve(store)
    statements = _count_statements(session_maker)

    second = await _resolve(AccountIdentityStore(session_maker))

    assert second == first
    assert statements == []


@pytest.mark.asyncio
async def test_each_session_maker_has_its_own_cache(session_maker, cache_config):
    """A maker that never cached an identity reads the database once, then serves it from its own cache."""
    cache_config(account_identity_cache_enabled=True)
    created = await _resolve(AccountIdentityStore(session_maker))
    other = async_sessionmaker(bind=session_maker.kw["bind"], class_=AsyncSession, expire_on_commit=False)
    store = AccountIdentityStore(other)
    statements = _count_statements(other)

    assert await _resolve(store) == created
    reads = len(statements)
    assert reads > 0
    assert await _resolve(store) == created
    assert len(statements) == reads


@pytest.mark.asyncio
async def test_disabled_cache_queries_every_time(session_maker, cache_config):
    cache_config(account_identity_cache_enabled=False)
    store = AccountIdentityStore(session_maker)
    await _resolve(store)
    statements = _count_statements(session_maker)

    await _resolve(store)

    assert statements != []


@pytest.mark.asyncio
async def test_cached_identity_survives_deactivation_only_until_the_ttl(session_maker, cache_config):
    cache_config(account_identity_cache_enabled=True, account_identity_cache_ttl_sec=60)
    store = AccountIdentityStore(session_maker)
    await _resolve(store)
    async with session_maker() as session, session.begin():
        await session.execute(update(DBAccountIdentity).values(status="disabled"))

    await _resolve(store)  # still served from the cache, by design

    cache = _identity_cache(session_maker)
    assert cache is not None
    cache.expire(cache.timer() + 61)
    with pytest.raises(AccountIdentityUnavailableError):
        await _resolve(store)


@pytest.mark.asyncio
async def test_inactive_identity_is_never_cached(session_maker, cache_config):
    cache_config(account_identity_cache_enabled=True)
    store = AccountIdentityStore(session_maker)
    await _resolve(store)
    async with session_maker() as session, session.begin():
        await session.execute(update(DBAccountIdentity).values(status="disabled"))
    other = async_sessionmaker(bind=session_maker.kw["bind"], class_=AsyncSession, expire_on_commit=False)

    for _ in range(2):
        with pytest.raises(AccountIdentityUnavailableError):
            await _resolve(AccountIdentityStore(other))

    async with session_maker() as session, session.begin():
        await session.execute(update(DBAccountIdentity).values(status="active"))
    assert (await _resolve(AccountIdentityStore(other))).account_id.startswith("account-")


@pytest.mark.asyncio
async def test_evicted_identity_is_read_from_the_database_again(session_maker, cache_config):
    cache_config(account_identity_cache_enabled=True)
    store = AccountIdentityStore(session_maker)
    await _resolve(store)
    async with session_maker() as session, session.begin():
        await session.execute(update(DBAccountIdentity).values(status="disabled"))

    evict_account_identity(session_maker, issuer=ISSUER, subject="sub-cache")

    with pytest.raises(AccountIdentityUnavailableError):
        await _resolve(store)


@pytest.mark.asyncio
async def test_lookup_in_flight_during_an_eviction_does_not_repopulate_the_cache(
    session_maker, cache_config, monkeypatch
):
    """A lookup that read the row before the eviction must not publish it afterwards."""
    cache_config(account_identity_cache_enabled=True)
    store = AccountIdentityStore(session_maker)
    await _resolve(store)
    evict_account_identity(session_maker, issuer=ISSUER, subject="sub-cache")
    read_row = store.get_active_identity

    async def read_then_evict(*, issuer: str, subject: str):
        record = await read_row(issuer=issuer, subject=subject)
        # Deactivation and its eviction land after the read and before the lookup caches the row.
        evict_account_identity(session_maker, issuer=issuer, subject=subject)
        return record

    monkeypatch.setattr(store, "get_active_identity", read_then_evict)
    assert (await _resolve(store)).account_id.startswith("account-")

    cache = _identity_cache(session_maker)
    assert cache is not None
    assert (ISSUER, "sub-cache") not in cache

    monkeypatch.setattr(store, "get_active_identity", read_row)
    await _resolve(store)
    assert (ISSUER, "sub-cache") in cache


def test_non_positive_cache_ttl_is_rejected():
    with pytest.raises(ValidationError):
        EntitiesConfig(account_identity_cache_ttl_sec=0)
