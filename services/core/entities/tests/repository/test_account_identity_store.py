# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest
from nmp.core.entities.app.repository.account_identity import AccountIdentityStore
from nmp.core.entities.app.repository.sqlalchemy.models import DBAccount, DBAccountIdentity
from sqlalchemy import func, select


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
