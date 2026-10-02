# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest
from nhx.core.entities.app.repository.account_credential import AccountCredentialStore, credential_lookup_hash
from nhx.core.entities.app.repository.account_identity import AccountIdentityStore
from nhx.core.entities.app.repository.sqlalchemy.models import DBAccount, DBAccountCredential, DBAccountIdentity
from sqlalchemy import select


async def _identity(session_maker):
    return await AccountIdentityStore(session_maker).resolve_or_materialize(
        issuer="https://idp.example.com",
        subject="user-1",
        subject_claim="sub",
        account_type="user",
    )


@pytest.mark.asyncio
async def test_create_hashes_handle_and_active_lookup_checks_identity(session_maker) -> None:
    identity = await _identity(session_maker)
    store = AccountCredentialStore(session_maker)

    created = await store.create(
        credential_type="web_session",
        handle="opaque-session",
        owner_account_id=identity.account_id,
        account_identity_id=identity.identity_id,
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        encrypted_payload="encrypted",
    )

    assert created.lookup_hash == credential_lookup_hash("opaque-session")
    assert "opaque-session" not in created.lookup_hash
    assert await store.get_active("web_session", "opaque-session") == created
    async with session_maker() as session:
        row = await session.get(DBAccountCredential, created.id)
        assert row is not None
        assert row.lookup_hash == credential_lookup_hash("opaque-session")

        identity_row = await session.get(DBAccountIdentity, identity.identity_id)
        assert identity_row is not None
        identity_row.status = "disabled"
        await session.commit()

    assert await store.get_active("web_session", "opaque-session") is None


@pytest.mark.asyncio
async def test_active_lookup_rejects_inactive_account_and_expired_credential(session_maker) -> None:
    identity = await _identity(session_maker)
    store = AccountCredentialStore(session_maker)
    await store.create(
        credential_type="cli_refresh",
        handle="expired",
        owner_account_id=identity.account_id,
        account_identity_id=identity.identity_id,
        expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
    )
    await store.create(
        credential_type="cli_refresh",
        handle="active",
        owner_account_id=identity.account_id,
        account_identity_id=identity.identity_id,
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
    )

    assert await store.get_active("cli_refresh", "expired") is None
    async with session_maker() as session:
        account = await session.get(DBAccount, identity.account_id)
        assert account is not None
        account.status = "suspended"
        await session.commit()
    assert await store.get_active("cli_refresh", "active") is None


@pytest.mark.asyncio
async def test_active_lookup_rejects_identity_bound_to_another_subject_account(session_maker) -> None:
    owner_identity = await _identity(session_maker)
    other_identity = await AccountIdentityStore(session_maker).resolve_or_materialize(
        issuer="https://idp.example.com",
        subject="user-2",
        subject_claim="sub",
        account_type="user",
    )
    store = AccountCredentialStore(session_maker)
    await store.create(
        credential_type="web_session",
        handle="mismatched-identity",
        owner_account_id=owner_identity.account_id,
        subject_account_id=owner_identity.account_id,
        account_identity_id=other_identity.identity_id,
    )

    assert await store.get_active("web_session", "mismatched-identity") is None


@pytest.mark.asyncio
async def test_consume_active_is_single_use_and_revoke_for_account_is_complete(session_maker) -> None:
    identity = await _identity(session_maker)
    store = AccountCredentialStore(session_maker)
    code = await store.create(
        credential_type="cli_login_code",
        handle="one-time",
        owner_account_id=identity.account_id,
        account_identity_id=identity.identity_id,
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=2),
    )
    await store.create(
        credential_type="web_session",
        handle="session",
        owner_account_id=identity.account_id,
        account_identity_id=identity.identity_id,
    )

    assert await store.consume_active("cli_login_code", "one-time") == code
    assert await store.consume_active("cli_login_code", "one-time") is None
    assert await store.revoke_for_account(identity.account_id, revoked_by="admin") == 1
    revoked = await store.get("web_session", "session")
    assert revoked is not None
    assert revoked.status == "REVOKED"
    assert revoked.revoked_by == "admin"


@pytest.mark.asyncio
async def test_versioned_update_replaces_encrypted_payload(session_maker) -> None:
    identity = await _identity(session_maker)
    store = AccountCredentialStore(session_maker)
    created = await store.create(
        credential_type="cli_refresh",
        handle="refresh-handle",
        owner_account_id=identity.account_id,
        encrypted_payload="first",
    )

    updated = await store.update(replace(created, encrypted_payload="second"))

    assert updated.db_version == created.db_version + 1
    stored = await store.get_active("cli_refresh", "refresh-handle")
    assert stored is not None
    assert stored.encrypted_payload == "second"
    async with session_maker() as session:
        hashes = list(await session.scalars(select(DBAccountCredential.lookup_hash)))
    assert hashes == [credential_lookup_hash("refresh-handle")]
