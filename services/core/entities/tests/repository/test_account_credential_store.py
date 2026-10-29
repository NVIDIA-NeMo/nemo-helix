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
        credential_type="broker_refresh",
        handle="opaque-refresh",
        owner_account_id=identity.account_id,
        account_identity_id=identity.identity_id,
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        encrypted_payload="encrypted",
    )

    assert created.lookup_hash == credential_lookup_hash("opaque-refresh")
    assert "opaque-refresh" not in created.lookup_hash
    assert await store.get_active("broker_refresh", "opaque-refresh") == created
    async with session_maker() as session:
        row = await session.get(DBAccountCredential, created.id)
        assert row is not None
        assert row.lookup_hash == credential_lookup_hash("opaque-refresh")

        identity_row = await session.get(DBAccountIdentity, identity.identity_id)
        assert identity_row is not None
        identity_row.status = "disabled"
        await session.commit()

    assert await store.get_active("broker_refresh", "opaque-refresh") is None


@pytest.mark.asyncio
async def test_active_lookup_rejects_inactive_account_and_expired_credential(session_maker) -> None:
    identity = await _identity(session_maker)
    store = AccountCredentialStore(session_maker)
    await store.create(
        credential_type="broker_refresh",
        handle="expired",
        owner_account_id=identity.account_id,
        account_identity_id=identity.identity_id,
        expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
    )
    await store.create(
        credential_type="broker_refresh",
        handle="active",
        owner_account_id=identity.account_id,
        account_identity_id=identity.identity_id,
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
    )

    assert await store.get_active("broker_refresh", "expired") is None
    async with session_maker() as session:
        account = await session.get(DBAccount, identity.account_id)
        assert account is not None
        account.status = "suspended"
        await session.commit()
    assert await store.get_active("broker_refresh", "active") is None


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
        credential_type="broker_refresh",
        handle="mismatched-identity",
        owner_account_id=owner_identity.account_id,
        subject_account_id=owner_identity.account_id,
        account_identity_id=other_identity.identity_id,
    )

    assert await store.get_active("broker_refresh", "mismatched-identity") is None


@pytest.mark.asyncio
async def test_consume_active_is_single_use_and_revoke_for_account_is_complete(session_maker) -> None:
    identity = await _identity(session_maker)
    store = AccountCredentialStore(session_maker)
    code = await store.create(
        credential_type="token_authorization_code",
        handle="one-time",
        owner_account_id=identity.account_id,
        account_identity_id=identity.identity_id,
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=2),
    )
    await store.create(
        credential_type="broker_refresh",
        handle="refresh",
        owner_account_id=identity.account_id,
        account_identity_id=identity.identity_id,
    )

    assert await store.consume_active("token_authorization_code", "one-time") == code
    assert await store.consume_active("token_authorization_code", "one-time") is None
    assert await store.revoke_for_account(identity.account_id, revoked_by="admin") == 1
    revoked = await store.get("broker_refresh", "refresh")
    assert revoked is not None
    assert revoked.status == "REVOKED"
    assert revoked.revoked_by == "admin"
    assert revoked.db_version == 2


@pytest.mark.asyncio
async def test_versioned_update_replaces_encrypted_payload(session_maker) -> None:
    identity = await _identity(session_maker)
    store = AccountCredentialStore(session_maker)
    created = await store.create(
        credential_type="broker_refresh",
        handle="refresh-handle",
        owner_account_id=identity.account_id,
        encrypted_payload="first",
    )

    updated = await store.update(replace(created, encrypted_payload="second"))

    assert updated.db_version == created.db_version + 1
    stored = await store.get_active("broker_refresh", "refresh-handle")
    assert stored is not None
    assert stored.encrypted_payload == "second"
    async with session_maker() as session:
        hashes = list(await session.scalars(select(DBAccountCredential.lookup_hash)))
    assert hashes == [credential_lookup_hash("refresh-handle")]


@pytest.mark.asyncio
async def test_rotate_active_replaces_handle_and_rejects_old_handle(session_maker) -> None:
    identity = await _identity(session_maker)
    store = AccountCredentialStore(session_maker)
    created = await store.create(
        credential_type="broker_refresh",
        handle="old-refresh-handle",
        owner_account_id=identity.account_id,
        account_identity_id=identity.identity_id,
        encrypted_payload="first",
        public_metadata={"oidc_client": "confidential"},
    )

    rotated = await store.rotate_active(
        created,
        new_handle="new-refresh-handle",
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        public_metadata=created.public_metadata,
        encrypted_payload="second",
    )

    assert rotated is not None
    assert rotated.id != created.id
    assert rotated.db_version == 1
    assert rotated.encrypted_payload == "second"
    assert await store.get_active("broker_refresh", "old-refresh-handle") is None
    assert await store.get_active("broker_refresh", "new-refresh-handle") == rotated

    async with session_maker() as session:
        hashes = list(await session.scalars(select(DBAccountCredential.lookup_hash)))
    assert hashes == [credential_lookup_hash("new-refresh-handle")]


async def _service_identity(session_maker, subject: str):
    return await AccountIdentityStore(session_maker).resolve_or_materialize(
        issuer="nemo:service-account",
        subject=subject,
        subject_claim="service_account_id",
        account_type="service",
    )


async def _access_key(store, *, handle, owner, subject, bound_workspace_id=None, status="ACTIVE", expires_at=None):
    return await store.create(
        credential_type="access_key",
        handle=handle,
        owner_account_id=owner,
        subject_account_id=subject,
        status=status,
        expires_at=expires_at,
        public_metadata={"bound_workspace_id": bound_workspace_id},
    )


@pytest.mark.asyncio
async def test_bound_workspace_ids_for_subject_reports_only_live_bound_keys(session_maker) -> None:
    owner = await _identity(session_maker)
    account = await _service_identity(session_maker, "otel-collector")
    store = AccountCredentialStore(session_maker)
    await _access_key(store, handle="a", owner=owner.account_id, subject=account.account_id, bound_workspace_id="ws-a")
    # Suspended keys still count: they can be unsuspended.
    await _access_key(
        store,
        handle="b",
        owner=owner.account_id,
        subject=account.account_id,
        bound_workspace_id="ws-b",
        status="SUSPENDED",
    )
    # Revoked and expired ones can never authenticate again, so they constrain nothing.
    await _access_key(
        store,
        handle="c",
        owner=owner.account_id,
        subject=account.account_id,
        bound_workspace_id="ws-c",
        status="REVOKED",
    )
    await _access_key(
        store,
        handle="d",
        owner=owner.account_id,
        subject=account.account_id,
        bound_workspace_id="ws-d",
        expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
    )
    # An unbound key contributes nothing.
    await _access_key(store, handle="e", owner=owner.account_id, subject=account.account_id)

    assert await store.bound_workspace_ids_for_subject("access_key", account.account_id) == {"ws-a", "ws-b"}


@pytest.mark.asyncio
async def test_bound_workspace_ids_for_subject_is_empty_for_an_account_without_keys(session_maker) -> None:
    account = await _service_identity(session_maker, "unused")

    store = AccountCredentialStore(session_maker)

    assert await store.bound_workspace_ids_for_subject("access_key", account.account_id) == set()


@pytest.mark.asyncio
async def test_list_for_owner_includes_other_owners_keys_bound_to_the_callers_workspaces(session_maker) -> None:
    caller = await _identity(session_maker)
    other = await AccountIdentityStore(session_maker).resolve_or_materialize(
        issuer="https://idp.example.com", subject="user-2", subject_claim="sub", account_type="user"
    )
    account = await _service_identity(session_maker, "otel-collector")
    store = AccountCredentialStore(session_maker)
    mine = await _access_key(store, handle="mine", owner=caller.account_id, subject=caller.account_id)
    theirs = await _access_key(
        store, handle="theirs", owner=other.account_id, subject=account.account_id, bound_workspace_id="ws-a"
    )
    await _access_key(
        store, handle="elsewhere", owner=other.account_id, subject=account.account_id, bound_workspace_id="ws-z"
    )

    listed, has_more = await store.list_for_owner(
        "access_key",
        caller.account_id,
        include_service_subjects=False,
        service_subject_bound_workspace_ids=["ws-a"],
        offset=0,
        limit=10,
    )

    assert {record.id for record in listed} == {mine.id, theirs.id}
    assert has_more is False


@pytest.mark.asyncio
async def test_list_for_subject_pages_by_id_across_every_status(session_maker) -> None:
    owner = await _identity(session_maker)
    account = await _service_identity(session_maker, "otel-collector")
    store = AccountCredentialStore(session_maker)
    created = [
        await _access_key(
            store,
            handle=f"key-{index}",
            owner=owner.account_id,
            subject=account.account_id,
            status="REVOKED" if index == 1 else "ACTIVE",
        )
        for index in range(3)
    ]

    seen = []
    after_id = None
    while True:
        page, has_more = await store.list_for_subject("access_key", account.account_id, after_id=after_id, limit=2)
        seen.extend(page)
        if not has_more:
            break
        after_id = page[-1].id

    assert {record.id for record in seen} == {record.id for record in created}
    assert [record.id for record in seen] == sorted(record.id for record in created)
