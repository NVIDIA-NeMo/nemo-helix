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


async def _service_subject_credentials(session_maker):
    """alice owns one credential; bob owns service-subject credentials bound to various workspaces."""
    identities = AccountIdentityStore(session_maker)
    alice = await _identity(session_maker)
    bob = await identities.resolve_or_materialize(
        issuer="https://idp.example.com", subject="user-2", subject_claim="sub", account_type="user"
    )
    service = await identities.resolve_or_materialize(
        issuer="nemo:service-account", subject="team-a/svc", subject_claim="service_account_id", account_type="service"
    )
    store = AccountCredentialStore(session_maker)
    now = datetime.now(timezone.utc)
    await store.create(
        credential_type="access_key", handle="alice-own", owner_account_id=alice.account_id, issued_at=now
    )
    for index, (handle, subject_account_id, bound_ids) in enumerate(
        [
            ("svc-a", service.account_id, ["ws-a"]),
            ("svc-c", service.account_id, ["ws-c"]),
            ("svc-a-c", service.account_id, ["ws-c", "ws-a"]),
            ("svc-unbound", service.account_id, []),
            # Not a service subject, so never surfaced through the metadata match.
            ("bob-own-a", bob.account_id, ["ws-a"]),
        ]
    ):
        await store.create(
            credential_type="access_key",
            handle=handle,
            owner_account_id=bob.account_id,
            subject_account_id=subject_account_id,
            issued_at=now + timedelta(minutes=index + 1),
            public_metadata={"handle": handle, "bound_workspace_ids": bound_ids},
        )
    return store, alice


def _handles(records) -> list[str]:
    return sorted(record.public_metadata.get("handle", "alice-own") for record in records)


@pytest.mark.asyncio
async def test_list_for_owner_adds_service_subjects_whose_metadata_array_contains_a_value(session_maker) -> None:
    store, alice = await _service_subject_credentials(session_maker)

    records, has_more = await store.list_for_owner(
        "access_key",
        alice.account_id,
        include_service_subjects=False,
        service_subject_metadata_contains=("bound_workspace_ids", ["ws-a"]),
        offset=0,
        limit=100,
    )

    assert _handles(records) == ["alice-own", "svc-a", "svc-a-c"]
    assert not has_more

    records, _ = await store.list_for_owner(
        "access_key",
        alice.account_id,
        include_service_subjects=False,
        service_subject_metadata_contains=("bound_workspace_ids", ["ws-a", "ws-c"]),
        offset=0,
        limit=100,
    )
    assert _handles(records) == ["alice-own", "svc-a", "svc-a-c", "svc-c"]


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata_contains", [None, ("bound_workspace_ids", []), ("bound_workspace_ids", ["ws-x"])])
async def test_list_for_owner_without_a_metadata_match_returns_only_owned_credentials(
    session_maker, metadata_contains
) -> None:
    store, alice = await _service_subject_credentials(session_maker)

    records, _ = await store.list_for_owner(
        "access_key",
        alice.account_id,
        include_service_subjects=False,
        service_subject_metadata_contains=metadata_contains,
        offset=0,
        limit=100,
    )

    assert _handles(records) == ["alice-own"]


@pytest.mark.asyncio
async def test_list_for_owner_metadata_match_pages_with_lookahead(session_maker) -> None:
    store, alice = await _service_subject_credentials(session_maker)

    records, has_more = await store.list_for_owner(
        "access_key",
        alice.account_id,
        include_service_subjects=False,
        service_subject_metadata_contains=("bound_workspace_ids", ["ws-a"]),
        offset=0,
        limit=2,
    )

    # Newest first: svc-a-c, then svc-a; alice's own (oldest) is on the next page.
    assert [record.public_metadata["handle"] for record in records] == ["svc-a-c", "svc-a"]
    assert has_more


@pytest.mark.asyncio
async def test_list_for_owner_include_service_subjects_ignores_the_metadata_match(session_maker) -> None:
    store, alice = await _service_subject_credentials(session_maker)

    records, _ = await store.list_for_owner(
        "access_key",
        alice.account_id,
        include_service_subjects=True,
        service_subject_metadata_contains=("bound_workspace_ids", ["ws-a"]),
        offset=0,
        limit=100,
    )

    assert _handles(records) == ["alice-own", "svc-a", "svc-a-c", "svc-c", "svc-unbound"]


@pytest.mark.asyncio
async def test_list_for_subject_keyset_scan_skips_nothing_when_a_read_row_is_deleted(session_maker) -> None:
    store, _ = await _service_subject_credentials(session_maker)
    service = await AccountIdentityStore(session_maker).get_active_identity(
        issuer="nemo:service-account", subject="team-a/svc"
    )
    assert service is not None

    first_page, has_more = await store.list_for_subject("access_key", service.account_id, limit=1)
    assert has_more
    seen = [record.public_metadata["handle"] for record in first_page]
    # A concurrent delete of a row already read must not shift an unread row onto that page.
    await store.delete("access_key", first_page[0].public_metadata["handle"])
    after_id = first_page[-1].id
    while has_more:
        page, has_more = await store.list_for_subject("access_key", service.account_id, after_id=after_id, limit=1)
        seen.extend(record.public_metadata["handle"] for record in page)
        after_id = page[-1].id

    assert sorted(seen) == ["svc-a", "svc-a-c", "svc-c", "svc-unbound"]
