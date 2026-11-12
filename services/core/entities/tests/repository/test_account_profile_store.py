# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timedelta

import pytest
from nhx.core.entities.app.repository import AccountProfileStore, IdentityClaims
from nhx.core.entities.app.repository.account_identity import AccountIdentityStore
from nhx.core.entities.app.repository.sqlalchemy.models import DBAccount, DBAccountIdentity, DBAccountProfile
from sqlalchemy import delete, select, update

ISSUER = "https://idp.example.com"
OTHER_ISSUER = "https://other-idp.example.com"


async def _materialize(store: AccountIdentityStore, subject: str, issuer: str = ISSUER, **kwargs):
    return await store.resolve_or_materialize(
        issuer=issuer, subject=subject, subject_claim="sub", account_type="user", **kwargs
    )


@pytest.mark.asyncio
async def test_first_sync_creates_profile_with_primary_identity_and_idp_fields(session_maker):
    record = await _materialize(
        AccountIdentityStore(session_maker),
        "sub-1",
        primary_email="alice@example.com",
        claims_snapshot={"sub": "sub-1", "email": "alice@example.com", "phone_number": "555"},
        profile_claims=IdentityClaims(
            email="alice@example.com",
            name="Alice Anderson",
            picture="https://cdn.example.com/a.png",
            snapshot={"sub": "sub-1", "name": "Alice Anderson", "address": "secret"},
            login=True,
        ),
    )

    profile = await AccountProfileStore(session_maker).get(record.account_id)
    assert profile is not None
    assert profile.idp_display_name == "Alice Anderson"
    assert profile.avatar_url == "https://cdn.example.com/a.png"
    assert profile.last_login_at is not None
    assert profile.display_name() == "Alice Anderson"
    async with session_maker() as session:
        db_profile = await session.get(DBAccountProfile, record.account_id)
        identity = await session.get(DBAccountIdentity, record.identity_id)
    assert db_profile is not None and db_profile.primary_identity_id == record.identity_id
    assert identity is not None
    assert identity.claims_snapshot == {"sub": "sub-1", "name": "Alice Anderson"}
    assert identity.last_seen_at is not None


@pytest.mark.asyncio
async def test_existing_identity_refreshes_idp_fields_and_email(session_maker):
    store = AccountIdentityStore(session_maker)
    record = await _materialize(store, "sub-1", primary_email="old@example.com")

    await _materialize(
        store, "sub-1", profile_claims=IdentityClaims(email="new@example.com", name="Renamed", picture="http://x")
    )

    profile = await AccountProfileStore(session_maker).get(record.account_id)
    assert profile is not None
    assert profile.primary_email == "new@example.com"
    assert profile.idp_display_name == "Renamed"
    assert profile.avatar_url is None  # non-https avatars are dropped


@pytest.mark.asyncio
async def test_sync_is_throttled_by_last_seen(session_maker):
    store = AccountIdentityStore(session_maker)
    record = await _materialize(store, "sub-1", profile_claims=IdentityClaims(name="First"))

    await _materialize(
        store, "sub-1", profile_claims=IdentityClaims(name="Second"), profile_sync_interval=timedelta(minutes=15)
    )
    profile = await AccountProfileStore(session_maker).get(record.account_id)
    assert profile is not None and profile.idp_display_name == "First"

    async with session_maker() as session, session.begin():
        await session.execute(update(DBAccountIdentity).values(last_seen_at=datetime(2000, 1, 1)))
    await _materialize(
        store, "sub-1", profile_claims=IdentityClaims(name="Third"), profile_sync_interval=timedelta(minutes=15)
    )
    profile = await AccountProfileStore(session_maker).get(record.account_id)
    assert profile is not None and profile.idp_display_name == "Third"


@pytest.mark.asyncio
async def test_only_primary_identity_feeds_idp_fields(session_maker):
    store = AccountIdentityStore(session_maker)
    first = await _materialize(store, "sub-1", link_key="alice", profile_claims=IdentityClaims(name="From IdP A"))
    second = await _materialize(
        store,
        "sub-1",
        issuer=OTHER_ISSUER,
        link_key="alice",
        profile_claims=IdentityClaims(name="From IdP B", email="b@example.com"),
    )
    assert second.account_id == first.account_id

    profiles = AccountProfileStore(session_maker)
    profile = await profiles.get(first.account_id)
    assert profile is not None
    assert profile.idp_display_name == "From IdP A"
    assert profile.primary_email is None

    # Unlinking the primary hands primary status to the next oldest identity.
    async with session_maker() as session, session.begin():
        await session.execute(delete(DBAccountIdentity).where(DBAccountIdentity.id == first.identity_id))
    await profiles.sync_identity(second.identity_id, IdentityClaims(name="From IdP B"))
    profile = await profiles.get(first.account_id)
    assert profile is not None and profile.idp_display_name == "From IdP B"


@pytest.mark.asyncio
async def test_update_sets_and_clears_override(session_maker):
    record = await _materialize(AccountIdentityStore(session_maker), "sub-1", profile_claims=IdentityClaims(name="IdP"))
    profiles = AccountProfileStore(session_maker)

    updated = await profiles.update(record.account_id, {"display_name": "Override", "timezone": "Europe/Paris"})
    assert updated.display_name() == "Override"
    assert updated.timezone == "Europe/Paris"

    cleared = await profiles.update(record.account_id, {"display_name": None})
    assert cleared.display_name() == "IdP"
    assert cleared.timezone == "Europe/Paris"


@pytest.mark.asyncio
async def test_resolve_matches_subject_or_email_and_flags_ambiguity(session_maker):
    store = AccountIdentityStore(session_maker)
    alice = await _materialize(
        store, "alice-sub", primary_email="alice@example.com", profile_claims=IdentityClaims(name="Alice")
    )
    await AccountProfileStore(session_maker).update(alice.account_id, {"display_name": "Al"})
    await _materialize(store, "shared-sub")
    await _materialize(store, "shared-sub", issuer=OTHER_ISSUER)
    await store.resolve_or_materialize(
        issuer="nemo:service", subject="jobs", subject_claim="service", account_type="service"
    )
    async with session_maker() as session, session.begin():
        await session.execute(update(DBAccount).where(DBAccount.id == alice.account_id).values(status="deactivated"))

    results = await AccountProfileStore(session_maker).resolve(
        ["alice-sub", "alice@example.com", "shared-sub", "jobs", "*", "unknown"]
    )

    by_principal = {result.principal: result for result in results}
    assert by_principal["alice-sub"].display_name == "Al"
    assert by_principal["alice-sub"].idp_display_name == "Alice"
    assert by_principal["alice-sub"].display_name_is_override is True
    assert by_principal["alice-sub"].status == "deactivated"
    assert by_principal["alice@example.com"].account_id == alice.account_id
    assert by_principal["shared-sub"].ambiguous is True
    assert by_principal["shared-sub"].display_name is None
    for unresolved in ("jobs", "*", "unknown"):
        assert by_principal[unresolved].account_id is None


@pytest.mark.asyncio
async def test_resolve_never_reveals_email_for_other_principals(session_maker):
    await _materialize(
        AccountIdentityStore(session_maker), "bob-sub", primary_email="bob@example.com", display_name="bob@example.com"
    )

    results = await AccountProfileStore(session_maker).resolve(["bob-sub", "bob@example.com"])

    by_principal = {result.principal: result for result in results}
    assert by_principal["bob-sub"].account_id is not None
    assert by_principal["bob-sub"].display_name is None
    assert by_principal["bob@example.com"].display_name == "bob@example.com"


@pytest.mark.asyncio
async def test_account_delete_cascades_to_profile(session_maker):
    record = await _materialize(AccountIdentityStore(session_maker), "sub-1", profile_claims=IdentityClaims(name="A"))
    async with session_maker() as session, session.begin():
        await session.execute(delete(DBAccount).where(DBAccount.id == record.account_id))
    async with session_maker() as session:
        assert (await session.scalars(select(DBAccountProfile))).all() == []
