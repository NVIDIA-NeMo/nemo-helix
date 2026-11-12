# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest
from nhx.core.entities.app.repository import (
    PREFERENCE_MAX_BYTES,
    AccountPreferenceStore,
    PreferenceLimitError,
    PreferenceTooLargeError,
    PreferenceVersionConflictError,
    merge_patch,
)
from nhx.core.entities.app.repository import account_preference as account_preference_module
from nhx.core.entities.app.repository.account_identity import AccountIdentityStore
from nhx.core.entities.app.repository.sqlalchemy.models import DBAccount, DBAccountPreference
from sqlalchemy import delete, select


@pytest.fixture
async def account_id(session_maker) -> str:
    record = await AccountIdentityStore(session_maker).resolve_or_materialize(
        issuer="https://idp.example.com", subject="sub-1", subject_claim="sub", account_type="user"
    )
    return record.account_id


@pytest.fixture
def store(session_maker) -> AccountPreferenceStore:
    return AccountPreferenceStore(session_maker)


def test_merge_patch_follows_rfc_7396() -> None:
    target = {"a": "b", "c": {"d": "e", "f": "g"}}
    assert merge_patch(target, {"a": "z", "c": {"f": None}}) == {"a": "z", "c": {"d": "e"}}
    assert merge_patch({"a": [1, 2]}, {"a": [3]}) == {"a": [3]}
    assert merge_patch({"a": "b"}, {"a": {"x": 1}}) == {"a": {"x": 1}}


@pytest.mark.asyncio
async def test_put_get_patch_delete_round_trip(store: AccountPreferenceStore, account_id: str) -> None:
    created = await store.put(account_id, "studio.ui", None, {"theme": "dark", "density": "compact"})
    assert created.db_version == 1

    patched = await store.patch(account_id, "studio.ui", None, {"density": None, "beta": True})
    assert patched.value == {"theme": "dark", "beta": True}
    assert patched.db_version == 2

    fetched = await store.get(account_id, "studio.ui", None)
    assert fetched == patched

    assert await store.delete(account_id, "studio.ui", None) is True
    assert await store.get(account_id, "studio.ui", None) is None
    assert await store.delete(account_id, "studio.ui", None) is False


@pytest.mark.asyncio
async def test_if_match_enforces_version(store: AccountPreferenceStore, account_id: str) -> None:
    with pytest.raises(PreferenceVersionConflictError):
        await store.put(account_id, "cli", None, {"x": 1}, if_match=1)
    created = await store.put(account_id, "cli", None, {"x": 1}, if_match=0)
    with pytest.raises(PreferenceVersionConflictError):
        await store.put(account_id, "cli", None, {"x": 2}, if_match=0)
    updated = await store.patch(account_id, "cli", None, {"x": 2}, if_match=created.db_version)
    assert updated.value == {"x": 2}
    with pytest.raises(PreferenceVersionConflictError):
        await store.delete(account_id, "cli", None, if_match=created.db_version)


@pytest.mark.asyncio
async def test_concurrent_patches_merge_without_losing_keys(store: AccountPreferenceStore, account_id: str) -> None:
    await asyncio.gather(*(store.patch(account_id, "studio.ui", None, {f"k{i}": i}) for i in range(5)))
    record = await store.get(account_id, "studio.ui", None)
    assert record is not None
    assert record.value == {f"k{i}": i for i in range(5)}


@pytest.mark.asyncio
async def test_global_and_workspace_documents_are_separate(
    store: AccountPreferenceStore, account_id: str, workspace_repo
) -> None:
    await workspace_repo.create_workspace(name="ws-1")
    await store.put(account_id, "studio.onboarding", None, {"tourSeen": True})
    await store.put(account_id, "studio.onboarding", "ws-1", {"quickstartDismissed": True})
    await store.put(account_id, "studio.ui", None, {"theme": "light"})
    await store.put(account_id, "studios", None, {"unrelated": True})

    global_docs = await store.list_documents(account_id, namespace_prefix="studio")
    assert [doc.namespace for doc in global_docs] == ["studio.onboarding", "studio.ui"]
    workspace_docs = await store.list_documents(account_id, workspace="ws-1")
    assert [(doc.namespace, doc.value) for doc in workspace_docs] == [
        ("studio.onboarding", {"quickstartDismissed": True})
    ]


@pytest.mark.asyncio
async def test_size_and_count_limits(
    store: AccountPreferenceStore, account_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(PreferenceTooLargeError):
        await store.put(account_id, "big", None, {"blob": "x" * PREFERENCE_MAX_BYTES})

    monkeypatch.setattr(account_preference_module, "PREFERENCE_MAX_DOCUMENTS", 2)
    await store.put(account_id, "one", None, {})
    await store.put(account_id, "two", None, {})
    with pytest.raises(PreferenceLimitError):
        await store.put(account_id, "three", None, {})
    # Updating an existing document is not limited by the count.
    await store.put(account_id, "two", None, {"x": 1})


@pytest.mark.asyncio
async def test_workspace_and_account_deletes_cascade(
    store: AccountPreferenceStore, account_id: str, workspace_repo, session_maker
) -> None:
    await workspace_repo.create_workspace(name="ws-1")
    await store.put(account_id, "studio.onboarding", "ws-1", {"dismissed": True})
    await store.put(account_id, "studio.ui", None, {"theme": "dark"})

    await workspace_repo.delete_workspace(name="ws-1")
    assert await store.get(account_id, "studio.onboarding", "ws-1") is None
    assert await store.get(account_id, "studio.ui", None) is not None

    async with session_maker() as session, session.begin():
        await session.execute(delete(DBAccount).where(DBAccount.id == account_id))
    async with session_maker() as session:
        assert (await session.scalars(select(DBAccountPreference))).all() == []


@pytest.mark.asyncio
async def test_put_drops_nulls_so_defaults_still_apply(store: AccountPreferenceStore, account_id: str) -> None:
    record = await store.put(account_id, "studio.ui", None, {"theme": None, "nested": {"a": None, "b": 1}})
    assert record.value == {"nested": {"b": 1}}
