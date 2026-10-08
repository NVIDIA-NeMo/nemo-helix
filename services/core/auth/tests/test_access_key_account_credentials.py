# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from nemo_helix_plugin.auth.access_keys.types import AccessKeyCreateResponse
from nhx.common.entities import EntityNotFoundError
from nhx.core.auth.app.access_key_credentials import AccessKeyCredentialAdapter
from nhx.core.auth.app.access_keys import AccessKeyRegistry
from nhx.core.auth.entities import AccessKeyEntity
from nhx.core.entities.app.repository import (
    AccountCredentialStore,
    AccountIdentityStore,
    AccountIdentityUnavailableError,
)
from nhx.core.entities.app.repository.sqlalchemy.base import Base
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

NOW = datetime(2026, 9, 29, tzinfo=UTC)


@pytest_asyncio.fixture
async def credential_backend():
    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    identity_store = AccountIdentityStore(session_maker)
    owner = await identity_store.resolve_or_materialize(
        issuer="https://idp.example.com",
        subject="alice",
        subject_claim="sub",
        account_type="user",
    )
    adapter = AccessKeyCredentialAdapter(AccountCredentialStore(session_maker), identity_store)
    try:
        yield adapter, owner.account_id
    finally:
        await engine.dispose()


def _key(jti: str = "ak_new") -> AccessKeyCreateResponse:
    return AccessKeyCreateResponse(
        jti=jti,
        name="new-key",
        principal="alice",
        created_at=NOW,
        expires_at=NOW + timedelta(days=30),
        description="stored in account credentials",
        status="ACTIVE",
        issuer="https://platform.example.com/apis/auth",
        audiences=["nemo-helix-access-key"],
        scope=["entities"],
        token="signed-token",
        token_type="Bearer",
    )


@pytest.mark.asyncio
async def test_new_access_key_writes_only_account_credentials(credential_backend) -> None:
    adapter, owner_account_id = credential_backend
    legacy_client = AsyncMock()
    registry = AccessKeyRegistry(legacy_client, adapter)

    await registry.add(_key(), owner_principal="alice", owner_account_id=owner_account_id)

    legacy_client.create.assert_not_awaited()
    assert await registry.is_active("ak_new", "alice") is True
    assert await registry.revoke("ak_new", "alice") is True
    assert await registry.is_active("ak_new", "alice") is False


@pytest.mark.asyncio
async def test_existing_legacy_access_key_remains_valid(credential_backend) -> None:
    adapter, _owner_account_id = credential_backend
    legacy = AccessKeyEntity(
        name="ak_legacy",
        workspace="system",
        principal="alice",
        issued_at=NOW,
        expires_at=datetime.now(tz=UTC) + timedelta(days=1),
        issuer="https://platform.example.com/apis/auth",
        audiences=["nemo-helix-access-key"],
        last_used_at=datetime.now(tz=UTC),
    )
    legacy_client = AsyncMock()
    legacy_client.get.return_value = legacy
    registry = AccessKeyRegistry(legacy_client, adapter)

    assert await registry.is_active("ak_legacy", "alice") is True
    legacy_client.get.assert_awaited()


@pytest.mark.asyncio
async def test_missing_new_and_legacy_records_fail_closed(credential_backend) -> None:
    adapter, _owner_account_id = credential_backend
    legacy_client = AsyncMock()
    legacy_client.get.side_effect = EntityNotFoundError("missing")
    registry = AccessKeyRegistry(legacy_client, adapter)

    assert await registry.is_active("ak_missing", "alice") is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("bound_workspace", "expected"),
    [("team-a", False), (None, True), ("team-b", True)],
)
async def test_credential_store_service_key_counts_as_access_outside_workspace(
    credential_backend, bound_workspace: str | None, expected: bool
) -> None:
    adapter, owner_account_id = credential_backend
    entity_client = AsyncMock()
    registry = AccessKeyRegistry(entity_client, adapter)
    service_key = _key("ak_service").model_copy(
        update={"principal": "service-account:otel-collector", "entity_type": "SERVICE_ACCOUNT"}
    )
    await registry.add(
        service_key,
        owner_principal="alice",
        owner_account_id=owner_account_id,
        bound_workspace=bound_workspace,
    )
    entity_client.list.return_value = SimpleNamespace(data=[], pagination=SimpleNamespace(total_pages=1))

    assert (
        await registry.has_service_account_access_outside_workspace("service-account:otel-collector", "team-a")
        is expected
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("workspace_id", "expected"), [("ws-old", False), ("ws-new", True), (None, False)])
async def test_credential_store_key_bound_to_an_earlier_workspace_of_the_same_name_counts_as_outside(
    credential_backend, workspace_id: str | None, expected: bool
) -> None:
    adapter, owner_account_id = credential_backend
    entity_client = AsyncMock()
    entity_client.list.return_value = SimpleNamespace(data=[], pagination=SimpleNamespace(total_pages=1))
    registry = AccessKeyRegistry(entity_client, adapter)
    service_key = _key("ak_service").model_copy(
        update={"principal": "service-account:otel-collector", "entity_type": "SERVICE_ACCOUNT"}
    )
    await registry.add(
        service_key,
        owner_principal="alice",
        owner_account_id=owner_account_id,
        bound_workspace="team-a",
        bound_workspace_id="ws-old",
    )

    assert (
        await registry.has_service_account_access_outside_workspace(
            "service-account:otel-collector", "team-a", workspace_id
        )
        is expected
    )


@pytest.mark.asyncio
async def test_credential_store_check_includes_revoked_keys_and_ignores_unknown_accounts(credential_backend) -> None:
    adapter, owner_account_id = credential_backend
    entity_client = AsyncMock()
    entity_client.list.return_value = SimpleNamespace(data=[], pagination=SimpleNamespace(total_pages=1))
    registry = AccessKeyRegistry(entity_client, adapter)
    assert not await registry.has_service_account_access_outside_workspace("service-account:never-issued", "team-a")

    service_key = _key("ak_service").model_copy(
        update={"principal": "service-account:otel-collector", "entity_type": "SERVICE_ACCOUNT"}
    )
    await registry.add(service_key, owner_principal="alice", owner_account_id=owner_account_id)
    await adapter.credential_store.revoke("access_key", "ak_service")

    assert await registry.has_service_account_access_outside_workspace("service-account:otel-collector", "team-a")


async def _add_team_b_service_keys(registry: AccessKeyRegistry, adapter: AccessKeyCredentialAdapter) -> None:
    """Three service-bound keys of another tenant, all newer than the keys the tests list."""
    bob = await adapter.identity_store.resolve_or_materialize(
        issuer="https://idp.example.com", subject="bob", subject_claim="sub", account_type="user"
    )
    for index in range(3):
        await registry.add(
            _key(f"ak_team_b_{index}").model_copy(
                update={
                    "principal": f"service-account:team-b/svc{index}",
                    "entity_type": "SERVICE_ACCOUNT",
                    "created_at": NOW + timedelta(hours=index + 1),
                }
            ),
            owner_principal="bob",
            owner_account_id=bob.account_id,
            bound_workspace="team-b",
            bound_workspace_id="ws-b",
        )


@pytest.mark.asyncio
async def test_credential_store_listing_does_not_page_other_tenants_service_keys_through_a_non_admin(
    credential_backend,
) -> None:
    adapter, alice_account_id = credential_backend
    entity_client = AsyncMock()
    entity_client.list.return_value = SimpleNamespace(data=[], pagination=SimpleNamespace(total_pages=1))
    registry = AccessKeyRegistry(entity_client, adapter)
    # alice's key is the oldest row, so other tenants' newer keys would crowd it off page 1.
    await registry.add(_key("ak_alice"), owner_principal="alice", owner_account_id=alice_account_id)
    await _add_team_b_service_keys(registry, adapter)
    admin_override = AsyncMock(return_value=False)

    listed = await registry.list_for_principal(
        "alice", page=1, page_size=2, owner_account_id=alice_account_id, admin_override=admin_override
    )

    assert [key.jti for key in listed.data] == ["ak_alice"]
    assert not listed.has_more
    # Nothing of team-b's was fetched, so nothing needed an admin check.
    admin_override.assert_not_awaited()


@pytest.mark.asyncio
async def test_credential_store_listing_keeps_service_keys_the_caller_created_for_workspaces_they_administer(
    credential_backend,
) -> None:
    adapter, alice_account_id = credential_backend
    entity_client = AsyncMock()
    entity_client.list.return_value = SimpleNamespace(data=[], pagination=SimpleNamespace(total_pages=1))
    registry = AccessKeyRegistry(entity_client, adapter)
    await registry.add(_key("ak_alice"), owner_principal="alice", owner_account_id=alice_account_id)
    for jti, workspace in (("ak_team_a", "team-a"), ("ak_team_c", "team-c")):
        await registry.add(
            _key(jti).model_copy(
                update={"principal": f"service-account:{workspace}/svc", "entity_type": "SERVICE_ACCOUNT"}
            ),
            owner_principal="alice",
            owner_account_id=alice_account_id,
            bound_workspace=workspace,
            bound_workspace_id=f"ws-{workspace}",
        )
    await _add_team_b_service_keys(registry, adapter)

    # alice administers team-a but no longer team-c.
    listed = await registry.list_for_principal(
        "alice",
        page=1,
        page_size=100,
        owner_account_id=alice_account_id,
        admin_override=AsyncMock(side_effect=lambda workspace, workspace_id=None: workspace == "team-a"),
    )

    assert sorted(key.jti for key in listed.data) == ["ak_alice", "ak_team_a"]


@pytest.mark.asyncio
async def test_credential_store_listing_for_a_helix_admin_includes_other_tenants_service_keys(
    credential_backend,
) -> None:
    adapter, alice_account_id = credential_backend
    entity_client = AsyncMock()
    entity_client.list.return_value = SimpleNamespace(data=[], pagination=SimpleNamespace(total_pages=1))
    registry = AccessKeyRegistry(entity_client, adapter)
    await registry.add(_key("ak_alice"), owner_principal="alice", owner_account_id=alice_account_id)
    await _add_team_b_service_keys(registry, adapter)

    listed = await registry.list_for_principal(
        "alice", page=1, page_size=100, include_service_accounts=True, owner_account_id=alice_account_id
    )

    assert sorted(key.jti for key in listed.data) == [
        "ak_alice",
        "ak_team_b_0",
        "ak_team_b_1",
        "ak_team_b_2",
    ]


@pytest.mark.asyncio
async def test_any_for_service_account_is_false_for_an_account_that_was_never_issued_a_key(credential_backend) -> None:
    adapter, _ = credential_backend

    assert await adapter.any_for_service_account("service-account:never-issued", lambda _record: True) is False


@pytest.mark.asyncio
async def test_any_for_service_account_fails_closed_for_a_disabled_identity(credential_backend) -> None:
    adapter, _ = credential_backend
    adapter.identity_store.get_active_identity = AsyncMock(side_effect=AccountIdentityUnavailableError("disabled"))

    assert await adapter.any_for_service_account("service-account:team-a/otel", lambda _record: False) is True
