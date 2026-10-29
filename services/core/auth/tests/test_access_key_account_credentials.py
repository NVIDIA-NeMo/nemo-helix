# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from nemo_helix_plugin.auth.access_keys.types import AccessKeyCreateResponse
from nhx.common.entities import EntityNotFoundError
from nhx.core.auth.app.access_key_credentials import AccessKeyCredentialAdapter
from nhx.core.auth.app.access_keys import AccessKeyRegistry
from nhx.core.auth.entities import AccessKeyEntity
from nhx.core.entities.app.repository import AccountCredentialStore, AccountIdentityStore
from nhx.core.entities.app.repository.sqlalchemy.base import Base
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

NOW = datetime.now()


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


def _service_key(jti: str = "ak_service") -> AccessKeyCreateResponse:
    return _key(jti).model_copy(
        update={"principal": "service-account:otel-collector", "entity_type": "SERVICE_ACCOUNT"}
    )


@pytest.mark.asyncio
async def test_bound_workspace_id_round_trips_through_the_credential_store(credential_backend) -> None:
    adapter, owner_account_id = credential_backend
    registry = AccessKeyRegistry(AsyncMock(), adapter)

    await registry.add(
        _service_key(),
        owner_principal="alice",
        owner_account_id=owner_account_id,
        bound_workspace_id="workspace-aaa",
    )

    stored = await adapter.get("ak_service")
    assert stored is not None
    assert stored.bound_workspace_id == "workspace-aaa"


@pytest.mark.asyncio
async def test_a_key_stored_before_binding_existed_reads_as_unbound(credential_backend) -> None:
    adapter, owner_account_id = credential_backend
    registry = AccessKeyRegistry(AsyncMock(), adapter)
    await registry.add(_service_key(), owner_principal="alice", owner_account_id=owner_account_id)

    # Rows written before this field existed have no such key at all; reading one must not fail.
    record = await adapter.credential_store.get("access_key", "ak_service")
    assert record is not None
    metadata = dict(record.public_metadata)
    metadata.pop("bound_workspace_id")
    await adapter.credential_store.update(replace(record, public_metadata=metadata))

    stored = await adapter.get("ak_service")
    assert stored is not None
    assert stored.bound_workspace_id is None


@pytest.mark.asyncio
async def test_service_account_keys_returns_every_key_for_the_account(credential_backend) -> None:
    adapter, owner_account_id = credential_backend
    registry = AccessKeyRegistry(AsyncMock(), adapter)
    await registry.add(
        _service_key("ak_one"),
        owner_principal="alice",
        owner_account_id=owner_account_id,
        bound_workspace_id="workspace-aaa",
    )
    await registry.add(_service_key("ak_two"), owner_principal="bob", owner_account_id=owner_account_id)

    identity, keys = await adapter.service_account_keys("service-account:otel-collector")

    assert identity is not None
    assert {key.name: key.bound_workspace_id for key in keys} == {"ak_one": "workspace-aaa", "ak_two": None}


@pytest.mark.asyncio
async def test_service_account_keys_is_empty_for_an_account_never_issued_one(credential_backend) -> None:
    adapter, _owner_account_id = credential_backend

    assert await adapter.service_account_keys("service-account:unknown") == (None, [])
    assert await adapter.is_service_account_disabled("service-account:unknown") is False
