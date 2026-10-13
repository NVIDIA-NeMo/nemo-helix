# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
