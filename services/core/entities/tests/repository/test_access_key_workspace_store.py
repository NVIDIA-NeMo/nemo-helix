# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The access_key_workspaces migration and AccessKeyWorkspaceStore against a migrated SQLite database."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from nhx.core.entities.app.repository import (
    AccessKeyWorkspaceStore,
    AccountCredentialStatus,
    AccountCredentialStore,
    AccountIdentityStore,
)
from nhx.core.entities.app.repository.sqlalchemy.models import DBAccessKeyWorkspace, DBWorkspace
from sqlalchemy import create_engine, inspect, select

JTI = "ak_" + "1" * 32
TEAM_A_ID = "workspace-TeamA1111111111111111111"
ALEMBIC_INI = Path(__file__).resolve().parent.parent.parent / "alembic.ini"


def test_migration_upgrades_and_downgrades(tmp_path) -> None:
    url = f"sqlite:///{tmp_path / 'entities.db'}"
    cfg = Config(str(ALEMBIC_INI))
    cfg.set_main_option("sqlalchemy.url", url)

    command.upgrade(cfg, "head")
    engine = create_engine(url)
    tables = inspect(engine).get_table_names()
    credential_indexes = {index["name"] for index in inspect(engine).get_indexes("account_credentials")}
    assert "access_key_workspaces" in tables
    assert "idx_account_credentials_type_issued" in credential_indexes

    command.downgrade(cfg, "-1")
    assert "access_key_workspaces" not in inspect(engine).get_table_names()
    assert "idx_account_credentials_type_issued" not in {
        index["name"] for index in inspect(engine).get_indexes("account_credentials")
    }
    engine.dispose()


async def _service_account_key(
    session_maker, *, status: AccountCredentialStatus = "ACTIVE", expires_at: datetime | None = None
) -> str:
    identity = await AccountIdentityStore(session_maker).resolve_or_materialize(
        issuer="nemo:service-account",
        subject="ci",
        subject_claim="service_account_id",
        account_type="service",
    )
    credential = await AccountCredentialStore(session_maker).create(
        credential_type="access_key",
        handle=JTI,
        owner_account_id=identity.account_id,
        status=status,
        expires_at=expires_at,
        public_metadata={"jti": JTI},
    )
    return credential.id


@pytest.mark.asyncio
async def test_store_reads_limits_and_active_workspace_names(session_maker) -> None:
    credential_id = await _service_account_key(session_maker)
    async with session_maker() as session:
        session.add(DBWorkspace(id=TEAM_A_ID, name="team-a"))
        session.add(DBWorkspace(id="workspace-TeamB1111111111111111111", name="team-b", deletion_stage="pending"))
        session.add(
            DBAccessKeyWorkspace(
                credential_id=credential_id,
                workspace_id=TEAM_A_ID,
                issued_at=datetime.now(timezone.utc).replace(tzinfo=None),
            )
        )
        await session.commit()
    store = AccessKeyWorkspaceStore(session_maker)

    assert await store.workspace_ids_for_key(JTI) == [TEAM_A_ID]
    assert await store.workspace_ids_for_key("ak_" + "9" * 32) == []
    assert await store.service_account_is_limited("ci") is True
    assert await store.service_account_is_limited("other") is False
    assert await store.active_workspace_names([TEAM_A_ID, "workspace-TeamB1111111111111111111"]) == ["team-a"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "key",
    [{"status": "REVOKED"}, {"expires_at": datetime.now(timezone.utc) - timedelta(days=1)}],
    ids=["revoked", "expired"],
)
async def test_revoked_or_expired_limited_key_still_marks_the_service_account_limited(session_maker, key) -> None:
    """Jobs the key started carry no key ID, so they must stay denied after the key is gone."""
    credential_id = await _service_account_key(session_maker, **key)
    async with session_maker() as session:
        session.add(
            DBAccessKeyWorkspace(
                credential_id=credential_id,
                workspace_id=TEAM_A_ID,
                issued_at=datetime.now(timezone.utc).replace(tzinfo=None),
            )
        )
        await session.commit()

    assert await AccessKeyWorkspaceStore(session_maker).service_account_is_limited("ci") is True


@pytest.mark.asyncio
async def test_deleting_the_credential_removes_its_limit_rows(session_maker) -> None:
    credential_id = await _service_account_key(session_maker)
    async with session_maker() as session:
        session.add(
            DBAccessKeyWorkspace(
                credential_id=credential_id,
                workspace_id=TEAM_A_ID,
                issued_at=datetime.now(timezone.utc).replace(tzinfo=None),
            )
        )
        await session.commit()

    assert await AccountCredentialStore(session_maker).delete("access_key", JTI) is True
    async with session_maker() as session:
        assert list(await session.scalars(select(DBAccessKeyWorkspace))) == []
