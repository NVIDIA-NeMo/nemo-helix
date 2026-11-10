# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from fastapi import HTTPException
from nemo_helix_plugin.iam.types import AuthzRequest
from nhx.core.auth.api.v2.authz import endpoints as authz_endpoints
from nhx.core.auth.app import access_key_workspace_limits
from nhx.core.auth.app.access_key_credentials import AccessKeyCredentialAdapter
from nhx.core.auth.app.access_key_workspace_limits import (
    AccessKeyWorkspaceLimitUnavailableError,
    resolve_access_key_workspaces,
)
from nhx.core.auth.app.account_resolution import ResolvedAccountContext
from nhx.core.auth.entities import AccessKeyEntity
from nhx.core.entities.app.repository import AccountCredentialStore, AccountIdentityStore
from nhx.core.entities.app.repository.sqlalchemy.base import Base
from nhx.core.entities.app.repository.sqlalchemy.models import DBAccessKeyWorkspace, DBWorkspace
from sqlalchemy import update
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

pytestmark = pytest.mark.asyncio

NOW = datetime.now(UTC)
CI_KEY = "ak_" + "1" * 32
OTHER_KEY = "ak_" + "2" * 32
UNLIMITED_KEY = "ak_" + "3" * 32
TEAM_A_ID = "workspace-TeamA1111111111111111111"
TEAM_B_ID = "workspace-TeamB1111111111111111111"


class _Backend:
    def __init__(self, session_maker: async_sessionmaker[AsyncSession], owner_account_id: str) -> None:
        self.session_maker = session_maker
        self.owner_account_id = owner_account_id
        self.credentials = AccessKeyCredentialAdapter(
            AccountCredentialStore(session_maker), AccountIdentityStore(session_maker)
        )

    async def add_workspace(self, workspace_id: str, name: str, *, deletion_stage: str | None = None) -> None:
        async with self.session_maker() as session:
            session.add(DBWorkspace(id=workspace_id, name=name, deletion_stage=deletion_stage))
            await session.commit()

    async def add_key(self, jti: str, service_account: str | None, workspace_ids: list[str]) -> None:
        """Store a key for ``service-account:<service_account>``, or a personal key of admin@example.com for None."""
        record = AccessKeyEntity(
            name=jti,
            workspace="system",
            principal="admin@example.com",
            subject_principal=f"service-account:{service_account}" if service_account else None,
            entity_type="SERVICE_ACCOUNT" if service_account else "USER",
            issuer="https://platform.example.com/apis/auth",
            audiences=["nemo-helix-access-key"],
            issued_at=NOW,
            expires_at=NOW + timedelta(days=30),
        )
        stored = await self.credentials.create(record, owner_account_id=self.owner_account_id)
        credential = await self.credentials.credential_store.get("access_key", stored.name)
        assert credential is not None
        async with self.session_maker() as session:
            for workspace_id in workspace_ids:
                session.add(
                    DBAccessKeyWorkspace(
                        credential_id=credential.id,
                        workspace_id=workspace_id,
                        issued_at=NOW.replace(tzinfo=None),
                    )
                )
            await session.commit()


@pytest_asyncio.fixture
async def backend(monkeypatch: pytest.MonkeyPatch):
    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    owner = await AccountIdentityStore(session_maker).resolve_or_materialize(
        issuer="https://idp.example.com",
        subject="admin",
        subject_claim="sub",
        account_type="user",
    )
    monkeypatch.setattr(access_key_workspace_limits, "get_account_session_maker", AsyncMock(return_value=session_maker))
    backend = _Backend(session_maker, owner.account_id)
    await backend.add_workspace(TEAM_A_ID, "team-a")
    await backend.add_workspace(TEAM_B_ID, "team-b")
    await backend.add_key(CI_KEY, "ci", [TEAM_A_ID])
    try:
        yield backend
    finally:
        await engine.dispose()


async def test_limited_key_resolves_to_its_workspace_names(backend: _Backend) -> None:
    assert await resolve_access_key_workspaces(
        "allow", {"principal_id": "service-account:ci", "access_key_jti": CI_KEY}
    ) == ["team-a"]


async def test_key_without_rows_is_unlimited(backend: _Backend) -> None:
    await backend.add_key(UNLIMITED_KEY, "other", [])

    assert (
        await resolve_access_key_workspaces(
            "allow", {"principal_id": "service-account:other", "access_key_jti": UNLIMITED_KEY}
        )
        is None
    )


async def test_principal_without_key_skips_the_database(monkeypatch: pytest.MonkeyPatch) -> None:
    session_maker = AsyncMock(side_effect=AssertionError("database must not be read"))
    monkeypatch.setattr(access_key_workspace_limits, "get_account_session_maker", session_maker)

    assert await resolve_access_key_workspaces("allow", {"principal_id": "alice@example.com"}) is None
    assert (
        await resolve_access_key_workspaces("allow", {"principal_id": "service:models", "access_key_jti": CI_KEY})
        is None
    )
    session_maker.assert_not_awaited()


@pytest.mark.parametrize("deletion_stage", ["pending", "deleting", "failed"])
async def test_workspace_marked_for_deletion_is_dropped(backend: _Backend, deletion_stage: str) -> None:
    async with backend.session_maker() as session:
        await session.execute(
            update(DBWorkspace).where(DBWorkspace.id == TEAM_A_ID).values(deletion_stage=deletion_stage)
        )
        await session.commit()

    assert (
        await resolve_access_key_workspaces("allow", {"principal_id": "service-account:ci", "access_key_jti": CI_KEY})
        == []
    )


async def test_recreated_workspace_with_the_same_name_is_not_allowed(backend: _Backend) -> None:
    async with backend.session_maker() as session:
        await session.execute(update(DBWorkspace).where(DBWorkspace.id == TEAM_A_ID).values(name="team-a-old"))
        await session.commit()
    await backend.add_workspace("workspace-TeamANew111111111111111", "team-a")

    assert await resolve_access_key_workspaces(
        "allow", {"principal_id": "service-account:ci", "access_key_jti": CI_KEY}
    ) == ["team-a-old"]


async def test_revoked_key_allows_nothing(backend: _Backend) -> None:
    stored = await backend.credentials.get(CI_KEY)
    assert stored is not None
    await backend.credentials.update(stored.model_copy(update={"status": "REVOKED"}))

    assert (
        await resolve_access_key_workspaces("allow", {"principal_id": "service-account:ci", "access_key_jti": CI_KEY})
        == []
    )


async def test_key_id_of_another_principal_allows_nothing(backend: _Backend) -> None:
    assert (
        await resolve_access_key_workspaces(
            "allow", {"principal_id": "service-account:other", "access_key_jti": CI_KEY}
        )
        == []
    )


async def test_delegated_call_uses_the_subject_key_limit(backend: _Backend) -> None:
    auth_input = {
        "principal_id": "service:jobs",
        "on_behalf_of_principal_id": "service-account:ci",
        "subject_access_key_jti": CI_KEY,
    }

    assert await resolve_access_key_workspaces("allow", auth_input) == ["team-a"]


async def test_actor_and_subject_limits_intersect(backend: _Backend) -> None:
    await backend.add_key(OTHER_KEY, "other", [TEAM_A_ID, TEAM_B_ID])
    auth_input = {
        "principal_id": "service-account:other",
        "access_key_jti": OTHER_KEY,
        "on_behalf_of_principal_id": "service-account:ci",
        "subject_access_key_jti": CI_KEY,
    }

    assert await resolve_access_key_workspaces("allow", auth_input) == ["team-a"]


async def test_service_account_with_limited_key_but_no_key_id_allows_nothing(backend: _Backend) -> None:
    delegated = {"principal_id": "service:jobs", "on_behalf_of_principal_id": "service-account:ci"}

    assert await resolve_access_key_workspaces("allow", delegated) == []
    assert await resolve_access_key_workspaces("allow", {"principal_id": "service-account:ci"}) == []


async def test_service_account_without_limited_key_is_unlimited(backend: _Backend) -> None:
    assert await resolve_access_key_workspaces("allow", {"principal_id": "service-account:no-keys"}) is None


async def test_database_error_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        access_key_workspace_limits,
        "get_account_session_maker",
        AsyncMock(side_effect=OperationalError("SELECT", {}, Exception("database is down"))),
    )

    with pytest.raises(AccessKeyWorkspaceLimitUnavailableError):
        await resolve_access_key_workspaces("allow", {"principal_id": "service-account:ci", "access_key_jti": CI_KEY})


def _patch_policy_endpoint(evaluate_result: dict):
    config = SimpleNamespace(bundle_cache_seconds=60)
    return (
        patch.object(authz_endpoints, "get_service_config", return_value=config),
        patch.object(
            authz_endpoints,
            "resolve_authz_account_context",
            AsyncMock(return_value=ResolvedAccountContext()),
        ),
        patch.object(authz_endpoints, "evaluate", return_value=evaluate_result),
    )


async def test_policy_endpoint_overrides_caller_supplied_limit_and_returns_it() -> None:
    config_patch, context_patch, evaluate_patch = _patch_policy_endpoint({"allowed": True})
    request = AuthzRequest(
        input={"principal_id": "service-account:ci", "access_key_jti": CI_KEY, "access_key_workspaces": None}
    )

    with (
        config_patch,
        context_patch,
        evaluate_patch as evaluate,
        patch.object(authz_endpoints, "resolve_access_key_workspaces", AsyncMock(return_value=["team-a"])),
    ):
        response = await authz_endpoints.evaluate_policy(request, "allow", entity_client=AsyncMock())

    assert evaluate.call_args.args[1]["access_key_workspaces"] == ["team-a"]
    assert response.result["access_key_workspaces"] == ["team-a"]


async def test_policy_endpoint_drops_caller_supplied_limit_when_no_key_applies() -> None:
    config_patch, context_patch, evaluate_patch = _patch_policy_endpoint({"allowed": True})
    request = AuthzRequest(input={"principal_id": "alice@example.com", "access_key_workspaces": ["team-a"]})

    with (
        config_patch,
        context_patch,
        evaluate_patch as evaluate,
        patch.object(authz_endpoints, "resolve_access_key_workspaces", AsyncMock(return_value=None)),
    ):
        response = await authz_endpoints.evaluate_policy(request, "allow", entity_client=AsyncMock())

    assert evaluate.call_args.args[1]["access_key_workspaces"] is None
    assert "access_key_workspaces" not in response.result


async def test_policy_endpoint_returns_503_when_limits_are_unavailable() -> None:
    config_patch, context_patch, evaluate_patch = _patch_policy_endpoint({"allowed": True})
    request = AuthzRequest(input={"principal_id": "service-account:ci", "access_key_jti": CI_KEY})

    with (
        config_patch,
        context_patch,
        evaluate_patch,
        patch.object(
            authz_endpoints,
            "resolve_access_key_workspaces",
            AsyncMock(side_effect=AccessKeyWorkspaceLimitUnavailableError("database is down")),
        ),
        pytest.raises(HTTPException) as exc_info,
    ):
        await authz_endpoints.evaluate_policy(request, "allow", entity_client=AsyncMock())

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Access key workspace limits are unavailable"


async def test_user_key_with_limit_rows_allows_nothing(backend: _Backend) -> None:
    await backend.add_key(OTHER_KEY, None, [TEAM_A_ID])

    assert (
        await resolve_access_key_workspaces("allow", {"principal_id": "admin@example.com", "access_key_jti": OTHER_KEY})
        == []
    )


async def test_same_identity_in_both_slots_is_looked_up_once(
    backend: _Backend, monkeypatch: pytest.MonkeyPatch
) -> None:
    lookups: list[str] = []
    original = access_key_workspace_limits.AccessKeyWorkspaceStore.workspace_ids_for_key

    async def counting(self, jti: str) -> list[str]:
        lookups.append(jti)
        return await original(self, jti)

    monkeypatch.setattr(access_key_workspace_limits.AccessKeyWorkspaceStore, "workspace_ids_for_key", counting)
    has_role_input = {
        "principal_id": "service-account:ci",
        "access_key_jti": CI_KEY,
        "on_behalf_of_principal_id": "service-account:ci",
        "subject_access_key_jti": CI_KEY,
    }

    assert await resolve_access_key_workspaces("allow", has_role_input) == ["team-a"]
    assert lookups == [CI_KEY]


async def test_revoked_limited_key_keeps_its_delegated_work_denied(backend: _Backend) -> None:
    stored = await backend.credentials.get(CI_KEY)
    assert stored is not None
    await backend.credentials.update(stored.model_copy(update={"status": "REVOKED"}))
    job_call = {"principal_id": "service:jobs", "on_behalf_of_principal_id": "service-account:ci"}

    assert await resolve_access_key_workspaces("allow", job_call) == []


async def test_role_binding_probe_ignores_key_limits(backend: _Backend) -> None:
    probe = {"principal_id": "service-account:ci", "workspace": "team-a", "role": "Viewer", "role_binding_probe": True}

    assert await resolve_access_key_workspaces("has_role", probe) is None


@pytest.mark.parametrize(
    ("entrypoint", "extra"),
    [("allow", {}), ("has_permissions", {}), ("has_role", {"access_key_jti": CI_KEY})],
    ids=["allow", "has-permissions", "has-role-with-key-id"],
)
async def test_role_binding_probe_is_ignored_outside_a_plain_role_check(
    backend: _Backend, entrypoint: str, extra: dict[str, str]
) -> None:
    probe = {"principal_id": "service-account:ci", "workspace": "team-a", "role_binding_probe": True, **extra}

    assert await resolve_access_key_workspaces(entrypoint, probe) is not None


async def test_limited_service_account_stays_limited_with_an_unlimited_key_id(backend: _Backend) -> None:
    await backend.add_key(UNLIMITED_KEY, "other", [])

    assert (
        await resolve_access_key_workspaces(
            "allow", {"principal_id": "service-account:ci", "access_key_jti": UNLIMITED_KEY}
        )
        == []
    )


@pytest.mark.parametrize("entrypoint", ["allow", "has_permissions", "has_role"])
async def test_policy_endpoint_passes_its_entrypoint_to_the_limit_lookup(entrypoint: str) -> None:
    config_patch, context_patch, evaluate_patch = _patch_policy_endpoint({"allowed": True})
    request = AuthzRequest(input={"principal_id": "service-account:ci", "role_binding_probe": True})

    with (
        config_patch,
        context_patch,
        evaluate_patch,
        patch.object(authz_endpoints, "resolve_access_key_workspaces", AsyncMock(return_value=None)) as resolve,
    ):
        await authz_endpoints.evaluate_policy(request, entrypoint, entity_client=AsyncMock())

    resolve.assert_awaited_once_with(entrypoint, request.input)


async def test_limited_service_account_cannot_use_its_own_unlimited_key(backend: _Backend) -> None:
    """A limited account is limited for good: its other keys stop working too (see security-model docs)."""
    await backend.add_key(UNLIMITED_KEY, "ci", [])

    assert (
        await resolve_access_key_workspaces(
            "allow", {"principal_id": "service-account:ci", "access_key_jti": UNLIMITED_KEY}
        )
        == []
    )
