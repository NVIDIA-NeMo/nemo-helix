# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Storage grants cannot be used as IAM control-data permissions."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from nemo_helix_plugin.api.pagination import PageParams
from nhx.common.api.filter import ComparisonOperation, FilterOperator
from nhx.common.auth import ALL_WORKSPACES
from nhx.common.auth.client import AuthClient
from nhx.common.auth.models import Principal
from nhx.common.config import AuthConfig
from nhx.core.entities.api.v2.entities import endpoints
from nhx.core.entities.api.v2.entities.endpoints import _validate_parent_access, create_entity, get_entity_by_id
from nhx.core.entities.api.v2.entities.schemas import EntityCreateInput
from nhx.core.entities.api.v2.entity_access import (
    EntityOperation,
    get_control_entity_list_workspaces,
    require_control_entity_access,
    require_relationship_access,
)
from nhx.core.entities.app.repository import EntityRepositoryInterface, WorkspaceRepositoryInterface
from nhx.core.entities.entities import Entity, Workspace
from nhx.core.entities.utils.filter import RelationshipFilterOperation
from nhx.core.entities.utils.relationships import Relationship


@pytest.fixture
def storage_auth(monkeypatch: pytest.MonkeyPatch) -> AuthClient:
    async def check_permissions(
        self: AuthClient, workspace: str, *, actor: list[str] | None = None, subject: list[str] | None = None
    ) -> bool:
        return subject is None and actor is not None and all(permission.startswith("entities.") for permission in actor)

    monkeypatch.setattr(AuthClient, "check_permissions", check_permissions)
    return AuthClient(principal=Principal(id="service:scaled-evals"), config=AuthConfig(enabled=True))


def control_entity() -> Entity:
    now = datetime.now(timezone.utc)
    return Entity(
        id="role-binding-test",
        name="binding",
        workspace="private",
        entity_type="role_binding",
        data={"role": "Admin"},
        created_at=now,
        updated_at=now,
        db_version=1,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("entity_type", ["role_binding", "access_key"])
@pytest.mark.parametrize("operation", ["create", "read", "list", "update", "delete"])
async def test_storage_only_grants_cannot_access_control_entities(
    storage_auth: AuthClient, entity_type: str, operation: EntityOperation
) -> None:
    with pytest.raises(HTTPException) as caught:
        await require_control_entity_access(storage_auth, "private", entity_type, operation)
    assert caught.value.status_code == 403


@pytest.mark.asyncio
async def test_operational_entities_need_no_iam_grant(storage_auth: AuthClient) -> None:
    await require_control_entity_access(storage_auth, "private", "scaled_evals_task", "create")


@pytest.mark.asyncio
async def test_explicit_iam_grants_and_auth_disabled_access_remain_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    allowed = AsyncMock(return_value=True)
    monkeypatch.setattr(AuthClient, "check_permissions", allowed)
    auth_client = AuthClient(principal=Principal(id="service:auth"), config=AuthConfig(enabled=True))
    await require_control_entity_access(auth_client, "private", "role_binding", "create")
    allowed.assert_awaited_once_with("private", actor=["iam.create"])
    allowed.reset_mock()
    disabled = AuthClient(principal=Principal(id="user"), config=AuthConfig(enabled=False))
    await require_control_entity_access(disabled, "private", "role_binding", "create")
    allowed.assert_not_called()


@pytest.mark.asyncio
async def test_control_create_is_denied_before_storage_mutation(storage_auth: AuthClient) -> None:
    repository = AsyncMock(spec=EntityRepositoryInterface)
    with pytest.raises(HTTPException) as caught:
        await create_entity(
            "private",
            "role_binding",
            EntityCreateInput(name="binding", data={"role": "Admin"}),
            repository,
            AsyncMock(spec=WorkspaceRepositoryInterface),
            storage_auth,
        )
    assert caught.value.status_code == 403
    repository.create_entity.assert_not_called()


@pytest.mark.asyncio
async def test_id_lookup_checks_authoritative_stored_control_type(storage_auth: AuthClient) -> None:
    repository = AsyncMock(spec=EntityRepositoryInterface)
    repository.get_entity_by_id.return_value = control_entity()
    with pytest.raises(HTTPException) as caught:
        await get_entity_by_id(
            "role-binding-test", repository, AsyncMock(spec=WorkspaceRepositoryInterface), storage_auth
        )
    assert caught.value.status_code == 403


@pytest.mark.asyncio
async def test_control_parent_cannot_be_referenced_by_storage_only_callers(storage_auth: AuthClient) -> None:
    repository = AsyncMock(spec=EntityRepositoryInterface)
    repository.get_entity_by_id.return_value = control_entity()
    with pytest.raises(HTTPException) as caught:
        await _validate_parent_access(None, repository, "role-binding-test", storage_auth)
    assert caught.value.status_code == 403


@pytest.mark.asyncio
async def test_control_relationship_filters_require_iam_access(storage_auth: AuthClient) -> None:
    operation = RelationshipFilterOperation(
        relationship_name="bindings",
        relationship=Relationship(kind="one_to_many", target_entity_type="role_binding", via="parent"),
    )
    with pytest.raises(HTTPException) as caught:
        await require_relationship_access(storage_auth, operation)
    assert caught.value.status_code == 403


@pytest.mark.asyncio
@pytest.mark.parametrize("entity_type", ["role_binding", "access_key"])
@pytest.mark.parametrize("delegated", [False, True])
@pytest.mark.parametrize("permitted", [False, True])
async def test_wildcard_control_lists_filter_rows_totals_and_counts_by_iam_permission(
    monkeypatch: pytest.MonkeyPatch, entity_type: str, delegated: bool, permitted: bool
) -> None:
    """Membership and a powerful actor must not replace the subject's IAM grant."""
    auth_client = AuthClient(
        principal=Principal(
            id="service:auth" if delegated else "system-admin", on_behalf_of="system-admin" if delegated else None
        ),
        config=AuthConfig(enabled=True),
    )
    checked: list[set[str]] = []

    async def filter_workspaces(
        self: AuthClient, workspaces: set[str], *, actor: list[str] | None = None, subject: list[str] | None = None
    ) -> set[str]:
        assert actor == (None if delegated else ["iam.list"])
        assert subject == (["iam.list"] if delegated else None)
        assert ALL_WORKSPACES not in workspaces
        checked.append(workspaces)
        return workspaces & {"system"} if permitted else set()

    monkeypatch.setattr(AuthClient, "filter_workspaces", filter_workspaces)
    monkeypatch.setattr(endpoints, "get_accessible_workspaces", AsyncMock(return_value={"system", "viewer-only"}))
    repository = AsyncMock(spec=EntityRepositoryInterface)
    repository.list_entities.return_value = ([], 0)
    repository.count_entities_by.return_value = {}
    workspace_repository = AsyncMock(spec=WorkspaceRepositoryInterface)
    user_filter = ComparisonOperation(operator=FilterOperator.EQ, field="data.role", value="Viewer")

    result = await endpoints.list_entities(
        ALL_WORKSPACES,
        entity_type,
        repository,
        workspace_repository,
        user_filter,
        auth_client,
        PageParams(page=2, page_size=3),
        sort="name",
        count_by="data.role",
    )

    assert checked == [{"system", "viewer-only"}]
    listing = repository.list_entities.call_args.kwargs
    counting = repository.count_entities_by.call_args.kwargs
    allowed = {"system"} if permitted else set()
    assert listing["relationship_child_workspaces"] == counting["relationship_child_workspaces"] == allowed
    assert listing["filter_op"] is counting["filter_op"]
    assert listing["filter_op"].to_dict() == {"$and": [{"workspace": {"$in": list(allowed)}}, user_filter.to_dict()]}
    assert listing["page"] == 2 and listing["page_size"] == 3
    assert result.pagination.total_results == 0
    assert result.group_counts == {}
    workspace_repository.list_workspaces.assert_not_awaited()


@pytest.mark.asyncio
async def test_unscoped_control_lists_check_every_page_of_real_workspaces(monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime.now(timezone.utc)
    workspaces = [Workspace(id=str(i), name=f"workspace-{i}", created_at=now, updated_at=now) for i in range(1001)]
    repository = AsyncMock(spec=WorkspaceRepositoryInterface)
    repository.list_workspaces.side_effect = [(workspaces[:1000], 1001), (workspaces[1000:], 1001)]
    permission = AsyncMock(return_value={"workspace-1000"})
    monkeypatch.setattr(AuthClient, "filter_workspaces", permission)
    auth_client = AuthClient(principal=Principal(id="service:limited"), config=AuthConfig(enabled=True))

    allowed = await get_control_entity_list_workspaces(auth_client, repository, "role_binding", None)

    assert allowed == {"workspace-1000"}
    assert repository.list_workspaces.await_count == 2
    repository.list_workspaces.assert_any_await(page=2, page_size=1000, sort="name")
    permission.assert_awaited_once_with({workspace.name for workspace in workspaces}, actor=["iam.list"])


@pytest.mark.asyncio
@pytest.mark.parametrize(("enabled", "entity_type"), [(False, "role_binding"), (True, "model")])
async def test_control_list_filter_preserves_auth_off_and_operational_queries(
    monkeypatch: pytest.MonkeyPatch, enabled: bool, entity_type: str
) -> None:
    permission = AsyncMock()
    monkeypatch.setattr(AuthClient, "filter_workspaces", permission)
    repository = AsyncMock(spec=WorkspaceRepositoryInterface)
    auth_client = AuthClient(principal=Principal(id="user"), config=AuthConfig(enabled=enabled))
    assert await get_control_entity_list_workspaces(auth_client, repository, entity_type, None) is None
    permission.assert_not_awaited()
    repository.list_workspaces.assert_not_awaited()


@pytest.mark.asyncio
async def test_control_list_permission_failure_does_not_query_rows_or_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(endpoints, "get_accessible_workspaces", AsyncMock(return_value={"system"}))
    monkeypatch.setattr(AuthClient, "filter_workspaces", AsyncMock(side_effect=HTTPException(503, "PDP unavailable")))
    repository = AsyncMock(spec=EntityRepositoryInterface)
    auth_client = AuthClient(
        principal=Principal(id="service:auth", on_behalf_of="system-admin"), config=AuthConfig(enabled=True)
    )
    with pytest.raises(HTTPException) as denied:
        await endpoints.list_entities(
            ALL_WORKSPACES,
            "role_binding",
            repository,
            AsyncMock(spec=WorkspaceRepositoryInterface),
            None,
            auth_client,
            PageParams(page=1, page_size=10),
            sort="name",
            count_by="data.role",
        )
    assert denied.value.status_code == 503
    repository.list_entities.assert_not_awaited()
    repository.count_entities_by.assert_not_awaited()
