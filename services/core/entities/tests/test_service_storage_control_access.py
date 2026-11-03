# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Storage grants cannot be used as IAM control-data permissions."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from nhx.common.auth.client import AuthClient
from nhx.common.auth.models import Principal
from nhx.common.config import AuthConfig
from nhx.core.entities.api.v2.entities.endpoints import _validate_parent_access, create_entity, get_entity_by_id
from nhx.core.entities.api.v2.entities.schemas import EntityCreateInput
from nhx.core.entities.api.v2.entity_access import (
    EntityOperation,
    require_control_entity_access,
    require_relationship_access,
)
from nhx.core.entities.app.repository import EntityRepositoryInterface, WorkspaceRepositoryInterface
from nhx.core.entities.entities import Entity
from nhx.core.entities.utils.filter import RelationshipFilterOperation
from nhx.core.entities.utils.relationships import Relationship


@pytest.fixture
def storage_auth(monkeypatch: pytest.MonkeyPatch) -> AuthClient:
    async def check_permissions(self: AuthClient, workspace: str, permissions: list[str]) -> bool:
        return all(permission.startswith("entities.") for permission in permissions)

    monkeypatch.setattr(AuthClient, "has_permissions", check_permissions)
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
    monkeypatch.setattr(AuthClient, "has_permissions", allowed)
    auth_client = AuthClient(principal=Principal(id="service:auth"), config=AuthConfig(enabled=True))
    await require_control_entity_access(auth_client, "private", "role_binding", "create")
    allowed.assert_awaited_once_with("private", ["iam.create"])
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
