# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repository policy preserves creator snapshots in the update transaction."""

import pytest
from nhx.common.api.filter import ComparisonOperation, FilterOperator, LogicalOperation
from nhx.core.entities.app.repository import SQLAlchemyEntityRepository
from nhx.core.entities.app.repository.exceptions import EntityVersionConflictError
from nhx.core.entities.app.repository.field_policy import ProtectedEntityField

pytestmark = pytest.mark.asyncio


async def test_repository_policy_preserves_snapshot_for_id_and_name_updates(
    entity_repo: SQLAlchemyEntityRepository, setup_workspaces
) -> None:
    original = {"principal_id": "user@example.com", "principal_groups": ["team-a"]}
    created = await entity_repo.create_entity(
        workspace="workspace-1", entity_type="test", name="protected", data={"_auth_context": original, "value": 1}
    )
    user_policy = ProtectedEntityField("_auth_context").bind(policy_enabled=True, can_read=False, can_write=False)
    user_repo = SQLAlchemyEntityRepository(entity_repo.session_maker, user_policy)
    supplied = {"value": 2, "_auth_context": {"principal_id": "forged"}}

    by_id = await user_repo.update_entity(entity_id=created.id, data=supplied)
    assert by_id.data == {"value": 2}
    assert supplied["_auth_context"] == {"principal_id": "forged"}
    stored = await entity_repo.get_entity_by_id(entity_id=created.id)
    assert stored is not None and stored.data["_auth_context"] == original

    by_name = await user_repo.update_entity_by_name(
        workspace="workspace-1", entity_type="test", name="protected", data={"value": 3, "_auth_context": None}
    )
    assert by_name.data == {"value": 3}
    stored = await entity_repo.get_entity_by_id(entity_id=created.id)
    assert stored is not None and stored.data["_auth_context"] == original

    with pytest.raises(EntityVersionConflictError):
        await user_repo.update_entity_by_name(
            workspace="workspace-1",
            entity_type="test",
            name="protected",
            data={"value": 4, "_auth_context": {"principal_id": "forged"}},
            expected_db_version=created.db_version,
        )
    stored = await entity_repo.get_entity_by_id(entity_id=created.id)
    assert stored is not None and stored.data == {"value": 3, "_auth_context": original}


async def test_repository_policy_preserves_absent_key_and_allows_service_replacement(
    entity_repo: SQLAlchemyEntityRepository, setup_workspaces
) -> None:
    original = {"principal_id": "user@example.com"}
    service_policy = ProtectedEntityField("_auth_context").bind(policy_enabled=True, can_read=True, can_write=True)
    service_repo = SQLAlchemyEntityRepository(entity_repo.session_maker, service_policy)
    created = await service_repo.create_entity(
        workspace="workspace-1", entity_type="test", name="without-context", data={"value": 1}
    )
    preserved_absence = await service_repo.update_entity(entity_id=created.id, data={"value": 2, "_auth_context": None})
    assert preserved_absence.data == {"value": 2}

    replaced = await service_repo.update_entity_by_name(
        workspace="workspace-1",
        entity_type="test",
        name="without-context",
        data={"value": 3, "_auth_context": original},
    )
    assert replaced.data["_auth_context"] == original
    no_wipe = await service_repo.update_entity(entity_id=created.id, data={"value": 4})
    assert no_wipe.data["_auth_context"] == original


async def test_protected_entity_field_works_for_another_key(
    entity_repo: SQLAlchemyEntityRepository, setup_workspaces
) -> None:
    field = ProtectedEntityField("private_metadata")
    user_policy = field.bind(policy_enabled=True, can_read=False, can_write=False)
    user_repo = SQLAlchemyEntityRepository(entity_repo.session_maker, user_policy)
    created = await user_repo.create_entity(
        workspace="workspace-1",
        entity_type="test",
        name="generic-field",
        data={"private_metadata": {"forged": True}, "value": 1},
    )
    assert created.data == {"value": 1}

    service_policy = field.bind(policy_enabled=True, can_read=True, can_write=True)
    service_repo = SQLAlchemyEntityRepository(entity_repo.session_maker, service_policy)
    stored = await service_repo.update_entity(
        entity_id=created.id, data={"private_metadata": {"trusted": True}, "value": 2}
    )
    assert stored.data["private_metadata"] == {"trusted": True}

    visible = await user_repo.update_entity(
        entity_id=created.id, data={"private_metadata": {"forged": True}, "value": 3}
    )
    assert visible.data == {"value": 3}
    unchanged = await service_repo.get_entity_by_id(entity_id=created.id)
    assert unchanged is not None and unchanged.data["private_metadata"] == {"trusted": True}

    protected = ComparisonOperation(operator=FilterOperator.EQ, field="data.private_metadata.trusted", value=True)
    nested = LogicalOperation(operator=FilterOperator.AND, operations=[protected])
    with pytest.raises(ValueError, match="not queryable"):
        user_policy.validate_query(nested)
    with pytest.raises(ValueError, match="not queryable"):
        user_policy.validate_query(None, sort="-data.private_metadata")
    with pytest.raises(ValueError, match="not queryable"):
        user_policy.validate_query(None, group_by="data")
    service_policy.validate_query(nested)
    field.bind(policy_enabled=False, can_read=False, can_write=False).validate_query(nested)


@pytest.mark.parametrize("key", ["", "a.b"])
async def test_protected_entity_field_requires_top_level_key(key: str) -> None:
    with pytest.raises(ValueError, match="top-level"):
        ProtectedEntityField(key)
