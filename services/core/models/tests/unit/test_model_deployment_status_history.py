# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ModelDeployment status history (entity-to-schema conversion and update_deployment_status)."""

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from nhx.common.entities.client import EntityClient
from nhx.core.models.api.service.model_deployment_service import (
    ModelDeploymentService,
    _compact_adjacent_status_history,
    _entity_to_schema,
)
from nhx.core.models.entities import ModelDeployment as ModelDeploymentEntity
from nhx.core.models.schemas import (
    ModelDeploymentStatus,
    UpdateModelDeploymentStatusRequest,
)


def _create_deployment_entity(
    entity_id: str = "deployment-id-123",
    created_at: datetime | None = None,
    updated_at: datetime | None = None,
    **kwargs: Any,
) -> ModelDeploymentEntity:
    """Helper to create ModelDeploymentEntity with proper private attributes."""
    entity = ModelDeploymentEntity(**kwargs)
    entity._id = entity_id
    entity._created_at = created_at or datetime.now(timezone.utc)
    entity._updated_at = updated_at or datetime.now(timezone.utc)
    return entity


def _deployment_entity(
    *,
    status: ModelDeploymentStatus = ModelDeploymentStatus.CREATED,
    status_message: str = "",
    status_history: list[dict[str, Any]] | None = None,
    created_at: datetime | None = None,
    name: str = "test-v1",
    workspace: str = "default",
    base_name: str = "test",
    entity_version: int = 1,
    project: str = "p",
    config: str = "c",
    config_version: int = 1,
    like: ModelDeploymentEntity | None = None,
    **kwargs: Any,
) -> ModelDeploymentEntity:
    """Factory for test deployment entities. Pass status_history, status, status_message (and optionally created_at); other fields have defaults. Use like=<entity> to copy base fields from an existing entity and override only what changes."""
    if like is not None:
        params: dict[str, Any] = {
            "name": like.name,
            "workspace": like.workspace,
            "base_name": like.base_name,
            "entity_version": like.entity_version,
            "project": like.project,
            "config": like.config,
            "config_version": like.config_version,
            "status": like.status,
            "status_message": like.status_message or "",
            "status_history": list(like.status_history),
            "created_at": like._created_at,
        }
    else:
        params = {
            "name": name,
            "workspace": workspace,
            "base_name": base_name,
            "entity_version": entity_version,
            "project": project,
            "config": config,
            "config_version": config_version,
            "status": status,
            "status_message": status_message,
            "status_history": status_history if status_history is not None else [],
        }
        if created_at is not None:
            params["created_at"] = created_at
    params.update(kwargs)
    if like is not None:
        if status != ModelDeploymentStatus.CREATED or status_message:
            params["status"] = status
            params["status_message"] = status_message
        if status_history is not None:
            params["status_history"] = status_history
        if created_at is not None:
            params["created_at"] = created_at
    return _create_deployment_entity(**params)


@pytest.fixture
def mock_entity_client() -> AsyncMock:
    return AsyncMock(spec=EntityClient)


@pytest.fixture
def deployment_service(mock_entity_client: AsyncMock) -> ModelDeploymentService:
    return ModelDeploymentService(mock_entity_client)


def test_entity_to_schema_status_history_empty():
    """Schema status_history is empty when entity has no history."""
    entity = _deployment_entity(status_history=[])
    schema = _entity_to_schema(entity)
    assert schema.status_history == []


def test_entity_to_schema_status_history_populated():
    """Schema status_history contains converted items from entity dicts."""
    ts = "2025-01-15T10:00:00+00:00"
    entity = _deployment_entity(
        status=ModelDeploymentStatus.PENDING,
        status_message="msg",
        status_history=[{"timestamp": ts, "status": "PENDING", "status_message": "msg"}],
    )
    schema = _entity_to_schema(entity)
    assert len(schema.status_history) == 1
    assert schema.status_history[0].status == ModelDeploymentStatus.PENDING
    assert schema.status_history[0].status_message == "msg"
    assert schema.status_history[0].timestamp == datetime.fromisoformat(ts)


def test_compact_adjacent_status_history_merges_duplicates_keeps_first_timestamp():
    """Adjacent (status, message) duplicates are merged; first-seen timestamp is kept."""
    t1, t2, t3 = "2025-01-01T10:00:00", "2025-01-01T10:01:00", "2025-01-01T10:02:00"
    history = [
        {"timestamp": t1, "status": "PENDING", "status_message": "msg"},
        {"timestamp": t2, "status": "PENDING", "status_message": "msg"},
        {"timestamp": t3, "status": "READY", "status_message": ""},
    ]
    compacted = _compact_adjacent_status_history(history)
    assert len(compacted) == 2
    assert compacted[0]["timestamp"] == t1
    assert compacted[0]["status"] == "PENDING" and compacted[0]["status_message"] == "msg"
    assert compacted[1]["timestamp"] == t3
    assert compacted[1]["status"] == "READY"


def test_compact_adjacent_status_history_preserves_non_adjacent_duplicates():
    """Non-adjacent duplicates (e.g. A, B, A) are preserved as logical progression."""
    t1, t2, t3 = "2025-01-01T10:00:00", "2025-01-01T10:01:00", "2025-01-01T10:02:00"
    history = [
        {"timestamp": t1, "status": "PENDING", "status_message": "a"},
        {"timestamp": t2, "status": "READY", "status_message": ""},
        {"timestamp": t3, "status": "PENDING", "status_message": "a"},
    ]
    compacted = _compact_adjacent_status_history(history)
    assert len(compacted) == 3
    assert [e["status"] for e in compacted] == ["PENDING", "READY", "PENDING"]


@pytest.mark.asyncio
async def test_update_deployment_status_deduplication_no_change(
    deployment_service: ModelDeploymentService, mock_entity_client: AsyncMock
):
    """When status and message unchanged, update is not called."""
    entity = _deployment_entity(
        status=ModelDeploymentStatus.PENDING,
        status_message="same",
        status_history=[],
    )
    mock_list_result = MagicMock()
    mock_list_result.data = [entity]
    mock_entity_client.list.return_value = mock_list_result
    mock_entity_client.get.return_value = entity

    request = UpdateModelDeploymentStatusRequest(
        status=ModelDeploymentStatus.PENDING,
        status_message="same",
    )
    result = await deployment_service.update_deployment_status("default", "test", request)

    assert result is not None
    assert result.status == ModelDeploymentStatus.PENDING
    assert result.status_message == "same"
    mock_entity_client.update.assert_not_called()


@pytest.mark.asyncio
async def test_update_deployment_status_appends_to_history(
    deployment_service: ModelDeploymentService, mock_entity_client: AsyncMock
):
    """First status update seeds history and appends current state before updating."""
    created_at = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    entity = _deployment_entity(
        status=ModelDeploymentStatus.CREATED,
        status_message="",
        status_history=[],
        created_at=created_at,
    )
    mock_list_result = MagicMock()
    mock_list_result.data = [entity]
    mock_entity_client.list.return_value = mock_list_result
    mock_entity_client.get.return_value = entity

    # Return an entity that has the history the service would have written
    updated_entity = _deployment_entity(
        like=entity,
        status=ModelDeploymentStatus.PENDING,
        status_message="Provisioning",
        status_history=[
            {"timestamp": created_at.isoformat(), "status": "CREATED", "status_message": ""},
            {"timestamp": datetime.now(timezone.utc).isoformat(), "status": "CREATED", "status_message": ""},
        ],
    )
    mock_entity_client.update.return_value = updated_entity

    request = UpdateModelDeploymentStatusRequest(
        status=ModelDeploymentStatus.PENDING,
        status_message="Provisioning",
    )
    result = await deployment_service.update_deployment_status("default", "test", request)

    assert result is not None
    assert result.status == ModelDeploymentStatus.PENDING
    assert result.status_message == "Provisioning"
    assert len(result.status_history) == 2
    assert result.status_history[0].status == ModelDeploymentStatus.CREATED
    assert result.status_history[1].status == ModelDeploymentStatus.CREATED
    mock_entity_client.update.assert_called_once()


@pytest.mark.asyncio
async def test_update_deployment_status_history_capped_at_100(
    deployment_service: ModelDeploymentService, mock_entity_client: AsyncMock
):
    """History is trimmed to 100 most recent entries."""
    base_ts = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    entity = _deployment_entity(
        status=ModelDeploymentStatus.PENDING,
        status_message="old",
        status_history=[
            {
                "timestamp": (base_ts + timedelta(seconds=i)).isoformat(),
                "status": "PENDING",
                "status_message": f"msg{i}",
            }
            for i in range(100)
        ],
    )
    mock_list_result = MagicMock()
    mock_list_result.data = [entity]
    mock_entity_client.list.return_value = mock_list_result
    mock_entity_client.get.return_value = entity

    # After update, service keeps last 100; simulate by returning 100 entries (dropped oldest)
    new_history = entity.status_history[-99:] + [
        {"timestamp": datetime.now(timezone.utc).isoformat(), "status": "PENDING", "status_message": "new"}
    ]
    updated_entity = _deployment_entity(
        like=entity,
        status_message="new",
        status_history=new_history,
    )
    mock_entity_client.update.return_value = updated_entity

    request = UpdateModelDeploymentStatusRequest(
        status=ModelDeploymentStatus.PENDING,
        status_message="new",
    )
    result = await deployment_service.update_deployment_status("default", "test", request)

    assert result is not None
    assert len(result.status_history) == 100
    mock_entity_client.update.assert_called_once()
    call_entity = mock_entity_client.update.call_args[0][0]
    assert len(call_entity.status_history) == 100


@pytest.mark.asyncio
async def test_update_deployment_status_message_only_change(
    deployment_service: ModelDeploymentService, mock_entity_client: AsyncMock
):
    """Message-only change triggers update and appends to history."""
    entity = _deployment_entity(
        status=ModelDeploymentStatus.PENDING,
        status_message="old",
        status_history=[],
    )
    mock_list_result = MagicMock()
    mock_list_result.data = [entity]
    mock_entity_client.list.return_value = mock_list_result
    mock_entity_client.get.return_value = entity

    updated_entity = _deployment_entity(
        like=entity,
        status_message="new",
        status_history=[
            {"timestamp": datetime.now(timezone.utc).isoformat(), "status": "PENDING", "status_message": "old"},
        ],
    )
    mock_entity_client.update.return_value = updated_entity

    request = UpdateModelDeploymentStatusRequest(
        status=ModelDeploymentStatus.PENDING,
        status_message="new",
    )
    result = await deployment_service.update_deployment_status("default", "test", request)

    assert result is not None
    assert result.status_message == "new"
    assert len(result.status_history) == 1
    assert result.status_history[0].status_message == "old"
    mock_entity_client.update.assert_called_once()


@pytest.mark.asyncio
async def test_update_deployment_status_appends_to_history_when_empty(
    deployment_service: ModelDeploymentService, mock_entity_client: AsyncMock
):
    """First status update appends previous entity state then current; no synthetic CREATED seed."""
    now = datetime.now(timezone.utc)
    entity = _deployment_entity(
        status=ModelDeploymentStatus.PENDING,
        status_message="already pending",
        status_history=[],
        created_at=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
    )
    mock_list_result = MagicMock()
    mock_list_result.data = [entity]
    mock_entity_client.list.return_value = mock_list_result
    mock_entity_client.get.return_value = entity

    updated_entity = _deployment_entity(
        like=entity,
        status_message="updated",
        status_history=[
            {"timestamp": now.isoformat(), "status": "PENDING", "status_message": "already pending"},
            {"timestamp": now.isoformat(), "status": "PENDING", "status_message": "updated"},
        ],
    )
    mock_entity_client.update.return_value = updated_entity

    request = UpdateModelDeploymentStatusRequest(
        status=ModelDeploymentStatus.PENDING,
        status_message="updated",
    )
    result = await deployment_service.update_deployment_status("default", "test", request)

    assert result is not None
    assert len(result.status_history) == 2
    assert result.status_history[0].status == ModelDeploymentStatus.PENDING
    assert result.status_history[0].status_message == "already pending"
    assert result.status_history[1].status == ModelDeploymentStatus.PENDING
    assert result.status_history[1].status_message == "updated"
