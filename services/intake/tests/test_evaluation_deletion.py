# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluation deletion cascades, protection, and recovery after partial cleanup."""

import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, call

import pytest
from fastapi import HTTPException
from nhx.common.entities.client import EntityClient, EntityConflictError
from nhx.intake.api.v2.experiments.endpoints import _cleanup_evaluation_sessions, delete_evaluation
from nhx.intake.entities.experiments import Experiment as Evaluation
from nhx.intake.repository.clickhouse.executor import ClickHouseQueryError
from nhx.intake.repository.session import TraceIndexCollisionError
from nhx.intake.spans.service import IntakeSpansService, SessionInEvaluationError, SessionNotFoundError


def _evaluation(*, deleted: bool = False) -> Evaluation:
    return Evaluation(
        name="evaluation", workspace="default", experiment_ids=["group"], dataset_name="test", is_deleted=deleted
    )


def _client(entity: Evaluation) -> SimpleNamespace:
    return SimpleNamespace(
        get=AsyncMock(return_value=entity),
        get_by_id=AsyncMock(
            side_effect=lambda kind, entity_id: (
                entity if kind is Evaluation else SimpleNamespace(is_deleted=False, baseline_evaluation_name=None)
            )
        ),
        update=AsyncMock(return_value=entity),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("deleted", [False, True])
async def test_delete_evaluation_relinks_before_releasing_original_name(deleted: bool) -> None:
    entity = _evaluation(deleted=deleted)
    client = _client(entity)

    async def cleanup(**kwargs: str) -> None:
        assert entity.is_deleted
        assert entity.name == "evaluation"
        assert entity.deletion_name is not None

    async def relink(**kwargs: str) -> None:
        assert entity.name == "evaluation"
        assert kwargs["deleted_name"] == entity.deletion_name

    service = SimpleNamespace(
        delete_evaluation_sessions=AsyncMock(side_effect=cleanup),
        rename_evaluation_references=AsyncMock(side_effect=relink),
    )
    await delete_evaluation("default", "evaluation", cast(EntityClient, client), cast(IntakeSpansService, service))

    assert client.update.await_count == 3
    assert entity.name == entity.deletion_name
    assert entity.name.startswith("evaluation-deleted-")
    service.delete_evaluation_sessions.assert_awaited_once_with(workspace="default", evaluation_name="evaluation")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "status_code"),
    [(ClickHouseQueryError("sessions.delete"), 503), (TraceIndexCollisionError("default", "s"), 409)],
)
async def test_delete_evaluation_failure_can_be_retried(failure: Exception, status_code: int) -> None:
    entity = _evaluation()
    client = _client(entity)
    service = SimpleNamespace(
        delete_evaluation_sessions=AsyncMock(side_effect=[failure, None]), rename_evaluation_references=AsyncMock()
    )
    with pytest.raises(HTTPException) as error:
        await delete_evaluation("default", "evaluation", cast(EntityClient, client), cast(IntakeSpansService, service))
    assert error.value.status_code == status_code
    assert entity.is_deleted
    assert entity.name == "evaluation"
    assert entity.deletion_owner is None
    target = entity.deletion_name
    service.rename_evaluation_references.assert_not_awaited()

    await delete_evaluation("default", "evaluation", cast(EntityClient, client), cast(IntakeSpansService, service))
    assert client.update.await_count == 5
    assert entity.name == target
    assert service.delete_evaluation_sessions.await_count == 2


@pytest.mark.asyncio
async def test_partial_relink_reuses_durable_destination_name() -> None:
    entity = _evaluation()
    client = _client(entity)
    service = SimpleNamespace(
        delete_evaluation_sessions=AsyncMock(),
        rename_evaluation_references=AsyncMock(
            side_effect=[ClickHouseQueryError("sessions.rename_evaluation_traces"), None]
        ),
    )
    with pytest.raises(HTTPException) as error:
        await delete_evaluation("default", "evaluation", cast(EntityClient, client), cast(IntakeSpansService, service))
    assert error.value.status_code == 503
    assert entity.name == "evaluation"
    target = entity.deletion_name
    assert target is not None

    await delete_evaluation("default", "evaluation", cast(EntityClient, client), cast(IntakeSpansService, service))

    assert entity.name == target
    assert service.rename_evaluation_references.await_args_list == [
        call(workspace="default", original_name="evaluation", deleted_name=target),
        call(workspace="default", original_name="evaluation", deleted_name=target),
    ]


@pytest.mark.asyncio
async def test_cleanup_pages_past_protected_and_concurrently_deleted_sessions() -> None:
    repository = AsyncMock()
    repository.list_evaluation_session_ids.side_effect = [["a", "b"], ["c", "d"], []]
    service = IntakeSpansService(AsyncMock(), AsyncMock(), repository, AsyncMock(), AsyncMock())
    service.delete_session = AsyncMock(
        side_effect=[SessionInEvaluationError("default", "a"), None, SessionNotFoundError("default", "c"), None]
    )

    await service.delete_evaluation_sessions(workspace="default", evaluation_name="evaluation")

    assert repository.list_evaluation_session_ids.await_args_list == [
        call(workspace="default", evaluation_name="evaluation", after=None, limit=100),
        call(workspace="default", evaluation_name="evaluation", after="b", limit=100),
        call(workspace="default", evaluation_name="evaluation", after="d", limit=100),
    ]
    assert service.delete_session.await_args_list == [
        call(workspace="default", session_id=session_id) for session_id in ("a", "b", "c", "d")
    ]


@pytest.mark.asyncio
async def test_cleanup_propagates_storage_failures() -> None:
    repository = AsyncMock()
    repository.list_evaluation_session_ids.return_value = ["a", "b"]
    service = IntakeSpansService(AsyncMock(), AsyncMock(), repository, AsyncMock(), AsyncMock())
    service.delete_session = AsyncMock(side_effect=ClickHouseQueryError("sessions.delete"))

    with pytest.raises(ClickHouseQueryError):
        await service.delete_evaluation_sessions(workspace="default", evaluation_name="evaluation")
    service.delete_session.assert_awaited_once_with(workspace="default", session_id="a")


@pytest.mark.asyncio
async def test_overlapping_cleanup_and_stale_request_cannot_relink_replacement() -> None:
    entity = _evaluation(deleted=True)
    entity._id = "original-id"
    entity._db_version = 1
    entity.deletion_name = "evaluation-deleted-target"
    stored = entity.model_copy(deep=True)

    async def get_by_id(kind, identity):
        assert kind is Evaluation
        assert identity == "original-id"
        return stored.model_copy(deep=True)

    async def update(candidate, **kwargs):
        nonlocal stored
        if candidate.db_version != stored.db_version:
            raise EntityConflictError("version changed")
        stored = candidate.model_copy(deep=True)
        stored._db_version += 1
        return stored.model_copy(deep=True)

    client = SimpleNamespace(get_by_id=AsyncMock(side_effect=get_by_id), update=AsyncMock(side_effect=update))
    entered = asyncio.Event()
    finish = asyncio.Event()

    async def cleanup(**kwargs):
        entered.set()
        await finish.wait()

    service = SimpleNamespace(
        delete_evaluation_sessions=AsyncMock(side_effect=cleanup), rename_evaluation_references=AsyncMock()
    )
    running = asyncio.create_task(
        _cleanup_evaluation_sessions(
            cast(IntakeSpansService, service),
            cast(EntityClient, client),
            entity.model_copy(deep=True),
            workspace="default",
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        with pytest.raises(HTTPException) as error:
            await _cleanup_evaluation_sessions(
                cast(IntakeSpansService, service), cast(EntityClient, client), entity, workspace="default"
            )
        assert error.value.status_code == 409
    finally:
        finish.set()
        await running

    # The old name may now belong to a replacement. A delayed request still
    # holding the original snapshot must reload by ID and perform no relink.
    assert stored.name == "evaluation-deleted-target"
    assert stored.deletion_owner is None
    await _cleanup_evaluation_sessions(
        cast(IntakeSpansService, service), cast(EntityClient, client), entity, workspace="default"
    )
    service.rename_evaluation_references.assert_awaited_once_with(
        workspace="default", original_name="evaluation", deleted_name="evaluation-deleted-target"
    )


@pytest.mark.asyncio
async def test_cleanup_version_conflict_never_touches_telemetry() -> None:
    entity = _evaluation(deleted=True)
    entity.deletion_name = "evaluation-deleted-target"
    client = _client(entity)
    client.update.side_effect = EntityConflictError("another worker claimed cleanup")
    service = AsyncMock()
    with pytest.raises(HTTPException) as error:
        await _cleanup_evaluation_sessions(service, cast(EntityClient, client), entity, workspace="default")
    assert error.value.status_code == 409
    service.delete_evaluation_sessions.assert_not_awaited()
    service.rename_evaluation_references.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancelled_cleanup_keeps_name_and_claim_reserved() -> None:
    entity = _evaluation(deleted=True)
    entity.deletion_name = "evaluation-deleted-target"
    client = _client(entity)
    service = SimpleNamespace(
        delete_evaluation_sessions=AsyncMock(side_effect=asyncio.CancelledError),
        rename_evaluation_references=AsyncMock(),
    )
    with pytest.raises(asyncio.CancelledError):
        await _cleanup_evaluation_sessions(
            cast(IntakeSpansService, service), cast(EntityClient, client), entity, workspace="default"
        )
    assert entity.name == "evaluation"
    assert entity.deletion_owner is not None
    with pytest.raises(HTTPException) as error:
        await _cleanup_evaluation_sessions(
            cast(IntakeSpansService, service), cast(EntityClient, client), entity, workspace="default"
        )
    assert error.value.status_code == 409
    assert service.delete_evaluation_sessions.await_count == 1
    service.rename_evaluation_references.assert_not_awaited()


@pytest.mark.asyncio
async def test_unexpected_cleanup_error_logs_identifiers_without_line_breaks(caplog: pytest.LogCaptureFixture) -> None:
    entity = _evaluation(deleted=True)
    entity.deletion_name = "evaluation-deleted-target"
    client = _client(entity)
    service = SimpleNamespace(
        delete_evaluation_sessions=AsyncMock(side_effect=RuntimeError("unexpected storage failure")),
        rename_evaluation_references=AsyncMock(),
    )
    with pytest.raises(RuntimeError, match="unexpected storage failure"):
        await _cleanup_evaluation_sessions(
            cast(IntakeSpansService, service),
            cast(EntityClient, client),
            entity,
            workspace="default\r\nforged-entry",
        )
    record = next(record for record in caplog.records if "Evaluation cleanup needs recovery" in record.message)
    assert "\r" not in record.getMessage()
    assert "\n" not in record.getMessage()
    assert "defaultforged-entry" in record.getMessage()
    assert str(entity.id) in record.getMessage()
    assert entity.deletion_owner is not None
