# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluation deletion cascades, protection, and recovery after partial cleanup."""

from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, call

import pytest
from fastapi import HTTPException
from nhx.common.entities.client import EntityClient
from nhx.intake.api.v2.experiments.endpoints import delete_evaluation
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
        get_by_id=AsyncMock(return_value=SimpleNamespace(is_deleted=False, baseline_evaluation_name=None)),
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

    assert client.update.await_count == 2
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
    target = entity.deletion_name
    service.rename_evaluation_references.assert_not_awaited()

    await delete_evaluation("default", "evaluation", cast(EntityClient, client), cast(IntakeSpansService, service))
    assert client.update.await_count == 2
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
