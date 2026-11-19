# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Session detail API tests."""

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from nhx.common.entities.client import EntityClient, EntityNotFoundError
from nhx.intake.entities.experiments import Experiment as Evaluation
from nhx.intake.repository.session import SessionRepository, TraceIndexCollisionError
from nhx.intake.spans.api.sessions import delete_session, get_session
from nhx.intake.spans.domain import IntakeSession, SpanStatus
from nhx.intake.spans.service import IntakeSpansService, SessionInEvaluationError, SessionNotFoundError


@pytest.mark.asyncio
async def test_get_session_returns_aggregate_schema_without_children() -> None:
    domain = IntakeSession(
        id="session-a",
        workspace="workspace-a",
        started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        status=SpanStatus.SUCCESS,
        trace_count=2,
        span_count=5,
        total_tokens=123,
    )
    service = SimpleNamespace(get_session=AsyncMock(return_value=domain))

    response = await get_session("workspace-a", "session-a", cast(IntakeSpansService, service))

    assert response.id == "session-a"
    assert response.trace_count == 2
    assert response.span_count == 5
    assert response.total_tokens == 123
    fields = type(response).model_fields
    assert "traces" not in fields
    assert "spans" not in fields
    assert "input" not in fields
    assert "output" not in fields


@pytest.mark.asyncio
async def test_get_session_returns_404_when_missing() -> None:
    service = SimpleNamespace(get_session=AsyncMock(side_effect=SessionNotFoundError("workspace-a", "missing")))

    with pytest.raises(HTTPException) as error:
        await get_session("workspace-a", "missing", cast(IntakeSpansService, service))

    assert error.value.status_code == 404
    assert error.value.detail == "Session workspace-a/missing not found"


@pytest.mark.asyncio
async def test_delete_session_delegates_to_service() -> None:
    service = SimpleNamespace(delete_session=AsyncMock())

    assert await delete_session("workspace-a", "session-a", cast(IntakeSpansService, service)) is None

    service.delete_session.assert_awaited_once_with(workspace="workspace-a", session_id="session-a")


@pytest.mark.asyncio
async def test_delete_session_returns_404_when_missing() -> None:
    service = SimpleNamespace(delete_session=AsyncMock(side_effect=SessionNotFoundError("workspace-a", "missing")))

    with pytest.raises(HTTPException) as error:
        await delete_session("workspace-a", "missing", cast(IntakeSpansService, service))

    assert error.value.status_code == 404
    assert error.value.detail == "Session workspace-a/missing not found"


@pytest.mark.asyncio
@pytest.mark.parametrize(("exists", "has_evaluation"), [(False, False), (True, False), (True, True)])
async def test_delete_session_service_checks_existence_and_evaluation(exists: bool, has_evaluation: bool) -> None:
    repository = SimpleNamespace(
        get_session=AsyncMock(return_value=object() if exists else None),
        has_live_trace_index=AsyncMock(return_value=False),
        get_evaluation_names=AsyncMock(return_value=["evaluation-a"] if has_evaluation else []),
        delete_session=AsyncMock(),
    )
    service = IntakeSpansService(
        span_repository=AsyncMock(),
        trace_repository=AsyncMock(),
        session_repository=cast(SessionRepository, repository),
        evaluator_results_repository=AsyncMock(),
        annotations_repository=AsyncMock(),
    )

    if exists and not has_evaluation:
        await service.delete_session(workspace="workspace-a", session_id="session-a")
        repository.delete_session.assert_awaited_once_with(workspace="workspace-a", session_id="session-a")
    else:
        with pytest.raises(SessionInEvaluationError if exists else SessionNotFoundError):
            await service.delete_session(workspace="workspace-a", session_id="session-a")
        repository.delete_session.assert_not_awaited()
    if exists:
        repository.get_evaluation_names.assert_awaited_once_with(
            workspace="workspace-a", session_id="session-a", include_deleted=False
        )
    else:
        repository.get_evaluation_names.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_session_returns_409_when_part_of_evaluation() -> None:
    service = SimpleNamespace(
        delete_session=AsyncMock(side_effect=SessionInEvaluationError("workspace-a", "session-a"))
    )

    with pytest.raises(HTTPException) as error:
        await delete_session("workspace-a", "session-a", cast(IntakeSpansService, service))

    assert error.value.status_code == 409
    assert error.value.detail == (
        "Session workspace-a/session-a is associated with an undeleted or unresolved evaluation and cannot be deleted"
    )


@pytest.mark.asyncio
async def test_delete_session_returns_409_when_trace_index_key_has_conflicting_owners() -> None:
    service = SimpleNamespace(
        delete_session=AsyncMock(side_effect=TraceIndexCollisionError("workspace-a", "session-a"))
    )
    with pytest.raises(HTTPException) as error:
        await delete_session("workspace-a", "session-a", cast(IntakeSpansService, service))

    assert error.value.status_code == 409
    assert error.value.detail == (
        "Session workspace-a/session-a cannot be deleted because its root spans have conflicting trace-index ownership"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("states", [[True], [True, True], [False], [True, False], [None], [True, None]])
async def test_delete_session_requires_all_evaluations_to_be_confirmed_deleted(states: list[bool | None]) -> None:
    names = [f"evaluation-{index}" for index in range(len(states))]
    repository = SimpleNamespace(
        get_session=AsyncMock(return_value=object()),
        get_evaluation_names=AsyncMock(return_value=names),
        delete_session=AsyncMock(),
    )
    client = SimpleNamespace(
        get=AsyncMock(
            side_effect=[
                EntityNotFoundError("missing")
                if state is None
                else Evaluation(
                    name=name,
                    workspace="workspace-a",
                    experiment_ids=["group-a"],
                    dataset_name="test",
                    is_deleted=state,
                )
                for name, state in zip(names, states, strict=True)
            ]
        )
    )
    service = IntakeSpansService(
        span_repository=AsyncMock(),
        trace_repository=AsyncMock(),
        session_repository=cast(SessionRepository, repository),
        evaluator_results_repository=AsyncMock(),
        annotations_repository=AsyncMock(),
        entity_client=cast(EntityClient, client),
    )

    if all(state is True for state in states):
        await service.delete_session(workspace="workspace-a", session_id="session-a")
        repository.delete_session.assert_awaited_once_with(workspace="workspace-a", session_id="session-a")
        assert client.get.await_count == len(names)
    else:
        with pytest.raises(SessionInEvaluationError):
            await service.delete_session(workspace="workspace-a", session_id="session-a")
        repository.delete_session.assert_not_awaited()
    client.get.assert_any_await(Evaluation, workspace="workspace-a", name=names[0])


@pytest.mark.asyncio
async def test_delete_session_does_not_delete_when_evaluation_lookup_fails() -> None:
    repository = SimpleNamespace(
        get_session=AsyncMock(return_value=object()),
        get_evaluation_names=AsyncMock(return_value=["evaluation-a"]),
        delete_session=AsyncMock(),
    )
    client = SimpleNamespace(get=AsyncMock(side_effect=RuntimeError("entity store unavailable")))
    service = IntakeSpansService(
        span_repository=AsyncMock(),
        trace_repository=AsyncMock(),
        session_repository=cast(SessionRepository, repository),
        evaluator_results_repository=AsyncMock(),
        annotations_repository=AsyncMock(),
        entity_client=cast(EntityClient, client),
    )
    with pytest.raises(RuntimeError, match="entity store unavailable"):
        await service.delete_session(workspace="workspace-a", session_id="session-a")
    repository.delete_session.assert_not_awaited()
