# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioral tests for Evaluation cross-store read composition."""

from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest
from nhx.common.entities.client import EntityClient, EntityNotFoundError
from nhx.intake.entities.experiments import Experiment as Evaluation
from nhx.intake.experiments.read_service import (
    EvaluationNotFoundError,
    EvaluationReadLimitExceededError,
    EvaluationReadService,
    EvaluationTelemetryUnavailableError,
)
from nhx.intake.repository.evaluation_rollup import EvaluationRollup, EvaluationRollupRepository
from nhx.intake.repository.evaluation_session import EvaluationSessionPage, EvaluationSessionRepository
from nhx.intake.spans.domain import IntakeResponseMode, SpanStatus


class _RollupRepository(EvaluationRollupRepository):
    def __init__(self, rollups: dict[str, EvaluationRollup] | Exception) -> None:
        self._rollups = rollups
        self.calls: list[tuple[str, list[str]]] = []

    async def get_rollups(
        self,
        *,
        workspace: str,
        evaluation_names: list[str],
    ) -> dict[str, EvaluationRollup]:
        self.calls.append((workspace, evaluation_names))
        if isinstance(self._rollups, Exception):
            raise self._rollups
        return self._rollups


class _SessionRepository(EvaluationSessionRepository):
    def __init__(self, result: EvaluationSessionPage | Exception) -> None:
        self._result = result

    async def list_sessions(
        self,
        *,
        workspace: str,
        evaluation_name: str,
        status: SpanStatus | None = None,
        test_case_name: str | None = None,
        page: int,
        page_size: int,
        mode: IntakeResponseMode,
        sort_keys: list[tuple[str, bool]] | None = None,
    ) -> EvaluationSessionPage:
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


def _evaluation(name: str, *, deleted: bool = False) -> Evaluation:
    evaluation = Evaluation(
        workspace="default",
        name=name,
        experiment_ids=["experiment-group-1"],
        dataset_name="dataset",
    )
    evaluation.is_deleted = deleted
    return evaluation


def _entity_client() -> tuple[EntityClient, AsyncMock, AsyncMock]:
    list_mock = AsyncMock()
    get_mock = AsyncMock()
    client = cast(EntityClient, SimpleNamespace(list=list_mock, get=get_mock))
    return client, list_mock, get_mock


@pytest.mark.asyncio
async def test_list_evaluations_batches_rollup_enrichment() -> None:
    first = _evaluation("eval-1")
    second = _evaluation("eval-2")
    client, list_mock, _ = _entity_client()
    list_mock.return_value = SimpleNamespace(
        data=[first, second],
        pagination=SimpleNamespace(total_results=2),
    )
    first_rollup = EvaluationRollup(evaluation_name=first.name, run_count=4)
    rollups = _RollupRepository({first.name: first_rollup})
    service = EvaluationReadService(
        entity_client=client,
        rollup_repository=rollups,
        session_repository=None,
    )

    result = await service.list_evaluations(
        workspace="default",
        filter_operation=None,
        limit=1000,
    )

    assert result.total == 2
    assert result.rollups_available is True
    assert result.evaluations[0].entity is first
    assert result.evaluations[0].rollup is first_rollup
    assert result.evaluations[1].entity is second
    assert result.evaluations[1].rollup is None
    assert list_mock.await_count == 1
    assert rollups.calls == [("default", ["eval-1", "eval-2"])]


@pytest.mark.asyncio
async def test_list_evaluations_degrades_when_rollups_fail() -> None:
    evaluation = _evaluation("eval-1")
    client, list_mock, _ = _entity_client()
    list_mock.return_value = SimpleNamespace(
        data=[evaluation],
        pagination=SimpleNamespace(total_results=1),
    )
    service = EvaluationReadService(
        entity_client=client,
        rollup_repository=_RollupRepository(RuntimeError("ClickHouse unavailable")),
        session_repository=None,
    )

    result = await service.list_evaluations(
        workspace="default",
        filter_operation=None,
        limit=1000,
    )

    assert result.rollups_available is False
    assert result.evaluations[0].entity is evaluation
    assert result.evaluations[0].rollup is None


@pytest.mark.asyncio
async def test_list_evaluations_rejects_over_limit_before_rollup_lookup() -> None:
    evaluation = _evaluation("eval-1")
    client, list_mock, _ = _entity_client()
    list_mock.return_value = SimpleNamespace(
        data=[evaluation],
        pagination=SimpleNamespace(total_results=1001),
    )
    rollups = _RollupRepository({})
    service = EvaluationReadService(
        entity_client=client,
        rollup_repository=rollups,
        session_repository=None,
    )

    with pytest.raises(EvaluationReadLimitExceededError) as exc_info:
        await service.list_evaluations(
            workspace="default",
            filter_operation=None,
            limit=1000,
        )

    assert exc_info.value.selected == 1001
    assert rollups.calls == []


@pytest.mark.asyncio
async def test_get_evaluation_rejects_missing_and_deleted_entities() -> None:
    client, _, get_mock = _entity_client()
    service = EvaluationReadService(
        entity_client=client,
        rollup_repository=_RollupRepository({}),
        session_repository=None,
    )
    get_mock.side_effect = EntityNotFoundError("missing")

    with pytest.raises(EvaluationNotFoundError):
        await service.get_evaluation(workspace="default", name="missing")

    get_mock.side_effect = None
    get_mock.return_value = _evaluation("deleted", deleted=True)
    with pytest.raises(EvaluationNotFoundError):
        await service.get_evaluation(workspace="default", name="deleted")


@pytest.mark.asyncio
async def test_session_failures_are_translated_after_entity_validation() -> None:
    client, _, get_mock = _entity_client()
    get_mock.return_value = _evaluation("eval-1")
    service = EvaluationReadService(
        entity_client=client,
        rollup_repository=None,
        session_repository=_SessionRepository(RuntimeError("ClickHouse unavailable")),
    )

    with pytest.raises(EvaluationTelemetryUnavailableError) as exc_info:
        await service.list_sessions(
            workspace="default",
            evaluation_name="eval-1",
            status=None,
            test_case_name=None,
            page=1,
            page_size=100,
            mode="detailed",
            sort_keys=None,
        )

    assert exc_info.value.configured is True
    assert get_mock.await_count == 1
