# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioral tests for Evaluation cross-store read composition."""

from datetime import datetime, timezone
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
from nhx.intake.repository.evaluation_rollup import (
    EvaluationRollup,
    EvaluationRollupRepository,
    MissingCreatedAfterError,
)
from nhx.intake.repository.evaluation_session import EvaluationSessionPage, EvaluationSessionRepository
from nhx.intake.spans.domain import IntakeResponseMode, SpanStatus


class _RollupRepository(EvaluationRollupRepository):
    def __init__(self, rollups: dict[str, EvaluationRollup] | Exception) -> None:
        self._rollups = rollups
        self.calls: list[tuple[str, list[str], dict[str, datetime | None]]] = []

    async def get_rollups(
        self,
        *,
        workspace: str,
        evaluation_names: list[str],
        created_after_by_name: dict[str, datetime | None],
    ) -> dict[str, EvaluationRollup]:
        self.calls.append((workspace, list(evaluation_names), dict(created_after_by_name)))
        if isinstance(self._rollups, Exception):
            raise self._rollups
        return self._rollups


class _SessionRepository(EvaluationSessionRepository):
    def __init__(self, result: EvaluationSessionPage | Exception) -> None:
        self._result = result
        self.calls: list[datetime | None] = []

    async def list_sessions(
        self,
        *,
        workspace: str,
        evaluation_name: str,
        status: SpanStatus | None = None,
        test_case_name: str | None = None,
        created_after: datetime | None,
        page: int,
        page_size: int,
        mode: IntakeResponseMode,
        sort_keys: list[tuple[str, bool]] | None = None,
    ) -> EvaluationSessionPage:
        self.calls.append(created_after)
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


def _evaluation(name: str, *, deleted: bool = False, created_at: datetime | None = None) -> Evaluation:
    evaluation = Evaluation(
        workspace="default",
        name=name,
        experiment_ids=["experiment-group-1"],
        dataset_name="dataset",
    )
    evaluation.is_deleted = deleted
    evaluation._created_at = created_at
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
    assert rollups.calls == [("default", ["eval-1", "eval-2"], {"eval-1": None, "eval-2": None})]


@pytest.mark.asyncio
async def test_list_evaluations_passes_each_created_at_as_rollup_bound() -> None:
    first_created_at = datetime(2026, 3, 1, tzinfo=timezone.utc)
    first = _evaluation("eval-1", created_at=first_created_at)
    second = _evaluation("eval-2")
    client, list_mock, _ = _entity_client()
    list_mock.return_value = SimpleNamespace(
        data=[first, second],
        pagination=SimpleNamespace(total_results=2),
    )
    rollups = _RollupRepository({})
    service = EvaluationReadService(
        entity_client=client,
        rollup_repository=rollups,
        session_repository=None,
    )

    await service.list_evaluations(
        workspace="default",
        filter_operation=None,
        limit=1000,
    )

    assert rollups.calls == [("default", ["eval-1", "eval-2"], {"eval-1": first_created_at, "eval-2": None})]


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
async def test_missing_created_after_entry_is_raised_not_masked_as_clickhouse_outage() -> None:
    client, _, get_mock = _entity_client()
    get_mock.return_value = _evaluation("eval-1")
    service = EvaluationReadService(
        entity_client=client,
        rollup_repository=_RollupRepository(
            MissingCreatedAfterError("created_after_by_name is missing entries for: eval-1")
        ),
        session_repository=None,
    )

    with pytest.raises(MissingCreatedAfterError, match="created_after_by_name"):
        await service.get_evaluation(workspace="default", name="eval-1")


@pytest.mark.asyncio
async def test_other_rollup_key_errors_still_read_as_rollups_unavailable() -> None:
    client, _, _ = _entity_client()
    service = EvaluationReadService(
        entity_client=client,
        rollup_repository=_RollupRepository(KeyError("run_count")),
        session_repository=None,
    )

    batch = await service.attach_rollups(workspace="default", evaluations=[_evaluation("eval-1")])

    assert batch.rollups_available is False
    assert batch.evaluations[0].rollup is None


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
async def test_get_evaluation_bounds_rollups_and_sessions_to_the_live_entity() -> None:
    created_at = datetime(2026, 3, 2, tzinfo=timezone.utc)
    client, _, get_mock = _entity_client()
    get_mock.return_value = _evaluation("eval-1", created_at=created_at)
    sessions = _SessionRepository(EvaluationSessionPage(rows=[], total=0))
    rollups = _RollupRepository({})
    service = EvaluationReadService(
        entity_client=client,
        rollup_repository=rollups,
        session_repository=sessions,
    )

    await service.get_evaluation(workspace="default", name="eval-1")
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

    assert rollups.calls == [("default", ["eval-1"], {"eval-1": created_at})]
    assert sessions.calls == [created_at]


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
