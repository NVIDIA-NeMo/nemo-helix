# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-store read composition for Evaluations."""

import logging
from dataclasses import dataclass

from nhx.common.api.filter import FilterOperation
from nhx.common.entities.client import EntityClient, EntityNotFoundError
from nhx.intake.entities.experiments import Experiment as Evaluation
from nhx.intake.repository.evaluation_rollup import EvaluationRollup, EvaluationRollupRepository
from nhx.intake.repository.evaluation_session import (
    EvaluationSessionPage,
    EvaluationSessionRepository,
    MetricSortTooLargeError,
)
from nhx.intake.spans.domain import IntakeResponseMode, SpanStatus

logger = logging.getLogger(__name__)


class EvaluationNotFoundError(Exception):
    def __init__(self, workspace: str, name: str) -> None:
        super().__init__(f"Evaluation {workspace}/{name} not found")
        self.workspace = workspace
        self.name = name


class EvaluationReadLimitExceededError(Exception):
    def __init__(self, selected: int, limit: int) -> None:
        super().__init__(f"Evaluation read selected {selected} rows, limit is {limit}")
        self.selected = selected
        self.limit = limit


class EvaluationTelemetryUnavailableError(Exception):
    """Raised when telemetry is required to serve an Evaluation read."""

    def __init__(self, *, configured: bool) -> None:
        self.configured = configured
        super().__init__("Evaluation telemetry store unavailable")


class InvalidEvaluationSessionStatusError(Exception):
    def __init__(self, value: str) -> None:
        self.value = value
        super().__init__(f"Invalid Evaluation session status: {value}")


@dataclass(frozen=True)
class EvaluationRead:
    entity: Evaluation
    rollup: EvaluationRollup | None


@dataclass(frozen=True)
class EvaluationReadBatch:
    evaluations: list[EvaluationRead]
    total: int
    rollups_available: bool


class EvaluationReadService:
    """Compose Evaluation entities with their ClickHouse-backed read data."""

    def __init__(
        self,
        *,
        entity_client: EntityClient,
        rollup_repository: EvaluationRollupRepository | None,
        session_repository: EvaluationSessionRepository | None,
    ) -> None:
        self._entities = entity_client
        self._rollups = rollup_repository
        self._sessions = session_repository

    async def list_evaluations(
        self,
        *,
        workspace: str,
        filter_operation: FilterOperation | None,
        limit: int,
    ) -> EvaluationReadBatch:
        result = await self._entities.list(
            Evaluation,
            workspace=workspace,
            filter_operation=filter_operation,
            page=1,
            page_size=limit,
        )
        total = result.pagination.total_results
        if total > limit:
            raise EvaluationReadLimitExceededError(total, limit)
        return await self.attach_rollups(
            workspace=workspace,
            evaluations=result.data,
            total=total,
        )

    async def get_evaluation(self, *, workspace: str, name: str) -> EvaluationRead:
        evaluation = await self._get_live_evaluation(workspace=workspace, name=name)
        batch = await self.attach_rollups(workspace=workspace, evaluations=[evaluation])
        return batch.evaluations[0]

    async def attach_rollups(
        self,
        *,
        workspace: str,
        evaluations: list[Evaluation],
        total: int | None = None,
    ) -> EvaluationReadBatch:
        """Attach rollups to entities already loaded by an Evaluation workflow."""

        if not evaluations:
            return EvaluationReadBatch(evaluations=[], total=total or 0, rollups_available=True)

        rollups, available = await self._get_rollups(
            workspace=workspace,
            evaluation_names=[evaluation.name for evaluation in evaluations],
        )
        return EvaluationReadBatch(
            evaluations=[
                EvaluationRead(entity=evaluation, rollup=rollups.get(evaluation.name)) for evaluation in evaluations
            ],
            total=len(evaluations) if total is None else total,
            rollups_available=available,
        )

    async def list_sessions(
        self,
        *,
        workspace: str,
        evaluation_name: str,
        status: str | None,
        test_case_name: str | None,
        page: int,
        page_size: int,
        mode: IntakeResponseMode,
        sort_keys: list[tuple[str, bool]] | None,
    ) -> EvaluationSessionPage:
        await self._get_live_evaluation(workspace=workspace, name=evaluation_name)
        if self._sessions is None:
            raise EvaluationTelemetryUnavailableError(configured=False)
        status_filter = None
        if status is not None:
            try:
                status_filter = SpanStatus(status)
            except ValueError as exc:
                raise InvalidEvaluationSessionStatusError(status) from exc
        try:
            return await self._sessions.list_sessions(
                workspace=workspace,
                evaluation_name=evaluation_name,
                status=status_filter,
                test_case_name=test_case_name,
                page=page,
                page_size=page_size,
                mode=mode,
                sort_keys=sort_keys,
            )
        except MetricSortTooLargeError:
            raise
        except Exception as exc:
            logger.exception(
                "Per-session read failed for workspace=%s evaluation=%s",
                _sanitize_for_log(workspace),
                _sanitize_for_log(evaluation_name),
            )
            raise EvaluationTelemetryUnavailableError(configured=True) from exc

    async def _get_live_evaluation(self, *, workspace: str, name: str) -> Evaluation:
        try:
            evaluation = await self._entities.get(Evaluation, workspace=workspace, name=name)
        except EntityNotFoundError as exc:
            raise EvaluationNotFoundError(workspace, name) from exc
        if evaluation.is_deleted:
            raise EvaluationNotFoundError(workspace, name)
        return evaluation

    async def _get_rollups(
        self,
        *,
        workspace: str,
        evaluation_names: list[str],
    ) -> tuple[dict[str, EvaluationRollup], bool]:
        if self._rollups is None:
            return {}, False
        try:
            return (
                await self._rollups.get_rollups(
                    workspace=workspace,
                    evaluation_names=evaluation_names,
                ),
                True,
            )
        except Exception:
            logger.exception("Skipping evaluation rollup hydration because ClickHouse is unavailable")
            return {}, False


def _sanitize_for_log(value: str) -> str:
    return value.replace("\r", "").replace("\n", "")
