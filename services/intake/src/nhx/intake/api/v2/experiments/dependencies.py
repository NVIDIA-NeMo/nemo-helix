# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependencies for the Evaluations API."""

from typing import Annotated

from fastapi import Depends
from nhx.common.entities.client import EntityClient
from nhx.common.service.dependencies import get_entity_client
from nhx.intake.dependencies import DenormalizerDep as DenormalizerDep
from nhx.intake.dependencies import IntakeServiceDep
from nhx.intake.experiments.read_service import EvaluationReadService
from nhx.intake.repository.clickhouse.evaluation_rollup import ClickHouseEvaluationRollupRepository
from nhx.intake.repository.clickhouse.evaluation_session import ClickHouseEvaluationSessionRepository
from nhx.intake.repository.clickhouse.executor import ClickHouseExecutor
from nhx.intake.repository.evaluation_rollup import EvaluationRollupRepository
from nhx.intake.repository.evaluation_session import EvaluationSessionRepository

EntityClientDep = Annotated[EntityClient, Depends(get_entity_client)]


def get_evaluation_rollup_repository(intake: IntakeServiceDep) -> EvaluationRollupRepository | None:
    # Rollups are enrichment only. Evaluation entity reads should continue when
    # ClickHouse is disabled or temporarily unavailable.
    client = intake.clickhouse_client
    return ClickHouseEvaluationRollupRepository(ClickHouseExecutor(client)) if client is not None else None


EvaluationRollupRepositoryDep = Annotated[EvaluationRollupRepository | None, Depends(get_evaluation_rollup_repository)]


def get_evaluation_session_repository(intake: IntakeServiceDep) -> EvaluationSessionRepository | None:
    client = intake.clickhouse_client
    return ClickHouseEvaluationSessionRepository(ClickHouseExecutor(client)) if client is not None else None


EvaluationSessionRepositoryDep = Annotated[
    EvaluationSessionRepository | None, Depends(get_evaluation_session_repository)
]


def get_evaluation_read_service(
    entity_client: EntityClientDep,
    rollup_repository: EvaluationRollupRepositoryDep,
    session_repository: EvaluationSessionRepositoryDep,
) -> EvaluationReadService:
    return EvaluationReadService(
        entity_client=entity_client,
        rollup_repository=rollup_repository,
        session_repository=session_repository,
    )


EvaluationReadServiceDep = Annotated[EvaluationReadService, Depends(get_evaluation_read_service)]
