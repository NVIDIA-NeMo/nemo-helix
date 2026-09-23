# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependencies for the Evaluations API."""

from typing import Annotated

from fastapi import Depends, Request
from nhx.common.entities.client import EntityClient
from nhx.common.service.dependencies import get_entity_client
from nhx.intake.experiments.denormalizer import EvaluationDenormalizer
from nhx.intake.experiments.read_service import EvaluationReadService
from nhx.intake.repository.clickhouse.evaluation_rollup import ClickHouseEvaluationRollupRepository
from nhx.intake.repository.clickhouse.evaluation_session import ClickHouseEvaluationSessionRepository
from nhx.intake.repository.clickhouse.executor import ClickHouseExecutor
from nhx.intake.repository.evaluation_rollup import EvaluationRollupRepository
from nhx.intake.repository.evaluation_session import EvaluationSessionRepository
from nhx.intake.spans.clickhouse_client import ClickHouseSpanClient

EntityClientDep = Annotated[EntityClient, Depends(get_entity_client)]


def _get_clickhouse_client(request: Request) -> ClickHouseSpanClient | None:
    service = getattr(request.app.state, "intake_service", None) or getattr(request.app.state, "service", None)
    if service is None:
        return None
    return getattr(service, "clickhouse_client", None)


def get_evaluation_rollup_repository(request: Request) -> EvaluationRollupRepository | None:
    # Rollups are enrichment only. Evaluation entity reads should continue when
    # ClickHouse is disabled or temporarily unavailable.
    client = _get_clickhouse_client(request)
    return ClickHouseEvaluationRollupRepository(ClickHouseExecutor(client)) if client is not None else None


EvaluationRollupRepositoryDep = Annotated[EvaluationRollupRepository | None, Depends(get_evaluation_rollup_repository)]


def get_evaluation_session_repository(request: Request) -> EvaluationSessionRepository | None:
    client = _get_clickhouse_client(request)
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


def get_denormalizer(request: Request) -> EvaluationDenormalizer | None:
    """Reach the service-owned denormalizer from the request (absent if startup didn't create one)."""
    service = getattr(request.app.state, "intake_service", None) or getattr(request.app.state, "service", None)
    return getattr(service, "denormalizer", None) if service is not None else None


DenormalizerDep = Annotated[EvaluationDenormalizer | None, Depends(get_denormalizer)]
