# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared dependencies for Intake trace endpoints."""

from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.workspaces.client import AsyncWorkspacesClient
from nhx.common.service.dependencies import get_nemo_client
from nhx.intake.experiments.denormalizer import EvaluationDenormalizer
from nhx.intake.repository.annotations import AnnotationsRepository
from nhx.intake.repository.clickhouse.annotations import ClickHouseAnnotationsRepository
from nhx.intake.repository.clickhouse.evaluator_results import ClickHouseEvaluatorResultsRepository
from nhx.intake.repository.clickhouse.executor import ClickHouseExecutor
from nhx.intake.repository.clickhouse.session import ClickHouseSessionRepository
from nhx.intake.repository.clickhouse.span import ClickHouseSpanRepository
from nhx.intake.repository.clickhouse.trace import ClickHouseTraceRepository
from nhx.intake.repository.evaluator_results import EvaluatorResultsRepository
from nhx.intake.repository.session import SessionRepository
from nhx.intake.repository.span import SpanRepository
from nhx.intake.repository.trace import TraceRepository
from nhx.intake.spans.clickhouse_client import ClickHouseSpanClient, get_clickhouse_client
from nhx.intake.spans.service import IntakeSpansService


async def require_workspace_access(
    workspace: str,
    client: AsyncNemoClient = Depends(get_nemo_client),
) -> None:
    """Validate that the request principal can access the path workspace."""

    (await AsyncWorkspacesClient.from_client(client).get_workspace(name=workspace)).data()


def validate_list_query_params(request: Request, additional_params: set[str] | None = None) -> None:
    """Reject unsupported top-level query params while allowing deep-object filters."""

    allowed = {"page", "page_size", "sort", "filter"}
    if additional_params is not None:
        allowed.update(additional_params)

    unsupported = []
    for key in request.query_params.keys():
        if key in allowed or key.startswith("filter["):
            continue
        unsupported.append(key)

    if unsupported:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported query parameter(s): {', '.join(sorted(set(unsupported)))}",
        )


def get_clickhouse_executor(
    client: Annotated[ClickHouseSpanClient, Depends(get_clickhouse_client)],
) -> ClickHouseExecutor:
    return ClickHouseExecutor(client)


def get_span_repository(
    executor: Annotated[ClickHouseExecutor, Depends(get_clickhouse_executor)],
) -> SpanRepository:
    return ClickHouseSpanRepository(executor)


def get_trace_repository(
    executor: Annotated[ClickHouseExecutor, Depends(get_clickhouse_executor)],
) -> TraceRepository:
    return ClickHouseTraceRepository(executor)


def get_session_repository(
    executor: Annotated[ClickHouseExecutor, Depends(get_clickhouse_executor)],
) -> SessionRepository:
    return ClickHouseSessionRepository(executor)


def get_evaluator_results_repository(
    executor: Annotated[ClickHouseExecutor, Depends(get_clickhouse_executor)],
) -> EvaluatorResultsRepository:
    return ClickHouseEvaluatorResultsRepository(executor)


def get_annotations_repository(
    executor: Annotated[ClickHouseExecutor, Depends(get_clickhouse_executor)],
) -> AnnotationsRepository:
    return ClickHouseAnnotationsRepository(executor)


def get_spans_service(
    span_repository: Annotated[SpanRepository, Depends(get_span_repository)],
    trace_repository: Annotated[TraceRepository, Depends(get_trace_repository)],
    session_repository: Annotated[SessionRepository, Depends(get_session_repository)],
    evaluator_results_repository: Annotated[EvaluatorResultsRepository, Depends(get_evaluator_results_repository)],
    annotations_repository: Annotated[AnnotationsRepository, Depends(get_annotations_repository)],
) -> IntakeSpansService:
    return IntakeSpansService(
        span_repository,
        trace_repository,
        session_repository,
        evaluator_results_repository,
        annotations_repository,
    )


SpansServiceDep = Annotated[IntakeSpansService, Depends(get_spans_service)]


def get_denormalizer(request: Request) -> EvaluationDenormalizer | None:
    """Reach the service-owned denormalizer from the request (absent if startup didn't create one)."""
    service = getattr(request.app.state, "intake_service", None) or getattr(request.app.state, "service", None)
    return getattr(service, "denormalizer", None) if service is not None else None


DenormalizerDep = Annotated[EvaluationDenormalizer | None, Depends(get_denormalizer)]
