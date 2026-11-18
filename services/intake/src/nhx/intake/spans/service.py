# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Application service for Intake trace spans."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from nhx.common.api.common import PaginatedResult
from nhx.common.entities.client import EntityClient, EntityNotFoundError
from nhx.intake.entities.experiments import Experiment as Evaluation
from nhx.intake.repository.annotations import AnnotationsRepository
from nhx.intake.repository.evaluator_results import EvaluatorResultsRepository
from nhx.intake.repository.session import SessionRepository
from nhx.intake.repository.span import SpanRepository
from nhx.intake.repository.trace import TraceRepository
from nhx.intake.spans.domain import (
    Annotation,
    AnnotationListFilter,
    EvaluatorResult,
    EvaluatorResultListFilter,
    IntakeResponseMode,
    IntakeSession,
    IntakeSpan,
    IntakeTrace,
    SpanGroup,
    SpanListFilter,
    TraceBatch,
    TraceListFilter,
    TraceMetricBucket,
    TraceMetricPoint,
    TraceMode,
)


class SpanNotFoundError(Exception):
    def __init__(self, workspace: str, span_id: str) -> None:
        super().__init__(f"Span {workspace}/{span_id} not found")
        self.workspace = workspace
        self.span_id = span_id


class EvaluatorResultNotFoundError(Exception):
    def __init__(self, workspace: str, evaluator_result_id: str) -> None:
        super().__init__(f"Evaluator result {workspace}/{evaluator_result_id} not found")
        self.workspace = workspace
        self.evaluator_result_id = evaluator_result_id


class TraceNotFoundError(Exception):
    def __init__(self, workspace: str, trace_id: str) -> None:
        super().__init__(f"Trace {workspace}/{trace_id} not found")
        self.workspace = workspace
        self.trace_id = trace_id


class SessionNotFoundError(Exception):
    def __init__(self, workspace: str, session_id: str) -> None:
        super().__init__(f"Session {workspace}/{session_id} not found")
        self.workspace = workspace
        self.session_id = session_id


class SessionInEvaluationError(Exception):
    def __init__(self, workspace: str, session_id: str) -> None:
        super().__init__(
            f"Session {workspace}/{session_id} is associated with an undeleted or unresolved evaluation "
            "and cannot be deleted"
        )


class AnnotationNotFoundError(Exception):
    def __init__(self, workspace: str, annotation_id: str) -> None:
        super().__init__(f"Annotation {workspace}/{annotation_id} not found")
        self.workspace = workspace
        self.annotation_id = annotation_id


class IntakeSpansService:
    def __init__(
        self,
        span_repository: SpanRepository,
        trace_repository: TraceRepository,
        session_repository: SessionRepository,
        evaluator_results_repository: EvaluatorResultsRepository,
        annotations_repository: AnnotationsRepository,
        entity_client: EntityClient | None = None,
    ) -> None:
        self._spans = span_repository
        self._traces = trace_repository
        self._sessions = session_repository
        self._evaluator_results = evaluator_results_repository
        self._annotations = annotations_repository
        self._entity_client = entity_client

    async def ingest_batch(self, batch: TraceBatch) -> None:
        await self._spans.save_spans(batch.spans)
        if batch.evaluator_results:
            await self._evaluator_results.save_evaluator_results(batch.evaluator_results)

    async def ingest_prepared_batch(
        self,
        *,
        span_rows: list[list[Any]],
        evaluator_result_rows: list[list[Any]],
    ) -> None:
        """Insert rows that an ingest worker process built off the event loop.

        Mirrors ``ingest_batch``: spans first, then evaluator results. These
        awaited inserts are the commit point of an ATIF ingest ``201``: it is
        returned only after both succeed. Like ``ingest_batch``, the inserts are
        sequential, so a failure between them can leave spans stored without
        their evaluator results; stable ids make a producer retry converge.
        """
        await self._spans.insert_span_rows(span_rows)
        if evaluator_result_rows:
            await self._evaluator_results.insert_evaluator_result_rows(evaluator_result_rows)

    async def list_spans(
        self,
        *,
        filters: SpanListFilter,
        page: int,
        page_size: int,
        sort: str,
        mode: IntakeResponseMode,
    ) -> PaginatedResult[IntakeSpan]:
        return await self._spans.list_spans(
            filters=filters,
            page=page,
            page_size=page_size,
            sort=sort,
            mode=mode,
        )

    async def list_span_groups(
        self,
        *,
        filters: SpanListFilter,
        group_by: list[str],
        page: int,
        page_size: int,
        sort: str,
    ) -> PaginatedResult[SpanGroup]:
        return await self._spans.list_span_groups(
            filters=filters,
            group_by=group_by,
            page=page,
            page_size=page_size,
            sort=sort,
        )

    async def get_span(self, *, workspace: str, span_id: str, session_id: str | None = None) -> IntakeSpan:
        span = await self._spans.get_span(workspace=workspace, span_id=span_id, session_id=session_id)
        if span is None:
            raise SpanNotFoundError(workspace, span_id)
        return span

    async def list_traces(
        self,
        *,
        filters: TraceListFilter,
        page: int,
        page_size: int,
        sort: str,
        mode: TraceMode,
    ) -> PaginatedResult[IntakeTrace]:
        return await self._traces.list_traces(filters=filters, page=page, page_size=page_size, sort=sort, mode=mode)

    async def get_trace(
        self, *, workspace: str, trace_id: str, mode: TraceMode, session_id: str | None = None
    ) -> IntakeTrace:
        trace = await self._traces.get_trace(workspace=workspace, trace_id=trace_id, mode=mode, session_id=session_id)
        if trace is None:
            raise TraceNotFoundError(workspace, trace_id)
        return trace

    async def trace_metrics(
        self,
        *,
        filters: TraceListFilter,
        bucket: TraceMetricBucket,
        timezone_name: str,
    ) -> list[TraceMetricPoint]:
        return await self._traces.trace_metrics(filters=filters, bucket=bucket, timezone_name=timezone_name)

    async def latest_trace_started_at_by_group(
        self,
        *,
        workspace: str,
        trace_refs_by_group: dict[str, list[str]],
    ) -> dict[str, datetime]:
        return await self._traces.latest_trace_started_at_by_group(
            workspace=workspace,
            trace_refs_by_group=trace_refs_by_group,
        )

    async def get_session(self, *, workspace: str, session_id: str) -> IntakeSession:
        session = await self._sessions.get_session(workspace=workspace, session_id=session_id)
        if session is None:
            raise SessionNotFoundError(workspace, session_id)
        return session

    async def delete_session(self, *, workspace: str, session_id: str) -> None:
        await self.get_session(workspace=workspace, session_id=session_id)
        evaluation_names = await self._sessions.get_evaluation_names(workspace=workspace, session_id=session_id)
        for name in evaluation_names:
            if self._entity_client is None:
                raise SessionInEvaluationError(workspace, session_id)
            try:
                evaluation = await self._entity_client.get(Evaluation, workspace=workspace, name=name)
            except EntityNotFoundError as exc:
                # A missing/renamed entity is not evidence of a confirmed deletion.
                raise SessionInEvaluationError(workspace, session_id) from exc
            if not evaluation.is_deleted:
                raise SessionInEvaluationError(workspace, session_id)
        await self._sessions.delete_session(workspace=workspace, session_id=session_id)

    async def create_evaluator_result(self, result: EvaluatorResult) -> EvaluatorResult:
        """Persist one evaluator_result. Loose target — no span existence check."""

        await self._evaluator_results.save_evaluator_results([result])
        return result

    async def list_evaluator_results(
        self,
        *,
        filters: EvaluatorResultListFilter,
        page: int,
        page_size: int,
        sort: str,
    ) -> PaginatedResult[EvaluatorResult]:
        return await self._evaluator_results.list_evaluator_results(
            filters=filters, page=page, page_size=page_size, sort=sort
        )

    async def get_evaluator_result(self, *, workspace: str, evaluator_result_id: str) -> EvaluatorResult:
        result = await self._evaluator_results.get_evaluator_result(
            workspace=workspace, evaluator_result_id=evaluator_result_id
        )
        if result is None:
            raise EvaluatorResultNotFoundError(workspace, evaluator_result_id)
        return result

    async def list_evaluator_results_for_span(self, *, workspace: str, span_id: str) -> list[EvaluatorResult]:
        return await self._evaluator_results.list_evaluator_results_for_span(workspace=workspace, span_id=span_id)

    async def create_annotation(self, annotation: Annotation) -> Annotation:
        """Persist one annotation. Target validation is permissive — no span/session existence check."""

        await self._annotations.save_annotations([annotation])
        return annotation

    async def get_annotation(self, *, workspace: str, annotation_id: str) -> Annotation:
        annotation = await self._annotations.get_annotation(workspace=workspace, annotation_id=annotation_id)
        if annotation is None:
            raise AnnotationNotFoundError(workspace, annotation_id)
        return annotation

    async def list_annotations(
        self,
        *,
        filters: AnnotationListFilter,
        page: int,
        page_size: int,
        sort: str,
    ) -> PaginatedResult[Annotation]:
        return await self._annotations.list_annotations(filters=filters, page=page, page_size=page_size, sort=sort)

    async def delete_annotation(self, *, workspace: str, annotation_id: str) -> None:
        existing = await self._annotations.get_annotation(workspace=workspace, annotation_id=annotation_id)
        if existing is None:
            raise AnnotationNotFoundError(workspace, annotation_id)
        await self._annotations.soft_delete_annotation(annotation=existing)
