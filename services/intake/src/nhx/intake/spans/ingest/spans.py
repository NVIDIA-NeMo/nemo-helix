# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider-neutral JSON span ingest for ClickHouse-backed Intake spans."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Annotated, Self

from fastapi import APIRouter, Depends, HTTPException, Response, status
from nhx.intake.config import DEFAULT_SPAN_RETENTION_DAYS
from nhx.intake.spans.api.dependencies import DenormalizerDep, SpansServiceDep, require_workspace_access
from nhx.intake.spans.domain import IntakeSpan, SpanKind, SpanStatus, TraceBatch
from nhx.intake.spans.span_attribute_bags import DIRECT_INGEST_RAW_ATTRIBUTES_KEY
from nhx.intake.spans.span_semantic_attributes import SpanSemanticAttributes
from nhx.intake.spans.storage import json_dumps_preserve, utc_now
from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

router = APIRouter(dependencies=[Depends(require_workspace_access)])
API_TAG = "Ingest"

DIRECT_SPAN_IDENTIFIER_MAX_LENGTH = 1024
DIRECT_SPAN_NAME_MAX_LENGTH = 1024
DirectSpanIdentifier = Annotated[str, Field(min_length=1, max_length=DIRECT_SPAN_IDENTIFIER_MAX_LENGTH)]
DirectSpanName = Annotated[str, Field(max_length=DIRECT_SPAN_NAME_MAX_LENGTH)]
SourceName = Annotated[
    str,
    Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$",
        description="Stable name for the source trace store, such as `langsmith` or `mlflow`.",
    ),
]


class DirectSpanInput(BaseModel):
    """One provider-neutral span supplied by a historical trace importer."""

    model_config = ConfigDict(extra="forbid")

    span_id: DirectSpanIdentifier
    trace_id: DirectSpanIdentifier
    session_id: DirectSpanIdentifier | None = None
    parent_span_id: DirectSpanIdentifier | None = None
    name: DirectSpanName = ""
    kind: SpanKind = SpanKind.UNKNOWN
    status: SpanStatus = SpanStatus.UNKNOWN
    started_at: datetime
    ended_at: datetime | None = None
    input: JsonValue | None = None
    output: JsonValue | None = None
    attributes: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("started_at", "ended_at")
    @classmethod
    def require_timezone(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamps must include a UTC offset")
        return value.astimezone(timezone.utc)

    @field_validator("attributes")
    @classmethod
    def validate_semantic_attributes(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        SpanSemanticAttributes.from_source_attributes(value)
        return value

    @model_validator(mode="after")
    def validate_span(self) -> Self:
        if self.parent_span_id == self.span_id:
            raise ValueError("parent_span_id must differ from span_id")
        if self.ended_at is not None and self.ended_at < self.started_at:
            raise ValueError("ended_at must be greater than or equal to started_at")
        return self


class DirectSpansIngestRequest(BaseModel):
    """A validated batch of spans from one source trace store."""

    model_config = ConfigDict(extra="forbid")

    source: SourceName
    spans: list[DirectSpanInput] = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def reject_duplicate_span_identities(self) -> Self:
        identities = [(span.trace_id, span.span_id) for span in self.spans]
        if len(identities) != len(set(identities)):
            raise ValueError("spans must not contain duplicate (trace_id, span_id) identities")
        return self


@router.post(
    "/v2/workspaces/{workspace}/ingest/spans",
    tags=[API_TAG],
    status_code=status.HTTP_201_CREATED,
    response_class=Response,
)
async def ingest_spans(
    workspace: str,
    body: DirectSpansIngestRequest,
    service: SpansServiceDep,
    denormalizer: DenormalizerDep,
) -> Response:
    ingested_at = utc_now()
    expired = [
        span
        for span in body.spans
        if span.started_at.date() + timedelta(days=DEFAULT_SPAN_RETENTION_DAYS) <= ingested_at.date()
    ]
    if expired:
        oldest = min(expired, key=lambda span: span.started_at)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"{len(expired)} span(s) fall outside Intake's {DEFAULT_SPAN_RETENTION_DAYS}-day "
                f"ClickHouse retention window; oldest span {oldest.span_id!r} started at "
                f"{oldest.started_at.isoformat()}. Increase the spans and trace_index table TTL "
                "before importing older data."
            ),
        )
    converted = [
        direct_span_to_domain(
            workspace=workspace,
            source=body.source,
            span=span,
            ingested_at=ingested_at,
        )
        for span in body.spans
    ]
    spans = [item for item, _ in converted]
    await service.ingest_batch(TraceBatch(spans=spans))
    if denormalizer is not None:
        evaluation_names = {semantic.evaluation_name for _, semantic in converted if semantic.evaluation_name}
        for evaluation_name in evaluation_names:
            denormalizer.mark_dirty(workspace=workspace, evaluation_name=evaluation_name)
    return Response(status_code=status.HTTP_201_CREATED)


def direct_span_to_domain(
    *,
    workspace: str,
    source: str,
    span: DirectSpanInput,
    ingested_at: datetime,
) -> tuple[IntakeSpan, SpanSemanticAttributes]:
    semantic_attributes, consumed_keys = SpanSemanticAttributes.from_source_attributes(span.attributes)
    attribute_bags = semantic_attributes.to_bags()
    raw_attributes = {key: value for key, value in span.attributes.items() if key not in consumed_keys}
    if raw_attributes:
        attribute_bags.put_json(DIRECT_INGEST_RAW_ATTRIBUTES_KEY, raw_attributes)
    intake_span = IntakeSpan(
        workspace=workspace,
        session_id=span.session_id or span.trace_id,
        trace_id=span.trace_id,
        source_format=source,
        external_span_id=span.span_id,
        external_parent_span_id=span.parent_span_id or "",
        kind=span.kind,
        name=span.name,
        status=span.status,
        start_time=span.started_at,
        end_time=span.ended_at,
        attributes_string=attribute_bags.string,
        attributes_number=attribute_bags.number,
        attributes_bool=attribute_bags.boolean,
        input=""
        if span.input is None
        else span.input
        if isinstance(span.input, str)
        else json_dumps_preserve(span.input),
        output=(
            ""
            if span.output is None
            else span.output
            if isinstance(span.output, str)
            else json_dumps_preserve(span.output)
        ),
        event_ts=ingested_at,
    )
    return intake_span, semantic_attributes
