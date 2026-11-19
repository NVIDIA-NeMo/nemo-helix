# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phoenix-shaped wire data; persistence and normalization belong to Intake."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Literal, Self

from nemo_helix_plugin.intake.types import DirectSpanInput, Span, SpanKind
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, JsonValue, model_validator


def _intake_time(value: datetime) -> datetime:
    # ClickHouse stores these timestamps in UTC; older Intake read responses
    # omit the offset. Never apply the adapter host's local timezone.
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


class SpanContext(BaseModel):
    model_config = ConfigDict(extra="forbid")
    trace_id: str = Field(min_length=1, max_length=256)
    span_id: str = Field(min_length=1, max_length=256)


class PhoenixSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(max_length=4096)
    context: SpanContext
    span_kind: SpanKind = "UNKNOWN"
    parent_id: str | None = Field(default=None, min_length=1, max_length=256)
    start_time: AwareDatetime
    end_time: AwareDatetime | None = None
    status_code: Literal["OK", "ERROR", "UNSET"] = "UNSET"
    status_message: str = Field(default="", max_length=4096)
    attributes: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_interval(self) -> Self:
        if self.end_time is not None and self.end_time < self.start_time:
            raise ValueError("end_time must not precede start_time")
        if self.parent_id == self.context.span_id:
            raise ValueError("A span cannot parent itself")
        return self

    def to_intake(self, project: str) -> DirectSpanInput:
        attributes = dict(self.attributes)
        # The selected project wins over every semantic project alias.
        for key in ("gen_ai.project", "project", "project.name"):
            attributes.pop(key, None)
        if project:
            attributes["project.name"] = project
        if self.status_message:
            attributes["phoenix.status_message"] = self.status_message
            if self.status_code == "ERROR":
                attributes["error.message"] = self.status_message
        session = attributes.get("session.id")
        status: Literal["success", "error", "unknown"] = (
            "success" if self.status_code == "OK" else "error" if self.status_code == "ERROR" else "unknown"
        )
        return DirectSpanInput(
            span_id=self.context.span_id,
            trace_id=self.context.trace_id,
            parent_span_id=self.parent_id,
            session_id=session if isinstance(session, str) and session else None,
            name=self.name,
            kind=self.span_kind,
            status=status,
            started_at=self.start_time,
            ended_at=self.end_time,
            input=attributes.get("input.value"),
            output=attributes.get("output.value"),
            attributes=attributes,
        )

    @classmethod
    def from_intake(cls, span: Span) -> Self:
        raw = json.loads(span.raw_attributes) if span.raw_attributes else {}
        if not isinstance(raw, dict):
            raise ValueError("Intake raw_attributes must contain an object")
        attributes: dict[str, JsonValue] = raw
        # Intake extracts semantic attributes into typed columns. Restore the
        # OpenInference keys the viewer understands, without flattening raw JSON.
        for key, value in {
            "input.value": span.input,
            "output.value": span.output,
            "session.id": span.session_id,
            "llm.model_name": span.model,
            "llm.provider": span.provider,
            "llm.token_count.prompt": span.input_tokens,
            "llm.token_count.completion": span.output_tokens,
            "llm.token_count.total": span.total_tokens,
            "llm.cost.total": span.cost_total_usd,
            "tool.name": span.tool_name,
            "project.name": span.project,
        }.items():
            if value is not None:
                attributes.setdefault(key, value)
        message = attributes.get("phoenix.status_message", span.error_message or "")
        return cls(
            name=span.name or span.span_id,
            context=SpanContext(trace_id=span.trace_id or span.session_id, span_id=span.span_id),
            span_kind=span.kind,
            parent_id=span.parent_span_id,
            start_time=_intake_time(span.started_at),
            end_time=_intake_time(span.ended_at) if span.ended_at is not None else None,
            status_code="OK"
            if span.status == "success"
            else "ERROR"
            if span.status in {"error", "cancelled"}
            else "UNSET",
            status_message=message if isinstance(message, str) else "",
            attributes=attributes,
        )


class SpanBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    spans: list[PhoenixSpan] = Field(min_length=1, max_length=1000)


class Project(BaseModel):
    # Empty name represents Intake spans without a project.
    name: str


class ProjectsPage(BaseModel):
    data: list[Project]
    next_cursor: str | None


class SpansPage(BaseModel):
    data: list[PhoenixSpan]
    next_cursor: str | None
