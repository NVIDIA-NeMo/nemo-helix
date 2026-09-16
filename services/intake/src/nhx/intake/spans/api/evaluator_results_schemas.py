# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic schemas for ClickHouse-backed evaluator_results."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Self

from nhx.common.entities.values import DatetimeFilter, Filter
from nhx.intake.spans.domain import EvaluatorResult as DomainEvaluatorResult
from nhx.intake.spans.domain import EvaluatorResultDataType
from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvaluatorResultSortField(StrEnum):
    CREATED_AT_ASC = "created_at"
    CREATED_AT_DESC = "-created_at"
    VALUE_ASC = "value"
    VALUE_DESC = "-value"


class FloatFilter(Filter):
    gte: float | None = Field(
        default=None,
        alias="$gte",
        serialization_alias="$gte",
        description="Filter for results greater than or equal to this value.",
    )
    lte: float | None = Field(
        default=None,
        alias="$lte",
        serialization_alias="$lte",
        description="Filter for results less than or equal to this value.",
    )

    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=(),
        populate_by_name=True,
    )


class EvaluatorResultFilter(BaseModel):
    span_id: str | None = Field(default=None, description="Filter by target span id.")
    session_id: str | None = Field(default=None, description="Filter by target session id.")
    name: str | None = Field(default=None, description="Filter by evaluator/metric name.")
    data_type: EvaluatorResultDataType | None = Field(default=None, description="Filter by data_type.")
    created_by: str | None = Field(default=None, description="Filter by principal/system that wrote the row.")
    value: FloatFilter | None = Field(default=None, description="Filter by numeric value (range supported).")
    created_at: DatetimeFilter | None = Field(
        default=None, description="Filter by row creation time (range supported)."
    )


class EvaluatorResultInput(BaseModel):
    """Request body for POST /evaluator-results.

    Server fills in `evaluator_result_id`, `created_at`, `ingested_at`, and
    `created_by`. Producer supplies the target span (loose target — not
    validated against the spans table), the score, and provenance.
    """

    model_config = ConfigDict(extra="forbid")

    span_id: str = Field(description="Target span id. Not validated against existing spans (loose target policy).")
    session_id: str = Field(
        description="Session id the target span belongs to. Denormalized so session-scoped reads stay fast."
    )
    name: str = Field(description="Evaluator / metric identity (e.g. 'faithfulness/v1').")
    value: float | None = Field(
        default=None,
        description="Numeric value. Required when data_type is NUMERIC or BOOLEAN (0|1).",
    )
    string_value: str | None = Field(
        default=None,
        description="String value. Required when data_type is CATEGORICAL or TEXT.",
    )
    data_type: EvaluatorResultDataType = Field(
        description="Discriminator for which of value / string_value carries the payload."
    )
    comment: str | None = Field(default=None, description="Free-text rationale or explanation.")

    @model_validator(mode="after")
    def _enforce_value_coherence(self) -> Self:
        if self.data_type in (EvaluatorResultDataType.NUMERIC, EvaluatorResultDataType.BOOLEAN):
            if self.value is None:
                raise ValueError(f"`value` is required when data_type is {self.data_type.value}.")
        if self.data_type in (EvaluatorResultDataType.CATEGORICAL, EvaluatorResultDataType.TEXT):
            if self.string_value is None:
                raise ValueError(f"`string_value` is required when data_type is {self.data_type.value}.")
        if self.data_type == EvaluatorResultDataType.BOOLEAN and self.value not in (0, 1, 0.0, 1.0):
            raise ValueError("`value` must be 0 or 1 when data_type is BOOLEAN.")
        return self


class EvaluatorResult(BaseModel):
    """Response model for evaluator_results read endpoints."""

    evaluator_result_id: str
    span_id: str
    session_id: str
    workspace: str
    name: str
    value: float | None = None
    string_value: str | None = None
    data_type: EvaluatorResultDataType
    comment: str | None = None
    created_by: str | None = None
    created_at: datetime
    ingested_at: datetime

    @classmethod
    def from_domain(cls, result: DomainEvaluatorResult) -> Self:
        return cls(
            evaluator_result_id=result.evaluator_result_id,
            span_id=result.span_id,
            session_id=result.session_id,
            workspace=result.workspace,
            name=result.name,
            value=result.value,
            string_value=result.string_value,
            data_type=result.data_type,
            comment=result.comment,
            created_by=result.created_by,
            created_at=result.created_at,
            ingested_at=result.ingested_at,
        )
