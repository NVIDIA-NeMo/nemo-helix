# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic schemas for time-bucketed Intake trace metrics."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Self

from nhx.intake.spans.domain import TraceMetricPoint
from pydantic import BaseModel, Field


class TraceMetricBucketParam(StrEnum):
    TOTAL = "total"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


class TokenRollupResponse(BaseModel):
    sum: int | None = Field(default=None, ge=0)
    mean: float | None = None
    p90: float | None = None
    p99: float | None = None


class CostRollupResponse(BaseModel):
    sum: float | None = None
    mean: float | None = None
    p90: float | None = None
    p99: float | None = None


class LatencyRollupResponse(BaseModel):
    mean: float | None = None
    p50: float | None = None
    p90: float | None = None
    p95: float | None = None
    p99: float | None = None


class TraceMetricPointResponse(BaseModel):
    bucket_start: datetime | None = Field(
        default=None,
        description="Start of the bucket in the requested timezone. Omitted when bucket=total.",
    )
    run_count: int = Field(ge=0, description="Agent runs started in this bucket.")
    failed_run_count: int = Field(ge=0, description="Runs whose root span ended in error.")
    input_tokens: TokenRollupResponse
    output_tokens: TokenRollupResponse
    cached_tokens: TokenRollupResponse
    total_tokens: TokenRollupResponse
    cost_usd: CostRollupResponse
    latency_ms: LatencyRollupResponse

    @classmethod
    def from_domain(cls, point: TraceMetricPoint) -> Self:
        return cls.model_validate(point, from_attributes=True)


class TraceMetrics(BaseModel):
    bucket: TraceMetricBucketParam
    timezone: str
    data: list[TraceMetricPointResponse]
