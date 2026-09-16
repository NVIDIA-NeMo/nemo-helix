# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic schemas for ClickHouse-backed Intake session details."""

from datetime import datetime
from typing import Self

from nhx.intake.spans.domain import IntakeSession, SpanStatus
from pydantic import BaseModel, Field


class Session(BaseModel):
    """Aggregate telemetry for one Intake session; does not include traces or span payloads."""

    id: str
    workspace: str
    started_at: datetime
    ended_at: datetime | None = None
    duration_ms: float | None = None
    status: SpanStatus
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cached_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    cost_usd: float | None = None
    cost_input_usd: float | None = None
    cost_output_usd: float | None = None
    trace_count: int = Field(ge=0)
    span_count: int = Field(ge=0)

    @classmethod
    def from_domain(cls, session: IntakeSession) -> Self:
        return cls.model_validate(session, from_attributes=True)
