# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repository interface and read models for Evaluation sessions."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

from nhx.intake.spans.domain import IntakeResponseMode, SpanStatus


class MetricSortTooLargeError(Exception):
    """Raised when a cost/tokens sort exceeds the bounded pre-metrics path."""

    def __init__(self, total: int, limit: int) -> None:
        self.total = total
        self.limit = limit
        super().__init__(f"Metric sort requested on {total} sessions, limit is {limit}")


@dataclass(frozen=True)
class EvaluationSessionRow:
    """One ingested session of an Evaluation."""

    workspace: str
    evaluation_name: str
    session_id: str
    test_case_name: str | None
    trace_id: str
    root_span_id: str
    started_at: datetime
    ended_at: datetime | None
    latency_ms: float | None
    status: SpanStatus
    input: str | None
    output: str | None
    input_tokens: int | None
    output_tokens: int | None
    cached_tokens: int | None
    cost_total_usd: float | None
    evaluator_scores: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationSessionPage:
    rows: list[EvaluationSessionRow]
    total: int


class EvaluationSessionRepository(ABC):
    """Domain-facing interface for Evaluation session reads."""

    @abstractmethod
    async def list_sessions(
        self,
        *,
        workspace: str,
        evaluation_name: str,
        status: SpanStatus | None = None,
        test_case_name: str | None = None,
        page: int,
        page_size: int,
        mode: IntakeResponseMode,
        sort_keys: list[tuple[str, bool]] | None = None,
    ) -> EvaluationSessionPage:
        """Return one page of sessions for an Evaluation."""
        pass
