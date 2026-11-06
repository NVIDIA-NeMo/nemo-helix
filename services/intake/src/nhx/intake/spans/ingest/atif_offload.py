# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ATIF mapping and row building, run in the Intake service's ingest process pool.

Everything here is pickled by reference into worker processes, so it stays module-level and
takes and returns only plain picklable data, never request or storage clients.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from nhx.intake.repository.clickhouse.evaluator_results import evaluator_result_rows_for_insert
from nhx.intake.repository.clickhouse.span import span_rows_for_insert
from nhx.intake.spans.ingest.atif_domain import AtifTrajectory
from nhx.intake.spans.ingest.atif_mapping import (
    trajectory_to_evaluator_results,
    trajectory_to_spans,
)


@dataclass(frozen=True)
class PreparedAtifBatch:
    """Picklable result of mapping one ATIF trajectory into ClickHouse insert rows."""

    span_rows: list[list[Any]] = field(default_factory=list)
    evaluator_result_rows: list[list[Any]] = field(default_factory=list)


def atif_ingest_worker_ready() -> None:
    """Instant task that pre-spawns an ingest worker process during startup."""


def atif_ingest_worker_hold(seconds: float) -> None:
    """Keep a worker busy so the next warm-up task forces another spawn."""
    time.sleep(seconds)


def prepare_atif_batch(
    workspace: str,
    trajectory: AtifTrajectory,
    ingested_at: datetime,
    max_subagent_depth: int,
) -> PreparedAtifBatch:
    """Map one ATIF trajectory and build its insert rows."""
    spans = trajectory_to_spans(
        workspace=workspace,
        trajectory=trajectory,
        ingested_at=ingested_at,
        max_subagent_depth=max_subagent_depth,
    )
    evaluator_results = trajectory_to_evaluator_results(
        workspace=workspace,
        trajectory=trajectory,
        spans=spans,
        ingested_at=ingested_at,
        max_subagent_depth=max_subagent_depth,
    )
    return PreparedAtifBatch(
        span_rows=span_rows_for_insert(spans),
        evaluator_result_rows=evaluator_result_rows_for_insert(evaluator_results),
    )
