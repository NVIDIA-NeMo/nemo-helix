# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process-pool offload for the CPU-heavy part of the ATIF ingest path.

``prepare_atif_batch`` runs in a ``ProcessPoolExecutor`` worker owned by the
Intake service. It maps one ATIF trajectory into spans and evaluator results and
builds the ClickHouse insert rows, so none of that CPU runs on the event loop.
The awaited insert stays on the loop in the request handler and remains the
commit point for the ``201``.

Constraints, enforced by tests:

- The function is pickled by reference, so it stays module-level and importable
  without the FastAPI request stack.
- It takes and returns only plain picklable data: the trajectory, the ingest
  timestamp, and prebuilt insert rows — never the request, the ClickHouse client,
  or a live session.
- An exception (including ``AtifTrajectoryDepthError``) propagates to the
  awaiting request handler, which fails the request and inserts nothing.
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
    """Hold one worker briefly so a second warm-up task forces a second spawn.

    ``ProcessPoolExecutor`` only spawns a new process when no worker is idle, so
    the short hold makes the warm-up cover the pool's full worker complement.
    """
    time.sleep(seconds)


def prepare_atif_batch(
    workspace: str,
    trajectory: AtifTrajectory,
    ingested_at: datetime,
    max_subagent_depth: int,
) -> PreparedAtifBatch:
    """Map one ATIF trajectory and build its insert rows off the event loop.

    Runs in an ingest worker process, so the arguments and the result must stay
    picklable and the function must not touch request or storage clients.
    """
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
