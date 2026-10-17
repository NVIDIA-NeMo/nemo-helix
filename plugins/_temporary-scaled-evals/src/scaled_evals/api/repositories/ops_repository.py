# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, cast

import psycopg
from scaled_evals.api.benchmark_run_store import BenchmarkRunStore
from scaled_evals.api.evaluation_store import EvaluationStore
from scaled_evals.api.task_store import TaskStore


class OperationsRepository:
    def __init__(self, conn: psycopg.Connection[Any]) -> None:
        self.conn = conn

    def ping(self) -> None:
        self.conn.execute("SELECT 1")

    def assert_schema_compatible(self) -> None:
        # Exercise the columns the request path selects so a DB missing a
        # shipped migration fails readiness instead of serving 500s.
        self.conn.execute(
            "SELECT benchmark_run_id, generation, status, claim_token, sha256, cleanup_checked_at FROM benchmark_run_archives WHERE false"
        )

    def heartbeat_service(self, service: str, instance_id: str) -> None:
        """Record liveness for a scaled-evals background service."""
        with self.conn.transaction(), self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO service_heartbeats (service, instance_id, heartbeat_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (service, instance_id) DO UPDATE
                SET heartbeat_at = EXCLUDED.heartbeat_at
                """,
                (service, instance_id),
            )

    def has_fresh_service_heartbeat(self, service: str, *, stale_seconds: float) -> bool:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM service_heartbeats
                    WHERE service = %s
                      AND heartbeat_at >= NOW() - (%s * INTERVAL '1 second')
                ) AS is_fresh
                """,
                (service, stale_seconds),
            )
            row = cast(dict[str, Any] | None, cur.fetchone())
        return bool(row and row["is_fresh"])

    def evaluation_status_counts(self) -> dict[str, int]:
        return EvaluationStore().status_counts()

    def fleet_totals(self) -> dict[str, int]:
        """Return all-time low-cardinality control-plane resource totals."""
        return {
            "task_definitions": TaskStore().count(include_deleted=True),
            **EvaluationStore().fleet_totals(),
            "benchmark_runs": BenchmarkRunStore().count(),
        }

    def dispatch_observability_snapshot(
        self,
        *,
        stuck_queued_seconds: float,
        stuck_provisioning_seconds: float,
        stuck_running_seconds: float,
        stale_worker_seconds: float,
    ) -> dict:
        """Return low-cardinality queue/worker/stuck-job diagnostics for metrics."""
        return EvaluationStore().dispatch_snapshot(
            stuck_queued_seconds=stuck_queued_seconds,
            stuck_provisioning_seconds=stuck_provisioning_seconds,
            stuck_running_seconds=stuck_running_seconds,
            stale_worker_seconds=stale_worker_seconds,
        )

    def ready_task_pack_revisions(self, *, limit: int) -> list[dict]:
        return TaskStore().ready_task_pack_revisions(limit=limit)
