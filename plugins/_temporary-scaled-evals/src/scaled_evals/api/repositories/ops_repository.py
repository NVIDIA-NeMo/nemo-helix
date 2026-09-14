# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import psycopg


class OperationsRepository:
    def __init__(self, conn: psycopg.Connection[Any]) -> None:
        self.conn = conn

    def ping(self) -> None:
        self.conn.execute("SELECT 1")

    def assert_schema_compatible(self) -> None:
        # Exercise the exact evaluation columns the request path selects so a DB
        # missing a shipped migration fails readiness instead of serving 500s.
        from scaled_evals.api.repositories.evaluation_repository import EVALUATION_COLUMNS

        self.conn.execute(f"SELECT {EVALUATION_COLUMNS} FROM evaluations WHERE false")
        self.conn.execute("SELECT id FROM evaluation_execution_cleanups WHERE false")
        self.conn.execute(
            "SELECT benchmark_run_id, generation, status, claim_token FROM benchmark_run_archives WHERE false"
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
            row = cur.fetchone()
        return bool(row and row["is_fresh"])

    def evaluation_status_counts(self) -> dict[str, int]:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT status::text AS status, COUNT(*) AS count
                FROM evaluations
                WHERE deleted_at IS NULL
                GROUP BY status
                """
            )
            return {str(row["status"]): int(row["count"]) for row in cur.fetchall()}

    def fleet_totals(self) -> dict[str, int]:
        """Return all-time low-cardinality control-plane resource totals."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM tasks) AS task_definitions,
                    (SELECT COUNT(DISTINCT task_id) FROM evaluations) AS tasks_run,
                    (SELECT COUNT(*) FROM evaluations) AS evaluation_jobs,
                    (SELECT COALESCE(SUM(current_execution), 0) FROM evaluations)
                        AS evaluation_executions,
                    (SELECT COALESCE(SUM(n_trials), 0) FROM evaluations) AS completed_trials,
                    (SELECT COUNT(*) FROM benchmark_runs) AS benchmark_runs
                """
            )
            row = cur.fetchone() or {}
        return {key: int(value or 0) for key, value in row.items()}

    def dispatch_observability_snapshot(
        self,
        *,
        stuck_queued_seconds: float,
        stuck_provisioning_seconds: float,
        stuck_running_seconds: float,
        stale_worker_seconds: float,
    ) -> dict:
        """Return low-cardinality queue/worker/stuck-job diagnostics for metrics."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                WITH active AS (
                    SELECT
                        status::text AS status,
                        runtime,
                        EXTRACT(EPOCH FROM (NOW() - created_at)) AS age_seconds,
                        EXTRACT(EPOCH FROM (NOW() - dispatch_claimed_at)) AS lease_age_seconds,
                        dispatch_claimed_by,
                        dispatch_claimed_at,
                        CASE
                            WHEN status = 'queued'
                                THEN created_at < NOW() - (%s * INTERVAL '1 second')
                            WHEN status = 'provisioning'
                                THEN updated_at < NOW() - (%s * INTERVAL '1 second')
                            WHEN status = 'running'
                                THEN updated_at < NOW() - (%s * INTERVAL '1 second')
                            ELSE false
                        END AS stuck
                    FROM evaluations
                    WHERE deleted_at IS NULL
                      AND status IN ('queued', 'provisioning', 'running')
                )
                SELECT
                    COALESCE(MAX(age_seconds) FILTER (WHERE status = 'queued'), 0)
                        AS oldest_queued_seconds,
                    COUNT(*) FILTER (WHERE status = 'queued' AND dispatch_claimed_by IS NULL)
                        AS unclaimed_queued,
                    COUNT(DISTINCT dispatch_claimed_by) FILTER (
                        WHERE dispatch_claimed_by IS NOT NULL
                          AND dispatch_claimed_at >= NOW() - (%s * INTERVAL '1 second')
                    ) AS live_workers,
                    COUNT(DISTINCT dispatch_claimed_by) FILTER (
                        WHERE dispatch_claimed_by IS NOT NULL
                          AND dispatch_claimed_at < NOW() - (%s * INTERVAL '1 second')
                    ) AS stale_workers,
                    COALESCE(MAX(lease_age_seconds) FILTER (
                        WHERE dispatch_claimed_by IS NOT NULL
                    ), 0) AS oldest_worker_lease_seconds
                FROM active
                """,
                (
                    stuck_queued_seconds,
                    stuck_provisioning_seconds,
                    stuck_running_seconds,
                    stale_worker_seconds,
                    stale_worker_seconds,
                ),
            )
            summary = cur.fetchone() or {}

            cur.execute(
                """
                WITH active AS (
                    SELECT
                        status::text AS status,
                        runtime,
                        CASE
                            WHEN status = 'queued'
                                THEN created_at < NOW() - (%s * INTERVAL '1 second')
                            WHEN status = 'provisioning'
                                THEN updated_at < NOW() - (%s * INTERVAL '1 second')
                            WHEN status = 'running'
                                THEN updated_at < NOW() - (%s * INTERVAL '1 second')
                            ELSE false
                        END AS stuck
                    FROM evaluations
                    WHERE deleted_at IS NULL
                      AND status IN ('queued', 'provisioning', 'running')
                )
                SELECT status, runtime, COUNT(*) AS count
                FROM active
                WHERE stuck
                GROUP BY status, runtime
                """,
                (stuck_queued_seconds, stuck_provisioning_seconds, stuck_running_seconds),
            )
            stuck_jobs = [
                {
                    "status": str(row["status"]),
                    "runtime": str(row["runtime"]),
                    "count": int(row["count"]),
                }
                for row in cur.fetchall()
            ]

            cur.execute(
                """
                SELECT runtime, COUNT(*) AS count
                FROM evaluations
                WHERE deleted_at IS NULL
                  AND status = 'failed'
                  AND updated_at >= NOW() - INTERVAL '1 hour'
                  AND (
                      status_detail ILIKE '%%backend%%'
                      OR status_detail ILIKE '%%launch%%'
                      OR status_detail ILIKE '%%status read%%'
                      OR status_detail ILIKE '%%timed out%%'
                  )
                GROUP BY runtime
                """
            )
            backend_failures = [{"runtime": str(row["runtime"]), "count": int(row["count"])} for row in cur.fetchall()]

            cur.execute(
                """
                SELECT status::text AS status, COUNT(*) AS count
                FROM evaluation_runtime_resources
                WHERE kind = 'switchyard'
                  AND status IN ('draining', 'delete_failed')
                GROUP BY status
                """
            )
            switchyard_teardown = {str(row["status"]): int(row["count"]) for row in cur.fetchall()}

        return {
            "oldest_queued_seconds": float(summary.get("oldest_queued_seconds") or 0),
            "unclaimed_queued": int(summary.get("unclaimed_queued") or 0),
            "live_workers": int(summary.get("live_workers") or 0),
            "stale_workers": int(summary.get("stale_workers") or 0),
            "oldest_worker_lease_seconds": float(summary.get("oldest_worker_lease_seconds") or 0),
            "stuck_jobs": stuck_jobs,
            "backend_failures": backend_failures,
            "switchyard_teardown": switchyard_teardown,
        }

    def ready_task_pack_revisions(self, *, limit: int) -> list[dict]:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT t.owner_id, r.task_id, r.revision, r.tarball_object_key
                FROM task_revisions r
                JOIN tasks t ON t.id = r.task_id
                WHERE t.deleted_at IS NULL
                  AND r.status = 'ready'
                ORDER BY t.updated_at DESC, r.task_id, r.revision DESC
                LIMIT %s
                """,
                (limit,),
            )
            return cur.fetchall()
