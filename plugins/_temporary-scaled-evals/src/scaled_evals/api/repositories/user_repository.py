# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import psycopg


class UserRepository:
    """Owner-scoped and cluster-wide views over live evaluation and task state."""

    def __init__(self, conn: psycopg.Connection) -> None:
        self.conn = conn

    def summary(self, owner_id: str) -> dict:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    COUNT(*) FILTER (WHERE status IN ('blocked', 'provisioning', 'running'))
                        AS active,
                    COUNT(*) FILTER (WHERE status = 'queued') AS queued,
                    COUNT(*) FILTER (
                        WHERE status = 'failed' AND updated_at >= NOW() - INTERVAL '24 hours'
                    ) AS failed_24h,
                    COUNT(*) FILTER (
                        WHERE status = 'succeeded' AND updated_at >= NOW() - INTERVAL '24 hours'
                    ) AS succeeded_24h
                FROM evaluations
                WHERE owner_id = %s AND deleted_at IS NULL
                """,
                (owner_id,),
            )
            evaluations = dict(cur.fetchone())
            cur.execute(
                """
                SELECT
                    COUNT(*) FILTER (WHERE r.status = 'ready') AS ready,
                    COUNT(*) FILTER (WHERE r.status IN ('pending', 'uploading', 'building'))
                        AS building
                FROM tasks t
                LEFT JOIN LATERAL (
                    SELECT status FROM task_revisions
                    WHERE task_id = t.id ORDER BY revision DESC LIMIT 1
                ) r ON TRUE
                WHERE t.owner_id = %s AND t.deleted_at IS NULL
                """,
                (owner_id,),
            )
            tasks = dict(cur.fetchone())
        return {
            "evaluations": {key: int(value or 0) for key, value in evaluations.items()},
            "tasks": {key: int(value or 0) for key, value in tasks.items()},
        }

    def capacity(self) -> dict[str, int]:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    COUNT(*) FILTER (WHERE status IN ('blocked', 'provisioning', 'running'))
                        AS active_runs,
                    COUNT(*) FILTER (WHERE status = 'queued') AS queued_runs,
                    COALESCE(SUM(parallelism) FILTER (
                        WHERE status IN ('provisioning', 'running')
                    ), 0) AS active_slots
                FROM evaluations WHERE deleted_at IS NULL
                """
            )
            row = cur.fetchone()
        return {key: int(value or 0) for key, value in row.items()}

    def live_usage(self) -> dict:
        """Return queued and active run counts overall and per owner, plus the task total."""
        live_columns = """
            COUNT(*) FILTER (WHERE status = 'queued') AS queued_runs,
            COUNT(*) FILTER (WHERE status IN ('blocked', 'provisioning', 'running')) AS active_runs,
            COALESCE(SUM(parallelism), 0) AS total_parallelism
        """
        live_filter = "deleted_at IS NULL AND status IN ('queued', 'blocked', 'provisioning', 'running')"
        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT {live_columns}, (SELECT COUNT(*) FROM tasks WHERE deleted_at IS NULL) AS total_tasks "
                f"FROM evaluations WHERE {live_filter}"
            )
            summary = {key: int(value or 0) for key, value in cur.fetchone().items()}
            cur.execute(
                f"SELECT owner_id, {live_columns}, MAX(created_at) AS last_run_at "
                f"FROM evaluations WHERE {live_filter} GROUP BY owner_id"
            )
            actors = cur.fetchall()
        return {**summary, "actors": actors}

    def quota_usage(self, owner_id: str) -> dict[str, int]:
        """Return real owner-backed counts for ``GET /users/me``."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    COUNT(*) FILTER (
                        WHERE status IN ('queued', 'provisioning', 'running')
                    ) AS evaluations_active,
                    COALESCE(SUM(parallelism) FILTER (
                        WHERE status IN ('provisioning', 'running')
                    ), 0) AS sandbox_slots_active
                FROM evaluations
                WHERE owner_id = %s AND deleted_at IS NULL
                """,
                (owner_id,),
            )
            usage = dict(cur.fetchone())
            cur.execute(
                """
                SELECT COUNT(*) AS tasks_owned
                FROM tasks
                WHERE owner_id = %s AND deleted_at IS NULL
                """,
                (owner_id,),
            )
            usage.update(cur.fetchone())
        return {key: int(value or 0) for key, value in usage.items()}

    def recent_activity(self, owner_id: str, *, limit: int) -> list[dict]:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, kind, name, status, created_at FROM (
                    SELECT id, 'evaluation' AS kind, name, status::text AS status, created_at
                    FROM evaluations WHERE owner_id = %s AND deleted_at IS NULL
                    UNION ALL
                    SELECT t.id, 'task' AS kind, t.name, r.status::text, t.created_at
                    FROM tasks t
                    LEFT JOIN LATERAL (
                        SELECT status FROM task_revisions
                        WHERE task_id = t.id ORDER BY revision DESC LIMIT 1
                    ) r ON TRUE
                    WHERE t.owner_id = %s AND t.deleted_at IS NULL
                ) activity
                ORDER BY created_at DESC, id DESC
                LIMIT %s
                """,
                (owner_id, owner_id, limit),
            )
            return cur.fetchall()
