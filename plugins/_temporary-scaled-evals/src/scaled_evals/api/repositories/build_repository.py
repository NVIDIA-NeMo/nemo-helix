# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import psycopg


@dataclass(frozen=True, slots=True)
class TaskBuildJob:
    task_id: str
    revision: int
    backend: str
    payload: dict[str, Any]
    credentials: dict[str, str]
    object_key: str
    attempt: int


class TaskBuildRepository:
    def __init__(self, conn: psycopg.Connection) -> None:
        self.conn = conn

    def heartbeat_worker(self, worker_id: str) -> None:
        with self.conn.transaction(), self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO service_heartbeats (service, instance_id, heartbeat_at)
                VALUES ('build_worker', %s, NOW())
                ON CONFLICT (service, instance_id) DO UPDATE
                SET heartbeat_at = EXCLUDED.heartbeat_at
                """,
                (worker_id,),
            )

    def record_success(
        self,
        task_id: str,
        revision: int,
        *,
        image_ref: str,
        image_digest: str,
    ) -> None:
        with self.conn.transaction(), self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE task_revisions
                SET status = 'ready',
                    image_ref = %s,
                    image_digest = %s,
                    build_completed_at = NOW()
                WHERE task_id = %s AND revision = %s AND status = 'building'
                """,
                (image_ref, image_digest, task_id, revision),
            )
            if cur.rowcount != 1:
                return
            cur.execute(
                """
                UPDATE tasks SET current_revision = %s, updated_at = NOW()
                WHERE id = %s AND (current_revision IS NULL OR current_revision <= %s)
                """,
                (revision, task_id, revision),
            )

    def record_failure(self, task_id: str, revision: int, build_error: str) -> None:
        with self.conn.transaction(), self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE task_revisions
                SET status = 'failed',
                    build_error = %s,
                    build_completed_at = NOW()
                WHERE task_id = %s AND revision = %s AND status = 'building'
                """,
                (build_error, task_id, revision),
            )

    def claim_next(
        self,
        *,
        worker_id: str,
        claim_timeout: float,
        max_attempts: int,
    ) -> TaskBuildJob | None:
        with self.conn.transaction(), self.conn.cursor() as cur:
            # A worker can die after claiming its final allowed attempt. Once
            # that lease expires there is no safe attempt left to claim, so
            # make the revision terminal instead of leaving it in `building`
            # forever. Live workers keep build_claimed_at fresh via heartbeat.
            cur.execute(
                """
                UPDATE task_revisions
                SET status = 'failed',
                    build_error = COALESCE(NULLIF(build_error, ''),
                        'build worker lease expired after final attempt'),
                    build_completed_at = NOW(),
                    build_claimed_at = NULL,
                    build_claimed_by = NULL,
                    build_next_attempt_at = NULL
                  WHERE status = 'building'
                    AND build_backend IN ('image_builder_service', 'cloudbuild', 'buildkit', 'prebuilt')
                    AND build_attempts >= %s
                  AND build_claimed_at IS NOT NULL
                  AND build_claimed_at < NOW() - (%s * INTERVAL '1 second')
                """,
                (max_attempts, claim_timeout),
            )
            cur.execute(
                """
                WITH candidate AS (
                    SELECT task_id, revision
                    FROM task_revisions
                      WHERE status = 'building'
                        AND build_backend IN ('image_builder_service', 'cloudbuild', 'buildkit', 'prebuilt')
                        AND build_attempts < %s
                      AND (build_next_attempt_at IS NULL OR build_next_attempt_at <= NOW())
                      AND (
                          build_claimed_at IS NULL
                          OR build_claimed_at < NOW() - (%s * INTERVAL '1 second')
                      )
                    ORDER BY build_started_at, task_id, revision
                    LIMIT 1
                    FOR UPDATE SKIP LOCKED
                )
                UPDATE task_revisions r
                SET build_claimed_at = NOW(),
                    build_first_claimed_at = COALESCE(r.build_first_claimed_at, NOW()),
                    build_claimed_by = %s,
                    build_attempts = build_attempts + 1
                FROM candidate
                WHERE r.task_id = candidate.task_id
                  AND r.revision = candidate.revision
                RETURNING r.task_id, r.revision, r.build_backend,
                          r.build_payload, r.build_credentials,
                          r.tarball_object_key, r.build_attempts
                """,
                (max_attempts, claim_timeout, worker_id),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return TaskBuildJob(
            task_id=row["task_id"],
            revision=row["revision"],
            backend=row["build_backend"],
            payload=dict(row["build_payload"] or {}),
            credentials=dict(row["build_credentials"] or {}),
            object_key=row["tarball_object_key"],
            attempt=row["build_attempts"],
        )

    def bind_platform_job(
        self,
        task_id: str,
        revision: int,
        *,
        worker_id: str,
        job_name: str,
    ) -> bool:
        """Transfer a short-lived controller claim to a durable Platform Job."""
        with self.conn.transaction(), self.conn.cursor() as cur:
            # ponytail: reuse the existing claim-owner column during the
            # temporary Postgres bridge. The column disappears at Entity Store
            # cutover, so adding a second durable pointer would only create
            # another migration and dual-write to remove.
            cur.execute(
                """
                UPDATE task_revisions
                SET build_claimed_by = %s,
                    build_claimed_at = NOW()
                WHERE task_id = %s AND revision = %s
                  AND status = 'building' AND build_claimed_by = %s
                """,
                (job_name, task_id, revision, worker_id),
            )
            return cur.rowcount == 1

    def list_platform_jobs(self, *, limit: int = 100) -> list[dict[str, Any]]:
        """List active task builds currently owned by Platform Jobs."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT task_id, revision, build_attempts, build_claimed_by AS job_name
                FROM task_revisions
                WHERE status = 'building'
                  AND build_claimed_by LIKE 'scaled-evals-build-%%'
                ORDER BY build_claimed_at, task_id, revision
                LIMIT %s
                """,
                (limit,),
            )
            return list(cur.fetchall())

    def heartbeat(self, task_id: str, revision: int, *, worker_id: str) -> bool:
        with self.conn.transaction(), self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE task_revisions
                SET build_claimed_at = NOW(),
                    build_first_claimed_at = COALESCE(build_first_claimed_at, NOW())
                WHERE task_id = %s AND revision = %s
                  AND status = 'building' AND build_claimed_by = %s
                """,
                (task_id, revision, worker_id),
            )
            return cur.rowcount == 1

    def complete(
        self,
        task_id: str,
        revision: int,
        *,
        worker_id: str,
        image_ref: str,
        image_digest: str,
    ) -> bool:
        with self.conn.transaction(), self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE task_revisions
                SET status = 'ready', image_ref = %s, image_digest = %s,
                    build_error = NULL, build_completed_at = NOW(),
                    build_claimed_at = NULL, build_claimed_by = NULL,
                    build_next_attempt_at = NULL
                WHERE task_id = %s AND revision = %s
                  AND status = 'building' AND build_claimed_by = %s
                """,
                (image_ref, image_digest, task_id, revision, worker_id),
            )
            updated = cur.rowcount == 1
            if updated:
                cur.execute(
                    """
                    UPDATE tasks SET current_revision = %s, updated_at = NOW()
                    WHERE id = %s AND (current_revision IS NULL OR current_revision <= %s)
                    """,
                    (revision, task_id, revision),
                )
            return updated

    def retry_or_fail(
        self,
        task_id: str,
        revision: int,
        *,
        worker_id: str,
        build_error: str,
        attempt: int,
        max_attempts: int,
        retry_delay: float,
    ) -> bool:
        terminal = attempt >= max_attempts
        with self.conn.transaction(), self.conn.cursor() as cur:
            if terminal:
                cur.execute(
                    """
                    UPDATE task_revisions
                    SET status = 'failed', build_error = %s,
                        build_completed_at = NOW(), build_claimed_at = NULL,
                        build_claimed_by = NULL, build_next_attempt_at = NULL
                    WHERE task_id = %s AND revision = %s
                      AND status = 'building' AND build_claimed_by = %s
                    """,
                    (build_error, task_id, revision, worker_id),
                )
            else:
                cur.execute(
                    """
                    UPDATE task_revisions
                    SET build_error = %s, build_claimed_at = NULL,
                        build_claimed_by = NULL,
                        build_next_attempt_at = NOW() + (%s * INTERVAL '1 second')
                    WHERE task_id = %s AND revision = %s
                      AND status = 'building' AND build_claimed_by = %s
                    """,
                    (build_error, retry_delay, task_id, revision, worker_id),
                )
            return cur.rowcount == 1
