# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from uuid import uuid4

import psycopg
from psycopg.types.json import Jsonb

from scaled_evals.api.repositories.base_repository import Conflict, NotFound
from scaled_evals.models.benchmark_archives import BenchmarkArchiveMember

_MEMBER_COLUMNS = """
    id, task_id, task_revision, status, current_execution,
    archive_object_key, archive_built_at, archive_size_bytes
"""


class BenchmarkArchiveRepository:
    def __init__(self, conn: psycopg.Connection[Any]) -> None:
        self.conn = conn

    def get(self, run_id: str) -> dict | None:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT a.* FROM benchmark_run_archives a JOIN benchmark_runs b "
                "ON b.id = a.benchmark_run_id WHERE b.id = %s AND b.deleted_at IS NULL",
                (run_id,),
            )
            return cur.fetchone()

    def request(self, run_id: str, *, force: bool = False) -> dict:
        with self.conn.transaction(), self.conn.cursor() as cur:
            cur.execute(
                "SELECT framework FROM benchmark_runs WHERE id = %s AND deleted_at IS NULL FOR UPDATE",
                (run_id,),
            )
            run = cur.fetchone()
            if run is None:
                raise NotFound("benchmark_run", "benchmark run not found")
            if run["framework"] != "harbor":
                raise Conflict("unsupported_framework", "benchmark archives require Harbor")
            existing = self.get(run_id)
            if existing and (
                existing["status"] in {"queued", "building"} or (existing["status"] == "ready" and not force)
            ):
                return existing
            cur.execute(
                f"SELECT {_MEMBER_COLUMNS}, "
                "execution_snapshot #>> '{task,slug}' AS task_slug, "
                "execution_snapshot #>> '{task,name}' AS task_name, "
                "archive_status, evidence_status, "
                "cancel_teardown_status FROM evaluations "
                "WHERE benchmark_run_id = %s AND deleted_at IS NULL ORDER BY id FOR SHARE",
                (run_id,),
            )
            rows = cur.fetchall()
            if not rows:
                raise Conflict("empty_run", "benchmark run has no evaluations")
            for row in rows:
                if row["status"] not in {"succeeded", "failed", "cancelled"}:
                    raise Conflict("run_not_terminal", "all member evaluations must be terminal")
                if (
                    row["archive_status"] != "ready"
                    or not row["archive_object_key"]
                    or row["evidence_status"] != "ready"
                    or row["cancel_teardown_status"] in {"pending", "failed"}
                ):
                    raise Conflict(
                        "member_artifacts_not_ready",
                        f"evaluation {row['id']} must finish evidence, teardown and archive generation before export",
                    )
            members = [BenchmarkArchiveMember.model_validate(row).model_dump(mode="json") for row in rows]
            cur.execute(
                """
                INSERT INTO benchmark_run_archives
                    (benchmark_run_id, generation, status, members)
                VALUES (%s, %s, 'queued', %s)
                ON CONFLICT (benchmark_run_id) DO UPDATE SET
                    generation = EXCLUDED.generation, status = 'queued',
                    members = EXCLUDED.members, requested_at = NOW(),
                    claimed_at = NULL, claim_token = NULL, attempts = 0,
                    object_key = NULL, size_bytes = NULL, sha256 = NULL, partial = NULL,
                    built_at = NULL, error = NULL
                RETURNING *
                """,
                (run_id, str(uuid4()), Jsonb(members)),
            )
            row = cur.fetchone()
            assert row is not None
            return row

    def claim(self, *, claim_timeout: float) -> dict | None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                WITH candidate AS (
                    SELECT a.benchmark_run_id FROM benchmark_run_archives a
                    JOIN benchmark_runs b ON b.id = a.benchmark_run_id
                    WHERE b.deleted_at IS NULL AND (a.status = 'queued' OR
                        (a.status = 'building' AND
                         a.claimed_at < NOW() - (%s * INTERVAL '1 second')))
                    ORDER BY a.requested_at FOR UPDATE OF a SKIP LOCKED LIMIT 1
                )
                UPDATE benchmark_run_archives a SET
                    status = CASE WHEN a.attempts >= 3 THEN 'failed' ELSE 'building' END,
                    error = CASE WHEN a.attempts >= 3 THEN 'archive worker lease expired'
                                 ELSE NULL END,
                    claimed_at = NOW(), claim_token = %s, attempts = a.attempts + 1
                FROM candidate c WHERE a.benchmark_run_id = c.benchmark_run_id
                RETURNING a.*
                """,
                (claim_timeout, str(uuid4())),
            )
            return cur.fetchone()

    def claim_cleanup(self, *, interval_seconds: float) -> str | None:
        """Schedule another bounded sweep, including failed or soft-deleted runs.

        The timestamp is a throttle, not a completion marker: crashes, transient
        delete failures and uploads completing late are retried on later sweeps.
        """
        with self.conn.cursor() as cur:
            cur.execute(
                """
                WITH candidate AS (
                    SELECT benchmark_run_id FROM benchmark_run_archives
                    WHERE cleanup_checked_at IS NULL OR
                        cleanup_checked_at < NOW() - (%s * INTERVAL '1 second')
                    ORDER BY cleanup_checked_at NULLS FIRST, requested_at, benchmark_run_id
                    FOR UPDATE SKIP LOCKED LIMIT 1
                )
                UPDATE benchmark_run_archives a SET cleanup_checked_at = NOW()
                FROM candidate c WHERE a.benchmark_run_id = c.benchmark_run_id
                RETURNING a.benchmark_run_id
                """,
                (interval_seconds,),
            )
            row = cur.fetchone()
            return None if row is None else row["benchmark_run_id"]

    def lock_for_cleanup(self, run_id: str) -> dict | None:
        """Read authoritative references; caller holds the transaction through deletion."""
        with self.conn.cursor() as cur:
            cur.execute("SELECT * FROM benchmark_run_archives WHERE benchmark_run_id = %s FOR UPDATE", (run_id,))
            return cur.fetchone()

    def heartbeat(self, run_id: str, token: str) -> bool:
        with self.conn.cursor() as cur:
            cur.execute(
                "UPDATE benchmark_run_archives SET claimed_at = NOW() "
                "WHERE benchmark_run_id = %s AND claim_token = %s AND status = 'building'",
                (run_id, token),
            )
            return cur.rowcount == 1

    def validate_members(self, run_id: str, members: list[dict]) -> None:
        """Called before reading sources and again under locks before publication."""
        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT {_MEMBER_COLUMNS}, archive_status FROM evaluations "
                "WHERE benchmark_run_id = %s AND deleted_at IS NULL ORDER BY id FOR SHARE",
                (run_id,),
            )
            rows = cur.fetchall()
            actual = [
                BenchmarkArchiveMember.model_validate(row).model_dump(mode="json", exclude={"task_slug", "task_name"})
                for row in rows
                if row["archive_status"] == "ready"
            ]
            expected = [{k: v for k, v in member.items() if k not in {"task_slug", "task_name"}} for member in members]
            if actual != expected:
                raise Conflict("members_changed", "member executions or archives changed; rebuild export")

    def finish(
        self,
        job: dict,
        *,
        object_key: str,
        size_bytes: int,
        sha256: str | None = None,
        partial: bool = False,
    ) -> bool:
        """Return whether this claim published; a commit error remains uncertain."""
        with self.conn.transaction(), self.conn.cursor() as cur:
            self.validate_members(job["benchmark_run_id"], job["members"])
            cur.execute(
                """
                UPDATE benchmark_run_archives SET status = 'ready', object_key = %s,
                    size_bytes = %s, sha256 = %s, partial = %s, built_at = NOW(), error = NULL, claimed_at = NULL
                WHERE benchmark_run_id = %s AND claim_token = %s AND status = 'building'
                """,
                (object_key, size_bytes, sha256, partial, job["benchmark_run_id"], job["claim_token"]),
            )
            return cur.rowcount == 1

    def fail(self, job: dict, error: str) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                "UPDATE benchmark_run_archives SET status = 'failed', error = %s, "
                "claimed_at = NULL WHERE benchmark_run_id = %s AND claim_token = %s "
                "AND status = 'building'",
                (error, job["benchmark_run_id"], job["claim_token"]),
            )
