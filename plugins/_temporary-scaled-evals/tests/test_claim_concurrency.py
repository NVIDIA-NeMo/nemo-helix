# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit

import pytest

try:
    import psycopg
    from nemo_scaled_evals_plugin.migrations import apply_sql
    from psycopg.rows import dict_row
    from psycopg.sql import SQL, Identifier
    from scaled_evals.api.repositories.build_repository import TaskBuildRepository
    from scaled_evals.api.repositories.evaluation_repository import EvaluationRepository
except ImportError as exc:
    pytest.skip(f"scaled-evals plugin not installed: {exc}", allow_module_level=True)

# Same opt-in as test_migrations: point at a throwaway Postgres to run.
TEST_DSN_ENV = "SCALED_EVALS_TEST_DATABASE_URL"


def _loses_to_an_uncommitted_claim(
    dsn: str, claim: Callable[[Any], Any], rival: Callable[[Any], Any] | None = None
) -> Any:
    """Claim on one connection while a second claimant waits on the same row.

    The second claimant selects the candidate before the first commits, blocks on
    the row lock, then must re-check and come back empty once the first commits.
    ``rival`` replaces the first claim with another write to the same row.
    """
    with (
        psycopg.connect(dsn, row_factory=dict_row) as first,
        psycopg.connect(dsn, row_factory=dict_row) as second,
        ThreadPoolExecutor(1) as pool,
    ):
        with first.transaction():
            won = (rival or claim)(first)
            assert won is not None
            losing = pool.submit(claim, second)
            time.sleep(0.5)
            assert not losing.done(), "second claimant never reached the locked row"
        assert losing.result(timeout=10) is None
        assert claim(second) is None
        return won


@pytest.mark.skipif(not os.environ.get(TEST_DSN_ENV), reason=f"{TEST_DSN_ENV} not set")
def test_claims_have_exactly_one_winner_without_row_locks() -> None:
    admin_dsn = os.environ[TEST_DSN_ENV]
    scratch = f"se_claim_test_{uuid.uuid4().hex[:12]}"
    parsed = urlsplit(admin_dsn)
    dsn = urlunsplit(
        parsed._replace(
            path=f"/{scratch}",
            query=f"options={quote('-c search_path=scaled_evals,public', safe='')}",
        )
    )
    with psycopg.connect(admin_dsn, autocommit=True) as conn:
        conn.execute(SQL("CREATE DATABASE {}").format(Identifier(scratch)))
    try:
        apply_sql(dsn, schema="scaled_evals")
        with psycopg.connect(dsn) as conn:
            conn.execute("INSERT INTO tasks (id, name, slug) VALUES ('t1', 't1', 't1')")
            conn.execute(
                "INSERT INTO task_revisions (task_id, revision, tarball_object_key, status, build_backend)"
                " VALUES ('t1', 1, 'k', 'building', 'prebuilt')"
            )
            conn.execute(
                "INSERT INTO evaluations (id, name, task_id, task_revision, status, parallelism)"
                " VALUES ('e1', 'e1', 't1', 1, 'queued', 1)"
            )

        won = _loses_to_an_uncommitted_claim(
            dsn,
            lambda conn: EvaluationRepository(conn).claim_next(claim_timeout=60, worker_id="w"),
        )
        assert (won["id"], won["previous_status"], won["status"]) == ("e1", "queued", "provisioning")

        with psycopg.connect(dsn) as conn:
            conn.execute(
                "UPDATE evaluations SET status = 'running', dispatch_job_name = 'job-1',"
                " dispatch_claimed_at = NULL WHERE id = 'e1'"
            )
        leased = _loses_to_an_uncommitted_claim(
            dsn,
            lambda conn: EvaluationRepository(conn).claim_stale_dispatch_job(
                stale_seconds=0, claim_timeout=60, worker_id="w"
            ),
        )
        assert (leased["id"], leased["dispatch_job_name"]) == ("e1", "job-1")

        # A Job resubmitted after the reconciler picked its row is no longer stale.
        with psycopg.connect(dsn) as conn:
            conn.execute(
                "UPDATE evaluations SET dispatch_claimed_at = NULL, dispatch_reconcile_claimed_at = NULL"
                " WHERE id = 'e1'"
            )
        _loses_to_an_uncommitted_claim(
            dsn,
            lambda conn: EvaluationRepository(conn).claim_stale_dispatch_job(
                stale_seconds=60, claim_timeout=60, worker_id="w"
            ),
            rival=lambda conn: conn.execute(
                "UPDATE evaluations SET dispatch_job_name = 'job-2', dispatch_claimed_at = NOW()"
                " WHERE id = 'e1' RETURNING id"
            ).fetchone(),
        )

        build = _loses_to_an_uncommitted_claim(
            dsn,
            lambda conn: TaskBuildRepository(conn).claim_next(worker_id="w", claim_timeout=60, max_attempts=3),
        )
        assert (build.task_id, build.revision) == ("t1", 1)

        # e1 holds one of the owner's slots, so a 50-slot evaluation waits for a 51-slot cap.
        with psycopg.connect(dsn, row_factory=dict_row) as conn:
            conn.execute(
                "INSERT INTO evaluations (id, name, task_id, task_revision, status, parallelism)"
                " VALUES ('e2', 'e2', 't1', 1, 'queued', 50)"
            )
            repo = EvaluationRepository(conn)
            assert repo.claim_next(claim_timeout=60, worker_id="w", per_user_slot_limit=50) is None
            assert (
                repo.claim_next(claim_timeout=60, worker_id="w", cluster_slot_limit=50, per_user_slot_limit=51) is None
            )
            admitted = repo.claim_next(claim_timeout=60, worker_id="w", cluster_slot_limit=51, per_user_slot_limit=51)
            assert admitted is not None and admitted["id"] == "e2"
    finally:
        with psycopg.connect(admin_dsn, autocommit=True) as conn:
            conn.execute(SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(Identifier(scratch)))
