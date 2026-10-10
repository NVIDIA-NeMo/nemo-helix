# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from unittest.mock import MagicMock
from urllib.parse import quote, urlsplit, urlunsplit

import pytest

try:
    import psycopg
    from nemo_scaled_evals_plugin.migrations import apply_sql
    from psycopg.rows import dict_row
    from psycopg.sql import SQL, Identifier
    from scaled_evals.api.repositories.build_repository import TaskBuildRepository
    from scaled_evals.api.repositories.evaluation_repository import DISPATCH_STATE_COLUMNS, EvaluationRepository
    from scaled_evals.dispatch.worker import Dispatcher
except ImportError as exc:
    pytest.skip(f"scaled-evals plugin not installed: {exc}", allow_module_level=True)

# Same opt-in as test_migrations: point at a throwaway Postgres to run.
TEST_DSN_ENV = "SCALED_EVALS_TEST_DATABASE_URL"


def _loses_to_an_uncommitted_claim(dsn: str, claim: Callable[[Any], Any]) -> Any:
    """Claim on one connection while a second claimant waits on the same row.

    The second claimant selects the candidate before the first commits, blocks on
    the row lock, then must re-check and come back empty once the first commits.
    """
    with (
        psycopg.connect(dsn, row_factory=dict_row) as first,
        psycopg.connect(dsn, row_factory=dict_row) as second,
        ThreadPoolExecutor(1) as pool,
    ):
        with first.transaction():
            won = claim(first)
            assert won is not None
            losing = pool.submit(claim, second)
            time.sleep(0.5)
            assert not losing.done(), "second claimant never reached the locked row"
        assert losing.result(timeout=10) is None
        assert claim(second) is None
        return won


@pytest.fixture
def dsn() -> Iterator[str]:
    """Yield a migrated scratch database seeded with one queued evaluation, ``e1``."""
    if not os.environ.get(TEST_DSN_ENV):
        pytest.skip(f"{TEST_DSN_ENV} not set")
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
            # Far above the old 50-slot per-user and 500-slot cluster caps.
            conn.execute(
                "INSERT INTO evaluations (id, name, task_id, task_revision, status, parallelism)"
                " VALUES ('e1', 'e1', 't1', 1, 'queued', 1000)"
            )
        yield dsn
    finally:
        with psycopg.connect(admin_dsn, autocommit=True) as conn:
            conn.execute(SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(Identifier(scratch)))


def test_claims_have_exactly_one_winner_without_row_locks_or_admission(dsn: str) -> None:
    with psycopg.connect(dsn, row_factory=dict_row) as conn:
        assert EvaluationRepository(conn).claim_next(claim_timeout=60, worker_id="w", evaluation_id="other") is None

    won = _loses_to_an_uncommitted_claim(
        dsn,
        lambda conn: EvaluationRepository(conn).claim_next(claim_timeout=60, worker_id="w", evaluation_id="e1"),
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

    build = _loses_to_an_uncommitted_claim(
        dsn,
        lambda conn: TaskBuildRepository(conn).claim_next(worker_id="w", claim_timeout=60, max_attempts=3),
    )
    assert (build.task_id, build.revision) == ("t1", 1)


def test_launch_inputs_and_live_state_rebuild_the_dispatch_row(dsn: str) -> None:
    """The Job's entity inputs plus the narrow state read equal the old full load."""
    with psycopg.connect(dsn, row_factory=dict_row, autocommit=True) as conn:
        full = EvaluationRepository(conn).load_for_dispatch("e1")
        assert full is not None
        inputs = {key: value for key, value in full.items() if key not in DISPATCH_STATE_COLUMNS}
        conn.execute("UPDATE evaluations SET status = 'running', current_execution = 2 WHERE id = 'e1'")
        merged = Dispatcher._load_state(conn, "e1", inputs)
        assert merged == EvaluationRepository(conn).load_for_dispatch("e1")
        assert (merged["status"], merged["current_execution"]) == ("running", 2)

    projected: list[dict[str, Any]] = []
    dispatcher = Dispatcher(connect=lambda: psycopg.connect(dsn, row_factory=dict_row), on_change=projected.append)
    dispatcher._changed("e1")
    assert [row["id"] for row in projected] == ["e1"]
    assert projected[0]["status"] == "running"
    # A failing projection never fails the run.
    dispatcher.on_change = MagicMock(side_effect=RuntimeError("entity store down"))
    dispatcher._changed("e1")
