# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("scaled_evals")

from scaled_evals.api.build.queue_worker import TaskBuildWorker
from scaled_evals.api.repositories.build_repository import TaskBuildRepository


def test_claim_expires_stale_final_attempt_before_selecting_work() -> None:
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchone.return_value = None

    assert (
        TaskBuildRepository(conn).claim_next(
            worker_id="worker-1",
            claim_timeout=90,
            max_attempts=3,
        )
        is None
    )

    expire_sql, expire_params = cur.execute.call_args_list[0].args
    claim_sql, claim_params = cur.execute.call_args_list[1].args
    assert "build worker lease expired after final attempt" in expire_sql
    assert "build_attempts >= %s" in expire_sql
    assert expire_params == (3, 90)
    assert "FOR UPDATE SKIP LOCKED" in claim_sql
    assert claim_params == (3, 90, "worker-1")


@pytest.mark.parametrize(
    ("attempt", "terminal"),
    [
        (1, False),
        (3, True),
    ],
)
def test_retry_or_fail_persists_retry_or_terminal_state(attempt: int, terminal: bool) -> None:
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.rowcount = 1

    updated = TaskBuildRepository(conn).retry_or_fail(
        "task_1",
        2,
        worker_id="worker-1",
        build_error="builder unavailable",
        attempt=attempt,
        max_attempts=3,
        retry_delay=30,
    )

    assert updated is True
    sql, params = cur.execute.call_args.args
    if terminal:
        assert "status = 'failed'" in sql
        assert "build_completed_at = NOW()" in sql
        assert params == ("builder unavailable", "task_1", 2, "worker-1")
    else:
        assert "build_next_attempt_at = NOW()" in sql
        assert "status = 'failed'" not in sql
        assert params == ("builder unavailable", 30, "task_1", 2, "worker-1")


def test_platform_job_claim_binding_and_listing() -> None:
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.rowcount = 1
    cur.fetchall.return_value = [
        {
            "task_id": "task_1",
            "revision": 2,
            "build_attempts": 1,
            "job_name": "scaled-evals-build-abc",
        }
    ]
    repository = TaskBuildRepository(conn)

    assert repository.bind_platform_job(
        "task_1",
        2,
        worker_id="controller-1",
        job_name="scaled-evals-build-abc",
    )
    bind_sql, bind_params = cur.execute.call_args.args
    assert "SET build_claimed_by = %s" in bind_sql
    assert bind_params == ("scaled-evals-build-abc", "task_1", 2, "controller-1")

    assert repository.list_platform_jobs(limit=10) == cur.fetchall.return_value
    list_sql, list_params = cur.execute.call_args.args
    assert "build_claimed_by LIKE 'scaled-evals-build-%%'" in list_sql
    assert list_params == (10,)


def test_legacy_worker_does_not_claim_platform_owned_queue(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = TaskBuildWorker()
    claim_next = MagicMock()
    monkeypatch.setattr(worker, "claim_next", claim_next)
    monkeypatch.setitem(
        TaskBuildWorker.work_once.__globals__,
        "settings",
        SimpleNamespace(platform_build_jobs_enabled=True),
    )

    assert worker.work_once() is False
    claim_next.assert_not_called()
