# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from datetime import UTC, datetime

import pytest

pytest.importorskip("scaled_evals")

from scaled_evals.api.repositories.base_repository import Conflict
from scaled_evals.api.repositories.benchmark_archive_repository import BenchmarkArchiveRepository
from task_fixtures import seed_benchmark_run, seed_evaluation, seed_task_revision
from test_repositories import MALICIOUS, _conn, _executed_sql_and_params

_RUN = {"framework": "harbor", "runtime": "sandbox_k8s", "parallelism": 1, "benchmark_id": "bm_1"}
_READY_MEMBER = {
    "task_id": "task_1",
    "task_revision": 1,
    "benchmark_run_id": "bmr_1",
    "status": "succeeded",
    "archive_status": "ready",
    "archive_object_key": "key",
    "evidence_status": "ready",
}


def _seed_run(**member_fields) -> None:  # noqa: ANN003
    """Store run `bmr_1`, with one member evaluation when fields are given."""
    seed_benchmark_run({**_RUN, "id": "bmr_1", "name": "run", "benchmark_revision": 1})
    if member_fields:
        seed_task_revision("task_1")
        seed_evaluation({**_READY_MEMBER, "id": "ev_1", **member_fields})


def test_benchmark_archive_request_snapshots_every_terminal_member() -> None:
    _seed_run(
        status="failed",
        current_execution=2,
        archive_built_at=datetime.now(UTC),
        archive_size_bytes=42,
        execution_snapshot={"task": {"slug": "smoke-one", "name": "Smoke One"}},
    )
    conn, cur = _conn()
    cur.fetchone.side_effect = [None, {"status": "queued"}]

    assert BenchmarkArchiveRepository(conn).request("bmr_1")["status"] == "queued"

    _sql, params = _executed_sql_and_params(cur, 1)
    [member] = params[2].obj
    assert (member["id"], member["current_execution"]) == ("ev_1", 2)
    assert (member["task_slug"], member["task_name"]) == ("smoke-one", "Smoke One")


@pytest.mark.parametrize("status, force", [("queued", True), ("building", True), ("ready", False)])
def test_benchmark_archive_request_is_idempotent(status, force) -> None:  # noqa: ANN001
    _seed_run()
    conn, cur = _conn()
    cur.fetchone.side_effect = [{"status": status}]
    assert BenchmarkArchiveRepository(conn).request("bmr_1", force=force) == {"status": status}
    assert cur.execute.call_count == 1


@pytest.mark.parametrize(
    "changes, code",
    [
        ({"status": "running"}, "run_not_terminal"),
        ({"archive_status": "building"}, "member_artifacts_not_ready"),
        ({"evidence_status": "building"}, "member_artifacts_not_ready"),
        ({"cancel_teardown_status": "pending"}, "member_artifacts_not_ready"),
    ],
)
def test_benchmark_archive_request_rejects_unstable_members(changes, code) -> None:  # noqa: ANN001
    _seed_run(**changes)
    conn, cur = _conn()
    cur.fetchone.side_effect = [None]
    with pytest.raises(Conflict) as exc:
        BenchmarkArchiveRepository(conn).request("bmr_1")
    assert exc.value.code == code
    assert cur.execute.call_count == 1


def test_benchmark_archive_lease_is_bounded_and_fences_stale_workers() -> None:
    _seed_run()
    conn, cur = _conn()
    repo = BenchmarkArchiveRepository(conn)
    assert repo.claim(claim_timeout=30, run_id="bmr_missing") is None
    assert cur.execute.call_count == 0
    repo.claim(claim_timeout=30, run_id="bmr_1")
    sql, params = _executed_sql_and_params(cur)
    assert "a.attempts >= 3" in sql and "a.benchmark_run_id = %s" in sql
    assert params[:2] == (30, "bmr_1")
    repo.fail({"benchmark_run_id": "bmr_1", "claim_token": MALICIOUS}, "failure")
    sql, params = _executed_sql_and_params(cur, 1)
    assert "claim_token = %s" in sql and "status = 'building'" in sql
    assert params[-1] == MALICIOUS and MALICIOUS not in sql


def test_benchmark_archive_finish_refuses_changed_execution() -> None:
    _seed_run()
    conn, cur = _conn()
    with pytest.raises(Conflict, match="members_changed"):
        BenchmarkArchiveRepository(conn).finish(
            {"benchmark_run_id": "bmr_1", "claim_token": "token", "members": [{"id": "ev_old"}]},
            object_key="key",
            size_bytes=123,
        )
    assert cur.execute.call_count == 0
