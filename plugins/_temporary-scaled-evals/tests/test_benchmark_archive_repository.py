# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from datetime import UTC, datetime

import pytest

pytest.importorskip("scaled_evals")

from test_repositories import MALICIOUS, _conn, _executed_sql_and_params


def test_benchmark_archive_request_snapshots_every_terminal_member() -> None:
    from scaled_evals.api.repositories.benchmark_archive_repository import (
        BenchmarkArchiveRepository,
    )

    conn, cur = _conn()
    cur.fetchone.side_effect = [{"framework": "harbor"}, None, {"status": "queued"}]
    cur.fetchall.return_value = [
        {
            "id": MALICIOUS,
            "task_id": "task_1",
            "task_slug": "smoke-one",
            "task_name": "Smoke One",
            "task_revision": 1,
            "status": "failed",
            "current_execution": 2,
            "archive_object_key": "key",
            "archive_built_at": datetime.now(UTC),
            "archive_size_bytes": 42,
            "archive_status": "ready",
            "evidence_status": "ready",
            "cancel_teardown_status": "not_requested",
        }
    ]
    assert BenchmarkArchiveRepository(conn).request(MALICIOUS)["status"] == "queued"
    sql, params = _executed_sql_and_params(cur, 0)
    assert MALICIOUS not in sql and params == (MALICIOUS,)
    assert "FOR UPDATE" in sql
    member_sql, member_params = _executed_sql_and_params(cur, 2)
    assert "FOR SHARE" in member_sql
    assert "LIMIT" not in member_sql
    assert member_params == (MALICIOUS,)
    _, params = _executed_sql_and_params(cur, 3)
    assert params[2].obj[0]["current_execution"] == 2
    assert params[2].obj[0]["task_slug"] == "smoke-one"
    assert params[2].obj[0]["task_name"] == "Smoke One"
    assert "execution_snapshot #>> '{task,slug}'" in member_sql


@pytest.mark.parametrize("status, force", [("queued", True), ("building", True), ("ready", False)])
def test_benchmark_archive_request_is_idempotent(status, force) -> None:
    from scaled_evals.api.repositories.benchmark_archive_repository import (
        BenchmarkArchiveRepository,
    )

    conn, cur = _conn()
    cur.fetchone.side_effect = [{"framework": "harbor"}, {"status": status}]
    assert BenchmarkArchiveRepository(conn).request("bmr_1", force=force) == {"status": status}
    assert cur.execute.call_count == 2


@pytest.mark.parametrize(
    "changes, code",
    [
        ({"status": "running"}, "run_not_terminal"),
        ({"archive_status": "building"}, "member_artifacts_not_ready"),
        ({"evidence_status": "building"}, "member_artifacts_not_ready"),
        ({"cancel_teardown_status": "pending"}, "member_artifacts_not_ready"),
    ],
)
def test_benchmark_archive_request_rejects_unstable_members(changes, code) -> None:
    from scaled_evals.api.repositories.base_repository import Conflict
    from scaled_evals.api.repositories.benchmark_archive_repository import (
        BenchmarkArchiveRepository,
    )

    conn, cur = _conn()
    cur.fetchone.side_effect = [{"framework": "harbor"}, None]
    cur.fetchall.return_value = [
        {
            "id": "ev_1",
            "status": "succeeded",
            "archive_status": "ready",
            "archive_object_key": "key",
            "evidence_status": "ready",
            "cancel_teardown_status": "not_requested",
            **changes,
        }
    ]
    with pytest.raises(Conflict) as exc:
        BenchmarkArchiveRepository(conn).request("bmr_1")
    assert exc.value.code == code
    assert cur.execute.call_count == 3


def test_benchmark_archive_lease_is_bounded_and_fences_stale_workers() -> None:
    from scaled_evals.api.repositories.benchmark_archive_repository import (
        BenchmarkArchiveRepository,
    )

    conn, cur = _conn()
    repo = BenchmarkArchiveRepository(conn)
    repo.claim(claim_timeout=30)
    sql, params = _executed_sql_and_params(cur)
    assert "SKIP LOCKED" in sql and "a.attempts >= 3" in sql
    assert "b.deleted_at IS NULL" in sql
    assert params[0] == 30
    repo.fail({"benchmark_run_id": "bmr_1", "claim_token": MALICIOUS}, "failure")
    sql, params = _executed_sql_and_params(cur, 1)
    assert "claim_token = %s" in sql and "status = 'building'" in sql
    assert params[-1] == MALICIOUS and MALICIOUS not in sql


def test_benchmark_archive_finish_refuses_changed_execution() -> None:
    from scaled_evals.api.repositories.base_repository import Conflict
    from scaled_evals.api.repositories.benchmark_archive_repository import (
        BenchmarkArchiveRepository,
    )

    conn, cur = _conn()
    cur.fetchall.return_value = []
    with pytest.raises(Conflict, match="members_changed"):
        BenchmarkArchiveRepository(conn).finish(
            {"benchmark_run_id": "bmr_1", "claim_token": "token", "members": [{"id": "ev_old"}]},
            object_key="key",
            size_bytes=123,
        )
    assert cur.execute.call_count == 1
