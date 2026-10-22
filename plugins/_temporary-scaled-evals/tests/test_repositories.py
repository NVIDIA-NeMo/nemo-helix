# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

try:
    from nemo_scaled_evals_plugin import migrations
    from scaled_evals.api.benchmark_store import BenchmarkStore
    from scaled_evals.api.repositories.base_repository import (
        order_by_clause,
        patch_set_clause,
        substring_search_pattern,
    )
    from scaled_evals.api.repositories.ops_repository import OperationsRepository
    from scaled_evals.api.task_store import TaskStore
    from task_fixtures import seed_benchmark, seed_evaluation, seed_task_revision, stored_revision
except ImportError as exc:
    pytest.skip(f"scaled-evals plugin not installed: {exc}", allow_module_level=True)

MALICIOUS = "x'); DROP TABLE evaluations; --"


def _conn() -> tuple[MagicMock, MagicMock]:
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchone.return_value = {"id": "ok"}
    cur.fetchall.return_value = []
    return conn, cur


def _executed_sql_and_params(cur: MagicMock, index: int = 0) -> tuple[str, tuple | list]:
    call = cur.execute.call_args_list[index]
    return call.args[0], call.args[1]


def test_benchmark_run_revision_returns_qualification_metadata() -> None:
    qualified_at = datetime(2026, 8, 6, tzinfo=UTC)
    seed_benchmark(
        "bm_safe",
        qualification_status="qualified",
        qualification_evidence={"receipt": "qe_1"},
        qualified_at=qualified_at,
    )
    store = BenchmarkStore()

    assert store.revision_for_run("bm_safe") == {
        "revision": 1,
        "qualification_status": "qualified",
        "qualification_evidence": {"receipt": "qe_1"},
        "qualified_at": qualified_at,
    }
    assert store.revision_for_run("bm_safe", 1) == store.revision_for_run("bm_safe")
    assert store.revision_for_run("bm_safe", 2) is None
    assert store.revision_for_run("bm_missing") is None
    store.soft_delete("bm_safe")
    assert store.revision_for_run("bm_safe") is None


def test_task_prebuilt_finalize_marks_ready_with_verifier_image() -> None:
    store = TaskStore()
    store.create_with_initial_revision(
        "task_safe", name="task", slug="task", description=None, visibility="private", object_key="k"
    )

    result = store.finalize_latest_revision_prebuilt(
        "task_safe",
        image_ref="registry/task:signed",
        image_digest="sha256:" + "a" * 64,
        tarball_sha256="package-hash",
        verifier_image_ref="registry/verifier:signed",
        verifier_image_digest="sha256:" + "b" * 64,
    )

    assert result is not None and result.finalized and result.status == "ready"
    revision = stored_revision("task_safe", 1)
    assert (revision["status"], revision["tarball_sha256"], revision["verifier_image_digest"]) == (
        "ready",
        "package-hash",
        "sha256:" + "b" * 64,
    )
    uploading = store.revision_for_finalize("task_safe", expected_revision=1)
    assert uploading is not None and uploading.status == "ready"
    again = store.finalize_latest_revision_prebuilt("task_safe", image_ref="x", image_digest="y", tarball_sha256=None)
    assert again is not None and again.finalized is False


def test_verifier_image_columns_exist_in_baseline_and_replayable_migration() -> None:
    root = migrations.sql_root()
    migration = (root / "migrations" / "043_task_verifier_image.sql").read_text()
    baseline = (root / "schema" / "01_tasks.sql").read_text()

    for column in ("verifier_image_ref", "verifier_image_digest"):
        assert f"ADD COLUMN IF NOT EXISTS {column} TEXT" in migration
        assert re.search(rf"^\s+{column}\s+TEXT,$", baseline, re.MULTILINE)


def test_substring_search_pattern_treats_like_metacharacters_as_literals() -> None:
    assert substring_search_pattern(r"100%_done\now") == r"%100\%\_done\\now%"
    assert substring_search_pattern("   ") is None


def test_operations_heartbeat_uses_dict_row_factory() -> None:
    conn, cur = _conn()
    cur.fetchone.return_value = {"is_fresh": True}

    assert OperationsRepository(conn).has_fresh_service_heartbeat("build_worker", stale_seconds=60)

    sql, params = _executed_sql_and_params(cur)
    assert "AS is_fresh" in sql
    assert params == ("build_worker", 60)


def test_operations_counts_and_snapshot_derive_from_stored_evaluations() -> None:
    seed_task_revision("task_x")
    now = datetime.now(UTC)
    old = now - timedelta(hours=2)
    seed_evaluation({"id": "ev_q", "status": "queued", "created_at": old})
    seed_evaluation(
        {"id": "ev_r", "status": "running", "updated_at": now, "dispatch_claimed_by": "w1", "dispatch_claimed_at": now}
    )
    seed_evaluation(
        {
            "id": "ev_f",
            "status": "failed",
            "updated_at": now,
            "status_detail": "backend launch failed",
            "current_execution": 2,
            "n_trials": 3,
        }
    )
    repository = OperationsRepository(MagicMock())

    assert repository.evaluation_status_counts() == {"queued": 1, "running": 1, "failed": 1}
    assert repository.fleet_totals() == {
        "task_definitions": 1,
        "tasks_run": 1,
        "evaluation_jobs": 3,
        "evaluation_executions": 4,
        "completed_trials": 3,
        "benchmark_runs": 0,
    }
    snapshot = repository.dispatch_observability_snapshot(
        stuck_queued_seconds=111,
        stuck_provisioning_seconds=222,
        stuck_running_seconds=333,
        stale_worker_seconds=444,
    )
    assert (snapshot["unclaimed_queued"], snapshot["live_workers"], snapshot["stale_workers"]) == (1, 1, 0)
    assert snapshot["oldest_queued_seconds"] >= 7200
    assert snapshot["stuck_jobs"] == [{"status": "queued", "runtime": "sandbox_k8s", "count": 1}]
    assert snapshot["backend_failures"] == [{"runtime": "sandbox_k8s", "count": 1}]


def test_order_by_clause_rejects_unallowed_columns() -> None:
    with pytest.raises(ValueError, match="not orderable"):
        order_by_clause(("created_at", "id; DROP TABLE evaluations"), "desc")


def test_order_by_clause_formats_allowed_columns() -> None:
    assert order_by_clause(("created_at", "id"), "asc") == "created_at ASC, id ASC"


def test_patch_set_clause_rejects_unlisted_patch_field() -> None:
    with pytest.raises(ValueError, match="not patchable"):
        patch_set_clause([("deleted_at", MALICIOUS)], frozenset({"name"}))
