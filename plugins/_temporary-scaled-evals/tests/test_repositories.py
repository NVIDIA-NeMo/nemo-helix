# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

try:
    from nemo_scaled_evals_plugin import migrations
    from scaled_evals.api.benchmark_store import BenchmarkStore
    from scaled_evals.api.config_profile_store import ConfigProfileStore
    from scaled_evals.api.repositories.base_repository import (
        InvalidReference,
        order_by_clause,
        patch_set_clause,
        substring_search_pattern,
    )
    from scaled_evals.api.repositories.benchmark_run_repository import BenchmarkRunRepository
    from scaled_evals.api.repositories.evaluation_repository import EvaluationRepository
    from scaled_evals.api.repositories.execution_cleanup_repository import (
        ExecutionCleanupRepository,
    )
    from scaled_evals.api.repositories.execution_telemetry_repository import (
        ExecutionTelemetryRepository,
    )
    from scaled_evals.api.repositories.ops_repository import OperationsRepository
    from scaled_evals.api.repositories.user_repository import UserRepository
    from scaled_evals.api.schemas.common import encode_cursor
    from scaled_evals.api.task_store import TaskStore
    from scaled_evals.models.evaluations import EvaluationResultSummary, EvaluationResultWrite
    from task_fixtures import seed_benchmark, seed_task_revision, stored_revision
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


def test_evaluation_list_parameterizes_filters_and_cursor() -> None:
    conn, cur = _conn()
    created_at = datetime(2026, 6, 23, tzinfo=UTC)
    cursor = encode_cursor(created_at, "ev_safe")

    EvaluationRepository(conn).list(
        limit=20,
        cursor=cursor,
        order="desc",
        status="queued",
        task_id=MALICIOUS,
        shared=True,
        q=MALICIOUS,
    )

    sql, params = _executed_sql_and_params(cur)
    assert MALICIOUS not in sql
    assert "queued" not in sql
    assert "status::text ILIKE %s" in sql
    assert MALICIOUS in params
    assert substring_search_pattern(MALICIOUS) in params
    assert created_at in params
    assert "ev_safe" in params


@pytest.mark.parametrize(
    ("repository", "kwargs", "expected_sql"),
    [
        (
            BenchmarkRunRepository,
            {"benchmark_id": None, "shared": False},
            "benchmark_id ILIKE %s",
        ),
    ],
)
def test_resource_list_search_is_parameterized(  # noqa: ANN001
    repository, kwargs, expected_sql
) -> None:
    conn, cur = _conn()

    repository(conn).list(
        limit=20,
        cursor=None,
        order="desc",
        q=MALICIOUS,
        **kwargs,
    )

    sql, params = _executed_sql_and_params(cur)
    assert MALICIOUS not in sql
    assert expected_sql in sql
    assert substring_search_pattern(MALICIOUS) in params


def test_benchmark_run_search_includes_case_insensitive_derived_status() -> None:
    conn, cur = _conn()

    BenchmarkRunRepository(conn).list(
        benchmark_id=None,
        shared=False,
        limit=20,
        cursor=None,
        order="desc",
        q="SUCCEEDED",
    )

    sql, params = _executed_sql_and_params(cur)
    assert "CASE" in sql
    assert "member.status NOT IN ('succeeded', 'failed', 'cancelled')" in sql
    assert ") ILIKE %s" in sql
    assert params.count("%SUCCEEDED%") == 6


def test_substring_search_pattern_treats_like_metacharacters_as_literals() -> None:
    assert substring_search_pattern(r"100%_done\now") == r"%100\%\_done\\now%"
    assert substring_search_pattern("   ") is None


def test_evaluation_load_for_dispatch_selects_provenance_inputs() -> None:
    seed_task_revision("task_safe", slug="task-safe", image_digest="sha256:task", verifier_image_ref="verifier:1")
    conn, cur = _conn()
    cur.fetchone.return_value = {"id": "ev_safe", "task_id": "task_safe", "task_revision": 1}

    row = EvaluationRepository(conn).load_for_dispatch("ev_safe")

    sql, params = _executed_sql_and_params(cur)
    assert params == ("ev_safe",)
    assert "e.task_id" in sql
    assert "e.task_revision" in sql
    assert "e.framework_profile_id" in sql
    assert "e.framework_version" in sql
    assert "e.runner_image_ref" in sql
    assert "e.runner_image_digest" in sql
    assert "e.framework_adapter_version" in sql
    assert "e.sandbox_k8s_version" in sql
    assert "e.runner_metadata" in sql
    assert "e.dispatch_job_name" in sql
    assert "e.dispatch_job_uid" in sql
    assert row is not None
    assert (row["task_slug"], row["image_digest"], row["verifier_image_ref"]) == (
        "task-safe",
        "sha256:task",
        "verifier:1",
    )
    assert {"verifier_image_digest", "tarball_sha256", "tarball_object_key"} <= row.keys()
    cur.fetchone.return_value = {"id": "ev_safe", "task_id": "task_safe", "task_revision": 9}
    assert EvaluationRepository(conn).load_for_dispatch("ev_safe") is None


def test_evaluation_stale_dispatch_jobs_parameterize_reconcile_window() -> None:
    conn, cur = _conn()

    EvaluationRepository(conn).list_stale_dispatch_jobs(stale_seconds=45.5, limit=7)

    sql, params = _executed_sql_and_params(cur)
    assert "dispatch_job_name IS NOT NULL" in sql
    assert "status IN ('provisioning', 'running')" in sql
    assert params == (45.5, 7)


def test_evaluation_reconciler_claim_is_leased_without_row_locks() -> None:
    conn, cur = _conn()

    EvaluationRepository(conn).claim_stale_dispatch_job(
        stale_seconds=45.5,
        claim_timeout=90,
        worker_id="worker-1",
    )

    sql, params = _executed_sql_and_params(cur)
    assert "SKIP LOCKED" not in sql
    assert "dispatch_reconcile_claimed_at" in sql
    assert "evaluation_execution_cleanups" in sql
    assert params == (45.5, 90, "worker-1", 45.5, 90)


def test_execution_cleanup_claim_is_bounded_and_skip_locked() -> None:
    conn, cur = _conn()

    ExecutionCleanupRepository(conn).claim_one(worker_id="worker-1", claim_timeout=30)

    sql, params = _executed_sql_and_params(cur)
    assert "LIMIT 1" in sql
    assert "FOR UPDATE SKIP LOCKED" in sql
    assert "teardown_attempts = teardown_attempts + 1" in sql
    assert params == (30, "worker-1")


def test_execution_cleanup_list_is_attempt_aware_and_omits_backend_handle() -> None:
    conn, cur = _conn()

    ExecutionCleanupRepository(conn).list_for_evaluation("ev_1")

    sql, params = _executed_sql_and_params(cur)
    assert "ORDER BY execution_number ASC" in sql
    assert "backend_handle" not in sql
    assert params == ("ev_1",)


def test_execution_cleanup_failure_releases_lease_with_bounded_backoff() -> None:
    conn, cur = _conn()

    ExecutionCleanupRepository(conn).mark_failed(
        17,
        worker_id="worker-1",
        detail="backend unavailable",
    )

    sql, params = _executed_sql_and_params(cur)
    assert "status = 'delete_failed'" in sql
    assert "LEAST(300" in sql
    assert "teardown_claimed_by = NULL" in sql
    assert params == ("backend unavailable", 17, "worker-1")


def test_evaluation_stale_dispatch_job_failure_is_guarded() -> None:
    conn, cur = _conn()
    cur.rowcount = 1

    assert EvaluationRepository(conn).fail_stale_dispatch_job(
        "ev_safe",
        execution_number=2,
        dispatch_job_name="scaled-evals-eval-safe",
        detail=MALICIOUS,
    )

    sql, params = _executed_sql_and_params(cur)
    assert MALICIOUS not in sql
    assert params == (MALICIOUS, "ev_safe", 2, "scaled-evals-eval-safe")
    assert "deleted_at IS NULL" in sql
    assert "status IN ('provisioning', 'running')" in sql
    assert "current_execution = %s" in sql
    assert "dispatch_job_name = %s" in sql
    event_params = _executed_sql_and_params(cur, 1)[1]
    assert event_params == ("ev_safe", "status", "failed", MALICIOUS)
    assert len(cur.execute.call_args_list) == 2


def test_evaluation_stale_dispatch_job_failure_noops_when_guard_misses() -> None:
    conn, cur = _conn()
    cur.rowcount = 0

    assert not EvaluationRepository(conn).fail_stale_dispatch_job(
        "ev_safe",
        execution_number=2,
        dispatch_job_name="scaled-evals-eval-safe",
        detail=MALICIOUS,
    )

    assert cur.execute.call_count == 1


def test_evaluation_manual_retry_preserves_identity_and_resets_terminal_state() -> None:
    conn, cur = _conn()
    cur.fetchone.return_value = {
        "id": "ev_safe",
        "current_execution": 4,
        "max_executions": 4,
    }

    row = EvaluationRepository(conn).retry_failed("ev_safe")

    assert row is not None
    retry_sql, retry_params = _executed_sql_and_params(cur)
    assert "status = 'queued'" in retry_sql
    assert "status = 'failed'" in retry_sql
    assert "current_execution = current_execution + 1" in retry_sql
    assert "current_execution - infrastructure_retries + 1" in retry_sql
    assert "benchmark_run_id" in retry_sql
    assert "result = NULL" in retry_sql
    assert "evidence_status = 'missing'" in retry_sql
    assert "archive_status = 'missing'" in retry_sql
    assert retry_params == ("ev_safe",)
    event_sql, event_params = _executed_sql_and_params(cur, 1)
    assert "INSERT INTO evaluation_events" in event_sql
    assert event_params == (
        "ev_safe",
        "retry",
        "queued",
        "manual retry 4/4 scheduled",
    )


def test_evaluation_retry_block_reason_distinguishes_terminal_artifact_claims() -> None:
    conn, cur = _conn()
    cur.fetchone.return_value = {"reason": "terminal_artifacts_finalizing"}

    reason = EvaluationRepository(conn).retry_block_reason("ev_safe")

    assert reason == "terminal_artifacts_finalizing"
    sql, params = _executed_sql_and_params(cur)
    assert "evidence_claimed_by IS NOT NULL" in sql
    assert "archive_claimed_by IS NOT NULL" in sql
    assert "benchmark_runs.cancelled_at IS NULL" in sql
    assert params == ("ev_safe",)


def test_evaluation_persist_result_uses_pydantic_write_model() -> None:
    conn, cur = _conn()
    result = EvaluationResultWrite(
        result={"note": MALICIOUS},
        summary=EvaluationResultSummary(
            reward=0.5,
            n_trials=2,
            n_completed=1,
            n_errored=1,
            finished_at="2026-06-23T20:00:00Z",
        ),
        extra_detail=MALICIOUS,
    )

    EvaluationRepository(conn).persist_result("ev_safe", result)

    sql, params = _executed_sql_and_params(cur)
    assert MALICIOUS not in sql
    assert params[1] == f"1/2 trials completed; {MALICIOUS}"
    assert params[3].obj == 0.5
    assert params[4] == 0.5
    assert params[5] == 2
    assert params[6] == 1


def test_evaluation_execution_writes_are_fenced() -> None:
    conn, cur = _conn()
    cur.rowcount = 1

    assert EvaluationRepository(conn).set_status(
        "ev_safe",
        "running",
        expected_execution_number=2,
    )
    status_sql, status_params = _executed_sql_and_params(cur)
    assert "current_execution = %s::integer" in status_sql
    # The two trailing params are the monotonicity and cancel guards' status.
    assert status_params[-4:-2] == (2, 2)

    result = EvaluationResultWrite(
        result={"reward": 1},
        summary=EvaluationResultSummary(
            reward=1,
            n_trials=1,
            n_completed=1,
            n_errored=0,
        ),
    )
    assert EvaluationRepository(conn).persist_result(
        "ev_safe",
        result,
        expected_execution_number=2,
    )
    result_sql, result_params = _executed_sql_and_params(cur, 2)
    assert "current_execution = %s::integer" in result_sql
    assert result_params[-2:] == (2, 2)


def test_evaluation_status_cannot_move_backward_off_terminal() -> None:
    """A terminal row only accepts another terminal status.

    Two writers can observe the same execution: the Platform Job status
    reconcile and the run itself. Without this guard a stale `running`
    observation landing after the run finished would resurrect a succeeded
    evaluation, so the guard has to live in the WHERE clause where it applies
    to every caller rather than in one call site.
    """
    conn, cur = _conn()
    cur.rowcount = 1
    repo = EvaluationRepository(conn)

    repo.set_status("ev_safe", "running")
    sql, params = _executed_sql_and_params(cur)
    assert "status NOT IN ('succeeded', 'failed', 'cancelled')" in sql
    # The guard compares the incoming status, so it is the last bound param.
    assert params[-1] == "running"

    # Terminal-to-terminal still lands: a cancel racing a success is not a
    # backward move, and refusing it would strand the cancel.
    repo.set_status("ev_safe", "cancelled")
    assert _executed_sql_and_params(cur, 2)[1][-1] == "cancelled"


def test_evaluation_artifact_publication_locks_current_execution() -> None:
    conn, cur = _conn()
    cur.rowcount = 1

    assert EvaluationRepository(conn).lock_current_execution(
        "ev_safe",
        expected_execution_number=3,
    )

    sql, params = _executed_sql_and_params(cur)
    assert "current_execution = %s" in sql
    assert "FOR UPDATE" in sql
    assert params == ("ev_safe", 3)


def test_evaluation_persist_result_preserves_boolean_without_numeric_coercion() -> None:
    conn, cur = _conn()
    result = EvaluationResultWrite(
        result={"verifier_result": {"score": True}},
        summary=EvaluationResultSummary(
            reward=True,
            n_trials=1,
            n_completed=1,
            n_errored=0,
        ),
    )

    EvaluationRepository(conn).persist_result("ev_bool", result)

    _sql, params = _executed_sql_and_params(cur)
    assert params[3].obj is True
    assert params[4] is None


def test_evaluation_status_detail_is_redacted_before_persistence() -> None:
    conn, cur = _conn()

    EvaluationRepository(conn).set_status(
        "ev_safe",
        "failed",
        detail="runner failed with api_key=sk-secret-value",
    )

    _sql, params = _executed_sql_and_params(cur)
    assert params[1] == "runner failed with api_key=<redacted>"
    event_params = _executed_sql_and_params(cur, 1)[1]
    assert event_params[3] == "runner failed with api_key=<redacted>"


def test_operations_snapshot_parameterizes_thresholds() -> None:
    conn, cur = _conn()
    cur.fetchone.return_value = {
        "oldest_queued_seconds": 42,
        "unclaimed_queued": 1,
        "live_workers": 2,
        "stale_workers": 0,
        "oldest_worker_lease_seconds": 12,
    }
    cur.fetchall.side_effect = [
        [{"status": "queued", "runtime": "sandbox_k8s", "count": 3}],
        [{"runtime": "sandbox_k8s", "count": 1}],
    ]

    snapshot = OperationsRepository(conn).dispatch_observability_snapshot(
        stuck_queued_seconds=111,
        stuck_provisioning_seconds=222,
        stuck_running_seconds=333,
        stale_worker_seconds=444,
    )

    first_sql, first_params = _executed_sql_and_params(cur, 0)
    second_sql, second_params = _executed_sql_and_params(cur, 1)
    assert "111" not in first_sql
    assert "222" not in first_sql
    assert "333" not in first_sql
    assert "444" not in first_sql
    assert first_params == (111, 222, 333, 444, 444)
    assert second_params == (111, 222, 333)
    assert snapshot["stuck_jobs"] == [{"status": "queued", "runtime": "sandbox_k8s", "count": 3}]
    assert snapshot["backend_failures"] == [{"runtime": "sandbox_k8s", "count": 1}]


def test_operations_heartbeat_uses_dict_row_factory() -> None:
    conn, cur = _conn()
    cur.fetchone.return_value = {"is_fresh": True}

    assert OperationsRepository(conn).has_fresh_service_heartbeat("build_worker", stale_seconds=60)

    sql, params = _executed_sql_and_params(cur)
    assert "AS is_fresh" in sql
    assert params == ("build_worker", 60)


def test_operations_status_counts_use_dict_rows() -> None:
    conn, cur = _conn()
    category_rows = [
        {"status": "queued", "count": 2},
        {"status": "succeeded", "count": 5},
    ]
    timeline_rows = [
        {
            "day": datetime(2026, 8, 6, tzinfo=UTC).date(),
            "total": 2,
            "counts": {"evaluation_timeout": 2},
        },
        {
            "day": datetime(2026, 8, 7, tzinfo=UTC).date(),
            "total": 4,
            "counts": {"inference_http_504": 4},
        },
    ]
    cur.fetchall.side_effect = [category_rows, timeline_rows]

    assert OperationsRepository(conn).evaluation_status_counts() == {
        "queued": 2,
        "succeeded": 5,
    }


def test_operations_fleet_totals_include_jobs_executions_and_trials() -> None:
    conn, cur = _conn()
    TaskStore().create_with_initial_revision(
        "task_1", name="t", slug="t", description=None, visibility="private", object_key="k"
    )
    cur.fetchone.return_value = {
        "tasks_run": 9,
        "evaluation_jobs": 34,
        "evaluation_executions": 39,
        "completed_trials": 81,
        "benchmark_runs": 4,
    }

    totals = OperationsRepository(conn).fleet_totals()

    assert totals == {
        "task_definitions": 1,
        "tasks_run": 9,
        "evaluation_jobs": 34,
        "evaluation_executions": 39,
        "completed_trials": 81,
        "benchmark_runs": 4,
    }
    sql = cur.execute.call_args.args[0]
    assert "FROM tasks" not in sql
    assert "COUNT(DISTINCT task_id)" in sql
    assert "COUNT(*) FROM evaluations" in sql
    assert "SUM(current_execution)" in sql
    assert "SUM(n_trials)" in sql
    assert len(cur.execute.call_args.args) == 1


def test_repository_order_rejects_unallowed_sort_direction() -> None:
    conn, _cur = _conn()

    with pytest.raises(ValueError, match="order"):
        BenchmarkRunRepository(conn).list(
            limit=20, cursor=None, order="desc; DROP TABLE tasks", benchmark_id=None, shared=False
        )


def test_order_by_clause_rejects_unallowed_columns() -> None:
    with pytest.raises(ValueError, match="not orderable"):
        order_by_clause(("created_at", "id; DROP TABLE evaluations"), "desc")


def test_order_by_clause_formats_allowed_columns() -> None:
    assert order_by_clause(("created_at", "id"), "asc") == "created_at ASC, id ASC"


def test_execution_telemetry_repository_records_attempt_aware_phases_and_summary() -> None:
    conn, cur = _conn()
    repository = ExecutionTelemetryRepository(conn)

    repository.record_phase("ev_1", execution_number=2, phase="running")
    repository.record_phase(
        "ev_1",
        execution_number=2,
        phase="terminal",
        terminal_status="failed",
    )
    repository.record_summary(
        "ev_1",
        execution_number=2,
        summary={
            "input_tokens": 10,
            "output_tokens": 2,
            "usage_source": "atif",
            "cost_usd": 0.1,
            "cost_source": "provider",
            "raw_artifact_refs": [{"relation": "trajectory", "path": "trajectory.json"}],
        },
    )

    running_sql, running_params = _executed_sql_and_params(cur)
    assert "running_started_at" in running_sql
    assert running_params == ("ev_1", 2)
    terminal_sql, terminal_params = _executed_sql_and_params(cur, 1)
    assert "failure_phase" in terminal_sql
    assert terminal_params == ("ev_1", 2, "failed", "failed")
    summary_sql, summary_params = _executed_sql_and_params(cur, 2)
    assert "raw_artifact_refs" in summary_sql
    assert summary_params[0:2] == ("ev_1", 2)


def test_live_usage_counts_only_unfinished_evaluations() -> None:
    seed_task_revision("task_1")
    seed_task_revision("task_2")
    conn, cur = _conn()
    cur.fetchone.return_value = {
        "queued_runs": 1,
        "active_runs": Decimal("2"),
        "total_parallelism": None,
    }
    cur.fetchall.return_value = [{"owner_id": "usr_1", "queued_runs": 1, "active_runs": 2}]

    result = UserRepository(conn).live_usage()

    summary_sql, actors_sql = (call.args[0] for call in cur.execute.call_args_list)
    for sql in (summary_sql, actors_sql):
        assert "status IN ('queued', 'blocked', 'provisioning', 'running')" in sql
        assert "deleted_at IS NULL" in sql
    assert "GROUP BY owner_id" in actors_sql
    assert result == {
        "queued_runs": 1,
        "active_runs": 2,
        "total_parallelism": 0,
        "total_tasks": 2,
        "actors": [{"owner_id": "usr_1", "queued_runs": 1, "active_runs": 2}],
    }


def test_patch_set_clause_rejects_unlisted_patch_field() -> None:
    with pytest.raises(ValueError, match="not patchable"):
        patch_set_clause([("deleted_at", MALICIOUS)], frozenset({"name"}))


def test_evaluation_create_persists_initial_user_turn_order() -> None:
    seed_task_revision("task_safe")
    conn, cur = _conn()

    EvaluationRepository(conn).create(
        "ev_safe",
        name="run",
        framework="harbor",
        task_id="task_safe",
        task_revision=1,
        framework_profile_id=None,
        harbor_profile_id=None,
        intake_profile_id=None,
        credentials={},
        extra_skill_object_keys=[],
        initial_user_turns=["turn zero", "/skill-b", "/skill-a"],
        runtime="sandbox_k8s",
        parallelism=1,
        visibility="private",
    )

    sql, params = _executed_sql_and_params(cur, len(cur.execute.call_args_list) - 2)
    assert "initial_user_turns" in sql
    assert ["turn zero", "/skill-b", "/skill-a"] in params


def test_evaluation_snapshot_captures_mutable_inputs() -> None:
    cur = MagicMock()
    store = TaskStore()
    store.create_with_initial_revision(
        "task_safe", name="task", slug="task", description=None, visibility="private", object_key="k"
    )
    store.mark_latest_revision_building("task_safe", build_backend="buildkit", build_payload={"source_ref": "abc123"})
    ConfigProfileStore().create("cfg_safe", name="profile", type="harbor", config={"env": {"SAFE": "1"}})

    snapshot = EvaluationRepository._capture_execution_snapshot(
        cur,
        evaluation={"id": "ev_safe", "instruction_prefix": "exact"},
        task_id="task_safe",
        task_revision=1,
        profile_ids={"harbor": "cfg_safe"},
        credentials={"openai": "team-a/openai-key"},
    )

    assert snapshot["profiles"]["harbor"]["config"] == {"env": {"SAFE": "1"}}
    assert snapshot["credentials"] == {"openai": "team-a/openai-key"}
    assert snapshot["task"]["build_payload"] == {"source_ref": "abc123"}
    assert snapshot["task"]["slug"] == "task"
    store.soft_delete("task_safe")
    with pytest.raises(InvalidReference):
        EvaluationRepository._capture_execution_snapshot(
            cur, evaluation={"id": "ev"}, task_id="task_safe", task_revision=1, profile_ids={}, credentials={}
        )
