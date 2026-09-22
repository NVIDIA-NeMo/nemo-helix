# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

try:
    from scaled_evals.api.repositories.base_repository import (
        order_by_clause,
        patch_set_clause,
        substring_search_pattern,
    )
    from scaled_evals.api.repositories.benchmark_repository import BenchmarkRepository
    from scaled_evals.api.repositories.benchmark_run_repository import BenchmarkRunRepository
    from scaled_evals.api.repositories.config_profile_repository import ConfigProfileRepository
    from scaled_evals.api.repositories.credential_repository import CredentialRepository
    from scaled_evals.api.repositories.evaluation_repository import EvaluationRepository
    from scaled_evals.api.repositories.execution_cleanup_repository import (
        ExecutionCleanupRepository,
    )
    from scaled_evals.api.repositories.execution_telemetry_repository import (
        ExecutionTelemetryRepository,
    )
    from scaled_evals.api.repositories.ops_repository import OperationsRepository
    from scaled_evals.api.repositories.runtime_resource_repository import RuntimeResourceRepository
    from scaled_evals.api.repositories.switchyard_campaign_repository import (
        SwitchyardCampaignRepository,
    )
    from scaled_evals.api.repositories.task_repository import TaskRepository
    from scaled_evals.api.repositories.user_repository import UserRepository
    from scaled_evals.api.schemas.common import encode_cursor
    from scaled_evals.models.evaluations import EvaluationResultSummary, EvaluationResultWrite
    from scaled_evals.models.runtime import SwitchyardLease
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


def test_task_create_parameterizes_name_and_slug() -> None:
    conn, cur = _conn()

    TaskRepository(conn).create_with_initial_revision(
        "task_safe",
        name=MALICIOUS,
        slug="safe-slug",
        description=MALICIOUS,
        visibility="private",
        object_key="task_safe/rev/1/tarball.tar.gz",
    )

    sql, params = _executed_sql_and_params(cur)
    assert MALICIOUS not in sql
    assert params[1] == MALICIOUS
    assert params[3] == MALICIOUS


def test_task_patch_parameterizes_slug_and_patch_values() -> None:
    conn, cur = _conn()

    TaskRepository(conn).update(
        "task_safe",
        name=MALICIOUS,
        slug="safe-slug",
        description=MALICIOUS,
        visibility="org",
    )

    sql, params = _executed_sql_and_params(cur)
    assert MALICIOUS not in sql
    assert "safe-slug" not in sql
    assert MALICIOUS in params
    assert "safe-slug" in params


def test_task_failed_revision_count_is_digest_scoped_and_parameterized() -> None:
    conn, cur = _conn()
    cur.fetchone.return_value = {"count": 2}

    count = TaskRepository(conn).count_failed_revisions("task_safe", MALICIOUS)

    sql, params = _executed_sql_and_params(cur)
    assert count == 2
    assert MALICIOUS not in sql
    assert params == ("task_safe", MALICIOUS)


def test_benchmark_resolved_tasks_uses_effective_revision_in_one_query() -> None:
    conn, cur = _conn()

    BenchmarkRepository(conn).list_resolved_tasks("bm_safe", 4)

    sql, params = _executed_sql_and_params(cur)
    assert "COALESCE(m.task_revision, t.current_revision)" in sql
    assert "LEFT JOIN task_revisions" in sql
    assert params == ("bm_safe", 4)


def test_benchmark_run_revision_returns_qualification_metadata() -> None:
    conn, cur = _conn()
    qualified_at = datetime(2026, 8, 6, tzinfo=UTC)
    cur.fetchone.return_value = {
        "current_revision": 3,
        "qualification_status": "qualified",
        "qualification_evidence": {"receipt": "qe_1"},
        "qualified_at": qualified_at,
    }

    resolved = BenchmarkRunRepository(conn).benchmark_revision_for_run("bm_safe")

    assert resolved == {
        "revision": 3,
        "qualification_status": "qualified",
        "qualification_evidence": {"receipt": "qe_1"},
        "qualified_at": qualified_at,
    }
    sql, params = _executed_sql_and_params(cur)
    assert "qualification_status" in sql
    assert params == ("bm_safe",)


def test_task_prebuilt_finalize_is_atomic_and_parameterized() -> None:
    conn, cur = _conn()
    cur.fetchone.side_effect = [
        {"id": "task_safe"},
        {
            "revision": 3,
            "status": "uploading",
            "tarball_object_key": "task_safe/rev/3/tarball.tar.gz",
        },
    ]
    cur.rowcount = 1

    result = TaskRepository(conn).finalize_latest_revision_prebuilt(
        "task_safe",
        image_ref=MALICIOUS,
        image_digest="sha256:" + "a" * 64,
        tarball_sha256="package-hash",
    )

    assert result is not None and result.finalized and result.status == "ready"
    update = cur.execute.call_args_list[2]
    assert MALICIOUS not in update.args[0]
    assert update.args[1][0] == MALICIOUS
    assert "status = 'ready'" in update.args[0]


def test_config_profile_create_parameterizes_name() -> None:
    conn, cur = _conn()

    ConfigProfileRepository(conn).create(
        "cfg_safe",
        name=MALICIOUS,
        type="switchyard",
        config={"route": MALICIOUS},
    )

    sql, params = _executed_sql_and_params(cur)
    assert MALICIOUS not in sql
    assert params[1] == MALICIOUS


def test_credential_create_parameterizes_name() -> None:
    conn, cur = _conn()

    CredentialRepository(conn).create(
        "cred_safe",
        name=MALICIOUS,
        provider="openai",
        payload_kind="key",
        encrypted_payload=b"encrypted",
        fingerprint="fp",
        owner_id="user_safe",
    )

    sql, params = _executed_sql_and_params(cur)
    assert MALICIOUS not in sql
    assert params[1] == "user_safe"
    assert params[2] == MALICIOUS


def test_credential_dispatch_load_can_be_owner_scoped() -> None:
    conn, cur = _conn()

    CredentialRepository(conn).load_for_dispatch(
        ["cred_safe"],
        owner_id="user_safe",
        include_unowned=False,
    )

    sql, params = _executed_sql_and_params(cur)
    assert "owner_id = %s" in sql
    assert params == [["cred_safe"], "user_safe", False]


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
        (TaskRepository, {"owner_id": None}, "id ILIKE %s"),
        (BenchmarkRepository, {}, "slug ILIKE %s"),
        (
            BenchmarkRunRepository,
            {"benchmark_id": None, "shared": False},
            "benchmark_id ILIKE %s",
        ),
        (
            CredentialRepository,
            {
                "provider": None,
                "owner_id": "user_safe",
                "include_unowned": False,
            },
            "provider::text ILIKE %s",
        ),
        (
            ConfigProfileRepository,
            {"type": None, "owner_id": None},
            "type::text ILIKE %s",
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


def test_credential_search_is_case_insensitive_across_visible_metadata() -> None:
    conn, cur = _conn()

    CredentialRepository(conn).list(
        provider=None,
        limit=20,
        cursor=None,
        order="desc",
        owner_id="user_safe",
        include_unowned=False,
        q="ANTHROPIC",
    )

    sql, params = _executed_sql_and_params(cur)
    assert "name ILIKE %s" in sql
    assert "provider::text ILIKE %s" in sql
    assert "payload_kind::text ILIKE %s" in sql
    assert "fingerprint ILIKE %s" in sql
    assert params.count("%ANTHROPIC%") == 5


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
    conn, cur = _conn()

    EvaluationRepository(conn).load_for_dispatch("ev_safe")

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
    assert "b.slug AS task_slug" in sql
    assert "r.image_digest" in sql
    assert "r.tarball_sha256" in sql
    assert "r.tarball_object_key" in sql


def test_evaluation_stale_dispatch_jobs_parameterize_reconcile_window() -> None:
    conn, cur = _conn()

    EvaluationRepository(conn).list_stale_dispatch_jobs(stale_seconds=45.5, limit=7)

    sql, params = _executed_sql_and_params(cur)
    assert "dispatch_job_name IS NOT NULL" in sql
    assert "status IN ('provisioning', 'running')" in sql
    assert params == (45.5, 7)


def test_evaluation_reconciler_claim_is_leased_and_skip_locked() -> None:
    conn, cur = _conn()

    EvaluationRepository(conn).claim_stale_dispatch_job(
        stale_seconds=45.5,
        claim_timeout=90,
        worker_id="worker-1",
    )

    sql, params = _executed_sql_and_params(cur)
    assert "FOR UPDATE OF e SKIP LOCKED" in sql
    assert "dispatch_reconcile_claimed_at" in sql
    assert "evaluation_execution_cleanups" in sql
    assert params == (45.5, 90, "worker-1")


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
    switchyard_sql, switchyard_params = _executed_sql_and_params(cur, 2)
    assert "UPDATE evaluation_runtime_resources" in switchyard_sql
    assert "kind = 'switchyard'" in switchyard_sql
    assert "status IN ('provisioned', 'draining', 'delete_failed')" in switchyard_sql
    assert "drain_until = NOW()" in switchyard_sql
    assert switchyard_params == ("ev_safe",)
    campaign_sql, campaign_params = _executed_sql_and_params(cur, 3)
    assert "UPDATE benchmark_switchyard_launches" in campaign_sql
    assert "status = 'cleanup_pending'" in campaign_sql
    assert "permit_expires_at = NOW()" in campaign_sql
    assert "status IN ('launching', 'running')" in campaign_sql
    assert campaign_params == ("ev_safe",)


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
    # The trailing param is the monotonicity guard's status.
    assert status_params[-3:-1] == (2, 2)

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
        [{"status": "delete_failed", "count": 1}],
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
    assert snapshot["switchyard_teardown"] == {"delete_failed": 1}


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
    cur.fetchone.return_value = {
        "task_definitions": 12,
        "tasks_run": 9,
        "evaluation_jobs": 34,
        "evaluation_executions": 39,
        "completed_trials": 81,
        "benchmark_runs": 4,
    }

    totals = OperationsRepository(conn).fleet_totals()

    assert totals == {
        "task_definitions": 12,
        "tasks_run": 9,
        "evaluation_jobs": 34,
        "evaluation_executions": 39,
        "completed_trials": 81,
        "benchmark_runs": 4,
    }
    sql = cur.execute.call_args.args[0]
    assert "COUNT(*) FROM tasks" in sql
    assert "COUNT(DISTINCT task_id)" in sql
    assert "COUNT(*) FROM evaluations" in sql
    assert "SUM(current_execution)" in sql
    assert "SUM(n_trials)" in sql
    assert len(cur.execute.call_args.args) == 1


def test_runtime_resource_upsert_switchyard_parameterizes_metadata() -> None:
    conn, cur = _conn()
    lease = SwitchyardLease(
        profile_id="cfg_sw",
        namespace="evals",
        name="switchyard-safe",
        service_name="switchyard-safe",
        config_map_name="switchyard-safe-routes",
        secret_name="switchyard-safe-secrets",
        endpoint=MALICIOUS,
        openai_base_url=f"{MALICIOUS}/v1",
        anthropic_base_url=MALICIOUS,
        inbound="openai",
        port=4000,
    )

    RuntimeResourceRepository(conn).upsert_switchyard_provisioned(
        evaluation_id="ev_safe",
        execution_number=2,
        lease=lease,
    )

    sql, params = _executed_sql_and_params(cur)
    assert MALICIOUS not in sql
    assert "ON CONFLICT (evaluation_id, execution_number, kind)" in sql
    assert params[0:2] == ("ev_safe", 2)
    assert params[5] == MALICIOUS


def test_switchyard_runtime_resource_queries_are_execution_scoped() -> None:
    conn, cur = _conn()
    repository = RuntimeResourceRepository(conn)

    repository.get_switchyard("ev_safe", 3)
    repository.mark_switchyard_draining("ev_safe", 3, drain_seconds=30)

    get_sql, get_params = _executed_sql_and_params(cur)
    assert "execution_number = %s" in get_sql
    assert get_params == ("ev_safe", 3)
    drain_sql, drain_params = _executed_sql_and_params(cur, 1)
    assert "execution_number = %s" in drain_sql
    assert drain_params[1:] == ("ev_safe", 3)


def test_switchyard_campaign_provision_claim_persists_before_external_mutation() -> None:
    conn, cur = _conn()
    cur.fetchone.side_effect = [
        {
            "benchmark_run_id": "bmr_safe",
            "status": "provisioning",
            "profile_id": "cfg_sw",
            "config_hash": "sha256:config",
            "credential_hash": "sha256:credential",
            "max_concurrent_members": 128,
            "cancel_requested_at": None,
            "claim_owner": None,
            "claim_expires_at": None,
        },
        {
            "benchmark_run_id": "bmr_safe",
            "status": "provisioning",
            "profile_id": "cfg_sw",
        },
    ]

    _row, owns = SwitchyardCampaignRepository(conn).ensure_and_claim_provisioning(
        benchmark_run_id="bmr_safe",
        profile_id="cfg_sw",
        config_hash="sha256:config",
        credential_hash="sha256:credential",
        max_concurrent_members=128,
        worker_id="worker-a",
        claim_seconds=30,
    )

    assert owns is True
    sql = [str(call.args[0]) for call in cur.execute.call_args_list]
    assert "INSERT INTO benchmark_switchyard_campaigns" in sql[0]
    assert "FOR UPDATE" in sql[1]
    assert "cancel_requested_at IS NULL" in sql[2]


def test_switchyard_campaign_failed_provision_persists_cleanup_identity() -> None:
    conn, cur = _conn()
    lease = SwitchyardLease(
        profile_id="cfg_sw",
        namespace="evals",
        name="switchyard-bmr-safe",
        service_name="switchyard-bmr-safe",
        config_map_name="switchyard-bmr-safe-routes",
        secret_name="switchyard-bmr-safe-secrets",
        endpoint="http://switchyard-bmr-safe.evals.svc.cluster.local:4000",
        openai_base_url="http://switchyard-bmr-safe.evals.svc.cluster.local:4000/v1",
        anthropic_base_url="http://switchyard-bmr-safe.evals.svc.cluster.local:4000",
        inbound="openai",
        port=4000,
    )

    SwitchyardCampaignRepository(conn).mark_provision_failed(
        "bmr_safe",
        worker_id="worker-a",
        claim_attempt=3,
        detail="readiness failed; rollback failed",
        lease=lease,
    )

    sql, params = _executed_sql_and_params(cur)
    assert "metadata = COALESCE(%s::jsonb, metadata::jsonb)" in sql
    assert params[1:4] == ("evals", "switchyard-bmr-safe", lease.endpoint)
    assert params[4].obj["name"] == "switchyard-bmr-safe"
    assert params[-1] == 3


def test_switchyard_campaign_heartbeat_is_fenced_by_claim_attempt() -> None:
    conn, cur = _conn()
    cur.rowcount = 1

    renewed = SwitchyardCampaignRepository(conn).renew_provisioning_claim(
        "bmr_safe",
        worker_id="worker-a",
        claim_attempt=4,
        claim_seconds=30,
    )

    assert renewed is True
    sql, params = _executed_sql_and_params(cur)
    assert "claim_expires_at = NOW()" in sql
    assert "claim_owner = %s AND claim_attempt = %s" in sql
    assert params == (30, "bmr_safe", "worker-a", 4)


def test_switchyard_campaign_missing_ready_resources_are_atomically_reclaimed() -> None:
    conn, cur = _conn()
    cur.fetchone.side_effect = [
        {
            "benchmark_run_id": "bmr_safe",
            "status": "ready",
            "cancel_requested_at": None,
        },
        {
            "benchmark_run_id": "bmr_safe",
            "status": "provisioning",
            "claim_owner": "worker-a",
            "claim_attempt": 5,
        },
    ]

    claimed = SwitchyardCampaignRepository(conn).claim_ready_reprovisioning(
        "bmr_safe",
        worker_id="worker-a",
        claim_seconds=30,
        detail="deployment not found",
    )

    assert claimed is not None
    assert claimed["claim_attempt"] == 5
    sql = [str(call.args[0]) for call in cur.execute.call_args_list]
    assert "FOR UPDATE" in sql[0]
    assert "SET status = 'provisioning'" in sql[1]
    assert "claim_attempt = claim_attempt + 1" in sql[1]


def test_switchyard_campaign_ready_unavailable_marks_stale_ready_campaign() -> None:
    conn, cur = _conn()

    SwitchyardCampaignRepository(conn).mark_ready_unavailable(
        "bmr_safe",
        detail="deployment not found",
    )

    sql, params = _executed_sql_and_params(cur)
    assert "status = 'provision_failed'" in sql
    assert "WHERE benchmark_run_id = %s AND status = 'ready'" in sql
    assert params == ("deployment not found", "bmr_safe")


def test_switchyard_campaign_permit_is_campaign_wide_and_atomic() -> None:
    conn, cur = _conn()
    cur.fetchone.side_effect = [
        {"status": "ready", "max_concurrent_members": 512, "cancel_requested_at": None},
        None,
        {"count": 511},
    ]

    decision = SwitchyardCampaignRepository(conn).acquire_launch_permit(
        benchmark_run_id="bmr_safe",
        evaluation_id="ev_safe",
        worker_id="worker-a",
        lease_seconds=30,
    )

    assert decision == "launch"
    sql = "\n".join(str(call.args[0]) for call in cur.execute.call_args_list)
    assert "FROM benchmark_switchyard_campaigns" in sql
    assert "FOR UPDATE" in sql
    assert "status IN ('launching', 'running', 'cleanup_pending')" in sql
    assert "INSERT INTO benchmark_switchyard_launches" in sql


def test_switchyard_campaign_permit_honors_campaign_capacity() -> None:
    conn, cur = _conn()
    cur.fetchone.side_effect = [
        {"status": "ready", "max_concurrent_members": 512, "cancel_requested_at": None},
        None,
        {"count": 512},
    ]

    decision = SwitchyardCampaignRepository(conn).acquire_launch_permit(
        benchmark_run_id="bmr_safe",
        evaluation_id="ev_wait",
        worker_id="worker-a",
        lease_seconds=30,
    )

    assert decision == "wait"
    sql = "\n".join(str(call.args[0]) for call in cur.execute.call_args_list)
    assert "INSERT INTO benchmark_switchyard_launches" not in sql


def test_switchyard_campaign_finalizer_waits_for_terminal_members_and_cleanup() -> None:
    conn, cur = _conn()
    cur.fetchone.return_value = None

    assert (
        SwitchyardCampaignRepository(conn).claim_finalizable(
            worker_id="worker-a",
            claim_seconds=30,
        )
        is None
    )

    sql, params = _executed_sql_and_params(cur)
    assert "e.status NOT IN ('succeeded', 'failed', 'cancelled')" in sql
    assert "c.status IN (" in sql and "'provisioning'" in sql
    assert "AND EXISTS" in sql
    assert "l.status IN ('launching', 'running', 'cleanup_pending')" in sql
    assert "'finalizing'" in sql
    assert "RETURNING" in sql and "c.benchmark_run_id" in sql
    assert "FOR UPDATE SKIP LOCKED" in sql
    assert params == ("worker-a", 30)


def test_switchyard_campaign_cleanup_claim_respects_retry_lease() -> None:
    conn, cur = _conn()

    SwitchyardCampaignRepository(conn).claim_cleanup(worker_id="worker-a", claim_seconds=30)

    sql, params = _executed_sql_and_params(cur)
    assert "l.permit_expires_at IS NULL OR l.permit_expires_at <= NOW()" in sql
    assert "OR l.status = 'cleanup_pending'" not in sql
    assert params == ("worker-a", 30)


def test_switchyard_campaign_cleanup_can_be_abandoned_with_diagnostics() -> None:
    conn, cur = _conn()

    SwitchyardCampaignRepository(conn).abandon_cleanup("ev_safe", detail="delete failed")

    sql, params = _executed_sql_and_params(cur)
    assert "status = 'cleanup_acknowledged'" in sql
    assert "cleanup_error = %s" in sql
    assert "permit_expires_at = NULL" in sql
    assert params == ("delete failed", "ev_safe")


def test_switchyard_campaign_records_managed_lease_before_apply() -> None:
    conn, cur = _conn()
    cur.rowcount = 1
    lease = SwitchyardLease(
        profile_id="cfg_sw",
        benchmark_run_id="bmr_safe",
        namespace="evals",
        name="switchyard-bmr-safe",
        service_name="switchyard-bmr-safe",
        config_map_name="switchyard-bmr-safe-routes",
        secret_name="switchyard-bmr-safe-secrets",
        endpoint="http://switchyard-bmr-safe.evals.svc.cluster.local:4000",
        openai_base_url="http://switchyard-bmr-safe.evals.svc.cluster.local:4000/v1",
        anthropic_base_url="http://switchyard-bmr-safe.evals.svc.cluster.local:4000",
        inbound="openai",
        port=4000,
    )

    assert SwitchyardCampaignRepository(conn).record_provisioning_lease(
        "bmr_safe",
        worker_id="worker-a",
        claim_attempt=2,
        lease=lease,
    )

    sql, params = _executed_sql_and_params(cur)
    assert "status = 'provisioning'" in sql
    assert "cancel_requested_at IS NULL" in sql
    assert "resource_name = %s" in sql
    assert params[0:3] == ("evals", "switchyard-bmr-safe", lease.endpoint)
    assert params[4:] == ("bmr_safe", "worker-a", 2)


def test_switchyard_campaign_deletion_does_not_wait_for_member_archives() -> None:
    conn, cur = _conn()

    SwitchyardCampaignRepository(conn).claim_due_deletion(
        worker_id="worker-a",
        claim_seconds=900,
    )

    sql, params = _executed_sql_and_params(cur)
    assert "c.drain_until <= NOW()" in sql
    assert "e.evidence_status" not in sql
    assert "e.archive_status" not in sql
    assert params == ("worker-a", 900)


def test_switchyard_campaign_failed_deletion_has_retry_backoff() -> None:
    conn, cur = _conn()

    SwitchyardCampaignRepository(conn).mark_delete_failed(
        "bmr_safe",
        worker_id="worker-a",
        detail="kubectl unavailable",
    )

    sql, params = _executed_sql_and_params(cur)
    assert "claim_expires_at = NOW() + INTERVAL '30 seconds'" in sql
    assert params == ("kubectl unavailable", "bmr_safe", "worker-a")


def test_switchyard_campaign_cannot_mark_deleted_without_resource_identity() -> None:
    conn, cur = _conn()

    SwitchyardCampaignRepository(conn).mark_deleted("bmr_safe", worker_id="worker-a")

    sql, params = _executed_sql_and_params(cur)
    assert "resource_name IS NOT NULL" in sql
    assert params == ("bmr_safe", "worker-a")


def test_switchyard_resource_teardown_recovers_terminal_provisioned_leaks() -> None:
    conn, cur = _conn()

    RuntimeResourceRepository(conn).claim_due_switchyard_teardown(
        claim_timeout=30,
        worker_id="worker-a",
    )

    sql, params = _executed_sql_and_params(cur)
    assert "JOIN evaluations e ON e.id = r.evaluation_id" in sql
    assert "r.status = 'provisioned'" in sql
    assert "e.status IN ('succeeded', 'failed', 'cancelled')" in sql
    assert "INTERVAL '5 minutes'" in sql
    assert params == (30, 30, "worker-a")


def test_evaluation_evidence_claim_waits_for_campaign_evidence() -> None:
    conn, cur = _conn()
    cur.fetchone.return_value = None

    EvaluationRepository(conn).claim_next_evidence(claim_timeout=30, worker_id="worker-a")

    sql, _params = _executed_sql_and_params(cur)
    assert "benchmark_switchyard_campaigns" in sql
    assert "c.evidence_status NOT IN ('ready', 'unavailable')" in sql


def test_repository_order_rejects_unallowed_sort_direction() -> None:
    conn, _cur = _conn()

    with pytest.raises(ValueError, match="order"):
        TaskRepository(conn).list(limit=20, cursor=None, order="desc; DROP TABLE tasks")


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


def test_admin_compute_summary_reports_coverage_and_normalizes_decimals() -> None:
    conn, cur = _conn()
    summary = {
        "evaluations": 3,
        "sampled_evaluations": 2,
        "samples": 9,
        "avg_cpu_cores": Decimal("0.25"),
        "peak_cpu_cores": 1.5,
        "avg_cpu_request_cores": Decimal("1.0"),
        "avg_cpu_limit_cores": Decimal("2.0"),
        "avg_cpu_request_utilization_percent": Decimal("25"),
        "avg_memory_bytes": Decimal("1024"),
        "peak_memory_bytes": 2048,
        "avg_memory_request_bytes": Decimal("4096"),
        "avg_memory_limit_bytes": Decimal("8192"),
        "avg_memory_request_utilization_percent": Decimal("25"),
        "requested_gpus": Decimal("1"),
        "gpu_utilization_available": False,
    }
    runtime = {"runtime": "sandbox_k8s", **summary}
    timeline = {"day": datetime(2026, 8, 7, tzinfo=UTC).date(), **summary}
    cur.fetchone.return_value = summary
    cur.fetchall.side_effect = [[runtime], [timeline]]
    window_start = datetime(2026, 8, 1, tzinfo=UTC)
    window_end = datetime(2026, 8, 8, tzinfo=UTC)

    result = UserRepository(conn).compute_summary(window_days=7, window_start=window_start, window_end=window_end)

    summary_sql, summary_params = _executed_sql_and_params(cur)
    assert "LEFT JOIN evaluation_resource_usage" in summary_sql
    assert "cpu_samples * cpu_request_cores" in summary_sql
    assert "FILTER (WHERE usage.cpu_sample_count > 0)" in summary_sql
    assert "FILTER (WHERE usage.memory_sample_count > 0)" in summary_sql
    assert summary_params == (window_start, window_end)
    timeline_sql, timeline_params = _executed_sql_and_params(cur, 2)
    assert "generate_series" in timeline_sql
    assert timeline_params == (window_start, window_end, window_start, window_end)
    assert result["avg_cpu_cores"] == 0.25
    assert result["avg_memory_request_bytes"] == 4096.0
    assert result["gpu_utilization_available"] is False
    assert "gpu_utilization_available" not in result["runtimes"][0]
    assert result["timeline"][0]["avg_cpu_request_cores"] == 1.0


def test_admin_failure_summary_categorizes_and_limits_examples() -> None:
    conn, cur = _conn()
    category_rows = [
        {
            "category": "inference_http_504",
            "category_count": 4,
            "evaluation_id": "ev_504",
            "evaluation_name": "gateway failure",
            "task_id": "task_1",
            "owner_id": "user_1",
            "owner_label": "User One",
            "runtime": "sandbox_k8s",
            "failure_code": "APIStatusError",
            "detail": "inference request returned 504 with api_key=sk-secret123456",
            "occurred_at": datetime(2026, 8, 7, tzinfo=UTC),
        },
        {
            "category": "evaluation_timeout",
            "category_count": 2,
            "evaluation_id": "ev_timeout",
            "evaluation_name": "deadline",
            "task_id": "task_2",
            "owner_id": None,
            "owner_label": None,
            "runtime": "sandbox_k8s",
            "failure_code": "poll_timeout",
            "detail": "timed out waiting for the run to finish",
            "occurred_at": datetime(2026, 8, 6, tzinfo=UTC),
        },
    ]
    timeline_rows = [
        {
            "day": datetime(2026, 8, 6, tzinfo=UTC).date(),
            "total": 2,
            "counts": {"evaluation_timeout": 2},
            "codes": {"evaluation_timeout": {"poll_timeout": 2}},
        },
        {
            "day": datetime(2026, 8, 7, tzinfo=UTC).date(),
            "total": 4,
            "counts": {"inference_http_504": 4},
            "codes": {"inference_http_504": {"APIStatusError": 4}},
        },
    ]
    cur.fetchall.side_effect = [category_rows, timeline_rows]

    window_start = datetime(2026, 7, 9, tzinfo=UTC)
    window_end = datetime(2026, 8, 8, tzinfo=UTC)
    result = UserRepository(conn).failure_summary(
        window_days=30,
        window_start=window_start,
        window_end=window_end,
        examples_per_category=3,
    )

    sql, params = _executed_sql_and_params(cur)
    assert "e.status = 'failed'" in sql
    assert "inference_http_504" in sql
    assert "inference_rate_limit" in sql
    assert "inference_timeout" in sql
    assert "sandbox_startup" in sql
    assert "runtime_cleanup" in sql
    assert "evaluation_timeout" in sql
    assert "trial_cancelled" in sql
    assert "object_storage" in sql
    assert "agent_exit" in sql
    assert "runtime_infrastructure" in sql
    assert "task_configuration" in sql
    assert "control_plane_state" in sql
    assert "ROW_NUMBER() OVER" in sql
    assert params == (window_start, window_end, 3)
    timeline_sql, timeline_params = _executed_sql_and_params(cur, 1)
    assert "generate_series" in timeline_sql
    assert "JSONB_OBJECT_AGG" in timeline_sql
    assert "e.result->>'finished_at'" in timeline_sql
    assert "daily_codes" in timeline_sql
    assert timeline_params == (window_start, window_end, window_start, window_end)
    assert result["total_failures"] == 6
    assert result["categories"][0]["count"] == 4
    assert result["categories"][0]["examples"][0]["evaluation_id"] == "ev_504"
    assert result["categories"][0]["examples"][0]["detail"] == (
        "inference request returned 504 with api_key=<redacted>"
    )
    assert result["timeline"][1]["counts"] == {"inference_http_504": 4}
    assert result["timeline"][1]["codes"] == {"inference_http_504": {"APIStatusError": 4}}


def test_patch_set_clause_rejects_unlisted_patch_field() -> None:
    with pytest.raises(ValueError, match="not patchable"):
        patch_set_clause([("deleted_at", MALICIOUS)], frozenset({"name"}))


def test_evaluation_create_persists_initial_user_turn_order() -> None:
    conn, cur = _conn()

    EvaluationRepository(conn).create(
        "ev_safe",
        name="run",
        framework="harbor",
        task_id="task_safe",
        task_revision=1,
        framework_profile_id=None,
        harbor_profile_id=None,
        switchyard_profile_id=None,
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


def test_evaluation_snapshot_locks_and_captures_mutable_inputs() -> None:
    cur = MagicMock()
    cur.fetchone.return_value = {
        "id": "task_safe",
        "name": "task",
        "slug": "task",
        "revision": 1,
        "status": "ready",
        "build_payload": {"source_ref": "abc123"},
    }
    cur.fetchall.side_effect = [
        [
            {
                "id": "cfg_safe",
                "name": "profile",
                "type": "harbor",
                "config": {"env": {"SAFE": "1"}},
                "updated_at": None,
            }
        ],
        [
            {
                "id": "cred_safe",
                "provider": "openai",
                "payload_kind": "key",
                "fingerprint": "sha256:actual",
                "updated_at": None,
            }
        ],
    ]

    snapshot = EvaluationRepository._capture_execution_snapshot(
        cur,
        evaluation={"id": "ev_safe", "instruction_prefix": "exact"},
        task_id="task_safe",
        task_revision=1,
        profile_ids={"harbor": "cfg_safe"},
        credentials={"openai": "cred_safe"},
    )

    assert snapshot["profiles"]["harbor"]["config"] == {"env": {"SAFE": "1"}}
    assert snapshot["credentials"]["openai"]["fingerprint"] == "sha256:actual"
    assert snapshot["task"]["build_payload"] == {"source_ref": "abc123"}
    assert all("FOR KEY SHARE" in call.args[0] for call in cur.execute.call_args_list)


def test_task_revision_for_finalize_binds_exact_revision() -> None:
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchone.return_value = {
        "revision": 2,
        "status": "uploading",
        "tarball_object_key": "task_one/rev/2/tarball.tar.gz",
    }

    result = TaskRepository(conn).revision_for_finalize("task_one", expected_revision=2)

    assert result is not None
    assert result.revision == 2
    query, params = cur.execute.call_args.args
    assert query.count("%s") == len(params)
    assert params == (2, "task_one")
