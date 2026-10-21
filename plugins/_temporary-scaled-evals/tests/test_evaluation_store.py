# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write guards of the entity-backed evaluation store."""

from datetime import UTC, datetime

import pytest

pytest.importorskip("scaled_evals")

from scaled_evals.api.config_profile_store import ConfigProfileStore
from scaled_evals.api.evaluation_store import EvaluationStore
from scaled_evals.api.repositories.base_repository import InvalidReference
from scaled_evals.api.task_store import TaskStore
from scaled_evals.models.evaluations import EvaluationResultSummary, EvaluationResultWrite
from task_fixtures import seed_evaluation, seed_task_revision


def _seed(evaluation_id: str, **fields) -> None:  # noqa: ANN003
    seed_task_revision("task_x", 1)
    seed_evaluation({"id": evaluation_id, **fields})


def _stored(evaluation_id: str) -> dict:
    row = EvaluationStore().load_for_dispatch(evaluation_id)
    assert row is not None
    return row


def _last_event(evaluation_id: str) -> dict:
    return EvaluationStore().list_events(evaluation_id, limit=100, cursor=None, offset=0)[-1]


def test_status_writes_are_fenced_monotonic_and_redacted() -> None:
    _seed("ev_1", status="provisioning", current_execution=2)
    store = EvaluationStore()

    # A stale execution cannot write.
    assert not store.set_status("ev_1", "running", expected_execution_number=1)
    assert store.set_status("ev_1", "running", expected_execution_number=2)
    assert store.set_status("ev_1", "failed", detail="runner failed with api_key=sk-secret-value")
    assert _stored("ev_1")["status_detail"] == "runner failed with api_key=<redacted>"
    assert _last_event("ev_1")["detail"] == "runner failed with api_key=<redacted>"
    # Terminal never moves back, a cancel may still land, and a cancel is final.
    assert not store.set_status("ev_1", "running")
    assert store.set_status("ev_1", "cancelled")
    assert not store.set_status("ev_1", "failed")
    assert _stored("ev_1")["status"] == "cancelled"


def test_persist_result_writes_the_summary_without_numeric_coercion() -> None:
    _seed("ev_1", status="running", current_execution=2)
    _seed("ev_bool", status="running")
    store = EvaluationStore()
    result = EvaluationResultWrite(
        result={"note": "x"},
        summary=EvaluationResultSummary(
            reward=0.5, n_trials=2, n_completed=1, n_errored=1, finished_at="2026-06-23T20:00:00"
        ),
        extra_detail="extra",
    )

    assert not store.persist_result("ev_1", result, expected_execution_number=1)
    assert store.persist_result("ev_1", result, expected_execution_number=2)
    stored = _stored("ev_1")
    assert (stored["status"], stored["status_detail"]) == ("succeeded", "1/2 trials completed; extra")
    assert (stored["reward"], stored["n_trials"], stored["n_completed"]) == (0.5, 2, 1)
    assert stored["finished_at"] == datetime(2026, 6, 23, 20, tzinfo=UTC)

    boolean = EvaluationResultWrite(
        result={"verifier_result": {"score": True}},
        summary=EvaluationResultSummary(reward=True, n_trials=1, n_completed=1, n_errored=0),
    )
    assert store.persist_result("ev_bool", boolean)
    stored = _stored("ev_bool")
    assert (stored["reward_value"], stored["reward"]) == (True, None)
    # A summary without a finish time keeps the moment the result was written.
    assert stored["finished_at"] is not None


def test_manual_retry_resets_terminal_state_unless_artifacts_are_finalizing() -> None:
    _seed("ev_busy", status="failed", archive_claimed_by="worker-1")
    _seed(
        "ev_1",
        status="failed",
        current_execution=4,
        max_executions=4,
        result={"reward": 0},
        archive_status="ready",
        archive_object_key="old",
        evidence_status="ready",
    )
    store = EvaluationStore()

    assert store.retry_block_reason("ev_busy") == "terminal_artifacts_finalizing"
    assert store.retry_failed("ev_busy") is None

    assert store.retry_block_reason("ev_1") is None
    assert store.retry_failed("ev_1") is not None
    stored = _stored("ev_1")
    assert (stored["status"], stored["current_execution"], stored["max_executions"]) == ("queued", 5, 5)
    assert (stored["archive_status"], stored["archive_object_key"], stored["evidence_status"]) == (
        "missing",
        None,
        "missing",
    )
    assert stored["result"] is None
    event = _last_event("ev_1")
    assert (event["type"], event["status"], event["detail"]) == ("retry", "queued", "manual retry 5/5 scheduled")
    # Only a failed evaluation retries.
    assert store.retry_failed("ev_1") is None


def test_failed_dispatch_job_hands_its_runtime_to_a_leased_cleanup_with_backoff() -> None:
    _seed(
        "ev_1",
        status="running",
        current_execution=2,
        dispatch_job_name="job-2",
        backend_handle='{"external_id": "pod-2"}',
        dispatch_reconcile_claimed_by="reconciler",
    )
    store = EvaluationStore()
    failure = {
        "execution_number": 2,
        "dispatch_job_name": "job-2",
        "failure_code": "runner_disappeared",
        "detail": "job died",
        "retry_delay_seconds": 0,
    }

    # Only the reconciler holding the lease can fail the Job.
    assert store.record_dispatch_job_infrastructure_failure("ev_1", reconcile_worker_id="other", **failure) is None
    outcome = store.record_dispatch_job_infrastructure_failure("ev_1", reconcile_worker_id="reconciler", **failure)
    assert outcome == {"action": "cleanup", "retry": True}

    claimed = store.claim_cleanup(worker_id="worker-1", claim_timeout=30)
    assert claimed is not None and claimed["evaluation_id"] == "ev_1" and claimed["execution_number"] == 2
    assert store.claim_cleanup(worker_id="worker-2", claim_timeout=30) is None  # leased
    store.mark_cleanup_failed(claimed["id"], worker_id="worker-1", detail="backend unavailable")
    [cleanup] = store.cleanups_for("ev_1")
    assert (cleanup["status"], cleanup["delete_error"]) == ("delete_failed", "backend unavailable")
    assert "backend_handle" not in cleanup
    assert store.claim_cleanup(worker_id="worker-2", claim_timeout=30) is None  # backing off


def test_dispatch_load_and_snapshot_freeze_task_and_profile_inputs() -> None:
    seed_task_revision("task_safe", slug="task-safe", image_digest="sha256:task", verifier_image_ref="verifier:1")
    seed_evaluation({"id": "ev_1", "task_id": "task_safe", "task_revision": 1})
    seed_evaluation({"id": "ev_gone", "task_id": "task_safe", "task_revision": 9})
    row = _stored("ev_1")
    assert (row["task_slug"], row["image_digest"], row["verifier_image_ref"]) == (
        "task-safe",
        "sha256:task",
        "verifier:1",
    )
    assert EvaluationStore().load_for_dispatch("ev_gone") is None

    tasks = TaskStore()
    tasks.create_with_initial_revision(
        "task_snap", name="task", slug="task", description=None, visibility="private", object_key="k"
    )
    tasks.mark_latest_revision_building("task_snap", build_backend="buildkit", build_payload={"source_ref": "abc"})
    ConfigProfileStore().create("cfg_safe", name="profile", type="harbor", config={"env": {"SAFE": "1"}})
    snapshot = EvaluationStore._capture_execution_snapshot(
        evaluation={"id": "ev_snap"},
        task_id="task_snap",
        task_revision=1,
        profile_ids={"harbor": "cfg_safe"},
        credentials={"openai": "team-a/openai-key"},
    )
    assert snapshot["profiles"]["harbor"]["config"] == {"env": {"SAFE": "1"}}
    assert snapshot["credentials"] == {"openai": "team-a/openai-key"}
    assert (snapshot["task"]["slug"], snapshot["task"]["build_payload"]) == ("task", {"source_ref": "abc"})
    tasks.soft_delete("task_snap")
    with pytest.raises(InvalidReference):
        EvaluationStore._capture_execution_snapshot(
            evaluation={"id": "ev"}, task_id="task_snap", task_revision=1, profile_ids={}, credentials={}
        )


def test_live_usage_counts_only_unfinished_evaluations() -> None:
    _seed("ev_q", status="queued", owner_id="usr_1", parallelism=2)
    _seed("ev_r", status="running", owner_id="usr_1", parallelism=3)
    _seed("ev_done", status="succeeded", owner_id="usr_1")

    usage = EvaluationStore().live_usage()

    assert (usage["queued_runs"], usage["active_runs"], usage["total_parallelism"]) == (1, 1, 5)
    assert usage["total_tasks"] == 1
    [actor] = usage["actors"]
    assert (actor["owner_id"], actor["queued_runs"], actor["active_runs"]) == ("usr_1", 1, 1)
