# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from datetime import UTC, datetime

import pytest

pytest.importorskip("scaled_evals")

from scaled_evals.api.benchmark_archive_store import BenchmarkArchiveStore
from scaled_evals.api.benchmark_run_store import BenchmarkRunStore
from scaled_evals.api.repositories.base_repository import Conflict
from task_fixtures import seed_benchmark_run, seed_evaluation, seed_task_revision

_RUN = {"framework": "harbor", "runtime": "sandbox_k8s", "parallelism": 1, "benchmark_id": "bm_1"}
_READY_MEMBER = {
    "task_id": "task_1",
    "task_revision": 1,
    "benchmark_run_id": "bmr_1",
    "status": "succeeded",
    "archive_status": "ready",
    "archive_object_key": "key",
    "evidence_status": "ready",
    "archive_built_at": datetime(2026, 1, 1, tzinfo=UTC),
    "archive_size_bytes": 42,
}


def _seed_run(**member_fields) -> None:  # noqa: ANN003
    """Store run `bmr_1` with one ready member evaluation `ev_1`."""
    seed_benchmark_run({**_RUN, "id": "bmr_1", "name": "run", "benchmark_revision": 1})
    seed_task_revision("task_1")
    seed_evaluation({**_READY_MEMBER, "id": "ev_1", **member_fields})


def _publish(store: BenchmarkArchiveStore, job: dict) -> bool:
    return store.finish(job, object_key="published", size_bytes=1)


def test_request_snapshots_members_and_is_idempotent_until_forced() -> None:
    _seed_run(
        status="failed",
        current_execution=2,
        execution_snapshot={"task": {"slug": "smoke-one", "name": "Smoke One"}},
    )
    store = BenchmarkArchiveStore()
    assert store.get("bmr_1") is None
    queued = store.request("bmr_1")
    assert queued["status"] == "queued"
    [member] = queued["members"]
    assert (member["id"], member["current_execution"]) == ("ev_1", 2)
    assert (member["task_slug"], member["task_name"]) == ("smoke-one", "Smoke One")
    assert store.request("bmr_1", force=True)["generation"] == queued["generation"]

    job = store.claim(claim_timeout=30, run_id="bmr_1")
    assert job is not None and job["status"] == "building"
    assert store.request("bmr_1", force=True)["generation"] == queued["generation"]
    assert _publish(store, job)
    ready = store.get("bmr_1")
    assert ready is not None and (ready["status"], ready["object_key"]) == ("ready", "published")
    assert store.request("bmr_1")["generation"] == queued["generation"]

    rebuilt = store.request("bmr_1", force=True)
    assert rebuilt["generation"] != queued["generation"]
    assert (rebuilt["status"], rebuilt["object_key"], rebuilt["attempts"]) == ("queued", None, 0)


@pytest.mark.parametrize(
    "changes, code",
    [
        ({"status": "running"}, "run_not_terminal"),
        ({"archive_status": "building"}, "member_artifacts_not_ready"),
        ({"evidence_status": "building"}, "member_artifacts_not_ready"),
        ({"cancel_teardown_status": "pending"}, "member_artifacts_not_ready"),
    ],
)
def test_request_rejects_unstable_members(changes, code) -> None:  # noqa: ANN001
    _seed_run(**changes)
    with pytest.raises(Conflict) as exc:
        BenchmarkArchiveStore().request("bmr_1")
    assert exc.value.code == code
    assert BenchmarkArchiveStore().get("bmr_1") is None


def test_lease_is_bounded_and_fences_stale_workers() -> None:
    _seed_run()
    store = BenchmarkArchiveStore()
    assert store.claim(claim_timeout=30, run_id="bmr_missing") is None
    store.request("bmr_1")
    assert [row["benchmark_run_id"] for row in store.list_claimable(claim_timeout=30, limit=10)] == ["bmr_1"]

    first = store.claim(claim_timeout=30, run_id="bmr_1")
    assert first is not None and (first["status"], first["attempts"]) == ("building", 1)
    assert store.claim(claim_timeout=30, run_id="bmr_1") is None
    assert store.list_claimable(claim_timeout=30, limit=10) == []
    assert not store.heartbeat("bmr_1", "other") and store.heartbeat("bmr_1", first["claim_token"])

    # An expired lease is reclaimed under a new token; the old worker can no
    # longer publish or fail it.
    second = store.claim(claim_timeout=-1, run_id="bmr_1")
    assert second is not None and second["claim_token"] != first["claim_token"]
    assert not _publish(store, first)
    store.fail(first, "stale")
    current = store.get("bmr_1")
    assert current is not None and (current["status"], current["error"]) == ("building", None)

    # A Job that ended before claiming counts as an attempt; the third fails
    # the archive, and a stale attempt count is ignored.
    store.skip_attempt("bmr_1", generation=second["generation"], attempts=1)
    store.skip_attempt("bmr_1", generation=second["generation"], attempts=2)
    store.skip_attempt("bmr_1", generation=second["generation"], attempts=3)
    exhausted = store.get("bmr_1")
    assert exhausted is not None and (exhausted["status"], exhausted["attempts"]) == ("failed", 3)
    assert exhausted["error"] == "archive build job ended before claiming"


def test_finish_refuses_changed_members() -> None:
    _seed_run()
    store = BenchmarkArchiveStore()
    store.request("bmr_1")
    job = store.claim(claim_timeout=30, run_id="bmr_1")
    assert job is not None
    with pytest.raises(Conflict, match="members_changed"):
        _publish(store, {**job, "members": [{"id": "ev_old"}]})
    current = store.get("bmr_1")
    assert current is not None and current["status"] == "building"


def test_cleanup_sweep_is_throttled_and_reaches_deleted_runs() -> None:
    _seed_run()
    store = BenchmarkArchiveStore()
    store.request("bmr_1")
    assert BenchmarkRunStore().soft_delete("bmr_1")
    assert store.get("bmr_1") is None and store.list_claimable(claim_timeout=30, limit=10) == []
    assert store.claim_cleanup(interval_seconds=300) == "bmr_1"
    assert store.claim_cleanup(interval_seconds=300) is None
    assert store.claim_cleanup(interval_seconds=-1) == "bmr_1"
    row = store.for_cleanup("bmr_1")
    assert row is not None and row["status"] == "queued"
