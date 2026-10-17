# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

pytest.importorskip("scaled_evals")

from nemo_scaled_evals_plugin.entities import ScaledEvalsTask
from scaled_evals.api.repositories.base_repository import Conflict
from scaled_evals.api.task_store import TaskStore
from task_fixtures import stored_revision, stored_task


def _building_task(store: TaskStore, task_id: str = "task_1", slug: str = "one") -> None:
    store.create_with_initial_revision(
        task_id, name=task_id, slug=slug, description=None, visibility="private", object_key=f"{task_id}/1"
    )
    store.mark_latest_revision_building(
        task_id, build_backend="buildkit", build_payload={"p": 1}, build_credentials={"registry": "secret-name"}
    )


def test_build_lease_lifecycle() -> None:
    store = TaskStore()
    _building_task(store)

    job = store.claim_next(worker_id="w1", claim_timeout=90, max_attempts=3)
    assert job is not None and (job.task_id, job.revision, job.attempt) == ("task_1", 1, 1)
    assert job.credentials == {"registry": "secret-name"}
    assert store.claim_next(worker_id="w2", claim_timeout=90, max_attempts=3) is None
    assert store.heartbeat("task_1", 1, worker_id="w1")
    assert not store.heartbeat("task_1", 1, worker_id="w2")

    assert store.bind_platform_job("task_1", 1, worker_id="w1", job_name="scaled-evals-build-abc")
    assert store.list_platform_jobs(limit=10) == [
        {"task_id": "task_1", "revision": 1, "build_attempts": 1, "job_name": "scaled-evals-build-abc"}
    ]

    owner = "scaled-evals-build-abc"
    assert store.retry_or_fail(
        "task_1", 1, worker_id=owner, build_error="flaky", attempt=1, max_attempts=3, retry_delay=0
    )
    assert stored_revision("task_1", 1)["status"] == "building"
    job = store.claim_next(worker_id="w3", claim_timeout=90, max_attempts=3)
    assert job is not None and job.attempt == 2
    assert store.complete(
        "task_1",
        1,
        worker_id="w3",
        image_ref="registry/task@sha256:a",
        image_digest="sha256:a",
        verifier_image_ref="registry/verifier@sha256:b",
        verifier_image_digest="sha256:b",
    )
    revision = stored_revision("task_1", 1)
    assert (revision["status"], revision["image_digest"], revision["verifier_image_digest"]) == (
        "ready",
        "sha256:a",
        "sha256:b",
    )
    assert revision["build_claimed_by"] is None and stored_task("task_1")["current_revision"] == 1
    assert not store.complete("task_1", 1, worker_id="w3", image_ref="x", image_digest="y")


def test_final_attempt_lease_expiry_and_terminal_failure(entity_store) -> None:
    store = TaskStore()
    _building_task(store)
    assert store.claim_next(worker_id="w1", claim_timeout=90, max_attempts=1) is not None

    entity = entity_store.get(ScaledEvalsTask, "task_1", workspace="scaled-evals-internal")
    entity.revisions[0].build_claimed_at = datetime.now(tz=UTC) - timedelta(seconds=120)
    entity_store.update(entity)

    assert store.claim_next(worker_id="w2", claim_timeout=90, max_attempts=1) is None
    revision = stored_revision("task_1", 1)
    assert revision["status"] == "failed"
    assert revision["build_error"] == "build worker lease expired after final attempt"

    _building_task(store, "task_2", "two")
    store.claim_next(worker_id="w1", claim_timeout=90, max_attempts=3)
    assert store.retry_or_fail(
        "task_2", 1, worker_id="w1", build_error="broken", attempt=3, max_attempts=3, retry_delay=30
    )
    assert stored_revision("task_2", 1)["status"] == "failed"


def test_views_soft_delete_and_slug_conflict() -> None:
    store = TaskStore()
    _building_task(store)
    with pytest.raises(Conflict):
        store.create_with_initial_revision(
            "task_2", name="dup", slug="one", description=None, visibility="private", object_key="k"
        )
    created = store.create_next_revision("task_1")
    assert created is not None and created.revision == 2

    members = [{"task_id": "task_1", "task_revision": None}, {"task_id": "task_1", "task_revision": 2}]
    resolved = store.resolve_members(members, include_deleted=False)
    assert [(task["slug"], rev and rev["revision"]) for _, task, rev in resolved] == [("one", 2), ("one", 2)]

    assert store.soft_delete("task_1") and not store.soft_delete("task_1")
    assert store.get("task_1") is None and store.count() == 0 and store.count(include_deleted=True) == 1
    assert store.revision("task_1", 1, include_deleted=False) is None
    assert stored_revision("task_1", 1)["task_deleted"] is True
    assert store.resolve_members(members, include_deleted=False) == []
    assert set(store.revisions([("task_1", 1), ("task_1", 9)])) == {("task_1", 1)}
