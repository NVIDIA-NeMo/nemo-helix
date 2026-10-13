# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

pytest.importorskip("scaled_evals")

import scaled_evals.api.build.queue_worker as queue_worker
from scaled_evals.api.build.queue_worker import TaskBuildWorker
from scaled_evals.api.repositories.build_repository import TaskBuildJob


@contextmanager
def _connect():
    yield MagicMock()


def _job(*, attempt: int = 1) -> TaskBuildJob:
    return TaskBuildJob(
        task_id="task_1",
        revision=2,
        backend="aces_uploaded",
        payload={
            "context_path": ".",
            "builder_source_commit": "c" * 40,
        },
        credentials={},
        object_key="task_1/rev/2/tarball.tar.gz",
        attempt=attempt,
    )


def test_worker_completes_claimed_job(monkeypatch: pytest.MonkeyPatch) -> None:
    repo = MagicMock()
    repo.complete.return_value = True
    monkeypatch.setattr(queue_worker, "TaskBuildRepository", lambda conn: repo)
    worker = TaskBuildWorker(connect=_connect, heartbeat_interval=60)
    monkeypatch.setattr(worker, "_execute", lambda job: ("registry/image", "sha256:abc"))

    worker.run(_job())

    repo.complete.assert_called_once_with(
        "task_1",
        2,
        worker_id=worker.worker_id,
        image_ref="registry/image",
        image_digest="sha256:abc",
        verifier_image_ref=None,
        verifier_image_digest=None,
    )


def _prebuilt_job(**payload: str) -> TaskBuildJob:
    return TaskBuildJob(
        task_id="task_1",
        revision=2,
        backend="prebuilt",
        payload={"image_ref": "registry.example.com/team/task:signed", **payload},
        credentials={},
        object_key="task_1/rev/2/tarball.tar.gz",
        attempt=1,
    )


def test_worker_completes_prebuilt_job_with_resolved_verifier_image(monkeypatch: pytest.MonkeyPatch) -> None:
    repo = MagicMock()
    repo.complete.return_value = True
    monkeypatch.setattr(queue_worker, "TaskBuildRepository", lambda conn: repo)
    resolved = MagicMock(
        runtime_ref="registry.example.com/team/task-verifier:signed",
        digest="sha256:" + "b" * 64,
    )
    verify = MagicMock(return_value=resolved)
    monkeypatch.setattr(queue_worker, "resolve_task_image", verify)
    worker = TaskBuildWorker(connect=_connect, heartbeat_interval=60)
    monkeypatch.setattr(worker, "_execute", lambda job: ("registry/image", "sha256:abc"))

    worker.run(
        _prebuilt_job(
            verifier_image_ref="registry.example.com/team/task-verifier:signed",
            verifier_expected_digest="sha256:" + "b" * 64,
        )
    )

    verify.assert_called_once_with(
        "registry.example.com/team/task-verifier:signed",
        expected_digest="sha256:" + "b" * 64,
    )
    repo.complete.assert_called_once_with(
        "task_1",
        2,
        worker_id=worker.worker_id,
        image_ref="registry/image",
        image_digest="sha256:abc",
        verifier_image_ref=resolved.runtime_ref,
        verifier_image_digest=resolved.digest,
    )


def test_worker_retries_when_verifier_image_does_not_resolve(monkeypatch: pytest.MonkeyPatch) -> None:
    repo = MagicMock()
    monkeypatch.setattr(queue_worker, "TaskBuildRepository", lambda conn: repo)
    monkeypatch.setattr(queue_worker, "resolve_task_image", MagicMock(side_effect=RuntimeError("digest mismatch")))
    worker = TaskBuildWorker(connect=_connect, heartbeat_interval=60, max_attempts=3)
    monkeypatch.setattr(worker, "_execute", lambda job: ("registry/image", "sha256:abc"))

    worker.run(_prebuilt_job(verifier_image_ref="registry.example.com/team/task-verifier:signed"))

    repo.complete.assert_not_called()
    assert repo.retry_or_fail.call_args.kwargs["build_error"] == "digest mismatch"


@pytest.mark.parametrize(
    "job",
    [
        _prebuilt_job(),
        TaskBuildJob(
            task_id="task_1",
            revision=2,
            backend="cloudbuild",
            payload={"verifier_image_ref": "registry.example.com/team/task-verifier:signed"},
            credentials={},
            object_key="k",
            attempt=1,
        ),
    ],
)
def test_resolve_verifier_image_skips_jobs_without_prebuilt_verifier(
    monkeypatch: pytest.MonkeyPatch, job: TaskBuildJob
) -> None:
    verify = MagicMock()
    monkeypatch.setattr(queue_worker, "resolve_task_image", verify)

    assert queue_worker.resolve_verifier_image(job) is None
    verify.assert_not_called()


def test_worker_retries_failure_without_losing_job(monkeypatch: pytest.MonkeyPatch) -> None:
    repo = MagicMock()
    monkeypatch.setattr(queue_worker, "TaskBuildRepository", lambda conn: repo)
    worker = TaskBuildWorker(connect=_connect, heartbeat_interval=60, max_attempts=3)

    def fail(job):  # noqa: ANN001
        raise RuntimeError("builder unavailable")

    monkeypatch.setattr(worker, "_execute", fail)
    worker.run(_job(attempt=1))

    repo.retry_or_fail.assert_called_once_with(
        "task_1",
        2,
        worker_id=worker.worker_id,
        build_error="builder unavailable",
        attempt=1,
        max_attempts=3,
        retry_delay=30.0,
    )


def test_worker_records_failure_after_final_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    repo = MagicMock()
    monkeypatch.setattr(queue_worker, "TaskBuildRepository", lambda conn: repo)
    worker = TaskBuildWorker(connect=_connect, heartbeat_interval=60, max_attempts=3)
    monkeypatch.setattr(
        worker,
        "_execute",
        MagicMock(side_effect=RuntimeError("builder still unavailable")),
    )

    worker.run(_job(attempt=3))

    repo.retry_or_fail.assert_called_once_with(
        "task_1",
        2,
        worker_id=worker.worker_id,
        build_error="builder still unavailable",
        attempt=3,
        max_attempts=3,
        retry_delay=30.0,
    )


def test_worker_survives_transient_claim_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = TaskBuildWorker(connect=_connect)
    work_once = MagicMock(side_effect=[RuntimeError("database restarting"), False])
    monkeypatch.setattr(worker, "work_once", work_once)

    def stop_after_idle(_seconds: float) -> None:
        if work_once.call_count == 2:
            raise KeyboardInterrupt

    worker.sleep = stop_after_idle

    with pytest.raises(KeyboardInterrupt):
        worker.work_forever(idle_sleep=0.01)

    assert work_once.call_count == 2


def test_idle_worker_publishes_presence_heartbeat(monkeypatch: pytest.MonkeyPatch) -> None:
    repo = MagicMock()
    monkeypatch.setattr(queue_worker, "TaskBuildRepository", lambda conn: repo)
    worker = TaskBuildWorker(connect=_connect)
    worker.work_once = MagicMock(return_value=False)
    worker.sleep = MagicMock(side_effect=KeyboardInterrupt)

    with pytest.raises(KeyboardInterrupt):
        worker.work_forever(idle_sleep=0.01)

    repo.heartbeat_worker.assert_called_once_with(worker.worker_id)


def test_cloudbuild_job_resolves_registry_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    build = MagicMock(return_value=("registry.example.com/team/task:rev2", "sha256:" + "a" * 64))
    monkeypatch.setattr(queue_worker, "build_cloud_revision_image", build)
    resolved = MagicMock(
        runtime_ref="registry.example.com/team/task@sha256:" + "a" * 64,
        digest="sha256:" + "a" * 64,
    )
    verify = MagicMock(return_value=resolved)
    monkeypatch.setattr(queue_worker, "resolve_task_image", verify)
    worker = TaskBuildWorker(connect=_connect)
    job = TaskBuildJob(
        task_id="task_1",
        revision=2,
        backend="cloudbuild",
        payload={},
        credentials={},
        object_key="task_1/rev/2/tarball.tar.gz",
        attempt=1,
    )

    assert worker._execute(job) == (resolved.runtime_ref, resolved.digest)
    build.assert_called_once_with("task_1", 2, "task_1/rev/2/tarball.tar.gz")
    verify.assert_called_once_with(
        "registry.example.com/team/task:rev2",
        expected_digest="sha256:" + "a" * 64,
    )


def test_prebuilt_job_resolves_registry_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    resolved = MagicMock(
        runtime_ref="registry.example.com/team/task:signed",
        digest="sha256:" + "a" * 64,
    )
    verify = MagicMock(return_value=resolved)
    monkeypatch.setattr(queue_worker, "resolve_task_image", verify)
    job = TaskBuildJob(
        task_id="task_1",
        revision=2,
        backend="prebuilt",
        payload={
            "image_ref": "registry.example.com/team/task:signed",
            "expected_digest": "sha256:" + "a" * 64,
        },
        credentials={},
        object_key="task_1/rev/2/tarball.tar.gz",
        attempt=1,
    )

    assert TaskBuildWorker(connect=_connect)._execute(job) == (
        resolved.runtime_ref,
        resolved.digest,
    )
    verify.assert_called_once_with(
        "registry.example.com/team/task:signed",
        expected_digest="sha256:" + "a" * 64,
    )
