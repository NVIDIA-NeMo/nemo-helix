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
    )


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
