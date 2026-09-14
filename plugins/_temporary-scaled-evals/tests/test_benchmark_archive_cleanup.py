# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publication fencing and crash-recoverable archive object cleanup."""

from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

pytest.importorskip("scaled_evals")

from scaled_evals.api import s3
from scaled_evals.api.repositories.base_repository import Conflict
from scaled_evals.api.repositories.benchmark_archive_repository import BenchmarkArchiveRepository
from scaled_evals.benchmark_archive import build_benchmark_archive
from scaled_evals.benchmark_archive_cleanup import cleanup_benchmark_archives
from scaled_evals.dispatch.worker import Dispatcher
from test_benchmark_archives import harbor_files, job, member, tar_bytes

KEY = "benchmark-runs/bmr_1/archives/gen/claim.tar.gz"
OTHER = "benchmark-runs/bmr_1/archives/gen/new-claim.tar.gz"


@contextmanager
def connect():
    yield MagicMock()


@pytest.mark.parametrize("failure", ["upload", "lease"])
@pytest.mark.parametrize("delete_fails", [False, True])
def test_upload_failure_or_lease_loss_removes_uploaded_object(monkeypatch, failure, delete_fails):
    source = tar_bytes(harbor_files())
    monkeypatch.setattr(s3, "stream_object", lambda key: iter([source]))
    monkeypatch.setattr(s3, "list_objects", lambda prefix: [])
    deleted = MagicMock(side_effect=OSError("delete unavailable") if delete_fails else None)
    monkeypatch.setattr(s3, "delete_object", deleted)
    uploaded = False

    def upload(path, key, *, content_type):
        nonlocal uploaded
        uploaded = True
        if failure == "upload":
            raise OSError("upload response lost")
        return path.stat().st_size

    def check_claim():
        if uploaded and failure == "lease":
            raise RuntimeError("lease lost during upload")

    monkeypatch.setattr(s3, "upload_file", upload)
    with pytest.raises((OSError, RuntimeError), match="upload"):
        build_benchmark_archive(job(members=[member(archive_size_bytes=len(source))]), check_claim=check_claim)
    deleted.assert_called_once_with(KEY)


@pytest.mark.parametrize("failure", ["success", "fenced", "members_changed", "commit_ack_lost", "database_down"])
def test_worker_cleans_only_unpublished_objects(monkeypatch, failure):
    state = job(object_key=None)
    repo = MagicMock(spec=BenchmarkArchiveRepository)
    repo.lock_for_cleanup.side_effect = lambda run_id: state
    monkeypatch.setattr("scaled_evals.dispatch.worker.BenchmarkArchiveRepository", lambda conn: repo)
    monkeypatch.setattr("scaled_evals.benchmark_archive_cleanup.BenchmarkArchiveRepository", lambda conn: repo)
    monkeypatch.setattr(
        "scaled_evals.dispatch.worker.build_benchmark_archive",
        lambda *args, **kwargs: {"object_key": KEY, "size_bytes": 123},
    )
    deleted = MagicMock()
    monkeypatch.setattr(s3, "delete_object", deleted)

    def finish(*args, **kwargs):
        if failure == "success":
            state.update(status="ready", object_key=KEY)
            return True
        if failure == "fenced":
            state["claim_token"] = "new-claim"
            return False
        if failure == "members_changed":
            raise Conflict("members_changed", "changed execution")
        if failure == "commit_ack_lost":
            state.update(status="ready", object_key=KEY)
            raise OSError("commit response lost")
        raise OSError("database down")

    def fail(*args):
        if failure == "database_down":
            repo.lock_for_cleanup.side_effect = OSError("database down")
            raise OSError("database down")
        if state["status"] == "building" and state["claim_token"] == "claim":
            state["status"] = "failed"

    repo.finish.side_effect = finish
    repo.fail.side_effect = fail
    worker = Dispatcher(connect=connect)
    if failure == "database_down":
        with pytest.raises(OSError, match="database down"):
            worker.build_benchmark_archive(job())
    else:
        worker.build_benchmark_archive(job())
    if failure in {"fenced", "members_changed"}:
        deleted.assert_called_once_with(KEY)
    else:
        deleted.assert_not_called()
    if failure == "success":
        repo.fail.assert_not_called()


@pytest.mark.parametrize("status", ["building", "ready", "failed"])
def test_reconciliation_protects_current_upload_and_published_object(monkeypatch, status):
    repo = MagicMock(spec=BenchmarkArchiveRepository)
    repo.lock_for_cleanup.return_value = job(status=status, object_key=KEY if status != "building" else None)
    monkeypatch.setattr("scaled_evals.benchmark_archive_cleanup.BenchmarkArchiveRepository", lambda conn: repo)
    keys = [
        KEY,
        OTHER,
        "benchmark-runs/bmr_other/archives/gen/claim.tar.gz",
        "benchmark-runs/bmr_1/artifacts/trace.json",
    ]
    monkeypatch.setattr(s3, "list_objects", lambda prefix: [{"key": key} for key in keys])
    deleted = MagicMock()
    monkeypatch.setattr(s3, "delete_object", deleted)
    cleanup_benchmark_archives(connect, "bmr_1")
    deleted.assert_called_once_with(OTHER)


def test_cleanup_delete_failure_can_be_retried(monkeypatch):
    repo = MagicMock(spec=BenchmarkArchiveRepository)
    repo.lock_for_cleanup.return_value = job(status="failed", object_key=None)
    monkeypatch.setattr("scaled_evals.benchmark_archive_cleanup.BenchmarkArchiveRepository", lambda conn: repo)
    monkeypatch.setattr(s3, "list_objects", lambda prefix: [{"key": KEY}])
    deleted = MagicMock(side_effect=[OSError("store unavailable"), None])
    monkeypatch.setattr(s3, "delete_object", deleted)
    with pytest.raises(OSError, match="store unavailable"):
        cleanup_benchmark_archives(connect, "bmr_1")
    assert cleanup_benchmark_archives(connect, "bmr_1") == 1
    assert deleted.call_count == 2


def test_idle_dispatcher_reconciles_crashed_upload(monkeypatch):
    from scaled_evals.api.settings import settings

    repo = MagicMock(spec=BenchmarkArchiveRepository)
    repo.claim.return_value = None
    repo.claim_cleanup.return_value = "bmr_1"
    repo.lock_for_cleanup.return_value = job(status="ready", object_key=OTHER)
    monkeypatch.setattr("scaled_evals.dispatch.worker.BenchmarkArchiveRepository", lambda conn: repo)
    monkeypatch.setattr("scaled_evals.benchmark_archive_cleanup.BenchmarkArchiveRepository", lambda conn: repo)
    monkeypatch.setattr(settings, "dispatch_kubernetes_jobs_enabled", False)
    monkeypatch.setattr(s3, "list_objects", lambda prefix: [{"key": KEY}, {"key": OTHER}])
    deleted = MagicMock()
    monkeypatch.setattr(s3, "delete_object", deleted)
    worker = Dispatcher(connect=connect)
    for method in (
        "claim_next_switchyard_teardown",
        "claim_next_switchyard_campaign_cleanup",
        "claim_next_switchyard_campaign_finalization",
        "claim_next_switchyard_campaign_deletion",
        "claim_next",
        "claim_next_evidence",
        "claim_next_archive",
    ):
        monkeypatch.setattr(worker, method, lambda: None)
    assert worker.work_once() is True
    repo.claim_cleanup.assert_called_once_with(interval_seconds=settings.benchmark_archive_cleanup_interval_seconds)
    deleted.assert_called_once_with(KEY)


def test_repeated_reconciliation_catches_upload_that_finished_after_worker_death(monkeypatch):
    repo = MagicMock(spec=BenchmarkArchiveRepository)
    repo.lock_for_cleanup.return_value = job(status="failed", object_key=None)
    monkeypatch.setattr("scaled_evals.benchmark_archive_cleanup.BenchmarkArchiveRepository", lambda conn: repo)
    objects = []
    monkeypatch.setattr(s3, "list_objects", lambda prefix: objects.copy())
    deleted = MagicMock()
    monkeypatch.setattr(s3, "delete_object", deleted)
    cleanup_benchmark_archives(connect, "bmr_1")
    deleted.assert_not_called()
    objects.append({"key": KEY})
    cleanup_benchmark_archives(connect, "bmr_1")
    deleted.assert_called_once_with(KEY)


def test_reconciliation_rechecks_reference_after_listing(monkeypatch):
    repo = MagicMock(spec=BenchmarkArchiveRepository)
    state = job(status="building", object_key=None)
    repo.lock_for_cleanup.side_effect = lambda run_id: state
    monkeypatch.setattr("scaled_evals.benchmark_archive_cleanup.BenchmarkArchiveRepository", lambda conn: repo)

    def list_objects(prefix):
        state.update(status="ready", object_key=KEY)
        return [{"key": KEY}]

    monkeypatch.setattr(s3, "list_objects", list_objects)
    deleted = MagicMock()
    monkeypatch.setattr(s3, "delete_object", deleted)
    cleanup_benchmark_archives(connect, "bmr_1")
    deleted.assert_not_called()
