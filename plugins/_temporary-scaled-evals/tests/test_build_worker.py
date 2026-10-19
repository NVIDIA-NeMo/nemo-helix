# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

pytest.importorskip("scaled_evals")

from scaled_evals.api.build import worker


def test_finalize_build_persists_success(monkeypatch: pytest.MonkeyPatch) -> None:
    async def build(task_id: str, revision: int, object_key: str) -> tuple[str, str]:
        assert (task_id, revision, object_key) == ("task_1", 2, "task_1/rev/2.tar.gz")
        return "registry.example/task:rev2", "sha256:abc"

    success = MagicMock()
    failure = MagicMock()
    monkeypatch.setattr(worker, "build_revision_image", build)
    monkeypatch.setattr(worker, "_record_success", success)
    monkeypatch.setattr(worker, "_record_failure", failure)

    worker.run_finalize_build("task_1", 2, "task_1/rev/2.tar.gz")

    success.assert_called_once_with("task_1", 2, "registry.example/task:rev2", "sha256:abc")
    failure.assert_not_called()


def test_finalize_build_persists_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    async def build(_task_id: str, _revision: int, _object_key: str) -> tuple[str, str]:
        raise RuntimeError("buildkit unavailable")

    success = MagicMock()
    failure = MagicMock()
    monkeypatch.setattr(worker, "build_revision_image", build)
    monkeypatch.setattr(worker, "_record_success", success)
    monkeypatch.setattr(worker, "_record_failure", failure)

    worker.run_finalize_build("task_1", 2, "task_1/rev/2.tar.gz")

    failure.assert_called_once_with("task_1", 2, "buildkit unavailable")
    success.assert_not_called()


@pytest.mark.parametrize("fails", [False, True])
def test_finalize_uploaded_persists_terminal_result(monkeypatch: pytest.MonkeyPatch, fails: bool) -> None:
    def resolve(**kwargs: object) -> tuple[str, str]:
        assert kwargs == {
            "tarball_object_key": "task_1/rev/2.tar.gz",
            "context_path": "tasks/demo",
        }
        if fails:
            raise RuntimeError("builder rejected archive")
        return "registry.example/task:signed", "sha256:def"

    success = MagicMock()
    failure = MagicMock()
    monkeypatch.setattr(worker, "resolve_uploaded_revision_image", resolve)
    monkeypatch.setattr(worker, "_record_success", success)
    monkeypatch.setattr(worker, "_record_failure", failure)

    worker.run_finalize_uploaded(
        "task_1",
        2,
        tarball_object_key="task_1/rev/2.tar.gz",
        context_path="tasks/demo",
    )

    if fails:
        failure.assert_called_once_with("task_1", 2, "builder rejected archive")
        success.assert_not_called()
    else:
        success.assert_called_once_with("task_1", 2, "registry.example/task:signed", "sha256:def")
        failure.assert_not_called()


def test_terminal_persistence_helpers_use_repository(monkeypatch: pytest.MonkeyPatch) -> None:
    connection = MagicMock()
    repository = MagicMock()

    @contextmanager
    def connect():
        yield connection

    monkeypatch.setattr(worker, "_connect", connect)
    repository_factory = MagicMock(return_value=repository)
    monkeypatch.setattr(worker, "TaskBuildRepository", repository_factory)

    worker._record_success("task_1", 2, "registry.example/task:rev2", "sha256:abc")
    worker._record_failure("task_1", 3, "terminal failure")

    assert repository_factory.call_count == 2
    repository_factory.assert_any_call(connection)
    repository.record_success.assert_called_once_with(
        "task_1",
        2,
        image_ref="registry.example/task:rev2",
        image_digest="sha256:abc",
    )
    repository.record_failure.assert_called_once_with("task_1", 3, "terminal failure")
