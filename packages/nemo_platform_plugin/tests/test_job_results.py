# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :mod:`nemo_platform_plugin.job_results`.

Pins the :class:`LocalJobResults` contract:

- ``save`` copies a file under ``<root>/<name>`` and returns a
  :class:`ResultRef` with a ``file://`` URL.
- ``save`` on a directory copies recursively and honours
  ``ignore_patterns``.
- ``save`` is idempotent on ``name`` (a second call overwrites,
  regardless of whether the artefact kind changes between calls).
- ``save`` is a no-op when ``local_path == <root>/<name>`` — a job that
  already wrote into the results directory can register without
  duplicating.
- :class:`LocalJobResults` satisfies the :class:`JobResults` ABC.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest
from nemo_platform_plugin.client.client import NemoClient
from nemo_platform_plugin.files.client import AsyncFilesClient, FilesClient
from nemo_platform_plugin.job_results import (
    JobResults,
    LocalJobResults,
    PlatformJobResults,
    ResultRef,
)
from nemo_platform_plugin.jobs.client import AsyncJobsClient, JobsClient
from nemo_platform_plugin.jobs.constants import NEMO_JOB_WORKSPACE_ENVVAR
from nemo_platform_plugin.jobs.result_manager import (
    AsyncResultManager,
    ResultManager,
    async_result_manager_factory,
    result_manager_factory,
)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "results"


class TestInheritance:
    def test_local_job_results_is_a_job_results(self, root: Path) -> None:
        assert isinstance(LocalJobResults(root=root), JobResults)


class TestSaveFile:
    def test_save_copies_file_into_root(self, tmp_path: Path, root: Path) -> None:
        src = tmp_path / "src.txt"
        src.write_text("hello")
        results = LocalJobResults(root=root)
        ref = results.save("greeting", src)
        assert isinstance(ref, ResultRef)
        assert ref.name == "greeting"
        copied = root / "greeting"
        assert copied.exists()
        assert copied.read_text() == "hello"
        assert ref.artifact_url == f"file://{copied.resolve()}"

    def test_save_in_place_does_not_recopy(self, root: Path) -> None:
        root.mkdir(parents=True)
        existing = root / "in-place"
        existing.write_text("existing")
        results = LocalJobResults(root=root)
        ref = results.save("in-place", existing)
        assert existing.read_text() == "existing"
        assert ref.artifact_url == f"file://{existing.resolve()}"

    def test_save_overwrites_existing_file(self, tmp_path: Path, root: Path) -> None:
        first = tmp_path / "first.txt"
        first.write_text("alpha")
        second = tmp_path / "second.txt"
        second.write_text("bravo-bravo")
        results = LocalJobResults(root=root)
        results.save("doc", first)
        results.save("doc", second)
        assert (root / "doc").read_text() == "bravo-bravo"

    def test_save_missing_file_raises(self, tmp_path: Path, root: Path) -> None:
        results = LocalJobResults(root=root)
        with pytest.raises(FileNotFoundError, match="missing"):
            results.save("missing", tmp_path / "nope.txt")


class TestSaveDirectory:
    def test_save_copies_directory_recursively(self, tmp_path: Path, root: Path) -> None:
        src = tmp_path / "payload"
        (src / "nested").mkdir(parents=True)
        (src / "a.txt").write_text("A")
        (src / "nested" / "b.txt").write_text("B")
        results = LocalJobResults(root=root)
        ref = results.save("artifacts", src)
        dst = root / "artifacts"
        assert dst.is_dir()
        assert (dst / "a.txt").read_text() == "A"
        assert (dst / "nested" / "b.txt").read_text() == "B"
        assert ref.artifact_url == f"file://{dst.resolve()}"

    def test_save_directory_honours_ignore_patterns(self, tmp_path: Path, root: Path) -> None:
        src = tmp_path / "payload"
        (src / "cache").mkdir(parents=True)
        (src / "cache" / "cache.db").write_text("junk")
        (src / "keep.txt").write_text("K")
        results = LocalJobResults(root=root)
        # Trailing-slash form mirrors the platform impl's accepted shape;
        # the local builder normalizes it before fnmatch.
        results.save("artifacts", src, ignore_patterns=["cache.db", "cache/"])
        dst = root / "artifacts"
        assert (dst / "keep.txt").read_text() == "K"
        assert not (dst / "cache").exists()

    def test_save_directory_overwrites(self, tmp_path: Path, root: Path) -> None:
        src1 = tmp_path / "p1"
        src1.mkdir()
        (src1 / "one.txt").write_text("1")
        src2 = tmp_path / "p2"
        src2.mkdir()
        (src2 / "two.txt").write_text("2")
        results = LocalJobResults(root=root)
        results.save("artifacts", src1)
        results.save("artifacts", src2)
        dst = root / "artifacts"
        assert (dst / "two.txt").exists()
        assert not (dst / "one.txt").exists()


class TestSaveOverwriteAcrossKinds:
    """``save(name, ...)`` overwrites by name regardless of artifact kind."""

    def test_directory_then_file(self, tmp_path: Path, root: Path) -> None:
        src_dir = tmp_path / "payload"
        src_dir.mkdir()
        (src_dir / "one.txt").write_text("1")
        src_file = tmp_path / "replacement.txt"
        src_file.write_text("replacement")
        results = LocalJobResults(root=root)
        results.save("artifact", src_dir)
        results.save("artifact", src_file)
        dst = root / "artifact"
        assert dst.is_file()
        assert dst.read_text() == "replacement"

    def test_file_then_directory(self, tmp_path: Path, root: Path) -> None:
        src_file = tmp_path / "first.txt"
        src_file.write_text("first")
        src_dir = tmp_path / "payload"
        src_dir.mkdir()
        (src_dir / "two.txt").write_text("2")
        results = LocalJobResults(root=root)
        results.save("artifact", src_file)
        results.save("artifact", src_dir)
        dst = root / "artifact"
        assert dst.is_dir()
        assert (dst / "two.txt").read_text() == "2"


def _platform_record(name: str = "metrics", url: str = "fileset://ws/fs#results/A1/metrics"):
    record = MagicMock()
    record.name = name  # MagicMock special-cases ``name``
    record.artifact_url = url
    return record


def _task_client() -> tuple[NemoClient, httpx.Client]:
    http_client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, request=request)))
    client = NemoClient(base_url="http://platform.test", workspace="ws", http_client=http_client)
    return client, http_client


def test_result_manager_factory_explicit_workspace_overrides_client_workspace() -> None:
    files_client = FilesClient(base_url="http://platform.test", workspace="client-ws")

    manager = result_manager_factory(
        job_name="j",
        workspace="explicit-ws",
        files_client=files_client,
        jobs_client=JobsClient.from_client(files_client),
    )

    assert isinstance(manager, ResultManager)
    assert manager.workspace == "explicit-ws"


def test_result_manager_factory_ignores_job_env_and_uses_client_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(NEMO_JOB_WORKSPACE_ENVVAR, "env-ws")
    files_client = FilesClient(base_url="http://platform.test", workspace="client-ws")

    manager = result_manager_factory(
        job_name="j",
        files_client=files_client,
        jobs_client=JobsClient.from_client(files_client),
    )

    assert manager.workspace == "client-ws"


@pytest.mark.asyncio
async def test_async_result_manager_factory_ignores_job_env_and_uses_client_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(NEMO_JOB_WORKSPACE_ENVVAR, "env-ws")
    files_client = AsyncFilesClient(base_url="http://platform.test", workspace="client-ws")

    manager = async_result_manager_factory(
        job_name="j",
        files_client=files_client,
        jobs_client=AsyncJobsClient.from_client(files_client),
    )

    assert isinstance(manager, AsyncResultManager)
    assert manager.workspace == "client-ws"


class TestPlatformJobResults:
    def test_platform_job_results_is_a_job_results(self) -> None:
        client = NemoClient(base_url="http://platform.test", workspace="ws")
        with patch("nemo_platform_plugin.job_results.result_manager_factory") as factory:
            factory.return_value = MagicMock()
            sink = PlatformJobResults(job_name="j", workspace="ws", client=client)
        assert isinstance(sink, JobResults)

    def test_platform_job_results_uses_typed_client_transport(self) -> None:
        http_client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, request=request)))
        client = NemoClient(base_url="http://platform.test", workspace="ws", http_client=http_client)

        with patch("nemo_platform_plugin.job_results.result_manager_factory") as factory:
            factory.return_value = MagicMock()
            PlatformJobResults(job_name="j", workspace="ws", client=client)

        files_client = factory.call_args.kwargs["files_client"]
        jobs_client = factory.call_args.kwargs["jobs_client"]
        assert isinstance(files_client, FilesClient)
        assert isinstance(jobs_client, JobsClient)
        assert files_client._http is http_client
        assert jobs_client._http is http_client

    def test_save_delegates_to_result_manager(self, tmp_path: Path) -> None:
        client, http_client = _task_client()
        local = tmp_path / "out.json"
        local.write_text("{}")
        manager = MagicMock()
        manager.create_result.return_value = _platform_record(
            name="metrics",
            url="fileset://ws/fs#results/A1/metrics",
        )
        with patch("nemo_platform_plugin.job_results.result_manager_factory", return_value=manager) as factory:
            sink = PlatformJobResults(job_name="j", workspace="ws", client=client, attempt_id="A1")
            ref = sink.save("metrics", local, ignore_patterns=["cache.db"])

        factory.assert_called_once()
        call_kwargs = factory.call_args.kwargs
        assert call_kwargs["job_name"] == "j"
        assert call_kwargs["workspace"] == "ws"
        assert call_kwargs["attempt_id"] == "A1"
        assert isinstance(call_kwargs["files_client"], FilesClient)
        assert isinstance(call_kwargs["jobs_client"], JobsClient)
        assert call_kwargs["files_client"]._http is http_client
        assert call_kwargs["jobs_client"]._http is http_client
        manager.create_result.assert_called_once_with(
            result_name="metrics",
            artifact_local_path=local,
            ignore_patterns=["cache.db"],
        )
        assert ref == ResultRef(name="metrics", artifact_url="fileset://ws/fs#results/A1/metrics")

    def test_save_forwards_directory_path_unchanged(self, tmp_path: Path) -> None:
        client, _http_client = _task_client()
        payload_dir = tmp_path / "payload"
        payload_dir.mkdir()
        manager = MagicMock()
        manager.create_result.return_value = _platform_record()
        with patch("nemo_platform_plugin.job_results.result_manager_factory", return_value=manager):
            sink = PlatformJobResults(job_name="j", workspace="ws", client=client)
            sink.save("artifacts", payload_dir, ignore_patterns=["cache.db", "cache/"])
        call = manager.create_result.call_args
        assert call.kwargs["artifact_local_path"] == payload_dir
        assert call.kwargs["ignore_patterns"] == ["cache.db", "cache/"]

    def test_save_propagates_manager_errors(self) -> None:
        client, _http_client = _task_client()
        manager = MagicMock()
        manager.create_result.side_effect = RuntimeError("boom")
        with patch("nemo_platform_plugin.job_results.result_manager_factory", return_value=manager):
            sink = PlatformJobResults(job_name="j", workspace="ws", client=client)
            with pytest.raises(RuntimeError, match="boom"):
                sink.save("metrics", Path("/tmp/whatever"))


def test_result_manager_fetches_metadata_from_typed_task_client() -> None:
    """Task result publishing receives ``NemoClient`` handles, not generated SDKs."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/apis/jobs/v2/workspaces/test-ws/jobs/test-job"
        return httpx.Response(
            200,
            json={
                "id": "job-1",
                "attempt_id": "att-123",
                "name": "test-job",
                "workspace": "test-ws",
                "source": "unit",
                "platform_spec": {
                    "steps": [
                        {
                            "name": "step-1",
                            "executor": {"provider": "cpu", "container": {}},
                        }
                    ]
                },
                "fileset": "test-fileset",
                "output_location": "shared-fs",
                "status": "created",
            },
            request=request,
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = NemoClient(base_url="http://platform.test", workspace="test-ws", http_client=http_client)
    manager = ResultManager(
        job_name="test-job",
        workspace="test-ws",
        files_client=FilesClient.from_client(client),
        jobs_client=JobsClient.from_client(client),
    )

    assert manager._fetch_job_metadata() == ("att-123", "test-fileset", "shared-fs")
