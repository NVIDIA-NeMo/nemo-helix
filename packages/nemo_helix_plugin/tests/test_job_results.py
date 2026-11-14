# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :mod:`nemo_helix_plugin.job_results`.

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
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.files.client import AsyncFilesClient, FilesClient
from nemo_helix_plugin.job_results import (
    HelixJobResults,
    JobResults,
    LocalJobResults,
    ResultRef,
)
from nemo_helix_plugin.jobs.client import AsyncJobsClient, JobsClient
from nemo_helix_plugin.jobs.constants import NEMO_JOB_WORKSPACE_ENVVAR
from nemo_helix_plugin.jobs.file_manager import _filter_files_by_patterns, _stage_local_files
from nemo_helix_plugin.jobs.result_manager import (
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
        # A trailing slash selects directories rather than regular files.
        results.save("artifacts", src, ignore_patterns=["cache.db", "cache/"])
        dst = root / "artifacts"
        assert (dst / "keep.txt").read_text() == "K"
        assert not (dst / "cache").exists()

    def test_save_directory_anchors_directory_path_patterns(self, tmp_path: Path, root: Path) -> None:
        """Exclude a root-relative directory while retaining deeper copies and same-prefix files."""
        src = tmp_path / "payload"
        for relative in (
            "home/.tmp/plugins/catalog.json",
            "home/.tmp/plugins.sha",
            "workspace/home/.tmp/plugins/kept.json",
        ):
            (src / relative).parent.mkdir(parents=True, exist_ok=True)
            (src / relative).write_text(relative)
        results = LocalJobResults(root=root)
        # Same anchoring as the platform impl: the directory path starts at the saved root.
        results.save("artifacts", src, ignore_patterns=["home/.tmp/plugins/"])
        dst = root / "artifacts"
        assert not (dst / "home" / ".tmp" / "plugins").exists()
        assert (dst / "home" / ".tmp" / "plugins.sha").exists()
        assert (dst / "workspace" / "home" / ".tmp" / "plugins" / "kept.json").exists()

    def test_save_directory_matches_globbed_directory_paths(self, tmp_path: Path, root: Path) -> None:
        """Match globs inside a root-relative directory path without filtering adjacent files."""
        src = tmp_path / "payload"
        for relative in ("home/.tmp/plugins-clone-a1/catalog.json", "home/.tmp/plugins.sha"):
            (src / relative).parent.mkdir(parents=True, exist_ok=True)
            (src / relative).write_text(relative)
        results = LocalJobResults(root=root)
        results.save("artifacts", src, ignore_patterns=["home/.tmp/plugins*/"])
        dst = root / "artifacts"
        assert not (dst / "home" / ".tmp" / "plugins-clone-a1").exists()
        assert (dst / "home" / ".tmp" / "plugins.sha").exists()

    def test_save_directory_matches_single_folder_globs(self, tmp_path: Path, root: Path) -> None:
        """Folder globs exclude matching directories at any depth and keep same-prefix files."""
        src = tmp_path / "payload"
        for relative in (
            "plugins/catalog.json",
            "plugins-clone-a1/nested/skill.md",
            "nested/plugins/catalog.json",
            "nested/plugins-clone-b2/catalog.json",
            "plugins.sha",
            "nested/plugins.sha",
            "other/keep.txt",
        ):
            (src / relative).parent.mkdir(parents=True, exist_ok=True)
            (src / relative).write_text(relative)

        LocalJobResults(root=root).save("artifacts", src, ignore_patterns=["plugins*/"])

        dst = root / "artifacts"
        assert sorted(path.relative_to(dst).as_posix() for path in dst.rglob("*") if path.is_file()) == [
            "nested/plugins.sha",
            "other/keep.txt",
            "plugins.sha",
        ]

    def test_save_directory_keeps_filename_glob_behavior(self, tmp_path: Path, root: Path) -> None:
        """Filename globs continue excluding matching files at every depth."""
        src = tmp_path / "payload"
        for relative in ("cache.tmp", "nested/cache.tmp", "keep.txt"):
            (src / relative).parent.mkdir(parents=True, exist_ok=True)
            (src / relative).write_text(relative)

        LocalJobResults(root=root).save("artifacts", src, ignore_patterns=["*.tmp"])

        dst = root / "artifacts"
        assert (dst / "keep.txt").is_file()
        assert not (dst / "cache.tmp").exists()
        assert not (dst / "nested/cache.tmp").exists()

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

    @pytest.mark.parametrize(
        ("patterns", "excluded"),
        [
            (None, set()),
            ([], set()),
            ("*.tmp", {"cache.tmp", "nested/cache.tmp"}),
            (
                ["plugins*/"],
                {
                    "plugins/catalog.json",
                    "nested/plugins-clone/catalog.json",
                    "home/.tmp/plugins/catalog.json",
                    "deep/home/.tmp/plugins/catalog.json",
                },
            ),
            (["home/.tmp/plugins*/"], {"home/.tmp/plugins/catalog.json"}),
            (["/home/.tmp/plugins*/"], {"home/.tmp/plugins/catalog.json"}),
            (["/plugins/"], {"plugins/catalog.json"}),
            (["/evidence/"], {"evidence/trace.json"}),
            (["nested/*.tmp"], {"nested/cache.tmp"}),
            (
                ["catalog.json"],
                {
                    "plugins/catalog.json",
                    "nested/plugins-clone/catalog.json",
                    "home/.tmp/plugins/catalog.json",
                    "deep/home/.tmp/plugins/catalog.json",
                },
            ),
            (["[Cc]ache.tmp"], {"cache.tmp", "nested/cache.tmp"}),
            ([".hidden"], {".hidden", "nested/.hidden"}),
        ],
    )
    def test_local_and_hosted_saves_keep_the_same_files(self, tmp_path: Path, root: Path, patterns, excluded) -> None:
        """Cover file-only globs, directory anchoring, and path globs in both save paths."""
        files = {
            "cache.tmp",
            "nested/cache.tmp",
            "logs.tmp/keep.txt",
            "nested/logs.tmp/keep.txt",
            "plugins/catalog.json",
            "nested/plugins-clone/catalog.json",
            "plugins.sha",
            "evidence/trace.json",
            "nested/evidence/trace.json",
            "home/.tmp/plugins/catalog.json",
            "deep/home/.tmp/plugins/catalog.json",
            ".hidden",
            "nested/.hidden",
        }
        src = tmp_path / "payload"
        for relative in files:
            (src / relative).parent.mkdir(parents=True, exist_ok=True)
            (src / relative).write_text(relative)

        LocalJobResults(root=root).save("artifacts", src, ignore_patterns=patterns)

        dst = root / "artifacts"
        saved = {path.relative_to(dst).as_posix() for path in dst.rglob("*") if path.is_file()}
        assert saved == files - excluded
        assert set(_filter_files_by_patterns(sorted(files), patterns)) == saved
        staging_root = tmp_path / "staging"
        staging_root.mkdir()
        snapshot, selected = _stage_local_files(src, staging_root, patterns)
        assert set(selected) == saved
        assert {path.relative_to(snapshot).as_posix() for path in snapshot.rglob("*") if path.is_file()} == saved


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
        attempt_id="attempt-1",
        workspace="explicit-ws",
        files_client=files_client,
        fileset_name="job-fileset",
        jobs_client=JobsClient.from_client(files_client),
    )

    assert isinstance(manager, ResultManager)
    assert manager.workspace == "explicit-ws"


def test_result_manager_factory_ignores_job_env_and_uses_client_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(NEMO_JOB_WORKSPACE_ENVVAR, "env-ws")
    files_client = FilesClient(base_url="http://platform.test", workspace="client-ws")

    manager = result_manager_factory(
        job_name="j",
        attempt_id="attempt-1",
        files_client=files_client,
        fileset_name="job-fileset",
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
        attempt_id="attempt-1",
        files_client=files_client,
        fileset_name="job-fileset",
        jobs_client=AsyncJobsClient.from_client(files_client),
    )

    assert isinstance(manager, AsyncResultManager)
    assert manager.workspace == "client-ws"


class TestHelixJobResults:
    def test_platform_job_results_is_a_job_results(self) -> None:
        client = NemoClient(base_url="http://platform.test", workspace="ws")
        with patch("nemo_helix_plugin.job_results.result_manager_factory") as factory:
            factory.return_value = MagicMock()
            sink = HelixJobResults(
                job_name="j", workspace="ws", client=client, attempt_id="attempt-1", fileset_name="job-fileset"
            )
        assert isinstance(sink, JobResults)

    def test_platform_job_results_uses_typed_client_transport(self) -> None:
        http_client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, request=request)))
        client = NemoClient(base_url="http://platform.test", workspace="ws", http_client=http_client)

        with patch("nemo_helix_plugin.job_results.result_manager_factory") as factory:
            factory.return_value = MagicMock()
            HelixJobResults(
                job_name="j", workspace="ws", client=client, attempt_id="attempt-1", fileset_name="job-fileset"
            )

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
        with patch("nemo_helix_plugin.job_results.result_manager_factory", return_value=manager) as factory:
            sink = HelixJobResults(
                job_name="j", workspace="ws", client=client, attempt_id="A1", fileset_name="job-fileset"
            )
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
            stage=False,
        )
        assert ref == ResultRef(name="metrics", artifact_url="fileset://ws/fs#results/A1/metrics")

    def test_save_forwards_directory_path_unchanged(self, tmp_path: Path) -> None:
        client, _http_client = _task_client()
        payload_dir = tmp_path / "payload"
        payload_dir.mkdir()
        manager = MagicMock()
        manager.create_result.return_value = _platform_record()
        with patch("nemo_helix_plugin.job_results.result_manager_factory", return_value=manager):
            sink = HelixJobResults(
                job_name="j", workspace="ws", client=client, attempt_id="attempt-1", fileset_name="job-fileset"
            )
            sink.save("artifacts", payload_dir, ignore_patterns=["cache.db", "cache/"], stage=True)
        call = manager.create_result.call_args
        assert call.kwargs["artifact_local_path"] == payload_dir
        assert call.kwargs["stage"] is True
        assert call.kwargs["ignore_patterns"] == ["cache.db", "cache/"]

    def test_save_propagates_manager_errors(self) -> None:
        client, _http_client = _task_client()
        manager = MagicMock()
        manager.create_result.side_effect = RuntimeError("boom")
        with patch("nemo_helix_plugin.job_results.result_manager_factory", return_value=manager):
            sink = HelixJobResults(
                job_name="j", workspace="ws", client=client, attempt_id="attempt-1", fileset_name="job-fileset"
            )
            with pytest.raises(RuntimeError, match="boom"):
                sink.save("metrics", Path("/tmp/whatever"))


def test_result_manager_forwards_staging_to_the_file_manager(tmp_path: Path) -> None:
    artifact = tmp_path / "bundle"
    artifact.mkdir()
    jobs_client = MagicMock()
    jobs_client.create_job_result.return_value.data.return_value = _platform_record()
    file_manager = MagicMock()
    file_manager.upload.return_value = "fileset://ws/results#results/attempt-a/bundle"
    file_manager.storage_type.return_value = "fileset"
    manager = ResultManager(
        job_name="job-a",
        workspace="ws",
        attempt_id="attempt-a",
        files_client=MagicMock(),
        jobs_client=jobs_client,
        fileset_name="results",
    )

    with patch.object(manager, "_create_file_manager", return_value=file_manager):
        manager.create_result("bundle", artifact, stage=True)

    assert file_manager.upload.call_args.kwargs["stage"] is True


def test_result_manager_uses_explicit_runtime_attempt_for_path_and_registration(tmp_path: Path) -> None:
    artifact = tmp_path / "metrics.json"
    artifact.write_text("{}")
    jobs_client = MagicMock()
    jobs_client.create_job_result.return_value.data.return_value = _platform_record(
        url="fileset://ws/results#results/attempt-a/metrics"
    )
    file_manager = MagicMock()
    file_manager.upload.return_value = "fileset://ws/results#results/attempt-a/metrics"
    file_manager.storage_type.return_value = "fileset"
    manager = ResultManager(
        job_name="job-a",
        workspace="ws",
        attempt_id="attempt-a",
        fileset_name="results",
        files_client=MagicMock(),
        jobs_client=jobs_client,
    )

    with patch.object(manager, "_create_file_manager", return_value=file_manager):
        manager.create_result("metrics", artifact)

    assert file_manager.upload.call_args.kwargs["remote_path"] == "results/attempt-a/metrics"
    assert file_manager.upload.call_args.kwargs["stage"] is False
    jobs_client.create_job_result.assert_called_once()
    call = jobs_client.create_job_result.call_args
    assert call.kwargs["body"].attempt_id == "attempt-a"


class TestRestrictedPolicyResultSave:
    """A provider-bound sandbox saves results without reading the job record.

    OpenShell sandboxes with gateway provider bindings get the restricted platform
    policy: they may post results, but ``GET /jobs/{job}`` is denied. ``ctx.results.save``
    must therefore work from the ``NEMO_JOB_*`` env alone.
    """

    def test_save_never_reads_the_job_record(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from nemo_helix_plugin.tasks import dispatcher as dispatcher_module

        artifact = tmp_path / "summary.json"
        artifact.write_text("{}")
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            path = request.url.path
            if request.method == "GET" and path.endswith("/filesets/job-fileset/files"):
                return httpx.Response(200, json={"data": []}, request=request)
            if request.method == "PUT" and "/filesets/job-fileset/-/" in path:
                return httpx.Response(
                    200, json={"file_ref": "f", "file_url": "u", "path": "p", "size": 2}, request=request
                )
            if request.method == "POST" and path.endswith("/jobs/job-a/results/summary"):
                return httpx.Response(
                    200,
                    json={
                        "name": "summary",
                        "job": "job-a",
                        "workspace": "ws",
                        "attempt_id": "attempt-a",
                        "created_at": "2026-01-01T00:00:00Z",
                        "updated_at": "2026-01-01T00:00:00Z",
                        "artifact_url": "ws/job-fileset#jobs/job-a/results/attempt-a/summary",
                        "artifact_storage_type": "fileset",
                    },
                    request=request,
                )
            return httpx.Response(404, json={"detail": f"{request.method} {path} is not permitted"}, request=request)

        for name, value in {
            "NEMO_JOB_WORKSPACE": "ws",
            "NEMO_JOB_ID": "job-a",
            "NEMO_JOB_ATTEMPT_ID": "attempt-a",
            "NEMO_JOB_EPHEMERAL_TASK_STORAGE_PATH": str(tmp_path / "e"),
            "NEMO_JOB_FILESET": "job-fileset",
            "NEMO_JOB_ARTIFACT_BASE_PATH": "jobs/job-a",
        }.items():
            monkeypatch.setenv(name, value)

        client = NemoClient(
            base_url="http://platform.test",
            workspace="ws",
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
            owns_http_client=True,
        )
        try:
            ref = dispatcher_module.build_ctx_from_env(client).results.save("summary", artifact)
        finally:
            client.close()

        assert [(r.method, r.url.path) for r in requests] == [
            ("GET", "/apis/files/v2/workspaces/ws/filesets/job-fileset/files"),
            ("PUT", "/apis/files/v2/workspaces/ws/filesets/job-fileset/-/jobs/job-a/results/attempt-a/summary"),
            ("POST", "/apis/jobs/v2/workspaces/ws/jobs/job-a/results/summary"),
        ]
        assert ref.artifact_url == "ws/job-fileset#jobs/job-a/results/attempt-a/summary"
