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

The sink tests drive real uploads through a recording transport that answers only the
Files and Jobs routes a result save may use, so a stray job-record read fails the test.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.files.client import AsyncFilesClient, FilesClient
from nemo_helix_plugin.job_results import (
    HelixJobResults,
    JobResults,
    LocalJobResults,
    ResultRef,
    job_storage_context_from_env,
)
from nemo_helix_plugin.jobs import result_manager as rm
from nemo_helix_plugin.jobs.client import AsyncJobsClient, JobsClient
from nemo_helix_plugin.jobs.constants import NEMO_JOB_WORKSPACE_ENVVAR
from nemo_helix_plugin.jobs.file_manager import _filter_files_by_patterns
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


WORKSPACE = "ws"
FILESET = "job-fileset"
JOB = "job-a"
ATTEMPT = "attempt-a"
BASE_PATH = "jobs/job-a"

FILES_ROOT = f"/apis/files/v2/workspaces/{WORKSPACE}/filesets/{FILESET}"
JOBS_ROOT = f"/apis/jobs/v2/workspaces/{WORKSPACE}/jobs/{JOB}"


class FakePlatform:
    """Records every request; answers only the routes a result save is allowed to use."""

    def __init__(self, *, result_exists: bool = False) -> None:
        self.requests: list[httpx.Request] = []
        self.artifacts: dict[str, bytes] = {}
        self.result_exists = result_exists

    def calls(self) -> list[tuple[str, str]]:
        return [(request.method, request.url.path) for request in self.requests]

    def uploads(self) -> list[str]:
        return [path.split("/-/", 1)[1] for method, path in self.calls() if method == "PUT"]

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        method, path = request.method, request.url.path
        if method == "GET" and path == f"{FILES_ROOT}/files":
            return httpx.Response(200, json={"data": []}, request=request)
        if method == "PUT" and path.startswith(f"{FILES_ROOT}/-/"):
            remote = path.split("/-/", 1)[1]
            self.artifacts[remote] = request.content
            return httpx.Response(
                200,
                json={"file_ref": remote, "file_url": "http://platform/f", "path": remote, "size": 1},
                request=request,
            )
        if path.startswith(f"{JOBS_ROOT}/results/"):
            name = path.rsplit("/", 1)[1]
            body = {
                "name": name,
                "job": JOB,
                "workspace": WORKSPACE,
                "attempt_id": ATTEMPT,
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
                "artifact_url": f"{WORKSPACE}/{FILESET}#{BASE_PATH}/results/{ATTEMPT}/{name}",
                "artifact_storage_type": "fileset",
            }
            if method == "POST":
                if self.result_exists:
                    return httpx.Response(409, json={"detail": "exists"}, request=request)
                return httpx.Response(200, json=body, request=request)
            if method == "GET":
                return httpx.Response(200, json=body, request=request)
        return httpx.Response(404, json={"detail": f"{method} {path} is not permitted"}, request=request)


def _client(platform: FakePlatform) -> NemoClient:
    return NemoClient(
        base_url="http://platform.test",
        workspace=WORKSPACE,
        http_client=httpx.Client(transport=httpx.MockTransport(platform.handler)),
    )


def _artifact(tmp_path: Path, name: str = "summary.json") -> Path:
    artifact = tmp_path / name
    artifact.write_text("{}")
    return artifact


def test_result_manager_factory_explicit_workspace_overrides_client_workspace() -> None:
    files_client = FilesClient(base_url="http://platform.test", workspace="client-ws")

    manager = result_manager_factory(
        job_name="j",
        attempt_id="attempt-1",
        workspace="explicit-ws",
        files_client=files_client,
        fileset=FILESET,
        jobs_client=JobsClient.from_client(files_client),
    )

    assert isinstance(manager, ResultManager)
    assert manager.workspace == "explicit-ws"
    assert manager.fileset == FILESET


def test_result_manager_factory_ignores_job_env_and_uses_client_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(NEMO_JOB_WORKSPACE_ENVVAR, "env-ws")
    files_client = FilesClient(base_url="http://platform.test", workspace="client-ws")

    manager = result_manager_factory(
        job_name="j",
        attempt_id="attempt-1",
        files_client=files_client,
        fileset=FILESET,
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
        fileset=FILESET,
        jobs_client=AsyncJobsClient.from_client(files_client),
    )

    assert isinstance(manager, AsyncResultManager)
    assert manager.workspace == "client-ws"


class TestHelixJobResults:
    def test_platform_job_results_is_a_job_results(self) -> None:
        platform = FakePlatform()
        sink = HelixJobResults(
            job_name=JOB,
            workspace=WORKSPACE,
            client=_client(platform),
            attempt_id=ATTEMPT,
            fileset=FILESET,
            artifact_base_path=BASE_PATH,
        )
        assert isinstance(sink, JobResults)

    def test_save_uploads_the_artifact_and_registers_the_result(self, tmp_path: Path) -> None:
        platform = FakePlatform()
        sink = HelixJobResults(
            job_name=JOB,
            workspace=WORKSPACE,
            client=_client(platform),
            attempt_id=ATTEMPT,
            fileset=FILESET,
            artifact_base_path=BASE_PATH,
        )

        ref = sink.save("summary", _artifact(tmp_path))

        assert platform.calls() == [
            ("GET", f"{FILES_ROOT}/files"),
            ("PUT", f"{FILES_ROOT}/-/{BASE_PATH}/results/{ATTEMPT}/summary"),
            ("POST", f"{JOBS_ROOT}/results/summary"),
        ]
        assert platform.artifacts[f"{BASE_PATH}/results/{ATTEMPT}/summary"] == b"{}"
        assert ref.name == "summary"
        assert ref.artifact_url == f"{WORKSPACE}/{FILESET}#{BASE_PATH}/results/{ATTEMPT}/summary"

    def test_save_skips_ignored_files_in_a_directory(self, tmp_path: Path) -> None:
        platform = FakePlatform()
        payload = tmp_path / "payload"
        (payload / "cache").mkdir(parents=True)
        (payload / "keep.json").write_text("{}")
        (payload / "cache.db").write_text("noise")
        (payload / "cache" / "entry.json").write_text("{}")
        sink = HelixJobResults(
            job_name=JOB,
            workspace=WORKSPACE,
            client=_client(platform),
            attempt_id=ATTEMPT,
            fileset=FILESET,
            artifact_base_path=BASE_PATH,
        )

        sink.save("artifacts", payload, ignore_patterns=["cache.db", "cache/"])

        assert platform.uploads() == [f"{BASE_PATH}/results/{ATTEMPT}/artifacts/keep.json"]

    def test_save_returns_the_existing_result_when_the_name_is_taken(self, tmp_path: Path) -> None:
        platform = FakePlatform(result_exists=True)
        sink = HelixJobResults(
            job_name=JOB,
            workspace=WORKSPACE,
            client=_client(platform),
            attempt_id=ATTEMPT,
            fileset=FILESET,
            artifact_base_path=BASE_PATH,
        )

        ref = sink.save("summary", _artifact(tmp_path))

        assert platform.calls()[-2:] == [
            ("POST", f"{JOBS_ROOT}/results/summary"),
            ("GET", f"{JOBS_ROOT}/results/summary"),
        ]
        assert ref.name == "summary"

    def test_save_surfaces_a_missing_artifact(self, tmp_path: Path) -> None:
        platform = FakePlatform()
        sink = HelixJobResults(
            job_name=JOB,
            workspace=WORKSPACE,
            client=_client(platform),
            attempt_id=ATTEMPT,
            fileset=FILESET,
            artifact_base_path=BASE_PATH,
        )

        with pytest.raises(rm.FileDoesNotExist):
            sink.save("summary", tmp_path / "absent.json")

        assert platform.uploads() == []


class TestJobStorageContextFromEnv:
    """Task entrypoints that build their own context read the storage context from env."""

    def test_reads_fileset_and_base_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NEMO_JOB_FILESET", FILESET)
        monkeypatch.setenv("NEMO_JOB_ARTIFACT_BASE_PATH", BASE_PATH)

        assert job_storage_context_from_env() == (FILESET, BASE_PATH)

    def test_empty_base_path_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NEMO_JOB_FILESET", FILESET)
        monkeypatch.setenv("NEMO_JOB_ARTIFACT_BASE_PATH", "")

        assert job_storage_context_from_env() == (FILESET, None)

    def test_missing_fileset_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NEMO_JOB_FILESET", raising=False)

        with pytest.raises(RuntimeError, match="NEMO_JOB_FILESET"):
            job_storage_context_from_env()


def test_result_manager_registers_the_runtime_attempt(tmp_path: Path) -> None:
    """The attempt in the path and the registration body comes from the manager.

    It names the concrete attempt that produced the artifact, not the attempt the job
    record currently points at.
    """
    platform = FakePlatform()
    manager = ResultManager(
        job_name=JOB,
        workspace=WORKSPACE,
        attempt_id="attempt-a",
        fileset=FILESET,
        files_client=FilesClient.from_client(_client(platform)),
        jobs_client=JobsClient.from_client(_client(platform)),
    )

    manager.create_result("metrics", _artifact(tmp_path, "metrics.json"))

    assert platform.uploads() == ["results/attempt-a/metrics"]
    post = [request for request in platform.requests if request.method == "POST"][0]
    assert json.loads(post.content)["attempt_id"] == "attempt-a"


class TestRestrictedPolicyResultSave:
    """A provider-bound sandbox saves results without reading the job record.

    OpenShell sandboxes with gateway provider bindings get the restricted platform
    policy: they may post and read results, but ``GET /jobs/{job}`` is denied. The sink
    must work from the ``NEMO_JOB_*`` env alone.
    """

    def test_save_never_reads_the_job_record(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from nemo_helix_plugin.tasks import dispatcher as dispatcher_module

        platform = FakePlatform()
        monkeypatch.setenv("NEMO_JOB_WORKSPACE", WORKSPACE)
        monkeypatch.setenv("NEMO_JOB_ID", JOB)
        monkeypatch.setenv("NEMO_JOB_ATTEMPT_ID", ATTEMPT)
        monkeypatch.setenv("NEMO_JOB_EPHEMERAL_TASK_STORAGE_PATH", str(tmp_path / "e"))
        monkeypatch.setenv("NEMO_JOB_FILESET", FILESET)
        monkeypatch.setenv("NEMO_JOB_ARTIFACT_BASE_PATH", BASE_PATH)

        client = _client(platform)
        ctx = dispatcher_module.build_ctx_from_env(client)
        ref = ctx.results.save("summary", _artifact(tmp_path))

        assert ("GET", JOBS_ROOT) not in platform.calls()
        assert platform.calls() == [
            ("GET", f"{FILES_ROOT}/files"),
            ("PUT", f"{FILES_ROOT}/-/{BASE_PATH}/results/{ATTEMPT}/summary"),
            ("POST", f"{JOBS_ROOT}/results/summary"),
        ]
        assert ref.artifact_url == f"{WORKSPACE}/{FILESET}#{BASE_PATH}/results/{ATTEMPT}/summary"
