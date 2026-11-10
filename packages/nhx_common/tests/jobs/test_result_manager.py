# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :mod:`nemo_helix_plugin.jobs.result_manager`.

These drive the manager against a fake Files + Jobs platform over an httpx transport,
so the assertions cover what actually goes over the wire: which routes are called, in
what order, the path an artifact lands at, and the result registration body. The fake
answers only the routes a result save is allowed to use and 404s everything else, so a
stray job-record read fails the test instead of passing silently.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.client.errors import PermissionDeniedError
from nemo_helix_plugin.files.client import AsyncFilesClient, FilesClient
from nemo_helix_plugin.jobs import result_manager as rm
from nemo_helix_plugin.jobs.client import AsyncJobsClient, JobsClient

JOB = "job-a"
WORKSPACE = "ws"
FILESET = "job-fileset"
ATTEMPT = "attempt-1"
BASE_PATH = "jobs/job-a"

FILES_ROOT = f"/apis/files/v2/workspaces/{WORKSPACE}/filesets/{FILESET}"
JOBS_ROOT = f"/apis/jobs/v2/workspaces/{WORKSPACE}/jobs/{JOB}"


def _result_body(name: str, artifact_url: str) -> dict[str, object]:
    return {
        "name": name,
        "job": JOB,
        "workspace": WORKSPACE,
        "attempt_id": ATTEMPT,
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        "artifact_url": artifact_url,
        "artifact_storage_type": "fileset",
    }


class FakePlatform:
    """The Files and Jobs routes a result save may use. Anything else is a 404.

    Recording every request lets a test assert the exact call sequence, which is how
    "the sink never reads the job record" is checked.
    """

    def __init__(self, *, result_exists: bool = False, transport_error: bool = False) -> None:
        self.requests: list[httpx.Request] = []
        self.artifacts: dict[str, bytes] = {}
        self.result_exists = result_exists
        self.transport_error = transport_error

    def calls(self) -> list[tuple[str, str]]:
        return [(request.method, request.url.path) for request in self.requests]

    def result_requests(self) -> list[httpx.Request]:
        return [request for request in self.requests if "/results/" in request.url.path]

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
                json={
                    "file_ref": f"{WORKSPACE}/{FILESET}#{remote}",
                    "file_url": f"http://platform/files/{remote}",
                    "path": remote,
                    "size": len(request.content),
                },
                request=request,
            )
        if path.startswith(f"{JOBS_ROOT}/results/"):
            name = path.rsplit("/", 1)[1]
            artifact_url = f"{WORKSPACE}/{FILESET}#{BASE_PATH}/results/{ATTEMPT}/{name}"
            if method == "POST":
                if self.transport_error:
                    raise httpx.ConnectError("connection refused", request=request)
                if self.result_exists:
                    return httpx.Response(409, json={"detail": "exists"}, request=request)
                return httpx.Response(200, json=_result_body(name, artifact_url), request=request)
            if method == "GET":
                return httpx.Response(200, json=_result_body(name, artifact_url), request=request)
        return httpx.Response(
            404,
            json={"detail": f"{method} {path} is not a permitted platform request"},
            request=request,
        )


def _sync_client(platform: FakePlatform) -> NemoClient:
    return NemoClient(
        base_url="http://platform",
        workspace=WORKSPACE,
        http_client=httpx.Client(transport=httpx.MockTransport(platform.handler)),
    )


def _async_client(platform: FakePlatform) -> AsyncNemoClient:
    return AsyncNemoClient(
        base_url="http://platform",
        workspace=WORKSPACE,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(platform.handler)),
    )


def _manager(platform: FakePlatform, *, artifact_base_path: str | None = BASE_PATH) -> rm.ResultManager:
    client = _sync_client(platform)
    return rm.ResultManager(
        job_name=JOB,
        workspace=WORKSPACE,
        attempt_id=ATTEMPT,
        fileset=FILESET,
        artifact_base_path=artifact_base_path,
        files_client=FilesClient.from_client(client),
        jobs_client=JobsClient.from_client(client),
    )


def _async_manager(platform: FakePlatform, *, artifact_base_path: str | None = BASE_PATH) -> rm.AsyncResultManager:
    client = _async_client(platform)
    return rm.AsyncResultManager(
        job_name=JOB,
        workspace=WORKSPACE,
        attempt_id=ATTEMPT,
        fileset=FILESET,
        artifact_base_path=artifact_base_path,
        files_client=AsyncFilesClient.from_client(client),
        jobs_client=AsyncJobsClient.from_client(client),
    )


def _artifact(tmp_path: Path, name: str = "summary.json", content: bytes = b'{"ok": true}') -> Path:
    artifact = tmp_path / name
    artifact.write_bytes(content)
    return artifact


def test_result_remote_path_nests_under_base():
    """_result_remote_path prepends the per-job base only when one is supplied."""
    manager = _manager(FakePlatform())
    assert manager._result_remote_path(ATTEMPT, "summary") == "results/attempt-1/summary"
    assert manager._result_remote_path(ATTEMPT, "summary", base="jobs/job-a") == (
        "jobs/job-a/results/attempt-1/summary"
    )


@pytest.mark.parametrize(
    "artifact_base_path,expected_remote",
    [(BASE_PATH, f"{BASE_PATH}/results/{ATTEMPT}/summary"), (None, f"results/{ATTEMPT}/summary")],
)
def test_create_result_uploads_then_registers(tmp_path, artifact_base_path, expected_remote):
    """A save lists the fileset, uploads the artifact, then registers the result."""
    platform = FakePlatform()
    artifact = _artifact(tmp_path)

    manager = _manager(platform, artifact_base_path=artifact_base_path)
    result = manager.create_result("summary", artifact)

    assert platform.calls() == [
        ("GET", f"{FILES_ROOT}/files"),
        ("PUT", f"{FILES_ROOT}/-/{expected_remote}"),
        ("POST", f"{JOBS_ROOT}/results/summary"),
    ]
    assert platform.artifacts[expected_remote] == b'{"ok": true}'
    assert result.name == "summary"
    post = platform.result_requests()[-1]
    assert post.read() and b'"attempt_id":"attempt-1"' in post.content
    assert f'"artifact_url":"{WORKSPACE}/{FILESET}#{expected_remote}"'.encode() in post.content


def test_create_result_never_reads_the_job_record(tmp_path):
    """The restricted sandbox policy allows results but not GET /jobs/{job}."""
    platform = FakePlatform()
    _manager(platform).create_result("summary", _artifact(tmp_path))

    assert ("GET", JOBS_ROOT) not in platform.calls()
    assert all(path.startswith((FILES_ROOT, f"{JOBS_ROOT}/results/")) for _, path in platform.calls())


def test_create_result_returns_existing_result_after_a_conflict(tmp_path):
    """Saving a result name twice reads the existing record instead of failing."""
    platform = FakePlatform(result_exists=True)

    result = _manager(platform).create_result("summary", _artifact(tmp_path))

    assert platform.calls() == [
        ("GET", f"{FILES_ROOT}/files"),
        ("PUT", f"{FILES_ROOT}/-/{BASE_PATH}/results/{ATTEMPT}/summary"),
        ("POST", f"{JOBS_ROOT}/results/summary"),
        ("GET", f"{JOBS_ROOT}/results/summary"),
    ]
    assert result.name == "summary"
    assert ("GET", JOBS_ROOT) not in platform.calls()


def test_create_result_wraps_transport_errors(tmp_path):
    """A failed registration surfaces as CreateJobResultError, not a raw transport error."""
    platform = FakePlatform(transport_error=True)

    with pytest.raises(rm.CreateJobResultError, match="Error creating job result"):
        _manager(platform).create_result("summary", _artifact(tmp_path))


def test_create_result_rejects_a_missing_artifact_before_uploading(tmp_path):
    """A save of a path that does not exist never reaches the artifact route."""
    platform = FakePlatform()

    with pytest.raises(rm.FileDoesNotExist):
        _manager(platform).create_result("summary", tmp_path / "absent.json")

    assert platform.calls() == [("GET", f"{FILES_ROOT}/files")]


def test_create_result_surfaces_a_fileset_it_cannot_create(tmp_path):
    """A save into a missing fileset fails: the sandbox may not create one.

    ``validate_storage`` creates the fileset when it is absent, and the restricted
    platform policy does not allow that route. The failure propagates rather than
    turning into a result record with no artifact.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        status = 403 if request.method == "POST" else 404
        return httpx.Response(status, json={"detail": "not permitted by policy"}, request=request)

    client = NemoClient(
        base_url="http://platform",
        workspace=WORKSPACE,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    manager = rm.ResultManager(
        job_name=JOB,
        workspace=WORKSPACE,
        attempt_id=ATTEMPT,
        fileset=FILESET,
        files_client=FilesClient.from_client(client),
        jobs_client=JobsClient.from_client(client),
    )

    with pytest.raises(PermissionDeniedError):
        manager.create_result("summary", _artifact(tmp_path))


@pytest.mark.asyncio
async def test_async_create_result_uploads_then_registers(tmp_path):
    """The async manager performs the same upload and registration sequence."""
    platform = FakePlatform()
    artifact = _artifact(tmp_path)

    result = await _async_manager(platform).create_result("summary", artifact)

    assert platform.calls() == [
        ("GET", f"{FILES_ROOT}/files"),
        ("PUT", f"{FILES_ROOT}/-/{BASE_PATH}/results/{ATTEMPT}/summary"),
        ("POST", f"{JOBS_ROOT}/results/summary"),
    ]
    assert platform.artifacts[f"{BASE_PATH}/results/{ATTEMPT}/summary"] == b'{"ok": true}'
    assert result.name == "summary"
    assert ("GET", JOBS_ROOT) not in platform.calls()


@pytest.mark.asyncio
async def test_async_create_result_returns_existing_result_after_a_conflict(tmp_path):
    """The async conflict path reads the existing record rather than the job."""
    platform = FakePlatform(result_exists=True)

    result = await _async_manager(platform).create_result("summary", _artifact(tmp_path))

    assert platform.calls()[-2:] == [
        ("POST", f"{JOBS_ROOT}/results/summary"),
        ("GET", f"{JOBS_ROOT}/results/summary"),
    ]
    assert result.name == "summary"
    assert ("GET", JOBS_ROOT) not in platform.calls()


@pytest.mark.asyncio
async def test_async_create_result_wraps_transport_errors(tmp_path):
    platform = FakePlatform(transport_error=True)

    with pytest.raises(rm.CreateJobResultError, match="Error creating job result"):
        await _async_manager(platform).create_result("summary", _artifact(tmp_path))
