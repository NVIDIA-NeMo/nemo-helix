# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.client.errors import ConflictError as ClientConflictError
from nemo_helix_plugin.client.errors import NemoTransportError
from nemo_helix_plugin.files.client import AsyncFilesClient, FilesClient
from nemo_helix_plugin.jobs import result_manager as rm
from nemo_helix_plugin.jobs.client import AsyncJobsClient, JobsClient
from nemo_helix_plugin.jobs.result_manager import CreateJobResultError


def _resp(data):
    """Wrap a payload in a NemoResponse-like object whose ``.data()`` returns it."""
    m = MagicMock()
    m.data.return_value = data
    return m


def _conflict_error() -> ClientConflictError:
    """Build the ConflictError the typed client raises on a 409 response."""
    request = httpx.Request("POST", "http://test")
    return ClientConflictError(httpx.Response(status_code=409, json={}, request=request))


def _sync_client() -> NemoClient:
    http_client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, request=request)))
    return NemoClient(base_url="http://test", workspace="test-ws", http_client=http_client)


def _async_client() -> AsyncNemoClient:
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, request=request)))
    return AsyncNemoClient(base_url="http://test", workspace="test-ws", http_client=http_client)


def test_result_remote_path_nests_under_base():
    """_result_remote_path prepends the per-job base only when one is supplied."""
    sdk = _sync_client()
    mgr = rm.ResultManager(
        job_name="my-job",
        workspace="ws",
        files_client=FilesClient.from_client(sdk),
        jobs_client=JobsClient.from_client(sdk),
    )
    assert mgr._result_remote_path("att-1", "summary") == "results/att-1/summary"
    assert mgr._result_remote_path("att-1", "summary", base="jobs/my-job") == "jobs/my-job/results/att-1/summary"


def test_create_result_nests_under_job_when_output_location_set(tmp_path, mock_sync_file_manager):
    """When the job has an output_location, results nest under jobs/<job_name>/results/…; else flat."""
    test_file = tmp_path / "artifact.bin"
    test_file.write_bytes(b"x")

    mock_jobs = MagicMock(spec=JobsClient)
    mock_jobs.create_job_result.return_value = _resp(MagicMock())

    sdk = _sync_client()
    mgr = rm.ResultManager(
        job_name="test-job",
        workspace="test-ws",
        files_client=FilesClient.from_client(sdk),
        jobs_client=mock_jobs,
    )

    for output_location, expected in [
        ("shared-fs", "jobs/test-job/results/att-1/summary"),
        (None, "results/att-1/summary"),
    ]:
        mock_jobs.get_job.return_value = _resp(
            MagicMock(attempt_id="att-1", fileset="shared-fs", output_location=output_location)
        )
        with (
            patch.object(mgr, "_create_file_manager", return_value=mock_sync_file_manager),
        ):
            mgr.create_result("summary", test_file)
        assert mock_sync_file_manager.upload.call_args.kwargs["remote_path"] == expected


def test_create_result_returns_existing_on_conflict_sync(tmp_path, mock_sync_file_manager):
    """Test that sync create_result returns existing result when ConflictError is raised."""
    test_file = tmp_path / "artifact.bin"
    test_file.write_bytes(b"test content")

    # Configure the typed jobs client to raise ConflictError on create, return existing on retrieve.
    existing_result = MagicMock(name="my-result")
    mock_jobs = MagicMock(spec=JobsClient)
    mock_jobs.create_job_result.side_effect = _conflict_error()
    mock_jobs.get_job_result.return_value = _resp(existing_result)
    # Job retrieval provides the fileset name.
    mock_jobs.get_job.return_value = _resp(
        MagicMock(attempt_id="att-123", fileset="test-fileset", output_location=None)
    )

    sdk = _sync_client()
    mgr = rm.ResultManager(
        job_name="test-job",
        workspace="test-ws",
        files_client=FilesClient.from_client(sdk),
        jobs_client=mock_jobs,
    )

    with patch.object(mgr, "_create_file_manager", return_value=mock_sync_file_manager):
        result = mgr.create_result("my-result", test_file)

    mock_jobs.get_job_result.assert_called_once_with(name="my-result", job="test-job", workspace="test-ws")
    assert result is existing_result


def test_create_result_wraps_transport_errors_sync(tmp_path, mock_sync_file_manager):
    test_file = tmp_path / "artifact.bin"
    test_file.write_bytes(b"test content")
    request = httpx.Request("POST", "http://test/apis/jobs/v2/workspaces/test-ws/jobs/test-job/results/my-result")
    mock_jobs = MagicMock(spec=JobsClient)
    mock_jobs.get_job.return_value = _resp(
        MagicMock(attempt_id="att-123", fileset="test-fileset", output_location=None)
    )
    mock_jobs.create_job_result.side_effect = NemoTransportError(
        httpx.ConnectError("Connection refused", request=request)
    )
    sdk = _sync_client()
    mgr = rm.ResultManager(
        job_name="test-job",
        workspace="test-ws",
        files_client=FilesClient.from_client(sdk),
        jobs_client=mock_jobs,
    )

    with (
        patch.object(mgr, "_create_file_manager", return_value=mock_sync_file_manager),
        pytest.raises(CreateJobResultError, match="Error creating job result"),
    ):
        mgr.create_result("my-result", test_file)


@pytest.mark.asyncio
async def test_create_result_returns_existing_on_conflict_async(tmp_path, mock_async_file_manager):
    """Test that async create_result returns existing result when ConflictError is raised."""
    test_file = tmp_path / "artifact.bin"
    test_file.write_bytes(b"test content")

    # Configure the typed async jobs client to raise ConflictError on create, return existing on retrieve.
    existing_result = MagicMock(name="my-result")
    mock_jobs = MagicMock(spec=AsyncJobsClient)
    mock_jobs.create_job_result = AsyncMock(side_effect=_conflict_error())
    mock_jobs.get_job_result = AsyncMock(return_value=_resp(existing_result))
    # Job retrieval provides the fileset name.
    mock_jobs.get_job = AsyncMock(
        return_value=_resp(MagicMock(attempt_id="att-123", fileset="test-fileset", output_location=None))
    )

    sdk = _async_client()
    mgr = rm.AsyncResultManager(
        job_name="test-job",
        workspace="test-ws",
        files_client=AsyncFilesClient.from_client(sdk),
        jobs_client=mock_jobs,
    )

    with patch.object(mgr, "_create_file_manager", return_value=mock_async_file_manager):
        result = await mgr.create_result("my-result", test_file)

    mock_jobs.get_job_result.assert_called_once_with(name="my-result", job="test-job", workspace="test-ws")
    assert result is existing_result


@pytest.mark.asyncio
async def test_create_result_wraps_transport_errors_async(tmp_path, mock_async_file_manager):
    test_file = tmp_path / "artifact.bin"
    test_file.write_bytes(b"test content")
    request = httpx.Request("POST", "http://test/apis/jobs/v2/workspaces/test-ws/jobs/test-job/results/my-result")
    mock_jobs = MagicMock(spec=AsyncJobsClient)
    mock_jobs.get_job = AsyncMock(
        return_value=_resp(MagicMock(attempt_id="att-123", fileset="test-fileset", output_location=None))
    )
    mock_jobs.create_job_result = AsyncMock(
        side_effect=NemoTransportError(httpx.ConnectError("Connection refused", request=request))
    )
    sdk = _async_client()
    mgr = rm.AsyncResultManager(
        job_name="test-job",
        workspace="test-ws",
        files_client=AsyncFilesClient.from_client(sdk),
        jobs_client=mock_jobs,
    )

    with (
        patch.object(mgr, "_create_file_manager", return_value=mock_async_file_manager),
        pytest.raises(CreateJobResultError, match="Error creating job result"),
    ):
        await mgr.create_result("my-result", test_file)
