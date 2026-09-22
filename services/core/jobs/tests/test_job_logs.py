# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nhx.common.entities import DEFAULT_WORKSPACE, EntityClient
from nhx.common.jobs.log_client import dep_job_logs_client
from nhx.common.jobs.schemas import InvalidPageCursorError, HelixJobLog, HelixJobLogPage
from nhx.core.jobs.api.v2.jobs.endpoints import dep_dispatcher, router
from nhx.core.jobs.app.dispatcher import JobDispatcher
from nhx.testing import create_test_client


@pytest.fixture
def sample_logs() -> list[HelixJobLog]:
    """Create sample job logs for testing."""
    return [
        HelixJobLog(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            job="test-job",
            job_step="step1",
            job_task="task1",
            message="Starting job execution",
        ),
        HelixJobLog(
            timestamp=datetime(2024, 1, 1, 12, 0, 5),
            job="test-job",
            job_step="step1",
            job_task="task1",
            message="Processing data",
        ),
        HelixJobLog(
            timestamp=datetime(2024, 1, 1, 12, 0, 10),
            job="test-job",
            job_step="step1",
            job_task="task1",
            message="Job completed successfully",
        ),
    ]


class TestJobLogsAPI:
    """Test suite for job logs API endpoints."""

    @pytest.fixture
    def mock_logs_client(self):
        """Create a mock logs client for querying job logs."""
        client = MagicMock()
        client.query_logs = AsyncMock()
        return client

    @pytest.fixture
    def dispatcher(self):
        """Create a real dispatcher with test entity store and mock SDK."""
        projects = ["default/test-project"]
        with create_test_client(client_type=EntityClient, projects=projects) as mock_store:
            mock_nhx_client = MagicMock()
            mock_files = AsyncMock()
            mock_fileset_obj = MagicMock()
            mock_fileset_obj.name = "test-fileset-id"
            mock_resp = MagicMock()
            mock_resp.data.return_value = mock_fileset_obj
            mock_files.create_fileset.return_value = mock_resp

            with patch("nhx.core.jobs.app.dispatcher.client_from_platform", return_value=mock_files):
                dispatcher = JobDispatcher(store=mock_store, sdk=mock_nhx_client)
                yield dispatcher

    @pytest.fixture
    def test_client(self, dispatcher, mock_logs_client):
        """Create a test client with real dispatcher and mocked logs client."""
        app = FastAPI()
        app.dependency_overrides[dep_dispatcher] = lambda: dispatcher
        app.dependency_overrides[dep_job_logs_client] = lambda: mock_logs_client
        app.include_router(router)
        return TestClient(app)

    async def test_get_job_logs_success(
        self, test_client, dispatcher, mock_logs_client, sample_logs, sample_platform_job_request
    ):
        """Test successful retrieval of job logs."""
        # Create a real job
        job = await dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)

        mock_logs_client.query_logs.return_value = HelixJobLogPage(
            data=sample_logs, total=3, next_page=None, prev_page=None
        )

        response = test_client.get(f"/v2/workspaces/{DEFAULT_WORKSPACE}/jobs/{job.name}/logs")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        assert len(data["data"]) == 3
        assert data["data"][0]["message"] == "Starting job execution"
        assert data["data"][0]["job"] == "test-job"
        assert data["data"][0]["job_step"] == "step1"
        assert data["next_page"] is None
        assert data["prev_page"] is None

        mock_logs_client.query_logs.assert_called_once_with(
            job.fileset,
            workspace=DEFAULT_WORKSPACE,
            filters={"job": job.name, "job_attempt": job.attempt_id},
            page_size=100,
            page_cursor=None,
            tail=None,
            artifact_base_path=None,
        )

    async def test_get_job_logs_with_pagination_params(
        self, test_client, dispatcher, mock_logs_client, sample_logs, sample_platform_job_request
    ):
        """Test job logs retrieval with pagination parameters."""
        # Create a real job
        job = await dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)

        mock_logs_client.query_logs.return_value = HelixJobLogPage(
            data=sample_logs[:2],
            total=3,
            next_page="next_cursor_123",
            prev_page=None,
        )

        response = test_client.get(
            f"/v2/workspaces/{DEFAULT_WORKSPACE}/jobs/{job.name}/logs?limit=2&page_cursor=some_cursor"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        assert len(data["data"]) == 2
        assert data["next_page"] == "next_cursor_123"
        assert data["prev_page"] is None

        mock_logs_client.query_logs.assert_called_once_with(
            job.fileset,
            workspace=DEFAULT_WORKSPACE,
            filters={"job": job.name, "job_attempt": job.attempt_id},
            page_size=2,
            page_cursor="some_cursor",
            tail=None,
            artifact_base_path=None,
        )

    async def test_get_job_logs_with_tail(
        self, test_client, dispatcher, mock_logs_client, sample_logs, sample_platform_job_request
    ):
        """Test job logs retrieval with tail parameter."""
        job = await dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)

        mock_logs_client.query_logs.return_value = HelixJobLogPage(
            data=sample_logs[-2:],
            total=3,
            next_page=None,
            prev_page="previous_cursor",
        )

        response = test_client.get(f"/v2/workspaces/{DEFAULT_WORKSPACE}/jobs/{job.name}/logs?tail=2")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        assert len(data["data"]) == 2
        assert data["next_page"] is None
        assert data["prev_page"] == "previous_cursor"

        mock_logs_client.query_logs.assert_called_once_with(
            job.fileset,
            workspace=DEFAULT_WORKSPACE,
            filters={"job": job.name, "job_attempt": job.attempt_id},
            page_size=100,
            page_cursor=None,
            tail=2,
            artifact_base_path=None,
        )

    async def test_get_job_logs_rejects_tail_with_limit(
        self, test_client, dispatcher, mock_logs_client, sample_platform_job_request
    ):
        """Tail controls returned window size and cannot be combined with limit."""
        job = await dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)

        response = test_client.get(f"/v2/workspaces/{DEFAULT_WORKSPACE}/jobs/{job.name}/logs?tail=2&limit=2")

        assert response.status_code == 400
        assert (
            response.json()["detail"]
            == "tail cannot be combined with limit; tail controls the returned log window size."
        )
        mock_logs_client.query_logs.assert_not_called()

    async def test_get_job_logs_rejects_tail_with_page_cursor(
        self, test_client, dispatcher, mock_logs_client, sample_platform_job_request
    ):
        """Tail starts at the end and cannot be combined with an existing cursor."""
        job = await dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)

        response = test_client.get(f"/v2/workspaces/{DEFAULT_WORKSPACE}/jobs/{job.name}/logs?tail=2&page_cursor=abc")

        assert response.status_code == 400
        assert (
            response.json()["detail"]
            == "tail cannot be combined with page_cursor; pass the returned prev_page as page_cursor without tail."
        )
        mock_logs_client.query_logs.assert_not_called()

    async def test_get_job_logs_scopes_to_artifact_base_path_when_output_location_set(
        self, test_client, dispatcher, mock_logs_client, sample_logs, sample_platform_job_request
    ):
        """A job created with output_location scopes its log query to the per-job artifact base path."""
        request = sample_platform_job_request.model_copy(update={"output_location": "shared-fs"})
        job = await dispatcher.create_job(request, DEFAULT_WORKSPACE)

        mock_logs_client.query_logs.return_value = HelixJobLogPage(
            data=sample_logs, total=3, next_page=None, prev_page=None
        )

        response = test_client.get(f"/v2/workspaces/{DEFAULT_WORKSPACE}/jobs/{job.name}/logs")

        assert response.status_code == 200
        assert mock_logs_client.query_logs.call_args.kwargs["artifact_base_path"] == f"jobs/{job.name}"

    async def test_get_job_logs_job_not_found(self, test_client, dispatcher, mock_logs_client):
        """Test job logs retrieval when job doesn't exist."""
        response = test_client.get(f"/v2/workspaces/{DEFAULT_WORKSPACE}/jobs/nonexistent-job/logs")

        assert response.status_code == 404
        assert response.json()["detail"] == "Job 'nonexistent-job' not found in workspace 'default'."
        mock_logs_client.query_logs.assert_not_called()

    async def test_get_job_logs_invalid_page_cursor(
        self, test_client, dispatcher, mock_logs_client, sample_platform_job_request
    ):
        """Test job logs retrieval with invalid page cursor."""
        # Create a real job
        job = await dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)

        mock_logs_client.query_logs.side_effect = InvalidPageCursorError("Invalid page cursor")

        response = test_client.get(
            f"/v2/workspaces/{DEFAULT_WORKSPACE}/jobs/{job.name}/logs?page_cursor=invalid_cursor"
        )

        assert response.status_code == 422
        assert response.json()["detail"] == "Invalid page cursor"

    async def test_get_job_logs_page_cursor_filter_mismatch_message(
        self, test_client, dispatcher, mock_logs_client, sample_platform_job_request
    ):
        """Cursor mismatch errors preserve their detail for callers."""
        job = await dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)

        mock_logs_client.query_logs.side_effect = InvalidPageCursorError(
            "page_cursor does not match the current log filters."
        )

        response = test_client.get(f"/v2/workspaces/{DEFAULT_WORKSPACE}/jobs/{job.name}/logs?page_cursor=old")

        assert response.status_code == 422
        assert response.json()["detail"] == "page_cursor does not match the current log filters."

    def test_get_job_logs_invalid_limit(self, test_client):
        """Test job logs retrieval with invalid limit parameter."""
        response = test_client.get(f"/v2/workspaces/{DEFAULT_WORKSPACE}/jobs/test-job/logs?limit=0")

        # Should get validation error before reaching our code
        assert response.status_code == 422

    async def test_get_job_logs_empty_logs(
        self, test_client, dispatcher, mock_logs_client, sample_platform_job_request
    ):
        """Test job logs retrieval when no logs exist."""
        # Create a real job
        job = await dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)

        mock_logs_client.query_logs.return_value = HelixJobLogPage(data=[], total=0, next_page=None, prev_page=None)

        response = test_client.get(f"/v2/workspaces/{DEFAULT_WORKSPACE}/jobs/{job.name}/logs")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert len(data["data"]) == 0
        assert data["next_page"] is None
        assert data["prev_page"] is None

    async def test_get_job_logs_large_limit(
        self, test_client, dispatcher, mock_logs_client, sample_logs, sample_platform_job_request
    ):
        """Test job logs retrieval with a large limit parameter."""
        # Create a real job
        job = await dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)

        mock_logs_client.query_logs.return_value = HelixJobLogPage(
            data=sample_logs,
            total=len(sample_logs),
            next_page=None,
            prev_page=None,
        )

        response = test_client.get(f"/v2/workspaces/{DEFAULT_WORKSPACE}/jobs/{job.name}/logs?limit=1000")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        assert len(data["data"]) == 3
        mock_logs_client.query_logs.assert_called_once()
