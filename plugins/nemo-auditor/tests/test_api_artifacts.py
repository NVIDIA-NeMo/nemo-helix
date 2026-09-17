# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the aggregate artifact download route (GET /results/artifacts/download).

The endpoint fetches individual garak report results from the Jobs service and
streams them back as a single tar.gz — no stored 'artifacts' result. Tests use
FastAPI's TestClient with dependency_overrides and targeted patches so no real
platform or fileset is needed.
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from nemo_auditor.api.v2 import artifacts as artifacts_module
from nemo_auditor.service import AuditorPluginService
from nemo_platform_plugin.client.errors import NotFoundError
from nemo_platform_plugin.dependencies import get_sdk_client
from nemo_platform_plugin.jobs.file_manager import TmpDirPath

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result_response(artifact_url: str = "fileset://default/fs/r") -> MagicMock:
    """A jobs-client response whose .data() returns a result info with artifact_url."""
    result_info = MagicMock()
    result_info.artifact_url = artifact_url
    resp = MagicMock()
    resp.data.return_value = result_info
    return resp


def _make_tmp(tmp_path: Path, filename: str, content: bytes = b"data") -> TmpDirPath:
    """Real TmpDirPath backed by a file under pytest's tmp_path."""
    d = tmp_path / filename
    d.mkdir()
    f = d / filename
    f.write_bytes(content)
    return TmpDirPath(path=f, tmp_dir=d)


def _tar_names(content: bytes) -> list[str]:
    with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as tar:
        return tar.getnames()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_sdk() -> MagicMock:
    return MagicMock()


@pytest.fixture
def test_app(mock_sdk: MagicMock) -> FastAPI:
    app = FastAPI()
    app.include_router(
        artifacts_module.router,
        prefix="/apis/auditor/v2/workspaces/{workspace}",
    )
    app.dependency_overrides[get_sdk_client] = lambda: mock_sdk
    return app


@pytest.fixture
def client(test_app: FastAPI) -> TestClient:
    return TestClient(test_app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDownloadAuditArtifacts:
    def test_all_results_present_returns_200_with_valid_targz(
        self, client: TestClient, mock_sdk: MagicMock, tmp_path: Path
    ) -> None:
        jsonl_tmp = _make_tmp(tmp_path, "run.report.jsonl", b"jsonl")
        html_tmp = _make_tmp(tmp_path, "run.report.html", b"<html/>")
        hitlog_tmp = _make_tmp(tmp_path, "run.hitlog.jsonl", b"hitlog")

        mock_jobs = AsyncMock()
        mock_jobs.get_job_result = AsyncMock(
            side_effect=[
                _make_result_response("fileset://default/fs/report-jsonl"),
                _make_result_response("fileset://default/fs/report-html"),
                _make_result_response("fileset://default/fs/report-hitlog-jsonl"),
            ]
        )

        mock_rm = AsyncMock()
        mock_rm.download_artifact = AsyncMock(side_effect=[jsonl_tmp, html_tmp, hitlog_tmp])

        with (
            patch.object(artifacts_module, "client_from_platform", return_value=mock_jobs),
            patch.object(artifacts_module, "async_result_manager_factory", return_value=mock_rm),
        ):
            resp = client.get("/apis/auditor/v2/workspaces/default/jobs/audit/job-1/results/artifacts/download")

        assert resp.status_code == 200
        assert "gzip" in resp.headers["content-type"]
        names = _tar_names(resp.content)
        assert "run.report.jsonl" in names
        assert "run.report.html" in names
        assert "run.hitlog.jsonl" in names

    def test_missing_optional_results_are_skipped(
        self, client: TestClient, mock_sdk: MagicMock, tmp_path: Path
    ) -> None:
        jsonl_tmp = _make_tmp(tmp_path, "run.report.jsonl", b"jsonl")

        mock_jobs = AsyncMock()
        mock_jobs.get_job_result = AsyncMock(
            side_effect=[
                _make_result_response("fileset://default/fs/report-jsonl"),
                NotFoundError(MagicMock()),
                NotFoundError(MagicMock()),
            ]
        )

        mock_rm = AsyncMock()
        mock_rm.download_artifact = AsyncMock(return_value=jsonl_tmp)

        with (
            patch.object(artifacts_module, "client_from_platform", return_value=mock_jobs),
            patch.object(artifacts_module, "async_result_manager_factory", return_value=mock_rm),
        ):
            resp = client.get("/apis/auditor/v2/workspaces/default/jobs/audit/job-1/results/artifacts/download")

        assert resp.status_code == 200
        names = _tar_names(resp.content)
        assert "run.report.jsonl" in names
        assert len(names) == 1

    def test_no_results_at_all_returns_404(self, client: TestClient, mock_sdk: MagicMock) -> None:
        mock_jobs = AsyncMock()
        mock_jobs.get_job_result = AsyncMock(side_effect=NotFoundError(MagicMock()))

        mock_rm = AsyncMock()

        with (
            patch.object(artifacts_module, "client_from_platform", return_value=mock_jobs),
            patch.object(artifacts_module, "async_result_manager_factory", return_value=mock_rm),
        ):
            resp = client.get("/apis/auditor/v2/workspaces/default/jobs/audit/job-1/results/artifacts/download")

        assert resp.status_code == 404
        assert "job-1" in resp.json()["detail"]

    def test_result_names_queried_in_order(self, client: TestClient, mock_sdk: MagicMock, tmp_path: Path) -> None:
        jsonl_tmp = _make_tmp(tmp_path, "run.report.jsonl")
        html_tmp = _make_tmp(tmp_path, "run.report.html")
        hitlog_tmp = _make_tmp(tmp_path, "run.hitlog.jsonl")

        mock_jobs = AsyncMock()
        mock_jobs.get_job_result = AsyncMock(
            side_effect=[
                _make_result_response(),
                _make_result_response(),
                _make_result_response(),
            ]
        )

        mock_rm = AsyncMock()
        mock_rm.download_artifact = AsyncMock(side_effect=[jsonl_tmp, html_tmp, hitlog_tmp])

        with (
            patch.object(artifacts_module, "client_from_platform", return_value=mock_jobs),
            patch.object(artifacts_module, "async_result_manager_factory", return_value=mock_rm),
        ):
            client.get("/apis/auditor/v2/workspaces/default/jobs/audit/job-1/results/artifacts/download")

        queried = [call.kwargs["name"] for call in mock_jobs.get_job_result.await_args_list]
        assert queried == ["report-jsonl", "report-html", "report-hitlog-jsonl"]

    def test_workspace_and_job_forwarded_to_jobs_client(
        self, client: TestClient, mock_sdk: MagicMock, tmp_path: Path
    ) -> None:
        tmp = _make_tmp(tmp_path, "run.report.jsonl")

        mock_jobs = AsyncMock()
        mock_jobs.get_job_result = AsyncMock(
            side_effect=[_make_result_response(), NotFoundError(MagicMock()), NotFoundError(MagicMock())]
        )

        mock_rm = AsyncMock()
        mock_rm.download_artifact = AsyncMock(return_value=tmp)

        with (
            patch.object(artifacts_module, "client_from_platform", return_value=mock_jobs),
            patch.object(artifacts_module, "async_result_manager_factory", return_value=mock_rm) as factory,
        ):
            client.get("/apis/auditor/v2/workspaces/prod/jobs/audit/my-job/results/artifacts/download")

        first_call = mock_jobs.get_job_result.await_args_list[0]
        assert first_call.kwargs["workspace"] == "prod"
        assert first_call.kwargs["job"] == "my-job"
        factory.assert_called_once_with(
            job_name="my-job",
            workspace="prod",
            files_client=mock_jobs,
            jobs_client=mock_jobs,
        )


class TestArtifactsRouteWiring:
    def test_artifacts_download_route_is_mounted(self) -> None:
        service = AuditorPluginService()
        paths: set[str] = set()
        for spec in service.get_routers():
            for route in spec.router.routes:
                if not isinstance(route, APIRoute):
                    continue
                methods = route.methods
                if methods is not None and "GET" in methods:
                    paths.add(f"/apis/auditor{spec.prefix}{route.path}")
        assert "/apis/auditor/v2/workspaces/{workspace}/jobs/audit/{job}/results/artifacts/download" in paths

    def test_artifacts_route_precedes_generic_name_route(self) -> None:
        # FastAPI matches in registration order; the specific /artifacts/download path
        # must appear before the wildcard /{name}/download catch-all within each router.
        service = AuditorPluginService()
        all_get_paths: list[str] = []
        for spec in service.get_routers():
            for route in spec.router.routes:
                if not isinstance(route, APIRoute):
                    continue
                methods = route.methods
                if methods is not None and "GET" in methods:
                    all_get_paths.append(f"/apis/auditor{spec.prefix}{route.path}")

        artifacts_idx = next(i for i, p in enumerate(all_get_paths) if p.endswith("/artifacts/download"))
        generic_idx = next(i for i, p in enumerate(all_get_paths) if p.endswith("/{name}/download"))
        assert artifacts_idx < generic_idx
