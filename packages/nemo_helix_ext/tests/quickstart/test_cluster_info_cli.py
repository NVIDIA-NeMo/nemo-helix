# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for cluster-info and quickstart-info CLI commands."""

import re
from unittest.mock import MagicMock, patch

import pytest
from nemo_helix_ext.cli.app import app
from typer.testing import CliRunner

runner = CliRunner()

_BASE_URL = "http://test:8080"
_QS_CONFIG = {
    "image": "nvcr.io/img:latest",
    "port": 8080,
    "storage_path": "/data",
    "docker_socket": "/var/run/docker.sock",
}


def _strip(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\x1b\[[0-9;]*m", "", text)).strip()


class TestClusterInfo:
    def test_no_config_file_uses_default_url(
        self, tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When no config file exists the CLI falls back to localhost:8080."""
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {
            "status": "healthy",
            "services": {"ready": [], "not_ready": []},
            "controllers": {"healthy": True, "status": {}},
        }
        with patch("httpx.get", return_value=mock_resp) as mock_get:
            result = runner.invoke(app, ["cluster-info"])
        assert result.exit_code == 0
        called_url = mock_get.call_args[0][0]
        assert "localhost:8080" in called_url

    @pytest.mark.parametrize(
        "status_data,expected_fragments",
        [
            (
                {
                    "status": "healthy",
                    "services": {"ready": ["auth", "jobs"], "not_ready": []},
                    "controllers": {"healthy": True, "status": {}},
                },
                ["✓ Status: healthy", "Ready (2): auth, jobs"],
            ),
            (
                {
                    "status": "degraded",
                    "services": {"ready": ["auth"], "not_ready": [{"name": "jobs", "message": "crash"}]},
                    "controllers": {"healthy": True, "status": {}},
                },
                ["! Status: degraded", "Ready (1): auth", "Not ready: jobs — crash"],
            ),
            (
                {
                    "status": "unhealthy",
                    "services": {
                        "ready": [],
                        "not_ready": [{"name": "auth", "message": ""}, {"name": "jobs", "message": ""}],
                    },
                    "controllers": {"healthy": False, "status": {"jobs-ctrl": False}},
                },
                ["✗ Status: unhealthy", "Not ready: auth", "Not ready: jobs", "Unhealthy controllers: jobs-ctrl"],
            ),
        ],
    )
    def test_platform_status_displayed(
        self,
        tmp_path: pytest.TempPathFactory,
        monkeypatch: pytest.MonkeyPatch,
        status_data: dict,
        expected_fragments: list[str],
    ) -> None:
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = status_data
        with (
            patch("nemo_helix_ext.cli.core.context.CLIContext.get_base_url", return_value=_BASE_URL),
            patch("httpx.get", return_value=mock_resp),
        ):
            result = runner.invoke(app, ["cluster-info"])
        assert result.exit_code == 0
        output = _strip(result.stderr)
        for fragment in expected_fragments:
            assert fragment in output, f"{fragment!r} not in output: {output!r}"

    def test_shows_url_in_output(self, tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {
            "status": "healthy",
            "services": {"ready": [], "not_ready": []},
            "controllers": {"healthy": True, "status": {}},
        }
        with (
            patch("nemo_helix_ext.cli.core.context.CLIContext.get_base_url", return_value=_BASE_URL),
            patch("httpx.get", return_value=mock_resp),
        ):
            result = runner.invoke(app, ["cluster-info"])
        assert _BASE_URL in result.stderr

    @pytest.mark.parametrize("base_url", ["http://test:8080/", "http://test:8080"])
    def test_trailing_slash_in_url_does_not_cause_double_slash(
        self, tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch, base_url: str
    ) -> None:
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {
            "status": "healthy",
            "services": {"ready": [], "not_ready": []},
            "controllers": {"healthy": True, "status": {}},
        }
        with (
            patch("nemo_helix_ext.cli.core.context.CLIContext.get_base_url", return_value=base_url),
            patch("httpx.get", return_value=mock_resp) as mock_get,
        ):
            runner.invoke(app, ["cluster-info"])
        called_url = mock_get.call_args[0][0]
        assert "//" not in called_url.split("://", 1)[1]

    def test_connection_error_exits_one(
        self, tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        with (
            patch("nemo_helix_ext.cli.core.context.CLIContext.get_base_url", return_value=_BASE_URL),
            patch("httpx.get", side_effect=Exception("Connection refused")),
        ):
            result = runner.invoke(app, ["cluster-info"])
        assert result.exit_code == 1
        assert "Could not reach cluster" in result.stderr

    @pytest.mark.parametrize("status_code", [500, 503])
    def test_non_200_response_exits_one(
        self,
        tmp_path: pytest.TempPathFactory,
        monkeypatch: pytest.MonkeyPatch,
        status_code: int,
    ) -> None:
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        mock_resp = MagicMock(status_code=status_code)
        with (
            patch("nemo_helix_ext.cli.core.context.CLIContext.get_base_url", return_value=_BASE_URL),
            patch("httpx.get", return_value=mock_resp),
        ):
            result = runner.invoke(app, ["cluster-info"])
        assert result.exit_code == 1
        assert str(status_code) in result.stderr


class TestQuickstartInfo:
    def test_docker_unavailable_exits_one(
        self, tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        mock_cluster = MagicMock()
        mock_cluster.cluster_info.return_value = {"status": "docker-unavailable", "running": False}
        with patch("nemo_helix_ext.quickstart.QuickstartCluster", return_value=mock_cluster):
            result = runner.invoke(app, ["quickstart", "status"])
        assert result.exit_code == 1
        assert "Docker is not running" in result.stderr

    @pytest.mark.parametrize(
        "info,expected_fragments",
        [
            (
                {"status": "not found", "running": False, "config": _QS_CONFIG},
                ["✗ Running: No", "not found"],
            ),
            (
                {
                    "status": "running",
                    "running": True,
                    "id": "abc123",
                    "health": "healthy",
                    "url": "http://localhost:8080",
                    "config": _QS_CONFIG,
                },
                ["✓ Running: Yes", "nvcr.io/img:latest", "✓ Health: healthy", "http://localhost:8080"],
            ),
            (
                {
                    "status": "running",
                    "running": True,
                    "id": "def456",
                    "health": "starting",
                    "url": "http://localhost:8080",
                    "config": _QS_CONFIG,
                },
                ["✓ Running: Yes", "! Health: starting"],
            ),
        ],
    )
    def test_cluster_states(
        self,
        tmp_path: pytest.TempPathFactory,
        monkeypatch: pytest.MonkeyPatch,
        info: dict,
        expected_fragments: list[str],
    ) -> None:
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        mock_cluster = MagicMock()
        mock_cluster.cluster_info.return_value = info
        with patch("nemo_helix_ext.quickstart.QuickstartCluster", return_value=mock_cluster):
            result = runner.invoke(app, ["quickstart", "status"])
        assert result.exit_code == 0
        output = _strip(result.stderr)
        for fragment in expected_fragments:
            assert fragment in output, f"{fragment!r} not in output: {output!r}"
