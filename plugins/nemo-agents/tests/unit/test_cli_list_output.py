# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``nemo agents`` list output formats."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from nemo_agents_plugin.cli import AgentsCLI
from typer.testing import CliRunner

runner = CliRunner()

_PATCH_PREFIX = "nemo_agents_plugin.cli"


@pytest.fixture
def app():
    """Build the ``nemo agents`` Typer app."""
    return AgentsCLI().get_cli()


def _agents_response() -> dict[str, Any]:
    return {
        "data": [
            {
                "name": "nemo-agent",
                "workspace": "default",
                "description": "Built-in NeMo agent",
                "config": {"llms": {"default": {"model": "test-model"}}},
                "config_format": "nat-workflow-v1",
                "created_at": "2026-05-12T19:56:53.332720",
            }
        ],
        "pagination": {
            "page": 1,
            "page_size": 1,
            "current_page_size": 1,
            "total_pages": 1,
            "total_results": 1,
        },
    }


def _deployments_response(
    *,
    name: str = "nemo-agent-deployment",
    endpoint: str = "http://localhost:8001",
    endpoints: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "name": name,
        "agent": "nemo-agent",
        "workspace": "default",
        "status": "running",
        "endpoint": endpoint,
        "config": {"workflow": {"_type": "react_agent"}},
        "port": 8001,
        "pid": 67890,
        "created_at": "2026-05-12T20:01:00.123456",
    }
    if endpoints is not None:
        row["endpoints"] = endpoints
    return {
        "data": [row],
        "pagination": {
            "page": 1,
            "page_size": 1,
            "current_page_size": 1,
            "total_pages": 1,
            "total_results": 1,
        },
    }


def _install_mock_transport(response: dict[str, Any]):
    transport = httpx.MockTransport(lambda request: httpx.Response(200, request=request, json=response))
    real_client = httpx.Client

    class _Client(real_client):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    return patch(f"{_PATCH_PREFIX}.httpx.Client", _Client)


class TestListAgentsOutput:
    def test_agents_list_defaults_to_table(self, app) -> None:
        with _install_mock_transport(_agents_response()):
            result = runner.invoke(app, ["list"])

        assert result.exit_code == 0, result.output
        assert "nemo-agent" in result.output
        assert "config_format" in result.output
        assert "nat-workflow-v1" in result.output
        assert '"data"' not in result.output
        assert "test-model" not in result.output

    @pytest.mark.parametrize("flag", ["--format", "-o", "--output-format", "-f"])
    def test_agents_list_supports_json_output(self, app, flag: str) -> None:
        response = _agents_response()
        with _install_mock_transport(response):
            result = runner.invoke(app, ["list", flag, "json"])

        assert result.exit_code == 0, result.output
        # The resolved-target banner goes to stderr; stdout stays clean JSON.
        assert json.loads(result.stdout) == response


class TestDeploymentsListOutput:
    def test_deployments_list_defaults_to_table(self, app) -> None:
        with _install_mock_transport(_deployments_response()):
            result = runner.invoke(app, ["deployments", "list"])

        assert result.exit_code == 0, result.output
        assert "nemo-agent-deployment" in result.output
        assert "nemo-agent" in result.output
        assert "running" in result.output
        assert "endpoint" in result.output
        assert "http://loc" in result.output
        assert '"data"' not in result.output
        assert "react_agent" not in result.output
        assert "67890" not in result.output

    @pytest.mark.parametrize("flag", ["--format", "-o", "--output-format", "-f"])
    def test_deployments_list_supports_json_output(self, app, flag: str) -> None:
        response = _deployments_response()
        with _install_mock_transport(response):
            result = runner.invoke(app, ["deployments", "list", flag, "json"])

        assert result.exit_code == 0, result.output
        # The resolved-target banner goes to stderr; stdout stays clean JSON.
        assert json.loads(result.stdout) == response

    @pytest.mark.parametrize("fmt", ["table", "markdown", "csv"])
    def test_container_list_shows_projected_http_url(self, app, fmt: str) -> None:
        response = _deployments_response(
            name="local-calc-docker",
            endpoint="",
            endpoints=[{"name": "port-8000", "url": "http://localhost:49154", "protocol": "http"}],
        )
        with _install_mock_transport(response):
            result = runner.invoke(app, ["deployments", "list", "--output-format", fmt])

        assert result.exit_code == 0, result.output
        # Rich clips the default table to the terminal; markdown and CSV keep the full URL.
        expected = "http://loc" if fmt == "table" else "http://localhost:49154"
        assert expected in result.stdout

    def test_list_prefers_scalar_endpoint_over_projected_urls(self, app) -> None:
        response = _deployments_response(
            endpoint="http://127.0.0.1:49153",
            endpoints=[{"name": "port-8000", "url": "http://localhost:49154", "protocol": "http"}],
        )
        with _install_mock_transport(response):
            result = runner.invoke(app, ["deployments", "list", "--output-format", "markdown"])

        assert result.exit_code == 0, result.output
        assert "http://127.0.0.1:49153" in result.stdout
        assert "http://localhost:49154" not in result.stdout

    def test_list_prefers_http_endpoint_over_earlier_non_http(self, app) -> None:
        response = _deployments_response(
            name="local-calc-docker",
            endpoint="",
            endpoints=[
                {"name": "metrics", "url": "tcp://localhost:9090", "protocol": "tcp"},
                {"name": "port-8000", "url": "http://localhost:49154", "protocol": "http"},
            ],
        )
        with _install_mock_transport(response):
            result = runner.invoke(app, ["deployments", "list", "--output-format", "markdown"])

        assert result.exit_code == 0, result.output
        assert "http://localhost:49154" in result.stdout
        assert "tcp://localhost:9090" not in result.stdout

    def test_list_shows_first_url_when_no_http_endpoint_exists(self, app) -> None:
        response = _deployments_response(
            name="local-calc-docker",
            endpoint="",
            endpoints=[{"name": "metrics", "url": "tcp://localhost:9090", "protocol": "tcp"}],
        )
        with _install_mock_transport(response):
            result = runner.invoke(app, ["deployments", "list", "--output-format", "markdown"])

        assert result.exit_code == 0, result.output
        assert "tcp://localhost:9090" in result.stdout

    def test_list_stays_blank_when_deployment_has_no_address(self, app) -> None:
        response = _deployments_response(name="local-calc-docker", endpoint="", endpoints=[])
        with _install_mock_transport(response):
            result = runner.invoke(app, ["deployments", "list"])

        assert result.exit_code == 0, result.output
        assert "http" not in result.stdout
        assert "tcp://" not in result.stdout

    def test_container_list_json_keeps_empty_scalar_endpoint(self, app) -> None:
        endpoints = [{"name": "port-8000", "url": "http://localhost:49154", "protocol": "http"}]
        response = _deployments_response(name="local-calc-docker", endpoint="", endpoints=endpoints)
        with _install_mock_transport(response):
            result = runner.invoke(app, ["deployments", "list", "--output-format", "json"])

        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout) == response
