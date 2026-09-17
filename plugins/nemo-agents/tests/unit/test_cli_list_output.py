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


def _deployments_response() -> dict[str, Any]:
    return {
        "data": [
            {
                "name": "nemo-agent-deployment",
                "agent": "nemo-agent",
                "workspace": "default",
                "status": "running",
                "endpoint": "http://localhost:8001",
                "config": {"workflow": {"_type": "react_agent"}},
                "port": 8001,
                "pid": 67890,
                "created_at": "2026-05-12T20:01:00.123456",
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
