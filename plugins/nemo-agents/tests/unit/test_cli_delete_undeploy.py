# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``nemo agents delete``, ``undeploy``, and ``deployments delete`` confirmation flags.

Verifies:
- Interactive confirmation prompts when ``--yes`` is not set
- Abort on user decline
- ``--yes`` / ``-y`` skips the prompt
- ``--all`` works as an alias for ``--agent`` on ``undeploy``
- Agent delete leaves the durable ``{agent}-ethos`` fileset in place
"""

from __future__ import annotations

from unittest.mock import ANY, patch

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


def _install_mock_transport(handler):
    transport = httpx.MockTransport(handler)
    real_client = httpx.Client

    class _Client(real_client):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    return patch(f"{_PATCH_PREFIX}.httpx.Client", _Client)


def _recording_delete_handler(requests: list[httpx.Request]):
    def handler(req: httpx.Request) -> httpx.Response:
        requests.append(req)
        return httpx.Response(204, request=req)

    return handler


def _recording_deployments_handler(requests: list[httpx.Request], deployments: list[dict]):
    def handler(req: httpx.Request) -> httpx.Response:
        requests.append(req)
        if req.method == "GET":
            return httpx.Response(200, request=req, json=_mock_deployments_response(deployments))
        return httpx.Response(204, request=req)

    return handler


def _recording_deployment_pages_handler(requests: list[httpx.Request], pages: list[list[dict]]):
    total_results = sum(len(page) for page in pages)

    def handler(req: httpx.Request) -> httpx.Response:
        requests.append(req)
        if req.method == "GET":
            page_number = int(req.url.params.get("page", "1"))
            return httpx.Response(
                200,
                request=req,
                json=_mock_deployments_response(
                    pages[page_number - 1],
                    page=page_number,
                    total_pages=len(pages),
                    total_results=total_results,
                ),
            )
        return httpx.Response(204, request=req)

    return handler


# ---------------------------------------------------------------------------
# nemo agents delete
# ---------------------------------------------------------------------------


class TestDeleteConfirmation:
    def test_delete_prompts_when_no_yes_flag(self, app) -> None:
        """Without --yes, the user is prompted; answering 'y' proceeds."""
        with patch(f"{_PATCH_PREFIX}._delete_agent_entity") as mock_delete:
            result = runner.invoke(app, ["delete", "my-agent"], input="y\n")

        assert result.exit_code == 0, result.output
        mock_delete.assert_called_once_with(
            agent_name="my-agent",
            workspace="default",
            base_url=ANY,
        )
        assert "deleted" in result.output.lower()

    def test_delete_aborts_when_user_declines(self, app) -> None:
        """Without --yes, answering 'n' aborts without calling the API."""
        with patch(f"{_PATCH_PREFIX}._delete_agent_entity") as mock_delete:
            result = runner.invoke(app, ["delete", "my-agent"], input="n\n")

        assert result.exit_code != 0
        mock_delete.assert_not_called()

    @pytest.mark.parametrize("flag", ["--yes", "-y"])
    def test_delete_skips_prompt_with_yes_flag(self, app, flag: str) -> None:
        """--yes and -y both skip the confirmation prompt."""
        with patch(f"{_PATCH_PREFIX}._delete_agent_entity") as mock_delete:
            result = runner.invoke(app, ["delete", "my-agent", flag])

        assert result.exit_code == 0, result.output
        mock_delete.assert_called_once()
        assert "deleted" in result.output.lower()

    def test_delete_removes_agent_but_preserves_ethos_fileset(self, app) -> None:
        """The ``{agent}-ethos`` fileset is durable: it holds ``ETHOS.md``."""
        methods: list[str] = []

        def handler(req: httpx.Request) -> httpx.Response:
            methods.append(req.method)
            assert req.url.path.endswith("/agents/my-agent")
            return httpx.Response(204)

        with (
            _install_mock_transport(handler),
            patch(f"{_PATCH_PREFIX}._platform_sdk") as mock_sdk,
        ):
            result = runner.invoke(app, ["delete", "my-agent", "--yes", "--base-url", "http://test"])

        assert result.exit_code == 0, result.output
        assert methods == ["DELETE"]
        mock_sdk.assert_not_called()


# ---------------------------------------------------------------------------
# nemo agents undeploy
# ---------------------------------------------------------------------------


def _mock_deployments_response(
    deployments: list[dict],
    *,
    page: int = 1,
    total_pages: int = 1,
    total_results: int | None = None,
) -> dict:
    """Build a fake paginated response for GET /deployments."""
    total = len(deployments) if total_results is None else total_results
    return {
        "data": deployments,
        "pagination": {
            "page": page,
            "page_size": len(deployments),
            "current_page_size": len(deployments),
            "total_pages": total_pages,
            "total_results": total,
        },
    }


class TestUndeployConfirmation:
    def test_undeploy_single_prompts_confirmation(self, app) -> None:
        """Undeploying a single deployment prompts for confirmation."""
        requests: list[httpx.Request] = []
        with _install_mock_transport(_recording_delete_handler(requests)):
            result = runner.invoke(app, ["undeploy", "dep-1"], input="y\n")

        assert result.exit_code == 0, result.output
        assert [request.method for request in requests] == ["DELETE"]

    def test_undeploy_single_aborts_on_decline(self, app) -> None:
        """Declining the prompt does not call the API."""
        requests: list[httpx.Request] = []
        with _install_mock_transport(_recording_delete_handler(requests)):
            result = runner.invoke(app, ["undeploy", "dep-1"], input="n\n")

        assert result.exit_code != 0
        assert requests == []

    def test_undeploy_single_yes_skips_prompt(self, app) -> None:
        """--yes skips the confirmation for single deployment undeploy."""
        requests: list[httpx.Request] = []
        with _install_mock_transport(_recording_delete_handler(requests)):
            result = runner.invoke(app, ["undeploy", "dep-1", "--yes"])

        assert result.exit_code == 0, result.output
        assert [request.method for request in requests] == ["DELETE"]

    def test_undeploy_by_agent_prompts_with_count(self, app) -> None:
        """Undeploying by --agent lists deployments, shows count in prompt, and deletes on 'y'."""
        requests: list[httpx.Request] = []
        deps = [
            {"name": "dep-1", "agent": "my-agent"},
            {"name": "dep-2", "agent": "my-agent"},
        ]

        with _install_mock_transport(_recording_deployments_handler(requests, deps)):
            result = runner.invoke(app, ["undeploy", "--agent", "my-agent"], input="y\n")

        assert result.exit_code == 0, result.output
        assert [request.method for request in requests].count("DELETE") == 2

    def test_undeploy_by_agent_aborts_on_decline(self, app) -> None:
        """Declining the prompt after listing does not delete anything."""
        requests: list[httpx.Request] = []
        deps = [
            {"name": "dep-1", "agent": "my-agent"},
            {"name": "dep-2", "agent": "my-agent"},
        ]

        with _install_mock_transport(_recording_deployments_handler(requests, deps)):
            result = runner.invoke(app, ["undeploy", "--agent", "my-agent"], input="n\n")

        assert result.exit_code != 0
        assert [request.method for request in requests] == ["GET"]

    def test_undeploy_all_flag_works_as_agent_alias(self, app) -> None:
        """--all is an alias for --agent on undeploy."""
        requests: list[httpx.Request] = []
        deps = [
            {"name": "dep-1", "agent": "my-agent"},
        ]

        with _install_mock_transport(_recording_deployments_handler(requests, deps)):
            result = runner.invoke(app, ["undeploy", "--all", "my-agent", "--yes"])

        assert result.exit_code == 0, result.output
        assert [request.method for request in requests].count("DELETE") == 1

    def test_undeploy_by_agent_fetches_all_pages(self, app) -> None:
        """Bulk undeploy deletes matching deployments beyond the first list page."""
        requests: list[httpx.Request] = []
        pages = [
            [
                {"name": "dep-1", "agent": "other-agent"},
            ],
            [
                {"name": "dep-2", "agent": "my-agent"},
                {"name": "dep-3", "agent": "my-agent"},
            ],
        ]

        with _install_mock_transport(_recording_deployment_pages_handler(requests, pages)):
            result = runner.invoke(app, ["undeploy", "--agent", "my-agent", "--yes"])

        assert result.exit_code == 0, result.output
        get_requests = [request for request in requests if request.method == "GET"]
        delete_paths = [request.url.path for request in requests if request.method == "DELETE"]
        assert len(get_requests) == 2
        assert get_requests[1].url.params["page"] == "2"
        assert delete_paths == [
            "/apis/agents/v2/workspaces/default/deployments/dep-2",
            "/apis/agents/v2/workspaces/default/deployments/dep-3",
        ]


# ---------------------------------------------------------------------------
# nemo agents deployments delete
# ---------------------------------------------------------------------------


class TestDeploymentsDeleteConfirmation:
    def test_deployments_delete_prompts_without_yes(self, app) -> None:
        """deployments delete prompts for confirmation."""
        requests: list[httpx.Request] = []
        with _install_mock_transport(_recording_delete_handler(requests)):
            result = runner.invoke(app, ["deployments", "delete", "dep-1"], input="y\n")

        assert result.exit_code == 0, result.output
        assert [request.method for request in requests] == ["DELETE"]

    def test_deployments_delete_aborts_on_decline(self, app) -> None:
        """Declining aborts without calling the API."""
        requests: list[httpx.Request] = []
        with _install_mock_transport(_recording_delete_handler(requests)):
            result = runner.invoke(app, ["deployments", "delete", "dep-1"], input="n\n")

        assert result.exit_code != 0
        assert requests == []

    def test_deployments_delete_skips_with_yes(self, app) -> None:
        """--yes skips the confirmation prompt."""
        requests: list[httpx.Request] = []
        with _install_mock_transport(_recording_delete_handler(requests)):
            result = runner.invoke(app, ["deployments", "delete", "dep-1", "--yes"])

        assert result.exit_code == 0, result.output
        assert [request.method for request in requests] == ["DELETE"]
