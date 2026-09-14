# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI tests for ``nemo agents deploy`` (wait-by-default) and ``nemo agents logs``.

Pin the user-visible contracts:

- ``deploy`` waits for a terminal deployment status by default and exits 1
  when the deployment fails, so the exit code reflects the actual outcome
  of the spawn instead of merely the API call.
- ``deploy --no-wait`` preserves the legacy fire-and-forget behaviour for
  scripted pipelines that prefer to poll separately.
- ``logs`` computes the absolute log path from the deployment name using
  the same convention the runner backend uses internally — no host-bound
  field is round-tripped through the public API.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
from nemo_agents_plugin.cli import _DEFAULT_WORKSPACE, AgentsCLI
from typer.testing import CliRunner


def _install_mock_transport(handler) -> AbstractContextManager[Any]:
    """Patch ``httpx.Client`` in the CLI module to use a ``MockTransport``."""
    transport = httpx.MockTransport(handler)
    real_client = httpx.Client

    class _Client(real_client):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    return patch("nemo_agents_plugin.cli.httpx.Client", _Client)


def _page(
    data: list[dict[str, Any]],
    *,
    page: int = 1,
    total_pages: int = 1,
    total_results: int | None = None,
) -> dict[str, Any]:
    total = len(data) if total_results is None else total_results
    return {
        "data": data,
        "pagination": {
            "page": page,
            "page_size": len(data),
            "current_page_size": len(data),
            "total_pages": total_pages,
            "total_results": total,
        },
    }


# ---------------------------------------------------------------------------
# deploy --wait (default) — exits 0 on running, 1 on failed
# ---------------------------------------------------------------------------


def test_deploy_default_waits_and_returns_success_on_running() -> None:
    """``deploy`` polls until status=running and exits 0."""
    statuses = iter(["pending", "starting", "running"])

    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "POST" and req.url.path.endswith("/deployments"):
            return httpx.Response(
                201,
                json={"name": "calc-abcd1234", "status": "pending", "agent": "calc"},
            )
        if req.method == "GET" and req.url.path.endswith("/deployments/calc-abcd1234"):
            status = next(statuses)
            return httpx.Response(
                200,
                json={
                    "name": "calc-abcd1234",
                    "status": status,
                    "endpoint": "http://127.0.0.1:49200" if status == "running" else "",
                },
            )
        return httpx.Response(404)

    app = AgentsCLI().get_cli()
    with _install_mock_transport(handler), patch("nemo_agents_plugin.cli.time.sleep"):
        result = CliRunner().invoke(
            app,
            ["deploy", "--agent", "calc", "--base-url", "http://test", "--timeout", "10"],
        )

    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    assert "is running" in result.stdout


def test_deploy_default_exits_failure_when_subprocess_dies() -> None:
    """Deploy exits 1 when the deployment reaches ``failed``.

    Before this fix the CLI would print the pending entity JSON and exit 0
    even when the subprocess immediately exited.  After: we wait for a
    terminal status and propagate failure as exit 1.
    """
    statuses = iter(["pending", "starting", "failed"])

    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "POST":
            return httpx.Response(
                201,
                json={"name": "calc-deadbeef", "status": "pending", "agent": "calc"},
            )
        if req.method == "GET":
            status = next(statuses)
            payload: dict[str, Any] = {
                "name": "calc-deadbeef",
                "status": status,
            }
            if status == "failed":
                payload["error"] = "Process exited with code 1"
            return httpx.Response(200, json=payload)
        return httpx.Response(404)

    app = AgentsCLI().get_cli()
    with _install_mock_transport(handler), patch("nemo_agents_plugin.cli.time.sleep"):
        result = CliRunner().invoke(
            app,
            ["deploy", "--agent", "calc", "--base-url", "http://test", "--timeout", "10"],
        )

    assert result.exit_code == 1, result.stdout
    # The error from the deployment entity is surfaced.
    assert "exited with code 1" in result.stdout
    assert "failed" in result.stdout


def test_deploy_polls_through_multiple_pending_responses() -> None:
    """Deploy keeps polling while status is non-terminal — even if the API
    initially returns ``pending`` repeatedly before the controller runs.

    Without this guarantee the wait loop could exit early on the first
    poll if it implicitly treated any non-running status as terminal.
    """
    # Five `pending`s then a single `running` — exercises the keep-polling path.
    statuses = iter(["pending", "pending", "pending", "pending", "pending", "running"])

    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "POST":
            return httpx.Response(201, json={"name": "slow-1", "status": "pending", "agent": "calc"})
        if req.method == "GET":
            return httpx.Response(
                200,
                json={
                    "name": "slow-1",
                    "status": next(statuses),
                    "endpoint": "http://127.0.0.1:49200",
                },
            )
        return httpx.Response(404)

    app = AgentsCLI().get_cli()
    with _install_mock_transport(handler), patch("nemo_agents_plugin.cli.time.sleep"):
        result = CliRunner().invoke(
            app,
            ["deploy", "--agent", "calc", "--base-url", "http://test", "--timeout", "60"],
        )

    assert result.exit_code == 0, result.stdout
    assert "is running" in result.stdout


def test_deploy_no_wait_returns_immediately_with_pending_json() -> None:
    """``--no-wait`` preserves the legacy behaviour: print JSON and exit 0."""
    posts: list[httpx.Request] = []
    gets: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "POST":
            posts.append(req)
            return httpx.Response(201, json={"name": "calc-abcd", "status": "pending", "agent": "calc"})
        gets.append(req)
        return httpx.Response(404)

    app = AgentsCLI().get_cli()
    with _install_mock_transport(handler):
        result = CliRunner().invoke(
            app,
            ["deploy", "--agent", "calc", "--no-wait", "--base-url", "http://test"],
        )

    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    assert "calc-abcd" in result.stdout
    # No GET polls should have happened — only the POST.
    assert len(posts) == 1
    assert gets == []


def test_deployments_wait_agent_resolves_latest_active_deployment_across_pages() -> None:
    """``deployments wait --agent`` fetches all pages and selects newest active deployment."""
    requests: list[httpx.Request] = []
    pages = [
        [
            {
                "name": "calc-new",
                "agent": "calc",
                "status": "pending",
                "created_at": "2026-05-18T12:00:00",
            },
        ],
        [
            {
                "name": "calc-old",
                "agent": "calc",
                "status": "pending",
                "created_at": "2026-05-17T12:00:00",
            },
        ],
    ]
    total_results = sum(len(page) for page in pages)

    def handler(req: httpx.Request) -> httpx.Response:
        requests.append(req)
        if req.method == "GET" and req.url.path.endswith("/deployments"):
            page_number = int(req.url.params.get("page", "1"))
            return httpx.Response(
                200,
                json=_page(
                    pages[page_number - 1],
                    page=page_number,
                    total_pages=len(pages),
                    total_results=total_results,
                ),
            )
        if req.method == "GET" and req.url.path.endswith("/deployments/calc-new"):
            return httpx.Response(
                200,
                json={
                    "name": "calc-new",
                    "agent": "calc",
                    "status": "running",
                    "created_at": "2026-05-18T12:00:00",
                },
            )
        return httpx.Response(404)

    app = AgentsCLI().get_cli()
    with _install_mock_transport(handler), patch("nemo_agents_plugin.cli.time.sleep"):
        result = CliRunner().invoke(
            app,
            ["deployments", "wait", "--agent", "calc", "--base-url", "http://test", "--timeout", "10"],
        )

    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    list_requests = [request for request in requests if request.url.path.endswith("/deployments")]
    assert len(list_requests) == 2
    assert list_requests[1].url.params["page"] == "2"
    assert requests[-1].url.path.endswith("/deployments/calc-new")


# ---------------------------------------------------------------------------
# logs subcommand — path derivation
# ---------------------------------------------------------------------------


def _make_log_for(workspace: str, name: str) -> Path:
    """Materialise a log file at the deterministic path the CLI will look up."""
    from nemo_agents_plugin.runner.in_memory import log_path_for_deployment

    path = log_path_for_deployment(workspace, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def test_logs_prints_file_contents_from_deterministic_path() -> None:
    """``nemo agents logs <name>`` reads the file at the conventional path."""
    log_file = _make_log_for(_DEFAULT_WORKSPACE, "calc-1")
    log_file.write_text("agent boot ok\nready on port 49200\n")

    def handler(req: httpx.Request) -> httpx.Response:
        # The CLI no longer fetches the deployment to learn the log path —
        # path is resolved client-side from the deployment name.
        return httpx.Response(404)

    app = AgentsCLI().get_cli()
    with _install_mock_transport(handler):
        result = CliRunner().invoke(app, ["logs", "calc-1", "--base-url", "http://test"])

    assert result.exit_code == 0, result.stderr or result.stdout
    assert "agent boot ok" in result.stdout
    assert "ready on port 49200" in result.stdout


def test_logs_path_only_prints_path_without_reading_file() -> None:
    """``--path`` prints the absolute path even if the file doesn't exist locally."""
    from nemo_agents_plugin.runner.in_memory import log_path_for_deployment

    expected_path = str(log_path_for_deployment(_DEFAULT_WORKSPACE, "calc-1"))

    app = AgentsCLI().get_cli()
    with _install_mock_transport(lambda r: httpx.Response(404)):
        result = CliRunner().invoke(app, ["logs", "calc-1", "--path", "--base-url", "http://test"])

    assert result.exit_code == 0, result.stderr or result.stdout
    assert expected_path in result.stdout


def test_logs_uses_workspace_to_separate_same_named_deployments() -> None:
    """The CLI's ``--workspace`` flag must drive the resolved log path.

    Two workspaces with deployment ``shared`` produce distinct log files
    (workspace-namespaced layout); ``nemo agents logs shared --workspace
    other`` should read ``other``'s log, not ``default``'s.
    """
    default_log = _make_log_for(_DEFAULT_WORKSPACE, "shared")
    default_log.write_text("from default workspace\n")
    other_log = _make_log_for("other-ws", "shared")
    other_log.write_text("from other workspace\n")

    app = AgentsCLI().get_cli()
    with _install_mock_transport(lambda r: httpx.Response(404)):
        default_result = CliRunner().invoke(app, ["logs", "shared", "--base-url", "http://test"])
        other_result = CliRunner().invoke(
            app, ["logs", "shared", "--workspace", "other-ws", "--base-url", "http://test"]
        )

    assert default_result.exit_code == 0
    assert "from default workspace" in default_result.stdout
    assert "from other workspace" not in default_result.stdout

    assert other_result.exit_code == 0
    assert "from other workspace" in other_result.stdout
    assert "from default workspace" not in other_result.stdout


def test_logs_reports_helpful_error_when_file_missing() -> None:
    """If the log file isn't on disk yet, exit 1 with a useful hint."""
    app = AgentsCLI().get_cli()
    with _install_mock_transport(lambda r: httpx.Response(404)):
        result = CliRunner().invoke(app, ["logs", "never-spawned", "--base-url", "http://test"])

    assert result.exit_code == 1
    assert "log file does not exist" in result.stderr
    assert "different host" in result.stderr  # part of the diagnostic hint


def test_logs_tail_prints_only_last_n_lines() -> None:
    """``--tail N`` prints only the last N lines."""
    log_file = _make_log_for(_DEFAULT_WORKSPACE, "calc-1")
    log_file.write_text("\n".join(f"line-{i}" for i in range(20)) + "\n")

    app = AgentsCLI().get_cli()
    with _install_mock_transport(lambda r: httpx.Response(404)):
        result = CliRunner().invoke(app, ["logs", "calc-1", "--tail", "3", "--base-url", "http://test"])

    assert result.exit_code == 0, result.stderr or result.stdout
    assert "line-19" in result.stdout
    assert "line-17" in result.stdout
    # Earlier lines are not included.
    assert "line-0" not in result.stdout


def test_logs_tail_rejects_non_positive_values() -> None:
    """Zero or negative ``--tail`` is a usage error — fail fast instead of
    silently printing the full log."""
    app = AgentsCLI().get_cli()
    for value in ("0", "-1"):
        with _install_mock_transport(lambda r: httpx.Response(404)):
            result = CliRunner().invoke(app, ["logs", "calc-1", "--tail", value, "--base-url", "http://test"])

        assert result.exit_code == 1, f"--tail {value} should reject"
        assert "positive" in result.stderr


def test_logs_resolves_most_recent_deployment_for_agent() -> None:
    """``--agent`` picks the deployment with the latest ``created_at``,
    not just the last list element.

    Without sorting, an API change to the default deployments-list ordering
    would silently make ``--agent`` pick the wrong deployment.
    """
    log_file = _make_log_for(_DEFAULT_WORKSPACE, "calc-2")
    log_file.write_text("calc-2 ok\n")

    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "GET" and req.url.path.endswith("/deployments"):
            # Return them in NON-creation order to confirm the CLI sorts.
            return httpx.Response(
                200,
                json=_page(
                    [
                        {
                            "name": "calc-2",
                            "agent": "calc",
                            "status": "running",
                            "created_at": "2026-05-18T12:00:00",
                        },
                        {
                            "name": "calc-1",
                            "agent": "calc",
                            "status": "failed",
                            "created_at": "2026-05-17T08:00:00",
                        },
                        {
                            "name": "other-1",
                            "agent": "other",
                            "status": "running",
                            "created_at": "2026-05-18T13:00:00",
                        },
                    ]
                ),
            )
        return httpx.Response(404)

    app = AgentsCLI().get_cli()
    with _install_mock_transport(handler):
        result = CliRunner().invoke(app, ["logs", "--agent", "calc", "--base-url", "http://test"])

    assert result.exit_code == 0, result.stderr or result.stdout
    assert "calc-2 ok" in result.stdout


def test_logs_agent_resolution_fetches_all_deployment_pages() -> None:
    """``logs --agent`` considers matching deployments beyond the first page."""
    log_file = _make_log_for(_DEFAULT_WORKSPACE, "calc-2")
    log_file.write_text("calc-2 from page 2\n")
    requests: list[httpx.Request] = []
    pages = [
        [
            {
                "name": "other-1",
                "agent": "other",
                "status": "running",
                "created_at": "2026-05-18T13:00:00",
            },
        ],
        [
            {
                "name": "calc-2",
                "agent": "calc",
                "status": "running",
                "created_at": "2026-05-18T12:00:00",
            },
        ],
    ]
    total_results = sum(len(page) for page in pages)

    def handler(req: httpx.Request) -> httpx.Response:
        requests.append(req)
        if req.method == "GET" and req.url.path.endswith("/deployments"):
            page_number = int(req.url.params.get("page", "1"))
            return httpx.Response(
                200,
                json=_page(
                    pages[page_number - 1],
                    page=page_number,
                    total_pages=len(pages),
                    total_results=total_results,
                ),
            )
        return httpx.Response(404)

    app = AgentsCLI().get_cli()
    with _install_mock_transport(handler):
        result = CliRunner().invoke(app, ["logs", "--agent", "calc", "--base-url", "http://test"])

    assert result.exit_code == 0, result.stderr or result.stdout
    assert "calc-2 from page 2" in result.stdout
    assert len(requests) == 2
    assert requests[1].url.params["page"] == "2"


def test_logs_requires_name_or_agent() -> None:
    """Calling ``logs`` with neither argument exits 1 with a usage error."""
    app = AgentsCLI().get_cli()
    result = CliRunner().invoke(app, ["logs", "--base-url", "http://test"])

    assert result.exit_code == 1
    assert "deployment name or --agent" in result.stderr
