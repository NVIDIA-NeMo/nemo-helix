# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI contract tests for ``nemo agents chat``."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from nemo_agents_plugin.cli import AgentsCLI, _run_resolved_session_chat
from nemo_agents_plugin.entities import (
    NAT_WORKFLOW_CONFIG_FORMAT,
    NEMO_AGENTS_SPEC_CONFIG_FORMAT,
    AgentDeployment,
    AgentSession,
    DeploymentStatus,
    SessionStatus,
)
from nemo_agents_plugin.session_protocol import SESSION_ID_HEADER
from nemo_platform_ext.cli.chat_tui import ExitAction, collect_stream_response
from typer.testing import CliRunner

runner = CliRunner()


def _deployment_response(
    *,
    name: str = "fabric-deployment",
    workspace: str = "default",
    config_format: str = NEMO_AGENTS_SPEC_CONFIG_FORMAT,
    status: DeploymentStatus = "running",
    endpoint: str = "http://127.0.0.1:9001",
    deployment_id: str = "deployment-id",
) -> dict[str, Any]:
    response = AgentDeployment(
        name=name,
        workspace=workspace,
        agent="fabric-agent",
        config={"config_format": config_format},
        status=status,
        endpoint=endpoint,
    ).model_dump(mode="json")
    response["id"] = deployment_id
    return response


def _session_response(
    *,
    name: str = "debug-auth",
    workspace: str = "default",
    deployment_id: str = "deployment-id",
    status: SessionStatus = SessionStatus.ACTIVE,
    expires_at: datetime | None = None,
) -> dict[str, Any]:
    response = AgentSession(
        name=name,
        workspace=workspace,
        deployment_id=deployment_id,
        status=status,
        expires_at=expires_at,
    ).model_dump(mode="json")
    response["id"] = "session-id"
    return response


def _deployment_page(
    *deployments: dict[str, Any],
    page: int = 1,
    total_pages: int = 1,
) -> dict[str, Any]:
    return {
        "data": list(deployments),
        "pagination": {
            "page": page,
            "page_size": 100,
            "current_page_size": len(deployments),
            "total_pages": total_pages,
            "total_results": len(deployments),
        },
    }


def _install_mock_transport(handler):
    transport = httpx.MockTransport(handler)
    real_client = httpx.Client

    class _Client(real_client):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    return patch("nemo_agents_plugin.cli.httpx.Client", _Client)


def _scripted_responses(*responses: dict[str, Any]):
    requests: list[httpx.Request] = []
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if not queue:
            raise AssertionError(f"unexpected request: {request.method} {request.url}")
        response = queue.pop(0)
        status_code = 201 if request.method == "POST" else 200
        return httpx.Response(status_code, request=request, json=response)

    return _install_mock_transport(handler), requests


def test_session_chat_help_describes_new_and_resumed_sessions() -> None:
    result = runner.invoke(AgentsCLI().get_cli(), ["chat", "--help"])

    assert result.exit_code == 0
    assert "Chat interactively with a new or existing deployed-agent session" in result.stdout
    assert "--input" in result.stdout
    assert "--agent-deployment" in result.stdout
    assert "--session" in result.stdout
    assert "--session-name" in result.stdout
    assert "--workspace" in result.stdout
    assert "--base-url" in result.stdout
    assert "--timeout" in result.stdout
    assert "--interactive" not in result.stdout


def test_session_chat_accepts_new_named_session_options() -> None:
    app = AgentsCLI().get_cli()
    with (
        patch("nemo_agents_plugin.cli._is_interactive_session_chat", return_value=True),
        patch("nemo_agents_plugin.cli._platform_session_chat") as session_chat,
    ):
        result = runner.invoke(
            app,
            [
                "chat",
                "--agent-deployment",
                "fabric-deployment",
                "--session-name",
                "debug-auth",
                "--input",
                "Help me debug this",
                "--workspace",
                "team-a",
                "--base-url",
                "http://platform.test",
                "--timeout",
                "42",
            ],
        )

    assert result.exit_code == 0, result.stderr
    session_chat.assert_called_once_with(
        base_url="http://platform.test",
        workspace="team-a",
        agent_deployment="fabric-deployment",
        session=None,
        session_name="debug-auth",
        input="Help me debug this",
        timeout=42.0,
    )


def test_session_chat_accepts_existing_session_without_initial_input() -> None:
    app = AgentsCLI().get_cli()
    with (
        patch("nemo_agents_plugin.cli._is_interactive_session_chat", return_value=True),
        patch("nemo_agents_plugin.cli._platform_session_chat") as session_chat,
    ):
        result = runner.invoke(app, ["chat", "--session", "debug-auth"])

    assert result.exit_code == 0, result.stderr
    session_chat.assert_called_once_with(
        base_url="http://localhost:8080",
        workspace="default",
        agent_deployment=None,
        session="debug-auth",
        session_name=None,
        input=None,
        timeout=300.0,
    )


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ([], "Provide exactly one of --agent-deployment or --session"),
        (
            ["--agent-deployment", "fabric-deployment", "--session", "debug-auth"],
            "Provide exactly one of --agent-deployment or --session",
        ),
        (
            ["--session", "debug-auth", "--session-name", "replacement"],
            "--session-name can only be used with --agent-deployment",
        ),
        (["--agent-deployment", ""], "--agent-deployment must not be empty"),
        (["--session", ""], "--session must not be empty"),
        (["--agent-deployment", "fabric-deployment", "--session-name", ""], "--session-name must not be empty"),
        (["--agent-deployment", "fabric-deployment", "--input", ""], "--input must not be empty"),
    ],
)
def test_session_chat_rejects_invalid_selector_combinations(arguments: list[str], message: str) -> None:
    with patch("nemo_agents_plugin.cli._platform_session_chat") as session_chat:
        result = runner.invoke(AgentsCLI().get_cli(), ["chat", *arguments])

    assert result.exit_code == 2
    assert message in result.stderr
    session_chat.assert_not_called()


def test_session_chat_requires_an_interactive_terminal() -> None:
    with (
        patch("nemo_agents_plugin.cli._is_interactive_session_chat", return_value=False),
        patch("nemo_agents_plugin.cli._platform_session_chat") as session_chat,
    ):
        result = runner.invoke(AgentsCLI().get_cli(), ["chat", "--session", "debug-auth"])

    assert result.exit_code == 2
    assert "Agent chat requires an interactive terminal" in result.stderr
    assert "nemo agents invoke" in result.stderr
    session_chat.assert_not_called()


def test_session_chat_creates_named_session_and_passes_values_to_transport() -> None:
    app = AgentsCLI().get_cli()
    transport, requests = _scripted_responses(_deployment_response(), _session_response())
    with (
        patch("nemo_agents_plugin.cli._is_interactive_session_chat", return_value=True),
        transport,
        patch("nemo_agents_plugin.cli._run_resolved_session_chat") as run_chat,
    ):
        result = runner.invoke(
            app,
            [
                "chat",
                "--agent-deployment",
                "fabric-deployment",
                "--session-name",
                "debug-auth",
                "--input",
                "Help me debug this",
                "--timeout",
                "42",
            ],
        )

    assert result.exit_code == 0, result.stderr
    assert (
        "Session 'debug-auth' created. Resume with:\n"
        "  nemo agents chat --session debug-auth --workspace default --base-url http://localhost:8080" in result.stdout
    )
    assert [(request.method, request.url.path) for request in requests] == [
        ("GET", "/apis/agents/v2/workspaces/default/deployments/fabric-deployment"),
        ("POST", "/apis/agents/v2/workspaces/default/sessions"),
    ]
    assert json.loads(requests[1].content) == {"deployment_id": "deployment-id", "name": "debug-auth"}
    run_chat.assert_called_once()
    run_kwargs = run_chat.call_args.kwargs
    assert run_kwargs["base_url"] == "http://localhost:8080"
    assert run_kwargs["workspace"] == "default"
    assert run_kwargs["input"] == "Help me debug this"
    assert run_kwargs["timeout"] == 42.0
    assert run_kwargs["deployment"].name == "fabric-deployment"
    assert run_kwargs["session"].name == "debug-auth"
    assert run_kwargs["session_id"] == "session-id"


def test_session_chat_preserves_api_generated_session_name() -> None:
    generated_name = "fabric-deployment-a1b2c3d4"
    transport, requests = _scripted_responses(_deployment_response(), _session_response(name=generated_name))
    with (
        patch("nemo_agents_plugin.cli._is_interactive_session_chat", return_value=True),
        transport,
        patch("nemo_agents_plugin.cli._run_resolved_session_chat") as run_chat,
    ):
        result = runner.invoke(
            AgentsCLI().get_cli(),
            ["chat", "--agent-deployment", "fabric-deployment"],
        )

    assert result.exit_code == 0, result.stderr
    assert requests[1].method == "POST"
    assert requests[1].url.path == "/apis/agents/v2/workspaces/default/sessions"
    assert json.loads(requests[1].content) == {"deployment_id": "deployment-id"}
    assert run_chat.call_args.kwargs["session"].name == generated_name
    assert run_chat.call_args.kwargs["session_id"] == "session-id"


def test_session_chat_resumes_active_session_and_passes_values_to_transport() -> None:
    transport, requests = _scripted_responses(_session_response(), _deployment_page(_deployment_response()))
    with (
        patch("nemo_agents_plugin.cli._is_interactive_session_chat", return_value=True),
        transport,
        patch("nemo_agents_plugin.cli._run_resolved_session_chat") as run_chat,
    ):
        result = runner.invoke(
            AgentsCLI().get_cli(),
            ["chat", "--session", "debug-auth", "--input", "Continue debugging", "--timeout", "42"],
        )

    assert result.exit_code == 0, result.stderr
    assert "Resuming runtime context; prior messages are not redisplayed." in result.stdout
    assert [(request.method, request.url.path) for request in requests] == [
        ("GET", "/apis/agents/v2/workspaces/default/sessions/debug-auth"),
        ("GET", "/apis/agents/v2/workspaces/default/deployments"),
    ]
    assert requests[1].url.params["page"] == "1"
    assert requests[1].url.params["page_size"] == "100"
    run_chat.assert_called_once()
    run_kwargs = run_chat.call_args.kwargs
    assert run_kwargs["base_url"] == "http://localhost:8080"
    assert run_kwargs["workspace"] == "default"
    assert run_kwargs["input"] == "Continue debugging"
    assert run_kwargs["timeout"] == 42.0
    assert run_kwargs["deployment"].name == "fabric-deployment"
    assert run_kwargs["session"].name == "debug-auth"
    assert run_kwargs["session_id"] == "session-id"


def test_session_chat_pages_until_it_finds_the_session_deployment() -> None:
    target_id = "target-deployment-id"
    transport, requests = _scripted_responses(
        _session_response(deployment_id=target_id),
        _deployment_page(
            _deployment_response(name="unrelated", deployment_id="unrelated-id"),
            page=1,
            total_pages=2,
        ),
        _deployment_page(
            _deployment_response(deployment_id=target_id),
            page=2,
            total_pages=2,
        ),
    )
    with (
        patch("nemo_agents_plugin.cli._is_interactive_session_chat", return_value=True),
        transport,
        patch("nemo_agents_plugin.cli._run_resolved_session_chat") as run_chat,
    ):
        result = runner.invoke(AgentsCLI().get_cli(), ["chat", "--session", "debug-auth"])

    assert result.exit_code == 0, result.stderr
    assert [request.url.params["page"] for request in requests[-2:]] == ["1", "2"]
    assert run_chat.call_args.kwargs["deployment"].name == "fabric-deployment"
    assert run_chat.call_args.kwargs["session_id"] == "session-id"


@pytest.mark.parametrize("status", [SessionStatus.CLOSED, SessionStatus.EXPIRED, SessionStatus.LOST])
def test_session_chat_rejects_terminal_session_before_deployment_lookup(status: SessionStatus) -> None:
    transport, requests = _scripted_responses(_session_response(status=status))
    with (
        patch("nemo_agents_plugin.cli._is_interactive_session_chat", return_value=True),
        transport,
        patch("nemo_agents_plugin.cli._run_resolved_session_chat") as run_chat,
    ):
        result = runner.invoke(AgentsCLI().get_cli(), ["chat", "--session", "debug-auth"])

    assert result.exit_code == 1
    assert f"is {status.value} and cannot be resumed" in result.stderr
    assert len(requests) == 1
    run_chat.assert_not_called()


def test_session_chat_reports_missing_session() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, request=request, json={"detail": "Session 'missing' not found."})

    real_client = httpx.Client
    with (
        patch("nemo_agents_plugin.cli._is_interactive_session_chat", return_value=True),
        patch(
            "nemo_agents_plugin.cli.httpx.Client",
            side_effect=lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
        ),
        patch("nemo_agents_plugin.cli._run_resolved_session_chat") as run_chat,
    ):
        result = runner.invoke(AgentsCLI().get_cli(), ["chat", "--session", "missing"])

    assert result.exit_code == 1
    assert "Session 'missing' not found. (HTTP 404 Not Found)" in result.stderr
    assert "/apis/agents/v2/workspaces/default/sessions/missing" in result.stderr
    run_chat.assert_not_called()


def test_session_chat_rejects_active_session_past_its_expiration_deadline() -> None:
    transport, requests = _scripted_responses(_session_response(expires_at=datetime(2000, 1, 1, tzinfo=UTC)))
    with (
        patch("nemo_agents_plugin.cli._is_interactive_session_chat", return_value=True),
        transport,
        patch("nemo_agents_plugin.cli._run_resolved_session_chat") as run_chat,
    ):
        result = runner.invoke(AgentsCLI().get_cli(), ["chat", "--session", "debug-auth"])

    assert result.exit_code == 1
    assert "is expired and cannot be resumed" in result.stderr
    assert len(requests) == 1
    run_chat.assert_not_called()


def test_session_chat_rejects_session_whose_deployment_no_longer_exists() -> None:
    transport, _requests = _scripted_responses(_session_response(), _deployment_page())
    with (
        patch("nemo_agents_plugin.cli._is_interactive_session_chat", return_value=True),
        transport,
        patch("nemo_agents_plugin.cli._run_resolved_session_chat") as run_chat,
    ):
        result = runner.invoke(AgentsCLI().get_cli(), ["chat", "--session", "debug-auth"])

    assert result.exit_code == 1
    assert "references deployment ID 'deployment-id'" in result.stderr
    assert "no matching deployment exists" in result.stderr
    run_chat.assert_not_called()


def test_session_chat_rejects_invalid_session_lookup_response() -> None:
    transport, _requests = _scripted_responses({"name": "debug-auth", "deployment_id": "deployment-id"})
    with (
        patch("nemo_agents_plugin.cli._is_interactive_session_chat", return_value=True),
        transport,
        patch("nemo_agents_plugin.cli._run_resolved_session_chat") as run_chat,
    ):
        result = runner.invoke(AgentsCLI().get_cli(), ["chat", "--session", "debug-auth"])

    assert result.exit_code == 1
    assert "returned an invalid response" in result.stderr
    run_chat.assert_not_called()


def test_resolved_session_chat_streams_each_current_turn_with_session_and_auth_headers() -> None:
    requests: list[httpx.Request] = []
    expires_at = datetime(2026, 9, 1, 18, 30, tzinfo=UTC)

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        user_input = json.loads(request.content)["messages"][0]["content"]
        chunk = {"choices": [{"delta": {"content": f"reply:{user_input}"}}]}
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode(),
        )

    real_client = httpx.Client
    transport = httpx.MockTransport(handler)

    def exercise_tui(**kwargs: Any) -> None:
        assert kwargs["display_info"] == {
            "Deployment": "fabric-deployment",
            "Session": "debug-auth",
            "Status": "active",
            "Expires": expires_at.isoformat(),
        }
        assert kwargs["initial_message"] == "first turn"
        assert kwargs["exit_action"] is ExitAction.DETACH
        for user_input in (kwargs["initial_message"], "second turn"):
            response_text, usage = collect_stream_response(kwargs["send_turn"](user_input))
            assert response_text == f"reply:{user_input}"
            assert usage is None

    with (
        patch("nemo_agents_plugin.cli._resolve_context_headers", return_value={"Authorization": "Bearer token"}),
        patch(
            "nemo_agents_plugin.cli.httpx.Client",
            side_effect=lambda **kwargs: real_client(transport=transport, **kwargs),
        ),
        patch("nemo_agents_plugin.cli.run_chat_tui", side_effect=exercise_tui),
    ):
        _run_resolved_session_chat(
            base_url="http://platform.test/",
            workspace="team-a",
            deployment=AgentDeployment.model_validate(_deployment_response()),
            session=AgentSession.model_validate(_session_response(expires_at=expires_at)),
            session_id="session-id",
            input="first turn",
            timeout=42,
        )

    expected_path = "/apis/agents/v2/workspaces/team-a/deployments/fabric-deployment/-/v1/chat/completions"
    assert [request.url.path for request in requests] == [expected_path, expected_path]
    assert [json.loads(request.content) for request in requests] == [
        {"messages": [{"role": "user", "content": "first turn"}], "stream": True},
        {"messages": [{"role": "user", "content": "second turn"}], "stream": True},
    ]
    assert all(request.headers[SESSION_ID_HEADER] == "session-id" for request in requests)
    assert all(request.headers["Authorization"] == "Bearer token" for request in requests)


@pytest.mark.parametrize("exception", [KeyboardInterrupt, EOFError])
def test_session_chat_interrupt_detaches_without_closing(exception: type[BaseException]) -> None:
    transport, requests = _scripted_responses(_session_response(), _deployment_page(_deployment_response()))
    with (
        patch("nemo_agents_plugin.cli._is_interactive_session_chat", return_value=True),
        transport,
        patch("nemo_platform_ext.cli.chat_tui.Prompt.ask", side_effect=exception),
    ):
        result = runner.invoke(AgentsCLI().get_cli(), ["chat", "--session", "debug-auth"])

    assert result.exit_code == 0, result.stderr
    assert "Session detached" in result.stdout
    assert [(request.method, request.url.path) for request in requests] == [
        ("GET", "/apis/agents/v2/workspaces/default/sessions/debug-auth"),
        ("GET", "/apis/agents/v2/workspaces/default/deployments"),
    ]


def test_session_chat_interrupt_during_initial_turn_detaches_without_closing() -> None:
    requests: list[httpx.Request] = []
    queue = [_session_response(), _deployment_page(_deployment_response())]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/-/v1/chat/completions"):
            raise KeyboardInterrupt
        if not queue:
            raise AssertionError(f"unexpected request: {request.method} {request.url}")
        return httpx.Response(200, request=request, json=queue.pop(0))

    with (
        patch("nemo_agents_plugin.cli._is_interactive_session_chat", return_value=True),
        _install_mock_transport(handler),
    ):
        result = runner.invoke(
            AgentsCLI().get_cli(),
            ["chat", "--session", "debug-auth", "--input", "first turn"],
        )

    assert result.exit_code == 0, result.stderr
    assert "Session detached" in result.stdout
    assert [request.method for request in requests[:2]] == ["GET", "GET"]


def test_session_chat_rejects_non_fabric_deployment_before_session_creation() -> None:
    transport, requests = _scripted_responses(_deployment_response(config_format=NAT_WORKFLOW_CONFIG_FORMAT))
    with (
        patch("nemo_agents_plugin.cli._is_interactive_session_chat", return_value=True),
        transport,
        patch("nemo_agents_plugin.cli._run_resolved_session_chat") as run_chat,
    ):
        result = runner.invoke(
            AgentsCLI().get_cli(),
            ["chat", "--agent-deployment", "fabric-deployment"],
        )

    assert result.exit_code == 1
    assert "is not Fabric-backed" in result.stderr
    assert len(requests) == 1
    run_chat.assert_not_called()


@pytest.mark.parametrize(
    ("status", "endpoint"),
    [
        ("starting", "http://127.0.0.1:9001"),
        ("running", ""),
    ],
)
def test_session_chat_rejects_unroutable_deployment_before_session_creation(
    status: DeploymentStatus,
    endpoint: str,
) -> None:
    transport, requests = _scripted_responses(_deployment_response(status=status, endpoint=endpoint))
    with (
        patch("nemo_agents_plugin.cli._is_interactive_session_chat", return_value=True),
        transport,
        patch("nemo_agents_plugin.cli._run_resolved_session_chat") as run_chat,
    ):
        result = runner.invoke(
            AgentsCLI().get_cli(),
            ["chat", "--agent-deployment", "fabric-deployment"],
        )

    assert result.exit_code == 1
    assert "is not routable" in result.stderr
    assert len(requests) == 1
    run_chat.assert_not_called()


def test_session_chat_rejects_invalid_session_creation_response() -> None:
    transport, _requests = _scripted_responses(
        _deployment_response(),
        {"name": "debug-auth", "deployment_id": "deployment-id"},
    )
    with (
        patch("nemo_agents_plugin.cli._is_interactive_session_chat", return_value=True),
        transport,
        patch("nemo_agents_plugin.cli._run_resolved_session_chat") as run_chat,
    ):
        result = runner.invoke(
            AgentsCLI().get_cli(),
            [
                "chat",
                "--agent-deployment",
                "fabric-deployment",
                "--session-name",
                "debug-auth",
            ],
        )

    assert result.exit_code == 1
    assert "returned an invalid response" in result.stderr
    run_chat.assert_not_called()
