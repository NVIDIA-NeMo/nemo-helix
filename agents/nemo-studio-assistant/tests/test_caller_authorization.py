# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline authorization regressions: no model, platform, or listening sockets."""

import asyncio
import json
import os
import ssl
from types import SimpleNamespace

import httpx
import pytest
from fastmcp.server.middleware import MiddlewareContext
from nemo_studio_assistant import authorization, mcp_server, register
from nemo_studio_assistant.mcp_server import CallerAuthorityMiddleware

SESSION = "90a877d5-19f6-49a8-bf09-d0020ae0833a"


def caller(identity="reader", workspace="default", studio=True):
    return {
        "workspace": workspace,
        "headers": {"authorization": f"Bearer {identity}"} if identity else {},
        "studio_session": SESSION if studio else None,
        "studio_capability": "private-callback-capability" if studio else None,
    }


@pytest.fixture
def platform(monkeypatch):
    requests = []
    monkeypatch.setenv("NHX_BASE_URL", "https://platform:8080")
    monkeypatch.setenv("NHX_ACCESS_TOKEN", "privileged-process-token")
    monkeypatch.setattr(register, "_api_error_streaks", {})

    def handle(request):
        requests.append(request)
        # Consent never grants a reader write privileges.
        if request.method != "GET" and request.headers.get("authorization") != "Bearer writer":
            return httpx.Response(403, json={"detail": "caller not authorized"})
        if request.method == "POST":
            return httpx.Response(
                201,
                json={
                    "id": "workspace-id",
                    "name": "demo",
                    "description": "demo",
                    "created_at": "2026-01-01T00:00:00Z",
                    "updated_at": "2026-01-01T00:00:00Z",
                },
            )
        return httpx.Response(
            200,
            json={
                "data": [],
                "pagination": {
                    "page": 1,
                    "page_size": 10,
                    "current_page_size": 0,
                    "total_pages": 1,
                    "total_results": 0,
                },
            },
        )

    monkeypatch.setattr(
        authorization, "Client", lambda **kwargs: httpx.Client(**kwargs, transport=httpx.MockTransport(handle))
    )
    return requests


def test_missing_caller_fails_closed_with_privileged_process_config(platform, monkeypatch, tmp_path):
    config = tmp_path / "privileged-config.yaml"
    config.write_text("not even a valid config: must never be consulted")
    monkeypatch.setenv("NHX_CONFIG_FILE", str(config))
    assert "Missing trusted caller context" in register.nemo_api("models", "list", workspace="default")
    assert "Missing trusted caller context" in register.check_status("evals", "job", workspace="default")
    with pytest.raises(PermissionError, match="Missing trusted caller"):
        register._get_client("default")
    assert platform == []


@pytest.mark.parametrize("studio", [True, False], ids=["studio", "direct-gateway"])
@pytest.mark.parametrize("action", ["list", "create"])
def test_cross_workspace_tools_denied_before_approval(platform, monkeypatch, studio, action):
    monkeypatch.setattr(register, "_request_mutation_approval", lambda *a, **k: pytest.fail("must not request consent"))
    with authorization.tool_authority(caller(studio=studio)):
        result = register.nemo_api("models", action, workspace="other")
        assert "invocation workspace" in result
        assert "invocation workspace" in register.check_status("evals", "job", workspace="other")
        with pytest.raises(PermissionError, match="invocation workspace"):
            register.deploy_guardrail(
                policy="policy",
                config_name="test",
                virtual_model_name="test",
                blocked_message="blocked",
                allowed_message="allowed",
                studio_session_id=SESSION,
                deployment_run_id="2a37cb64-6e28-4674-94a7-a122cdf0d08f",
                workspace="other",
            )
    assert platform == []


@pytest.mark.parametrize("params", [{"workspace": "other"}, {"name": "other"}])
def test_parameter_workspace_overrides_are_denied(platform, params):
    with authorization.tool_authority(caller()):
        result = register.nemo_api("workspaces", "retrieve", params=params, workspace="default")
        assert "PermissionError" in result
    assert platform == []


@pytest.mark.parametrize(
    "path",
    [
        "/apis/models/v2/workspaces/other/models",
        "/apis/models/v2/workspaces/%6fther/models",
        "/apis/models/v2/workspaces/default/../../other/models",
        "/apis/auth/v1/admin",
        "https://attacker.invalid/apis/models/v2/workspaces/default/models",
    ],
)
def test_final_http_target_is_bound_not_just_tool_workspace(platform, path):
    with authorization.tool_authority(caller()):
        client = register._get_client("default")
        with pytest.raises(PermissionError):
            client.http_client.get(path if path.startswith("https:") else f"https://platform:8080{path}")
    assert platform == []


def test_query_workspace_override_is_denied(platform):
    with authorization.tool_authority(caller()):
        client = register._get_client("default")
        with pytest.raises(PermissionError, match="query"):
            client.http_client.get("https://platform:8080/apis/models/v2/workspaces/default/models?workspace=other")
    assert platform == []


def test_self_approval_does_not_override_caller_authorization(platform, monkeypatch):
    monkeypatch.setattr(register, "_call_studio_tool", lambda *a, **k: {"behavior": "allow"})
    with authorization.tool_authority(caller()):
        result = register.nemo_api("workspaces", "create", {"name": "demo"}, SESSION, "default")
    assert "403" in result
    assert len(platform) == 1
    assert platform[0].headers["authorization"] == "Bearer reader"


def test_authorized_writer_with_consent_succeeds(platform, monkeypatch):
    monkeypatch.setattr(register, "_call_studio_tool", lambda *a, **k: {"behavior": "allow"})
    with authorization.tool_authority(caller("writer")):
        result = register.nemo_api("workspaces", "create", {"name": "demo"}, SESSION, "default")
    assert json.loads(result)["name"] == "demo"
    assert platform[0].headers["authorization"] == "Bearer writer"


@pytest.mark.parametrize("studio", [True, False])
def test_legitimate_reads_and_no_auth_remain_supported(platform, studio):
    for identity in ("reader", None):
        with authorization.tool_authority(caller(identity, studio=studio)):
            assert json.loads(register.nemo_api("models", "list", workspace="default"))["data"] == []
    assert platform[0].headers["authorization"] == "Bearer reader"
    assert "authorization" not in platform[1].headers


@pytest.mark.parametrize("headers", [{"authorization": "Bearer reader"}, {"cookie": "reader=session"}])
def test_caller_secrets_are_not_sent_to_cleartext_remote_platform(platform, monkeypatch, headers):
    monkeypatch.setenv("NHX_BASE_URL", "http://platform:8080")
    with authorization.tool_authority({"workspace": "default", "headers": headers}):
        with pytest.raises(PermissionError, match="HTTPS"):
            register._get_client("default")
    assert platform == []


def test_platform_url_cannot_supply_privileged_ambient_credentials(platform, monkeypatch):
    monkeypatch.setenv(
        "NHX_BASE_URL",
        str(httpx.URL("https://platform:8080").copy_with(username="test-user", password="test-password")),
    )
    with authorization.tool_authority(caller(None)):
        with pytest.raises(PermissionError, match="ambient credentials"):
            register._get_client("default")
    assert platform == []


def test_approved_input_cannot_override_target_workspace(platform, monkeypatch):
    monkeypatch.setattr(
        register,
        "_call_studio_tool",
        lambda *a, **k: {
            "behavior": "allow",
            "updatedInput": {"params": {"name": "other"}},
        },
    )
    with authorization.tool_authority(caller("writer")):
        result = register.nemo_api("workspaces", "delete", {"name": "default"}, SESSION, "default")
    assert "PermissionError" in result
    assert platform == []


def test_redirect_does_not_forward_caller_credentials(platform, monkeypatch):
    def redirect(request):
        platform.append(request)
        return httpx.Response(302, headers={"location": "https://attacker.invalid/steal"})

    monkeypatch.setattr(
        authorization, "Client", lambda **kwargs: httpx.Client(**kwargs, transport=httpx.MockTransport(redirect))
    )
    with authorization.tool_authority(caller()):
        result = register.nemo_api("models", "list", workspace="default")
    assert "Error" in result
    assert len(platform) == 1 and platform[0].url.host == "platform"


@pytest.mark.parametrize(
    ("url", "opt_in", "identity", "allowed"),
    [
        ("https://platform:8080", False, "reader", True),
        ("http://127.0.0.1:8080", False, None, True),
        ("http://platform:8080", True, None, False),
        ("http://host.docker.internal:8080", False, None, False),
        ("http://host.docker.internal:8080", True, None, True),
        ("http://host.docker.internal.evil:8080", True, None, False),
        ("http://host.docker.internal:8080", True, "reader", False),
        (
            str(httpx.URL("https://platform").copy_with(username="test-user", password="test-password")),
            True,
            None,
            False,
        ),
        ("ftp://127.0.0.1", True, None, False),
    ],
)
def test_callback_capability_transport_policy(platform, monkeypatch, url, opt_in, identity, allowed):
    monkeypatch.setenv("NHX_BASE_URL", url)
    monkeypatch.setenv("NHX_STUDIO_LOCAL_HTTP", "1" if opt_in else "0")

    def callback(request):
        platform.append(request)
        assert request.headers["x-nhx-studio-capability"] == "private-callback-capability"
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "result": {"content": [{"type": "text", "text": "{}"}]},
            },
        )

    monkeypatch.setattr(
        register, "Client", lambda **kwargs: httpx.Client(**kwargs, transport=httpx.MockTransport(callback))
    )
    with authorization.tool_authority(caller(identity)):
        if allowed:
            assert register._call_studio_tool(SESSION, "studio_link", {}) == {}
            assert len(platform) == 1
        else:
            with pytest.raises(PermissionError):
                register._call_studio_tool(SESSION, "studio_link", {})
            assert platform == []


def test_docker_callback_exception_never_applies_to_cookies(monkeypatch):
    monkeypatch.setenv("NHX_STUDIO_LOCAL_HTTP", "1")
    with authorization.tool_authority({"workspace": "default", "headers": {"cookie": "secret"}}) as current:
        with pytest.raises(PermissionError):
            authorization.check_callback_transport("http://host.docker.internal:8080", current)


def test_both_platform_clients_require_certificate_and_hostname_validation(platform, monkeypatch):
    contexts = []

    def client(**kwargs):
        context = kwargs["verify"]
        assert isinstance(context, ssl.SSLContext)
        assert context.verify_mode == ssl.CERT_REQUIRED
        assert context.check_hostname is True
        contexts.append(context)
        return httpx.Client(
            **kwargs,
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    json={"result": {"content": [{"type": "text", "text": "{}"}]}},
                )
            ),
        )

    monkeypatch.setattr(authorization, "Client", client)
    monkeypatch.setattr(register, "Client", client)
    with authorization.tool_authority(caller()):
        register._get_client("default")
        assert register._call_studio_tool(SESSION, "studio_link", {}) == {}
    assert len(contexts) == 2
    assert platform == []


@pytest.mark.parametrize("envvar", ["NHX_CLIENT_SSL_CERT_FILE", "SSL_CERT_FILE"])
@pytest.mark.parametrize("malformed", [False, True], ids=["missing", "invalid-pem"])
def test_invalid_platform_ca_fails_closed_for_caller_and_callback(platform, monkeypatch, tmp_path, envvar, malformed):
    monkeypatch.delenv("NHX_CLIENT_SSL_CERT_FILE", raising=False)
    ca_file = tmp_path / "invalid-ca.pem"
    if malformed:
        ca_file.write_text("not a certificate")
    monkeypatch.setenv(envvar, str(ca_file))
    with authorization.tool_authority(caller()):
        for operation in (
            lambda: register._get_client("default"),
            lambda: register._call_studio_tool(SESSION, "studio_link", {}),
        ):
            with pytest.raises((FileNotFoundError, ssl.SSLError)):
                operation()
    assert platform == []


def test_direct_gateway_cannot_invent_studio_session(platform):
    with authorization.tool_authority(caller(studio=False)):
        result = register.nemo_api("workspaces", "create", {"name": "demo"}, SESSION, "default")
    assert "invocation session" in result
    assert platform == []


async def test_concurrent_identities_have_distinct_clients_and_credentials(platform):
    async def run(identity):
        with authorization.tool_authority(caller(identity)):
            client = register._get_client("default")
            await asyncio.sleep(0)
            register.nemo_api("models", "list", workspace="default")
            assert register._get_client("default") is client
            return client

    first, second = await asyncio.gather(run("alice"), run("bob"))
    assert first is not second
    assert {request.headers["authorization"] for request in platform} == {"Bearer alice", "Bearer bob"}
    assert first.http_client.is_closed and second.http_client.is_closed
    with pytest.raises(PermissionError):
        authorization.authority()


async def test_mcp_redeems_capability_outside_tool_arguments_and_cleans_up(monkeypatch):
    original_client = httpx.AsyncClient
    seen = []

    def broker(request):
        seen.append(request)
        assert str(request.url) == "http://127.0.0.1:8123/_internal/studio-tool-authority"
        assert request.headers["authorization"] == "Bearer invocation-capability"
        return httpx.Response(200, json=caller())

    monkeypatch.setattr(
        mcp_server,
        "AsyncClient",
        lambda **kwargs: original_client(
            **kwargs,
            transport=httpx.MockTransport(broker),
        ),
    )
    meta = SimpleNamespace(nhx_tool_authority={"broker_port": 8123, "capability": "invocation-capability"})
    context = MiddlewareContext(message=SimpleNamespace(meta=meta))

    async def execute(_context):
        assert authorization.authority().headers == {"authorization": "Bearer reader"}
        return "success"

    assert await CallerAuthorityMiddleware().on_call_tool(context, execute) == "success"
    assert len(seen) == 1
    with pytest.raises(PermissionError):
        authorization.authority()


async def test_real_in_memory_mcp_protocol_redeems_request_scoped_broker_authority(platform, monkeypatch):
    from fastmcp import Client
    from nemo_agents_plugin.fabric.tool_authority import CONTEXT_KEY, ToolAuthorityBroker
    from nemo_studio_assistant.mcp_server import create_server
    from starlette.responses import JSONResponse

    original_client = httpx.AsyncClient
    results = []

    async def invocation(scope, receive, send):
        descriptor = scope["state"][CONTEXT_KEY]
        async with Client(create_server()) as mcp:
            result = await mcp.call_tool(
                "nemo_api",
                {"resource": "models", "action": "list", "workspace": "default"},
                meta={CONTEXT_KEY: descriptor},
            )
            results.append(result)
            denied = await mcp.call_tool(
                "nemo_api",
                {"resource": "models", "action": "list", "workspace": "other"},
                meta={CONTEXT_KEY: descriptor},
            )
            assert "invocation workspace" in str(denied.content)
        await JSONResponse({"ok": True})(scope, receive, send)

    broker = ToolAuthorityBroker(invocation, enabled=lambda: True)
    transport = httpx.ASGITransport(app=broker)
    monkeypatch.setattr(mcp_server, "AsyncClient", lambda **kwargs: original_client(**kwargs, transport=transport))
    async with original_client(transport=transport, base_url="http://127.0.0.1:8123") as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={
                "authorization": "Bearer reader",
                "x-nhx-tool-workspace": "default",
            },
        )
    assert response.status_code == 200
    assert len(results) == 1
    assert len(platform) == 1 and platform[0].headers["authorization"] == "Bearer reader"
    assert broker._active == {}


def test_fabric_bridge_overwrites_model_metadata_and_isolates_concurrent_turns(monkeypatch):
    import subprocess
    import sys
    from pathlib import Path

    # Pytest configures source imports through sys.path, not necessarily the
    # shell's PYTHONPATH. Reproduce that environment and propagate it explicitly.
    monkeypatch.delenv("PYTHONPATH", raising=False)
    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).resolve().parents[1] / "src"), *sys.path])
    result = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("fabric_authority_contract.py"))],
        env=child_env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "metadata", [None, {}, {"broker_port": 8080, "capability": ""}, {"broker_port": "8080", "capability": "x"}]
)
async def test_mcp_boundary_rejects_missing_or_invalid_broker_capability(metadata):
    message = SimpleNamespace(meta=SimpleNamespace(nhx_tool_authority=metadata))
    context = MiddlewareContext(message=message)

    async def execute(_context):
        pytest.fail("unauthenticated tool must not run")

    with pytest.raises(PermissionError):
        await CallerAuthorityMiddleware().on_call_tool(context, execute)
