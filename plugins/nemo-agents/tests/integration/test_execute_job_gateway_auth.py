# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real Fabric/Deep Agents execution against a local authenticated model stub."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from nemo_agents_plugin.jobs import gateway_proxy
from nemo_agents_plugin.jobs.execute import ExecuteAgentJob
from nemo_platform_plugin.client.oidc import WorkloadTokenExchangeProvider
from nemo_platform_plugin.config import Configuration
from nemo_platform_plugin.job_context import JobContext, StoragePaths
from nemo_platform_plugin.job_results import LocalJobResults
from pytest_httpserver import HTTPServer
from werkzeug.wrappers import Request, Response

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("identity_mode", ["workload", "principal", "anonymous"])
def test_real_agent_calls_gateway_and_returns_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, httpserver: HTTPServer, identity_mode: str
) -> None:
    monkeypatch.setenv("NMP_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("NMP_WORKLOAD_IDENTITY_TOKEN_FILE", raising=False)
    monkeypatch.delenv("NMP_PRINCIPAL", raising=False)
    monkeypatch.delenv("NMP_AUTH_ENABLED", raising=False)
    Configuration.clear_cache()
    proof = tmp_path / "proof"
    proof.write_text("test-job-subject-token")
    monkeypatch.setenv("NMP_BASE_URL", httpserver.url_for(""))
    provider = WorkloadTokenExchangeProvider(
        token_endpoint=httpserver.url_for("/token"),
        client_id="job",
        subject_token_file=proof,
        allow_http=True,
    )
    if identity_mode == "workload":
        monkeypatch.setenv("NMP_WORKLOAD_IDENTITY_TOKEN_FILE", str(proof))
        # Only discovery is replaced. Token exchange, expiry handling, the proxy,
        # Fabric, Deep Agents and its file tool all execute normally.
        monkeypatch.setattr(gateway_proxy, "resolve_workload_exchange_provider", lambda **kwargs: provider)
    elif identity_mode == "principal":
        monkeypatch.setenv("NMP_PRINCIPAL", json.dumps({"id": "test-job-user", "groups": ["test-workspace-users"]}))
    else:
        # Jobs also carry an anonymous principal when platform auth is disabled
        # in YAML rather than through NMP_AUTH_ENABLED.
        config_file = tmp_path / "config.yaml"
        config_file.write_text("auth:\n  enabled: false\n")
        monkeypatch.setenv("NMP_CONFIG_FILE_PATH", str(config_file))
        monkeypatch.setenv("NMP_PRINCIPAL", '{"id":"","groups":[]}')
    issued: list[str] = []
    seen: list[str] = []

    def exchange(request: Request) -> Response:
        assert request.form["subject_token"] == "test-job-subject-token"
        token = f"test-job-access-{len(issued) + 1}"
        issued.append(token)
        return Response(
            json.dumps({"access_token": token, "token_type": "Bearer", "expires_in": 300}),
            content_type="application/json",
        )

    def model(request: Request) -> Response:
        # Fabric sends a known-length JSON body. Preserve its framing: deployed
        # gateways may add Content-Length without removing Transfer-Encoding.
        if request.headers.get("Transfer-Encoding") or request.content_length != len(request.get_data()):
            return Response("Unexpected request framing", status=400)
        authorization = request.headers.get("Authorization", "")
        identity = request.headers.get("X-NMP-Principal-Id", "")
        if identity_mode == "workload":
            authenticated = bool(issued) and authorization == f"Bearer {issued[-1]}" and not identity
        elif identity_mode == "principal":
            authenticated = (
                not authorization
                and identity == "test-job-user"
                and request.headers.get("X-NMP-Principal-Groups") == "test-workspace-users"
                and not request.headers.get("X-NMP-Internal")
            )
        else:
            authenticated = authorization == "Bearer not-used" and not identity
        if not authenticated:
            return Response('{"detail":"Invalid or expired token"}', status=401, content_type="application/json")
        seen.append(authorization or identity)
        if len(seen) == 1:
            if identity_mode == "workload":
                assert provider.tokens is not None
                provider.tokens.expires_at = 0
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "write-file",
                        "type": "function",
                        "function": {
                            "name": "write_file",
                            "arguments": json.dumps(
                                {"file_path": "/auth-proof.txt", "content": "authenticated artifact\n"}
                            ),
                        },
                    }
                ],
            }
            finish = "tool_calls"
        else:
            message = {"role": "assistant", "content": "hello"}
            finish = "stop"
        return Response(
            json.dumps(
                {
                    "id": f"chatcmpl-{len(seen)}",
                    "object": "chat.completion",
                    "model": "test-model",
                    "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            ),
            content_type="application/json",
        )

    httpserver.expect_request("/token", method="POST").respond_with_handler(exchange)
    gateway_path = "/apis/inference-gateway/v2/workspaces/default/openai/-/v1"
    httpserver.expect_request(f"{gateway_path}/chat/completions", method="POST").respond_with_handler(model)
    config = {
        "config_format": "nemo-agents-spec-v1",
        "name": "gateway-auth-test",
        "default_harness": "deepagents",
        "harnesses": {"deepagents": {"kind": "deepagents", "settings": {"deepagents": {}}}},
        "models": {
            "default": {"provider": "openai", "model": "test-model", "base_url": httpserver.url_for(gateway_path)}
        },
        "environment": {"provider": "local", "workspace": "./workspace", "artifacts": "./artifacts"},
    }
    spec = {
        "request": {
            "agent": {"config_format": "nemo-agents-spec-v1", "config": config},
            "input": "Write auth-proof.txt, then say hello.",
            "auto_telemetry": False,
            "timeout_seconds": 60,
        },
        "agent": {
            "name": "gateway-auth-test",
            "workspace": "default",
            "config_format": "nemo-agents-spec-v1",
            "config": config,
        },
    }
    storage = StoragePaths(ephemeral=tmp_path / "ephemeral", persistent=tmp_path / "persistent")
    storage.ephemeral.mkdir()
    storage.persistent.mkdir()
    results = storage.persistent / "results"
    ctx = JobContext(workspace="default", storage=storage, results=LocalJobResults(root=results))
    try:
        result = ExecuteAgentJob().run(spec, ctx=ctx)
        assert result["status"] == "completed", result
        if identity_mode == "workload":
            assert seen == ["Bearer test-job-access-1", "Bearer test-job-access-2"]
            assert len(issued) == 2
        elif identity_mode == "principal":
            assert seen == ["test-job-user", "test-job-user"]
            assert not issued
        else:
            assert seen == ["Bearer not-used", "Bearer not-used"]
            assert not issued
        assert (results / "output_workdir" / "auth-proof.txt").read_text() == "authenticated artifact\n"
        assert json.loads((results / "fabric_run_result").read_text())["response"] == "hello"
        for path in results.rglob("*"):
            if path.is_file():
                content = path.read_bytes()
                assert b"test-job-access-" not in content
                assert b"test-job-subject-token" not in content
        httpserver.check_assertions()
    finally:
        Configuration.clear_cache()


@pytest.mark.parametrize("token_expires_mid_run", [False, True])
def test_real_agent_exports_its_trajectory_through_the_proxy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, httpserver: HTTPServer, token_expires_mid_run: bool
) -> None:
    """The auto-wired export reaches Intake authenticated, carrying no credential itself.

    The agent above proves inference routes through the proxy; this proves the
    trajectory does too, which is the point of sharing one mechanism. Relay, the
    real exporter, posts it -- so this also covers the assumption that the POST
    lands before the proxy shuts down.

    The expiring variant reproduces the access token expiring between the
    agent's last model call and its export. Pinning one token at job start
    is what used to make that export arrive expired and be dropped; because the
    proxy exchanges on demand, the export here goes out under a token that did
    not exist when the job began.
    """
    monkeypatch.setenv("NMP_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("NMP_PRINCIPAL", raising=False)
    monkeypatch.delenv("NMP_AUTH_ENABLED", raising=False)
    Configuration.clear_cache()
    proof = tmp_path / "proof"
    proof.write_text("test-job-subject-token")
    monkeypatch.setenv("NMP_BASE_URL", httpserver.url_for(""))
    monkeypatch.setenv("NMP_WORKLOAD_IDENTITY_TOKEN_FILE", str(proof))
    provider = WorkloadTokenExchangeProvider(
        token_endpoint=httpserver.url_for("/token"),
        client_id="job",
        subject_token_file=proof,
        allow_http=True,
    )
    monkeypatch.setattr(gateway_proxy, "resolve_workload_exchange_provider", lambda **kwargs: provider)
    issued: list[str] = []
    exports: list[tuple[str, int]] = []

    def exchange(request: Request) -> Response:
        token = f"test-job-access-{len(issued) + 1}"
        issued.append(token)
        return Response(
            json.dumps({"access_token": token, "token_type": "Bearer", "expires_in": 300}),
            content_type="application/json",
        )

    def model(request: Request) -> Response:
        if token_expires_mid_run:
            # Expire the credential the export will need, after inference is done
            # with it. Nothing refreshes on a timer; the next request exchanges.
            assert provider.tokens is not None
            provider.tokens.expires_at = 0
        return Response(
            json.dumps(
                {
                    "id": "chatcmpl-1",
                    "object": "chat.completion",
                    "model": "test-model",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            ),
            content_type="application/json",
        )

    def ingest(request: Request) -> Response:
        exports.append((request.headers.get("Authorization", ""), len(request.get_data())))
        return Response('{"status":"accepted"}', status=202, content_type="application/json")

    httpserver.expect_request("/token", method="POST").respond_with_handler(exchange)
    gateway_path = "/apis/inference-gateway/v2/workspaces/default/openai/-/v1"
    httpserver.expect_request(f"{gateway_path}/chat/completions", method="POST").respond_with_handler(model)
    httpserver.expect_request("/apis/intake/v2/workspaces/default/ingest/atif", method="POST").respond_with_handler(
        ingest
    )
    config = {
        "config_format": "nemo-agents-spec-v1",
        "name": "telemetry-auth-test",
        "default_harness": "deepagents",
        "harnesses": {"deepagents": {"kind": "deepagents", "settings": {"deepagents": {}}}},
        "models": {
            "default": {"provider": "openai", "model": "test-model", "base_url": httpserver.url_for(gateway_path)}
        },
        "environment": {"provider": "local", "workspace": "./workspace", "artifacts": "./artifacts"},
    }
    spec = {
        "request": {
            "agent": {"config_format": "nemo-agents-spec-v1", "config": config},
            "input": "Say hello.",
            "timeout_seconds": 60,
        },
        "agent": {
            "name": "telemetry-auth-test",
            "workspace": "default",
            "config_format": "nemo-agents-spec-v1",
            "config": config,
        },
    }
    storage = StoragePaths(ephemeral=tmp_path / "ephemeral", persistent=tmp_path / "persistent")
    storage.ephemeral.mkdir()
    storage.persistent.mkdir()
    results = storage.persistent / "results"
    ctx = JobContext(workspace="default", storage=storage, results=LocalJobResults(root=results))
    try:
        result = ExecuteAgentJob().run(spec, ctx=ctx)
        assert result["status"] == "completed", result
        assert exports, "Relay posted no trajectory to Intake"
        for authorization, size in exports:
            assert authorization == f"Bearer {issued[-1]}", "the proxy authenticates every export it forwards"
            assert size, "the proxy must forward the trajectory body, not an empty POST"
        if token_expires_mid_run:
            assert len(issued) > 1, "the expired token was never re-exchanged"
            assert exports[-1][0] != f"Bearer {issued[0]}", (
                "the export went out under the token the job started with, which had expired"
            )
        else:
            # The contrast that makes the variant above mean something: a token
            # that stays valid is exchanged once and reused, so the second
            # exchange there is the expiry being noticed, not routine churn.
            assert len(issued) == 1
        for path in results.rglob("*"):
            if path.is_file():
                assert b"test-job-access-" not in path.read_bytes()
        httpserver.check_assertions()
    finally:
        Configuration.clear_cache()
