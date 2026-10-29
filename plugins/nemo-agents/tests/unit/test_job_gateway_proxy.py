# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import copy
import json
import socket
from pathlib import Path
from unittest.mock import Mock
from urllib.parse import urlsplit

import httpx
import pytest
from nemo_agents_plugin.agent_config import AgentConfig
from nemo_agents_plugin.fabric.invocation import AgentConfigInvocationRequest
from nemo_agents_plugin.fabric.runtime import FabricRuntimeResult
from nemo_agents_plugin.jobs import execute, gateway_proxy
from nemo_helix_plugin.client.auth import StaticToken
from nemo_helix_plugin.job_context import JobContext
from pytest_httpserver import HTTPServer

GATEWAY_PATH = "/apis/inference-gateway/v2/workspaces/test/openai/-/v1"


@pytest.fixture(autouse=True)
def isolate_job_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NHX_PRINCIPAL", raising=False)
    monkeypatch.delenv("NHX_WORKLOAD_IDENTITY_TOKEN_FILE", raising=False)
    monkeypatch.delenv("NHX_AUTH_ENABLED", raising=False)


def _config() -> AgentConfig:
    return AgentConfig.model_validate(
        {
            "config_format": "nemo-agents-spec-v1",
            "name": "auth-test",
            "default_harness": "deepagents",
            "harnesses": {"deepagents": {"kind": "deepagents"}},
            "models": {
                "default": {"provider": "openai", "model": "test", "base_url": f"http://platform{GATEWAY_PATH}"}
            },
        }
    )


@pytest.fixture
def workload(monkeypatch: pytest.MonkeyPatch, httpserver: HTTPServer) -> Mock:
    monkeypatch.setenv("NHX_WORKLOAD_IDENTITY_TOKEN_FILE", "/job/proof-token")
    monkeypatch.setenv("NHX_BASE_URL", httpserver.url_for(""))
    factory = Mock(return_value=StaticToken("job-access-token"))
    monkeypatch.setattr(gateway_proxy, "resolve_workload_exchange_provider", factory)
    return factory


def _assert_closed(url: str) -> None:
    parsed = urlsplit(url)
    with socket.socket() as connection:
        connection.settimeout(1)
        assert connection.connect_ex((parsed.hostname, parsed.port)) != 0


def test_rebases_all_gateway_models_without_mutating_input(workload: Mock) -> None:
    config = _config()
    model = config.models["default"]
    assert model.base_url is not None
    model.base_url += "?version=1"
    config.models["direct"] = model.model_copy(update={"base_url": "https://provider.test/v1"})
    config.models["legacy"] = model.model_copy(update={"base_url": None, "settings": {"base_url": model.base_url}})
    config.harnesses["deepagents"].model = model.model_copy(deep=True)
    config.telemetry.atif = {"storage": [{"type": "http", "endpoint": "http://localhost:8080/intake"}]}
    before = config.model_dump()
    with gateway_proxy.platform_auth_proxy() as origin:
        runtime = gateway_proxy.rewrite_gateway_models(config, origin)
        proxy_url = runtime.models["default"].base_url
        assert proxy_url is not None
        assert urlsplit(proxy_url).hostname == "127.0.0.1"
        assert proxy_url.endswith(f"{GATEWAY_PATH}?version=1")
        assert runtime.models["legacy"].base_url == proxy_url
        assert "base_url" not in runtime.models["legacy"].settings
        harness_model = runtime.harnesses["deepagents"].model
        assert harness_model is not None
        assert harness_model.base_url == proxy_url
        assert runtime.models["direct"].base_url == "https://provider.test/v1"
        assert runtime.telemetry == config.telemetry
        assert "job-access-token" not in runtime.model_dump_json()
    assert config.model_dump() == before
    workload.assert_called_once()
    _assert_closed(proxy_url)


def test_rebases_translated_fabric_config_without_mutating_input() -> None:
    config = {
        "harness": {"adapter_id": "nvidia.fabric.deepagents"},
        "models": {
            "default": {
                "provider": "openai",
                "model": "test",
                "base_url": f"http://platform{GATEWAY_PATH}?version=1",
            },
            "legacy": {
                "provider": "openai",
                "model": "test",
                "settings": {"base_url": f"http://platform{GATEWAY_PATH}"},
            },
            "direct": {"provider": "openai", "model": "test", "base_url": "https://provider.test/v1"},
        },
    }
    before = copy.deepcopy(config)

    assert gateway_proxy.routes_fabric_inference_through_gateway(config)
    runtime = gateway_proxy.rewrite_fabric_gateway_models(config, "http://127.0.0.1:9019")

    assert runtime["models"]["default"]["base_url"] == f"http://127.0.0.1:9019{GATEWAY_PATH}?version=1"
    assert runtime["models"]["legacy"]["base_url"] == f"http://127.0.0.1:9019{GATEWAY_PATH}"
    assert "base_url" not in runtime["models"]["legacy"]["settings"]
    assert runtime["models"]["direct"]["base_url"] == "https://provider.test/v1"
    assert config == before


@pytest.mark.parametrize("models", [None, "not-a-mapping", {"default": "not-a-mapping", "bad": None}])
def test_malformed_fabric_models_are_skipped_instead_of_crashing(models: object) -> None:
    config = {"harness": {"adapter_id": "nvidia.fabric.deepagents"}, "models": models}

    assert not gateway_proxy.routes_fabric_inference_through_gateway(config)
    assert gateway_proxy.rewrite_fabric_gateway_models(config, "http://127.0.0.1:9019") is config


def test_fabric_model_with_non_mapping_settings_is_still_rebased() -> None:
    config = {"models": {"default": {"base_url": f"http://platform{GATEWAY_PATH}", "settings": "oops"}}}

    runtime = gateway_proxy.rewrite_fabric_gateway_models(config, "http://127.0.0.1:9019")

    assert runtime["models"]["default"]["base_url"] == f"http://127.0.0.1:9019{GATEWAY_PATH}"


def test_ambient_user_access_token_does_not_start_a_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    """NHX_ACCESS_TOKEN is the user's CLI override, not a delegated job token."""
    monkeypatch.delenv("NHX_ACCESS_TOKEN_FILE", raising=False)
    monkeypatch.setenv("NHX_ACCESS_TOKEN", "user-cli-token")

    with gateway_proxy.platform_auth_proxy() as origin:
        assert origin is None


@pytest.mark.parametrize("principal", [None, ""])
def test_a_job_with_no_identity_runs_without_a_proxy(
    monkeypatch: pytest.MonkeyPatch, workload: Mock, principal: str | None
) -> None:
    """Nothing to forward, so nothing is resolved and callers reach the platform directly."""
    monkeypatch.delenv("NHX_WORKLOAD_IDENTITY_TOKEN_FILE")
    if principal is not None:
        monkeypatch.setenv("NHX_PRINCIPAL", principal)
    principal_client = Mock(side_effect=AssertionError("Must not resolve principal headers"))
    monkeypatch.setattr(gateway_proxy, "get_nemo_client", principal_client)

    with gateway_proxy.platform_auth_proxy() as origin:
        assert origin is None
        # A config handed no origin is returned untouched, not deep-copied.
        config = _config()
        assert gateway_proxy.rewrite_gateway_models(config, origin) is config

    workload.assert_not_called()
    principal_client.assert_not_called()


def test_only_gateway_models_ask_for_a_proxy() -> None:
    """The predicate ``run()`` ORs with its telemetry decision to size the proxy."""
    assert gateway_proxy.routes_inference_through_gateway(_config())

    direct = _config()
    direct.models["default"].base_url = "https://provider.test/v1"
    assert not gateway_proxy.routes_inference_through_gateway(direct)
    # A run with nothing to route is handed back its own config.
    assert gateway_proxy.rewrite_gateway_models(direct, "http://127.0.0.1:1") is direct


def test_intake_otlp_endpoints_are_rebased_onto_the_proxy_like_everything_else_platform_bound() -> None:
    """The same spec exports through the auth-proxy sidecar when deployed; a job must not post it to the
    literal URL from inside a pod where that URL is unreachable or unauthenticated."""
    intake = "/apis/intake/v2/workspaces/team-a/ingest/otlp/v1"
    config = _config()
    config.telemetry.opentelemetry = {
        "enabled": True,
        "endpoints": [
            {"type": "gen_ai", "endpoint": f"https://helix.example.com{intake}/traces"},
            {"type": "gen_ai", "endpoint": "https://otel.example.com/v1/traces"},
        ],
        "logs": {"enabled": True, "endpoints": [{"endpoint": f"http://localhost:8080{intake}/logs"}]},
        "metrics": {"enabled": True, "endpoints": [{"endpoint": f"http://localhost:8080{intake}/metrics"}]},
    }
    before = config.model_dump()
    assert gateway_proxy.exports_otlp_to_intake(config)

    runtime = gateway_proxy.rewrite_intake_otlp_endpoints(config, "http://127.0.0.1:9019")

    opentelemetry = runtime.telemetry.opentelemetry
    assert opentelemetry is not None
    assert [e["endpoint"] for e in opentelemetry["endpoints"]] == [
        f"http://127.0.0.1:9019{intake}/traces",
        "https://otel.example.com/v1/traces",
    ]
    assert opentelemetry["logs"]["endpoints"][0]["endpoint"] == f"http://127.0.0.1:9019{intake}/logs"
    assert opentelemetry["metrics"]["endpoints"][0]["endpoint"] == f"http://127.0.0.1:9019{intake}/metrics"
    assert config.model_dump() == before
    assert gateway_proxy.rewrite_intake_otlp_endpoints(config, None) is config

    own_collector = _config()
    own_collector.telemetry.opentelemetry = {"endpoints": [{"type": "gen_ai", "endpoint": "https://mine/otlp"}]}
    assert not gateway_proxy.exports_otlp_to_intake(own_collector)
    assert gateway_proxy.rewrite_intake_otlp_endpoints(own_collector, "http://127.0.0.1:1") is own_collector


@pytest.mark.parametrize(
    "failure", [RuntimeError("invoke failed"), TimeoutError("invoke timed out"), asyncio.CancelledError()]
)
def test_proxy_closes_on_invocation_failure(workload: Mock, failure: BaseException) -> None:
    with pytest.raises(type(failure)):
        with gateway_proxy.platform_auth_proxy() as origin:
            proxy_url = gateway_proxy.rewrite_gateway_models(_config(), origin).models["default"].base_url
            assert proxy_url is not None
            raise failure
    assert proxy_url is not None
    _assert_closed(proxy_url)


@pytest.mark.parametrize("identity_mode", ["workload", "principal"])
def test_bad_platform_url_fails_without_resolving_credentials(
    monkeypatch: pytest.MonkeyPatch, workload: Mock, identity_mode: str
) -> None:
    if identity_mode == "principal":
        monkeypatch.delenv("NHX_WORKLOAD_IDENTITY_TOKEN_FILE")
        monkeypatch.setenv("NHX_PRINCIPAL", '{"id":"job-user"}')
    monkeypatch.delenv("NHX_BASE_URL")
    with pytest.raises(ValueError, match="NHX_BASE_URL"):
        with gateway_proxy.platform_auth_proxy():
            pytest.fail("Must not invoke the agent")
    workload.assert_not_called()


@pytest.mark.parametrize("exchange_fails", [False, True])
def test_workload_identity_takes_precedence_without_fallback(
    monkeypatch: pytest.MonkeyPatch, workload: Mock, httpserver: HTTPServer, exchange_fails: bool
) -> None:
    monkeypatch.setenv("NHX_PRINCIPAL", '{"id":"job-user"}')
    principal_client = Mock(side_effect=AssertionError("Must not resolve principal headers"))
    monkeypatch.setattr(gateway_proxy, "get_nemo_client", principal_client)
    if exchange_fails:
        workload.return_value = Mock(get_access_token=Mock(side_effect=RuntimeError("exchange unavailable")))
        with pytest.raises(RuntimeError, match="exchange unavailable"):
            with gateway_proxy.platform_auth_proxy():
                pytest.fail("Must not invoke the agent")
    else:
        httpserver.expect_request(GATEWAY_PATH, headers={"Authorization": "Bearer job-access-token"}).respond_with_json(
            {"ok": True}
        )
        with gateway_proxy.platform_auth_proxy() as origin:
            proxy_url = gateway_proxy.rewrite_gateway_models(_config(), origin).models["default"].base_url
            assert proxy_url is not None
            assert httpx.get(proxy_url).status_code == 200
        httpserver.check_assertions()
    principal_client.assert_not_called()


def test_principal_headers_preserve_identity_and_replace_caller_headers(
    monkeypatch: pytest.MonkeyPatch, httpserver: HTTPServer
) -> None:
    principal = {
        "id": "job-user",
        "email": "actor@example.test",
        "groups": ["actor-group"],
        "on_behalf_of": "delegated-user",
        "on_behalf_of_email": "subject@example.test",
        "on_behalf_of_groups": ["subject-group"],
    }
    headers = {
        "X-NHX-Principal-Id": "job-user",
        "X-NHX-Principal-Email": "actor@example.test",
        "X-NHX-Principal-Groups": "actor-group",
        "X-NHX-Principal-On-Behalf-Of": "delegated-user",
        "X-NHX-Principal-On-Behalf-Of-Email": "subject@example.test",
        "X-NHX-Principal-On-Behalf-Of-Groups": "subject-group",
    }
    monkeypatch.setenv("NHX_PRINCIPAL", json.dumps(principal))
    monkeypatch.setenv("NHX_BASE_URL", httpserver.url_for(""))
    httpserver.expect_request(GATEWAY_PATH, headers=headers).respond_with_json({"ok": True})
    with gateway_proxy.platform_auth_proxy() as origin:
        runtime = gateway_proxy.rewrite_gateway_models(_config(), origin)
        proxy_url = runtime.models["default"].base_url
        assert proxy_url is not None
        response = httpx.get(
            proxy_url, headers={**dict.fromkeys(headers, "spoofed"), "Authorization": "Bearer not-used"}
        )
        assert response.status_code == 200
        assert "job-user" not in runtime.model_dump_json()
    httpserver.check_assertions()
    forwarded = httpserver.log[0][0].headers
    assert "Authorization" not in forwarded
    assert "X-NHX-Internal" not in forwarded
    _assert_closed(proxy_url)


def test_proxy_forwards_the_rotating_subprocess_access_token(
    monkeypatch: pytest.MonkeyPatch, httpserver: HTTPServer, tmp_path: Path
) -> None:
    token_file = tmp_path / ".nhx-access-token"
    token_file.write_text("first-token", encoding="utf-8")
    monkeypatch.setenv("NHX_ACCESS_TOKEN_FILE", str(token_file))
    monkeypatch.setenv("NHX_PRINCIPAL", '{"id":"job-user"}')
    monkeypatch.setenv("NHX_BASE_URL", httpserver.url_for(""))
    httpserver.expect_request(GATEWAY_PATH).respond_with_json({"ok": True})
    with gateway_proxy.platform_auth_proxy() as origin:
        runtime = gateway_proxy.rewrite_gateway_models(_config(), origin)
        proxy_url = runtime.models["default"].base_url
        assert proxy_url is not None
        assert httpx.get(proxy_url).status_code == 200
        token_file.write_text("second-token", encoding="utf-8")
        assert httpx.get(proxy_url).status_code == 200
    forwarded = [request.headers for request, _ in httpserver.log]
    assert [headers["Authorization"] for headers in forwarded] == ["Bearer first-token", "Bearer second-token"]
    assert all("X-NHX-Principal-Id" not in headers for headers in forwarded)
    _assert_closed(proxy_url)


@pytest.mark.parametrize(
    "auth_enabled,principal",
    [("true", "broken-json"), ("false", "broken-json"), ("true", "{}"), ("true", '{"id":""}')],
)
def test_invalid_principal_fails_before_invocation(
    monkeypatch: pytest.MonkeyPatch, httpserver: HTTPServer, auth_enabled: str, principal: str
) -> None:
    monkeypatch.setenv("NHX_AUTH_ENABLED", auth_enabled)
    monkeypatch.setenv("NHX_PRINCIPAL", principal)
    monkeypatch.setenv("NHX_BASE_URL", httpserver.url_for(""))
    with pytest.raises(ValueError):
        with gateway_proxy.platform_auth_proxy():
            pytest.fail("Must not invoke the agent")


@pytest.mark.parametrize("auth_enabled", [False, True])
@pytest.mark.parametrize("source", ["environment", "config-file"])
def test_anonymous_job_respects_platform_auth_configuration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, auth_enabled: bool, source: str
) -> None:
    value = str(auth_enabled).lower()
    config_file = tmp_path / "config.yaml"
    config_file.write_text(f"auth:\n  enabled: {value}\n" if source == "config-file" else "auth: {}\n")
    monkeypatch.setenv("NHX_CONFIG_FILE_PATH", str(config_file))
    if source == "environment":
        monkeypatch.setenv("NHX_AUTH_ENABLED", value)
    monkeypatch.setenv("NHX_PRINCIPAL", '{"id":"","groups":[]}')
    monkeypatch.setenv("NHX_BASE_URL", "http://localhost:8080")
    if auth_enabled:
        with pytest.raises(ValueError, match="principal ID"):
            with gateway_proxy.platform_auth_proxy():
                pytest.fail("Must not invoke the agent")
    else:
        with gateway_proxy.platform_auth_proxy() as origin:
            assert origin is None


@pytest.mark.parametrize("startup", ["error", "timeout"])
def test_startup_failure_closes_listener(monkeypatch: pytest.MonkeyPatch, workload: Mock, startup: str) -> None:
    address = []

    def run(server, *, sockets):
        address.append(sockets[0].getsockname())
        if startup == "error":
            raise RuntimeError("cannot start")
        while not server.should_exit:
            import time

            time.sleep(0.01)

    monkeypatch.setattr(gateway_proxy.uvicorn.Server, "run", run)
    monkeypatch.setattr(gateway_proxy, "_STARTUP_TIMEOUT_SECONDS", 0.05)
    with pytest.raises((RuntimeError, TimeoutError), match="proxy"):
        with gateway_proxy.platform_auth_proxy():
            pytest.fail("Must not invoke the agent")
    _assert_closed(f"http://{address[0][0]}:{address[0][1]}")


def test_job_authenticates_with_telemetry_disabled(
    monkeypatch: pytest.MonkeyPatch, workload: Mock, httpserver: HTTPServer, ctx: JobContext
) -> None:
    httpserver.expect_request(
        f"{GATEWAY_PATH}/chat/completions", headers={"Authorization": "Bearer job-access-token"}
    ).respond_with_json({"ok": True})
    config = _config().model_dump()
    spec = {
        "request": {
            "agent": {"config_format": "nemo-agents-spec-v1", "config": config},
            "input": "hello",
            "auto_telemetry": False,
        },
        "agent": {
            "name": "auth-test",
            "workspace": "default",
            "config_format": "nemo-agents-spec-v1",
            "config": config,
        },
    }

    async def invoke(request: AgentConfigInvocationRequest) -> FabricRuntimeResult:
        base_url = request.agent_config.models["default"].base_url
        assert base_url is not None
        async with httpx.AsyncClient() as client:
            response = await client.post(
                base_url + "/chat/completions",
                headers={"Authorization": "Bearer not-used"},
            )
        response.raise_for_status()
        (request.base_dir / "artifacts" / "config.json").write_text(request.agent_config.model_dump_json())
        return FabricRuntimeResult(status="succeeded", response="hello")

    monkeypatch.setattr(execute, "invoke_agent_config_request_once", invoke)
    assert execute.ExecuteAgentJob().run(spec, ctx=ctx)["status"] == "completed"
    for path in ctx.storage.persistent.rglob("*"):
        if path.is_file():
            assert b"job-access-token" not in path.read_bytes()
    assert "127.0.0.1" not in json.dumps(config)
