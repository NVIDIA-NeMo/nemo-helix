# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harbor agents that name an Inference Gateway model."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from nemo_agents_plugin.runner.deployments_backend import UnreachableGatewayURLError
from nemo_evals.jobs import gateway_models
from nemo_evals.jobs.agent_evaluate import AgentEvalJob
from nemo_evals.jobs.agent_spec import HarborBuiltinAgentSource, HarborImportedAgentSource, HarborRunnerTarget
from nemo_evals.jobs.gateway_models import (
    agent_gateway_origin,
    check_gateway_model,
    check_tasks_reach_the_gateway,
    harbor_gateway_access,
)
from nemo_helix_plugin.job_context import JobContext, StoragePaths
from nemo_helix_plugin.job_results import LocalJobResults
from nhx_evals_sdk.agent_eval.runtimes.harbor.runtime import HarborEnvironmentConfig, HarborRuntimeConfig
from nhx_evals_sdk.values import Model, ModelRef, SecretRef
from pydantic import ValidationError
from pytest_mock import MockerFixture

ROUTE = "/apis/inference-gateway/v2/workspaces/ws/model/cb-gpt/-"


def _codex(
    *, env_vars: dict[str, str] | None = None, env_secrets: dict[str, SecretRef] | None = None
) -> HarborRunnerTarget:
    return HarborRunnerTarget(
        source=HarborBuiltinAgentSource(name="codex", model=ModelRef("ws/cb-gpt")),
        env_vars=env_vars or {},
        env_secrets=env_secrets or {},
    )


@pytest.fixture(autouse=True)
def _platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NEMO_BASE_URL", raising=False)
    monkeypatch.delenv("NEMO_INTERNAL_BASE_URL", raising=False)
    monkeypatch.delenv("NHX_INTERNAL_BASE_URL", raising=False)
    monkeypatch.setenv("NHX_BASE_URL", "http://localhost:8080")


@pytest.mark.parametrize(
    ("base_url", "internal_base_url", "sandboxed", "expected"),
    [
        ("http://localhost:8080", None, False, "http://host.docker.internal:8080"),
        ("http://nemo.example:8080", None, False, "http://nemo.example:8080"),
        ("http://localhost:8080", "http://nemo-api.ns.svc:8080", True, "http://nemo-api.ns.svc:8080"),
        ("http://nemo-api.ns.svc:8080", None, True, "http://nemo-api.ns.svc:8080"),
    ],
    ids=["docker-loopback", "docker-routable", "sandbox-internal-url", "sandbox-routable-base"],
)
def test_origin_is_reachable_from_where_the_agent_runs(
    monkeypatch: pytest.MonkeyPatch, base_url: str, internal_base_url: str | None, sandboxed: bool, expected: str
) -> None:
    monkeypatch.setenv("NHX_BASE_URL", base_url)
    if internal_base_url is not None:
        monkeypatch.setenv("NHX_INTERNAL_BASE_URL", internal_base_url)

    assert agent_gateway_origin(sandboxed=sandboxed) == expected


def test_a_sandbox_cannot_reach_a_loopback_platform() -> None:
    with pytest.raises(UnreachableGatewayURLError):
        agent_gateway_origin(sandboxed=True)


@pytest.mark.parametrize(
    ("source", "base_url", "base_url_env", "api_key_env"),
    [
        (
            HarborBuiltinAgentSource(name="codex", model=ModelRef("ws/cb-gpt")),
            f"{ROUTE}/v1",
            "OPENAI_BASE_URL",
            "OPENAI_API_KEY",
        ),
        (
            HarborImportedAgentSource(
                import_path="nemo_evals.harbor.agents:SandboxedCodex", model=ModelRef("ws/cb-gpt")
            ),
            f"{ROUTE}/v1",
            "OPENAI_BASE_URL",
            "OPENAI_API_KEY",
        ),
        (
            HarborBuiltinAgentSource(name="claude-code", model=ModelRef("ws/cb-gpt")),
            ROUTE,
            "ANTHROPIC_BASE_URL",
            "ANTHROPIC_API_KEY",
        ),
    ],
    ids=["codex", "sandboxed-codex", "claude-code"],
)
def test_each_agent_gets_the_route_on_the_variables_it_reads(
    source: HarborBuiltinAgentSource | HarborImportedAgentSource, base_url: str, base_url_env: str, api_key_env: str
) -> None:
    """OpenAI clients take the ``/v1`` root; the Anthropic client appends ``/v1`` itself."""
    access = harbor_gateway_access(HarborRunnerTarget(source=source), "http://gw:8080")

    assert access is not None
    assert (access.base_url, access.base_url_env, access.api_key_env) == (
        f"http://gw:8080{base_url}",
        base_url_env,
        api_key_env,
    )


def test_the_agent_gets_the_bare_model_name() -> None:
    assert _codex().agent_model_name == "cb-gpt"


def test_model_and_model_name_are_exclusive() -> None:
    with pytest.raises(ValidationError, match="set `model` or `model_name`, not both"):
        HarborBuiltinAgentSource(name="codex", model=ModelRef("ws/cb-gpt"), model_name="cb-gpt")


def test_an_agent_without_known_gateway_variables_is_refused() -> None:
    with pytest.raises(ValidationError, match="for 'oracle', set `model_name`"):
        HarborRunnerTarget(source=HarborBuiltinAgentSource(name="oracle", model=ModelRef("ws/cb-gpt")))


def test_a_base_url_the_caller_also_sets_is_refused() -> None:
    with pytest.raises(ValidationError, match="OPENAI_BASE_URL is set from the gateway model"):
        _codex(env_vars={"OPENAI_BASE_URL": "http://elsewhere/v1"})


async def test_check_refuses_auth_on_without_a_platform_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gateway_models, "platform_auth_enabled", lambda: True)

    with pytest.raises(ValueError, match="under OPENAI_API_KEY"):
        await check_gateway_model(_codex(), async_sdk=AsyncMock(), authorize=None)


async def test_check_resolves_the_model_as_the_caller(mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    """With auth on and a platform key, the model must still exist and be readable by the caller."""
    monkeypatch.setattr(gateway_models, "platform_auth_enabled", lambda: True)
    mocker.patch.object(gateway_models, "client_from_platform")
    resolve = mocker.patch.object(
        gateway_models.HelixMetricModelResolver, "resolve_model", AsyncMock(return_value=Model(url="u", name="n"))
    )
    target = _codex(env_secrets={"OPENAI_API_KEY": SecretRef("ws/agent-access-key")})

    await check_gateway_model(target, async_sdk=AsyncMock(), authorize=None)

    resolve.assert_awaited_once_with(ModelRef("ws/cb-gpt"))


async def test_check_ignores_targets_without_a_gateway_model(mocker: MockerFixture) -> None:
    resolve = mocker.patch.object(gateway_models.HelixMetricModelResolver, "resolve_model", AsyncMock())

    await check_gateway_model(
        HarborRunnerTarget(source=HarborBuiltinAgentSource(name="codex", model_name="gpt")),
        async_sdk=None,
        authorize=None,
    )

    resolve.assert_not_awaited()


def _job_config(tmp_path: Path, target: HarborRunnerTarget, *, sandboxed: bool) -> HarborRuntimeConfig:
    storage = StoragePaths(ephemeral=tmp_path / "ephemeral", persistent=tmp_path / "persistent")
    storage.ephemeral.mkdir()
    storage.persistent.mkdir()
    ctx = JobContext(workspace="ws", storage=storage, results=LocalJobResults(root=storage.persistent / "results"))

    def to_sandbox(config: HarborRuntimeConfig) -> HarborRuntimeConfig:
        return config.model_copy(update={"environment": HarborEnvironmentConfig(import_path="x:Sandbox")})

    runtime, _, _ = AgentEvalJob._resolve_target(target, ctx, configure_harbor=to_sandbox if sandboxed else None)
    config = getattr(runtime, "_config")
    assert isinstance(config, HarborRuntimeConfig)
    return config


@pytest.mark.parametrize(
    ("sandboxed", "origin"),
    [(False, "http://host.docker.internal:8080"), (True, "http://nemo-api.ns.svc:8080")],
    ids=["docker", "sandbox"],
)
def test_the_job_points_the_agent_at_the_route_for_its_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sandboxed: bool, origin: str
) -> None:
    monkeypatch.setenv("NHX_INTERNAL_BASE_URL", "http://nemo-api.ns.svc:8080")

    config = _job_config(tmp_path, _codex(), sandboxed=sandboxed)

    assert config.agent_model_name == "cb-gpt"
    assert config.gateway is not None
    assert config.gateway.base_url == f"{origin}{ROUTE}/v1"


_NO_NETWORK = {"environment": {"allow_internet": False}}


@pytest.mark.parametrize(
    ("host", "agent_channel"),
    [
        ("nemo-api.ns.svc", ("Nemo-API.ns.svc",)),
        ("nemo-api.ns.svc", (" nemo-api.ns.svc. ",)),
        ("10.1.2.3", ("10.0.0.0/8",)),
        ("nemo-api.ns.svc", ("*.ns.svc",)),
    ],
    ids=["host", "trailing-dot", "cidr", "wildcard"],
)
def test_a_no_network_task_reaches_a_gateway_on_its_agent_channel(
    monkeypatch: pytest.MonkeyPatch, host: str, agent_channel: tuple[str, ...]
) -> None:
    """Matches the sandbox's own allowlist: hosts trimmed and case-insensitively, IPs by CIDR, ``*.`` as a suffix."""
    monkeypatch.setenv("NHX_INTERNAL_BASE_URL", f"http://{host}:8080")

    check_tasks_reach_the_gateway(_codex(), [("t", _NO_NETWORK)], agent_channel)


def test_a_no_network_task_off_the_agent_channel_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """Its sandbox would deny the agent's model calls, so the run fails at submit rather than mid-trial."""
    monkeypatch.setenv("NHX_INTERNAL_BASE_URL", "http://nemo-api.ns.svc:8080")

    with pytest.raises(ValueError, match=r"no-network tasks \['offline'\] can't reach the Inference Gateway"):
        check_tasks_reach_the_gateway(_codex(), [("online", {}), ("offline", _NO_NETWORK)], ())


def test_tasks_with_network_access_or_targets_without_a_gateway_model_are_not_checked() -> None:
    check_tasks_reach_the_gateway(_codex(), [("online", {})], ())
    check_tasks_reach_the_gateway(
        HarborRunnerTarget(source=HarborBuiltinAgentSource(name="codex", model_name="gpt")), [("t", _NO_NETWORK)], ()
    )
