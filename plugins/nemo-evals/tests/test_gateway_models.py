# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harbor agents that name an Inference Gateway model."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from harbor.agents.model_connection import ModelConnectionSpec, ProviderAccess
from nemo_agents_plugin.runner.deployments_backend import UnreachableGatewayURLError
from nemo_evals.api.fields import AgentRef
from nemo_evals.gateway.policy import ModelAccess
from nemo_evals.harbor.trial_credentials import TrialCredentialIssuer
from nemo_evals.jobs import agent_compiler, gateway_agents, gateway_models
from nemo_evals.jobs.agent_evaluate import AgentEvalJob
from nemo_evals.jobs.agent_spec import (
    AgentEvalSpec,
    HarborBuiltinAgentSource,
    HarborGatewayModel,
    HarborImportedAgentSource,
    HarborModelName,
    HarborRunnerTarget,
    RegisteredAgentSource,
)
from nemo_evals.jobs.gateway_agents import GATEWAY_PLACEHOLDER_KEY
from nemo_evals.jobs.gateway_models import (
    AGENT_CHANNEL_HOSTS_KWARG,
    FABRIC_GATEWAY_KEY_ENV,
    TrialRoute,
    add_gateway_key_placeholder,
    agent_gateway_origin,
    check_gateway_model,
    check_tasks_reach_the_gateway,
    fabric_gateway_key_envs,
    fabric_gateway_route,
    point_agent_at_gateway_model,
    require_closed_book_route,
)
from nemo_evals.jobs.secret_env import SecretEnvError
from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.inference_middleware_models import MiddlewareCall, VirtualModel
from nemo_helix_plugin.job_context import JobContext, StoragePaths
from nemo_helix_plugin.job_results import LocalJobResults
from nemo_helix_plugin.jobs.api_factory import EnvironmentVariable, EnvironmentVariableFromSecret
from nhx_evals_sdk.agent_eval.runtimes.harbor.runtime import (
    HarborAgentTaskRunner,
    HarborEnvironmentConfig,
    HarborRuntimeConfig,
)
from nhx_evals_sdk.values import Model, ModelRef, SecretRef
from pydantic import ValidationError
from pytest_mock import MockerFixture

ROUTE = "/apis/inference-gateway/v2/workspaces/ws/model/cb-gpt/-"


def _gateway_source(agent: str) -> HarborBuiltinAgentSource:
    return HarborBuiltinAgentSource(name=agent, model=HarborGatewayModel(ref=ModelRef("ws/cb-gpt")))


def _codex(
    *, env_vars: dict[str, str] | None = None, env_secrets: dict[str, SecretRef] | None = None
) -> HarborRunnerTarget:
    return HarborRunnerTarget(
        source=_gateway_source("codex"),
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


@pytest.mark.parametrize(
    ("agent_channel", "expected"),
    [
        (["nemo-api.ns.svc"], "http://nemo-api.ns.svc:8080"),
        (["other.ns.svc.cluster.local", "nemo-api.ns.svc.cluster.local"], "http://nemo-api.ns.svc.cluster.local:8080"),
        (["nemo-api.ns.svcx"], "http://nemo-api.ns.svc:8080"),
        ([], "http://nemo-api.ns.svc:8080"),
    ],
    ids=["same-name", "fully-qualified-name", "label-prefix-is-not-a-name", "no-channel"],
)
def test_a_sandbox_gets_the_gateway_by_its_agent_channel_name(
    monkeypatch: pytest.MonkeyPatch, agent_channel: list[str], expected: str
) -> None:
    monkeypatch.setenv("NHX_INTERNAL_BASE_URL", "http://nemo-api.ns.svc:8080")

    assert agent_gateway_origin(sandboxed=True, agent_channel=agent_channel) == expected


def test_a_sandbox_cannot_reach_a_loopback_platform() -> None:
    with pytest.raises(UnreachableGatewayURLError):
        agent_gateway_origin(sandboxed=True)


@pytest.mark.parametrize(
    ("agent", "base_url", "base_url_env", "api_key_env", "model_name"),
    [
        ("codex", f"{ROUTE}/v1", "OPENAI_BASE_URL", "OPENAI_API_KEY", "cb-gpt"),
        ("claude-code", ROUTE, "ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY", "cb-gpt"),
        ("mini-swe-agent", f"{ROUTE}/v1", "OPENAI_BASE_URL", "MSWEA_API_KEY", "openai/cb-gpt"),
    ],
    ids=["openai-provider", "anthropic-provider", "provider-from-model-name"],
)
def test_each_agent_gets_the_route_on_the_variables_its_model_connection_declares(
    agent: str, base_url: str, base_url_env: str, api_key_env: str, model_name: str
) -> None:
    """Harbor's declared model connection decides the variables and the API style.

    OpenAI clients take the ``/v1`` root; the Anthropic client appends ``/v1`` itself. An agent that takes its provider
    from the model name is sent ``openai/<name>``, so it picks the OpenAI-style route.
    """
    target = HarborRunnerTarget(source=_gateway_source(agent))
    assert target.agent_model_name == model_name

    config = point_agent_at_gateway_model(HarborRuntimeConfig(agent_name="codex"), target, "http://gw:8080")

    assert config.env_vars == {base_url_env: f"http://gw:8080{base_url}"}
    assert config.env_secrets == {api_key_env: SecretRef(GATEWAY_PLACEHOLDER_KEY)}
    assert add_gateway_key_placeholder([], target) == [
        EnvironmentVariable(name=api_key_env, value=GATEWAY_PLACEHOLDER_KEY)
    ]


def test_a_platform_key_in_env_secrets_replaces_the_placeholder() -> None:
    """With platform auth on, the agent calls the gateway with the caller's access key, never the placeholder."""
    key = SecretRef("ws/agent-access-key")
    target = _codex(env_secrets={"OPENAI_API_KEY": key})

    config = point_agent_at_gateway_model(
        HarborRuntimeConfig(agent_name="codex", env_secrets=target.env_secrets), target, "http://gw:8080"
    )

    assert config.env_secrets == {"OPENAI_API_KEY": key}
    assert add_gateway_key_placeholder([], target) == []


def test_targets_without_a_gateway_model_get_no_placeholder() -> None:
    assert add_gateway_key_placeholder([], HarborRunnerTarget(source=HarborBuiltinAgentSource(name="codex"))) == []


def test_a_placeholder_whose_variable_another_secret_fills_is_refused() -> None:
    """Otherwise the job injects that secret under the agent's key variable, handing another credential to the agent."""
    judge_key = EnvironmentVariable(
        name="OPENAI_API_KEY", from_secret=EnvironmentVariableFromSecret(name="ws/OPENAI_API_KEY")
    )

    with pytest.raises(SecretEnvError, match="OPENAI_API_KEY is already sourced from a secret"):
        add_gateway_key_placeholder([judge_key], _codex())


def test_the_agent_gets_the_bare_model_name() -> None:
    assert _codex().agent_model_name == "cb-gpt"


@pytest.mark.parametrize(
    ("model", "expected"),
    [({"name": "gpt-x"}, HarborModelName), ({"ref": "ws/cb-gpt"}, HarborGatewayModel)],
    ids=["by-name", "gateway-ref"],
)
def test_a_model_document_is_the_one_shape_its_key_fits(model: dict[str, str], expected: type) -> None:
    target = HarborRunnerTarget.model_validate({"source": {"name": "codex", "model": model}})
    assert isinstance(target.source, HarborBuiltinAgentSource)
    assert type(target.source.model) is expected


def test_model_name_is_read_as_model_name_and_written_in_the_new_shape() -> None:
    """Specs saved before `model` took a shape still load, and are stored in the new shape from then on."""
    target = HarborRunnerTarget.model_validate({"source": {"name": "codex", "model_name": "gpt-x"}})

    assert target.source == HarborBuiltinAgentSource(name="codex", model=HarborModelName(name="gpt-x"))
    assert target.model_dump(mode="json")["source"] == {"name": "codex", "model": {"name": "gpt-x"}}


def test_an_imported_source_with_model_name_is_not_reported_as_deprecated(caplog: pytest.LogCaptureFixture) -> None:
    """`model_name` is current on an imported source; only the built-in source's is deprecated."""
    target = HarborRunnerTarget.model_validate({"source": {"import_path": "x:Y", "model_name": "m"}})

    assert target.agent_model_name == "m"
    assert "deprecated" not in caplog.text


@pytest.mark.parametrize(
    "source",
    [
        {"name": "codex", "model": {"name": "gpt-x"}, "model_name": "gpt-x"},
        {"name": "codex", "model": {"name": "gpt-x", "ref": "ws/cb-gpt"}},
        {"name": "codex", "model": {}},
        {"name": "codex", "model": "ws/cb-gpt"},
        {"name": "oracle", "model": {"ref": "ws/cb-gpt"}},
        {"name": "gemini-cli", "model": {"ref": "ws/cb-gpt"}},
    ],
    ids=[
        "model-and-model-name",
        "name-and-ref",
        "neither-name-nor-ref",
        "bare-string",
        "no-model-connection",
        "provider-the-gateway-does-not-serve",
    ],
)
def test_a_model_no_shape_fits_is_refused(source: dict[str, object]) -> None:
    """A model is given once, by name or by ref, and only agents whose model connection the gateway serves take a ref."""
    with pytest.raises(ValidationError):
        HarborRunnerTarget.model_validate({"source": source})


def test_a_base_url_the_caller_also_sets_is_refused() -> None:
    with pytest.raises(ValidationError, match="OPENAI_BASE_URL is set from the gateway model"):
        _codex(env_vars={"OPENAI_BASE_URL": "http://elsewhere/v1"})


async def test_check_refuses_auth_on_without_a_platform_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gateway_models, "platform_auth_enabled", lambda: True)

    with pytest.raises(ValueError, match="under OPENAI_API_KEY"):
        await check_gateway_model(_codex(), async_sdk=AsyncMock(), authorize=None)


async def test_check_accepts_an_anthropic_style_agent_with_auth_on(
    mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It sends its key as `x-api-key`, which the gateway accepts on model routes."""
    monkeypatch.setattr(gateway_models, "platform_auth_enabled", lambda: True)
    mocker.patch.object(gateway_models, "client_from_platform")
    resolve = mocker.patch.object(
        gateway_models.HelixMetricModelResolver, "resolve_model", AsyncMock(return_value=Model(url="u", name="n"))
    )
    target = HarborRunnerTarget(
        source=_gateway_source("claude-code"),
        env_secrets={"ANTHROPIC_API_KEY": SecretRef("ws/agent-access-key")},
    )

    await check_gateway_model(target, async_sdk=AsyncMock(), authorize=None)

    resolve.assert_awaited_once()


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
        HarborRunnerTarget(source=HarborBuiltinAgentSource(name="codex", model=HarborModelName(name="gpt"))),
        async_sdk=None,
        authorize=None,
    )

    resolve.assert_not_awaited()


def _runtime(
    tmp_path: Path,
    target: HarborRunnerTarget,
    *,
    sandboxed: bool,
    trial_issuer: TrialCredentialIssuer | None = None,
) -> HarborAgentTaskRunner:
    storage = StoragePaths(ephemeral=tmp_path / "ephemeral", persistent=tmp_path / "persistent")
    storage.ephemeral.mkdir()
    storage.persistent.mkdir()
    ctx = JobContext(workspace="ws", storage=storage, results=LocalJobResults(root=storage.persistent / "results"))

    def to_sandbox(config: HarborRuntimeConfig) -> HarborRuntimeConfig:
        environment = HarborEnvironmentConfig(
            import_path="x:Sandbox", kwargs={AGENT_CHANNEL_HOSTS_KWARG: ["nemo-api.ns.svc.cluster.local"]}
        )
        return config.model_copy(update={"environment": environment})

    runtime, _, _ = AgentEvalJob._resolve_target(
        target, ctx, configure_harbor=to_sandbox if sandboxed else None, trial_issuer=trial_issuer
    )
    assert isinstance(runtime, HarborAgentTaskRunner)
    return runtime


def _job_config(tmp_path: Path, target: HarborRunnerTarget, *, sandboxed: bool) -> HarborRuntimeConfig:
    config = getattr(_runtime(tmp_path, target, sandboxed=sandboxed), "_config")
    assert isinstance(config, HarborRuntimeConfig)
    return config


@pytest.mark.parametrize(
    ("sandboxed", "origin"),
    [(False, "http://host.docker.internal:8080"), (True, "http://nemo-api.ns.svc.cluster.local:8080")],
    ids=["docker", "sandbox"],
)
def test_the_job_points_the_agent_at_the_route_for_its_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sandboxed: bool, origin: str
) -> None:
    """A sandbox is given the gateway by the name its agent_channel allows, since that is the name it resolves."""
    monkeypatch.setenv("NHX_INTERNAL_BASE_URL", "http://nemo-api.ns.svc:8080")

    config = _job_config(tmp_path, _codex(), sandboxed=sandboxed)

    assert config.agent_model_name == "cb-gpt"
    assert config.env_vars["OPENAI_BASE_URL"] == f"{origin}{ROUTE}/v1"


@pytest.mark.parametrize("sandboxed", [False, True], ids=["docker", "sandbox"])
def test_the_job_gives_each_trial_its_own_gateway_credential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sandboxed: bool
) -> None:
    """The runner hands each planned job to the issuer, and runs the agent the user chose."""
    monkeypatch.setenv("NHX_INTERNAL_BASE_URL", "http://nemo-api.ns.svc:8080")
    issuer = TrialCredentialIssuer(AsyncNemoClient(base_url="http://platform.test"), run_id="r", intake_workspace="ws")

    runtime = _runtime(tmp_path, _codex(), sandboxed=sandboxed, trial_issuer=issuer)

    assert getattr(runtime, "_job_scope") is not None
    assert getattr(runtime, "_config").agent_name == "codex"


_NO_NETWORK = {"environment": {"allow_internet": False}}


@pytest.mark.parametrize(
    ("host", "agent_channel"),
    [
        ("nemo-api.ns.svc", ("Nemo-API.ns.svc",)),
        ("nemo-api.ns.svc", (" nemo-api.ns.svc. ",)),
        ("10.1.2.3", ("10.0.0.0/8",)),
        ("nemo-api.ns.svc", ("nemo-api.ns.svc.cluster.local",)),
    ],
    ids=["host", "trailing-dot", "cidr", "fully-qualified-name"],
)
def test_a_no_network_task_reaches_a_gateway_on_its_agent_channel(
    monkeypatch: pytest.MonkeyPatch, host: str, agent_channel: tuple[str, ...]
) -> None:
    """Hosts trimmed and case-insensitively, IPs by CIDR, or the host followed by more DNS labels."""
    monkeypatch.setenv("NHX_INTERNAL_BASE_URL", f"http://{host}:8080")

    check_tasks_reach_the_gateway(_codex(), [("t", _NO_NETWORK)], agent_channel)


@pytest.mark.parametrize("agent_channel", [(), ("*.ns.svc",)], ids=["absent", "wildcard"])
def test_a_no_network_task_off_the_agent_channel_is_refused(
    monkeypatch: pytest.MonkeyPatch, agent_channel: tuple[str, ...]
) -> None:
    """Its sandbox would deny the agent's model calls, so the run fails before any sandbox starts.

    A wildcard entry doesn't count: the sandbox keeps private addresses denied for wildcards, and an in-cluster
    gateway is usually on one.
    """
    monkeypatch.setenv("NHX_INTERNAL_BASE_URL", "http://nemo-api.ns.svc:8080")

    with pytest.raises(ValueError, match=r"no-network tasks \['offline'\] can't reach the Inference Gateway"):
        check_tasks_reach_the_gateway(_codex(), [("online", {}), ("offline", _NO_NETWORK)], agent_channel)


@pytest.mark.parametrize(
    "config",
    [
        {"agent": {"network_mode": "no-network"}},
        {"steps": [{"name": "build"}, {"name": "solve", "agent": {"network_mode": "no-network"}}]},
    ],
    ids=["agent-phase", "one-step"],
)
def test_a_task_that_takes_only_the_agent_offline_is_checked(
    monkeypatch: pytest.MonkeyPatch, config: dict[str, object]
) -> None:
    """Harbor runs the agent under its own or a step's network mode, ahead of the environment's."""
    monkeypatch.setenv("NHX_INTERNAL_BASE_URL", "http://nemo-api.ns.svc:8080")

    with pytest.raises(ValueError, match=r"gateway_unreachable: no-network tasks \['offline'\]"):
        check_tasks_reach_the_gateway(_codex(), [("offline", config)], ())


def test_tasks_with_network_access_or_targets_without_a_gateway_model_are_not_checked() -> None:
    check_tasks_reach_the_gateway(_codex(), [("online", {})], ())
    check_tasks_reach_the_gateway(
        HarborRunnerTarget(source=HarborBuiltinAgentSource(name="codex", model=HarborModelName(name="gpt"))),
        [("t", _NO_NETWORK)],
        (),
    )


def test_the_compiled_job_carries_the_placeholder_key(harbor_resolved_task) -> None:
    spec = AgentEvalSpec(tasks=[harbor_resolved_task("hello")], target=_codex())

    variables = {variable.name: variable for variable in agent_compiler._environment(spec, sandbox_plan=None)}

    assert variables["OPENAI_API_KEY"].value == GATEWAY_PLACEHOLDER_KEY


def test_an_agent_whose_connection_names_no_variables_cannot_call_the_gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    """Evals would have nowhere to put the gateway's URL or key, so the agent is refused like one with no connection."""
    agent = SimpleNamespace(MODEL_CONNECTION=ModelConnectionSpec(default_provider="anthropic"))
    monkeypatch.setattr(gateway_agents.AgentFactory, "get_agent_class", lambda name: agent)
    monkeypatch.setitem(
        gateway_agents.PROVIDERS, "anthropic", ProviderAccess(api_key_envs=(), base_url="https://x", base_url_envs=())
    )
    gateway_agents.gateway_agent_env.cache_clear()
    try:
        assert gateway_agents.gateway_agent_env("claude-code") is None
    finally:
        gateway_agents.gateway_agent_env.cache_clear()


_CLOSED_BOOK = MiddlewareCall(name="nemo-evals-closed-book", config_type="closed_book", config={})
_ROUTER = MiddlewareCall(name="nemo-switchyard", config_type="routellm_config", config={})
_NO_NETWORK_TASKS = [("open", {}), ("offline", {"environment": {"network_mode": "no-network"}})]


def _virtual_models(monkeypatch: pytest.MonkeyPatch, virtual_model: VirtualModel | Exception) -> AsyncMock:
    get = AsyncMock(
        side_effect=virtual_model
        if isinstance(virtual_model, Exception)
        else lambda **_: SimpleNamespace(data=lambda: virtual_model)
    )
    monkeypatch.setattr(gateway_models, "client_from_platform", lambda *_: SimpleNamespace(get_virtual_model=get))
    return get


async def test_a_closed_book_virtual_model_serves_no_network_tasks(monkeypatch: pytest.MonkeyPatch) -> None:
    get = _virtual_models(
        monkeypatch, VirtualModel(name="cb-gpt", workspace="ws", request_middleware=[_ROUTER, _CLOSED_BOOK])
    )

    await require_closed_book_route(_codex(), _NO_NETWORK_TASKS, async_sdk=AsyncMock(), authorize=None)

    get.assert_awaited_once_with(name="cb-gpt", workspace="ws")


@pytest.mark.parametrize(
    "virtual_model",
    [
        VirtualModel(name="cb-gpt", workspace="ws"),
        VirtualModel(name="cb-gpt", workspace="ws", response_middleware=[_CLOSED_BOOK]),
        VirtualModel(name="cb-gpt", workspace="ws", request_middleware=[_CLOSED_BOOK, _ROUTER]),
        NotFoundError(httpx.Response(404, request=httpx.Request("GET", "http://gw"))),
    ],
    ids=["no-middleware", "closed-book-on-the-response", "router-after-closed-book", "model-entity-only"],
)
async def test_a_gateway_model_that_is_not_closed_book_is_refused(
    monkeypatch: pytest.MonkeyPatch, virtual_model: VirtualModel | Exception
) -> None:
    """Only the last request middleware checks the model the request is sent to; an entity alone passes tools on."""
    _virtual_models(monkeypatch, virtual_model)

    with pytest.raises(ValueError, match=r"closed_book_required: no-network tasks \['offline'\] need a closed-book"):
        await require_closed_book_route(_codex(), _NO_NETWORK_TASKS, async_sdk=AsyncMock(), authorize=None)


@pytest.mark.parametrize(
    "target",
    [
        HarborRunnerTarget(source=HarborBuiltinAgentSource(name="codex", model=HarborModelName(name="gpt-5"))),
        HarborRunnerTarget(source=HarborBuiltinAgentSource(name="terminus-2")),
        HarborRunnerTarget(source=HarborImportedAgentSource(import_path="my_agent:Agent", model_name="gpt-5")),
        HarborRunnerTarget(
            source=RegisteredAgentSource(agent=AgentRef("ws/coder")),
            agent_kwargs={"fabric_config": {"name": "coder"}},
        ),
    ],
    ids=["model-by-name", "built-in-default-model", "imported-agent", "registered-agent"],
)
async def test_a_model_route_evals_cannot_check_is_refused_for_no_network_tasks(target: HarborRunnerTarget) -> None:
    with pytest.raises(ValueError, match=r"closed_book_required: no-network tasks \['offline'\]"):
        await require_closed_book_route(target, _NO_NETWORK_TASKS, async_sdk=None, authorize=None)


_IGW = "http://h/apis/inference-gateway/v2/workspaces/ws"


def _registered(config: dict[str, object]) -> HarborRunnerTarget:
    return HarborRunnerTarget(
        source=RegisteredAgentSource(agent=AgentRef("ws/coder")), agent_kwargs={"fabric_config": config}
    )


async def test_a_registered_agent_whose_gateway_models_are_closed_book_serves_no_network_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every model the agent calls through the gateway is checked; one it calls elsewhere can't leave the sandbox."""
    get = _virtual_models(monkeypatch, VirtualModel(name="cb", workspace="ws", request_middleware=[_CLOSED_BOOK]))
    target = _registered(
        {
            "models": {
                "default": {"base_url": f"{_IGW}/openai/-/v1", "model": "cb"},
                "elsewhere": {"base_url": "https://integrate.api.nvidia.com/v1", "model": "x"},
            },
            "harnesses": {"codex": {"model": {"base_url": f"{_IGW}/model/cb-harness/-/v1"}}},
        }
    )

    await require_closed_book_route(target, _NO_NETWORK_TASKS, async_sdk=AsyncMock(), authorize=None)

    assert [call.kwargs["name"] for call in get.await_args_list] == ["cb", "cb-harness"]


async def test_a_registered_agent_with_a_harness_model_that_is_not_closed_book_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    by_name = {
        "cb": VirtualModel(name="cb", workspace="ws", request_middleware=[_CLOSED_BOOK]),
        "open": VirtualModel(name="open", workspace="ws"),
    }
    get = AsyncMock(side_effect=lambda name, **_: SimpleNamespace(data=lambda: by_name[name]))
    monkeypatch.setattr(gateway_models, "client_from_platform", lambda *_: SimpleNamespace(get_virtual_model=get))
    target = _registered(
        {
            "models": {"default": {"base_url": f"{_IGW}/openai/-/v1", "model": "cb"}},
            "harnesses": {"codex": {"model": {"base_url": f"{_IGW}/openai/-/v1", "model": "open"}}},
        }
    )

    with pytest.raises(ValueError, match=r"ws/open isn't one"):
        await require_closed_book_route(target, _NO_NETWORK_TASKS, async_sdk=AsyncMock(), authorize=None)


@pytest.mark.parametrize(
    ("model", "match"),
    [
        pytest.param(
            {"settings": {"base_url": f"{_IGW}/openai/-/v1"}, "model": "open"},
            r"ws/open isn't one",
            id="url-in-settings",
        ),
        pytest.param(
            {"base_url": f"{_IGW}/provider/stub/-/v1", "model": "x"}, r"names no virtual model", id="provider-route"
        ),
        pytest.param({"base_url": "${GATEWAY_URL}/v1", "model": "cb"}, r"is a template", id="templated-url"),
    ],
)
async def test_a_registered_agent_model_evals_cannot_check_is_refused(
    monkeypatch: pytest.MonkeyPatch, model: dict[str, object], match: str
) -> None:
    """The check reads a model's URL as the runtime does, and fails closed on a gateway route it can't resolve."""
    by_name = {
        "cb": VirtualModel(name="cb", workspace="ws", request_middleware=[_CLOSED_BOOK]),
        "open": VirtualModel(name="open", workspace="ws"),
    }
    get = AsyncMock(side_effect=lambda name, **_: SimpleNamespace(data=lambda: by_name[name]))
    monkeypatch.setattr(gateway_models, "client_from_platform", lambda *_: SimpleNamespace(get_virtual_model=get))
    target = _registered({"models": {"default": {"base_url": f"{_IGW}/openai/-/v1", "model": "cb"}, "other": model}})

    with pytest.raises(ValueError, match=match):
        await require_closed_book_route(target, _NO_NETWORK_TASKS, async_sdk=AsyncMock(), authorize=None)


@pytest.mark.parametrize("agent", ["oracle", "nop"])
async def test_agents_that_call_no_model_run_no_network_tasks(agent: str) -> None:
    target = HarborRunnerTarget(source=HarborBuiltinAgentSource(name=agent))

    await require_closed_book_route(target, _NO_NETWORK_TASKS, async_sdk=None, authorize=None)


async def test_runs_without_no_network_tasks_are_not_checked() -> None:
    target = HarborRunnerTarget(source=HarborBuiltinAgentSource(name="codex", model=HarborModelName(name="gpt-5")))

    await require_closed_book_route(target, [("open", {})], async_sdk=None, authorize=None)


@pytest.mark.parametrize(
    "config",
    [
        {"environment": {"allow_internet": False}},
        {"agent": {"network_mode": "no-network"}},
        {"steps": [{"name": "build"}, {"name": "solve", "agent": {"network_mode": "no-network"}}]},
    ],
    ids=["deprecated-allow-internet", "agent-phase", "one-step"],
)
async def test_every_way_a_task_takes_the_agent_offline_is_checked(config: dict[str, object]) -> None:
    """Harbor runs the agent under its own or a step's network mode, ahead of the environment's."""
    with pytest.raises(ValueError, match=r"closed_book_required: no-network tasks \['offline'\]"):
        await require_closed_book_route(
            HarborRunnerTarget(source=HarborBuiltinAgentSource(name="codex", model=HarborModelName(name="gpt-5"))),
            [("offline", config)],
            async_sdk=None,
            authorize=None,
        )


async def test_the_caller_must_be_able_to_read_the_virtual_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Otherwise the refusal would tell them what its middleware is."""
    get = _virtual_models(monkeypatch, VirtualModel(name="cb-gpt", workspace="ws", request_middleware=[_CLOSED_BOOK]))
    authorize = AsyncMock(side_effect=PermissionError("denied"))

    with pytest.raises(PermissionError):
        await require_closed_book_route(_codex(), _NO_NETWORK_TASKS, async_sdk=AsyncMock(), authorize=authorize)

    authorize.assert_awaited_once_with("GET", "/apis/inference-gateway/v2/workspaces/ws/virtual-models/cb-gpt")
    get.assert_not_awaited()


_RECORDING = {"name": "nemo-evals-trial-recording", "config_type": "trial_recording", "config": {}}


def _virtual_model(request: list[dict], response: list[dict]) -> dict:
    return {
        "name": "agent-model",
        "workspace": "models",
        "request_middleware": request,
        "response_middleware": response,
    }


def _models_client(answer: httpx.Response, *, color: str) -> NemoClient | AsyncNemoClient:
    transport = httpx.MockTransport(lambda request: answer)
    if color == "async":
        return AsyncNemoClient(base_url="http://nemo-api:8080", http_client=httpx.AsyncClient(transport=transport))
    return NemoClient(base_url="http://nemo-api:8080", http_client=httpx.Client(transport=transport))


_CLOSED_BOOK_CALL = {"name": "nemo-evals-closed-book", "config_type": "closed_book", "config": {}}


@pytest.mark.parametrize(
    ("virtual_model", "warning", "route"),
    [
        pytest.param(
            _virtual_model([_RECORDING], [_RECORDING]), None, TrialRoute(ModelAccess.OPEN, True), id="recorded"
        ),
        pytest.param(
            _virtual_model([_RECORDING, _CLOSED_BOOK_CALL], [_RECORDING]),
            None,
            TrialRoute(ModelAccess.CLOSED, True),
            id="closed-book",
        ),
        pytest.param(
            _virtual_model([_CLOSED_BOOK_CALL, _RECORDING], [_RECORDING]),
            None,
            TrialRoute(ModelAccess.OPEN, True),
            id="closed-book not last",
        ),
        pytest.param(
            _virtual_model([], []),
            "won't be recorded in their trials",
            TrialRoute(ModelAccess.OPEN, False),
            id="not listed",
        ),
        pytest.param(
            _virtual_model([_RECORDING], []),
            "won't be recorded in their trials",
            TrialRoute(ModelAccess.OPEN, False),
            id="request only",
        ),
        pytest.param(
            _virtual_model([], [_RECORDING]),
            "span starts when the gateway has the reply",
            TrialRoute(ModelAccess.OPEN, True),
            id="response only",
        ),
        pytest.param(
            None, "Couldn't read virtual model models/agent-model", TrialRoute(ModelAccess.OPEN, False), id="unreadable"
        ),
    ],
)
@pytest.mark.parametrize("color", ["sync", "async"])
def test_the_job_reads_its_trials_policy_from_the_route_and_warns_when_it_wont_record_them(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    virtual_model: dict | None,
    warning: str | None,
    route: TrialRoute,
    color: str,
) -> None:
    """Recording is opt-in per virtual model, so a run that won't be attributed says so in its log."""
    monkeypatch.setattr(gateway_models, "run_with_isolated_async_client", lambda client, fn: asyncio.run(fn(client)))
    answer = (
        httpx.Response(200, json=virtual_model)
        if virtual_model is not None
        else httpx.Response(403, json={"detail": "forbidden"})
    )

    with caplog.at_level("WARNING", logger=gateway_models.__name__):
        result = gateway_models.check_trial_route(_models_client(answer, color=color), ModelRef("models/agent-model"))

    assert result == route
    if warning is None:
        assert caplog.records == []
    else:
        assert warning in caplog.text


@pytest.mark.parametrize(
    ("model", "route"),
    [
        pytest.param(
            {"base_url": "http://h:8080/apis/inference-gateway/v2/workspaces/dev/model/judge/-/v1"},
            (ModelRef("dev/judge"), "NEMO_AGENTS_IGW_API_KEY"),
            id="the model's own route",
        ),
        pytest.param(
            {"base_url": "http://h/apis/inference-gateway/v2/workspaces/dev/openai/-/v1", "model": "ws/recorded"},
            (ModelRef("ws/recorded"), "NEMO_AGENTS_IGW_API_KEY"),
            id="the OpenAI route, model with its workspace",
        ),
        pytest.param(
            {
                "base_url": "http://h/apis/inference-gateway/v2/workspaces/dev/openai/-/v1",
                "model": "recorded",
                "api_key_env": "MY_KEY",
            },
            (ModelRef("dev/recorded"), "MY_KEY"),
            id="the OpenAI route, bare model name",
        ),
        pytest.param({"base_url": "https://integrate.api.nvidia.com/v1", "model": "x"}, None, id="not the gateway"),
        pytest.param(
            {"base_url": "http://h/apis/inference-gateway/v2/workspaces/dev/openai/-/v1"}, None, id="no model to name"
        ),
    ],
)
def test_a_registered_agents_gateway_route_is_read_from_its_default_model(
    model: dict[str, str], route: tuple[ModelRef, str] | None
) -> None:
    assert fabric_gateway_route({"models": {"default": model}}) == route


def test_a_registered_agents_gateway_models_each_name_the_variable_they_read_their_key_from() -> None:
    config = {
        "models": {
            "other": {"base_url": f"{_IGW}/openai/-/v1", "model": "cb", "api_key_env": "OTHER_KEY"},
            "default": {"base_url": f"{_IGW}/model/cb/-/v1"},
            "elsewhere": {"base_url": "https://integrate.api.nvidia.com/v1", "model": "x", "api_key_env": "NVIDIA_KEY"},
        },
        "harnesses": {
            "codex": {"model": {"base_url": f"{_IGW}/model/cb-harness/-/v1", "api_key_env": "HARNESS_KEY"}},
            "claude": {"model": {"base_url": f"{_IGW}/openai/-", "model": "cb-claude"}},
        },
    }

    assert fabric_gateway_key_envs(config) == [FABRIC_GATEWAY_KEY_ENV, "OTHER_KEY", "HARNESS_KEY"]
