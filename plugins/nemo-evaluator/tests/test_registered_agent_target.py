# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A registered platform agent as the source of a Fabric or Harbor runner target."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from nemo_evaluator.api.schemas import AgentRef
from nemo_evaluator.jobs.agent_compiler import _secret_refs, compile_agent_eval_job
from nemo_evaluator.jobs.agent_evaluate import (
    REGISTERED_AGENT_HARBOR_IMPORT_PATH,
    AgentEvalJob,
    _resolve_registered_agent,
)
from nemo_evaluator.jobs.agent_spec import (
    AgentEvalSpec,
    FabricRunnerTarget,
    HarborRunnerTarget,
    registered_agent_name,
    target_agent_identity,
)
from nemo_evaluator.jobs.environment_stage import ENVIRONMENT_STORAGE_DIR
from nemo_evaluator_sdk.values import SecretRef
from nemo_helix_plugin.agents.types import EnvironmentSpecInline, McpFulfillment
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.job_context import JobContext, StoragePaths
from nemo_helix_plugin.job_results import LocalJobResults
from nemo_helix_plugin.sdk import AsyncNeMoHelix
from pydantic import ValidationError
from pytest_mock import MockerFixture

pytestmark = pytest.mark.usefixtures("_platform_base_url")

_AGENT = AgentRef(root="calculator-agent")


@pytest.fixture
def _platform_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    # The agents plugin binds the agent's models to the Inference Gateway of the platform it runs in.
    monkeypatch.setenv("NHX_BASE_URL", "http://platform.test")


def _async_platform() -> AsyncNeMoHelix:
    return AsyncNeMoHelix(
        base_url="http://platform.test",
        workspace="default",
        http_client=AsyncMock(spec=httpx.AsyncClient),
    )


def _calculator_config() -> dict[str, Any]:
    """The shipped calculator example, as ``nemo agents create`` stores it."""
    return {
        "config_format": "nemo-agents-spec-v1",
        "name": "calculator-agent",
        "description": "Calculator agent executed with DeepAgents",
        "instructions": {"system": {"content": "You are a concise calculator agent."}},
        "default_harness": "deepagents",
        "harnesses": {"deepagents": {"kind": "deepagents", "settings": {"deepagents": {}}}},
        "models": {
            "default": {
                "provider": "nvidia",
                "model": "nvidia-nemotron-3-5-lightning-30b-a3b",
                "api_key_env": "NVIDIA_API_KEY",
            }
        },
    }


def _calculator_with_mcp_config() -> dict[str, Any]:
    """The calculator example's MCP variant: it *declares* a calculator server the environment must provide."""
    config = _calculator_config()
    config["mcp"] = {
        "servers": {"calculator": {"transport": "streamable-http", "url": "http://calculator.internal/mcp"}}
    }
    return config


def _not_found() -> NotFoundError:
    return NotFoundError(httpx.Response(404, request=httpx.Request("GET", "http://platform.test/x")))


def _platform(mocker: MockerFixture, agent: object, *, ethos_fileset: bool = False) -> Any:
    """Agents and Files clients as ``client_from_platform`` hands them out, by requested client class."""
    response = mocker.Mock()
    response.data.return_value = agent
    agents = mocker.Mock()
    agents.get_agent = AsyncMock(return_value=response)
    files = mocker.Mock()
    files.get_fileset = AsyncMock(return_value=mocker.Mock()) if ethos_fileset else AsyncMock(side_effect=_not_found())

    def by_class(_platform: object, client_cls: type) -> Any:
        return agents if client_cls.__name__ == "AsyncAgentsClient" else files

    mocker.patch("nemo_evaluator.jobs.agent_evaluate.client_from_platform", side_effect=by_class)
    return agents


def _agent(config: dict[str, Any] | None = None, *, config_format: str = "nemo-agents-spec-v1") -> Any:
    return SimpleNamespace(name="calculator-agent", config_format=config_format, config=config or _calculator_config())


async def _resolve(target: FabricRunnerTarget | HarborRunnerTarget, workspace: str = "dev") -> Any:
    return await _resolve_registered_agent(target, workspace=workspace, async_sdk=_async_platform())


# --- Fabric ------------------------------------------------------------------------------------------


async def test_fabric_target_by_agent_becomes_the_config_a_deployment_would_run(mocker: MockerFixture) -> None:
    agents = _platform(mocker, _agent())

    resolved = await _resolve(FabricRunnerTarget(agent=_AGENT, timeout_s=120))

    agents.get_agent.assert_awaited_once_with(workspace="dev", name="calculator-agent")
    assert isinstance(resolved, FabricRunnerTarget)
    assert resolved.agent is None and resolved.environment is None  # the ref is consumed, not carried
    assert resolved.timeout_s == 120
    assert resolved.config is not None
    # The platform agent.yaml became a Fabric config: harness selected by adapter id, ...
    assert resolved.config["harness"]["adapter_id"] == "nvidia.fabric.langchain.deepagents"
    # ... and the model bound to the *agent's* workspace gateway, exactly as a deployment would be.
    model = resolved.config["models"]["default"]
    assert model["model"] == "nvidia-nemotron-3-5-lightning-30b-a3b"
    assert model["base_url"] == "http://platform.test/apis/inference-gateway/v2/workspaces/dev/openai/-/v1"
    # No credential travels: the gateway authenticates upstream from Secrets, the harness gets a placeholder.
    assert resolved.config["environment"]["env"]["NVIDIA_API_KEY"] == "not-used"
    assert resolved.env_secrets == {}
    assert resolved.agent_files is None  # config-only agent: nothing to stage


async def test_a_qualified_ref_names_the_agents_workspace(mocker: MockerFixture) -> None:
    agents = _platform(mocker, _agent())

    resolved = await _resolve(FabricRunnerTarget(agent=AgentRef(root="shared/calculator-agent")))

    agents.get_agent.assert_awaited_once_with(workspace="shared", name="calculator-agent")
    assert isinstance(resolved, FabricRunnerTarget) and resolved.config is not None
    assert "/workspaces/shared/" in resolved.config["models"]["default"]["base_url"]


async def test_an_agent_with_an_ethos_fileset_records_it_for_staging(mocker: MockerFixture) -> None:
    config = _calculator_config()
    config["skills"] = {"paths": ["skills/arithmetic"]}
    _platform(mocker, _agent(config), ethos_fileset=True)

    resolved = await _resolve(FabricRunnerTarget(agent=_AGENT))

    assert isinstance(resolved, FabricRunnerTarget)
    assert resolved.agent_files is not None and resolved.agent_files.root == "dev/calculator-agent-ethos"


async def test_skills_without_an_ethos_fileset_are_a_submit_error(mocker: MockerFixture) -> None:
    config = _calculator_config()
    config["skills"] = {"paths": ["skills/arithmetic"]}
    _platform(mocker, _agent(config), ethos_fileset=False)

    with pytest.raises(ValueError, match="references skills but has no Ethos FileSet"):
        await _resolve(FabricRunnerTarget(agent=_AGENT))


async def test_a_missing_agent_is_a_submit_error(mocker: MockerFixture) -> None:
    agents = _platform(mocker, _agent())
    agents.get_agent = AsyncMock(side_effect=_not_found())

    with pytest.raises(ValueError, match="registered agent dev/missing does not exist"):
        await _resolve(FabricRunnerTarget(agent=AgentRef(root="missing")))


async def test_a_legacy_nat_agent_is_rejected_rather_than_guessed_at(mocker: MockerFixture) -> None:
    _platform(mocker, _agent({"workflow": {}}, config_format="nat-workflow-v1"))

    with pytest.raises(ValueError, match="nat-workflow-v1"):
        await _resolve(FabricRunnerTarget(agent=_AGENT))


async def test_inline_targets_pass_through_untouched() -> None:
    target = FabricRunnerTarget(config={"harness": {"adapter_id": "nvidia.fabric.codex"}})
    assert await _resolve_registered_agent(target, workspace="dev", async_sdk=None) is target
    assert await _resolve_registered_agent(None, workspace="dev", async_sdk=None) is None


# --- Environment overlay ------------------------------------------------------------------------------


async def test_environment_spec_reshapes_the_agent_exactly_as_a_deployment_would(mocker: MockerFixture) -> None:
    _platform(mocker, _agent(_calculator_with_mcp_config()))
    environment = EnvironmentSpecInline(
        env={"EVAL_MODE": "mock"},
        secrets={"CALC_TOKEN": "dev/calc-token"},
        mcp={"calculator": McpFulfillment(url="http://mock-calculator.test/mcp")},
    )

    resolved = await _resolve(FabricRunnerTarget(agent=_AGENT, environment=environment))

    assert isinstance(resolved, FabricRunnerTarget) and resolved.config is not None
    # The declared MCP server now points at the mock. (Per-server ``env`` is a stdio-only Fabric
    # setting; for an HTTP server a fulfilment carrying it fails translation exactly as a deployment would.)
    assert resolved.config["mcp"]["servers"]["calculator"]["url"] == "http://mock-calculator.test/mcp"
    # Process env is layered on top of the gateway placeholder the resolution injected.
    env = resolved.config["environment"]["env"]
    assert env["EVAL_MODE"] == "mock"
    assert env["NVIDIA_API_KEY"] == "not-used"
    # Secret *refs* travel on the target; the compiler turns them into from_secret job env.
    assert resolved.env_secrets == {"CALC_TOKEN": SecretRef(root="dev/calc-token")}
    assert "CALC_TOKEN" not in env


async def test_environment_spec_only_fulfils_servers_the_agent_declares(mocker: MockerFixture) -> None:
    _platform(mocker, _agent())
    environment = EnvironmentSpecInline(mcp={"exfil": McpFulfillment(url="http://attacker.test/mcp")})

    resolved = await _resolve(FabricRunnerTarget(agent=_AGENT, environment=environment))

    assert isinstance(resolved, FabricRunnerTarget) and resolved.config is not None
    # Mock what exists; an environment cannot hand the agent a tool it never had.
    assert resolved.config["mcp"]["servers"] == {}


async def test_conflicting_environment_secret_bindings_are_a_submit_error(mocker: MockerFixture) -> None:
    _platform(mocker, _agent())
    # Bound both as a secret ref and as plaintext env: the merge refuses rather than picking an order.
    environment = EnvironmentSpecInline(env={"CALC_TOKEN": "plain"}, secrets={"CALC_TOKEN": "dev/calc-token"})

    with pytest.raises(ValueError, match="cannot be run through Fabric"):
        await _resolve(FabricRunnerTarget(agent=_AGENT, environment=environment))


# --- Harbor ------------------------------------------------------------------------------------------


async def test_harbor_target_by_agent_selects_the_installed_fabric_agent_with_the_whole_config(
    mocker: MockerFixture,
) -> None:
    _platform(mocker, _agent())

    resolved = await _resolve(HarborRunnerTarget(agent=_AGENT, n_attempts=2, agent_kwargs={"fabric_workspace": "/app"}))

    assert isinstance(resolved, HarborRunnerTarget)
    assert resolved.agent is None
    assert resolved.agent_import_path == REGISTERED_AGENT_HARBOR_IMPORT_PATH
    assert resolved.agent_name is None
    assert resolved.n_attempts == 2
    kwargs = resolved.agent_kwargs
    # The agent travels whole: identity, harness, gateway-bound model -- nothing flattened to keywords.
    config = kwargs["fabric_config"]
    assert isinstance(config, dict)
    assert config["metadata"]["name"] == "calculator-agent"
    assert config["harness"]["adapter_id"] == "nvidia.fabric.langchain.deepagents"
    assert "/inference-gateway/" in config["models"]["default"]["base_url"]
    # The harness extra is derived from the adapter and pinned to this service's Fabric version.
    package = kwargs["fabric_package"]
    assert isinstance(package, str) and package.startswith("nemo-fabric[deepagents,relay]==")
    # Caller-supplied install/run knobs survive.
    assert kwargs["fabric_workspace"] == "/app"


async def test_harbor_caller_may_pin_the_fabric_package(mocker: MockerFixture) -> None:
    _platform(mocker, _agent())

    resolved = await _resolve(
        HarborRunnerTarget(agent=_AGENT, agent_kwargs={"fabric_package": "nemo-fabric[deepagents]==9.9.9"})
    )

    assert isinstance(resolved, HarborRunnerTarget)
    assert resolved.agent_kwargs["fabric_package"] == "nemo-fabric[deepagents]==9.9.9"


async def test_harbor_environment_secrets_join_the_targets_own(mocker: MockerFixture) -> None:
    _platform(mocker, _agent())
    environment = EnvironmentSpecInline(secrets={"CALC_TOKEN": "dev/calc-token"})

    resolved = await _resolve(
        HarborRunnerTarget(agent=_AGENT, environment=environment, env_secrets={"OTHER": SecretRef(root="dev/other")})
    )

    assert isinstance(resolved, HarborRunnerTarget)
    assert resolved.env_secrets == {
        "OTHER": SecretRef(root="dev/other"),
        "CALC_TOKEN": SecretRef(root="dev/calc-token"),
    }


def test_harbor_fabric_config_is_checked_by_credential_shape_not_key_name() -> None:
    # A registered agent's config legitimately *names* a credential variable and carries the gateway
    # placeholder; neither is a credential. An issued token in it is.
    config = {
        "models": {"default": {"api_key_env": "NVIDIA_API_KEY"}},
        "environment": {"env": {"NVIDIA_API_KEY": "not-used"}},
    }
    HarborRunnerTarget(agent_import_path="x:Y", agent_kwargs={"fabric_config": config})

    leaked = {"environment": {"env": {"NVIDIA_API_KEY": "nvapi-" + "a" * 60}}}
    with pytest.raises(ValidationError, match="look like plaintext credentials"):
        HarborRunnerTarget(agent_import_path="x:Y", agent_kwargs={"fabric_config": leaked})


# --- Spec boundary -------------------------------------------------------------------------------------


def test_fabric_target_needs_exactly_one_of_config_or_agent() -> None:
    with pytest.raises(ValidationError, match="exactly one of `config`"):
        FabricRunnerTarget()
    with pytest.raises(ValidationError, match="exactly one of `config`"):
        FabricRunnerTarget(config={"harness": {"adapter_id": "x"}}, agent=_AGENT)


def test_a_registered_agents_model_is_not_overridable() -> None:
    # A different model is a different registered agent.
    with pytest.raises(ValidationError, match="different model is a different registered agent"):
        FabricRunnerTarget(agent=_AGENT, model="nvidia/other")
    with pytest.raises(ValidationError, match="part of what it is"):
        HarborRunnerTarget(agent=_AGENT, agent_model_name="nvidia/other")


def test_environment_requires_a_registered_agent() -> None:
    with pytest.raises(ValidationError, match="`environment` applies to a registered `agent`"):
        FabricRunnerTarget(config={"harness": {"adapter_id": "x"}}, environment=EnvironmentSpecInline())
    with pytest.raises(ValidationError, match="`environment` applies to a registered `agent`"):
        HarborRunnerTarget(environment=EnvironmentSpecInline())


def test_harbor_registered_agent_owns_the_harbor_agent_selection() -> None:
    with pytest.raises(ValidationError, match="`agent_import_path` cannot be combined"):
        HarborRunnerTarget(agent=_AGENT, agent_import_path="x:Y")
    with pytest.raises(ValidationError, match="derived from `agent`"):
        HarborRunnerTarget(agent=_AGENT, agent_kwargs={"fabric_config": {}})


@pytest.mark.parametrize("ref", ["", "a/b/c", "/calc", "calc/", "has space", "ws/name#rev"])
def test_agent_ref_is_validated_at_the_spec_boundary(ref: str) -> None:
    # Same charset and shape as MetricRef: a malformed ref is a 422 on submit, not a lookup failure.
    with pytest.raises(ValidationError):
        FabricRunnerTarget.model_validate({"agent": ref})


def test_a_registered_agent_names_the_agent_for_publication() -> None:
    assert registered_agent_name(FabricRunnerTarget(agent=AgentRef(root="shared/calc"))) == "calc"
    assert target_agent_identity(FabricRunnerTarget(agent=AgentRef(root="shared/calc"))) == ("calc", None)
    assert target_agent_identity(HarborRunnerTarget(agent=AgentRef(root="calc"))) == ("calc", None)
    assert registered_agent_name(FabricRunnerTarget(config={"harness": {"adapter_id": "x"}})) is None


def test_canonical_spec_refuses_an_unresolved_registered_agent() -> None:
    for target in ({"kind": "fabric", "agent": "calc"}, {"kind": "harbor", "agent": "calc"}):
        with pytest.raises(ValidationError, match="must be resolved before run"):
            AgentEvalSpec.model_validate({"tasks": [{"id": "t", "intent": "x", "metrics": []}], "target": target})


# --- Job side --------------------------------------------------------------------------------------------


def _spec(target: dict[str, Any]) -> AgentEvalSpec:
    return AgentEvalSpec.model_validate({"tasks": [{"id": "t", "intent": "x", "metrics": []}], "target": target})


def test_fabric_env_secrets_reach_the_job_environment_through_the_compiler() -> None:
    spec = _spec(
        {
            "kind": "fabric",
            "config": {"harness": {"adapter_id": "nvidia.fabric.langchain.deepagents"}},
            "env_secrets": {"CALC_TOKEN": "dev/calc-token"},
        }
    )
    assert list(_secret_refs(spec)) == [("CALC_TOKEN", "dev/calc-token")]


def test_agent_files_add_a_staging_step_before_the_evaluation() -> None:
    spec = _spec(
        {
            "kind": "fabric",
            "config": {"harness": {"adapter_id": "nvidia.fabric.langchain.deepagents"}},
            "agent_files": "dev/calculator-agent-ethos",
        }
    )
    job = compile_agent_eval_job(spec, use_subprocess=True)
    assert [step.name for step in job.steps] == ["stage-environment", "agent-evaluate"]
    assert job.steps[0].config == {"environment": "dev/calculator-agent-ethos"}


def _job_context(tmp_path: Path) -> JobContext:
    storage = StoragePaths(ephemeral=tmp_path / "ephemeral", persistent=tmp_path / "persistent")
    storage.ephemeral.mkdir()
    storage.persistent.mkdir()
    return JobContext(workspace="dev", storage=storage, results=LocalJobResults(root=storage.persistent / "results"))


def _fabric_target_with_files() -> FabricRunnerTarget:
    return FabricRunnerTarget.model_validate(
        {
            "config": {"harness": {"adapter_id": "nvidia.fabric.langchain.deepagents"}},
            "agent_files": "dev/calculator-agent-ethos",
        }
    )


def test_staged_agent_files_become_the_fabric_base_dir(tmp_path: Path) -> None:
    ctx = _job_context(tmp_path)
    (ctx.storage.persistent / ENVIRONMENT_STORAGE_DIR).mkdir()
    runtime, _, _ = AgentEvalJob._resolve_target(_fabric_target_with_files(), ctx)
    assert getattr(runtime, "_base_dir") == ctx.storage.persistent / ENVIRONMENT_STORAGE_DIR


def test_unstaged_agent_files_fail_before_the_run(tmp_path: Path) -> None:
    ctx = _job_context(tmp_path)
    with pytest.raises(ValueError, match="were not staged"):
        AgentEvalJob._resolve_target(_fabric_target_with_files(), ctx)


def test_harbor_timeout_multipliers_reach_the_runtime(tmp_path: Path) -> None:
    ctx = _job_context(tmp_path)
    target = HarborRunnerTarget(
        agent_import_path="x:Y", agent_setup_timeout_multiplier=12.0, agent_timeout_multiplier=5.0
    )
    runtime, _, _ = AgentEvalJob._resolve_target(target, ctx)
    config = getattr(runtime, "_config")
    assert (config.agent_setup_timeout_multiplier, config.agent_timeout_multiplier) == (12.0, 5.0)
