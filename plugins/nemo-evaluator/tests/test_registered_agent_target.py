# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A registered platform agent as the source of a Fabric runner target."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from nemo_evaluator.api.schemas import AgentRef
from nemo_evaluator.filesets import FilesetRef
from nemo_evaluator.jobs.agent_compiler import _secret_refs, compile_agent_eval_job
from nemo_evaluator.jobs.agent_evaluate import AgentEvalJob, _resolve_registered_agent
from nemo_evaluator.jobs.agent_spec import (
    AgentEvalInputSpec,
    AgentEvalSpec,
    FabricConfigSource,
    FabricRegisteredAgentSource,
    FabricRunnerTarget,
    registered_agent_files,
    registered_agent_name,
    target_agent_identity,
)
from nemo_evaluator.jobs.environment_stage import ENVIRONMENT_STORAGE_DIR
from nemo_evaluator.shared.metric_bundles.bundles import bundle_metric
from nemo_evaluator.shared.metric_bundles.inline import InlineMetricBundlePackager
from nemo_evaluator_sdk import ExactMatchMetric
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
_INLINE = FabricConfigSource(config={"harness": {"adapter_id": "x"}})


def _by_agent(
    agent: AgentRef = _AGENT, environment: EnvironmentSpecInline | None = None, **kwargs: Any
) -> FabricRunnerTarget:
    return FabricRunnerTarget(source=FabricRegisteredAgentSource(agent=agent, environment=environment), **kwargs)


#: A canonical task snapshot, in the shape `AgentEvalSpec.tasks` carries after submission.
_RESOLVED_TASK: dict[str, Any] = {"id": "t", "spec": {"kind": "evaluator", "intent": "x", "inputs": {}, "metrics": []}}


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


async def _resolve(target: FabricRunnerTarget, workspace: str = "dev") -> Any:
    return await _resolve_registered_agent(target, workspace=workspace, async_sdk=_async_platform())


# --- Fabric ------------------------------------------------------------------------------------------


async def test_fabric_target_by_agent_becomes_the_config_a_deployment_would_run(mocker: MockerFixture) -> None:
    agents = _platform(mocker, _agent())

    resolved = await _resolve(_by_agent(timeout_s=120))

    agents.get_agent.assert_awaited_once_with(workspace="dev", name="calculator-agent")
    assert isinstance(resolved, FabricRunnerTarget)
    assert isinstance(resolved.source, FabricRegisteredAgentSource)
    assert resolved.source.agent == AgentRef(root="dev/calculator-agent")  # kept, qualified, as provenance
    assert resolved.timeout_s == 120
    assert resolved.config is not None and resolved.config is resolved.resolved_config
    # The platform agent.yaml became a Fabric config: harness selected by adapter id, ...
    assert resolved.config["harness"]["adapter_id"] == "nvidia.fabric.langchain.deepagents"
    # ... and the model bound to the *agent's* workspace gateway, exactly as a deployment would be.
    model = resolved.config["models"]["default"]
    assert model["model"] == "nvidia-nemotron-3-5-lightning-30b-a3b"
    assert model["base_url"] == "http://platform.test/apis/inference-gateway/v2/workspaces/dev/openai/-/v1"
    # No credential travels: the gateway authenticates upstream from Secrets, the harness gets a placeholder.
    assert resolved.config["environment"]["env"]["NVIDIA_API_KEY"] == "not-used"
    assert resolved.env_secrets == {}
    assert registered_agent_files(resolved) is None  # config-only agent: nothing to stage


async def test_a_qualified_ref_names_the_agents_workspace(mocker: MockerFixture) -> None:
    agents = _platform(mocker, _agent())

    resolved = await _resolve(_by_agent(AgentRef(root="shared/calculator-agent")))

    agents.get_agent.assert_awaited_once_with(workspace="shared", name="calculator-agent")
    assert isinstance(resolved, FabricRunnerTarget) and resolved.config is not None
    assert "/workspaces/shared/" in resolved.config["models"]["default"]["base_url"]


async def test_an_agent_whose_config_refers_to_files_stages_its_ethos_fileset(mocker: MockerFixture) -> None:
    config = _calculator_config()
    config["skills"] = {"paths": ["skills/arithmetic"]}
    _platform(mocker, _agent(config), ethos_fileset=True)

    resolved = await _resolve(_by_agent())

    assert isinstance(resolved, FabricRunnerTarget)
    assert registered_agent_files(resolved) == FilesetRef(root="dev/calculator-agent-ethos")


async def test_skills_without_an_ethos_fileset_are_a_submit_error(mocker: MockerFixture) -> None:
    config = _calculator_config()
    config["skills"] = {"paths": ["skills/arithmetic"]}
    _platform(mocker, _agent(config), ethos_fileset=False)

    with pytest.raises(ValueError, match="refers to files by relative path but has no Ethos FileSet"):
        await _resolve(_by_agent())


async def test_a_missing_agent_is_a_submit_error(mocker: MockerFixture) -> None:
    agents = _platform(mocker, _agent())
    agents.get_agent = AsyncMock(side_effect=_not_found())

    with pytest.raises(ValueError, match="registered agent dev/missing does not exist"):
        await _resolve(_by_agent(AgentRef(root="missing")))


async def test_a_legacy_nat_agent_is_rejected_rather_than_guessed_at(mocker: MockerFixture) -> None:
    _platform(mocker, _agent({"workflow": {}}, config_format="nat-workflow-v1"))

    with pytest.raises(ValueError, match="nat-workflow-v1"):
        await _resolve(_by_agent())


async def test_the_published_agent_name_survives_resolution(mocker: MockerFixture) -> None:
    """The canonical spec demands an Intake agent name; the kept ref supplies it without a pin."""
    _platform(mocker, _agent())
    metric = bundle_metric(
        ExactMatchMetric(reference="{{reference.expected}}", candidate="{{sample.output_text}}"),
        InlineMetricBundlePackager(),
    ).model_dump(mode="json")
    input_spec = AgentEvalInputSpec.model_validate(
        {
            "tasks": [{"id": "t", "intent": "x", "inputs": {"instruction": "hi"}, "metrics": [metric]}],
            "target": {"kind": "fabric", "source": {"agent": "calculator-agent"}},
            "publication": {"intake": {"evaluation_id": "eval-1"}},
        }
    )

    spec = await AgentEvalJob.to_spec(
        input_spec, workspace="dev", entity_client=None, async_sdk=_async_platform(), is_local=True
    )

    assert isinstance(spec, AgentEvalSpec)
    assert isinstance(spec.target, FabricRunnerTarget) and spec.target.config is not None
    assert target_agent_identity(spec.target) == ("calculator-agent", None)


async def test_inline_targets_pass_through_untouched() -> None:
    target = FabricRunnerTarget(source=_INLINE)
    assert await _resolve_registered_agent(target, workspace="dev", async_sdk=None) is target
    resolved = _by_agent(AgentRef(root="dev/calc"), resolved_config={"harness": {"adapter_id": "x"}})
    assert await _resolve_registered_agent(resolved, workspace="dev", async_sdk=None) is resolved
    assert await _resolve_registered_agent(None, workspace="dev", async_sdk=None) is None


# --- Environment overlay ------------------------------------------------------------------------------


async def test_environment_spec_reshapes_the_agent_exactly_as_a_deployment_would(mocker: MockerFixture) -> None:
    _platform(mocker, _agent(_calculator_with_mcp_config()))
    environment = EnvironmentSpecInline(
        env={"EVAL_MODE": "mock"},
        secrets={"CALC_TOKEN": "dev/calc-token"},
        mcp={"calculator": McpFulfillment(url="http://mock-calculator.test/mcp")},
    )

    resolved = await _resolve(_by_agent(environment=environment))

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

    resolved = await _resolve(_by_agent(environment=environment))

    assert isinstance(resolved, FabricRunnerTarget) and resolved.config is not None
    # Mock what exists; an environment cannot hand the agent a tool it never had.
    assert resolved.config["mcp"]["servers"] == {}


async def test_conflicting_environment_secret_bindings_are_a_submit_error(mocker: MockerFixture) -> None:
    _platform(mocker, _agent())
    # Bound both as a secret ref and as plaintext env: the merge refuses rather than picking an order.
    environment = EnvironmentSpecInline(env={"CALC_TOKEN": "plain"}, secrets={"CALC_TOKEN": "dev/calc-token"})

    with pytest.raises(ValueError, match="cannot be run through Fabric"):
        await _resolve(_by_agent(environment=environment))


# --- Spec boundary -------------------------------------------------------------------------------------


def test_a_fabric_source_is_one_shape_or_the_other() -> None:
    """The wire type cannot express a config and an agent at once; the schema, not a validator, says so."""
    for source in ({}, {"config": {"harness": {}}, "agent": "calc"}, {"agent": "calc", "model": "nvidia/other"}):
        with pytest.raises(ValidationError):
            FabricRunnerTarget.model_validate({"source": source})
    with pytest.raises(ValidationError):
        FabricRunnerTarget.model_validate({"source": {"config": {"harness": {}}, "environment": {}}})


def test_the_legacy_flat_inline_shape_is_lifted_into_source_with_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    """One release of tolerance for specs and persisted jobs written before `source` existed."""
    legacy = {"kind": "fabric", "config": {"harness": {"adapter_id": "x"}}, "model": "p/m", "timeout_s": 30}
    with caplog.at_level("WARNING", logger="nemo_evaluator.jobs.agent_spec"):
        target = FabricRunnerTarget.model_validate(legacy)
    assert target.source == FabricConfigSource(config={"harness": {"adapter_id": "x"}}, model="p/m")
    assert target.timeout_s == 30
    assert "deprecated" in caplog.text
    # Only the inline shape is lifted; the pre-release flat `agent` shape never shipped.
    with pytest.raises(ValidationError):
        FabricRunnerTarget.model_validate({"kind": "fabric", "agent": "calc"})


def test_resolved_config_is_derived_never_submitted() -> None:
    with pytest.raises(ValidationError, match="an inline `config` needs none"):
        FabricRunnerTarget(source=_INLINE, resolved_config={"harness": {}})
    submitted = {"kind": "fabric", "source": {"agent": "calc"}, "resolved_config": {"harness": {}}}
    with pytest.raises(ValidationError, match="not the submitter"):
        AgentEvalInputSpec.model_validate({"tasks": [{"id": "t", "intent": "x", "inputs": {}}], "target": submitted})


def test_staged_files_are_derived_from_a_resolved_agent_never_named_by_the_submitter() -> None:
    """Only a qualified ``agent`` plus a config with relative paths yields a FileSet; nothing is settable."""
    with_skills = {"harness": {"adapter_id": "x"}, "skills": {"paths": ["skills/a"]}}
    assert registered_agent_files(FabricRunnerTarget(source=FabricConfigSource(config=with_skills))) is None
    assert registered_agent_files(_by_agent(AgentRef(root="calc"), resolved_config=with_skills)) is None
    assert registered_agent_files(_by_agent(AgentRef(root="ws/calc"), resolved_config={"harness": {}})) is None
    assert registered_agent_files(_by_agent(AgentRef(root="ws/calc"), resolved_config=with_skills)) == FilesetRef(
        root="ws/calc-ethos"
    )
    with_discovery = {"harness": {"adapter_id": "x"}, "discovery": {"local_paths": ["adapters"]}}
    assert registered_agent_files(_by_agent(AgentRef(root="ws/calc"), resolved_config=with_discovery)) is not None
    assert "agent_files" not in FabricRunnerTarget.model_fields


@pytest.mark.parametrize("ref", ["", "a/b/c", "/calc", "calc/", "has space", "ws/name#rev"])
def test_agent_ref_is_validated_at_the_spec_boundary(ref: str) -> None:
    # Same charset and shape as MetricRef: a malformed ref is a 422 on submit, not a lookup failure.
    with pytest.raises(ValidationError):
        FabricRunnerTarget.model_validate({"source": {"agent": ref}})


def test_a_registered_agent_names_the_agent_for_publication() -> None:
    assert registered_agent_name(_by_agent(AgentRef(root="shared/calc"))) == "calc"
    assert target_agent_identity(_by_agent(AgentRef(root="shared/calc"))) == ("calc", None)
    assert registered_agent_name(FabricRunnerTarget(source=_INLINE)) is None
    assert target_agent_identity(FabricRunnerTarget(source=FabricConfigSource(config={}, model="p/m"))) == (None, "p/m")


def test_canonical_spec_refuses_an_unresolved_registered_agent() -> None:
    with pytest.raises(ValidationError, match="must be resolved before run"):
        AgentEvalSpec.model_validate(
            {"tasks": [_RESOLVED_TASK], "target": {"kind": "fabric", "source": {"agent": "calc"}}}
        )
    resolved = {"kind": "fabric", "source": {"agent": "dev/calc"}, "resolved_config": {"harness": {"adapter_id": "x"}}}
    AgentEvalSpec.model_validate({"tasks": [_RESOLVED_TASK], "target": resolved})


# --- Job side --------------------------------------------------------------------------------------------


def _spec(target: dict[str, Any]) -> AgentEvalSpec:
    return AgentEvalSpec.model_validate({"tasks": [_RESOLVED_TASK], "target": target})


def test_fabric_env_secrets_reach_the_job_environment_through_the_compiler() -> None:
    spec = _spec(
        {
            "kind": "fabric",
            "source": {"config": {"harness": {"adapter_id": "nvidia.fabric.langchain.deepagents"}}},
            "env_secrets": {"CALC_TOKEN": "dev/calc-token"},
        }
    )
    assert list(_secret_refs(spec)) == [("CALC_TOKEN", "dev/calc-token")]


def test_a_resolved_agent_with_files_adds_a_staging_step_before_the_evaluation() -> None:
    spec = _spec(_resolved_target_with_files())
    job = compile_agent_eval_job(spec, use_subprocess=True)
    assert [step.name for step in job.steps] == ["stage-environment", "agent-evaluate"]
    assert job.steps[0].config == {"environment": "dev/calculator-agent-ethos"}

    config_only = _spec(
        {
            "kind": "fabric",
            "source": {"agent": "dev/calculator-agent"},
            "resolved_config": {"harness": {"adapter_id": "x"}},
        }
    )
    assert [step.name for step in compile_agent_eval_job(config_only, use_subprocess=True).steps] == ["agent-evaluate"]


def _resolved_target_with_files() -> dict[str, Any]:
    return {
        "kind": "fabric",
        "source": {"agent": "dev/calculator-agent"},
        "resolved_config": {
            "harness": {"adapter_id": "nvidia.fabric.langchain.deepagents"},
            "skills": {"paths": ["skills/a"]},
        },
    }


def _job_context(tmp_path: Path) -> JobContext:
    storage = StoragePaths(ephemeral=tmp_path / "ephemeral", persistent=tmp_path / "persistent")
    storage.ephemeral.mkdir()
    storage.persistent.mkdir()
    return JobContext(workspace="dev", storage=storage, results=LocalJobResults(root=storage.persistent / "results"))


def _fabric_target_with_files() -> FabricRunnerTarget:
    return FabricRunnerTarget.model_validate(_resolved_target_with_files())


def test_staged_agent_files_become_the_fabric_base_dir(tmp_path: Path) -> None:
    ctx = _job_context(tmp_path)
    (ctx.storage.persistent / ENVIRONMENT_STORAGE_DIR).mkdir()
    runtime, _, _ = AgentEvalJob._resolve_target(_fabric_target_with_files(), ctx)
    assert getattr(runtime, "_base_dir") == ctx.storage.persistent / ENVIRONMENT_STORAGE_DIR


def test_unstaged_agent_files_fail_before_the_run(tmp_path: Path) -> None:
    ctx = _job_context(tmp_path)
    with pytest.raises(ValueError, match="were not staged"):
        AgentEvalJob._resolve_target(_fabric_target_with_files(), ctx)
