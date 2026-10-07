# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A registered platform agent as the source of a Fabric or Harbor runner target."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from nemo_evaluator.api.schemas import AgentRef
from nemo_evaluator.filesets import FilesetRef
from nemo_evaluator.jobs.agent_compiler import _secret_refs, compile_agent_eval_job
from nemo_evaluator.jobs.agent_evaluate import AgentEvalJob
from nemo_evaluator.jobs.agent_spec import (
    REGISTERED_AGENT_HARBOR_IMPORT_PATH,
    AgentEvalInputSpec,
    AgentEvalSpec,
    FabricConfigSource,
    FabricRunnerTarget,
    HarborBuiltinAgentSource,
    HarborImportedAgentSource,
    HarborRunnerTarget,
    RegisteredAgentSource,
    registered_agent_config_needs_files,
    registered_agent_files,
    registered_agent_name,
    target_agent_identity,
)
from nemo_evaluator.jobs.environment_stage import ENVIRONMENT_STORAGE_DIR
from nemo_evaluator.jobs.fabric_harness_packages import FABRIC_ADAPTER_EXTRAS
from nemo_evaluator.jobs.registered_agent_resolution import resolve_registered_agent
from nemo_evaluator.shared.metric_bundles.bundles import bundle_metric
from nemo_evaluator.shared.metric_bundles.inline import InlineMetricBundlePackager
from nemo_evaluator_sdk import ExactMatchMetric
from nemo_evaluator_sdk.agent_eval.runtimes.harbor.runtime import HarborRuntimeConfig
from nemo_evaluator_sdk.values import SecretRef
from nemo_helix_plugin.agents.types import EnvironmentSpecInline, McpFulfillment
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.job_context import JobContext, StoragePaths
from nemo_helix_plugin.job_results import LocalJobResults
from pydantic import ValidationError
from pytest_mock import MockerFixture

pytestmark = pytest.mark.usefixtures("_platform_base_url")

_AGENT = AgentRef(root="calculator-agent")
_INLINE = FabricConfigSource(config={"harness": {"adapter_id": "x"}})


def _by_agent(
    agent: AgentRef = _AGENT, environment: EnvironmentSpecInline | None = None, **kwargs: Any
) -> FabricRunnerTarget:
    return FabricRunnerTarget(source=RegisteredAgentSource(agent=agent, environment=environment), **kwargs)


#: A canonical task snapshot, in the shape `AgentEvalSpec.tasks` carries after submission.
_RESOLVED_TASK: dict[str, Any] = {"id": "t", "spec": {"kind": "evaluator", "intent": "x", "inputs": {}, "metrics": []}}


@pytest.fixture
def _platform_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    # The agents plugin binds the agent's models to the Inference Gateway of the platform it runs in.
    monkeypatch.setenv("NHX_BASE_URL", "http://platform.test")


def _async_platform() -> AsyncNemoClient:
    return AsyncNemoClient(
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
    files.create_fileset = AsyncMock(return_value=mocker.Mock())
    files.upload_file = AsyncMock(return_value=mocker.Mock())
    files.delete_fileset = AsyncMock(return_value=mocker.Mock())

    async def fake_download(ref: FilesetRef, destination: str, **_: Any) -> Path:
        root = Path(destination) / ref.root
        (root / "skills" / "arithmetic").mkdir(parents=True)
        (root / "skills" / "arithmetic" / "SKILL.md").write_text("# add\n")
        return root

    mocker.patch("nemo_evaluator.jobs.agent_files_snapshot._download_fileset_ref", side_effect=fake_download)
    mocker.patch("nemo_evaluator.jobs.agent_files_snapshot.AsyncFilesetFileSystem")
    agents.files = files

    def by_class(_platform: object, client_cls: type) -> Any:
        return agents if client_cls.__name__ == "AsyncAgentsClient" else files

    mocker.patch("nemo_evaluator.jobs.registered_agent_resolution.client_from_platform", side_effect=by_class)
    mocker.patch("nemo_evaluator.jobs.agent_evaluate.client_from_platform", side_effect=by_class)
    return agents


def _agent(config: dict[str, Any] | None = None, *, config_format: str = "nemo-agents-spec-v1") -> Any:
    return SimpleNamespace(name="calculator-agent", config_format=config_format, config=config or _calculator_config())


async def _resolve(target: FabricRunnerTarget | HarborRunnerTarget, workspace: str = "dev") -> Any:
    return await resolve_registered_agent(target, workspace=workspace, async_sdk=_async_platform())


# --- Fabric ------------------------------------------------------------------------------------------


async def test_fabric_target_by_agent_becomes_the_config_a_deployment_would_run(mocker: MockerFixture) -> None:
    agents = _platform(mocker, _agent())

    resolved = await _resolve(_by_agent(timeout_s=120))

    agents.get_agent.assert_awaited_once_with(workspace="dev", name="calculator-agent")
    assert isinstance(resolved, FabricRunnerTarget)
    assert isinstance(resolved.source, RegisteredAgentSource)
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


async def test_an_agent_whose_config_refers_to_files_snapshots_its_ethos_fileset(mocker: MockerFixture) -> None:
    """The job stages a copy taken at submit, not the live `<agent>-ethos` FileSet a re-registration rewrites."""
    config = _calculator_config()
    config["skills"] = {"paths": ["skills/arithmetic"]}
    files = _platform(mocker, _agent(config), ethos_fileset=True).files

    resolved = await _resolve(_by_agent())

    assert isinstance(resolved, FabricRunnerTarget)
    snapshot = registered_agent_files(resolved)
    assert snapshot is not None and snapshot.root.startswith("dev/agent-files-")
    assert snapshot.root != "dev/calculator-agent-ethos"
    created = files.create_fileset.await_args.kwargs
    assert created["workspace"] == "dev" and f"dev/{created['body'].name}" == snapshot.root
    assert "snapshot of dev/calculator-agent-ethos" in created["body"].description
    uploaded = files.upload_file.await_args.kwargs
    assert (uploaded["name"], uploaded["path"], uploaded["content"]) == (
        created["body"].name,
        "skills/arithmetic/SKILL.md",
        b"# add\n",
    )


async def test_a_config_without_relative_paths_takes_no_snapshot(mocker: MockerFixture) -> None:
    files = _platform(mocker, _agent(), ethos_fileset=True).files

    resolved = await _resolve(_by_agent())

    assert isinstance(resolved, FabricRunnerTarget) and registered_agent_files(resolved) is None
    files.create_fileset.assert_not_awaited()


async def test_a_submission_that_fails_after_resolution_discards_its_snapshot(mocker: MockerFixture) -> None:
    """No job exists to clean up after a rejected submission, so `to_spec` deletes the copy it just took."""
    config = _calculator_config()
    config["skills"] = {"paths": ["skills/arithmetic"]}
    files = _platform(mocker, _agent(config), ethos_fileset=True).files
    mocker.patch("nemo_evaluator.jobs.agent_evaluate.validate_scoring", side_effect=ValueError("no scorer"))
    input_spec = AgentEvalInputSpec.model_validate(
        {"tasks": [{"id": "t", "intent": "x", "inputs": {}}], "target": {"kind": "fabric", "source": {"agent": "calc"}}}
    )

    with pytest.raises(ValueError, match="no scorer"):
        await AgentEvalJob.to_spec(
            input_spec, workspace="dev", entity_client=None, async_sdk=_async_platform(), is_local=False
        )

    deleted = files.delete_fileset.await_args.kwargs
    assert deleted == {"workspace": "dev", "name": files.create_fileset.await_args.kwargs["body"].name}


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
    assert await resolve_registered_agent(target, workspace="dev", async_sdk=None) is target
    resolved = _by_agent(AgentRef(root="dev/calc"), resolved_config={"harness": {"adapter_id": "x"}})
    assert await resolve_registered_agent(resolved, workspace="dev", async_sdk=None) is resolved
    assert await resolve_registered_agent(None, workspace="dev", async_sdk=None) is None


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


async def test_a_stdio_servers_bound_secret_is_named_in_its_env_as_a_template(mocker: MockerFixture) -> None:
    """The MCP SDK spawns stdio servers without the harness's environment; the server must name the variable."""
    config = _calculator_with_mcp_config()
    config["mcp"]["servers"]["calculator"] = {"transport": "stdio", "url": "/usr/bin/python3", "args": ["srv.py"]}
    config["mcp"]["servers"]["remote"] = {"transport": "streamable-http", "url": "http://calc.internal/mcp"}
    _platform(mocker, _agent(config))
    environment = EnvironmentSpecInline(
        mcp={
            "calculator": McpFulfillment(url="/usr/bin/python3", secrets={"CALC_TOKEN": "dev/calc-token"}),
            "remote": McpFulfillment(url="http://mock/mcp", secrets={"REMOTE_TOKEN": "dev/remote-token"}),
        }
    )

    resolved = await _resolve(_by_agent(environment=environment))

    assert isinstance(resolved, FabricRunnerTarget) and resolved.config is not None
    servers = resolved.config["mcp"]["servers"]
    assert servers["calculator"]["env"] == {"CALC_TOKEN": "${CALC_TOKEN}"}
    assert "env" not in servers["remote"]  # nothing to spawn; the secret still reaches the process env
    assert resolved.env_secrets == {
        "CALC_TOKEN": SecretRef(root="dev/calc-token"),
        "REMOTE_TOKEN": SecretRef(root="dev/remote-token"),
    }


def test_the_host_runtime_expands_stdio_env_templates_only_in_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _job_context(tmp_path)
    monkeypatch.setenv("CALC_TOKEN", "sekrit")
    config = {
        "harness": {"adapter_id": "nvidia.fabric.langchain.deepagents"},
        "mcp": {
            "servers": {
                "calculator": {
                    "transport": "stdio",
                    "url": "python3",
                    "env": {"CALC_TOKEN": "${CALC_TOKEN}", "OTHER": "${UNSET_VAR}"},
                }
            }
        },
    }
    target = FabricRunnerTarget(
        source=FabricConfigSource(config=config), env_secrets={"CALC_TOKEN": SecretRef(root="dev/calc-token")}
    )

    runtime, _, _ = AgentEvalJob._resolve_target(target, ctx)

    launched = getattr(runtime, "_config")
    assert launched["mcp"]["servers"]["calculator"]["env"] == {"CALC_TOKEN": "sekrit", "OTHER": "${UNSET_VAR}"}
    assert (
        target.config is not None
        and target.config["mcp"]["servers"]["calculator"]["env"]["CALC_TOKEN"] == "${CALC_TOKEN}"
    )


async def test_conflicting_environment_secret_bindings_are_a_submit_error(mocker: MockerFixture) -> None:
    _platform(mocker, _agent())
    # Bound both as a secret ref and as plaintext env: the merge refuses rather than picking an order.
    environment = EnvironmentSpecInline(env={"CALC_TOKEN": "plain"}, secrets={"CALC_TOKEN": "dev/calc-token"})

    with pytest.raises(ValueError, match="cannot be run through Fabric"):
        await _resolve(_by_agent(environment=environment))


async def test_the_submitters_env_secret_wins_over_the_agents_with_a_warning(
    mocker: MockerFixture, caplog: pytest.LogCaptureFixture
) -> None:
    """A name both bind keeps the target's ref: the submitter chose it for this run, and is told it overrode."""
    environment = EnvironmentSpecInline(secrets={"NVIDIA_API_KEY": "agent-key", "OTHER": "agent-other"})
    _platform(mocker, _agent())
    target = _by_agent(environment=environment, env_secrets={"NVIDIA_API_KEY": SecretRef(root="eval-key")})

    with caplog.at_level("WARNING", logger="nemo_evaluator.jobs.agent_evaluate"):
        resolved = await _resolve(target)

    assert isinstance(resolved, FabricRunnerTarget)
    assert resolved.env_secrets == {
        "NVIDIA_API_KEY": SecretRef(root="eval-key"),
        "OTHER": SecretRef(root="agent-other"),
    }
    assert any(
        "override registered agent dev/calculator-agent's binding of 1 environment" in r.getMessage()
        for r in caplog.records
    )
    assert not any(
        "eval-key" in r.getMessage() or "agent-key" in r.getMessage() or "NVIDIA_API_KEY" in r.getMessage()
        for r in caplog.records
    )


# --- Harbor ------------------------------------------------------------------------------------------


def _harbor_by_agent(
    agent: AgentRef = _AGENT, environment: EnvironmentSpecInline | None = None, **kwargs: Any
) -> HarborRunnerTarget:
    return HarborRunnerTarget(source=RegisteredAgentSource(agent=agent, environment=environment), **kwargs)


async def test_harbor_target_by_agent_selects_the_installed_fabric_agent_with_the_whole_config(
    mocker: MockerFixture,
) -> None:
    _platform(mocker, _agent())

    resolved = await _resolve(_harbor_by_agent(n_attempts=2, agent_kwargs={"fabric_workspace": "/app"}))

    assert isinstance(resolved, HarborRunnerTarget)
    assert isinstance(resolved.source, RegisteredAgentSource)
    assert resolved.source.agent == AgentRef(root="dev/calculator-agent")  # kept, qualified
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
    # The gateway placeholder stays out of the persisted kwargs; the agent re-adds it per trial from
    # ``api_key_env``, so the target still passes Harbor's credential check job-side.
    assert config["models"]["default"]["api_key_env"] == "NVIDIA_API_KEY"
    assert "NVIDIA_API_KEY" not in config["environment"]["env"]
    HarborRuntimeConfig(jobs_dir=Path("/tmp/x"), agent_import_path="x:Y", agent_kwargs=kwargs)
    # The harness extra is derived from the adapter and pinned to this service's Fabric version, and so is
    # the MCP client stack deepagents leaves unpinned (a clean install picks an `mcp` major it cannot import).
    package = kwargs["fabric_package"]
    assert isinstance(package, str) and package.startswith("nemo-fabric[deepagents,relay]==")
    assert f"mcp=={importlib.metadata.version('mcp')}" in package.split()
    assert f"langchain-mcp-adapters=={importlib.metadata.version('langchain-mcp-adapters')}" in package.split()
    # Caller-supplied install/run knobs survive.
    assert kwargs["fabric_workspace"] == "/app"
    # Resolution is idempotent: a resolved target is left alone.
    assert await _resolve(resolved) is resolved


def test_every_harness_extra_the_resolver_names_exists_in_the_installed_fabric() -> None:
    """`pip install nemo-fabric[<extra>]` only warns on an unknown extra; the container would run without its harness."""
    provided = set(importlib.metadata.distribution("nemo-fabric").metadata.get_all("Provides-Extra") or [])
    assert set(FABRIC_ADAPTER_EXTRAS.values()) <= provided, set(FABRIC_ADAPTER_EXTRAS.values()) - provided
    assert "relay" in provided


async def test_harbor_drops_stdio_env_templates_the_container_cannot_expand(mocker: MockerFixture) -> None:
    config = _calculator_with_mcp_config()
    config["mcp"]["servers"]["calculator"] = {"transport": "stdio", "url": "/usr/bin/python3", "args": ["srv.py"]}
    _platform(mocker, _agent(config))
    environment = EnvironmentSpecInline(
        mcp={
            "calculator": McpFulfillment(url="/usr/bin/python3", env={"MODE": "mock"}, secrets={"CALC_TOKEN": "dev/t"})
        }
    )

    resolved = await _resolve(_harbor_by_agent(environment=environment))

    assert isinstance(resolved, HarborRunnerTarget)
    server = resolved.agent_kwargs["fabric_config"]["mcp"]["servers"]["calculator"]
    assert server["env"] == {"MODE": "mock"}  # the plain env stays; the template that would arrive literal is gone
    assert resolved.env_secrets["CALC_TOKEN"] == SecretRef(root="dev/t")  # the process still gets the secret


async def test_harbor_caller_may_pin_the_fabric_package(mocker: MockerFixture) -> None:
    _platform(mocker, _agent())

    resolved = await _resolve(_harbor_by_agent(agent_kwargs={"fabric_package": "nemo-fabric[deepagents]==9.9.9"}))

    assert isinstance(resolved, HarborRunnerTarget)
    assert resolved.agent_kwargs["fabric_package"] == "nemo-fabric[deepagents]==9.9.9"


async def test_a_non_string_fabric_package_override_is_rejected_not_replaced(mocker: MockerFixture) -> None:
    """A list where a requirement string belongs used to be swapped for the derived default without a word."""
    _platform(mocker, _agent())

    with pytest.raises(
        ValueError, match="`agent_kwargs.fabric_package` must be a requirement string or null, not list"
    ):
        await _resolve(_harbor_by_agent(agent_kwargs={"fabric_package": ["nemo-fabric[deepagents]"]}))


async def test_a_harbor_fill_that_fails_after_the_snapshot_discards_it(mocker: MockerFixture) -> None:
    """Resolution owns the snapshot until a resolved target exists, so its own failures roll it back."""
    config = _calculator_config()
    config["skills"] = {"paths": ["skills/arithmetic"]}
    config["default_harness"] = "mystery"
    config["harnesses"] = {"mystery": {"kind": "deepagents", "settings": {}}}
    files = _platform(mocker, _agent(config), ethos_fileset=True).files
    mocker.patch(
        "nemo_evaluator.jobs.registered_agent_resolution.fabric_harness_package",
        side_effect=ValueError("no known Fabric package extra installs harness 'x'"),
    )

    with pytest.raises(ValueError, match="no known Fabric package extra"):
        await _resolve(_harbor_by_agent())

    files.create_fileset.assert_awaited_once()
    deleted = files.delete_fileset.await_args.kwargs
    assert deleted == {"workspace": "dev", "name": files.create_fileset.await_args.kwargs["body"].name}


async def test_harbor_environment_secrets_join_the_targets_own(mocker: MockerFixture) -> None:
    _platform(mocker, _agent())
    environment = EnvironmentSpecInline(secrets={"CALC_TOKEN": "dev/calc-token"})

    resolved = await _resolve(
        _harbor_by_agent(environment=environment, env_secrets={"OTHER": SecretRef(root="dev/other")})
    )

    assert isinstance(resolved, HarborRunnerTarget)
    assert resolved.env_secrets == {
        "OTHER": SecretRef(root="dev/other"),
        "CALC_TOKEN": SecretRef(root="dev/calc-token"),
    }


async def test_a_harbor_agent_whose_config_refers_to_files_snapshots_its_ethos_fileset(mocker: MockerFixture) -> None:
    config = _calculator_config()
    config["skills"] = {"paths": ["skills/arithmetic"]}
    files = _platform(mocker, _agent(config), ethos_fileset=True).files

    resolved = await _resolve(_harbor_by_agent())

    assert isinstance(resolved, HarborRunnerTarget)
    snapshot = registered_agent_files(resolved)
    assert snapshot is not None and snapshot.root == f"dev/{files.create_fileset.await_args.kwargs['body'].name}"


def test_harbor_fabric_config_may_name_a_credential_variable_but_not_hold_one() -> None:
    HarborRunnerTarget(
        source=HarborImportedAgentSource(import_path="x:Y"),
        agent_kwargs={"fabric_config": {"models": {"default": {"api_key_env": "K"}}}},
    )

    for value in ("nvapi-" + "a" * 60, "not-used"):
        leaked = {"environment": {"env": {"NVIDIA_API_KEY": value}}}
        with pytest.raises(ValidationError, match="look like plaintext credentials"):
            HarborRunnerTarget(
                source=HarborImportedAgentSource(import_path="x:Y"), agent_kwargs={"fabric_config": leaked}
            )


def test_a_registered_agent_is_one_of_the_harbor_sources_and_excludes_the_others() -> None:
    """The union admits a registered `agent` next to a built-in `name` or an `import_path`, never two of them."""
    for source in (
        {"name": "oracle", "agent": "calc"},
        {"import_path": "x:Y", "agent": "calc"},
        {"agent": "calc", "model_name": "m"},
    ):
        with pytest.raises(ValidationError):
            HarborRunnerTarget.model_validate({"source": source})
    registered = _harbor_by_agent(AgentRef(root="calc"))
    assert (registered.agent_name, registered.agent_import_path, registered.agent_model_name) == (
        None,
        REGISTERED_AGENT_HARBOR_IMPORT_PATH,
        None,
    )
    assert target_agent_identity(registered) == ("calc", None)
    assert HarborRunnerTarget(source=HarborBuiltinAgentSource(name="codex")).agent_import_path is None


def test_a_submitted_harbor_registered_agent_owns_its_fabric_config() -> None:
    tasks = [{"id": "t", "intent": "x", "inputs": {}}]
    target = {"kind": "harbor", "source": {"agent": "calc"}, "agent_kwargs": {"fabric_config": {}}}
    with pytest.raises(ValidationError, match="derived from the registered `agent`"):
        AgentEvalInputSpec.model_validate({"tasks": tasks, "target": target})
    AgentEvalInputSpec.model_validate({"tasks": tasks, "target": {**target, "source": {"import_path": "x:Y"}}})


def test_canonical_spec_refuses_an_unresolved_harbor_registered_agent() -> None:
    with pytest.raises(ValidationError, match="must be resolved before run"):
        AgentEvalSpec.model_validate(
            {"tasks": [_RESOLVED_TASK], "target": {"kind": "harbor", "source": {"agent": "calc"}}}
        )
    resolved = {
        "kind": "harbor",
        "source": {"agent": "dev/calc"},
        "agent_kwargs": {"fabric_config": {"harness": {"adapter_id": "x"}}},
    }
    AgentEvalSpec.model_validate({"tasks": [_RESOLVED_TASK], "target": resolved})


def test_harbor_timeout_multipliers_reach_the_runtime(tmp_path: Path) -> None:
    ctx = _job_context(tmp_path)
    target = HarborRunnerTarget(
        source=HarborImportedAgentSource(import_path="x:Y"),
        agent_setup_timeout_multiplier=12.0,
        agent_timeout_multiplier=5.0,
    )
    runtime, _, _ = AgentEvalJob._resolve_target(target, ctx)
    config = getattr(runtime, "_config")
    assert (config.agent_setup_timeout_multiplier, config.agent_timeout_multiplier) == (12.0, 5.0)


def test_staged_harbor_agent_files_become_the_fabric_config_bundle(tmp_path: Path) -> None:
    ctx = _job_context(tmp_path)
    target = HarborRunnerTarget.model_validate(
        {
            "source": {"agent": "dev/calculator-agent", "files": "dev/agent-files-0123abcd4567"},
            "agent_kwargs": {"fabric_config": {"harness": {"adapter_id": "x"}, "skills": {"paths": ["skills/a"]}}},
        }
    )
    with pytest.raises(ValueError, match="were not staged"):
        AgentEvalJob._resolve_target(target, ctx)
    (ctx.storage.persistent / ENVIRONMENT_STORAGE_DIR).mkdir()
    runtime, _, _ = AgentEvalJob._resolve_target(target, ctx)
    config = getattr(runtime, "_config")
    assert config.agent_kwargs["fabric_config_bundle"] == str(ctx.storage.persistent / ENVIRONMENT_STORAGE_DIR)
    spec = AgentEvalSpec.model_validate(
        {"tasks": [_RESOLVED_TASK], "target": {"kind": "harbor", **target.model_dump(mode="json", exclude={"kind"})}}
    )
    assert [step.name for step in compile_agent_eval_job(spec, use_subprocess=True).steps] == [
        "stage-environment",
        "agent-evaluate",
    ]


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


def test_staged_files_are_the_snapshot_resolution_took_never_named_by_the_submitter() -> None:
    """``source.files`` is resolution's output: read by staging, rejected on a submitted spec."""
    with_skills = {"harness": {"adapter_id": "x"}, "skills": {"paths": ["skills/a"]}}
    assert registered_agent_files(FabricRunnerTarget(source=FabricConfigSource(config=with_skills))) is None
    assert registered_agent_files(_by_agent(AgentRef(root="ws/calc"), resolved_config=with_skills)) is None
    snapshot = FilesetRef(root="ws/agent-files-0123abcd4567")
    resolved = FabricRunnerTarget(
        source=RegisteredAgentSource(agent=AgentRef(root="ws/calc"), files=snapshot), resolved_config=with_skills
    )
    assert registered_agent_files(resolved) == snapshot

    submitted = {"kind": "fabric", "source": {"agent": "calc", "files": "ws/agent-files-0123abcd4567"}}
    with pytest.raises(ValidationError, match="`source.files` is set by registered-agent resolution"):
        AgentEvalInputSpec.model_validate({"tasks": [{"id": "t", "intent": "x", "inputs": {}}], "target": submitted})


def test_relative_paths_in_skills_or_discovery_need_files() -> None:
    assert not registered_agent_config_needs_files({"harness": {}})
    assert not registered_agent_config_needs_files({"skills": {"paths": []}, "discovery": {}})
    assert registered_agent_config_needs_files({"skills": {"paths": ["skills/a"]}})
    assert registered_agent_config_needs_files({"discovery": {"local_paths": ["adapters"]}})


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
    assert job.steps[0].config == {"environment": "dev/agent-files-0123abcd4567"}

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
        "source": {"agent": "dev/calculator-agent", "files": "dev/agent-files-0123abcd4567"},
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
