# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``NemoFabricAgent`` runs the ``fabric_config`` it is given; Harbor only decides where the trial lives."""

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("harbor", reason="NemoFabricAgent subclasses Harbor's BaseAgent")

from harbor.models.task.config import MCPServerConfig
from nemo_evaluator_sdk.agent_eval.runtimes.harbor_fabric_agent import FLAT_CONFIG_KWARGS, NemoFabricAgent
from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import HarborRuntimeConfig
from nemo_evaluator_sdk.agent_eval.runtimes.provenance import redact_credentials
from nemo_fabric import RelayAtifConfig, RelayAtofConfig
from nemo_fabric.integrations.harbor.fabric_agent import HARBOR_ARTIFACT_ROOT

_DEEPAGENTS = "nvidia.fabric.langchain.deepagents"


def _registered_agent_config() -> dict[str, Any]:
    """What a platform-registered agent looks like after translation: identity, gateway, skills, telemetry."""
    return {
        "metadata": {"name": "calculator-agent", "description": "Calculator agent executed with DeepAgents"},
        "harness": {"adapter_id": _DEEPAGENTS, "resolution": "preinstalled", "settings": {"deepagents": {}}},
        "environment": {
            "provider": "local",
            "workspace": "./workspace",
            "artifacts": "./artifacts",
            "env": {"NHX_BASE_URL": "http://platform.test", "NVIDIA_API_KEY": "not-used"},
        },
        "models": {
            "default": {
                "provider": "nvidia",
                "model": "nvidia-nemotron-3-5-lightning-30b-a3b",
                "api_key_env": "NVIDIA_API_KEY",
                "base_url": "http://platform.test/apis/inference-gateway/v2/workspaces/default/openai/-/v1",
            }
        },
        "instructions": {"system": {"content": "You are a concise calculator agent.", "mode": "replace"}},
        "skills": {"paths": ["skills/arithmetic"]},
        "mcp": {"servers": {"calculator": {"transport": "streamable-http", "url": "http://calc.internal/mcp"}}},
        "tools": {"blocked": ["web_search"]},
        "telemetry": {"providers": {"relay": {}}},
        "relay": {
            "project": "calculator-agent",
            "output_dir": "./artifacts/relay",
            "observability": {
                "version": 3,
                "atif": {"enabled": True, "agent_name": "calculator-agent", "output_directory": "./artifacts/relay"},
                "atof": {
                    "enabled": True,
                    "sinks": [
                        {"type": "file", "output_directory": "./artifacts/relay", "filename": "events.atof.jsonl"}
                    ],
                },
            },
        },
    }


def _agent(tmp_path: Path, **kwargs: Any) -> NemoFabricAgent:
    return NemoFabricAgent(
        logs_dir=tmp_path, fabric_config=_registered_agent_config(), fabric_workspace="/app", **kwargs
    )


def test_the_config_runs_verbatim_with_its_own_identity(tmp_path: Path) -> None:
    """Nothing the agent declares is dropped or renamed -- the point of taking a config over keywords."""
    config = _agent(tmp_path)._build_config()

    assert config.metadata.name == "calculator-agent"
    assert config.harness is not None and config.harness.adapter_id == _DEEPAGENTS
    assert config.skills is not None and [str(p) for p in config.skills.paths] == ["skills/arithmetic"]
    assert config.mcp is not None and set(config.mcp.servers) == {"calculator"}
    assert config.tools is not None and config.tools.blocked == ["web_search"]
    assert config.instructions is not None and config.instructions.system is not None
    model = config.models["default"]
    assert model.model == "nvidia-nemotron-3-5-lightning-30b-a3b"
    assert model.base_url == "http://platform.test/apis/inference-gateway/v2/workspaces/default/openai/-/v1"
    assert model.api_key_env == "NVIDIA_API_KEY"
    assert config.relay is not None and config.relay.observability is not None
    atif = config.relay.observability.atif
    assert isinstance(atif, RelayAtifConfig) and atif.agent_name == "calculator-agent"


def test_harbor_owns_only_where_the_trial_lives(tmp_path: Path) -> None:
    config = _agent(tmp_path)._build_config()
    artifact_root = f"{HARBOR_ARTIFACT_ROOT}/calculator-agent"

    assert config.environment is not None
    assert config.environment.provider == "local"
    assert str(config.environment.workspace) == "/app"
    assert str(config.environment.artifacts) == artifact_root
    assert str(config.runtime.artifacts) == artifact_root
    # The config's own env survives: the gateway placeholder is how the harness authenticates.
    assert config.environment.env == {"NHX_BASE_URL": "http://platform.test", "NVIDIA_API_KEY": "not-used"}
    # Relay output moves under the artifact root so Harbor collects the trajectory with the trial.
    assert config.relay is not None and str(config.relay.output_dir) == f"{artifact_root}/relay"
    observability = config.relay.observability
    assert observability is not None
    assert isinstance(observability.atif, RelayAtifConfig)
    assert str(observability.atif.output_directory) == f"{artifact_root}/relay"
    assert isinstance(observability.atof, RelayAtofConfig) and observability.atof.sinks is not None
    assert [str(getattr(sink, "output_directory")) for sink in observability.atof.sinks] == [f"{artifact_root}/relay"]


def test_harbor_runner_schemas_are_applied_unless_the_config_sets_them(tmp_path: Path) -> None:
    config = _agent(tmp_path)._build_config()
    assert (config.runtime.input_schema, config.runtime.output_schema) == ("text", "message")

    explicit = _registered_agent_config()
    explicit["runtime"] = {"input_schema": "message", "output_schema": "text"}
    config = NemoFabricAgent(logs_dir=tmp_path, fabric_config=explicit)._build_config()
    assert (config.runtime.input_schema, config.runtime.output_schema) == ("message", "text")


def test_the_supplied_config_is_not_mutated_between_trials(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    agent._build_config()
    assert str(agent.fabric_config.environment.workspace) == "./workspace"  # type: ignore[union-attr]


def test_default_turn_budget_applies_only_when_the_config_names_none(tmp_path: Path) -> None:
    assert _agent(tmp_path)._build_config().runtime.max_turns is None
    assert _agent(tmp_path, fabric_default_max_turns=50)._build_config().runtime.max_turns == 50

    budgeted = _registered_agent_config()
    budgeted["runtime"] = {"max_turns": 5}
    agent = NemoFabricAgent(logs_dir=tmp_path, fabric_config=budgeted, fabric_default_max_turns=50)
    assert agent._build_config().runtime.max_turns == 5


def test_harbor_model_name_swaps_the_model_id_but_keeps_the_endpoint_wiring(tmp_path: Path) -> None:
    model = _agent(tmp_path, model_name="nvidia-nemotron-3-super")._build_config().models["default"]

    assert model.model == "nvidia-nemotron-3-super"
    assert model.base_url == "http://platform.test/apis/inference-gateway/v2/workspaces/default/openai/-/v1"
    assert model.api_key_env == "NVIDIA_API_KEY"


def test_harbor_model_name_without_a_default_model_is_refused_not_invented(tmp_path: Path) -> None:
    config = _registered_agent_config()
    config["models"] = {}
    with pytest.raises(ValueError, match="declares no `models.default`"):
        NemoFabricAgent(logs_dir=tmp_path, fabric_config=config, model_name="x")._build_config()


def test_harbor_provided_mcp_servers_and_skills_are_added_to_the_agents_own(tmp_path: Path) -> None:
    agent = _agent(
        tmp_path,
        mcp_servers=[MCPServerConfig(name="task-tools", transport="stdio", command="tools", args=["--serve"])],
        skills_dir="/task/skills",
    )
    config = agent._build_config()

    assert config.mcp is not None and set(config.mcp.servers) == {"calculator", "task-tools"}
    assert config.skills is not None
    assert [str(p) for p in config.skills.paths] == ["skills/arithmetic", "/task/skills"]


@pytest.mark.parametrize("keyword", sorted(FLAT_CONFIG_KWARGS))
def test_flat_agent_description_keywords_are_refused(tmp_path: Path, keyword: str) -> None:
    """Two descriptions of one agent cannot be merged sensibly; the config is the description."""
    with pytest.raises(ValueError, match=keyword):
        _agent(tmp_path, **{keyword: "x"})


def test_a_config_without_a_harness_is_refused_at_construction(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="harness.adapter_id"):
        NemoFabricAgent(logs_dir=tmp_path, fabric_config={"metadata": {"name": "no-harness"}})


def test_install_and_run_keywords_still_pass_through(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    agent = _agent(
        tmp_path,
        fabric_package="nemo-fabric[deepagents]==0.3.0",
        fabric_config_bundle=bundle,
        fabric_config_target="/tmp/agent",
    )
    assert agent.fabric_package == "nemo-fabric[deepagents]==0.3.0"
    assert agent.fabric_config_bundle == bundle
    # Relative paths in the config (skills) resolve against the uploaded bundle.
    assert str(agent._build_spec("hi").config_base_dir) == "/tmp/agent"


def test_reports_its_own_name_to_harbor() -> None:
    assert NemoFabricAgent.name() == "nemo-fabric"


def test_a_fabric_config_in_agent_kwargs_is_judged_by_value_shape_not_key_name(tmp_path: Path) -> None:
    """The runtime config persists kwargs unredacted, so it refuses credentials -- but a registered agent's
    config *names* its credential variable and carries the gateway placeholder, which are not credentials."""
    config = HarborRuntimeConfig(
        jobs_dir=tmp_path,
        agent_import_path="nemo_evaluator_sdk.agent_eval.runtimes.harbor_fabric_agent:NemoFabricAgent",
        agent_kwargs={"fabric_config": _registered_agent_config()},
    )
    kept = config.agent_kwargs["fabric_config"]
    assert isinstance(kept, dict) and kept["models"]["default"]["api_key_env"] == "NVIDIA_API_KEY"

    leaked = _registered_agent_config()
    leaked["environment"]["env"]["NVIDIA_API_KEY"] = "nvapi-" + "a" * 60
    with pytest.raises(ValueError, match="fabric_config.environment.env.NVIDIA_API_KEY"):
        HarborRuntimeConfig(jobs_dir=tmp_path, agent_import_path="x:Y", agent_kwargs={"fabric_config": leaked})


def test_provenance_redaction_keeps_a_fabric_configs_variable_names() -> None:
    recorded = redact_credentials({"fabric_config": _registered_agent_config(), "api_key": "plain"})
    assert recorded["fabric_config"]["models"]["default"]["api_key_env"] == "NVIDIA_API_KEY"
    assert recorded["fabric_config"]["environment"]["env"]["NVIDIA_API_KEY"] == "not-used"
    assert recorded["api_key"] == "<redacted>"  # loose kwargs keep the key-marker rule
