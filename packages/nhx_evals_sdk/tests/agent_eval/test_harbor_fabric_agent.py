# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``NemoFabricAgent`` runs the ``fabric_config`` it is given; Harbor only decides where the trial lives."""

import importlib.metadata
import json
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("harbor", reason="NemoFabricAgent subclasses Harbor's BaseAgent")

from harbor.models.task.config import MCPServerConfig
from nemo_fabric import RelayAtifConfig, RelayAtofConfig
from nemo_fabric.integrations.harbor.fabric_agent import HARBOR_ARTIFACT_ROOT, FabricAgentOptions
from nhx_evals_sdk.agent_eval.runtimes.harbor.fabric_agent import (
    FLAT_CONFIG_KWARGS,
    INSTALL_RUN_KWARGS,
    NemoFabricAgent,
)
from nhx_evals_sdk.agent_eval.runtimes.harbor.runtime import HarborRuntimeConfig
from nhx_evals_sdk.agent_eval.runtimes.provenance import credential_shaped_settings, redact_credentials

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
            "env": {"NHX_BASE_URL": "http://platform.test"},
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
                "version": 4,
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
    # Relay reaches a plain-HTTP gateway through the loopback forwarder.
    assert model.base_url == "http://127.0.0.1:14320/apis/inference-gateway/v2/workspaces/default/openai/-/v1"
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


def test_an_explicit_null_turn_budget_keeps_the_harness_unbounded(tmp_path: Path) -> None:
    """``max_turns: null`` is a choice, not an omission; the default must not paper over it."""
    unbounded = _registered_agent_config()
    unbounded["runtime"] = {"max_turns": None}
    agent = NemoFabricAgent(logs_dir=tmp_path, fabric_config=unbounded, fabric_default_max_turns=50)
    assert agent._build_config().runtime.max_turns is None


def test_unattended_harness_settings_are_filled_in_but_never_overridden(tmp_path: Path) -> None:
    """A Claude harness prompts for permissions unless told otherwise; in a task container nobody answers."""
    claude = _registered_agent_config()
    claude["harness"] = {"adapter_id": "nvidia.fabric.claude"}
    config = NemoFabricAgent(logs_dir=tmp_path, fabric_config=claude, fabric_workspace="/app")._build_config()
    assert config.harness is not None
    assert config.harness.settings["permission_mode"] == "bypassPermissions"
    assert config.environment is not None and config.environment.env["IS_SANDBOX"] == "1"

    claude["harness"] = {"adapter_id": "nvidia.fabric.claude", "settings": {"permission_mode": "acceptEdits"}}
    claude["environment"]["env"]["IS_SANDBOX"] = "0"
    config = NemoFabricAgent(logs_dir=tmp_path, fabric_config=claude, fabric_workspace="/app")._build_config()
    assert config.harness is not None and config.harness.settings["permission_mode"] == "acceptEdits"
    assert config.environment is not None and config.environment.env["IS_SANDBOX"] == "0"

    # Other harnesses need nothing; the agent's own settings pass through untouched.
    deepagents = _agent(tmp_path)._build_config()
    assert deepagents.harness is not None and deepagents.harness.settings == {"deepagents": {}}


def test_harbor_model_name_swaps_the_model_id_but_keeps_the_endpoint_wiring(tmp_path: Path) -> None:
    model = _agent(tmp_path, model_name="nvidia-nemotron-3-super")._build_config().models["default"]

    assert model.model == "nvidia-nemotron-3-super"
    # Relay reaches a plain-HTTP gateway through the loopback forwarder.
    assert model.base_url == "http://127.0.0.1:14320/apis/inference-gateway/v2/workspaces/default/openai/-/v1"
    assert model.api_key_env == "NVIDIA_API_KEY"


def test_harbor_model_name_without_a_default_model_is_refused_not_invented(tmp_path: Path) -> None:
    config = _registered_agent_config()
    config["models"] = {}
    with pytest.raises(ValueError, match="declares no `models.default`"):
        NemoFabricAgent(logs_dir=tmp_path, fabric_config=config, model_name="x")._build_config()


def test_harbor_provided_mcp_servers_are_added_and_skills_travel_through_the_payload(tmp_path: Path) -> None:
    """The task runner expands ``skills_dir`` into its ``<skill>/SKILL.md`` children; the root itself is not a
    skill and the Claude/Codex adapters refuse it as one."""
    agent = _agent(
        tmp_path,
        mcp_servers=[MCPServerConfig(name="task-tools", transport="stdio", command="tools", args=["--serve"])],
        skills_dir="/task/skills",
    )
    spec = agent._build_spec("hi")

    assert spec.config.mcp is not None and set(spec.config.mcp.servers) == {"calculator", "task-tools"}
    assert spec.config.skills is not None
    assert [str(p) for p in spec.config.skills.paths] == ["skills/arithmetic"]
    assert str(spec.skills_dir) == "/task/skills"


@pytest.mark.parametrize("keyword", sorted(FLAT_CONFIG_KWARGS))
def test_flat_agent_description_keywords_are_refused(tmp_path: Path, keyword: str) -> None:
    """Two descriptions of one agent cannot be merged sensibly; the config is the description."""
    with pytest.raises(ValueError, match=keyword):
        _agent(tmp_path, **{keyword: "x"})


def test_the_refused_keywords_are_everything_upstream_accepts_that_is_not_install_or_run() -> None:
    """A keyword upstream adds lands in one set or the other; drifting into neither would merge silently."""
    assert FLAT_CONFIG_KWARGS | INSTALL_RUN_KWARGS == set(FabricAgentOptions.model_fields)
    assert not FLAT_CONFIG_KWARGS & INSTALL_RUN_KWARGS
    assert "fabric_adapter_id" in FLAT_CONFIG_KWARGS and "fabric_discovery_paths" in INSTALL_RUN_KWARGS


def test_harbor_config_validate_sees_the_same_refusals_as_preflight() -> None:
    """``harbor config validate`` calls ``parse_options``, not ``preflight``; the two must not disagree."""
    with pytest.raises(ValueError, match="fabric_max_turns"):
        NemoFabricAgent.parse_options({"fabric_config": _registered_agent_config(), "fabric_max_turns": 3})
    with pytest.raises(ValueError, match="fabric_adapter_id"):
        NemoFabricAgent.parse_options(
            {"fabric_config": _registered_agent_config(), "fabric_adapter_id": "nvidia.fabric.codex"}
        )
    # The adapter the config itself names is what the constructor hands upstream, so it passes.
    options = NemoFabricAgent.parse_options(
        {"fabric_config": _registered_agent_config(), "fabric_adapter_id": _DEEPAGENTS}
    )
    assert options is not None and options.fabric_adapter_id == _DEEPAGENTS


def test_the_options_schema_lists_only_what_the_agent_honours() -> None:
    schema = NemoFabricAgent.options_schema()
    assert not FLAT_CONFIG_KWARGS & set(schema["properties"])
    assert {"fabric_config", "fabric_default_max_turns", "fabric_package", "fabric_discovery_paths"} <= set(
        schema["properties"]
    )
    assert schema.get("required") == ["fabric_config"]


def test_task_local_discovery_paths_reach_the_config(tmp_path: Path) -> None:
    """Upstream applies ``fabric_discovery_paths`` in its ``_build_config``; overriding it must not drop them."""
    config = _agent(tmp_path, fabric_discovery_paths=["/tmp/adapters"])._build_spec("hi").config
    assert config.discovery is not None and [str(p) for p in config.discovery.local_paths] == ["/tmp/adapters"]

    listed = _registered_agent_config()
    listed["discovery"] = {"local_paths": ["adapters", "/tmp/adapters"]}
    agent = NemoFabricAgent(logs_dir=tmp_path, fabric_config=listed, fabric_discovery_paths=["/tmp/adapters", "/opt/x"])
    config = agent._build_spec("hi").config
    assert config.discovery is not None
    assert [str(p) for p in config.discovery.local_paths] == ["adapters", "/tmp/adapters", "/opt/x"]


def test_a_config_without_a_harness_is_refused_at_construction(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="harness.adapter_id"):
        NemoFabricAgent(logs_dir=tmp_path, fabric_config={"metadata": {"name": "no-harness"}})


def test_harbor_preflight_accepts_the_agent_kwargs_a_target_carries() -> None:
    NemoFabricAgent.preflight(
        {
            "fabric_config": _registered_agent_config(),
            "fabric_default_max_turns": 20,
            "fabric_package": "nemo-fabric[deepagents]==0.5.0",
            "fabric_workspace": "/app",
        }
    )


@pytest.mark.parametrize("keyword", sorted(FLAT_CONFIG_KWARGS))
def test_harbor_preflight_refuses_flat_agent_description_keywords(keyword: str) -> None:
    with pytest.raises(ValueError, match=keyword):
        NemoFabricAgent.preflight({"fabric_config": _registered_agent_config(), keyword: "x"})


@pytest.mark.parametrize("turns", [0, -1])
def test_harbor_preflight_refuses_a_non_positive_default_turn_budget(turns: int) -> None:
    """Fabric's ``runtime.max_turns`` is ``gt=0``; catching it here keeps the failure out of ``Trial.create``."""
    with pytest.raises(ValueError, match="fabric_default_max_turns"):
        NemoFabricAgent.preflight({"fabric_config": _registered_agent_config(), "fabric_default_max_turns": turns})


def test_harbor_preflight_refuses_a_config_without_a_harness() -> None:
    with pytest.raises(ValueError, match="harness.adapter_id"):
        NemoFabricAgent.preflight({"fabric_config": {"metadata": {"name": "no-harness"}}})


def test_harbor_preflight_requires_a_fabric_config() -> None:
    with pytest.raises(ValueError, match="fabric_config"):
        NemoFabricAgent.preflight({"fabric_workspace": "/app"})


def _custom_adapter_descriptor(tmp_path: Path, adapter_id: str) -> Path:
    descriptor = {
        "contract_version": "fabric.adapter/v1alpha2",
        "adapter_id": adapter_id,
        "adapter_kind": "process",
        "runner": {"command": "python3", "script": "adapter.py"},
        "requirements": {"binaries": ["python3"]},
        "config": {
            "accepts": ["models", "models.base_url", "instructions.system", "skills", "mcp", "tools.blocked"],
            "system_instruction_modes": ["replace"],
        },
        "telemetry": {"providers": {"relay": {"outputs": ["atif"]}}},
        "capabilities": {"streaming": False, "cancellation": False, "updates": False, "service": False},
    }
    path = tmp_path / "custom.fabric-adapter.json"
    path.write_text(json.dumps(descriptor), encoding="utf-8")
    return path


def test_harbor_preflight_refuses_a_custom_adapter_the_host_cannot_inspect(tmp_path: Path) -> None:
    """Fabric runs this check in ``__init__``; Harbor does not catch construction errors per trial, so
    the job would abort with Fabric's wording and no pointer at what is missing."""
    custom = _registered_agent_config()
    custom["harness"] = {"adapter_id": "acme.fabric.custom"}
    custom["discovery"] = {"local_paths": ["adapters"]}

    with pytest.raises(ValueError, match="fabric_adapter_descriptor"):
        NemoFabricAgent.preflight({"fabric_config": custom})

    descriptor = _custom_adapter_descriptor(tmp_path, "acme.fabric.custom")
    NemoFabricAgent.preflight({"fabric_config": custom, "fabric_adapter_descriptor": str(descriptor)})


def test_harbor_preflight_refuses_a_fabric_package_pinned_to_another_release() -> None:
    """The task-side runner rejects a payload serialised by a host on a different Fabric release."""
    with pytest.raises(ValueError, match="task-side Fabric runner must match"):
        NemoFabricAgent.preflight(
            {"fabric_config": _registered_agent_config(), "fabric_package": "nemo-fabric[deepagents]==0.4.0"}
        )
    for package in (
        "nemo-fabric[deepagents,relay]==0.4.0 mcp==1.29.0",
        "'nemo-fabric[deepagents]==0.4.0; python_version<\"3.13\"'",
        "nemo-fabric==0.4.0,!=0.4.1",
    ):
        with pytest.raises(ValueError, match="task-side Fabric runner must match"):
            NemoFabricAgent.preflight({"fabric_config": _registered_agent_config(), "fabric_package": package})

    host = importlib.metadata.version("nemo-fabric")
    for package in (
        f"nemo-fabric[deepagents]=={host}",
        "nemo-fabric[deepagents]>=0.5,<0.6",
        f"mcp==1.29.0 nemo-fabric=={host}",
    ):
        NemoFabricAgent.preflight({"fabric_config": _registered_agent_config(), "fabric_package": package})


def test_parsed_options_carry_the_config_and_its_adapter(tmp_path: Path) -> None:
    options = _agent(tmp_path, fabric_default_max_turns=7).options
    assert options.fabric_adapter_id == _DEEPAGENTS
    assert options.fabric_config == _registered_agent_config()
    assert options.fabric_default_max_turns == 7


def test_install_and_run_keywords_still_pass_through(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    agent = _agent(
        tmp_path,
        fabric_package="nemo-fabric[deepagents]==0.5.0",
        fabric_config_bundle=bundle,
        fabric_config_target="/tmp/agent",
    )
    assert agent.fabric_package == "nemo-fabric[deepagents]==0.5.0"
    assert agent.fabric_config_bundle == bundle
    # Relative paths in the config (skills) resolve against the uploaded bundle.
    assert str(agent._build_spec("hi").config_base_dir) == "/tmp/agent"


def test_reports_its_own_name_to_harbor() -> None:
    assert NemoFabricAgent.name() == "nemo-fabric"


def test_the_module_imports_without_the_harbor_extra_and_names_it_when_the_agent_is_used() -> None:
    """The examples import ``NVIDIA_MODEL_BASE_URL`` from here; upstream exports its options model only
    beside its real ``FabricAgent``, so the import has to tolerate the placeholder."""
    script = textwrap.dedent(
        """
        import sys
        sys.modules["harbor"] = None  # the extra is absent
        from nhx_evals_sdk.agent_eval.runtimes.harbor import fabric_agent
        print(fabric_agent.NVIDIA_MODEL_BASE_URL)
        try:
            fabric_agent.NemoFabricAgent.preflight({"fabric_config": {}})
        except ModuleNotFoundError as error:
            print(error)
        """
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    base_url, hint = result.stdout.strip().splitlines()
    assert base_url == "https://integrate.api.nvidia.com/v1"
    assert "nemo-fabric[harbor]" in hint


def test_gateway_models_get_the_placeholder_key_their_client_insists_on(tmp_path: Path) -> None:
    """The gateway authenticates the job, but the harness's client refuses an empty key. The placeholder is
    added per trial so the config handed to Harbor never has to carry a credential-named value."""
    env = _agent(tmp_path)._build_config().environment
    assert env is not None and env.env["NVIDIA_API_KEY"] == "not-used"

    unnamed = _registered_agent_config()
    del unnamed["models"]["default"]["api_key_env"]
    config = NemoFabricAgent(logs_dir=tmp_path, fabric_config=unnamed)._build_config()
    assert config.models["default"].api_key_env == "NEMO_AGENTS_IGW_API_KEY"
    assert config.environment is not None and config.environment.env["NEMO_AGENTS_IGW_API_KEY"] == "not-used"

    direct = _registered_agent_config()
    direct["models"]["default"]["base_url"] = "https://integrate.api.nvidia.com/v1"
    config = NemoFabricAgent(logs_dir=tmp_path, fabric_config=direct)._build_config()
    assert config.environment is not None and "NVIDIA_API_KEY" not in config.environment.env


def test_a_gateway_key_harbor_hands_the_agent_is_not_shadowed_by_the_placeholder(tmp_path: Path) -> None:
    """A trial's own gateway key reaches the harness through the agent's environment."""
    agent = NemoFabricAgent(
        logs_dir=tmp_path, fabric_config=_registered_agent_config(), extra_env={"NVIDIA_API_KEY": "trial-token"}
    )

    env = agent._build_config().environment
    assert env is not None and "NVIDIA_API_KEY" not in env.env
    assert agent._runner_env["NVIDIA_API_KEY"] == "trial-token"


def test_a_fabric_config_may_name_its_credential_variable_but_never_hold_a_value(tmp_path: Path) -> None:
    """Harbor persists ``agent_kwargs`` unredacted. ``api_key_env`` names a variable and passes; anything
    under a credential-named key is refused -- an issued token, an unrecognised secret, or the placeholder."""
    config = HarborRuntimeConfig(
        jobs_dir=tmp_path,
        agent_import_path="nhx_evals_sdk.agent_eval.runtimes.harbor.fabric_agent:NemoFabricAgent",
        agent_kwargs={"fabric_config": _registered_agent_config()},
    )
    kept = config.agent_kwargs["fabric_config"]
    assert kept == _registered_agent_config()

    for name, value in [
        ("NVIDIA_API_KEY", "nvapi-" + "a" * 60),
        ("MY_API_KEY", "an-unrecognised-real-secret"),
        ("NVIDIA_API_KEY", "not-used"),
        ("MY_API_KEY_ENV", "SUPER_SECRET_VALUE_123"),  # a user's key that happens to end in _ENV
        ("api_key_env", "SUPER_SECRET_VALUE_123"),  # even Fabric's own field name, under the env map
    ]:
        leaked = _registered_agent_config()
        leaked["environment"]["env"][name] = value
        with pytest.raises(ValueError, match=f"fabric_config.environment.env.{name}"):
            HarborRuntimeConfig(jobs_dir=tmp_path, agent_import_path="x:Y", agent_kwargs={"fabric_config": leaked})

    # The name exemption is for names: a value that is not shaped like one gets no pass from the key.
    misnamed = _registered_agent_config()
    misnamed["models"]["default"]["api_key_env"] = "an-unrecognised-real-secret"
    with pytest.raises(ValueError, match="fabric_config.models.default.api_key_env"):
        HarborRuntimeConfig(jobs_dir=tmp_path, agent_import_path="x:Y", agent_kwargs={"fabric_config": misnamed})


def test_only_fabrics_own_name_fields_are_exempt_from_the_marker_checks() -> None:
    """The exemption is a list of Fabric fields, not an ``_env`` suffix: a harness setting a user named
    ``token_env`` is a value under a credential-marked key and is reported like any other."""
    assert credential_shaped_settings({"fabric_config": {"harness": {"token_env": "SUPER_SECRET_VALUE_123"}}}) == [
        "fabric_config.harness.token_env"
    ]
    assert credential_shaped_settings({"fabric_config": {"models": {"m": {"api_key_env": "MY_KEY"}}}}) == []


def test_provenance_redaction_keeps_variable_names_and_redacts_values() -> None:
    config = _registered_agent_config()
    config["environment"]["env"]["MY_API_KEY"] = "an-unrecognised-real-secret"
    config["models"]["default"]["api_key_env"] = "not a variable name"  # a value where a name belongs

    recorded = redact_credentials({"fabric_config": config, "api_key": "plain"})

    assert recorded["fabric_config"]["environment"]["env"]["MY_API_KEY"] == "<redacted>"
    assert recorded["fabric_config"]["models"]["default"]["api_key_env"] == "<redacted>"
    assert recorded["api_key"] == "<redacted>"
    assert (
        redact_credentials({"fabric_config": _registered_agent_config()})["fabric_config"]["models"]["default"][
            "api_key_env"
        ]
        == "NVIDIA_API_KEY"
    )
