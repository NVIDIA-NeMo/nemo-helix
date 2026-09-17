# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for service-side sandboxed Gym execution.

Three things are covered: that a submitted target maps onto Gym's own configuration mechanism, that
a deployment which cannot sandbox refuses rather than falls back, and that a plaintext credential
cannot ride into user-supplied environment code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from nemo_evaluator.config import EvaluatorConfig
from nemo_evaluator.filesets import FilesetRef
from nemo_evaluator.jobs.agent_spec import GymRunnerTarget
from nemo_evaluator.jobs.gym_sandbox import (
    SandboxPlan,
    SandboxUnavailableError,
    credential_shaped_env_vars,
    gym_global_config,
    require_fileset_environment_sandboxed,
    require_fileset_sandbox_storage_identity,
    resolve_sandbox_plan,
    serve_config,
)
from nemo_platform_plugin.jobs.execution_profiles import (
    KubernetesJobExecutionProfile,
    KubernetesJobExecutionProfileConfig,
    KubernetesJobStorageConfig,
)


def target(**overrides: Any) -> GymRunnerTarget:
    fields: dict[str, Any] = {
        "agent": "simple_agent",
        "agent_config": "responses_api_agents/simple_agent/configs/simple_agent.yaml",
        "resources_server": "mcqa",
    }
    fields.update(overrides)
    return GymRunnerTarget(**fields)


def capable_config(**overrides: Any) -> EvaluatorConfig:
    fields: dict[str, Any] = {
        "sandboxed_gym_default": True,
        "sandbox_cluster_capable": True,
        "sandbox_runtime_image": "registry.example.com/nmp-gym-runtime:1.0",
        "sandbox_job_storage_pvc_claim": "job-storage",
        "sandbox_policy_base_urls": ("https://integrate.api.nvidia.com/v1",),
    }
    fields.update(overrides)
    return EvaluatorConfig(**fields)


def capable_plan(**overrides) -> SandboxPlan:
    """The plan the compiler would resolve from a sandbox-capable deployment."""
    plan = resolve_sandbox_plan(capable_config(**overrides), target())
    assert plan is not None
    return plan


# --------------------------------------------------------------------------------------------
# Selection -> Gym config
# --------------------------------------------------------------------------------------------


def test_the_selection_becomes_config_paths_gym_resolves_itself() -> None:
    """The CLI's `--config`/`--model-type`/`--resources-server` are sugar over `config_paths`.

    Emitting the same paths means the sandboxed run loads the same configs, and means we never have
    to understand Gym's YAML schema -- Gym merges what it is handed and then loads these itself.
    """
    config = gym_global_config(target(model_type="openai_model"))

    assert config["config_paths"] == [
        "responses_api_agents/simple_agent/configs/simple_agent.yaml",
        "responses_api_models/openai_model/configs/openai_model.yaml",
        "resources_servers/mcqa/configs/mcqa.yaml",
    ]


def test_asset_paths_stay_relative_so_the_sandbox_resolves_them() -> None:
    # Gym searches NEMO_GYM_EXTRA_ROOTS, then cwd, then its install root. Resolving here would bake
    # in this process's filesystem, which is not the one the environment is mounted on.
    for path in gym_global_config(target())["config_paths"]:
        assert not path.startswith("/")


def test_a_name_slash_flavor_selector_picks_that_flavor() -> None:
    config = gym_global_config(target(resources_server="mcqa/variant"))

    assert "resources_servers/mcqa/configs/variant.yaml" in config["config_paths"]


def test_the_resources_server_binding_travels_as_data_not_a_hydra_string() -> None:
    # The CLI has to serialize this as `+a.b.c=value`; a config dict does not, so there is no
    # quoting grammar to get wrong.
    config = gym_global_config(target())

    assert config["simple_agent"]["responses_api_agents"]["simple_agent"]["resources_server"]["name"] == "mcqa"


def test_binding_is_omitted_for_self_contained_agents() -> None:
    assert "simple_agent" not in gym_global_config(target(bind_resources_server=False))


def test_an_explicit_override_wins_over_the_derived_binding() -> None:
    # Same precedence as the CLI, where hydra_params are appended after the derived override.
    config = gym_global_config(
        target(hydra_params={"simple_agent": {"responses_api_agents": {"simple_agent": {"extra": 1}}}})
    )

    agent = config["simple_agent"]["responses_api_agents"]["simple_agent"]
    assert agent == {"extra": 1}, "the caller's value replaces the derived one at the same key"


def test_custom_environment_carries_component_selection_for_host_composition() -> None:
    config = gym_global_config(
        target(
            environment=FilesetRef(root="default/custom-gym"),
            agent_ref_name="custom_agent",
        )
    )

    assert config["config_paths"][0] == "responses_api_agents/simple_agent/configs/simple_agent.yaml"
    assert config["_nmp_environment_component_selection"] == {
        "agent_instance": "custom_agent",
        "agent_config": "responses_api_agents/simple_agent/configs/simple_agent.yaml",
        "resources_server_instance": "mcqa",
        "resources_server_config": "resources_servers/mcqa/configs/mcqa.yaml",
        "model_config": "responses_api_models/inference_provider/configs/inference_provider.yaml",
    }
    assert config["custom_agent"]["responses_api_agents"]["simple_agent"]["resources_server"]["name"] == "mcqa"


def test_custom_environment_omits_agent_config_when_the_package_supplies_the_agent() -> None:
    config = gym_global_config(
        target(
            environment=FilesetRef(root="default/custom-gym"),
            agent_config=None,
            agent_ref_name="custom_agent",
        )
    )

    assert "responses_api_agents/simple_agent/configs/simple_agent.yaml" not in config["config_paths"]
    assert config["_nmp_environment_component_selection"]["agent_config"] is None


def test_agent_config_is_required_without_an_environment_package() -> None:
    with pytest.raises(ValueError, match="agent_config field is required"):
        target(agent_config=None)


def test_environment_fileset_rejects_file_fragments() -> None:
    with pytest.raises(ValueError, match="file fragment"):
        target(environment=FilesetRef(root="default/custom-gym#resources_servers/custom/app.py"))


def test_serve_config_takes_cluster_facts_from_the_deployment_not_the_job() -> None:
    payload = serve_config(target(), capable_plan(), job_id="job-7")

    assert payload["sandbox"]["image"] == "registry.example.com/nmp-gym-runtime:1.0"
    assert payload["sandbox"]["environment_pvc_claim"] == "job-storage"
    assert payload["episode_broker"]["job_id"] == "job-7"
    # ...and the job's half is the environment selection, nothing else.
    assert payload["gym_global_config"]["config_paths"]


def test_unset_runtime_image_uses_the_qualified_gym_host(monkeypatch: pytest.MonkeyPatch) -> None:
    qualified = "registry.example.com/nemo/nmp-gym-host:same-platform-tag"
    monkeypatch.setattr(
        "nemo_evaluator.jobs.gym_sandbox.get_qualified_image",
        lambda name: qualified if name == "nmp-gym-host" else "",
    )

    plan = resolve_sandbox_plan(capable_config(sandbox_runtime_image=None), target())

    assert plan is not None
    assert plan.runtime_image == qualified


def test_explicit_runtime_image_bypasses_qualification(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_qualification(name: str) -> str:
        raise AssertionError(f"explicit runtime override unexpectedly qualified {name}")

    monkeypatch.setattr("nemo_evaluator.jobs.gym_sandbox.get_qualified_image", unexpected_qualification)

    plan = resolve_sandbox_plan(capable_config(sandbox_runtime_image="registry.example.com/custom/host:1"), target())

    assert plan is not None
    assert plan.runtime_image == "registry.example.com/custom/host:1"


def test_sandbox_server_protocol_reaches_the_opensandbox_host_provider() -> None:
    plan = resolve_sandbox_plan(
        capable_config(),
        target(),
        sandbox_server_protocol="http",
    )

    assert plan is not None
    assert plan.host_provider_options == {"connection": {"protocol": "http"}}


def test_sandbox_server_protocol_does_not_change_the_docker_host_provider() -> None:
    plan = resolve_sandbox_plan(
        capable_config(
            sandbox_host_provider="docker",
            sandbox_host_provider_options={"root_dir": "/tmp/gym"},
        ),
        target(),
        sandbox_server_protocol="http",
    )

    assert plan is not None
    assert plan.host_provider_options == {"root_dir": "/tmp/gym"}


# --------------------------------------------------------------------------------------------
# Gates
# --------------------------------------------------------------------------------------------


def test_sandboxing_is_off_by_default() -> None:
    # An existing deployment has no OpenSandbox to provision against, so the default must not
    # change what it does today.
    assert resolve_sandbox_plan(EvaluatorConfig(), target()) is None


def test_fileset_environment_is_rejected_when_sandboxing_is_off() -> None:
    with pytest.raises(SandboxUnavailableError, match="require sandboxed execution"):
        require_fileset_environment_sandboxed(
            target(environment=FilesetRef(root="default/custom-gym")),
            EvaluatorConfig(),
        )


def test_fileset_environment_is_rejected_when_opensandbox_pvc_differs_from_job_storage() -> None:
    profile = KubernetesJobExecutionProfile(
        config=KubernetesJobExecutionProfileConfig(storage=KubernetesJobStorageConfig(pvc_name="jobs-pvc"))
    )

    with pytest.raises(SandboxUnavailableError, match="sandbox_job_storage_pvc_claim"):
        require_fileset_sandbox_storage_identity(
            target(environment=FilesetRef(root="default/custom-gym")),
            capable_config(),
            execution_profile=profile,
        )


def test_fileset_environment_skips_pvc_identity_for_docker_hosts() -> None:
    profile = KubernetesJobExecutionProfile(
        config=KubernetesJobExecutionProfileConfig(storage=KubernetesJobStorageConfig(pvc_name="jobs-pvc"))
    )
    require_fileset_sandbox_storage_identity(
        target(environment=FilesetRef(root="default/custom-gym")),
        capable_config(sandbox_host_provider="docker"),
        execution_profile=profile,
    )


def test_an_incapable_cluster_refuses_rather_than_running_colocated() -> None:
    """Falling back would run user environment code beside the job's credentials, unannounced."""
    config = capable_config(sandbox_cluster_capable=False)

    with pytest.raises(SandboxUnavailableError, match="sandbox_cluster_capable"):
        resolve_sandbox_plan(config, target())


def test_a_host_without_storage_cannot_be_provisioned() -> None:
    config = capable_config(sandbox_job_storage_pvc_claim=None)

    with pytest.raises(SandboxUnavailableError, match="sandbox_job_storage_pvc_claim"):
        resolve_sandbox_plan(config, target())


# --------------------------------------------------------------------------------------------
# Credentials
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["OPENAI_API_KEY", "NVIDIA_APIKEY", "HF_TOKEN", "DB_PASSWORD", "MY_SECRET", "SSH_PRIVATE_KEY"],
)
def test_credential_shaped_env_vars_are_recognised(name: str) -> None:
    assert credential_shaped_env_vars(target(env_vars={name: "x"})) == [name]


def test_ordinary_env_vars_are_left_alone() -> None:
    # `wmt_translation` needs WMT_TRANSLATION_COMET_PY_CACHE; refusing that would break a real
    # environment for no security gain.
    assert credential_shaped_env_vars(target(env_vars={"WMT_TRANSLATION_COMET_PY_CACHE": "/shared"})) == []


def test_a_plaintext_credential_blocks_a_sandboxed_run_and_names_the_route_out() -> None:
    """In sandboxed mode `env_vars` reaches a host running user-supplied environment code."""
    with pytest.raises(SandboxUnavailableError) as excinfo:
        resolve_sandbox_plan(capable_config(), target(env_vars={"OPENAI_API_KEY": "sk-real"}))

    message = str(excinfo.value)
    assert "OPENAI_API_KEY" in message
    assert "env_secrets" in message, "the message must name the supported alternative"
    assert "sk-real" not in message, "the refusal must not echo the credential it is protecting"


def test_the_same_credential_as_a_secret_ref_is_allowed() -> None:
    # The whole point of the gate: it blocks the plaintext route, not the capability.
    assert resolve_sandbox_plan(capable_config(), target(env_secrets={"OPENAI_API_KEY": "ws/openai"})) is not None


# --------------------------------------------------------------------------------------------
# Egress
#
# A sandboxed host denies everything but the broker, so what reaches the serve config is the whole
# of what the environment can talk to. The original wiring passed an empty allowlist and every test
# still passed, because a stubbed host needs no egress -- these exist so that cannot recur.
# --------------------------------------------------------------------------------------------


def test_the_model_endpoint_reaches_the_host_as_an_egress_allowance() -> None:
    payload = serve_config(target(), capable_plan(), job_id="job-1")

    assert payload["policy_base_urls"] == ("https://integrate.api.nvidia.com/v1",)


def test_extra_destinations_are_parsed_into_host_and_port() -> None:
    plan = capable_plan(sandbox_egress_allow=("cache.internal:6379", "metrics.svc.cluster.local:9090"))

    rules = serve_config(target(), plan, job_id="job-1")["sandbox"]["network_policy"]["egress_allow"]

    assert rules == [
        {"host": "cache.internal", "port": 6379},
        {"host": "metrics.svc.cluster.local", "port": 9090},
    ]


@pytest.mark.parametrize("entry", ["no-port", "host:", ":8080", "host:not-a-number"])
def test_a_malformed_egress_entry_is_refused_rather_than_skipped(entry: str) -> None:
    # Skipping it would grant less than the operator asked for, and the run would fail at a network
    # call with nothing pointing back at the typo.
    plan = capable_plan(sandbox_egress_allow=(entry,))

    with pytest.raises(SandboxUnavailableError, match="host:port"):
        serve_config(target(), plan, job_id="job-1")


def test_a_deployment_with_no_egress_at_all_is_refused() -> None:
    """The bug this pair of settings was added for.

    Without either, the host comes up healthy and then fails at the first rollout, because it has
    no route to a model -- a failure as far as possible from the setting that caused it.
    """
    config = capable_config(sandbox_policy_base_urls=(), sandbox_egress_allow=())

    with pytest.raises(SandboxUnavailableError, match="no route to a model"):
        resolve_sandbox_plan(config, target())


def test_egress_is_not_something_a_job_can_widen() -> None:
    # A submitted target must not be able to grant its own sandbox a destination; egress is
    # trusted-side policy, which is why it is read only from deployment config.
    hostile = target(hydra_params={"policy_base_urls": ["https://exfiltrate.example"]})

    payload = serve_config(hostile, capable_plan(), job_id="job-1")

    assert payload["policy_base_urls"] == ("https://integrate.api.nvidia.com/v1",)


# --------------------------------------------------------------------------------------------
# The assembled host spec
#
# `serve_config` returns a dict; what matters is the `GymHostSpec` the orchestrator builds from it,
# because that is what a provider acts on. Asserting the dict alone is how the empty egress
# allowlist survived. These build the real spec.
# --------------------------------------------------------------------------------------------


def built_host_spec(
    plan: SandboxPlan,
    gym_target: GymRunnerTarget | None = None,
    *,
    persistent_storage_path: Path | None = None,
):
    from sandboxed_gym import SandboxedGymServeConfig
    from sandboxed_gym.config import BrokerEndpoint
    from sandboxed_gym.orchestrator import build_gym_host_spec

    payload = serve_config(
        gym_target or target(),
        plan,
        job_id="job-9",
        persistent_storage_path=persistent_storage_path,
    )
    broker = BrokerEndpoint(url="http://10.0.0.5:51234", host="10.0.0.5", port=51234, token="tok")
    return build_gym_host_spec(SandboxedGymServeConfig.model_validate(payload), broker)


def test_the_model_endpoint_and_broker_both_reach_the_built_spec() -> None:
    spec = built_host_spec(capable_plan(sandbox_egress_allow=("cache.internal:6379",)))

    assert {(rule.host, rule.port) for rule in spec.egress_allow} == {
        ("integrate.api.nvidia.com", 443),
        ("10.0.0.5", 51234),
        ("cache.internal", 6379),
    }


def test_the_environment_and_workspace_mounts_are_not_the_same_directory() -> None:
    """They share a PVC claim, so only the sub-path separates them.

    Equal sub-paths would mount one directory twice -- once read-only as the environment, once
    writable as the workspace -- letting a run modify the environment it is evaluating.
    """
    spec = built_host_spec(capable_plan())

    assert spec.environment_mount.read_only is True
    assert spec.workspace_mount.read_only is False
    assert spec.environment_mount.sub_path == "environment"
    assert spec.workspace_mount.sub_path == "workspace/job-9"


def test_custom_environment_uses_the_read_only_host_mount() -> None:
    spec = built_host_spec(
        capable_plan(),
        target(environment=FilesetRef(root="default/custom-gym")),
    )

    assert spec.bootstrap_env["NMP_ENVIRONMENT_PATH"] == "/job/environment"


def test_fileset_backed_docker_mounts_the_subprocess_persistent_storage(tmp_path: Path) -> None:
    persistent = tmp_path / "default" / "job-9" / "1" / "job-storage"
    persistent.mkdir(parents=True)
    plan = capable_plan(
        sandbox_host_provider="docker",
        sandbox_host_provider_options={"root_dir": "/ignored", "network": "nmp-test"},
    )

    payload = serve_config(
        target(environment=FilesetRef(root="default/custom-gym")),
        plan,
        job_id="job-9",
        persistent_storage_path=persistent,
    )

    sandbox = payload["sandbox"]
    assert sandbox["host_provider_options"] == {
        "root_dir": str(persistent.parent),
        "network": "nmp-test",
    }
    assert (
        Path(sandbox["host_provider_options"]["root_dir"])
        / sandbox["environment_pvc_claim"]
        / sandbox["environment_sub_path"]
        == persistent / "environment"
    )
    assert (
        Path(sandbox["host_provider_options"]["root_dir"])
        / sandbox["workspace_pvc_claim"]
        / sandbox["workspace_sub_path"]
        == persistent / "workspace"
    )


def test_fileset_backed_docker_requires_the_trusted_persistent_storage_path() -> None:
    plan = capable_plan(sandbox_host_provider="docker")

    with pytest.raises(SandboxUnavailableError, match="persistent storage path"):
        serve_config(
            target(environment=FilesetRef(root="default/custom-gym")),
            plan,
            job_id="job-9",
        )


def test_opensandbox_keeps_pvc_subpaths_when_the_local_storage_path_is_available(tmp_path: Path) -> None:
    persistent = tmp_path / "job-storage"

    payload = serve_config(
        target(environment=FilesetRef(root="default/custom-gym")),
        capable_plan(),
        job_id="job-9",
        workspace="dev",
        persistent_storage_path=persistent,
    )

    sandbox = payload["sandbox"]
    assert sandbox["environment_pvc_claim"] == "job-storage"
    assert sandbox["environment_sub_path"] == "jobs/dev/job-9/environment"
    assert sandbox["workspace_sub_path"] == "jobs/dev/job-9/workspace"


def test_resource_requests_reach_the_host_when_configured() -> None:
    spec = built_host_spec(capable_plan(sandbox_resources={"cpu": "2", "memory": "8Gi"}))

    assert spec.resources == {"cpu": "2", "memory": "8Gi"}


def test_the_runtime_image_and_job_id_reach_the_host() -> None:
    spec = built_host_spec(capable_plan())

    assert spec.runtime_image == "registry.example.com/nmp-gym-runtime:1.0"
    assert spec.job_id == "job-9"


def test_the_selection_reaches_the_host_as_gym_config_paths() -> None:
    # The end of the mapping chain: what the submitter chose is what the host will load.
    import json as _json

    from sandboxed_gym.runtime.gym_host_runtime import GYM_GLOBAL_CONFIG_ENV_KEY

    spec = built_host_spec(capable_plan())

    global_config = _json.loads(spec.bootstrap_env[GYM_GLOBAL_CONFIG_ENV_KEY])
    assert "resources_servers/mcqa/configs/mcqa.yaml" in global_config["config_paths"]


# --------------------------------------------------------------------------------------------
# The job's own environment, and the writable workspace
# --------------------------------------------------------------------------------------------


def test_concurrent_runs_do_not_share_one_writable_workspace() -> None:
    # The configured sub-path is deployment-wide and the workspace mount is read-write, so an
    # unscoped path would let two evaluations running at once overwrite each other's state.
    plan = capable_plan()
    first = serve_config(target(), plan, job_id="job-a")["sandbox"]["workspace_sub_path"]
    second = serve_config(target(), plan, job_id="job-b")["sandbox"]["workspace_sub_path"]

    assert first != second
    assert first.endswith("/job-a")
    assert second.endswith("/job-b")


def test_the_environment_mount_stays_shared_across_runs() -> None:
    # Read-only and identical for every run: scoping it per job would defeat the shared cache.
    plan = capable_plan()
    first = serve_config(target(), plan, job_id="job-a")["sandbox"]["environment_sub_path"]
    second = serve_config(target(), plan, job_id="job-b")["sandbox"]["environment_sub_path"]

    assert first == second


def test_env_vars_reach_the_host_rather_than_being_dropped() -> None:
    # `wmt_translation` and friends are configurable only this way, and the host is where the
    # environment actually runs.
    payload = serve_config(
        target(env_vars={"WMT_TRANSLATION_COMET_PY_CACHE": "/shared"}), capable_plan(), job_id="job-1"
    )

    assert payload["host_env"]["WMT_TRANSLATION_COMET_PY_CACHE"] == "/shared"


def test_a_resolved_env_secret_reaches_the_host(monkeypatch: pytest.MonkeyPatch) -> None:
    # The service resolves the reference into this container; its value exists nowhere else, so if
    # it is not forwarded the environment runs inside the sandbox without its credential.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-resolved")
    payload = serve_config(target(env_secrets={"OPENAI_API_KEY": "ws/openai"}), capable_plan(), job_id="job-1")

    assert payload["host_env"]["OPENAI_API_KEY"] == "sk-resolved"


def test_an_unresolved_env_secret_is_named_rather_than_silently_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(SandboxUnavailableError, match="OPENAI_API_KEY"):
        serve_config(target(env_secrets={"OPENAI_API_KEY": "ws/openai"}), capable_plan(), job_id="job-1")


def test_the_host_env_reaches_the_built_spec() -> None:
    from sandboxed_gym.host.models import BROKER_URL_ENV

    spec = built_host_spec(capable_plan(), target(env_vars={"WMT_TRANSLATION_COMET_PY_CACHE": "/shared"}))

    assert spec.bootstrap_env["WMT_TRANSLATION_COMET_PY_CACHE"] == "/shared"
    assert spec.bootstrap_env[BROKER_URL_ENV] == "http://10.0.0.5:51234"


def test_a_job_cannot_move_the_broker_by_naming_its_variable() -> None:
    # `env_vars` is job-authored. Redefining NMP_BROKER_URL would point the host's rollouts at the
    # job's own listener, outside the broker's mediation.
    with pytest.raises(ValueError, match="NMP_BROKER_URL"):
        built_host_spec(capable_plan(), target(env_vars={"NMP_BROKER_URL": "http://evil"}))


# --------------------------------------------------------------------------------------------
# The plan travels from the service to the job
# --------------------------------------------------------------------------------------------


def test_the_job_reads_the_plan_the_service_resolved(monkeypatch: pytest.MonkeyPatch) -> None:
    # The job container has none of the operator's `NEMO_EVALUATOR_SANDBOX_*` variables, so
    # re-deriving the plan there would silently yield model defaults.
    from nemo_evaluator.jobs.gym_sandbox import GYM_SANDBOX_PLAN_ENVVAR, sandbox_plan_from_environment

    monkeypatch.setenv(GYM_SANDBOX_PLAN_ENVVAR, capable_plan().model_dump_json())

    assert sandbox_plan_from_environment() == capable_plan()


def test_no_plan_in_the_environment_means_gym_runs_colocated(monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_evaluator.jobs.gym_sandbox import GYM_SANDBOX_PLAN_ENVVAR, sandbox_plan_from_environment

    monkeypatch.delenv(GYM_SANDBOX_PLAN_ENVVAR, raising=False)

    assert sandbox_plan_from_environment() is None
