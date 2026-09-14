# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run provenance: target identity via the runner contract, and timings."""

from __future__ import annotations

import json
from typing import cast

from nemo_evaluator_sdk.agent_eval.evaluator import _describe_target
from nemo_evaluator_sdk.agent_eval.runtimes.sandbox.base import SandboxProvider
from nemo_evaluator_sdk.agent_eval.trials import AgentTaskRunner, RunnerInfo
from nemo_evaluator_sdk.values import Model


class _Runner:
    """A minimal runner fulfilling the AgentTaskRunner contract."""

    def runner_info(self) -> RunnerInfo:
        return RunnerInfo(name="gym", kind="runner", version="1.2.3", config={"resources_server": "mcqa"})

    async def run_tasks(self, tasks, config=None):  # pragma: no cover - not exercised here
        return []


class _RunnerMissingInfo:
    """A would-be runner without runner_info — no longer an AgentTaskRunner."""

    async def run_tasks(self, tasks, config=None):  # pragma: no cover - not exercised here
        return []


def test_runner_contract_is_satisfied_and_used() -> None:
    runner = _Runner()
    assert isinstance(runner, AgentTaskRunner)

    info = _describe_target(runner)
    assert (info.name, info.kind, info.version) == ("gym", "runner", "1.2.3")
    assert info.config == {"resources_server": "mcqa"}


def test_runner_info_is_required_by_the_runner_contract() -> None:
    # runner_info is part of AgentTaskRunner, not an optional add-on: a class without it is not a
    # runner, so the evaluator won't dispatch to it and provenance can never be missing.
    assert not isinstance(_RunnerMissingInfo(), AgentTaskRunner)


def test_every_shipped_runner_reports_a_stable_name_and_result_shaping_config() -> None:
    """Every shipped runner: a curated name (not a class name) and the settings that change results.

    Provenance that omits a result-shaping setting is worse than none — two runs that behaved
    differently would record identical metadata — so assert each runner surfaces its own knobs.
    """
    from pathlib import Path

    from nemo_evaluator_sdk.agent_eval.runtimes.callable_runtime import CallableAgentTaskRunner
    from nemo_evaluator_sdk.agent_eval.runtimes.docker_sandbox import DockerSandboxAgentRuntime
    from nemo_evaluator_sdk.agent_eval.runtimes.fabric.container_runtime import FabricContainerRuntime
    from nemo_evaluator_sdk.agent_eval.runtimes.fabric.runtime import FabricAgentRuntime
    from nemo_evaluator_sdk.agent_eval.runtimes.gym import GymAgentTaskRunner, GymRuntimeConfig
    from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import HarborAgentTaskRunner, HarborRuntimeConfig

    async def _agent_fn(task):  # pragma: no cover - never called
        return None

    class _Provider:
        """Identity-only stub: runner_info reads `provider.name` and nothing else, so the rest of the
        SandboxProvider protocol is deliberately unimplemented and cast at the call site."""

        name = "docker"

    harness = {"harness": {"adapter_id": "nvidia.fabric.codex"}}
    runners = [
        (CallableAgentTaskRunner(_agent_fn), "callable", {"agent_fn", "parallelism"}),
        (DockerSandboxAgentRuntime(), "docker_sandbox", {"model", "image", "timeout_s", "instructions"}),
        (
            FabricAgentRuntime(config=harness),
            "fabric",
            {"model", "timeout_s", "adapter_id", "skills", "capture_trajectory"},
        ),
        (
            FabricContainerRuntime(config=harness, provider=cast(SandboxProvider, _Provider())),
            "fabric_container",
            {"provider", "image", "adapter_id", "skills"},
        ),
        (
            GymAgentTaskRunner(config=GymRuntimeConfig(agent="a", agent_config="c", resources_server="r")),
            "gym",
            {
                "resources_server",
                "agent",
                "model_type",
                "num_repeats",
                "bind_resources_server",
                "hydra_params",
                "reward_key",
            },
        ),
        (
            HarborAgentTaskRunner(config=HarborRuntimeConfig(jobs_dir=Path("/jobs"))),
            "harbor",
            {
                "agent_name",
                "agent_import_path",
                "agent_kwargs",
                "effective_agent",
                "n_attempts",
                "jobs_dir",
                "reward_key",
            },
        ),
    ]

    for runner, expected_name, expected_config_keys in runners:
        info = runner.runner_info()
        assert info.name == expected_name, f"{type(runner).__name__} reported {info.name!r}"
        assert info.kind == "runner"
        missing = expected_config_keys - set(info.config)
        assert not missing, f"{expected_name} omits result-shaping config: {sorted(missing)}"


def test_provider_identity_is_stable_not_a_repr() -> None:
    # str(provider) yields a memory address, so two identical runs would record different metadata.
    from nemo_evaluator_sdk.agent_eval.runtimes.fabric.container_runtime import FabricContainerRuntime

    class _Provider:
        """Identity-only stub; see the note in the shipped-runners test above."""

        name = "docker"

    provider = cast(SandboxProvider, _Provider())
    info = FabricContainerRuntime(config={"harness": {"adapter_id": "x"}}, provider=provider).runner_info()
    assert info.config["provider"] == "docker"
    assert "0x" not in info.config["provider"]


def test_fabric_records_a_config_supplied_model_not_just_an_explicit_one() -> None:
    # _compose_config only overrides the config's default model when `model=` was passed explicitly, so
    # a config-supplied model is what actually runs. Reporting the constructor arg alone would give two
    # runs with different models identical provenance.
    from nemo_evaluator_sdk.agent_eval.runtimes.fabric.runtime import FabricAgentRuntime

    config = {"harness": {"adapter_id": "nvidia.fabric.codex"}, "models": {"default": {"model": "gpt-from-config"}}}

    assert FabricAgentRuntime(config=config).runner_info().config["model"] == "gpt-from-config"
    # An explicit model still wins — that is the precedence _compose_config applies.
    assert FabricAgentRuntime(config=config, model="gpt-explicit").runner_info().config["model"] == "gpt-explicit"
    assert FabricAgentRuntime(config={"harness": {"adapter_id": "x"}}).runner_info().config["model"] is None


def test_fabric_records_whether_trajectory_evidence_was_captured() -> None:
    # With capture off, no relay/ATIF exporter runs and the trial carries no trajectory evidence — so a
    # trajectory-scoring metric sees something different. Both modes must be distinguishable afterwards.
    from nemo_evaluator_sdk.agent_eval.runtimes.fabric.runtime import FabricAgentRuntime

    config = {"harness": {"adapter_id": "nvidia.fabric.codex"}}
    assert FabricAgentRuntime(config=config, capture_trajectory=True).runner_info().config["capture_trajectory"] is True
    assert (
        FabricAgentRuntime(config=config, capture_trajectory=False).runner_info().config["capture_trajectory"] is False
    )


def test_model_target_and_imported_trials_are_identified() -> None:
    model = _describe_target(Model(name="gpt-x", url="https://example/v1/chat/completions"))
    assert (model.name, model.kind) == ("gpt-x", "model")

    imported = _describe_target(None)  # trials supplied directly, no target ran
    assert (imported.name, imported.kind) == ("imported", "imported")


def test_harbor_records_the_effective_agent_when_a_custom_import_path_overrides_the_name() -> None:
    # run_job uses agent_import_path when set and ignores agent_name, so recording agent_name alone
    # would give two runs with different custom agents identical provenance.
    from pathlib import Path

    from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import HarborAgentTaskRunner, HarborRuntimeConfig

    def _info(**kwargs):
        return HarborAgentTaskRunner(config=HarborRuntimeConfig(jobs_dir=Path("/jobs"), **kwargs)).runner_info().config

    custom = _info(agent_import_path="pkg_a:Agent", agent_model_name="model-a")
    assert custom["effective_agent"] == "pkg_a:Agent"
    assert custom["agent_model_name"] == "model-a"

    other = _info(agent_import_path="pkg_b:Agent", agent_model_name="model-b")
    assert other["effective_agent"] != custom["effective_agent"]

    # Built-in agents still resolve through agent_name, defaulting to Harbor's oracle.
    assert _info(agent_name="oracle")["effective_agent"] == "oracle"
    assert _info()["effective_agent"] == "oracle"


def test_gym_redacts_credential_looking_hydra_params() -> None:
    # hydra_params is a free-form Hydra escape hatch forwarded to `gym env start`, and RunnerInfo.config
    # is persisted into the run bundle — so a value that looks like a credential must not be written there.
    from nemo_evaluator_sdk.agent_eval.runtimes.gym import GymAgentTaskRunner, GymRuntimeConfig

    runner = GymAgentTaskRunner(
        config=GymRuntimeConfig(
            agent="a",
            agent_config="c",
            resources_server="r",
            hydra_params={
                "model": {"api_key": "sk-should-not-be-recorded", "temperature": 0.7},
                "env": {"HF_TOKEN": "hf_should-not-be-recorded"},
                "agent": {"nested": {"secret": "deep-should-not-be-recorded"}},
                "models": [
                    {"name": "m", "api_key": "sk-in-a-list-should-not-be-recorded"},
                ],
                "api_keys": ["sk-whole-list-should-not-be-recorded"],
            },
        )
    )

    recorded = runner.runner_info().config["hydra_params"]

    assert recorded == {
        "model": {
            "api_key": "<redacted>",
            "temperature": 0.7,  # not credential-shaped, kept verbatim for reproducibility
        },
        "env": {"HF_TOKEN": "<redacted>"},
        # Matching is on the full dotted path, so a credential stays redacted at any depth.
        "agent": {"nested": {"secret": "<redacted>"}},
        # A mapping inside a list reaches Gym just as a nested one does, so it is walked too. The
        # index contributes no path segment: the key marks the credential, not the position.
        "models": [{"name": "m", "api_key": "<redacted>"}],
        # A credential-shaped key wins over descending into it — the whole list goes.
        "api_keys": "<redacted>",
    }
    assert "should-not-be-recorded" not in json.dumps(recorded)


def test_harbor_redacts_credential_looking_agent_kwargs() -> None:
    # agent_kwargs is forwarded verbatim to the Harbor agent's constructor and RunnerInfo.config is
    # persisted into the run bundle — the same exposure as Gym's hydra_params, so the same redaction.
    from pathlib import Path

    from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import HarborAgentTaskRunner, HarborRuntimeConfig

    runner = HarborAgentTaskRunner(
        config=HarborRuntimeConfig(
            jobs_dir=Path("/jobs"),
            agent_import_path="pkg:Agent",
            agent_kwargs={
                "fabric_adapter_id": "nvidia.fabric.codex",
                "extra_env": {"OPENAI_API_KEY": "sk-should-not-be-recorded", "HOME": "/root"},
                "fabric_harness_settings": {"auth": {"token": "tok-should-not-be-recorded"}},
            },
        )
    )

    recorded = runner.runner_info().config["agent_kwargs"]

    assert recorded == {
        "fabric_adapter_id": "nvidia.fabric.codex",
        "extra_env": {"OPENAI_API_KEY": "<redacted>", "HOME": "/root"},
        "fabric_harness_settings": {"auth": {"token": "<redacted>"}},
    }
    assert "should-not-be-recorded" not in json.dumps(recorded)


def test_gym_redacts_credential_looking_env_vars() -> None:
    # env_vars needs redaction at least as much as hydra_params: an environment variable is the
    # conventional way to hand a process an API key, so a caller doing the obvious thing would
    # otherwise write one straight into the run bundle.
    from nemo_evaluator_sdk.agent_eval.runtimes.gym import GymAgentTaskRunner, GymRuntimeConfig

    runner = GymAgentTaskRunner(
        config=GymRuntimeConfig(
            agent="a",
            agent_config="c",
            resources_server="r",
            env_vars={
                "WMT_TRANSLATION_COMET_PY_CACHE": "/shared/cache",
                "OPENAI_API_KEY": "sk-should-not-be-recorded",
                "HF_TOKEN": "hf_should-not-be-recorded",
                "AWS_SECRET_ACCESS_KEY": "should-not-be-recorded",
                "DB_PASSWORD": "should-not-be-recorded",
            },
        )
    )

    recorded = runner.runner_info().config["env_vars"]

    assert recorded == {
        # The reason the field exists — a cache path is provenance worth keeping verbatim.
        "WMT_TRANSLATION_COMET_PY_CACHE": "/shared/cache",
        "OPENAI_API_KEY": "<redacted>",
        "HF_TOKEN": "<redacted>",
        "AWS_SECRET_ACCESS_KEY": "<redacted>",
        "DB_PASSWORD": "<redacted>",
    }
    assert "should-not-be-recorded" not in json.dumps(recorded)


def test_model_provenance_records_the_endpoint_and_invocation_params() -> None:
    # A name alone is not an identity: the same model served from two URLs, or run at two different
    # temperatures, would otherwise record identical provenance.
    from nemo_evaluator_sdk.values import RunConfigOnlineModel
    from nemo_evaluator_sdk.values.params import InferenceParams

    params = RunConfigOnlineModel(inference=InferenceParams(temperature=0.0, max_tokens=256), max_retries=5)
    info = _describe_target(Model(name="gpt-x", url="https://a.example/v1/chat/completions"), params)

    assert (info.name, info.kind) == ("gpt-x", "model")
    assert info.config["url"] == "https://a.example/v1/chat/completions"
    assert info.config["params"]["inference"] == {"temperature": 0.0, "max_tokens": 256}
    assert info.config["params"]["max_retries"] == 5

    # Same name, different endpoint -> distinguishable.
    other = _describe_target(Model(name="gpt-x", url="https://b.example/v1/chat/completions"), params)
    assert other.config["url"] != info.config["url"]


def _load_example(name: str):
    """Import an example runtime module.

    The example runtimes use relative imports, so they are imported as a package with the SDK
    package root on sys.path.
    """
    import importlib
    import sys
    from pathlib import Path

    root = str(Path(__file__).resolve().parents[2])
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module(f"examples.run_agent_eval.{name}")


def _example_runtimes() -> list[object]:
    aut = _load_example("aut_runtime")
    platform = _load_example("platform_runtime")
    workflow = _load_example("workflow_runtime")
    return [
        workflow.WorkflowAgentRuntime(),
        aut.NatAutRuntime(aut.AutConfig(aut_agent_name="an-agent")),
        platform.NatWorkflowRuntime(),
    ]


def test_example_runtimes_still_satisfy_the_runner_contract() -> None:
    """Making runner_info required silently breaks any runner that lacks it.

    `AgentTaskRunner` is runtime_checkable and isinstance-dispatched, so a runner without
    `runner_info` stops matching and the evaluator raises "unsupported agent-eval target type" — at
    run time, with nothing at import time to warn you. The shipped runtimes were updated; the example
    runtimes were missed once already, so assert on those too.
    """
    for runtime in _example_runtimes():
        assert isinstance(runtime, AgentTaskRunner), f"{type(runtime).__name__} is no longer an AgentTaskRunner"
        assert runtime.runner_info().name, f"{type(runtime).__name__} reported an empty name"


def test_example_runner_provenance_excludes_the_api_keys_its_config_carries() -> None:
    # AutConfig and NatWorkflowConfig hold nvidia/anthropic API keys alongside their result-shaping
    # settings, and RunnerInfo.config is persisted with the run — so these must enumerate fields
    # rather than dump the config.
    aut_module = _load_example("aut_runtime")
    platform_module = _load_example("platform_runtime")

    aut = aut_module.NatAutRuntime(
        aut_module.AutConfig(
            aut_agent_name="an-agent",
            nvidia_api_key="nvapi-secret",
            inference_nvidia_api_key="nvapi-secret-2",
            anthropic_api_key="sk-ant-secret",
        )
    ).runner_info()
    workflow = platform_module.NatWorkflowRuntime(
        platform_module.NatWorkflowConfig(nvidia_api_key="nvapi-secret")
    ).runner_info()

    for info in (aut, workflow):
        assert "secret" not in str(info.config), f"{info.name} leaked a credential into provenance"
        assert not any("api_key" in key for key in info.config)
