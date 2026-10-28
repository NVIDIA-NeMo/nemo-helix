# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A task Job's input, its compiled Job, and what the dispatcher changes inside it."""

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from harbor.agents.base import BaseAgent
from harbor.agents.factory import AgentFactory
from harbor.environments.factory import EnvironmentFactory
from harbor.models.task.config import EnvironmentConfig, TaskConfig
from harbor.models.task.verifier_mode import resolve_effective_verifier_env_config
from harbor.models.trial.config import AgentConfig, ResourceMode
from harbor.models.trial.config import EnvironmentConfig as TrialEnvironmentConfig
from harbor.models.trial.paths import TrialPaths
from nemo_builder_plugin.entities import ImageRuntimeConfig
from nemo_evals.filesets import FilesetRef
from nemo_evals.harbor.agent_setup_egress import INNER_AGENT_KWARG, SETUP_EGRESS_AGENT, SETUP_HOSTS_KWARG
from nemo_evals.harbor.compose.models import ComposeError
from nemo_evals.harbor.remote.profile import ResourceOverrides, resource_defaults
from nemo_evals.jobs.agent_evaluate import AsyncAgentEvalJob
from nemo_evals.jobs.agent_spec import (
    AgentEvalSpec,
    HarborBuiltinAgentSource,
    HarborGatewayModel,
    HarborRunnerTarget,
    RegisteredAgentSource,
)
from nemo_evals.jobs.gateway_agents import GATEWAY_PLACEHOLDER_KEY
from nemo_evals.jobs.harbor_task_dispatch import (
    DISPATCH_COMMAND,
    OPENSANDBOX_ENVIRONMENT,
    ExecutionDeadlinePassedError,
    HarborTaskAgentEvalJob,
    HarborTaskDispatchJob,
    SandboxServerChangedError,
    bind_task_image,
    check_execution_deadline,
    check_sandbox_server,
    configure_runtime,
    image_entrypoints,
    prepared_compose_services,
)
from nemo_evals.jobs.harbor_task_spec import (
    HarborTaskJobSpec,
    PreparedHarborEnvironment,
    PreparedImage,
    build_task_job_input,
)
from nemo_evals.tasks import harbor_task_dispatch as entrypoint
from nemo_evals.taskset_runs.specs import RunIntakePublication
from nemo_helix_plugin.job_context import JobContext
from nemo_helix_plugin.jobs.api_factory import CPUExecutionProvider
from nhx_evals_sdk.agent_eval.runtimes.harbor.runtime import HarborRuntimeConfig
from nhx_evals_sdk.agent_eval.runtimes.harbor.tasks import HARBOR_TASK_DIR_KEY
from nhx_evals_sdk.agent_eval.tasks import AgentEvalTask
from nhx_evals_sdk.values import ModelRef
from pydantic import ValidationError

NOW = datetime(2026, 10, 15, 12, tzinfo=timezone.utc)
AGENT = PreparedImage(image_ref="registry.invalid/agent@sha256:" + "a" * 64, preparation_key="ka")
VERIFIER = PreparedImage(image_ref="registry.invalid/verifier@sha256:" + "b" * 64, preparation_key="kb")
ENVIRONMENT = PreparedHarborEnvironment(task_revision_digest="r", archive_digest="d", agent=AGENT, verifier=VERIFIER)


@pytest.fixture
def task_spec(harbor_plan, harbor_run) -> HarborTaskJobSpec:
    return build_task_job_input(harbor_plan, harbor_run, 1, ENVIRONMENT)


def test_input_runs_only_its_task_and_leaves_the_plan_alone(harbor_plan, task_spec):
    assert [t.id for t in task_spec.spec.tasks] == ["hello/sum-three"]
    assert len(harbor_plan.spec.tasks) == 2
    assert task_spec.member_id == "task-1" and task_spec.task_index == 1
    assert task_spec.execution_deadline_at == NOW + timedelta(hours=1)
    assert task_spec.sandbox_policy == harbor_plan.sandbox_policy
    assert HarborTaskJobSpec.model_validate_json(task_spec.model_dump_json()) == task_spec


def test_a_run_without_publication_gives_its_task_jobs_none(task_spec):
    assert task_spec.spec.publication is None


def test_a_publishing_run_gives_each_task_job_the_runs_evaluation_never_required(harbor_plan, harbor_run):
    intake = RunIntakePublication(
        experiment="swe",
        experiment_id="exp-1",
        agent_name="codex",
        dataset_name="d",
        dataset_version="1",
    )
    plan = harbor_plan.model_copy(update={"intake": intake})
    publication = build_task_job_input(plan, harbor_run, 1, ENVIRONMENT).spec.publication
    assert publication is not None and publication.intake is not None
    # Named after the run record, not the plan: a keyless run's id can change during acceptance.
    assert publication.intake.evaluation_id == harbor_run.name
    # The version is left to each trial's own trajectory (codex records its release).
    assert (publication.intake.agent_name, publication.intake.agent_version) == ("codex", "unknown")
    # A failed publish must not fail the task Job and drop its scored result from the run's bundle.
    assert publication.intake.required is False


def test_a_task_job_publishes_under_its_own_name_and_leaves_the_evaluation_to_the_controller(task_spec):
    job = HarborTaskAgentEvalJob()
    job._task_spec = task_spec
    # Stable sessions (<run>-task-<i>:<trial>): a re-publish replaces them instead of adding a random run.
    assert job._run_id() == f"{task_spec.run_id}-task-1"
    assert HarborTaskAgentEvalJob.record_publication_durations is False
    assert AsyncAgentEvalJob.record_publication_durations is True
    assert AsyncAgentEvalJob()._run_id() is None


def test_only_task_rejects_an_out_of_range_index(harbor_spec):
    with pytest.raises(IndexError):
        harbor_spec.only_task(2)


async def test_compiled_job_has_one_dispatch_step_without_sandbox_credentials(task_spec, harbor_config):
    compiled = await HarborTaskDispatchJob.compile(spec=task_spec, profile="k8s-cpu")
    (step,) = compiled.steps
    assert isinstance(step.executor, CPUExecutionProvider) and step.executor.profile == "k8s-cpu"
    assert step.executor.container.image == harbor_config.dispatcher_image
    assert step.executor.container.command == DISPATCH_COMMAND
    assert HarborTaskJobSpec.model_validate(step.config) == task_spec
    names = {variable.name for variable in step.environment or []}
    assert not {name for name in names if "SANDBOX" in name}
    assert not {name for name in names if "API_KEY" in name}


async def test_registered_agent_files_are_staged_first_on_the_dispatcher_image(task_spec, harbor_config):
    target = task_spec.spec.target.model_copy(
        update={
            "source": RegisteredAgentSource(agent="agents/coder", files=FilesetRef("agents/agent-files-1")),
            "agent_kwargs": {"fabric_config": {"harness": {"adapter_id": "deepagents"}}},
        }
    )
    spec = task_spec.model_copy(
        update={"spec": AgentEvalSpec.model_validate(task_spec.spec.model_copy(update={"target": target}).model_dump())}
    )
    stage, dispatch = (await HarborTaskDispatchJob.compile(spec=spec, profile="k8s-cpu")).steps
    assert isinstance(stage.executor, CPUExecutionProvider)
    assert stage.name == "stage-environment" and stage.executor.container.image == harbor_config.dispatcher_image
    assert stage.config["environment"] == "agents/agent-files-1"
    assert dispatch.name == "harbor-task-dispatch"


def test_task_jobs_keep_the_shared_agent_files_snapshot():
    assert HarborTaskAgentEvalJob.discard_agent_files is False
    assert AsyncAgentEvalJob.discard_agent_files is True


def test_dispatcher_refuses_another_sandbox_server(task_spec, monkeypatch):
    monkeypatch.setenv("OPEN_SANDBOX_DOMAIN", "other.svc")
    with pytest.raises(SandboxServerChangedError, match="sandbox_server_changed"):
        check_sandbox_server(task_spec)
    monkeypatch.setenv("OPEN_SANDBOX_DOMAIN", task_spec.sandbox_policy.connection.domain)
    check_sandbox_server(task_spec)


def test_dispatcher_refuses_to_start_after_the_execution_deadline(task_spec):
    deadline = task_spec.execution_deadline_at
    check_execution_deadline(task_spec, now=deadline - timedelta(seconds=1))
    with pytest.raises(ExecutionDeadlinePassedError, match="execution_deadline_passed"):
        check_execution_deadline(task_spec, now=deadline)


def test_a_task_job_started_after_the_deadline_stops_before_any_sandbox(task_spec, monkeypatch):
    # A submit retry can create a task Job after the controller expired the run and cancelled the Jobs it saw.
    assert task_spec.execution_deadline_at < datetime.now(timezone.utc)
    monkeypatch.setenv("OPEN_SANDBOX_DOMAIN", "other.svc")  # the deadline check comes first
    with pytest.raises(ExecutionDeadlinePassedError):
        HarborTaskAgentEvalJob().run(task_spec.model_dump(mode="json"), ctx=MagicMock(), async_client=MagicMock())


def test_entrypoint_aliases_jobs_injected_variables(monkeypatch):
    monkeypatch.delenv("OPENSANDBOX_API_KEY", raising=False)
    monkeypatch.setenv("OPEN_SANDBOX_API_KEY", "injected-at-runtime")
    monkeypatch.setattr(
        entrypoint, "run_async_task_main", lambda job, **kwargs: 0 if job is HarborTaskAgentEvalJob else 1
    )
    assert entrypoint.main() == 0
    assert os.environ["OPENSANDBOX_API_KEY"] == "injected-at-runtime"


def test_runtime_selects_opensandbox_with_frozen_policy_and_no_key(task_spec, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENSANDBOX_API_KEY", "private-worker-value")
    ctx = MagicMock(spec=JobContext)
    ctx.job_id = "run-0123456789abcdef-task-1"
    configured = configure_runtime(HarborRuntimeConfig(jobs_dir=tmp_path), ctx, task_spec)
    assert configured.environment is not None and configured.environment.import_path == OPENSANDBOX_ENVIRONMENT
    kwargs = configured.environment.kwargs
    assert kwargs["domain"] == task_spec.sandbox_policy.connection.domain
    assert kwargs["trusted_allowed_hosts"] == ["igw.svc.cluster.local"]
    assert kwargs["agent_channel_hosts"] == []
    assert kwargs["sandbox_timeout_sec"] == 30_600 and kwargs["use_server_proxy"] is True
    assert kwargs["enforce_disk_limit"] is True and kwargs["max_disk_mb"] == 131_072
    assert not any(key.startswith("override_") for key in kwargs)
    assert kwargs["metadata"] == {"nemo-run": "run-0123456789abcdef", "nemo-task-index": "1", "nemo-job": ctx.job_id}
    assert configured.n_concurrent_trials == 1
    assert "private-worker-value" not in json.dumps(configured.model_dump(mode="json"))


def _ctx() -> JobContext:
    ctx = MagicMock(spec=JobContext)
    ctx.job_id = "run-0123456789abcdef-task-1"
    return ctx


def test_runtime_passes_each_recorded_images_own_entrypoint(harbor_plan, harbor_run, tmp_path):
    agent = AGENT.model_copy(update={"runtime_config": ImageRuntimeConfig(entrypoint=["/bin/bash", "-l", "-c"])})
    environment = ENVIRONMENT.model_copy(update={"agent": agent})
    spec = build_task_job_input(harbor_plan, harbor_run, 1, environment)
    configured = configure_runtime(HarborRuntimeConfig(jobs_dir=tmp_path), _ctx(), spec)
    assert configured.environment is not None
    # The verifier's image wasn't recorded, so its sandbox starts as it did before.
    assert configured.environment.kwargs["image_entrypoints"] == {AGENT.image_ref: ["/bin/bash", "-l", "-c"]}


def test_image_entrypoints_name_an_image_without_one_and_skip_unrecorded_ones():
    agent = AGENT.model_copy(update={"runtime_config": ImageRuntimeConfig(cmd=["/app/run.sh"])})
    assert image_entrypoints(ENVIRONMENT.model_copy(update={"agent": agent})) == {AGENT.image_ref: []}
    assert image_entrypoints(ENVIRONMENT) == {}


def test_runtime_without_recorded_images_passes_no_entrypoints(task_spec, tmp_path):
    configured = configure_runtime(HarborRuntimeConfig(jobs_dir=tmp_path), _ctx(), task_spec)
    assert configured.environment is not None
    assert "image_entrypoints" not in configured.environment.kwargs


def test_runtime_passes_the_frozen_agent_channel(task_spec, tmp_path):
    policy = task_spec.sandbox_policy.model_copy(update={"agent_channel": ("igw.svc.cluster.local",)})
    spec = task_spec.model_copy(update={"sandbox_policy": policy})
    ctx = MagicMock(spec=JobContext)
    ctx.job_id = "run-0123456789abcdef-task-1"
    configured = configure_runtime(HarborRuntimeConfig(jobs_dir=tmp_path), ctx, spec)
    assert configured.environment is not None
    assert configured.environment.kwargs["agent_channel_hosts"] == ["igw.svc.cluster.local"]


def test_runtime_passes_agent_overrides_and_capped_defaults(task_spec, tmp_path):
    overrides = ResourceOverrides(cpus=4, storage_mb=20_480)
    policy = task_spec.sandbox_policy.model_copy(update={"resource_overrides": overrides, "max_memory_mb": 1024})
    spec = task_spec.model_copy(update={"sandbox_policy": policy})
    ctx = MagicMock(spec=JobContext)
    ctx.job_id = "run-0123456789abcdef-task-1"
    configured = configure_runtime(HarborRuntimeConfig(jobs_dir=tmp_path), ctx, spec)
    assert configured.environment is not None
    kwargs = configured.environment.kwargs
    assert kwargs["agent_resource_overrides"] == {"cpus": 4, "storage_mb": 20_480}
    assert kwargs["sandbox_resource_defaults"] == resource_defaults(policy) == {"cpus": 1, "memory_mb": 1024}
    assert not [key for key in kwargs if key.startswith("override_")]
    harbor = TrialEnvironmentConfig.model_validate(configured.environment.model_dump())  # as the runtime hands it over
    assert harbor.cpu_enforcement_policy == harbor.memory_enforcement_policy == ResourceMode.AUTO


def _sandbox(configured, tmp_path: Path, session_id: str, task_env_config: EnvironmentConfig) -> Any:
    """The sandbox environment Harbor's factory builds for ``session_id`` from the dispatcher's config."""
    (tmp_path / "env").mkdir(exist_ok=True)
    (tmp_path / "trial-x").mkdir(exist_ok=True)
    return EnvironmentFactory.create_environment_from_config(
        TrialEnvironmentConfig.model_validate(configured.environment.model_dump()),
        environment_dir=tmp_path / "env",
        environment_name="t",
        session_id=session_id,
        trial_paths=TrialPaths(trial_dir=tmp_path / "trial-x"),
        task_env_config=task_env_config,
    )


@pytest.mark.parametrize("verifier", ["explicit", "inherited"])
def test_overrides_size_the_agent_sandbox_only(task_spec, tmp_path, monkeypatch, verifier):
    monkeypatch.setenv("OPEN_SANDBOX_API_KEY", "")
    overrides = ResourceOverrides(cpus=4, memory_mb=8192)
    policy = task_spec.sandbox_policy.model_copy(update={"resource_overrides": overrides, "max_memory_mb": 1024})
    configured = configure_runtime(
        HarborRuntimeConfig(jobs_dir=tmp_path),
        MagicMock(spec=JobContext, job_id="j"),
        task_spec.model_copy(update={"sandbox_policy": policy}),
    )
    task = TaskConfig.model_validate(
        {
            "environment": {"docker_image": AGENT.image_ref, "cpus": 2, "storage_mb": 1024},
            "verifier": {"environment_mode": "separate"}
            | ({"environment": {"docker_image": VERIFIER.image_ref, "cpus": 1}} if verifier == "explicit" else {}),
        }
    )
    agent = _sandbox(configured, tmp_path, "trial-x__env", task.environment)
    verifier_config = resolve_effective_verifier_env_config(task, None)  # Harbor copies it after the agent's
    assert verifier_config is not None
    verifier_sandbox = _sandbox(configured, tmp_path, "trial-x__verifier__main", verifier_config)
    assert agent._resource()["cpu"] == "4" and agent._resource()["memory"] == "8192Mi"
    expected_cpus = "1" if verifier == "explicit" else "2"  # task.toml's, never the override
    assert verifier_sandbox._resource()["cpu"] == expected_cpus
    assert verifier_sandbox._resource()["memory"] == "1024Mi"  # the capped default
    assert task.environment.cpus == 2 and task.environment.memory_mb is None  # the task's own config is untouched


def test_a_storage_only_task_still_gets_cpu_and_memory_limits(task_spec, tmp_path, monkeypatch):
    monkeypatch.setenv("OPEN_SANDBOX_API_KEY", "")
    policy = task_spec.sandbox_policy.model_copy(update={"enforce_disk_limit": True, "max_disk_mb": None})
    configured = configure_runtime(
        HarborRuntimeConfig(jobs_dir=tmp_path),
        MagicMock(spec=JobContext, job_id="j"),
        task_spec.model_copy(update={"sandbox_policy": policy}),
    )
    config = EnvironmentConfig(docker_image=AGENT.image_ref, storage_mb=4096)
    resource = _sandbox(configured, tmp_path, "trial-x", config)._resource()
    assert resource == {"cpu": "1", "memory": "2048Mi", "ephemeral-storage": "4096Mi"}


def _configured_with_setup_hosts(task_spec, tmp_path, hosts: tuple[str, ...], **runtime: Any) -> HarborRuntimeConfig:
    policy = task_spec.sandbox_policy.model_copy(update={"agent_setup_egress": hosts})
    spec = task_spec.model_copy(update={"sandbox_policy": policy})
    ctx = MagicMock(spec=JobContext)
    ctx.job_id = "run-0123456789abcdef-task-1"
    return configure_runtime(HarborRuntimeConfig(jobs_dir=tmp_path, **runtime), ctx, spec)


@pytest.mark.parametrize("agent", ["claude-code", "codex"])
def test_setup_hosts_reach_the_environment_and_wrap_a_builtin_agent(task_spec, tmp_path, agent):
    configured = _configured_with_setup_hosts(
        task_spec, tmp_path, ("registry.npmjs.org",), agent_name=agent, agent_kwargs={"version": "2.1.0"}
    )
    assert configured.environment is not None
    assert configured.environment.kwargs["agent_setup_hosts"] == ["registry.npmjs.org"]
    assert configured.agent_import_path == SETUP_EGRESS_AGENT
    assert configured.agent_kwargs == {
        "version": "2.1.0",
        INNER_AGENT_KWARG: agent,
        SETUP_HOSTS_KWARG: ["registry.npmjs.org"],
    }


def _harbor_setup_timeout_sec(configured: HarborRuntimeConfig) -> float:
    """The setup timeout Harbor gives the trial: its agent class's default times the setup multiplier."""
    agent_class = AgentFactory.get_agent_class_from_config(AgentConfig(import_path=configured.agent_import_path))
    assert configured.agent_setup_timeout_multiplier is not None
    return agent_class.DEFAULT_SETUP_TIMEOUT_SEC * configured.agent_setup_timeout_multiplier


@pytest.mark.skipif(
    not hasattr(BaseAgent, "DEFAULT_SETUP_TIMEOUT_SEC"), reason="Harbor 0.23 uses one setup timeout for every agent"
)
@pytest.mark.parametrize(
    ("agent", "multipliers", "timeout_sec"),
    [
        ("codex", {}, 360.0),
        ("hermes", {}, 1200.0),
        ("hermes", {"agent_setup_timeout_multiplier": 2.0}, 2400.0),
        ("hermes", {"timeout_multiplier": 0.5}, 600.0),
        ("hermes", {"timeout_multiplier": 0.5, "agent_setup_timeout_multiplier": 3.0}, 3600.0),
    ],
)
def test_a_wrapped_agent_keeps_its_own_setup_timeout(task_spec, tmp_path, agent, multipliers, timeout_sec):
    """Harbor reads the setup timeout from the wrapper's class, so the dispatcher scales it to the inner agent's."""
    configured = _configured_with_setup_hosts(
        task_spec, tmp_path, ("registry.npmjs.org",), agent_name=agent, **multipliers
    )
    assert _harbor_setup_timeout_sec(configured) == pytest.approx(timeout_sec)


def test_an_unwrapped_agent_keeps_the_callers_setup_multiplier(task_spec, tmp_path):
    configured = _configured_with_setup_hosts(task_spec, tmp_path, (), agent_name="hermes")
    assert configured.agent_setup_timeout_multiplier is None


def test_setup_hosts_wrap_an_importable_custom_agent(task_spec, tmp_path):
    configured = _configured_with_setup_hosts(
        task_spec, tmp_path, ("registry.npmjs.org",), agent_import_path="harbor.agents.installed.codex:Codex"
    )
    assert configured.agent_import_path == SETUP_EGRESS_AGENT
    assert configured.agent_kwargs[INNER_AGENT_KWARG] == "harbor.agents.installed.codex:Codex"


@pytest.mark.parametrize("agent", ["oracle", "nop"])
def test_agents_that_install_nothing_are_not_wrapped(task_spec, tmp_path, agent, caplog):
    """Harbor constructs and scores oracle and nop by name, so wrapping them would break them."""
    with caplog.at_level(logging.INFO, logger="nemo_evals.jobs.harbor_task_dispatch"):
        configured = _configured_with_setup_hosts(task_spec, tmp_path, ("registry.npmjs.org",), agent_name=agent)
    assert configured.agent_import_path is None and configured.agent_name == agent
    assert f"Agent {agent} runs setup without the agent setup egress window: it installs nothing" in caplog.text


def test_without_setup_hosts_the_agent_is_not_wrapped(task_spec, tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger="nemo_evals.jobs.harbor_task_dispatch"):
        configured = _configured_with_setup_hosts(task_spec, tmp_path, (), agent_name="claude-code")
    assert "agent setup egress window" not in caplog.text
    assert configured.agent_import_path is None and configured.agent_name == "claude-code"
    assert INNER_AGENT_KWARG not in configured.agent_kwargs


def test_a_loose_agent_file_runs_unwrapped(task_spec, tmp_path, caplog):
    """The runtime picks a loose file's import path at run time, so it can't be named for the wrapper.

    Its install then fails on the task's own policy, so the log says why, for whoever debugs it.
    """
    with caplog.at_level(logging.INFO, logger="nemo_evals.jobs.harbor_task_dispatch"):
        configured = _configured_with_setup_hosts(
            task_spec, tmp_path, ("registry.npmjs.org",), agent_import_path="wrapper:Agent", agent_dir=tmp_path
        )
    assert "Agent wrapper:Agent runs setup without the agent setup egress window: a loose agent file" in caplog.text
    assert configured.agent_import_path == "wrapper:Agent"
    assert INNER_AGENT_KWARG not in configured.agent_kwargs


def write_task(root: Path, toml: str, *, compose: bool = False) -> Path:
    (root / "environment").mkdir(parents=True)
    (root / "task.toml").write_text(toml)
    if compose:
        (root / "environment" / "docker-compose.yaml").write_text("services: {}\n")
    return root


def test_bind_sets_agent_and_separate_verifier_images(tmp_path):
    import tomllib

    task = write_task(
        tmp_path / "t", '# keep me\n[environment]\ncpus = 1\n\n[verifier]\nenvironment_mode = "separate"\n'
    )
    assert bind_task_image(task, AGENT.image_ref, verifier_image_ref=VERIFIER.image_ref) is True
    text = (task / "task.toml").read_text()
    config = tomllib.loads(text)
    assert config["environment"]["docker_image"] == AGENT.image_ref
    assert config["verifier"]["environment"] == {"cpus": 1, "docker_image": VERIFIER.image_ref}
    assert text.startswith("# keep me")


def test_bind_leaves_a_shared_verifier_alone(tmp_path):
    import tomllib

    task = write_task(tmp_path / "t", "[environment]\n")
    assert bind_task_image(task, AGENT.image_ref, verifier_image_ref=VERIFIER.image_ref) is False
    assert "verifier" not in tomllib.loads((task / "task.toml").read_text())


def test_bind_leaves_a_compose_tasks_files_alone(tmp_path):
    import tomllib

    task = write_task(tmp_path / "t", "[environment]\n", compose=True)
    bind_task_image(task, AGENT.image_ref)
    assert tomllib.loads((task / "task.toml").read_text())["environment"]["docker_image"] == AGENT.image_ref
    assert (task / "environment" / "docker-compose.yaml").read_text() == "services: {}\n"


def test_a_no_network_task_off_the_agent_channel_stops_before_any_sandbox(task_spec, harbor_resolved_task, monkeypatch):
    """Its sandbox would deny every call the agent makes to its gateway model, so no trial is worth starting."""
    monkeypatch.setenv("OPEN_SANDBOX_DOMAIN", task_spec.sandbox_policy.connection.domain)
    monkeypatch.setenv("NHX_INTERNAL_BASE_URL", "http://nemo-api.ns.svc:8080")
    spec = task_spec.spec.model_copy(
        update={
            "tasks": [harbor_resolved_task("offline", config={"environment": {"allow_internet": False}})],
            "target": HarborRunnerTarget(
                source=HarborBuiltinAgentSource(name="codex", model=HarborGatewayModel(ref=ModelRef("ws/cb-gpt")))
            ),
        }
    )
    future = datetime.now(timezone.utc) + timedelta(hours=1)
    blocked = task_spec.model_copy(update={"spec": spec, "execution_deadline_at": future})

    with pytest.raises(ValueError, match="gateway_unreachable"):
        HarborTaskAgentEvalJob().run(blocked.model_dump(mode="json"), ctx=MagicMock(), async_client=MagicMock())


async def test_a_task_job_for_a_gateway_model_carries_the_placeholder_key(task_spec):
    """The agent reads it like any other key; the gateway checks none with platform auth off."""
    target = HarborRunnerTarget(
        source=HarborBuiltinAgentSource(name="claude-code", model=HarborGatewayModel(ref=ModelRef("ws/cb-gpt")))
    )
    spec = task_spec.model_copy(update={"spec": task_spec.spec.model_copy(update={"target": target})})

    compiled = await HarborTaskDispatchJob.compile(spec=spec, profile="k8s-cpu")

    (step,) = compiled.steps
    variables = {variable.name: variable for variable in step.environment or []}
    assert variables["ANTHROPIC_API_KEY"].value == GATEWAY_PLACEHOLDER_KEY
    assert variables["ANTHROPIC_API_KEY"].from_secret is None


def test_every_sandbox_ends_at_the_runs_execution_deadline(task_spec, tmp_path):
    """The deadline goes to the environment as an absolute time, so each create (a retry, a separate verifier)
    gets the time left then, not a lifetime measured once when the dispatcher started."""
    ctx = MagicMock(spec=JobContext)
    ctx.job_id = "run-0123456789abcdef-task-1"
    configured = configure_runtime(HarborRuntimeConfig(jobs_dir=tmp_path), ctx, task_spec)
    assert configured.environment is not None
    kwargs = configured.environment.kwargs
    assert kwargs["sandbox_timeout_sec"] == task_spec.sandbox_policy.max_sandbox_lifetime_s
    expires_at = kwargs["sandbox_expires_at"]
    assert isinstance(expires_at, str) and datetime.fromisoformat(expires_at) == task_spec.execution_deadline_at


SERVICE_DB = PreparedImage(image_ref="registry.invalid/db@sha256:" + "c" * 64, preparation_key="kc")
SERVICE_API = PreparedImage(image_ref="registry.invalid/api@sha256:" + "d" * 64, preparation_key="kd")
COMPOSE_ENVIRONMENT = ENVIRONMENT.model_copy(update={"compose_images": {"db": SERVICE_DB, "api": SERVICE_API}})


def write_compose_task(root: Path) -> Path:
    """A task with a built main (its Dockerfile has an ENTRYPOINT), a published ``db`` and a built ``api``."""
    task = write_task(root, "[environment]\n", compose=True)
    (task / "environment" / "Dockerfile").write_text('FROM python:3.12-slim\nENTRYPOINT ["/start.sh"]\n')
    (task / "environment" / "api").mkdir()
    (task / "environment" / "api" / "Dockerfile").write_text("FROM python:3.12-slim\n")
    (task / "environment" / "docker-compose.yaml").write_text(
        "services:\n  db:\n    image: postgres:16\n  api:\n    build: ./api\n"
    )
    return task


def test_compose_services_run_the_images_the_run_prepared(tmp_path):
    task = write_compose_task(tmp_path / "t")
    spec = prepared_compose_services(task, COMPOSE_ENVIRONMENT)
    assert spec is not None
    assert {service["name"]: service["image"] for service in spec["services"]} == {
        "db": SERVICE_DB.image_ref,
        "api": SERVICE_API.image_ref,
    }
    assert spec["main"]["entrypoint"] == ["/start.sh"]


def test_compose_services_are_derived_before_the_image_is_bound(tmp_path):
    """Binding sets docker_image, and the parser then ignores main's Dockerfile: its ENTRYPOINT would be lost."""
    task = write_compose_task(tmp_path / "t")
    bind_task_image(task, AGENT.image_ref)
    spec = prepared_compose_services(task, COMPOSE_ENVIRONMENT)
    assert spec is not None and "main" not in spec


def test_a_compose_tasks_env_values_stay_out_of_the_config_harbor_persists(task_spec, tmp_path, monkeypatch):
    """Harbor writes ``environment.kwargs`` to ``config.json`` and every trial's ``result.json``; the values are the task's own."""
    task = write_compose_task(tmp_path / "t")
    (task / "environment" / "docker-compose.yaml").write_text(
        "services:\n"
        "  db:\n    image: postgres:16\n    environment:\n      POSTGRES_PASSWORD: pg-pass\n      PGDATA: /data\n"
        "  api:\n    build: ./api\n    environment:\n      VERIFIER_TOKEN: api-tok-value\n"
        "  main:\n    environment:\n      MYSQL_PASSWORD: my-pass\n"
    )
    ctx = MagicMock(spec=JobContext)
    ctx.job_id = "run-0123456789abcdef-task-1"
    runtime = HarborRuntimeConfig(jobs_dir=tmp_path)
    raw = prepared_compose_services(task, COMPOSE_ENVIRONMENT)
    assert raw is not None
    # The same spec, with its values in it, is what the credential guard refuses.
    with pytest.raises(ValidationError, match="look like plaintext credentials"):
        configure_runtime(runtime, ctx, task_spec, raw)

    monkeypatch.setattr(os, "environ", {})
    job = HarborTaskAgentEvalJob()
    job._task_spec = task_spec.model_copy(update={"environment": COMPOSE_ENVIRONMENT})
    job._prepared_tasks([MagicMock(spec=AgentEvalTask, metadata={HARBOR_TASK_DIR_KEY: str(task)})])

    configured = configure_runtime(runtime, ctx, job._task_spec, job._compose_services)

    assert configured.environment is not None
    persisted = json.dumps(configured.environment.kwargs)
    values = ("pg-pass", "/data", "api-tok-value", "my-pass")
    assert not [value for value in values if value in persisted]
    assert persisted.count("${NEMO_COMPOSE_ENV_") == len(values)
    assert sorted(os.environ.values()) == sorted(values)


def test_a_recorded_main_image_leaves_its_entrypoint_to_the_sandbox(tmp_path):
    """With Builder's record of main's image, the Dockerfile isn't guessed from: the sandbox restores the real one."""
    task = write_compose_task(tmp_path / "t")
    agent = AGENT.model_copy(update={"runtime_config": ImageRuntimeConfig(entrypoint=["/bin/bash", "-l", "-c"])})
    spec = prepared_compose_services(task, COMPOSE_ENVIRONMENT.model_copy(update={"agent": agent}))
    assert spec is not None and "main" not in spec


def write_main_only_task(root: Path) -> Path:
    """A task whose compose file only changes how ``main`` starts, as swe-lancer's do."""
    task = write_task(root, "[environment]\n", compose=True)
    (task / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\n")
    (task / "environment" / "docker-compose.yaml").write_text(
        "services:\n  main:\n    command: [/app/tests/run.sh]\n    working_dir: /app\n"
    )
    return task


def test_a_main_only_task_runs_main_s_command_after_its_recorded_entrypoint(tmp_path):
    agent = AGENT.model_copy(update={"runtime_config": ImageRuntimeConfig(entrypoint=["/bin/bash", "-l", "-c"])})
    spec = prepared_compose_services(
        write_main_only_task(tmp_path / "t"), ENVIRONMENT.model_copy(update={"agent": agent})
    )
    assert spec is not None
    # The sandbox restores the ENTRYPOINT from image_entrypoints, so the spec carries only the compose settings.
    assert spec["main"] == {"command": ["/app/tests/run.sh"], "working_dir": "/app"}
    assert spec.get("services", []) == []


def test_a_main_command_whose_entrypoint_is_unknown_is_refused(tmp_path):
    """Without Builder's record, main's command would run without the ENTRYPOINT its image may inherit."""
    with pytest.raises(ComposeError, match="compose_entrypoint_unknown"):
        prepared_compose_services(write_main_only_task(tmp_path / "t"), ENVIRONMENT)


def test_a_single_container_task_has_no_compose_services(tmp_path):
    task = write_task(tmp_path / "t", "[environment]\n")
    assert prepared_compose_services(task, ENVIRONMENT) is None


def test_a_built_service_without_a_prepared_image_is_refused(tmp_path):
    task = write_compose_task(tmp_path / "t")
    only_db = ENVIRONMENT.model_copy(update={"compose_images": {"db": SERVICE_DB}})
    with pytest.raises(ComposeError, match="compose_build_not_prepared"):
        prepared_compose_services(task, only_db)


def test_prepared_images_must_be_for_the_tasks_services(tmp_path):
    task = write_compose_task(tmp_path / "t")
    extra = COMPOSE_ENVIRONMENT.model_copy(
        update={"compose_images": {**COMPOSE_ENVIRONMENT.compose_images, "ghost": SERVICE_DB}}
    )
    with pytest.raises(ValueError, match="not the services the run prepared images for"):
        prepared_compose_services(task, extra)


def test_a_compose_file_the_pod_cannot_run_fails_the_job_with_the_parsers_message(tmp_path):
    task = write_task(tmp_path / "t", "[environment]\n")
    (task / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\n")
    (task / "environment" / "docker-compose.yaml").write_text(
        "services:\n  odoo:\n    image: odoo:17\n    expose: ['18080']\n"
    )
    with pytest.raises(ComposeError, match="compose_reserved_port"):
        prepared_compose_services(task, COMPOSE_ENVIRONMENT)


def test_runtime_passes_compose_services_to_the_environment(task_spec, tmp_path):
    ctx = MagicMock(spec=JobContext)
    ctx.job_id = "run-0123456789abcdef-task-1"
    services = {"services": [{"name": "db", "image": SERVICE_DB.image_ref}]}
    with_services = configure_runtime(HarborRuntimeConfig(jobs_dir=tmp_path), ctx, task_spec, services)
    without = configure_runtime(HarborRuntimeConfig(jobs_dir=tmp_path), ctx, task_spec)
    assert with_services.environment is not None and without.environment is not None
    assert with_services.environment.kwargs["compose_services"] == services
    assert "compose_services" not in without.environment.kwargs
