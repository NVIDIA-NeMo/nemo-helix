# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A task Job's input, its compiled Job, and what the dispatcher changes inside it."""

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from nemo_evals.filesets import FilesetRef
from nemo_evals.jobs.agent_evaluate import AsyncAgentEvalJob
from nemo_evals.jobs.agent_spec import AgentEvalSpec, RegisteredAgentSource
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
    assert kwargs["metadata"] == {"nemo-run": "run-0123456789abcdef", "nemo-task-index": "1", "nemo-job": ctx.job_id}
    assert configured.n_concurrent_trials == 1
    assert "private-worker-value" not in json.dumps(configured.model_dump(mode="json"))


def test_runtime_passes_the_frozen_agent_channel(task_spec, tmp_path):
    policy = task_spec.sandbox_policy.model_copy(update={"agent_channel": ("igw.svc.cluster.local",)})
    spec = task_spec.model_copy(update={"sandbox_policy": policy})
    ctx = MagicMock(spec=JobContext)
    ctx.job_id = "run-0123456789abcdef-task-1"
    configured = configure_runtime(HarborRuntimeConfig(jobs_dir=tmp_path), ctx, spec)
    assert configured.environment is not None
    assert configured.environment.kwargs["agent_channel_hosts"] == ["igw.svc.cluster.local"]


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


def test_bind_rejects_compose_tasks(tmp_path):
    task = write_task(tmp_path / "t", "[environment]\n", compose=True)
    with pytest.raises(ValueError, match="compose tasks"):
        bind_task_image(task, AGENT.image_ref)
