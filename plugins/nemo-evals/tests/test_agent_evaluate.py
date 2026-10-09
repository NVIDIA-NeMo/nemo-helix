# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the agent-evaluation job."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from nemo_evals.api.schemas import AgentRef, MetadataItem, MetricInline, TaskInputs, TasksetRef
from nemo_evals.api.task_definitions.evaluator import ResolvedEvaluatorTaskDefinition
from nemo_evals.cli import EvaluatorPluginCLI
from nemo_evals.config import EvaluatorConfig
from nemo_evals.filesets import FilesetRef
from nemo_evals.jobs.agent_compiler import _environment
from nemo_evals.jobs.agent_evaluate import (
    AGENT_BUNDLE_DIR,
    DEFAULT_RESULT_NAME,
    SUMMARY_RESULT_NAME,
    AgentEvalJob,
    AsyncAgentEvalJob,
)
from nemo_evals.jobs.agent_spec import (
    AgentEvalInputSpec,
    AgentEvalSpec,
    AgentEvalTaskInput,
    AgentTarget,
    FabricConfigSource,
    FabricRunnerTarget,
    GymAgentSource,
    GymRunnerTarget,
    HarborBuiltinAgentSource,
    HarborImportedAgentSource,
    HarborRunnerTarget,
    ModelTarget,
    RegisteredAgentSource,
    ResolvedTask,
    Target,
)
from nemo_evals.jobs.gym_sandbox import (
    GYM_SANDBOX_PLAN_ENVVAR,
    SandboxPlan,
    SandboxUnavailableError,
    SessionBackedGymRunner,
)
from nemo_evals.jobs.gym_submission import resolve_gym_environment
from nemo_evals.jobs.kinds.evaluator import _to_runtime_task
from nemo_evals.jobs.publication import PublicationOutcome
from nemo_evals.jobs.publication_spec import IntakePublicationSpec, PublicationSpec
from nemo_evals.jobs.run_outcome import STATUS_DETAILS_KEY
from nemo_evals.jobs.secret_env import JobEnvSecretSource
from nemo_evals.metric_refs import MetricRef
from nemo_evals.shared.metric_bundles.bundles import MetricBundle, bundle_metric
from nemo_evals.shared.metric_bundles.hybrid import HybridMetricBundlePackager
from nemo_evals.tasks.agent_evaluate import main as agent_eval_task_main
from nemo_evals.tasks.runner import SDK_INITIALIZATION_EXIT_CODE
from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.client.errors import InternalServerError, NemoResponseValidationError, NemoTransportError
from nemo_helix_plugin.commands import add_job_commands
from nemo_helix_plugin.files.types import FilesetPurpose
from nemo_helix_plugin.intake.client import AsyncIntakeClient
from nemo_helix_plugin.job_context import JobContext, StoragePaths
from nemo_helix_plugin.job_results import LocalJobResults
from nemo_helix_plugin.jobs.constants import PERSISTENT_JOB_STORAGE_PATH_ENVVAR
from nemo_helix_plugin.jobs.exceptions import HelixJobCompilationError, HelixJobDependencyUnavailableError
from nemo_helix_plugin.jobs.execution_profiles import (
    DockerJobExecutionProfile,
    DockerJobExecutionProfileConfig,
    KubernetesJobExecutionProfile,
    KubernetesJobExecutionProfileConfig,
    KubernetesJobStorageConfig,
    SubprocessJobExecutionProfile,
    VolcanoJobExecutionProfile,
    VolcanoJobExecutionProfileConfig,
)
from nemo_helix_plugin.jobs.providers import SubprocessExecutionProvider
from nemo_helix_plugin.jobs.schemas import HelixJobStatus
from nemo_helix_plugin.jobs.spec import BaseExecutionProfile, HelixJobSpec
from nhx_evals_sdk.agent_eval.results import AgentEvalResult, AgentEvalSummary
from nhx_evals_sdk.agent_eval.runtimes.fabric.runtime import FabricAgentRuntime
from nhx_evals_sdk.agent_eval.runtimes.gym import GymAgentTaskRunner
from nhx_evals_sdk.agent_eval.runtimes.gym.records import NG_ROLLOUT_INDEX, NG_TASK_INDEX
from nhx_evals_sdk.agent_eval.runtimes.gym.sandboxed import (
    MODEL_CALLS_RESULT_KEY,
    SandboxedGymAgentTaskRunner,
    SandboxedGymRuntimeConfig,
)
from nhx_evals_sdk.agent_eval.runtimes.harbor.env import harbor_env_templates
from nhx_evals_sdk.agent_eval.runtimes.harbor.runtime import HarborAgentTaskRunner
from nhx_evals_sdk.agent_eval.scores import AgentEvalScoreStatus, AgentEvalTaskScore
from nhx_evals_sdk.agent_eval.tasks import AgentEvalRunConfig, AgentEvalTask
from nhx_evals_sdk.agent_eval.trials import (
    AgentEvalTarget,
    AgentEvalTrial,
    AgentEvalTrialStatus,
    AgentOutput,
    TrialError,
    TrialMeasurements,
)
from nhx_evals_sdk.enums import AgentFormat
from nhx_evals_sdk.execution.metric_execution import run_sync
from nhx_evals_sdk.metrics.exact_match import ExactMatchMetric
from nhx_evals_sdk.values import Agent, GenericAgent, Model, RunConfigOnline, RunConfigOnlineModel, SecretRef
from nhx_evals_sdk.values.evidence import CandidateEvidence, EvidenceDescriptor
from nhx_evals_sdk.values.protocol import MetricOutput
from pydantic import JsonValue, ValidationError
from pytest_mock import MockerFixture
from typer.testing import CliRunner


def _inline_metric() -> MetricInline:
    bundle = bundle_metric(
        ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.model_output}}"),
        HybridMetricBundlePackager(),
    )
    return MetricInline.model_validate(bundle.model_dump(mode="json"))


def _task_inputs(**values: Any) -> TaskInputs:
    return TaskInputs.model_validate(values)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _task_spec() -> ResolvedTask:
    return ResolvedTask(
        id="task-1",
        spec=ResolvedEvaluatorTaskDefinition(
            kind="evaluator",
            intent="Answer the question.",
            inputs=_task_inputs(instruction="What is 2+2?"),
            metrics=[_inline_metric()],
        ),
    )


def _runner_target(model: str | None = None) -> FabricRunnerTarget:
    """A minimal agent-runner target, for tests about runners in general rather than a specific one."""
    return FabricRunnerTarget(
        source=FabricConfigSource(
            config={"metadata": {"name": "a"}, "harness": {"adapter_id": "nvidia.fabric.codex"}}, model=model
        )
    )


def _job_context(tmp_path: Path) -> JobContext:
    storage = StoragePaths(ephemeral=tmp_path / "ephemeral", persistent=tmp_path / "persistent")
    storage.ephemeral.mkdir()
    storage.persistent.mkdir()
    return JobContext(
        workspace="dev",
        storage=storage,
        results=LocalJobResults(root=storage.persistent / "results"),
    )


def test_cli_agent_evaluate_uses_flat_submit_without_local_run() -> None:
    app = EvaluatorPluginCLI().get_cli()
    add_job_commands(app, {"evals.agent-evaluate": AgentEvalJob}, cli=EvaluatorPluginCLI())

    result = CliRunner().invoke(app, ["agent-evaluate", "--help"])

    assert result.exit_code == 0
    output = result.output
    assert "--spec" in output
    assert "--base-url" not in output
    assert "--profile" in output
    assert "Run locally, in-process." not in result.output
    assert "explain" in output

    run_result = CliRunner().invoke(app, ["agent-evaluate", "run"])
    assert run_result.exit_code != 0


class _FakeEvaluator:
    """Stand-in for AgentEvaluator: records the tasks it was handed and returns canned trials.

    With ``failed=True`` every trial fails to generate and every score fails with it, the shape a dead
    agent endpoint produces under ``ignore_request_failure``.
    """

    def __init__(self, *, failed: bool = False) -> None:
        self.failed = failed
        self.received_tasks: list[AgentEvalTask] = []
        self.received_trials: list[AgentEvalTrial] | None = None
        self.received_target: AgentEvalTarget | None = None
        self.received_config: AgentEvalRunConfig | None = None

    def run_sync(
        self,
        *,
        tasks: Sequence[AgentEvalTask],
        trials: Sequence[AgentEvalTrial] | None = None,
        target: AgentEvalTarget | None = None,
        config: AgentEvalRunConfig | None = None,
    ) -> AgentEvalResult:
        self.received_tasks = list(tasks)
        self.received_trials = list(trials) if trials is not None else None
        self.received_target = target
        self.received_config = config
        generated_trials = [
            AgentEvalTrial(
                id=f"{task.id}:trial",
                task_id=task.id,
                status=AgentEvalTrialStatus.FAILED if self.failed else AgentEvalTrialStatus.COMPLETED,
                output=None if self.failed else AgentOutput(output_text="4"),
                error=TrialError(type="ConnectError", message="SSL: WRONG_VERSION_NUMBER") if self.failed else None,
            )
            for task in tasks
        ]
        scores = [
            AgentEvalTaskScore(
                id=f"{trial.id}:exact_match",
                run_id="run-1",
                task_id=trial.task_id,
                trial_id=trial.id,
                metric_type="exact_match",
                status=AgentEvalScoreStatus.FAILED if self.failed else AgentEvalScoreStatus.COMPLETED,
                outputs=[] if self.failed else [MetricOutput(name="score", value=1.0)],
            )
            for trial in generated_trials
        ]
        return AgentEvalResult(
            run_id="run-1", tasks=list(tasks), trials=generated_trials, scores=scores, summary=AgentEvalSummary()
        )


def test_to_runtime_task_reconstructs_runtime_metric_instances() -> None:
    task = _to_runtime_task(_task_spec())
    assert isinstance(task, AgentEvalTask)
    assert task.id == "task-1"
    assert len(task.metrics) == 1
    assert isinstance(task.metrics[0], ExactMatchMetric)


async def test_reference_round_trips_from_input_spec_to_runtime_task() -> None:
    # Grader-only ``reference`` must survive the wire DTO -> canonical spec -> runtime task path so
    # metrics can grade against held-out ground truth (never seeded into the agent workspace).
    reference = {"test_calculator.py": "def test_add(): assert add(2, 3) == 5"}
    input_spec = AgentEvalInputSpec(
        target=_runner_target(),
        tasks=[
            AgentEvalTaskInput(
                id="fix-bug",
                intent="Fix the bug.",
                inputs=_task_inputs(instruction="Fix calculator.py."),
                reference=reference,
                metrics=[_inline_metric()],
            )
        ],
    )

    spec = await AgentEvalJob.to_spec(
        input_spec, workspace="dev", entity_client=None, async_sdk=_async_sdk(), is_local=True
    )
    assert isinstance(spec, AgentEvalSpec)
    assert all(task.spec.kind == "evaluator" for task in spec.tasks)
    assert spec.tasks[0].spec.kind == "evaluator"
    assert spec.tasks[0].spec.reference == reference
    assert _to_runtime_task(spec.tasks[0]).reference == reference


async def test_arbitrary_inputs_round_trip_from_input_spec_to_runtime_task() -> None:
    gym_row = {"input": [{"role": "user", "content": "Choose A"}], "temperature": 0.2}
    gym_row_extras = {"expected": "A", "scores": [1.0, None], "verified": True}
    input_spec = AgentEvalInputSpec(
        tasks=[
            AgentEvalTaskInput(
                id="gym-task",
                intent="Run a Gym task.",
                inputs=TaskInputs.model_validate({"gym_row": gym_row}),
                metrics=[_inline_metric()],
                metadata=[MetadataItem(key="gym_row_extras", value=gym_row_extras)],
            )
        ],
        target=GymRunnerTarget(
            source=GymAgentSource(
                component="simple_agent", config="responses_api_agents/simple_agent/configs/simple_agent.yaml"
            ),
            resources_server="mcqa",
        ),
    )

    spec = await AgentEvalJob.to_spec(
        input_spec, workspace="dev", entity_client=None, async_sdk=_async_sdk(), is_local=True
    )

    assert isinstance(spec, AgentEvalSpec)
    assert all(task.spec.kind == "evaluator" for task in spec.tasks)
    assert spec.tasks[0].spec.kind == "evaluator"
    assert spec.tasks[0].spec.inputs.model_dump(exclude_none=True)["gym_row"] == gym_row
    runtime_task = _to_runtime_task(spec.tasks[0])
    assert runtime_task.inputs["gym_row"] == gym_row
    assert runtime_task.metadata["gym_row_extras"] == gym_row_extras


def test_agent_eval_job_reconstructs_tasks_and_persists_bundle(tmp_path: Path, mocker: MockerFixture) -> None:
    fake = _FakeEvaluator()
    mocker.patch.object(AgentEvalJob, "_build_evaluator", return_value=fake)
    ctx = _job_context(tmp_path)

    spec = AgentEvalSpec(tasks=[_task_spec()], target=_runner_target("openai/gpt-5.4"))
    result = AgentEvalJob().run(spec.model_dump(), ctx=ctx, client=_sync_sdk_with_identity())

    # The job reconstructed runtime tasks (bundled metric round-tripped) before handing off.
    assert [task.id for task in fake.received_tasks] == ["task-1"]
    assert isinstance(fake.received_tasks[0].metrics[0], ExactMatchMetric)

    # The run bundle is persisted under job storage and registered as artifacts.
    assert result["status"] == "completed"
    bundle = ctx.storage.persistent / AGENT_BUNDLE_DIR
    assert (bundle / "trials.jsonl").exists()
    assert (bundle / "scores.jsonl").exists()
    assert (bundle / "summary.json").exists()
    assert (ctx.storage.persistent / "results" / DEFAULT_RESULT_NAME).exists()
    assert (ctx.storage.persistent / "results" / SUMMARY_RESULT_NAME).exists()
    assert result["artifact"]["name"] == DEFAULT_RESULT_NAME


def _run_sandboxed_gym_job(ctx: JobContext, mocker: MockerFixture) -> Path:
    """Run a one-task sandboxed Gym job with the host call faked; return the downloaded artifact."""

    async def host(_runner: SandboxedGymAgentTaskRunner, examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                NG_TASK_INDEX: example[NG_TASK_INDEX],
                NG_ROLLOUT_INDEX: example[NG_ROLLOUT_INDEX],
                "reward": 1.0,
                MODEL_CALLS_RESULT_KEY: [{"model_call_id": "c0", "started_at": 1788534870.5}],
            }
            for example in examples
        ]

    mocker.patch.object(SandboxedGymAgentTaskRunner, "_collect", autospec=True, side_effect=host)
    runner = SandboxedGymAgentTaskRunner(config=SandboxedGymRuntimeConfig(rollout_url="http://gym-host.example/run"))
    mocker.patch.object(AgentEvalJob, "_resolve_target", return_value=(runner, None, None))
    task = ResolvedTask(
        id="task-1",
        spec=ResolvedEvaluatorTaskDefinition(
            kind="evaluator",
            intent="Answer the question.",
            inputs=_task_inputs(gym_row={"input": "What is 2+2?"}),
            metrics=[_inline_metric()],
        ),
        metadata=[MetadataItem(key="gym_row_extras", value={})],
    )
    spec = AgentEvalSpec(
        tasks=[task],
        target=GymRunnerTarget(
            source=GymAgentSource(component="simple_agent", config="simple_agent.yaml"), resources_server="mcqa"
        ),
    )
    AgentEvalJob().run(spec.model_dump(), ctx=ctx, client=_sync_sdk_with_identity())
    return ctx.storage.persistent / "results" / DEFAULT_RESULT_NAME


def test_agent_eval_job_keeps_sandboxed_gym_evidence_inside_the_downloadable_bundle(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    """Gym rollouts and model-call captures must ship in the artifact with bundle-relative refs.

    Anywhere else, they die with the Job's container and the trials reference paths nobody can open.
    """
    downloaded = _run_sandboxed_gym_job(_job_context(tmp_path), mocker)

    [trial] = [json.loads(line) for line in (downloaded / "trials.jsonl").read_text(encoding="utf-8").splitlines()]
    refs = {
        name: descriptor["ref"]
        for name, descriptor in trial["evidence"]["descriptors"].items()
        if descriptor.get("ref")
    }
    assert set(refs) == {"rollouts", "ng_trajectory"}
    for ref in refs.values():
        assert not Path(ref).is_absolute()
        assert (downloaded / ref).is_file()
    capture = (downloaded / refs["ng_trajectory"]).read_text(encoding="utf-8")
    assert json.loads(capture)["model_call_id"] == "c0"


def _fabric_evidence_path(runtime: FabricAgentRuntime, task: AgentEvalTask, config: AgentEvalRunConfig) -> Path:
    return runtime._evidence_dir(0, task, config) / "fabric_result.json"


def _harbor_evidence_path(runtime: HarborAgentTaskRunner, task: AgentEvalTask, config: AgentEvalRunConfig) -> Path:
    assert runtime._config is not None and runtime._config.jobs_dir is not None
    return runtime._config.jobs_dir / "job" / task.id / "result.json"


@pytest.mark.parametrize(
    ("runtime_cls", "target", "evidence_path"),
    [
        (FabricAgentRuntime, _runner_target("openai/gpt-5.4"), _fabric_evidence_path),
        (HarborAgentTaskRunner, HarborRunnerTarget(), _harbor_evidence_path),
    ],
    ids=["fabric", "harbor"],
)
def test_agent_eval_job_keeps_runner_evidence_inside_the_downloadable_bundle(
    tmp_path: Path,
    mocker: MockerFixture,
    runtime_cls: type,
    target: FabricRunnerTarget | HarborRunnerTarget,
    evidence_path: Callable[[Any, AgentEvalTask, AgentEvalRunConfig], Path],
) -> None:
    """Each runner writes evidence where the job tells it to; that place must be inside the bundle.

    Anywhere else, the evidence dies with the Job's container and the trial references a path nobody can open.
    """

    async def run_tasks(
        runtime: Any, tasks: Sequence[AgentEvalTask], config: AgentEvalRunConfig | None = None
    ) -> list[AgentEvalTrial]:
        assert config is not None
        [task] = tasks
        path = evidence_path(runtime, task, config)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        return [
            AgentEvalTrial(
                id=f"{task.id}-trial",
                task_id=task.id,
                status=AgentEvalTrialStatus.COMPLETED,
                output=AgentOutput(output_text="4"),
                evidence=CandidateEvidence(descriptors={"result": EvidenceDescriptor(kind="log", ref=str(path))}),
            )
        ]

    mocker.patch.object(runtime_cls, "run_tasks", autospec=True, side_effect=run_tasks)
    ctx = _job_context(tmp_path)
    spec = AgentEvalSpec(tasks=[_task_spec()], target=target)

    AgentEvalJob().run(spec.model_dump(), ctx=ctx, client=_sync_sdk_with_identity())

    downloaded = ctx.storage.persistent / "results" / DEFAULT_RESULT_NAME
    [trial] = [json.loads(line) for line in (downloaded / "trials.jsonl").read_text(encoding="utf-8").splitlines()]
    ref = trial["evidence"]["descriptors"]["result"]["ref"]
    assert not Path(ref).is_absolute()
    assert (downloaded / ref).is_file()


def test_agent_eval_job_retry_replaces_a_failed_attempts_bundle(tmp_path: Path, mocker: MockerFixture) -> None:
    """A retried job (Volcano ``maxRetry``) reuses the failed attempt's persistent storage; leftover Gym output
    must not block or leak.

    Gym refuses to collect into a directory that holds rollouts, and leftover files would upload with
    the new attempt's artifact.
    """
    ctx = _job_context(tmp_path)
    leftover = ctx.storage.persistent / AGENT_BUNDLE_DIR
    (leftover / "gym_run" / "model_calls").mkdir(parents=True)
    (leftover / "gym_run" / "rollouts.jsonl").write_text('{"reward": 0.0}\n', encoding="utf-8")
    (leftover / "gym_run" / "model_calls" / "stale.capture.jsonl").write_text("{}\n", encoding="utf-8")

    downloaded = _run_sandboxed_gym_job(ctx, mocker)

    assert not (downloaded / "gym_run" / "model_calls" / "stale.capture.jsonl").exists()
    rollouts = [json.loads(line) for line in (downloaded / "gym_run" / "rollouts.jsonl").read_text().splitlines()]
    assert [rollout["reward"] for rollout in rollouts] == [1.0]


def test_agent_eval_job_survives_result_persistence_failure(tmp_path: Path, mocker: MockerFixture) -> None:
    # The queryable result record is a best-effort convenience index; the authoritative output (bundle
    # + summary artifacts) is already saved. A persistence failure must not fail a successful eval.
    mocker.patch.object(AgentEvalJob, "_build_evaluator", return_value=_FakeEvaluator())
    persist = mocker.patch(
        "nemo_evals.jobs.agent_evaluate.persist_agent_eval_result",
        side_effect=RuntimeError("entity store unavailable"),
    )
    ctx = _job_context(tmp_path)

    spec = AgentEvalSpec(tasks=[_task_spec()], target=_runner_target("openai/gpt-5.4"))
    result = AgentEvalJob().run(spec.model_dump(), ctx=ctx, client=_sync_sdk_with_identity())

    # Persistence was attempted and raised, yet the job still completed with its artifacts intact.
    persist.assert_called_once()
    async_client = persist.call_args.kwargs["async_client"]
    assert isinstance(async_client, AsyncNemoClient)
    assert async_client.default_headers == _SDK_IDENTITY_HEADERS
    assert result["status"] == "completed"
    assert result["artifact"]["name"] == DEFAULT_RESULT_NAME
    assert (ctx.storage.persistent / "results" / DEFAULT_RESULT_NAME).exists()


def test_agent_eval_job_passes_async_client_to_publication(tmp_path: Path, mocker: MockerFixture) -> None:
    mocker.patch.object(AgentEvalJob, "_build_evaluator", return_value=_FakeEvaluator())
    mocker.patch("nemo_evals.jobs.agent_evaluate.persist_agent_eval_result")
    publish = mocker.patch(
        "nemo_evals.jobs.agent_evaluate.publish_agent_eval_result",
        return_value=PublicationOutcome(status=HelixJobStatus.COMPLETED, evaluation_id="eval-a"),
    )
    ctx = _job_context(tmp_path)
    spec = AgentEvalSpec(
        tasks=[_task_spec()],
        target=_runner_target("openai/gpt-5.4"),
        publication=PublicationSpec(intake=IntakePublicationSpec(evaluation_id="eval-a", agent_name="agent-a")),
    )

    result = AgentEvalJob().run(spec.model_dump(), ctx=ctx, client=_sync_sdk_with_identity())

    assert result["status"] == "completed"
    intake = publish.call_args.kwargs["intake"]
    assert isinstance(intake, AsyncIntakeClient)
    assert intake.default_headers == _SDK_IDENTITY_HEADERS


def test_agent_eval_job_reports_the_run_outcome_on_success(tmp_path: Path, mocker: MockerFixture) -> None:
    mocker.patch.object(AgentEvalJob, "_build_evaluator", return_value=_FakeEvaluator())
    report = mocker.patch("nemo_evals.jobs.agent_evaluate.report_run_outcome")

    spec = AgentEvalSpec(tasks=[_task_spec()], target=_runner_target("openai/gpt-5.4"))
    result = AgentEvalJob().run(spec.model_dump(), ctx=_job_context(tmp_path), client=_sync_sdk_with_identity())

    assert result["status"] == "completed"
    assert result[STATUS_DETAILS_KEY] == {
        "unit": "trials",
        "total": 1,
        "errored": 0,
        "scored": 1,
        "failed": False,
        "message": "1 of 1 trials scored; 0 reported errors.",
    }
    assert report.call_args.args[0].scored == 1


def test_agent_eval_job_fails_when_no_trial_scored(tmp_path: Path, mocker: MockerFixture) -> None:
    """A run whose every agent call died must not end as a completed job with zero scores.

    Everything else still happens — artifacts (they hold the per-trial errors), the queryable record
    (Studio reaches the bundle through it), publication (Intake gets the failed traces) — and the job
    reports ``failed`` with the reason.
    """
    mocker.patch.object(AgentEvalJob, "_build_evaluator", return_value=_FakeEvaluator(failed=True))
    persist = mocker.patch("nemo_evals.jobs.agent_evaluate.persist_agent_eval_result")
    publish = mocker.patch(
        "nemo_evals.jobs.agent_evaluate.publish_agent_eval_result",
        return_value=PublicationOutcome(status=HelixJobStatus.COMPLETED, evaluation_id="eval-a"),
    )
    report = mocker.patch("nemo_evals.jobs.agent_evaluate.report_run_outcome")
    ctx = _job_context(tmp_path)
    spec = AgentEvalSpec(
        tasks=[_task_spec()],
        target=_runner_target("openai/gpt-5.4"),
        publication=PublicationSpec(intake=IntakePublicationSpec(evaluation_id="eval-a", agent_name="agent-a")),
    )

    result = AgentEvalJob().run(spec.model_dump(), ctx=ctx, client=_sync_sdk_with_identity())

    assert result["status"] == "failed"
    assert result["reason"].startswith("No usable scores across 1 trials (1 reported errors)")
    assert result[STATUS_DETAILS_KEY]["failed"] is True
    assert (ctx.storage.persistent / "results" / SUMMARY_RESULT_NAME).exists()
    persist.assert_called_once()
    publish.assert_called_once()
    assert report.call_args.args[0].failed


def test_agent_eval_spec_requires_at_least_one_task() -> None:
    with pytest.raises(ValueError, match="at least 1 item|too_short|min_length"):
        AgentEvalSpec(tasks=[])


def _agent() -> Agent:
    return GenericAgent(
        url="http://agent.test",
        name="test-agent",
        format=AgentFormat.GENERIC,
        body={"question": "{{item.prompt}}"},
        response_path="$.answer",
    )


def test_resolve_target_builds_fabric_runtime_from_runner_target(tmp_path: Path) -> None:
    ctx = _job_context(tmp_path)
    fabric_target = FabricRunnerTarget(
        source=FabricConfigSource(
            config={"metadata": {"name": "a"}, "harness": {"adapter_id": "nvidia.fabric.codex"}},
            model="openai/gpt-5.4",
        )
    )
    target, prompt_template, params = AgentEvalJob._resolve_target(fabric_target, ctx)
    assert isinstance(target, FabricAgentRuntime)
    assert target._model == "openai/gpt-5.4"
    # A runner shapes its own request, so it contributes no prompt template or inference params.
    assert prompt_template is None
    assert params is None


def test_resolve_target_builds_harbor_runtime_from_runner_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _job_context(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-resolved-by-the-service")
    harbor_target = HarborRunnerTarget(
        source=HarborBuiltinAgentSource(name="oracle", model_name="openai/gpt-5.4"),
        agent_kwargs={"fabric_adapter_id": "nvidia.fabric.codex", "fabric_harness_settings": {"max_turns": 3}},
        env_secrets={"OPENAI_API_KEY": SecretRef(root="my-workspace/openai-key")},
        env_vars={"FABRIC_LOG": "debug"},
        n_attempts=2,
        n_concurrent_trials=8,
        max_retries=1,
        reward_key="score",
    )
    target, prompt_template, params = AgentEvalJob._resolve_target(harbor_target, ctx)
    assert isinstance(target, HarborAgentTaskRunner)
    assert target._config is not None
    assert target._config.jobs_dir == ctx.storage.persistent / AGENT_BUNDLE_DIR / "evidence" / "harbor"
    # Spec knobs are forwarded onto the Harbor runtime config.
    assert target._config.agent_model_name == "openai/gpt-5.4"
    assert target._config.agent_kwargs == {
        "fabric_adapter_id": "nvidia.fabric.codex",
        "fabric_harness_settings": {"max_turns": 3},
    }
    assert target._config.env_secrets == {"OPENAI_API_KEY": SecretRef(root="my-workspace/openai-key")}
    assert target._config.env_vars == {"FABRIC_LOG": "debug"}
    # The service injected the secret under its key, so Harbor gets a `${OPENAI_API_KEY}` template.
    assert isinstance(target._secret_resolver, JobEnvSecretSource)
    assert harbor_env_templates(target._config.env_secrets, target._secret_resolver) == {
        "OPENAI_API_KEY": "${OPENAI_API_KEY}"
    }
    assert target._config.n_attempts == 2
    assert target._config.reward_key == "score"
    # A runner shapes its own request, so it contributes no prompt template or inference params.
    assert prompt_template is None
    assert params is None


def test_resolve_target_unpacks_model_target_request_config(tmp_path: Path) -> None:
    ctx = _job_context(tmp_path)
    model = Model(url="http://model.test/v1/chat/completions", name="m")
    target, prompt_template, params = AgentEvalJob._resolve_target(
        ModelTarget(model=model, prompt_template="{{item.prompt}}", params=RunConfigOnlineModel()), ctx
    )
    assert target is model
    assert prompt_template == "{{item.prompt}}"
    assert isinstance(params, RunConfigOnlineModel)


def test_resolve_target_agent_carries_no_prompt_template(tmp_path: Path) -> None:
    ctx = _job_context(tmp_path)
    agent = _agent()
    target, prompt_template, params = AgentEvalJob._resolve_target(AgentTarget(agent=agent), ctx)
    # The agent shapes its own request via body/response_path — no separate prompt template.
    assert target is agent
    assert prompt_template is None
    assert isinstance(params, RunConfigOnline)


def test_resolve_target_resolves_none_to_no_target(tmp_path: Path) -> None:
    ctx = _job_context(tmp_path)
    assert AgentEvalJob._resolve_target(None, ctx) == (None, None, None)


def test_resolve_target_builds_gym_runtime_from_runner_target(tmp_path: Path) -> None:
    ctx = _job_context(tmp_path)
    gym_target = GymRunnerTarget(
        source=GymAgentSource(
            component="simple_agent", config="responses_api_agents/simple_agent/configs/simple_agent.yaml"
        ),
        resources_server="mcqa",
        num_repeats=2,
        concurrency=4,
        reward_key="score",
        hydra_params={"model": {"temperature": 0.7}},
        env_secrets={"OPENAI_API_KEY": SecretRef("dev/openai-key")},
    )
    target, prompt_template, params = AgentEvalJob._resolve_target(gym_target, ctx)
    assert isinstance(target, GymAgentTaskRunner)
    assert target._config.agent == "simple_agent"
    assert target._config.resources_server == "mcqa"
    assert target._config.num_repeats == 2
    assert target._config.reward_key == "score"
    # Overrides are nested data on both sides of the seam — the spec model and the runtime config
    # must agree on the shape, or the spec validates and the runtime rejects it.
    assert target._config.hydra_params == {"model": {"temperature": 0.7}}
    assert target._config.env_secrets == {"OPENAI_API_KEY": SecretRef("dev/openai-key")}
    assert isinstance(target._secret_resolver, JobEnvSecretSource)
    # A runner shapes its own request, so it contributes no prompt template or inference params.
    assert prompt_template is None
    assert params is None


def test_colocated_gym_checks_injected_secret_before_start(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    gym_target = GymRunnerTarget(
        source=GymAgentSource(component="simple_agent", config="a.yaml"),
        resources_server="mcqa",
        env_secrets={"OPENAI_API_KEY": SecretRef("ws/openai")},
    )
    runner, _, _ = AgentEvalJob._resolve_target(gym_target, _job_context(tmp_path))
    assert isinstance(runner, GymAgentTaskRunner)
    with pytest.raises(ValueError, match="nemo secrets get openai --workspace ws"):
        asyncio.run(runner.run_tasks([]))


def _sandbox_plan() -> SandboxPlan:
    return SandboxPlan(
        host_provider="opensandbox",
        runtime_image="registry.example.com/nhx-gym-runtime:1.0",
        job_storage_pvc_claim="job-storage",
        environment_sub_path="environment",
        workspace_sub_path="workspace",
        episode_backend="memory",
        policy_base_urls=("https://integrate.api.nvidia.com/v1",),
    )


def test_resolve_target_passes_job_storage_to_sandboxed_gym(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _job_context(tmp_path)
    gym_target = GymRunnerTarget(
        source=GymAgentSource(
            component="simple_agent", config="responses_api_agents/simple_agent/configs/simple_agent.yaml"
        ),
        resources_server="custom",
        environment=FilesetRef(root="dev/custom-environment"),
    )
    monkeypatch.setenv(GYM_SANDBOX_PLAN_ENVVAR, _sandbox_plan().model_dump_json())

    target, prompt_template, params = AgentEvalJob._resolve_target(gym_target, ctx)

    assert isinstance(target, SessionBackedGymRunner)
    assert target._persistent_storage_path == ctx.storage.persistent
    assert target._workspace == ctx.workspace
    assert prompt_template is None
    assert params is None


def test_resolve_target_rejects_unsandboxed_custom_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _job_context(tmp_path)
    gym_target = GymRunnerTarget(
        source=GymAgentSource(
            component="simple_agent", config="responses_api_agents/simple_agent/configs/simple_agent.yaml"
        ),
        resources_server="custom",
        environment=FilesetRef(root="dev/custom-environment"),
    )
    monkeypatch.delenv(GYM_SANDBOX_PLAN_ENVVAR, raising=False)

    with pytest.raises(SandboxUnavailableError, match="require sandboxed execution"):
        AgentEvalJob._resolve_target(gym_target, ctx)


def test_resolve_target_rejects_unsandboxed_agent_ref_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Colocated execution builds a GymRuntimeConfig, which has no agent_ref_name, so an accepted one
    # would be dropped and the run would route to `agent` -- a plausible score for a different agent.
    ctx = _job_context(tmp_path)
    gym_target = GymRunnerTarget(
        source=GymAgentSource(
            component="simple_agent",
            config="responses_api_agents/simple_agent/configs/simple_agent.yaml",
            instance="mcqa_simple_agent",
        ),
        resources_server="mcqa",
    )
    monkeypatch.delenv(GYM_SANDBOX_PLAN_ENVVAR, raising=False)

    with pytest.raises(SandboxUnavailableError, match="agent_ref_name"):
        AgentEvalJob._resolve_target(gym_target, ctx)


def test_runner_target_is_accepted(tmp_path: Path) -> None:
    spec = AgentEvalSpec(tasks=[_task_spec()], target=_runner_target("openai/gpt-5.4"))
    assert isinstance(spec.target, FabricRunnerTarget)


def test_harbor_runner_target_is_accepted() -> None:
    spec = AgentEvalSpec(tasks=[_task_spec()], target=HarborRunnerTarget())
    assert isinstance(spec.target, HarborRunnerTarget)


@pytest.mark.parametrize(("ref", "workspace"), [("my-workspace/openai-key", "my-workspace"), ("openai-key", "dev")])
def test_harbor_env_secret_missing_from_job_environment_names_the_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ref: str, workspace: str
) -> None:
    """An unresolved `env_secrets` entry fails by name before Docker starts, not inside a trial as an auth error.

    The runner raises when it builds the templates, at the start of each execution; target resolution
    no longer checks. A bare ref points at the job's workspace.
    """
    ctx = _job_context(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    harbor_target = HarborRunnerTarget(env_secrets={"OPENAI_API_KEY": SecretRef(root=ref)})
    runner, _, _ = AgentEvalJob._resolve_target(harbor_target, ctx)
    assert isinstance(runner, HarborAgentTaskRunner) and runner._config is not None

    with pytest.raises(ValueError) as excinfo:
        harbor_env_templates(runner._config.env_secrets, runner._secret_resolver)
    assert str(excinfo.value) == (
        f"env_secrets['OPENAI_API_KEY'] -> secret {ref!r} was not injected into this job's environment. "
        f"Check the secret exists in workspace {workspace!r}: nemo secrets get openai-key --workspace {workspace}"
    )


def test_harbor_target_refuses_plaintext_credentials_in_agent_kwargs() -> None:
    """A submitted spec carrying a credential in ``agent_kwargs`` is refused at the API, not mid-job.

    The value would otherwise be stored on the spec and then copied by Harbor across the job dir.
    Validating on the target means the submitter is told at submit time, while they still have the
    key in hand to move it to `env_secrets`.
    """
    with pytest.raises(ValidationError, match="env_secrets"):
        HarborRunnerTarget(agent_kwargs={"fabric_environment_env": {"OPENAI_API_KEY": "nvapi-not-a-real-key"}})


def test_model_target_validation_error_does_not_echo_rejected_auth_header() -> None:
    """The target wrapper must preserve credential redaction when its nested Model rejects auth."""
    with pytest.raises(ValidationError, match="authentication headers") as excinfo:
        ModelTarget.model_validate(
            {
                "model": {
                    "url": "http://model.test",
                    "name": "test",
                    "default_headers": {"Authorization": "LEAKME"},
                }
            }
        )
    assert "LEAKME" not in str(excinfo.value)


def test_gym_target_validation_error_does_not_echo_rejected_secret() -> None:
    """Direct target validation must not print a credential rejected as an environment collision."""
    with pytest.raises(ValidationError, match="env_vars and env_secrets") as excinfo:
        GymRunnerTarget.model_validate(
            {
                "agent": "simple_agent",
                "agent_config": "config.yaml",
                "resources_server": "mcqa",
                "env_secrets": {"KEY": "ws/key"},
                "env_vars": {"KEY": "LEAKME"},
            }
        )
    assert "LEAKME" not in str(excinfo.value)


def test_harbor_agent_kwargs_round_trip_the_wire_unchanged() -> None:
    """Nested kwargs survive JSON serialization, so what the submitter wrote is what the agent's ``__init__`` gets."""
    agent_kwargs: dict[str, JsonValue] = {
        "fabric_adapter_id": "nvidia.fabric.codex",
        "fabric_package": "nemo-fabric[codex]==0.4.0",
        "fabric_harness_settings": {"max_turns": 3, "tools": ["shell", None], "strict": True},
    }
    spec = AgentEvalSpec(
        tasks=[_task_spec()],
        target=HarborRunnerTarget(
            source=HarborImportedAgentSource(import_path="nemo_fabric.integrations.harbor:FabricAgent"),
            agent_kwargs=agent_kwargs,
        ),
    )

    restored = AgentEvalSpec.model_validate(json.loads(spec.model_dump_json()))

    assert isinstance(restored.target, HarborRunnerTarget)
    assert restored.target.agent_kwargs == agent_kwargs
    assert HarborRunnerTarget().agent_kwargs == {}


def test_agent_eval_job_source_is_distinct_from_evaluate_job() -> None:
    """The two evaluator job types must not share a ``source`` (mixed-list-crash regression).

    A shared source makes the agent-evaluate and evaluate list endpoints return each other's jobs and
    500 rendering the foreign spec. ``service.py`` passes ``AGENT_EVAL_JOB_SOURCE`` as the agent-eval
    routes' ``service_name`` so it lands a distinct ``source`` from EvaluateJob's derived default.
    """
    from nemo_evals.jobs.evaluate import EvaluateJob
    from nemo_evals.service import AGENT_EVAL_JOB_SOURCE
    from nemo_helix_plugin.jobs.routes import _derive_service_name

    assert AGENT_EVAL_JOB_SOURCE != _derive_service_name(EvaluateJob)


def _async_sdk() -> AsyncNemoClient:
    return AsyncNemoClient(
        base_url="http://platform.test",
        workspace="default",
        http_client=AsyncMock(spec=httpx.AsyncClient),
    )


_SDK_IDENTITY_HEADERS = {
    "X-NHX-Principal-Id": "service:evaluator",
    "X-NHX-Actor-Account-Id": "account-service",
    "X-NHX-Actor-Aliases": "service:evaluator",
    "X-NHX-Principal-On-Behalf-Of": "user-1",
    "X-NHX-Principal-On-Behalf-Of-Email": "user@corp.test",  # PII - must stay in-platform
    "X-NHX-Internal": "true",
    "X-NHX-Subject-Account-Id": "account-user",
    "X-NHX-Subject-Aliases": "user-1,user@corp.test",
    "X-NHX-Scopes": "platform:read evaluator:write",
    "X-NHX-Trace-Id": "must-not-forward",  # non-identity X-NHX-* must be dropped
    "Authorization": "Bearer super-secret",  # bearer must never reach any endpoint
}
_FORWARDED_IDENTITY_HEADERS = {
    "X-NHX-Principal-Id": "service:evaluator",
    "X-NHX-Actor-Account-Id": "account-service",
    "X-NHX-Actor-Aliases": "service:evaluator",
    "X-NHX-Principal-On-Behalf-Of": "user-1",
    "X-NHX-Principal-On-Behalf-Of-Email": "user@corp.test",
    "X-NHX-Internal": "true",
    "X-NHX-Subject-Account-Id": "account-user",
    "X-NHX-Subject-Aliases": "user-1,user@corp.test",
    "X-NHX-Scopes": "platform:read evaluator:write",
}


def _sync_sdk_with_identity(base_url: str = "http://platform") -> NemoClient:
    return NemoClient(
        base_url=base_url,
        default_headers=_SDK_IDENTITY_HEADERS,
        http_client=MagicMock(spec=httpx.Client),
    )


def _async_sdk_with_identity(base_url: str = "http://platform") -> AsyncNemoClient:
    return AsyncNemoClient(
        base_url=base_url,
        default_headers=_SDK_IDENTITY_HEADERS,
        http_client=AsyncMock(spec=httpx.AsyncClient),
    )


def _model_target(url: str) -> ModelTarget:
    return ModelTarget(model=Model(url=url, name="m"))


@pytest.mark.parametrize("client_factory", [_sync_sdk_with_identity, _async_sdk_with_identity])
def test_build_evaluator_forwards_identity_headers_to_platform_routed_target(
    client_factory: Callable[[], NemoClient | AsyncNemoClient],
) -> None:
    """A platform-routed target (same host as the SDK base URL, e.g. an IGW route) must act as the
    job's principal, so identity headers are forwarded. Forwarding is an explicit allowlist: the
    service principal id and on-behalf-of go through, but transport noise and other ``X-NHX-*`` (e.g.
    trace) headers and the bearer do not."""
    client = client_factory()
    target = _model_target("http://platform/apis/inference-gateway/v2/workspaces/default/model/m/-/v1/chat/completions")

    evaluator = AgentEvalJob._build_evaluator(client, target)

    assert evaluator.default_headers == _FORWARDED_IDENTITY_HEADERS


@pytest.mark.parametrize("client_factory", [_sync_sdk_with_identity, _async_sdk_with_identity])
def test_build_evaluator_forwards_identity_headers_to_matching_default_port(
    client_factory: Callable[[str], NemoClient | AsyncNemoClient],
) -> None:
    client = client_factory("https://platform")
    target = _model_target(
        "https://platform:443/apis/inference-gateway/v2/workspaces/default/model/m/-/v1/chat/completions"
    )

    assert AgentEvalJob._build_evaluator(client, target).default_headers == _FORWARDED_IDENTITY_HEADERS


@pytest.mark.parametrize("client_factory", [_sync_sdk_with_identity, _async_sdk_with_identity])
def test_build_evaluator_sends_no_identity_when_platform_origin_scheme_differs(
    client_factory: Callable[[str], NemoClient | AsyncNemoClient],
) -> None:
    client = client_factory("https://platform")
    target = _model_target("http://platform/apis/inference-gateway/v2/workspaces/default/model/m/-/v1/chat/completions")

    assert AgentEvalJob._build_evaluator(client, target).default_headers is None


@pytest.mark.parametrize("client_factory", [_sync_sdk_with_identity, _async_sdk_with_identity])
def test_build_evaluator_sends_no_identity_to_third_party_target(
    client_factory: Callable[[], NemoClient | AsyncNemoClient],
) -> None:
    """A third-party target the user configured must receive no on-behalf-of identity — that would
    leak the delegated user's id/email/groups PII to an external host (it authenticates via its own
    api key anyway)."""
    sdk = client_factory()
    target = _model_target("https://api.openai.com/v1/chat/completions")

    assert AgentEvalJob._build_evaluator(sdk, target).default_headers is None


def test_build_evaluator_runner_target_forwards_no_headers() -> None:
    # A runner has no platform HTTP endpoint of its own, so there's no identity to forward.
    sdk = _sync_sdk_with_identity()
    assert AgentEvalJob._build_evaluator(sdk, _runner_target("openai/gpt-5.4")).default_headers is None


def test_build_evaluator_without_identity_headers_forwards_no_headers() -> None:
    sdk = NemoClient(
        base_url="http://platform",
        workspace="dev",
        http_client=MagicMock(spec=httpx.Client),
    )
    target = _model_target("http://platform/apis/inference-gateway/v2/workspaces/default/model/m/-/v1/chat/completions")
    assert AgentEvalJob._build_evaluator(sdk, target).default_headers is None


def _capture_evaluator_headers(mocker: MockerFixture) -> dict[str, dict[str, str] | None]:
    """Swap in a fake ``AgentEvaluator`` that records the headers ``_build_evaluator`` computed."""
    captured: dict[str, dict[str, str] | None] = {}

    def _factory(*, default_headers: dict[str, str] | None = None) -> _FakeEvaluator:
        captured["default_headers"] = default_headers
        return _FakeEvaluator()

    mocker.patch("nemo_evals.jobs.agent_evaluate.AgentEvaluator", _factory)
    return captured


def test_scheduler_passes_the_declared_sync_agent_eval_client(
    tmp_path: Path,
    mocker: MockerFixture,
) -> None:
    """The sync scheduler entrypoint follows the job class' single client contract."""
    captured = _capture_evaluator_headers(mocker)
    config = AgentEvalSpec(
        tasks=[_task_spec()],
        target=_model_target(
            "http://platform/apis/inference-gateway/v2/workspaces/default/model/m/-/v1/chat/completions"
        ),
    ).model_dump()
    ctx = _job_context(tmp_path)

    result = AgentEvalJob().run(
        config,
        ctx=ctx,
        client=_sync_sdk_with_identity(),
    )

    assert result["status"] == "completed"
    assert captured["default_headers"] == _FORWARDED_IDENTITY_HEADERS


def test_scheduler_passes_the_declared_async_agent_eval_client(
    tmp_path: Path,
    mocker: MockerFixture,
) -> None:
    """The async scheduler entrypoint follows the job class' single client contract."""
    captured = _capture_evaluator_headers(mocker)
    config = AgentEvalSpec(
        tasks=[_task_spec()],
        target=_model_target(
            "http://platform/apis/inference-gateway/v2/workspaces/default/model/m/-/v1/chat/completions"
        ),
    ).model_dump()
    ctx = _job_context(tmp_path)

    result = AsyncAgentEvalJob().run(
        config,
        ctx=ctx,
        async_client=_async_sdk_with_identity(),
    )

    assert result["status"] == "completed"
    assert captured["default_headers"] == _FORWARDED_IDENTITY_HEADERS


def test_run_sends_no_identity_to_a_third_party_target_via_typed_async_client(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    """The same-origin guard is what keeps the delegated user's email and groups inside the platform.
    Scheduler adaptation reconstructs the client the guard compares against, so a regression here
    would leak that PII to whatever endpoint the submitter named."""
    captured = _capture_evaluator_headers(mocker)
    spec = AgentEvalSpec(tasks=[_task_spec()], target=_model_target("https://api.openai.com/v1/chat/completions"))
    ctx = _job_context(tmp_path)

    result = AsyncAgentEvalJob().run(
        spec.model_dump(),
        ctx=ctx,
        async_client=_async_sdk_with_identity(),
    )

    assert result["status"] == "completed"
    assert captured["default_headers"] is None


def test_input_spec_accepts_stored_metric_reference() -> None:
    spec = AgentEvalInputSpec(
        tasks=[
            AgentEvalTaskInput(id="task-1", intent="Answer.", inputs=TaskInputs(), metrics=[MetricRef("stored-metric")])
        ],
        target=_runner_target("openai/gpt-5.4"),
    )
    assert isinstance(spec.tasks, list)
    assert isinstance(spec.tasks[0], AgentEvalTaskInput)
    assert isinstance(spec.tasks[0].metrics[0], MetricRef)


def test_input_spec_accepts_a_taskset_reference() -> None:
    spec = AgentEvalInputSpec(tasks=TasksetRef("default/geo-suite"), target=_runner_target("openai/gpt-5.4"))
    assert isinstance(spec.tasks, TasksetRef)
    assert spec.tasks.root == "default/geo-suite"
    # A JSON string round-trips back to the TasksetRef arm of the union, not a list.
    assert isinstance(AgentEvalInputSpec.model_validate_json(spec.model_dump_json()).tasks, TasksetRef)


def test_input_spec_rejects_empty_inline_task_list() -> None:
    with pytest.raises(ValueError, match="at least 1 item"):
        AgentEvalInputSpec(tasks=[], target=_runner_target("openai/gpt-5.4"))


async def test_to_spec_resolves_inline_task_metrics_without_metric_refs() -> None:
    # Inline metrics need no entity lookup or Files read; refs would, but none are used here.
    input_spec = AgentEvalInputSpec(
        tasks=[
            AgentEvalTaskInput(
                id="task-1",
                intent="Answer the question.",
                inputs=_task_inputs(instruction="What is 2+2?"),
                metrics=[_inline_metric()],
            )
        ],
        target=_runner_target("openai/gpt-5.4"),
    )

    spec = await AgentEvalJob.to_spec(
        input_spec, workspace="dev", entity_client=None, async_sdk=_async_sdk(), is_local=True
    )

    assert isinstance(spec, AgentEvalSpec)
    assert all(task.spec.kind == "evaluator" for task in spec.tasks)
    assert len(spec.tasks) == 1
    assert isinstance(spec.tasks[0].spec.metrics[0], MetricInline)
    # Canonical metrics reconstruct to runtime instances.
    assert isinstance(_to_runtime_task(spec.tasks[0]).metrics[0], ExactMatchMetric)


@pytest.mark.parametrize("task_id", ["", "   ", "\t\n"])
async def test_to_spec_rejects_a_blank_inline_task_id(task_id: str) -> None:
    """A blank id must fail at submit, before a job exists, rather than run as an unnamed task."""
    input_spec = AgentEvalInputSpec(
        tasks=[
            AgentEvalTaskInput(
                id=task_id, intent="Answer.", inputs=_task_inputs(instruction="Reply DONE."), metrics=[_inline_metric()]
            )
        ],
        target=_runner_target("openai/gpt-5.4"),
    )
    with pytest.raises(ValueError, match="task id must not be empty"):
        await AgentEvalJob.to_spec(
            input_spec, workspace="dev", entity_client=None, async_sdk=_async_sdk(), is_local=True
        )


async def test_to_spec_requires_entity_store_to_resolve_a_metric_reference() -> None:
    # A stored MetricRef can only be loaded with an entity store and Files service; without one,
    # to_spec must fail loudly rather than silently drop the metric.
    input_spec = AgentEvalInputSpec(
        tasks=[
            AgentEvalTaskInput(id="task-1", intent="Answer.", inputs=TaskInputs(), metrics=[MetricRef("stored-metric")])
        ],
        target=_runner_target("openai/gpt-5.4"),
    )
    with pytest.raises(ValueError, match="platform connection"):
        await AgentEvalJob.to_spec(
            input_spec, workspace="dev", entity_client=None, async_sdk=_async_sdk(), is_local=True
        )


# --- compile: the submit/service-side path ----------------------------------


def _assert_agent_eval_step_entrypoint(
    job_spec: HelixJobSpec,
    *,
    expected_image: str | None = None,
    expected_entrypoint: Sequence[str] = ("python", "-m"),
) -> None:
    step = job_spec.steps[0]
    container = cast(Any, step.executor).container
    if expected_image is not None:
        assert container.image == expected_image
    assert container.entrypoint == list(expected_entrypoint)
    assert container.command == ["nemo_evals.tasks.agent_evaluate"]


async def test_checked_fabric_spec_transforms_and_compiles() -> None:
    path = _repo_root() / "skills/nemo-evals-plugin/assets/specs/fabric_agent_eval.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    input_spec = AgentEvalInputSpec.model_validate(payload)

    spec = await AgentEvalJob.to_spec(
        input_spec,
        workspace="default",
        entity_client=None,
        async_sdk=_async_sdk(),
        is_local=False,
    )

    assert isinstance(spec, AgentEvalSpec)
    assert all(task.spec.kind == "evaluator" for task in spec.tasks)
    bundle = MetricBundle.model_validate(spec.tasks[0].spec.metrics[0].model_dump(mode="json"))
    assert bundle.payload.kind == "inline"

    compiled = await AgentEvalJob.compile(
        workspace="default",
        spec=spec,
        entity_client=None,
        job_name=None,
        async_sdk=_async_sdk(),
    )
    job_spec = HelixJobSpec.model_validate(compiled)
    _assert_agent_eval_step_entrypoint(job_spec)
    config = cast(dict[str, Any], job_spec.steps[0].config)
    assert config["target"]["kind"] == "fabric"
    assert config["tasks"][0]["spec"]["metrics"][0]["payload"]["kind"] == "inline"


def _patch_execution_profiles(mocker: MockerFixture, profiles: list[BaseExecutionProfile]) -> None:
    response = mocker.Mock()
    response.data.return_value = profiles
    jobs_client = mocker.Mock()
    jobs_client.get_execution_profiles = mocker.AsyncMock(return_value=response)
    mocker.patch("nemo_evals.jobs.agent_evaluate.client_from_platform", return_value=jobs_client)


def _kubernetes_profile_with_job_storage(pvc_name: str = "job-storage") -> KubernetesJobExecutionProfile:
    return KubernetesJobExecutionProfile(
        config=KubernetesJobExecutionProfileConfig(storage=KubernetesJobStorageConfig(pvc_name=pvc_name))
    )


async def _compile_harbor(*, async_sdk: AsyncNemoClient, profile: str | None = None) -> HelixJobSpec:
    """Compile the minimal Harbor submission every backend-guard test makes."""
    compiled = await AgentEvalJob.compile(
        workspace="default",
        spec=AgentEvalSpec(tasks=[_task_spec()], target=HarborRunnerTarget()),
        entity_client=object(),
        job_name=None,
        async_sdk=async_sdk,
        profile=profile,
    )
    return HelixJobSpec.model_validate(compiled)


@pytest.mark.parametrize(
    ("target", "expected_kind", "expected_endpoint_name", "expected_image_name", "expected_entrypoint"),
    [
        (
            FabricRunnerTarget(
                source=FabricConfigSource(
                    config={"metadata": {"name": "a"}, "harness": {"adapter_id": "nvidia.fabric.codex"}}
                )
            ),
            "fabric",
            None,
            "nhx-tasks",
            ("python", "-m"),
        ),
        (
            GymRunnerTarget(
                source=GymAgentSource(
                    component="simple_agent", config="responses_api_agents/simple_agent/configs/simple_agent.yaml"
                ),
                resources_server="mcqa",
            ),
            "gym",
            None,
            "nhx-gym-tasks",
            ("/app/.venv/bin/python", "-m"),
        ),
        (
            ModelTarget(
                model=Model(url="http://model.test/v1/chat/completions", name="test-model"),
                params=RunConfigOnlineModel(),
            ),
            "model",
            "test-model",
            "nhx-tasks",
            ("python", "-m"),
        ),
        (
            AgentTarget(agent=_agent(), params=RunConfigOnline()),
            "agent",
            "test-agent",
            "nhx-tasks",
            ("python", "-m"),
        ),
    ],
)
async def test_compile_produces_cpu_task_step_carrying_each_target(
    mocker: MockerFixture,
    target: Target,
    expected_kind: str,
    expected_endpoint_name: str | None,
    expected_image_name: str,
    expected_entrypoint: Sequence[str],
) -> None:
    _patch_execution_profiles(mocker, [])
    mocker.patch("nemo_evals.jobs.agent_compiler.config.gym_tasks_image", None)
    mocker.patch(
        "nemo_evals.jobs.agent_compiler.get_qualified_image",
        side_effect=lambda name: f"registry.example/{name}:test",
    )
    spec = AgentEvalSpec(tasks=[_task_spec()], target=target)

    compiled = await AgentEvalJob.compile(
        workspace="default", spec=spec, entity_client=object(), job_name=None, async_sdk=_async_sdk()
    )

    job_spec = HelixJobSpec.model_validate(compiled)
    assert len(job_spec.steps) == 1
    step = job_spec.steps[0]
    assert step.name == "agent-evaluate"
    _assert_agent_eval_step_entrypoint(
        job_spec,
        expected_image=f"registry.example/{expected_image_name}:test",
        expected_entrypoint=expected_entrypoint,
    )
    config = cast(dict[str, Any], step.config)
    assert len(config["tasks"]) == 1
    assert config["target"]["kind"] == expected_kind
    if expected_endpoint_name is not None:
        endpoint = config["target"].get("model") or config["target"].get("agent")
        assert endpoint["name"] == expected_endpoint_name


async def test_compile_gym_target_honors_configured_image_override(mocker: MockerFixture) -> None:
    _patch_execution_profiles(mocker, [])
    image = "ghcr.io/nvidia-nemo/nemo-helix/nhx-gym-tasks:internal-test"
    mocker.patch("nemo_evals.jobs.agent_compiler.config.gym_tasks_image", image)
    qualify = mocker.patch("nemo_evals.jobs.agent_compiler.get_qualified_image")
    spec = AgentEvalSpec(
        tasks=[_task_spec()],
        target=GymRunnerTarget(
            source=GymAgentSource(
                component="simple_agent", config="responses_api_agents/simple_agent/configs/simple_agent.yaml"
            ),
            resources_server="mcqa",
        ),
    )

    compiled = await AgentEvalJob.compile(
        workspace="default", spec=spec, entity_client=object(), job_name=None, async_sdk=_async_sdk()
    )

    job_spec = HelixJobSpec.model_validate(compiled)
    _assert_agent_eval_step_entrypoint(
        job_spec,
        expected_image=image,
        expected_entrypoint=("/app/.venv/bin/python", "-m"),
    )
    qualify.assert_not_called()


def _gym_environment_target() -> GymRunnerTarget:
    return GymRunnerTarget(
        environment=FilesetRef(root="dev/custom-gym"),
        source=GymAgentSource(
            component="custom_agent", config="responses_api_agents/custom_agent/configs/custom_agent.yaml"
        ),
        resources_server="custom_resources",
    )


def _enable_fileset_sandbox(mocker: MockerFixture) -> None:
    mocker.patch(
        "nemo_evals.jobs.agent_evaluate.require_fileset_environment_sandboxed",
        return_value=None,
    )
    mocker.patch(
        "nemo_evals.jobs.agent_evaluate.require_fileset_sandbox_storage_identity",
        return_value=None,
    )
    mocker.patch("nemo_evals.jobs.agent_compiler.config.sandboxed_gym_default", True)
    mocker.patch("nemo_evals.jobs.agent_compiler.config.sandbox_cluster_capable", True)
    mocker.patch(
        "nemo_evals.jobs.agent_compiler.config.sandbox_runtime_image",
        "registry.example.com/nhx-gym-runtime:1.0",
    )
    mocker.patch(
        "nemo_evals.jobs.agent_compiler.config.sandbox_job_storage_pvc_claim",
        "job-storage",
    )
    mocker.patch(
        "nemo_evals.jobs.agent_compiler.config.sandbox_policy_base_urls",
        ("https://integrate.api.nvidia.com/v1",),
    )


async def test_compile_gym_environment_adds_staging_step_before_evaluation(mocker: MockerFixture) -> None:
    _patch_execution_profiles(mocker, [_kubernetes_profile_with_job_storage()])
    mocker.patch("nemo_evals.jobs.agent_compiler.config.gym_tasks_image", None)
    mocker.patch(
        "nemo_evals.jobs.agent_compiler.get_qualified_image",
        side_effect=lambda name: f"registry.example/{name}:test",
    )
    _enable_fileset_sandbox(mocker)
    spec = AgentEvalSpec(tasks=[_task_spec()], target=_gym_environment_target())

    compiled = await AgentEvalJob.compile(
        workspace="dev",
        spec=spec,
        entity_client=object(),
        job_name=None,
        async_sdk=_async_sdk(),
    )

    job_spec = HelixJobSpec.model_validate(compiled)
    assert [step.name for step in job_spec.steps] == ["stage-environment", "agent-evaluate"]
    stage, evaluate = job_spec.steps
    stage_container = cast(Any, stage.executor).container
    assert stage_container.image == "registry.example/nhx-tasks:test"
    assert stage_container.entrypoint == ["python", "-m"]
    assert stage_container.command == ["nemo_evals.tasks.stage_environment"]
    assert stage.config == {"environment": "dev/custom-gym"}
    evaluate_container = cast(Any, evaluate.executor).container
    assert evaluate_container.image == "registry.example/nhx-tasks:test"
    assert evaluate_container.entrypoint == ["python", "-m"]
    assert evaluate_container.command == ["nemo_evals.tasks.agent_evaluate"]
    evaluate_config = cast(dict[str, Any], evaluate.config)
    assert evaluate_config["target"]["environment"] == "dev/custom-gym"


async def test_compile_sandboxed_gym_uses_cpu_tasks_and_ignores_colocated_image_override(
    mocker: MockerFixture,
) -> None:
    _patch_execution_profiles(mocker, [])
    mocker.patch(
        "nemo_evals.jobs.agent_compiler.config.gym_tasks_image",
        "registry.example/nhx-gym-tasks:colocated-only",
    )
    mocker.patch(
        "nemo_evals.jobs.agent_compiler.get_qualified_image",
        side_effect=lambda name: f"registry.example/{name}:test",
    )
    _enable_fileset_sandbox(mocker)
    spec = AgentEvalSpec(
        tasks=[_task_spec()],
        target=GymRunnerTarget(
            source=GymAgentSource(
                component="simple_agent", config="responses_api_agents/simple_agent/configs/simple_agent.yaml"
            ),
            resources_server="mcqa",
        ),
    )

    compiled = await AgentEvalJob.compile(
        workspace="dev",
        spec=spec,
        entity_client=object(),
        job_name=None,
        async_sdk=_async_sdk(),
    )

    job_spec = HelixJobSpec.model_validate(compiled)
    _assert_agent_eval_step_entrypoint(
        job_spec,
        expected_image="registry.example/nhx-tasks:test",
        expected_entrypoint=("python", "-m"),
    )


async def test_compile_gym_environment_propagates_platform_sandbox_protocol(mocker: MockerFixture) -> None:
    _patch_execution_profiles(mocker, [_kubernetes_profile_with_job_storage()])
    mocker.patch(
        "nemo_evals.jobs.agent_compiler.get_qualified_image",
        return_value="registry.example/nhx-gym-tasks:test",
    )
    _enable_fileset_sandbox(mocker)
    mocker.patch("nemo_evals.jobs.agent_compiler.platform_config.sandbox_server_protocol", "http")

    compiled = await AgentEvalJob.compile(
        workspace="dev",
        spec=AgentEvalSpec(tasks=[_task_spec()], target=_gym_environment_target()),
        entity_client=object(),
        job_name=None,
        async_sdk=_async_sdk(),
    )

    evaluate = HelixJobSpec.model_validate(compiled).steps[-1]
    serialized_plan = next(
        variable.value for variable in evaluate.environment or [] if variable.name == GYM_SANDBOX_PLAN_ENVVAR
    )
    assert serialized_plan is not None
    plan = json.loads(serialized_plan)
    assert plan["host_provider_options"] == {"connection": {"protocol": "http"}}


async def test_compile_gym_environment_uses_host_commands_for_subprocess_profile(
    mocker: MockerFixture,
) -> None:
    mocker.patch(
        "nemo_evals.jobs.agent_compiler.get_qualified_image",
        return_value="registry.example/nhx-gym-tasks:test",
    )
    _enable_fileset_sandbox(mocker)
    mocker.patch("nemo_evals.jobs.agent_compiler.config.sandbox_host_provider", "docker")
    evaluator_config = mocker.Mock()
    evaluator_config.sandbox_host_provider = "docker"
    mocker.patch(
        "nemo_evals.jobs.agent_evaluate.get_config",
        return_value=evaluator_config,
    )
    _patch_execution_profiles(mocker, [SubprocessJobExecutionProfile()])
    spec = AgentEvalSpec(tasks=[_task_spec()], target=_gym_environment_target())

    compiled = await AgentEvalJob.compile(
        workspace="dev",
        spec=spec,
        entity_client=object(),
        job_name=None,
        async_sdk=_async_sdk(),
    )

    job_spec = HelixJobSpec.model_validate(compiled)
    assert [step.name for step in job_spec.steps] == ["stage-environment", "agent-evaluate"]

    stage, evaluate = job_spec.steps
    assert isinstance(stage.executor, SubprocessExecutionProvider)
    assert stage.executor.command == ["python", "-m", "nemo_evals.tasks.stage_environment"]
    assert isinstance(evaluate.executor, SubprocessExecutionProvider)
    assert evaluate.executor.command == ["python", "-m", "nemo_evals.tasks.agent_evaluate"]


async def test_compile_rejects_fileset_environment_when_sandboxing_is_disabled(mocker: MockerFixture) -> None:
    _patch_execution_profiles(mocker, [_kubernetes_profile_with_job_storage()])
    mocker.patch("nemo_evals.jobs.agent_compiler.config.gym_tasks_image", None)
    mocker.patch(
        "nemo_evals.jobs.agent_compiler.get_qualified_image",
        return_value="registry.example/nhx-gym-tasks:test",
    )

    with pytest.raises(HelixJobCompilationError, match="require sandboxed execution"):
        await AgentEvalJob.compile(
            workspace="dev",
            spec=AgentEvalSpec(tasks=[_task_spec()], target=_gym_environment_target()),
            entity_client=object(),
            job_name=None,
            async_sdk=_async_sdk(),
        )


async def test_compile_rejects_a_registered_gym_agent_when_sandboxing_is_disabled(mocker: MockerFixture) -> None:
    """A registered agent stages an environment tree with no `target.environment`; compile must still check it."""
    _patch_execution_profiles(mocker, [_kubernetes_profile_with_job_storage()])
    mocker.patch("nemo_evals.jobs.agent_compiler.config.gym_tasks_image", None)
    mocker.patch(
        "nemo_evals.jobs.agent_compiler.get_qualified_image",
        return_value="registry.example/nhx-gym-tasks:test",
    )
    target = GymRunnerTarget(
        source=RegisteredAgentSource(agent=AgentRef(root="dev/calc")),
        resources_server="mcqa",
        resolved_config={"harness": {"adapter_id": "nvidia.fabric.langchain.deepagents"}},
    )

    with pytest.raises(HelixJobCompilationError, match="registered agent's Gym package require sandboxed execution"):
        await AgentEvalJob.compile(
            workspace="dev",
            spec=AgentEvalSpec(tasks=[_task_spec()], target=target),
            entity_client=object(),
            job_name=None,
            async_sdk=_async_sdk(),
        )


async def test_compile_rejects_mismatched_job_and_sandbox_storage_pvcs(mocker: MockerFixture) -> None:
    mocker.patch("nemo_evals.jobs.agent_compiler.config.gym_tasks_image", None)
    mocker.patch(
        "nemo_evals.jobs.agent_compiler.get_qualified_image",
        return_value="registry.example/nhx-gym-tasks:test",
    )
    mocker.patch(
        "nemo_evals.jobs.agent_evaluate.require_fileset_environment_sandboxed",
        return_value=None,
    )
    mocker.patch(
        "nemo_evals.jobs.agent_evaluate.get_config",
        return_value=EvaluatorConfig(
            sandboxed_gym_default=True,
            sandbox_cluster_capable=True,
            sandbox_runtime_image="registry.example.com/nhx-gym-runtime:1.0",
            sandbox_job_storage_pvc_claim="job-storage",
            sandbox_policy_base_urls=("https://integrate.api.nvidia.com/v1",),
        ),
    )
    _patch_execution_profiles(
        mocker,
        [
            KubernetesJobExecutionProfile(
                config=KubernetesJobExecutionProfileConfig(storage=KubernetesJobStorageConfig(pvc_name="jobs-pvc"))
            )
        ],
    )

    with pytest.raises(HelixJobCompilationError, match="sandbox_job_storage_pvc_claim"):
        await AgentEvalJob.compile(
            workspace="dev",
            spec=AgentEvalSpec(tasks=[_task_spec()], target=_gym_environment_target()),
            entity_client=object(),
            job_name=None,
            async_sdk=_async_sdk(),
        )


async def test_compile_prefers_kubernetes_profile_for_opensandbox_fileset(mocker: MockerFixture) -> None:
    mocker.patch("nemo_evals.jobs.agent_compiler.config.gym_tasks_image", None)
    mocker.patch(
        "nemo_evals.jobs.agent_compiler.get_qualified_image",
        return_value="registry.example/nhx-gym-tasks:test",
    )
    _enable_fileset_sandbox(mocker)
    mocker.patch("nemo_evals.jobs.agent_compiler.config.sandbox_host_provider", "opensandbox")
    mocker.patch("nemo_evals.jobs.agent_compiler.config.sandbox_job_storage_pvc_claim", "jobs-pvc")
    evaluator_config = mocker.Mock()
    evaluator_config.sandboxed_gym_default = True
    evaluator_config.sandbox_host_provider = "opensandbox"
    evaluator_config.sandbox_job_storage_pvc_claim = "jobs-pvc"
    mocker.patch("nemo_evals.jobs.agent_evaluate.get_config", return_value=evaluator_config)
    _patch_execution_profiles(
        mocker,
        [
            SubprocessJobExecutionProfile(profile="default"),
            KubernetesJobExecutionProfile(
                profile="default",
                config=KubernetesJobExecutionProfileConfig(storage=KubernetesJobStorageConfig(pvc_name="jobs-pvc")),
            ),
        ],
    )

    compiled = await AgentEvalJob.compile(
        workspace="dev",
        spec=AgentEvalSpec(tasks=[_task_spec()], target=_gym_environment_target()),
        entity_client=object(),
        job_name=None,
        async_sdk=_async_sdk(),
    )

    job_spec = HelixJobSpec.model_validate(compiled)
    assert all(not isinstance(step.executor, SubprocessExecutionProvider) for step in job_spec.steps)


async def test_compile_marks_gym_profile_transport_failure_as_retryable(mocker: MockerFixture) -> None:
    request = httpx.Request("GET", "http://jobs.test/v2/execution-profiles")
    jobs_client = mocker.Mock()
    jobs_client.get_execution_profiles = mocker.AsyncMock(
        side_effect=NemoTransportError(httpx.ConnectError("profiles unavailable", request=request))
    )
    mocker.patch("nemo_evals.jobs.agent_evaluate.client_from_platform", return_value=jobs_client)
    spec = AgentEvalSpec(tasks=[_task_spec()], target=_gym_environment_target())

    with pytest.raises(HelixJobDependencyUnavailableError, match="Jobs service is temporarily unavailable"):
        await AgentEvalJob.compile(
            workspace="dev",
            spec=spec,
            entity_client=object(),
            job_name=None,
            async_sdk=_async_sdk(),
        )


def _wheels_manifest_bytes() -> bytes:
    return (
        b"format: wheels-v1\n"
        b"config_paths:\n"
        b"  - resources_servers/custom/configs/custom.yaml\n"
        b"metadata:\n"
        b"  name: custom-gym\n"
    )


async def test_resolve_gym_environment_qualifies_and_validates_purpose(mocker: MockerFixture) -> None:
    response = mocker.Mock()
    response.data.return_value = SimpleNamespace(purpose=FilesetPurpose.ENVIRONMENT)
    listing = mocker.Mock()
    listing.data.return_value = SimpleNamespace(
        data=[
            SimpleNamespace(path="nemo-environment.yaml"),
            SimpleNamespace(path="resources_servers/custom/configs/custom.yaml"),
            SimpleNamespace(path="wheels/custom_dependency-1.0-py3-none-any.whl"),
        ]
    )
    manifest = mocker.Mock()
    manifest.read = mocker.AsyncMock(return_value=_wheels_manifest_bytes())
    files = mocker.Mock()
    files.get_fileset = mocker.AsyncMock(return_value=response)
    files.list_files = mocker.AsyncMock(return_value=listing)
    files.download_file = mocker.AsyncMock(return_value=manifest)
    mocker.patch("nemo_evals.jobs.gym_submission.client_from_platform", return_value=files)
    target = GymRunnerTarget(
        environment=FilesetRef(root="custom-gym"),
        source=GymAgentSource(
            component="custom_agent", config="responses_api_agents/custom_agent/configs/custom_agent.yaml"
        ),
        resources_server="custom_resources",
    )

    resolved = await resolve_gym_environment(
        target,
        workspace="dev",
        async_client=_async_sdk(),
    )

    assert isinstance(resolved, GymRunnerTarget)
    assert resolved.environment == FilesetRef(root="dev/custom-gym")
    files.get_fileset.assert_awaited_once_with(workspace="dev", name="custom-gym")
    files.list_files.assert_awaited_once_with(workspace="dev", name="custom-gym")
    files.download_file.assert_awaited_once_with(
        workspace="dev",
        name="custom-gym",
        path="nemo-environment.yaml",
    )


async def test_resolve_gym_environment_accepts_native_v1(mocker: MockerFixture) -> None:
    response = mocker.Mock()
    response.data.return_value = SimpleNamespace(purpose=FilesetPurpose.ENVIRONMENT)
    listing = mocker.Mock()
    listing.data.return_value = SimpleNamespace(
        data=[
            SimpleNamespace(path="nemo-environment.yaml"),
            SimpleNamespace(path="resources_servers/custom/configs/custom.yaml"),
        ]
    )
    manifest = mocker.Mock()
    manifest.read = mocker.AsyncMock(
        return_value=(
            b"format: native-v1\n"
            b"config_paths:\n"
            b"  - resources_servers/custom/configs/custom.yaml\n"
            b"metadata:\n"
            b"  name: custom-gym\n"
        )
    )
    files = mocker.Mock()
    files.get_fileset = mocker.AsyncMock(return_value=response)
    files.list_files = mocker.AsyncMock(return_value=listing)
    files.download_file = mocker.AsyncMock(return_value=manifest)
    mocker.patch("nemo_evals.jobs.gym_submission.client_from_platform", return_value=files)

    resolved = await resolve_gym_environment(
        _gym_environment_target(),
        workspace="dev",
        async_client=_async_sdk(),
    )

    assert isinstance(resolved, GymRunnerTarget)
    assert resolved.environment == FilesetRef(root="dev/custom-gym")


async def test_resolve_gym_environment_rejects_wrong_purpose(mocker: MockerFixture) -> None:
    response = mocker.Mock()
    response.data.return_value = SimpleNamespace(purpose=FilesetPurpose.DATASET)
    files = mocker.Mock()
    files.get_fileset = mocker.AsyncMock(return_value=response)
    mocker.patch("nemo_evals.jobs.gym_submission.client_from_platform", return_value=files)
    target = GymRunnerTarget(
        environment=FilesetRef(root="dev/not-an-environment"),
        source=GymAgentSource(
            component="custom_agent", config="responses_api_agents/custom_agent/configs/custom_agent.yaml"
        ),
        resources_server="custom_resources",
    )

    with pytest.raises(ValueError, match="expected 'environment'"):
        await resolve_gym_environment(
            target,
            workspace="dev",
            async_client=_async_sdk(),
        )


async def test_resolve_gym_environment_rejects_missing_manifest(mocker: MockerFixture) -> None:
    response = mocker.Mock()
    response.data.return_value = SimpleNamespace(purpose=FilesetPurpose.ENVIRONMENT)
    listing = mocker.Mock()
    listing.data.return_value = SimpleNamespace(
        data=[SimpleNamespace(path="resources_servers/custom/configs/custom.yaml")]
    )
    files = mocker.Mock()
    files.get_fileset = mocker.AsyncMock(return_value=response)
    files.list_files = mocker.AsyncMock(return_value=listing)
    files.download_file = mocker.AsyncMock()
    mocker.patch("nemo_evals.jobs.gym_submission.client_from_platform", return_value=files)

    with pytest.raises(ValueError, match="has no nemo-environment.yaml at its root"):
        await resolve_gym_environment(
            _gym_environment_target(),
            workspace="dev",
            async_client=_async_sdk(),
        )

    files.download_file.assert_not_awaited()


async def test_resolve_gym_environment_rejects_manifest_listing_mismatch(mocker: MockerFixture) -> None:
    response = mocker.Mock()
    response.data.return_value = SimpleNamespace(purpose=FilesetPurpose.ENVIRONMENT)
    listing = mocker.Mock()
    listing.data.return_value = SimpleNamespace(data=[SimpleNamespace(path="nemo-environment.yaml")])
    manifest = mocker.Mock()
    manifest.read = mocker.AsyncMock(return_value=_wheels_manifest_bytes())
    files = mocker.Mock()
    files.get_fileset = mocker.AsyncMock(return_value=response)
    files.list_files = mocker.AsyncMock(return_value=listing)
    files.download_file = mocker.AsyncMock(return_value=manifest)
    mocker.patch("nemo_evals.jobs.gym_submission.client_from_platform", return_value=files)

    with pytest.raises(ValueError, match="config_paths reference files that are not in the package"):
        await resolve_gym_environment(
            _gym_environment_target(),
            workspace="dev",
            async_client=_async_sdk(),
        )


async def test_compile_rejects_harbor_target_for_docker_profile(mocker: MockerFixture) -> None:
    """The full rejection message: the resolved backend plus the standing Harbor requirement."""
    _patch_execution_profiles(
        mocker,
        [DockerJobExecutionProfile(provider="cpu", profile="default", config=DockerJobExecutionProfileConfig())],
    )

    with pytest.raises(HelixJobCompilationError) as exc_info:
        await _compile_harbor(async_sdk=_async_sdk())

    message = str(exc_info.value)
    assert "profile 'default'" in message
    assert "backend 'docker'" in message
    assert "Harbor targets currently require the subprocess backend" in message


@pytest.mark.parametrize(
    ("execution_profile", "backend"),
    [
        (
            KubernetesJobExecutionProfile(config=KubernetesJobExecutionProfileConfig()),
            "kubernetes_job",
        ),
        (
            VolcanoJobExecutionProfile(config=VolcanoJobExecutionProfileConfig()),
            "volcano_job",
        ),
    ],
)
async def test_compile_rejects_harbor_target_for_containerized_profile(
    execution_profile: BaseExecutionProfile,
    backend: str,
    mocker: MockerFixture,
) -> None:
    _patch_execution_profiles(mocker, [execution_profile])

    with pytest.raises(HelixJobCompilationError, match=rf"backend '{backend}'"):
        await _compile_harbor(async_sdk=_async_sdk())


async def test_compile_routes_harbor_directly_to_advertised_subprocess_profile(mocker: MockerFixture) -> None:
    """An advertised default subprocess backend must be selected, not merely inferred as a translation."""
    _patch_execution_profiles(
        mocker,
        [
            DockerJobExecutionProfile(provider="cpu", profile="default", config=DockerJobExecutionProfileConfig()),
            SubprocessJobExecutionProfile(),
        ],
    )

    job_spec = await _compile_harbor(async_sdk=_async_sdk())

    executor = job_spec.steps[0].executor
    assert isinstance(executor, SubprocessExecutionProvider)
    assert executor.profile == "default"
    assert executor.command == ["python", "-m", "nemo_evals.tasks.agent_evaluate"]
    assert cast(dict[str, Any], job_spec.steps[0].config)["target"]["kind"] == "harbor"


async def test_compile_rejects_harbor_when_profile_is_missing(mocker: MockerFixture) -> None:
    _patch_execution_profiles(mocker, [SubprocessJobExecutionProfile.model_validate({"profile": "other"})])

    with pytest.raises(HelixJobCompilationError, match="profile 'default'.*does not resolve"):
        await _compile_harbor(async_sdk=_async_sdk())


async def test_compile_marks_profile_transport_failure_as_retryable(mocker: MockerFixture) -> None:
    request = httpx.Request("GET", "http://jobs.test/v2/execution-profiles")
    jobs_client = mocker.Mock()
    jobs_client.get_execution_profiles = mocker.AsyncMock(
        side_effect=NemoTransportError(httpx.ConnectError("profiles unavailable", request=request))
    )
    mocker.patch("nemo_evals.jobs.agent_evaluate.client_from_platform", return_value=jobs_client)

    with pytest.raises(HelixJobDependencyUnavailableError, match="Jobs service is temporarily unavailable"):
        await _compile_harbor(async_sdk=_async_sdk())


@pytest.mark.parametrize("failure_kind", ["invalid-response", "server-error"])
async def test_compile_marks_profile_dependency_failure_as_retryable(failure_kind: str, mocker: MockerFixture) -> None:
    request = httpx.Request("GET", "http://jobs.test/v2/execution-profiles")
    response = httpx.Response(503, json={"detail": "jobs unavailable"}, request=request)
    error: Exception
    if failure_kind == "invalid-response":
        error = NemoResponseValidationError(response, ValueError("invalid profile response"))
    else:
        error = InternalServerError(response)
    jobs_client = mocker.Mock()
    jobs_client.get_execution_profiles = mocker.AsyncMock(side_effect=error)
    mocker.patch("nemo_evals.jobs.agent_evaluate.client_from_platform", return_value=jobs_client)

    with pytest.raises(HelixJobDependencyUnavailableError, match="Jobs service is temporarily unavailable"):
        await _compile_harbor(async_sdk=_async_sdk())


async def test_compile_does_not_classify_unexpected_profile_failure_as_invalid(mocker: MockerFixture) -> None:
    jobs_client = mocker.Mock()
    jobs_client.get_execution_profiles = mocker.AsyncMock(side_effect=RuntimeError("unexpected lookup bug"))
    mocker.patch("nemo_evals.jobs.agent_evaluate.client_from_platform", return_value=jobs_client)

    with pytest.raises(RuntimeError, match="unexpected lookup bug"):
        await _compile_harbor(async_sdk=_async_sdk())


async def test_compile_non_harbor_target_does_not_resolve_execution_profiles(mocker: MockerFixture) -> None:
    mocker.patch(
        "nemo_evals.jobs.agent_evaluate.client_from_platform",
        side_effect=AssertionError("non-Harbor compilation must not query execution profiles"),
    )
    spec = AgentEvalSpec(tasks=[_task_spec()], target=_runner_target("openai/gpt-5.4"))

    compiled = await AgentEvalJob.compile(
        workspace="default",
        spec=spec,
        entity_client=object(),
        job_name=None,
        async_sdk=_async_sdk(),
    )

    assert cast(dict[str, Any], HelixJobSpec.model_validate(compiled).steps[0].config)["target"]["kind"] == "fabric"


async def test_compile_resolves_fabric_runner_env_secrets() -> None:
    target = FabricRunnerTarget(
        source=FabricConfigSource(config={"harness": {"adapter_id": "nvidia.fabric.codex"}}),
        env_secrets={"NVIDIA_API_KEY": SecretRef("my-workspace/nvidia-key")},
    )
    spec = AgentEvalSpec(tasks=[_task_spec()], target=target)
    compiled = await AgentEvalJob.compile(
        workspace="default", spec=spec, entity_client=object(), job_name=None, async_sdk=_async_sdk()
    )
    step = HelixJobSpec.model_validate(compiled).steps[0]
    secrets = {env.name: env.from_secret.name for env in step.environment or [] if env.from_secret}
    assert secrets == {"NVIDIA_API_KEY": "my-workspace/nvidia-key"}
    assert cast(dict[str, Any], step.config)["target"]["env_secrets"] == {"NVIDIA_API_KEY": "my-workspace/nvidia-key"}


def test_fabric_worker_uses_the_job_env_secret_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    target = FabricRunnerTarget(
        source=FabricConfigSource(config={"harness": {"adapter_id": "nvidia.fabric.codex"}}),
        env_secrets={"NVIDIA_API_KEY": SecretRef("my-workspace/nvidia-key")},
    )
    runtime, _, _ = AgentEvalJob._resolve_target(target, _job_context(tmp_path))
    assert isinstance(runtime, FabricAgentRuntime)
    assert runtime._env_secrets == target.env_secrets
    assert isinstance(runtime._secret_resolver, JobEnvSecretSource)


@pytest.mark.parametrize("registered", [False, True])
@pytest.mark.parametrize(
    "config,field",
    [
        ({"environment": {"env": {"KEY": "override"}}}, "KEY"),
        ({"environment": "local"}, "config.environment"),
        ({"environment": {"env": []}}, "config.environment.env"),
    ],
)
def test_fabric_target_rejects_invalid_secret_environment_at_submit(
    config: dict[str, Any], field: str, registered: bool
) -> None:
    with pytest.raises(ValidationError, match=field):
        if registered:
            FabricRunnerTarget(
                source=RegisteredAgentSource(agent="ws/agent"),
                resolved_config=config,
                env_secrets={"KEY": SecretRef("ws/key")},
            )
        else:
            FabricRunnerTarget(source=FabricConfigSource(config=config), env_secrets={"KEY": SecretRef("ws/key")})


@pytest.mark.parametrize("registered", [False, True])
def test_fabric_target_validation_error_does_not_echo_rejected_secret(registered: bool) -> None:
    """Reject credential collisions without printing either inline or resolved Fabric config values."""
    config = {"environment": {"env": {"KEY": "LEAKME"}}}
    target: dict[str, Any] = {"env_secrets": {"KEY": "ws/key"}}
    if registered:
        target["source"] = {"agent": "ws/agent"}
        target["resolved_config"] = config
    else:
        target["source"] = {"config": config}
    with pytest.raises(ValidationError, match="also provided by env_secrets") as excinfo:
        FabricRunnerTarget.model_validate(target)
    assert "LEAKME" not in str(excinfo.value)


async def test_compile_injects_target_api_key_secret() -> None:
    spec = AgentEvalSpec(
        tasks=[_task_spec()],
        target=ModelTarget(
            model=Model(
                url="https://integrate.api.nvidia.com/v1/chat/completions",
                name="nvidia/model",
                api_key_secret=SecretRef(root="NVIDIA_API_KEY"),
            ),
            params=RunConfigOnlineModel(),
        ),
    )

    compiled = await AgentEvalJob.compile(
        workspace="default", spec=spec, entity_client=object(), job_name=None, async_sdk=_async_sdk()
    )

    step = HelixJobSpec.model_validate(compiled).steps[0]
    secrets = {env.name: env.from_secret.name for env in step.environment or [] if env.from_secret}
    assert secrets == {"NVIDIA_API_KEY": "NVIDIA_API_KEY"}


async def test_compile_rejects_reserved_secret_env_name() -> None:
    spec = AgentEvalSpec(
        tasks=[_task_spec()],
        target=ModelTarget(
            model=Model(
                url="https://integrate.api.nvidia.com/v1/chat/completions",
                name="nvidia/model",
                api_key_secret=SecretRef(root=PERSISTENT_JOB_STORAGE_PATH_ENVVAR),
            ),
            params=RunConfigOnlineModel(),
        ),
    )

    with pytest.raises(ValueError, match="reserved"):
        await AgentEvalJob.compile(
            workspace="default", spec=spec, entity_client=object(), job_name=None, async_sdk=_async_sdk()
        )


@pytest.mark.parametrize("source", ["gym", "harbor", "fabric", "metric"])
async def test_compile_rejects_sandbox_plan_secret_name_from_all_sources(source: str, mocker: MockerFixture) -> None:
    name = GYM_SANDBOX_PLAN_ENVVAR
    task = _task_spec()
    if source == "gym":
        _patch_execution_profiles(mocker, [])
        target = GymRunnerTarget(
            source=GymAgentSource(component="simple_agent", config="a.yaml"),
            resources_server="mcqa",
            env_secrets={name: SecretRef("ws/x")},
        )
    elif source == "harbor":
        target = HarborRunnerTarget(env_secrets={name: SecretRef("ws/x")})
    elif source == "fabric":
        target = FabricRunnerTarget(source=FabricConfigSource(config={}), env_secrets={name: SecretRef("ws/x")})
    else:
        target = HarborRunnerTarget()
        metric = task.spec.metrics[0].model_copy(update={"secrets": {name: SecretRef("ws/x")}})
        task = task.model_copy(update={"spec": task.spec.model_copy(update={"metrics": [metric]})})
    spec = AgentEvalSpec(tasks=[task], target=target)
    with pytest.raises(ValueError, match="NEMO_EVALS_GYM_SANDBOX_PLAN.*reserved"):
        await AgentEvalJob.compile(
            workspace="default", spec=spec, entity_client=object(), job_name=None, async_sdk=_async_sdk()
        )


def test_sandbox_plan_secret_name_is_rejected_before_plan_is_appended() -> None:
    spec = AgentEvalSpec(
        tasks=[_task_spec()],
        target=GymRunnerTarget(
            source=GymAgentSource(component="simple_agent", config="a.yaml"),
            resources_server="mcqa",
            env_secrets={GYM_SANDBOX_PLAN_ENVVAR: SecretRef("ws/x")},
        ),
    )
    with pytest.raises(ValueError, match="NEMO_EVALS_GYM_SANDBOX_PLAN.*reserved"):
        _environment(spec, sandbox_plan=_sandbox_plan())


# --- sync job entrypoint: the in-process run path, across target types -------


@pytest.mark.parametrize(
    "target",
    [
        ModelTarget(
            model=Model(url="http://model.test/v1/chat/completions", name="test-model"), params=RunConfigOnlineModel()
        ),
        AgentTarget(agent=_agent(), params=RunConfigOnline()),
        _runner_target("openai/gpt-5.4"),
        HarborRunnerTarget(),
        GymRunnerTarget(
            source=GymAgentSource(
                component="simple_agent", config="responses_api_agents/simple_agent/configs/simple_agent.yaml"
            ),
            resources_server="mcqa",
        ),
    ],
)
def test_sync_job_executes_each_target_type(target: Target, tmp_path: Path, mocker: MockerFixture) -> None:
    # The evaluator is faked so the target is threaded (a runner resolved to its runtime, an endpoint
    # passed through) without real inference.
    fake = _FakeEvaluator()
    mocker.patch.object(AgentEvalJob, "_build_evaluator", return_value=fake)
    input_spec = AgentEvalInputSpec(
        tasks=[
            AgentEvalTaskInput(
                id="task-1",
                intent="Answer.",
                inputs=_task_inputs(instruction="What is 2+2?", gym_row={}),
                metadata=[MetadataItem(key="gym_row_extras", value={})],
                metrics=[_inline_metric()],
            )
        ],
        target=target,
    )

    canonical = run_sync(
        lambda: AgentEvalJob.to_spec(
            input_spec, workspace="default", entity_client=None, async_sdk=_async_sdk(), is_local=True
        )
    )
    result = AgentEvalJob().run(
        canonical.model_dump(mode="json"),
        ctx=_job_context(tmp_path),
        client=_sync_sdk_with_identity(),
    )

    assert result["status"] == "completed"
    assert result["artifact"]["name"] == DEFAULT_RESULT_NAME
    assert [task.id for task in fake.received_tasks] == ["task-1"]
    assert fake.received_trials is None  # online generation, not precomputed
    if isinstance(target, FabricRunnerTarget):
        assert isinstance(fake.received_target, FabricAgentRuntime)
    elif isinstance(target, HarborRunnerTarget):
        assert isinstance(fake.received_target, HarborAgentTaskRunner)
    elif isinstance(target, GymRunnerTarget):
        assert isinstance(fake.received_target, GymAgentTaskRunner)
    elif isinstance(target, ModelTarget):
        assert getattr(fake.received_target, "name", None) == target.model.name
    else:
        assert getattr(fake.received_target, "name", None) == target.agent.name


def test_sync_job_scores_precomputed_trials_offline(tmp_path: Path, mocker: MockerFixture) -> None:
    # Offline eval: precomputed trials are scored directly, with no target / no generation.
    fake = _FakeEvaluator()
    mocker.patch.object(AgentEvalJob, "_build_evaluator", return_value=fake)
    precomputed = [
        AgentEvalTrial(
            id="t-1", task_id="task-1", status=AgentEvalTrialStatus.COMPLETED, output=AgentOutput(output_text="4")
        )
    ]
    input_spec = AgentEvalInputSpec(
        tasks=[AgentEvalTaskInput(id="task-1", intent="Answer.", inputs=TaskInputs(), metrics=[_inline_metric()])],
        trials=precomputed,
    )

    canonical = run_sync(
        lambda: AgentEvalJob.to_spec(
            input_spec, workspace="default", entity_client=None, async_sdk=_async_sdk(), is_local=True
        )
    )
    result = AgentEvalJob().run(
        canonical.model_dump(mode="json"),
        ctx=_job_context(tmp_path),
        client=_sync_sdk_with_identity(),
    )

    assert result["status"] == "completed"
    assert fake.received_target is None
    assert [trial.id for trial in fake.received_trials or []] == ["t-1"]


def test_spec_requires_exactly_one_of_target_or_trials() -> None:
    trial = AgentEvalTrial(
        id="t-1", task_id="task-1", status=AgentEvalTrialStatus.COMPLETED, output=AgentOutput(output_text="4")
    )
    with pytest.raises(ValueError, match="exactly one"):
        AgentEvalSpec(tasks=[_task_spec()])  # neither target nor trials
    with pytest.raises(ValueError, match="exactly one"):
        AgentEvalSpec(tasks=[_task_spec()], target=_runner_target("openai/gpt-5.4"), trials=[trial])  # both


# --- container task entrypoint ----------------------------------------------


class TestAgentEvalTask:
    """Coverage for the compiled container/subprocess task entrypoint."""

    def test_main_dispatches_agent_eval_job_with_task_client(self, mocker: MockerFixture) -> None:
        client = NemoClient(
            base_url="http://platform.test", workspace="default", http_client=MagicMock(spec=httpx.Client)
        )
        async_client = AsyncNemoClient(
            base_url="http://platform.test", workspace="default", http_client=AsyncMock(spec=httpx.AsyncClient)
        )
        ctx = MagicMock()
        build_ctx = mocker.patch("nemo_evals.tasks.runner.build_ctx_from_env", return_value=ctx)
        get_task_client = mocker.patch("nemo_evals.tasks.runner.get_task_nemo_client", return_value=client)
        get_async_task_client = mocker.patch(
            "nemo_evals.tasks.runner.get_async_task_nemo_client", return_value=async_client
        )
        run_task = mocker.patch("nemo_evals.tasks.runner.run_task_with_async_client", return_value=0)

        exit_code = agent_eval_task_main()

        assert exit_code == 0
        get_task_client.assert_called_once_with("evals")
        build_ctx.assert_called_once_with(client)
        get_async_task_client.assert_called_once_with("evals")
        run_task.assert_called_once_with(AsyncAgentEvalJob, async_client=async_client, ctx=ctx)

    def test_main_returns_setup_exit_code_when_task_client_fails(self, mocker: MockerFixture) -> None:
        get_task_client = mocker.patch("nemo_evals.tasks.runner.get_task_nemo_client", side_effect=RuntimeError("boom"))
        get_async_task_client = mocker.patch("nemo_evals.tasks.runner.get_async_task_nemo_client")
        run_task = mocker.patch("nemo_evals.tasks.runner.run_task_with_async_client")

        exit_code = agent_eval_task_main()

        assert exit_code == SDK_INITIALIZATION_EXIT_CODE
        get_task_client.assert_called_once_with("evals")
        get_async_task_client.assert_not_called()
        run_task.assert_not_called()


async def test_trial_error_survives_the_job_spec_wire_contract() -> None:
    """``AgentEvalTrial.error`` is public API, not just an SDK-internal field.

    Precomputed trials are accepted straight off the wire by ``AgentEvalInputSpec.trials``, and
    ``AgentEvalTrial`` forbids extras — so a typed error has to survive JSON round-tripping through
    the DTO. Regenerating the OpenAPI schema proves the shapes agree; only this proves a request
    carrying one actually validates.
    """
    payload = {
        "id": "debug-agent-runtime-error__KFtcHEw",
        "task_id": "fix-bug",
        "status": "partial",
        "error": {
            "type": "RuntimeError",
            "message": "Agent process failed with exit code 127",
            "traceback": "Traceback (most recent call last):\n",
            "occurred_at": "2026-08-13T17:22:32.230852",
        },
    }

    input_spec = AgentEvalInputSpec.model_validate(
        {
            "trials": [payload],
            "tasks": [
                {
                    "id": "fix-bug",
                    "intent": "Fix the bug.",
                    "inputs": {"instruction": "Fix calculator.py."},
                    "metrics": [_inline_metric().model_dump()],
                }
            ],
        }
    )

    assert input_spec.trials is not None
    error = input_spec.trials[0].error
    assert error is not None
    assert error.type == "RuntimeError"
    assert error.message == "Agent process failed with exit code 127"

    spec = await AgentEvalJob.to_spec(
        input_spec, workspace="dev", entity_client=None, async_sdk=_async_sdk(), is_local=True
    )
    assert isinstance(spec, AgentEvalSpec)
    assert spec.trials is not None
    assert spec.trials[0].error == error
    # And it survives a full serialize -> deserialize hop, which is how the job actually receives it.
    round_tripped = AgentEvalSpec.model_validate(json.loads(json.dumps(spec.model_dump(mode="json"))))
    assert round_tripped.trials is not None
    assert round_tripped.trials[0].error == error


@pytest.mark.parametrize("spec_type", [AgentEvalInputSpec, AgentEvalSpec])
@pytest.mark.parametrize(
    ("payload", "expected_measurements"),
    [
        pytest.param(
            {
                "measurements": {
                    "prompt_tokens": 8,
                    "completion_tokens": 2,
                    "runtime_sec": 0,
                    "cost_usd": 0,
                },
                "metadata": {"reward": 0.8},
            },
            TrialMeasurements(prompt_tokens=8, completion_tokens=2, runtime_sec=0, cost_usd=0),
            id="typed",
        ),
        pytest.param(
            {"metadata": {"prompt_tokens": 8, "completion_tokens": 2, "duration_ms": 1500}},
            TrialMeasurements(),
            id="metadata-only",
        ),
        pytest.param(
            {
                "measurements": {"prompt_tokens": 8, "completion_tokens": 2},
                "metadata": {"prompt_tokens": 999, "completion_tokens": 999, "duration_ms": 1500},
            },
            TrialMeasurements(prompt_tokens=8, completion_tokens=2),
            id="conflicting-metadata",
        ),
    ],
)
def test_precomputed_trial_measurements_validate_across_both_job_specs(
    spec_type: type[AgentEvalInputSpec] | type[AgentEvalSpec],
    payload: dict[str, object],
    expected_measurements: TrialMeasurements,
) -> None:
    row = {"id": "stored-trial", "task_id": "task-1", "status": "partial", **payload}

    spec = spec_type.model_validate(
        {
            "trials": [row],
            "tasks": [
                _task_spec().model_dump(mode="json")
                if spec_type is AgentEvalSpec
                else {"id": "task-1", **_task_spec().spec.model_dump(mode="json", exclude={"kind", "provenance"})}
            ],
        }
    )

    assert spec.trials is not None
    assert spec.trials[0].measurements == expected_measurements
    assert spec.trials[0].metadata == payload.get("metadata", {})


@pytest.mark.parametrize("spec_type", [AgentEvalInputSpec, AgentEvalSpec])
@pytest.mark.parametrize("measurements", [None, {"prompt_tokens": -1}])
def test_job_specs_reject_invalid_typed_measurements_without_metadata_fallback(
    spec_type: type[AgentEvalInputSpec] | type[AgentEvalSpec],
    measurements: object,
) -> None:
    with pytest.raises(ValidationError):
        spec_type.model_validate(
            {
                "trials": [
                    {
                        "id": "stored-trial",
                        "task_id": "task-1",
                        "status": "partial",
                        "measurements": measurements,
                        "metadata": {"prompt_tokens": 8, "duration_ms": 1500},
                    }
                ],
                "tasks": [
                    _task_spec().model_dump(mode="json")
                    if spec_type is AgentEvalSpec
                    else {"id": "task-1", **_task_spec().spec.model_dump(mode="json", exclude={"kind", "provenance"})}
                ],
            }
        )


async def test_compile_resolves_gym_runner_env_secrets(mocker: MockerFixture) -> None:
    """A Gym environment's model key reaches it through the OS environment, not an endpoint spec.

    Without this the only route was `env_vars`, which stores the credential in plaintext on the spec
    and writes it into whatever persists that spec.
    """
    _patch_execution_profiles(mocker, [])
    spec = AgentEvalSpec(
        tasks=[_task_spec()],
        target=GymRunnerTarget(
            source=GymAgentSource(
                component="simple_agent", config="responses_api_agents/simple_agent/configs/simple_agent.yaml"
            ),
            resources_server="mcqa",
            env_vars={"WMT_TRANSLATION_COMET_PY_CACHE": "/shared/cache"},
            env_secrets={"OPENAI_API_KEY": SecretRef(root="my-workspace/openai-key")},
        ),
    )

    compiled = await AgentEvalJob.compile(
        workspace="default", spec=spec, entity_client=object(), job_name=None, async_sdk=_async_sdk()
    )

    step = HelixJobSpec.model_validate(compiled).steps[0]
    secrets = {env.name: env.from_secret.name for env in step.environment or [] if env.from_secret}
    assert secrets == {"OPENAI_API_KEY": "my-workspace/openai-key"}
    # The reference travels; the value never does.
    stored_target = cast(dict[str, Any], step.config)["target"]
    assert stored_target["env_secrets"] == {"OPENAI_API_KEY": "my-workspace/openai-key"}
    assert stored_target["env_vars"] == {"WMT_TRANSLATION_COMET_PY_CACHE": "/shared/cache"}


async def test_compile_resolves_harbor_runner_env_secrets(mocker: MockerFixture) -> None:
    """A Harbor agent's model key reaches it through the job environment, not through `agent_kwargs`.

    `agent_kwargs` is persisted verbatim by Harbor into the job dir's config.json, so a credential there
    lands in plaintext on disk; the secret reference is the only thing allowed to travel on the spec.
    """
    _patch_execution_profiles(mocker, [SubprocessJobExecutionProfile()])
    spec = AgentEvalSpec(
        tasks=[_task_spec()],
        target=HarborRunnerTarget(
            source=HarborImportedAgentSource(import_path="nemo_fabric.integrations.harbor:FabricAgent"),
            agent_kwargs={"fabric_adapter_id": "nvidia.fabric.codex"},
            env_secrets={"OPENAI_API_KEY": SecretRef(root="my-workspace/openai-key")},
        ),
    )

    compiled = await AgentEvalJob.compile(
        workspace="default", spec=spec, entity_client=object(), job_name=None, async_sdk=_async_sdk()
    )

    step = HelixJobSpec.model_validate(compiled).steps[0]
    secrets = {env.name: env.from_secret.name for env in step.environment or [] if env.from_secret}
    assert secrets == {"OPENAI_API_KEY": "my-workspace/openai-key"}
    stored_target = cast(dict[str, Any], step.config)["target"]
    assert stored_target["env_secrets"] == {"OPENAI_API_KEY": "my-workspace/openai-key"}


@pytest.mark.parametrize("valid", [False, True])
async def test_gym_submission_validates_before_environment_resolution(monkeypatch, valid) -> None:
    """Invalid Gym rows stop submission before environment lookups; valid rows reach them."""
    from unittest.mock import AsyncMock

    from nemo_evals.api.schemas import MetadataItem

    resolver = AsyncMock(side_effect=lambda target, **kwargs: target)
    monkeypatch.setattr("nemo_evals.jobs.gym_submission.resolve_gym_environment", resolver)
    request = AgentEvalInputSpec(
        tasks=[
            AgentEvalTaskInput(
                id="task",
                intent="Run",
                inputs=TaskInputs.model_validate({"gym_row": {}} if valid else {}),
                metadata=[MetadataItem(key="gym_row_extras", value={})],
            )
        ],
        target=GymRunnerTarget(
            source=GymAgentSource(component="simple_agent", config="config.yaml"), resources_server="mcqa"
        ),
    )
    if valid:
        await AgentEvalJob.to_spec(
            request, workspace="default", entity_client=None, async_sdk=_async_sdk(), is_local=True
        )
        resolver.assert_awaited_once()
    else:
        with pytest.raises(ValueError, match="missing inputs"):
            await AgentEvalJob.to_spec(
                request, workspace="default", entity_client=None, async_sdk=_async_sdk(), is_local=True
            )
        resolver.assert_not_awaited()


@pytest.mark.parametrize(
    "target",
    [
        None,
        ModelTarget(model=Model(url="http://model.test", name="test")),
        AgentTarget(agent=_agent()),
        FabricRunnerTarget(source=FabricConfigSource(config={})),
        HarborRunnerTarget(),
    ],
)
async def test_non_gym_submission_never_prepares_gym(monkeypatch, target: Target | None) -> None:
    async def forbidden(*args, **kwargs):
        pytest.fail("Non-Gym submission invoked Gym preparation")

    monkeypatch.setattr("nemo_evals.jobs.agent_evaluate.prepare_gym_submission", forbidden)
    request = AgentEvalInputSpec(
        tasks=[AgentEvalTaskInput(id="task", intent="Answer")],
        target=target,
        trials=[
            AgentEvalTrial(
                id="trial",
                task_id="task",
                status=AgentEvalTrialStatus.COMPLETED,
                output=AgentOutput(output_text="Answer"),
            )
        ]
        if target is None
        else None,
    )
    result = await AgentEvalJob.to_spec(
        request,
        workspace="default",
        entity_client=None,
        async_sdk=_async_sdk(),
        is_local=True,
    )
    assert isinstance(result, AgentEvalSpec)
    assert result.target == target


async def test_compile_registered_gym_agent_stages_its_package_before_evaluation(mocker: MockerFixture) -> None:
    """A registered agent stages even with no environment FileSet: the package it runs from is built by that step."""
    _patch_execution_profiles(mocker, [_kubernetes_profile_with_job_storage()])
    mocker.patch("nemo_evals.jobs.agent_compiler.config.gym_tasks_image", None)
    mocker.patch(
        "nemo_evals.jobs.agent_compiler.get_qualified_image",
        side_effect=lambda name: f"registry.example/{name}:test",
    )
    _enable_fileset_sandbox(mocker)
    target = GymRunnerTarget(
        source=RegisteredAgentSource(
            agent=AgentRef(root="dev/calc"), files=FilesetRef(root="dev/agent-files-0123abcd4567")
        ),
        resources_server="mcqa",
        resolved_config={
            "harness": {"adapter_id": "nvidia.fabric.langchain.deepagents"},
            "skills": {"paths": ["skills/a"]},
        },
    )

    compiled = await AgentEvalJob.compile(
        workspace="dev",
        spec=AgentEvalSpec(tasks=[_task_spec()], target=target),
        entity_client=object(),
        job_name=None,
        async_sdk=_async_sdk(),
    )

    stage, evaluate = HelixJobSpec.model_validate(compiled).steps
    assert (stage.name, evaluate.name) == ("stage-environment", "agent-evaluate")
    config = cast(dict[str, Any], stage.config)
    assert "environment" not in config and config["agent_files"] == "dev/agent-files-0123abcd4567"
    package = config["gym_registered_agent"]
    assert package["agent"] == "dev/calc" and package["resolved_config"] == target.resolved_config
    assert package["requirements"][0].startswith("nemo-fabric[deepagents,relay]==")
    assert "constraints" not in package  # the host image's pins ship with the plugin, not with the spec
    assert package["requirements"] == [package["requirements"][0]]  # the extra alone; the host lock pins companions
