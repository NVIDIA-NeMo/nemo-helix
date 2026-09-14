# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the agent-evaluation job (AALGO-297)."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from nemo_evaluator.api.schemas import MetadataItem, MetricInline, TaskInputs, TasksetRef
from nemo_evaluator.cli import EvaluatorPluginCLI
from nemo_evaluator.config import EvaluatorConfig
from nemo_evaluator.filesets import FilesetRef
from nemo_evaluator.jobs.agent_evaluate import (
    AGENT_BUNDLE_DIR,
    DEFAULT_RESULT_NAME,
    SUMMARY_RESULT_NAME,
    AgentEvalJob,
    _resolve_gym_environment,
    _to_runtime_task,
)
from nemo_evaluator.jobs.agent_spec import (
    AgentEvalInputSpec,
    AgentEvalSpec,
    AgentEvalTaskInput,
    AgentEvalTaskSpec,
    AgentTarget,
    FabricRunnerTarget,
    GymRunnerTarget,
    HarborRunnerTarget,
    ModelTarget,
    Target,
)
from nemo_evaluator.jobs.gym_sandbox import (
    GYM_SANDBOX_PLAN_ENVVAR,
    SandboxPlan,
    SandboxUnavailableError,
    SessionBackedGymRunner,
)
from nemo_evaluator.metric_refs import MetricRef
from nemo_evaluator.shared.metric_bundles.bundles import MetricBundle, bundle_metric
from nemo_evaluator.shared.metric_bundles.cloudpickle import CloudpickleMetricBundlePackager
from nemo_evaluator.tasks.agent_evaluate import main as agent_eval_task_main
from nemo_evaluator.tasks.runner import SDK_INITIALIZATION_EXIT_CODE
from nemo_evaluator_sdk.agent_eval.results import AgentEvalResult, AgentEvalSummary
from nemo_evaluator_sdk.agent_eval.runtimes.fabric.runtime import FabricAgentRuntime
from nemo_evaluator_sdk.agent_eval.runtimes.gym import GymAgentTaskRunner
from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import HarborAgentTaskRunner
from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalRunConfig, AgentEvalTask
from nemo_evaluator_sdk.agent_eval.trials import (
    AgentEvalTarget,
    AgentEvalTrial,
    AgentEvalTrialStatus,
    AgentOutput,
    TrialMeasurements,
)
from nemo_evaluator_sdk.enums import AgentFormat
from nemo_evaluator_sdk.metrics.exact_match import ExactMatchMetric
from nemo_evaluator_sdk.values import Agent, GenericAgent, Model, RunConfigOnline, RunConfigOnlineModel, SecretRef
from nemo_platform_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_platform_plugin.client.errors import InternalServerError, NemoResponseValidationError, NemoTransportError
from nemo_platform_plugin.commands import add_job_commands
from nemo_platform_plugin.files.types import FilesetPurpose
from nemo_platform_plugin.job_context import JobContext, StoragePaths
from nemo_platform_plugin.job_results import LocalJobResults
from nemo_platform_plugin.jobs.constants import PERSISTENT_JOB_STORAGE_PATH_ENVVAR
from nemo_platform_plugin.jobs.exceptions import PlatformJobCompilationError, PlatformJobDependencyUnavailableError
from nemo_platform_plugin.jobs.execution_profiles import (
    DockerJobExecutionProfile,
    DockerJobExecutionProfileConfig,
    KubernetesJobExecutionProfile,
    KubernetesJobExecutionProfileConfig,
    KubernetesJobStorageConfig,
    SubprocessJobExecutionProfile,
    VolcanoJobExecutionProfile,
    VolcanoJobExecutionProfileConfig,
)
from nemo_platform_plugin.jobs.providers import SubprocessExecutionProvider
from nemo_platform_plugin.jobs.spec import BaseExecutionProfile, PlatformJobSpec
from nemo_platform_plugin.scheduler import NemoJobScheduler
from nemo_platform_plugin.sdk import AsyncNeMoPlatform, NeMoPlatform
from pydantic import ValidationError
from pytest_mock import MockerFixture
from typer.testing import CliRunner


def _inline_metric() -> MetricInline:
    bundle = bundle_metric(
        ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.model_output}}"),
        CloudpickleMetricBundlePackager(),
    )
    return MetricInline.model_validate(bundle.model_dump(mode="json"))


def _task_inputs(**values: Any) -> TaskInputs:
    return TaskInputs.model_validate(values)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _task_spec() -> AgentEvalTaskSpec:
    return AgentEvalTaskSpec(
        id="task-1",
        intent="Answer the question.",
        inputs=_task_inputs(instruction="What is 2+2?"),
        metrics=[_inline_metric()],
    )


def _runner_target(model: str | None = None) -> FabricRunnerTarget:
    """A minimal agent-runner target, for tests about runners in general rather than a specific one."""
    return FabricRunnerTarget(
        config={"metadata": {"name": "a"}, "harness": {"adapter_id": "nvidia.fabric.codex"}}, model=model
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


def test_cli_agent_evaluate_still_exposes_run_during_stainless_migration() -> None:
    app = EvaluatorPluginCLI().get_cli()
    add_job_commands(app, {"evaluator.agent-evaluate": AgentEvalJob}, cli=EvaluatorPluginCLI())

    result = CliRunner().invoke(app, ["agent-evaluate", "--help"])

    assert result.exit_code == 0
    assert "run" in result.output
    assert "submit" in result.output
    assert "explain" in result.output


class _FakeEvaluator:
    """Stand-in for AgentEvaluator: records the tasks it was handed and returns canned trials."""

    def __init__(self) -> None:
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
                status=AgentEvalTrialStatus.COMPLETED,
                output=AgentOutput(output_text="4"),
            )
            for task in tasks
        ]
        return AgentEvalResult(
            run_id="run-1", tasks=list(tasks), trials=generated_trials, scores=[], summary=AgentEvalSummary()
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

    spec = await AgentEvalJob.to_spec(input_spec, workspace="dev", entity_client=None, async_sdk=None, is_local=True)
    assert isinstance(spec, AgentEvalSpec)
    assert spec.tasks[0].reference == reference
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
            agent="simple_agent",
            agent_config="responses_api_agents/simple_agent/configs/simple_agent.yaml",
            resources_server="mcqa",
        ),
    )

    spec = await AgentEvalJob.to_spec(input_spec, workspace="dev", entity_client=None, async_sdk=None, is_local=True)

    assert isinstance(spec, AgentEvalSpec)
    assert spec.tasks[0].inputs.model_dump(exclude_none=True)["gym_row"] == gym_row
    runtime_task = _to_runtime_task(spec.tasks[0])
    assert runtime_task.inputs["gym_row"] == gym_row
    assert runtime_task.metadata["gym_row_extras"] == gym_row_extras


def test_agent_eval_job_reconstructs_tasks_and_persists_bundle(tmp_path: Path, mocker: MockerFixture) -> None:
    fake = _FakeEvaluator()
    mocker.patch.object(AgentEvalJob, "_build_evaluator", return_value=fake)
    ctx = _job_context(tmp_path)

    spec = AgentEvalSpec(tasks=[_task_spec()], target=_runner_target("openai/gpt-5.4"))
    result = AgentEvalJob().run(spec.model_dump(), ctx=ctx)

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


def test_agent_eval_job_survives_result_persistence_failure(tmp_path: Path, mocker: MockerFixture) -> None:
    # The queryable result record is a best-effort convenience index; the authoritative output (bundle
    # + summary artifacts) is already saved. A persistence failure must not fail a successful eval.
    mocker.patch.object(AgentEvalJob, "_build_evaluator", return_value=_FakeEvaluator())
    persist = mocker.patch(
        "nemo_evaluator.jobs.agent_evaluate.persist_agent_eval_result",
        side_effect=RuntimeError("entity store unavailable"),
    )
    ctx = _job_context(tmp_path)

    spec = AgentEvalSpec(tasks=[_task_spec()], target=_runner_target("openai/gpt-5.4"))
    result = AgentEvalJob().run(spec.model_dump(), ctx=ctx)

    # Persistence was attempted and raised, yet the job still completed with its artifacts intact.
    persist.assert_called_once()
    assert result["status"] == "completed"
    assert result["artifact"]["name"] == DEFAULT_RESULT_NAME
    assert (ctx.storage.persistent / "results" / DEFAULT_RESULT_NAME).exists()


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
        config={"metadata": {"name": "a"}, "harness": {"adapter_id": "nvidia.fabric.codex"}},
        model="openai/gpt-5.4",
    )
    target, prompt_template, params = AgentEvalJob._resolve_target(fabric_target, ctx)
    assert isinstance(target, FabricAgentRuntime)
    assert target._model == "openai/gpt-5.4"
    assert target._work_root == ctx.storage.persistent / "fabric"
    # A runner shapes its own request, so it contributes no prompt template or inference params.
    assert prompt_template is None
    assert params is None


def test_resolve_target_builds_harbor_runtime_from_runner_target(tmp_path: Path) -> None:
    ctx = _job_context(tmp_path)
    harbor_target = HarborRunnerTarget(
        agent_name="oracle",
        agent_model_name="openai/gpt-5.4",
        agent_kwargs={"fabric_adapter_id": "nvidia.fabric.codex", "fabric_harness_settings": {"max_turns": 3}},
        n_attempts=2,
        n_concurrent_trials=8,
        max_retries=1,
        reward_key="score",
    )
    target, prompt_template, params = AgentEvalJob._resolve_target(harbor_target, ctx)
    assert isinstance(target, HarborAgentTaskRunner)
    # The runtime-only jobs directory is injected from the job's persistent storage.
    assert target._config is not None
    assert target._config.jobs_dir == ctx.storage.persistent / "harbor"
    # Spec knobs are forwarded onto the Harbor runtime config.
    assert target._config.agent_model_name == "openai/gpt-5.4"
    assert target._config.agent_kwargs == {
        "fabric_adapter_id": "nvidia.fabric.codex",
        "fabric_harness_settings": {"max_turns": 3},
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
        agent="simple_agent",
        agent_config="responses_api_agents/simple_agent/configs/simple_agent.yaml",
        resources_server="mcqa",
        num_repeats=2,
        concurrency=4,
        reward_key="score",
        hydra_params={"model": {"temperature": 0.7}},
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
    # A runner shapes its own request, so it contributes no prompt template or inference params.
    assert prompt_template is None
    assert params is None


def _sandbox_plan() -> SandboxPlan:
    return SandboxPlan(
        host_provider="opensandbox",
        runtime_image="registry.example.com/nmp-gym-runtime:1.0",
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
        agent="simple_agent",
        agent_config="responses_api_agents/simple_agent/configs/simple_agent.yaml",
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
        agent="simple_agent",
        agent_config="responses_api_agents/simple_agent/configs/simple_agent.yaml",
        resources_server="custom",
        environment=FilesetRef(root="dev/custom-environment"),
    )
    monkeypatch.delenv(GYM_SANDBOX_PLAN_ENVVAR, raising=False)

    with pytest.raises(SandboxUnavailableError, match="require sandboxed execution"):
        AgentEvalJob._resolve_target(gym_target, ctx)


def test_runner_target_is_accepted(tmp_path: Path) -> None:
    spec = AgentEvalSpec(tasks=[_task_spec()], target=_runner_target("openai/gpt-5.4"))
    assert isinstance(spec.target, FabricRunnerTarget)


def test_harbor_runner_target_is_accepted() -> None:
    spec = AgentEvalSpec(tasks=[_task_spec()], target=HarborRunnerTarget(agent_name="oracle"))
    assert isinstance(spec.target, HarborRunnerTarget)


def test_harbor_agent_kwargs_round_trip_the_wire_unchanged() -> None:
    """Nested kwargs survive JSON serialization, so what the submitter wrote is what the agent's ``__init__`` gets."""
    agent_kwargs = {
        "fabric_adapter_id": "nvidia.fabric.codex",
        "fabric_package": "nemo-fabric[codex]==0.3.0b1",
        "fabric_harness_settings": {"max_turns": 3, "tools": ["shell", None], "strict": True},
    }
    spec = AgentEvalSpec(
        tasks=[_task_spec()],
        target=HarborRunnerTarget(
            agent_import_path="nemo_fabric.integrations.harbor:FabricAgent", agent_kwargs=agent_kwargs
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
    from nemo_evaluator.jobs.evaluate import EvaluateJob
    from nemo_evaluator.service import AGENT_EVAL_JOB_SOURCE
    from nemo_platform_plugin.jobs.routes import _derive_service_name

    assert AGENT_EVAL_JOB_SOURCE != _derive_service_name(EvaluateJob)


def _async_sdk() -> AsyncNemoClient:
    return AsyncNemoClient(
        base_url="http://platform.test",
        workspace="default",
        http_client=AsyncMock(spec=httpx.AsyncClient),
    )


def _async_platform() -> AsyncNeMoPlatform:
    return AsyncNeMoPlatform(
        base_url="http://platform.test",
        workspace="default",
        http_client=AsyncMock(spec=httpx.AsyncClient),
    )


_SDK_IDENTITY_HEADERS = {
    "X-NMP-Principal-Id": "service:evaluator",
    "X-NMP-Principal-On-Behalf-Of": "user-1",
    "X-NMP-Principal-On-Behalf-Of-Email": "user@corp.test",  # PII - must stay in-platform
    "X-NMP-Internal": "true",
    "X-NMP-Trace-Id": "must-not-forward",  # non-identity X-NMP-* must be dropped
    "Authorization": "Bearer super-secret",  # bearer must never reach any endpoint
}
_FORWARDED_IDENTITY_HEADERS = {
    "X-NMP-Principal-Id": "service:evaluator",
    "X-NMP-Principal-On-Behalf-Of": "user-1",
    "X-NMP-Principal-On-Behalf-Of-Email": "user@corp.test",
    "X-NMP-Internal": "true",
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
    service principal id and on-behalf-of go through, but transport noise and other ``X-NMP-*`` (e.g.
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


def test_build_evaluator_without_platform_forwards_no_headers() -> None:
    target = _model_target("http://platform/apis/inference-gateway/v2/workspaces/default/model/m/-/v1/chat/completions")
    assert AgentEvalJob._build_evaluator(None, target).default_headers is None


def _sync_platform_with_identity() -> NeMoPlatform:
    return NeMoPlatform(
        base_url="http://platform",
        workspace="dev",
        default_headers=_SDK_IDENTITY_HEADERS,
        http_client=MagicMock(spec=httpx.Client),
    )


def _async_platform_with_identity() -> AsyncNeMoPlatform:
    return AsyncNeMoPlatform(
        base_url="http://platform",
        workspace="dev",
        default_headers=_SDK_IDENTITY_HEADERS,
        http_client=AsyncMock(spec=httpx.AsyncClient),
    )


def _capture_evaluator_headers(mocker: MockerFixture) -> dict[str, dict[str, str] | None]:
    """Swap in a fake ``AgentEvaluator`` that records the headers ``_build_evaluator`` computed."""
    captured: dict[str, dict[str, str] | None] = {}

    def _factory(*, default_headers: dict[str, str] | None = None) -> _FakeEvaluator:
        captured["default_headers"] = default_headers
        return _FakeEvaluator()

    mocker.patch("nemo_evaluator.jobs.agent_evaluate.AgentEvaluator", _factory)
    return captured


@pytest.mark.parametrize("inject_async", [False, True], ids=["sdk", "async_sdk"])
def test_run_accepts_the_generated_sdk_the_local_cli_injects(
    tmp_path: Path, mocker: MockerFixture, inject_async: bool
) -> None:
    """A local ``nemo evaluator agent-evaluate run`` is handed a generated ``NeMoPlatform``, not a
    typed client, and every platform call in ``run`` is typed-client-only. Without adaptation the job
    dies with ``AttributeError: 'AsyncNeMoPlatform' object has no attribute 'is_platform_url'``
    before it issues a single target request.

    Adapting must not widen what reaches the target: the allowlist still applies, so the bearer and
    the trace header the generated SDK carried stay behind.
    """
    captured = _capture_evaluator_headers(mocker)
    config = AgentEvalSpec(
        tasks=[_task_spec()],
        target=_model_target(
            "http://platform/apis/inference-gateway/v2/workspaces/default/model/m/-/v1/chat/completions"
        ),
    ).model_dump()
    ctx = _job_context(tmp_path)

    if inject_async:
        result = AgentEvalJob().run(config, ctx=ctx, async_sdk=_async_platform_with_identity())
    else:
        result = AgentEvalJob().run(config, ctx=ctx, sdk=_sync_platform_with_identity())

    assert result["status"] == "completed"
    assert captured["default_headers"] == _FORWARDED_IDENTITY_HEADERS


def test_run_sends_no_identity_to_a_third_party_target_via_generated_sdk(tmp_path: Path, mocker: MockerFixture) -> None:
    """The same-origin guard is what keeps the delegated user's email and groups inside the platform.
    Adaptation reconstructs the client the guard compares against, so a regression here would leak
    that PII to whatever endpoint the submitter named."""
    captured = _capture_evaluator_headers(mocker)
    spec = AgentEvalSpec(tasks=[_task_spec()], target=_model_target("https://api.openai.com/v1/chat/completions"))

    result = AgentEvalJob().run(
        spec.model_dump(), ctx=_job_context(tmp_path), async_sdk=_async_platform_with_identity()
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
    assert isinstance(spec.tasks[0].metrics[0], MetricRef)


def test_input_spec_accepts_a_taskset_reference() -> None:
    spec = AgentEvalInputSpec(tasks=TasksetRef("default/geo-suite"), target=_runner_target("openai/gpt-5.4"))
    assert isinstance(spec.tasks, TasksetRef)
    assert spec.tasks.root == "default/geo-suite"
    # A JSON string round-trips back to the TasksetRef arm of the union, not a list.
    assert isinstance(AgentEvalInputSpec.model_validate_json(spec.model_dump_json()).tasks, TasksetRef)


def test_input_spec_rejects_empty_inline_task_list() -> None:
    with pytest.raises(ValueError, match="at least one task"):
        AgentEvalInputSpec(tasks=[], target=_runner_target("openai/gpt-5.4"))


async def test_to_spec_resolves_inline_task_metrics_without_a_platform() -> None:
    # Inline metrics need no entity client/SDK; refs would, but none are used here.
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

    spec = await AgentEvalJob.to_spec(input_spec, workspace="dev", entity_client=None, async_sdk=None, is_local=True)

    assert isinstance(spec, AgentEvalSpec)
    assert len(spec.tasks) == 1
    assert isinstance(spec.tasks[0].metrics[0], MetricInline)
    # Canonical metrics reconstruct to runtime instances.
    assert isinstance(_to_runtime_task(spec.tasks[0]).metrics[0], ExactMatchMetric)


async def test_to_spec_requires_platform_to_resolve_a_metric_reference() -> None:
    # A stored MetricRef can only be loaded with an entity store + SDK; without one, to_spec must
    # fail loudly rather than silently drop the metric.
    input_spec = AgentEvalInputSpec(
        tasks=[
            AgentEvalTaskInput(id="task-1", intent="Answer.", inputs=TaskInputs(), metrics=[MetricRef("stored-metric")])
        ],
        target=_runner_target("openai/gpt-5.4"),
    )
    with pytest.raises(ValueError, match="platform connection"):
        await AgentEvalJob.to_spec(input_spec, workspace="dev", entity_client=None, async_sdk=None, is_local=True)


# --- compile: the submit/service-side path ----------------------------------


def _assert_agent_eval_step_entrypoint(
    job_spec: PlatformJobSpec,
    *,
    expected_image: str | None = None,
    expected_entrypoint: Sequence[str] = ("python", "-m"),
) -> None:
    step = job_spec.steps[0]
    container = cast(Any, step.executor).container
    if expected_image is not None:
        assert container.image == expected_image
    assert container.entrypoint == list(expected_entrypoint)
    assert container.command == ["nemo_evaluator.tasks.agent_evaluate"]


async def test_checked_fabric_spec_transforms_and_compiles() -> None:
    path = _repo_root() / "skills/nemo-evaluator-plugin/assets/specs/fabric_agent_eval.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    input_spec = AgentEvalInputSpec.model_validate(payload)

    spec = await AgentEvalJob.to_spec(
        input_spec,
        workspace="default",
        entity_client=None,
        async_sdk=None,
        is_local=False,
    )

    assert isinstance(spec, AgentEvalSpec)
    assert isinstance(spec.tasks, list)
    bundle = MetricBundle.model_validate(spec.tasks[0].metrics[0].model_dump(mode="json"))
    assert bundle.payload.kind == "inline"

    compiled = await AgentEvalJob.compile(
        workspace="default",
        spec=spec,
        entity_client=None,
        job_name=None,
        async_sdk=None,
    )
    job_spec = PlatformJobSpec.model_validate(compiled)
    _assert_agent_eval_step_entrypoint(job_spec)
    config = cast(dict[str, Any], job_spec.steps[0].config)
    assert config["target"]["kind"] == "fabric"
    assert config["tasks"][0]["metrics"][0]["payload"]["kind"] == "inline"


def _patch_execution_profiles(mocker: MockerFixture, profiles: list[BaseExecutionProfile]) -> None:
    response = mocker.Mock()
    response.data.return_value = profiles
    jobs_client = mocker.Mock()
    jobs_client.get_execution_profiles = mocker.AsyncMock(return_value=response)
    mocker.patch("nemo_evaluator.jobs.agent_evaluate.client_from_platform", return_value=jobs_client)


async def _compile_harbor(*, async_sdk: AsyncNeMoPlatform | None, profile: str | None = None) -> PlatformJobSpec:
    """Compile the minimal Harbor submission every backend-guard test makes."""
    compiled = await AgentEvalJob.compile(
        workspace="default",
        spec=AgentEvalSpec(tasks=[_task_spec()], target=HarborRunnerTarget(agent_name="oracle")),
        entity_client=object(),
        job_name=None,
        async_sdk=async_sdk,
        profile=profile,
    )
    return PlatformJobSpec.model_validate(compiled)


@pytest.mark.parametrize(
    ("target", "expected_kind", "expected_endpoint_name", "expected_image_name", "expected_entrypoint"),
    [
        (
            FabricRunnerTarget(config={"metadata": {"name": "a"}, "harness": {"adapter_id": "nvidia.fabric.codex"}}),
            "fabric",
            None,
            "nmp-cpu-tasks",
            ("python", "-m"),
        ),
        (
            GymRunnerTarget(
                agent="simple_agent",
                agent_config="responses_api_agents/simple_agent/configs/simple_agent.yaml",
                resources_server="mcqa",
            ),
            "gym",
            None,
            "nmp-gym-tasks",
            ("/app/.venv/bin/python", "-m"),
        ),
        (
            ModelTarget(
                model=Model(url="http://model.test/v1/chat/completions", name="test-model"),
                params=RunConfigOnlineModel(),
            ),
            "model",
            "test-model",
            "nmp-cpu-tasks",
            ("python", "-m"),
        ),
        (
            AgentTarget(agent=_agent(), params=RunConfigOnline()),
            "agent",
            "test-agent",
            "nmp-cpu-tasks",
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
    mocker.patch("nemo_evaluator.jobs.agent_compiler.config.gym_tasks_image", None)
    mocker.patch(
        "nemo_evaluator.jobs.agent_compiler.get_qualified_image",
        side_effect=lambda name: f"registry.example/{name}:test",
    )
    spec = AgentEvalSpec(tasks=[_task_spec()], target=target)

    compiled = await AgentEvalJob.compile(
        workspace="default", spec=spec, entity_client=object(), job_name=None, async_sdk=None
    )

    job_spec = PlatformJobSpec.model_validate(compiled)
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
    image = "ghcr.io/nvidia-nemo/nemo-platform/nmp-gym-tasks:internal-test"
    mocker.patch("nemo_evaluator.jobs.agent_compiler.config.gym_tasks_image", image)
    qualify = mocker.patch("nemo_evaluator.jobs.agent_compiler.get_qualified_image")
    spec = AgentEvalSpec(
        tasks=[_task_spec()],
        target=GymRunnerTarget(
            agent="simple_agent",
            agent_config="responses_api_agents/simple_agent/configs/simple_agent.yaml",
            resources_server="mcqa",
        ),
    )

    compiled = await AgentEvalJob.compile(
        workspace="default", spec=spec, entity_client=object(), job_name=None, async_sdk=None
    )

    job_spec = PlatformJobSpec.model_validate(compiled)
    _assert_agent_eval_step_entrypoint(
        job_spec,
        expected_image=image,
        expected_entrypoint=("/app/.venv/bin/python", "-m"),
    )
    qualify.assert_not_called()


def _gym_environment_target() -> GymRunnerTarget:
    return GymRunnerTarget(
        environment=FilesetRef(root="dev/custom-gym"),
        agent="custom_agent",
        agent_config="responses_api_agents/custom_agent/configs/custom_agent.yaml",
        resources_server="custom_resources",
    )


def _enable_fileset_sandbox(mocker: MockerFixture) -> None:
    mocker.patch(
        "nemo_evaluator.jobs.agent_evaluate.require_fileset_environment_sandboxed",
        return_value=None,
    )
    mocker.patch(
        "nemo_evaluator.jobs.agent_evaluate.require_fileset_sandbox_storage_identity",
        return_value=None,
    )
    mocker.patch("nemo_evaluator.jobs.agent_compiler.config.sandboxed_gym_default", True)
    mocker.patch("nemo_evaluator.jobs.agent_compiler.config.sandbox_cluster_capable", True)
    mocker.patch(
        "nemo_evaluator.jobs.agent_compiler.config.sandbox_runtime_image",
        "registry.example.com/nmp-gym-runtime:1.0",
    )
    mocker.patch(
        "nemo_evaluator.jobs.agent_compiler.config.sandbox_job_storage_pvc_claim",
        "job-storage",
    )
    mocker.patch(
        "nemo_evaluator.jobs.agent_compiler.config.sandbox_policy_base_urls",
        ("https://integrate.api.nvidia.com/v1",),
    )


async def test_compile_gym_environment_adds_staging_step_before_evaluation(mocker: MockerFixture) -> None:
    mocker.patch("nemo_evaluator.jobs.agent_compiler.config.gym_tasks_image", None)
    mocker.patch(
        "nemo_evaluator.jobs.agent_compiler.get_qualified_image",
        side_effect=lambda name: f"registry.example/{name}:test",
    )
    _enable_fileset_sandbox(mocker)
    spec = AgentEvalSpec(tasks=[_task_spec()], target=_gym_environment_target())

    compiled = await AgentEvalJob.compile(
        workspace="dev",
        spec=spec,
        entity_client=object(),
        job_name=None,
        async_sdk=None,
    )

    job_spec = PlatformJobSpec.model_validate(compiled)
    assert [step.name for step in job_spec.steps] == ["stage-environment", "agent-evaluate"]
    stage, evaluate = job_spec.steps
    stage_container = cast(Any, stage.executor).container
    assert stage_container.image == "registry.example/nmp-cpu-tasks:test"
    assert stage_container.entrypoint == ["python", "-m"]
    assert stage_container.command == ["nemo_evaluator.tasks.stage_environment"]
    assert stage.config == {"environment": "dev/custom-gym"}
    evaluate_container = cast(Any, evaluate.executor).container
    assert evaluate_container.image == "registry.example/nmp-cpu-tasks:test"
    assert evaluate_container.entrypoint == ["python", "-m"]
    assert evaluate_container.command == ["nemo_evaluator.tasks.agent_evaluate"]
    evaluate_config = cast(dict[str, Any], evaluate.config)
    assert evaluate_config["target"]["environment"] == "dev/custom-gym"


async def test_compile_sandboxed_gym_uses_cpu_tasks_and_ignores_colocated_image_override(
    mocker: MockerFixture,
) -> None:
    mocker.patch(
        "nemo_evaluator.jobs.agent_compiler.config.gym_tasks_image",
        "registry.example/nmp-gym-tasks:colocated-only",
    )
    mocker.patch(
        "nemo_evaluator.jobs.agent_compiler.get_qualified_image",
        side_effect=lambda name: f"registry.example/{name}:test",
    )
    _enable_fileset_sandbox(mocker)
    spec = AgentEvalSpec(
        tasks=[_task_spec()],
        target=GymRunnerTarget(
            agent="simple_agent",
            agent_config="responses_api_agents/simple_agent/configs/simple_agent.yaml",
            resources_server="mcqa",
        ),
    )

    compiled = await AgentEvalJob.compile(
        workspace="dev",
        spec=spec,
        entity_client=object(),
        job_name=None,
        async_sdk=None,
    )

    job_spec = PlatformJobSpec.model_validate(compiled)
    _assert_agent_eval_step_entrypoint(
        job_spec,
        expected_image="registry.example/nmp-cpu-tasks:test",
        expected_entrypoint=("python", "-m"),
    )


async def test_compile_gym_environment_propagates_platform_sandbox_protocol(mocker: MockerFixture) -> None:
    mocker.patch(
        "nemo_evaluator.jobs.agent_compiler.get_qualified_image",
        return_value="registry.example/nmp-gym-tasks:test",
    )
    _enable_fileset_sandbox(mocker)
    mocker.patch("nemo_evaluator.jobs.agent_compiler.platform_config.sandbox_server_protocol", "http")

    compiled = await AgentEvalJob.compile(
        workspace="dev",
        spec=AgentEvalSpec(tasks=[_task_spec()], target=_gym_environment_target()),
        entity_client=object(),
        job_name=None,
        async_sdk=None,
    )

    evaluate = PlatformJobSpec.model_validate(compiled).steps[-1]
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
        "nemo_evaluator.jobs.agent_compiler.get_qualified_image",
        return_value="registry.example/nmp-gym-tasks:test",
    )
    _enable_fileset_sandbox(mocker)
    mocker.patch("nemo_evaluator.jobs.agent_compiler.config.sandbox_host_provider", "docker")
    evaluator_config = mocker.Mock()
    evaluator_config.sandbox_host_provider = "docker"
    mocker.patch(
        "nemo_evaluator.jobs.agent_evaluate.get_config",
        return_value=evaluator_config,
    )
    _patch_execution_profiles(mocker, [SubprocessJobExecutionProfile()])
    spec = AgentEvalSpec(tasks=[_task_spec()], target=_gym_environment_target())

    compiled = await AgentEvalJob.compile(
        workspace="dev",
        spec=spec,
        entity_client=object(),
        job_name=None,
        async_sdk=_async_platform(),
    )

    job_spec = PlatformJobSpec.model_validate(compiled)
    assert [step.name for step in job_spec.steps] == ["stage-environment", "agent-evaluate"]

    stage, evaluate = job_spec.steps
    assert isinstance(stage.executor, SubprocessExecutionProvider)
    assert stage.executor.command == ["python", "-m", "nemo_evaluator.tasks.stage_environment"]
    assert isinstance(evaluate.executor, SubprocessExecutionProvider)
    assert evaluate.executor.command == ["python", "-m", "nemo_evaluator.tasks.agent_evaluate"]


async def test_compile_rejects_fileset_environment_when_sandboxing_is_disabled(mocker: MockerFixture) -> None:
    mocker.patch("nemo_evaluator.jobs.agent_compiler.config.gym_tasks_image", None)
    mocker.patch(
        "nemo_evaluator.jobs.agent_compiler.get_qualified_image",
        return_value="registry.example/nmp-gym-tasks:test",
    )

    with pytest.raises(PlatformJobCompilationError, match="require sandboxed execution"):
        await AgentEvalJob.compile(
            workspace="dev",
            spec=AgentEvalSpec(tasks=[_task_spec()], target=_gym_environment_target()),
            entity_client=object(),
            job_name=None,
            async_sdk=None,
        )


async def test_compile_rejects_mismatched_job_and_sandbox_storage_pvcs(mocker: MockerFixture) -> None:
    mocker.patch("nemo_evaluator.jobs.agent_compiler.config.gym_tasks_image", None)
    mocker.patch(
        "nemo_evaluator.jobs.agent_compiler.get_qualified_image",
        return_value="registry.example/nmp-gym-tasks:test",
    )
    mocker.patch(
        "nemo_evaluator.jobs.agent_evaluate.require_fileset_environment_sandboxed",
        return_value=None,
    )
    mocker.patch(
        "nemo_evaluator.jobs.agent_evaluate.get_config",
        return_value=EvaluatorConfig(
            sandboxed_gym_default=True,
            sandbox_cluster_capable=True,
            sandbox_runtime_image="registry.example.com/nmp-gym-runtime:1.0",
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

    with pytest.raises(PlatformJobCompilationError, match="sandbox_job_storage_pvc_claim"):
        await AgentEvalJob.compile(
            workspace="dev",
            spec=AgentEvalSpec(tasks=[_task_spec()], target=_gym_environment_target()),
            entity_client=object(),
            job_name=None,
            async_sdk=_async_platform(),
        )


async def test_compile_prefers_kubernetes_profile_for_opensandbox_fileset(mocker: MockerFixture) -> None:
    mocker.patch("nemo_evaluator.jobs.agent_compiler.config.gym_tasks_image", None)
    mocker.patch(
        "nemo_evaluator.jobs.agent_compiler.get_qualified_image",
        return_value="registry.example/nmp-gym-tasks:test",
    )
    _enable_fileset_sandbox(mocker)
    mocker.patch("nemo_evaluator.jobs.agent_compiler.config.sandbox_host_provider", "opensandbox")
    mocker.patch("nemo_evaluator.jobs.agent_compiler.config.sandbox_job_storage_pvc_claim", "jobs-pvc")
    evaluator_config = mocker.Mock()
    evaluator_config.sandboxed_gym_default = True
    evaluator_config.sandbox_host_provider = "opensandbox"
    evaluator_config.sandbox_job_storage_pvc_claim = "jobs-pvc"
    mocker.patch("nemo_evaluator.jobs.agent_evaluate.get_config", return_value=evaluator_config)
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
        async_sdk=_async_platform(),
    )

    job_spec = PlatformJobSpec.model_validate(compiled)
    assert all(not isinstance(step.executor, SubprocessExecutionProvider) for step in job_spec.steps)


async def test_compile_marks_gym_profile_transport_failure_as_retryable(mocker: MockerFixture) -> None:
    request = httpx.Request("GET", "http://jobs.test/v2/execution-profiles")
    jobs_client = mocker.Mock()
    jobs_client.get_execution_profiles = mocker.AsyncMock(
        side_effect=NemoTransportError(httpx.ConnectError("profiles unavailable", request=request))
    )
    mocker.patch("nemo_evaluator.jobs.agent_evaluate.client_from_platform", return_value=jobs_client)
    spec = AgentEvalSpec(tasks=[_task_spec()], target=_gym_environment_target())

    with pytest.raises(PlatformJobDependencyUnavailableError, match="Jobs service is temporarily unavailable"):
        await AgentEvalJob.compile(
            workspace="dev",
            spec=spec,
            entity_client=object(),
            job_name=None,
            async_sdk=_async_platform(),
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
    mocker.patch("nemo_evaluator.jobs.agent_evaluate.client_from_platform", return_value=files)
    target = GymRunnerTarget(
        environment=FilesetRef(root="custom-gym"),
        agent="custom_agent",
        agent_config="responses_api_agents/custom_agent/configs/custom_agent.yaml",
        resources_server="custom_resources",
    )

    resolved = await _resolve_gym_environment(
        target,
        workspace="dev",
        async_sdk=_async_platform(),
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
    mocker.patch("nemo_evaluator.jobs.agent_evaluate.client_from_platform", return_value=files)

    resolved = await _resolve_gym_environment(
        _gym_environment_target(),
        workspace="dev",
        async_sdk=_async_platform(),
    )

    assert isinstance(resolved, GymRunnerTarget)
    assert resolved.environment == FilesetRef(root="dev/custom-gym")


async def test_resolve_gym_environment_rejects_wrong_purpose(mocker: MockerFixture) -> None:
    response = mocker.Mock()
    response.data.return_value = SimpleNamespace(purpose=FilesetPurpose.DATASET)
    files = mocker.Mock()
    files.get_fileset = mocker.AsyncMock(return_value=response)
    mocker.patch("nemo_evaluator.jobs.agent_evaluate.client_from_platform", return_value=files)
    target = GymRunnerTarget(
        environment=FilesetRef(root="dev/not-an-environment"),
        agent="custom_agent",
        agent_config="responses_api_agents/custom_agent/configs/custom_agent.yaml",
        resources_server="custom_resources",
    )

    with pytest.raises(ValueError, match="expected 'environment'"):
        await _resolve_gym_environment(
            target,
            workspace="dev",
            async_sdk=_async_platform(),
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
    mocker.patch("nemo_evaluator.jobs.agent_evaluate.client_from_platform", return_value=files)

    with pytest.raises(ValueError, match="has no nemo-environment.yaml at its root"):
        await _resolve_gym_environment(
            _gym_environment_target(),
            workspace="dev",
            async_sdk=_async_platform(),
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
    mocker.patch("nemo_evaluator.jobs.agent_evaluate.client_from_platform", return_value=files)

    with pytest.raises(ValueError, match="config_paths reference files that are not in the package"):
        await _resolve_gym_environment(
            _gym_environment_target(),
            workspace="dev",
            async_sdk=_async_platform(),
        )


async def test_compile_rejects_harbor_target_for_docker_profile(mocker: MockerFixture) -> None:
    """The full rejection message: the resolved backend plus the standing Harbor requirement."""
    _patch_execution_profiles(
        mocker,
        [DockerJobExecutionProfile(provider="cpu", profile="default", config=DockerJobExecutionProfileConfig())],
    )

    with pytest.raises(PlatformJobCompilationError) as exc_info:
        await _compile_harbor(async_sdk=_async_platform())

    message = str(exc_info.value)
    assert "profile 'default'" in message
    assert "backend 'docker'" in message
    assert "Harbor targets currently require local execution or the subprocess backend" in message


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

    with pytest.raises(PlatformJobCompilationError, match=rf"backend '{backend}'"):
        await _compile_harbor(async_sdk=_async_platform())


async def test_compile_routes_harbor_directly_to_advertised_subprocess_profile(mocker: MockerFixture) -> None:
    """An advertised default subprocess backend must be selected, not merely inferred as a translation."""
    _patch_execution_profiles(
        mocker,
        [
            DockerJobExecutionProfile(provider="cpu", profile="default", config=DockerJobExecutionProfileConfig()),
            SubprocessJobExecutionProfile(),
        ],
    )

    job_spec = await _compile_harbor(async_sdk=_async_platform())

    executor = job_spec.steps[0].executor
    assert isinstance(executor, SubprocessExecutionProvider)
    assert executor.profile == "default"
    assert executor.command == ["python", "-m", "nemo_evaluator.tasks.agent_evaluate"]
    assert cast(dict[str, Any], job_spec.steps[0].config)["target"]["kind"] == "harbor"


async def test_compile_rejects_harbor_when_profile_is_missing(mocker: MockerFixture) -> None:
    _patch_execution_profiles(mocker, [SubprocessJobExecutionProfile.model_validate({"profile": "other"})])

    with pytest.raises(PlatformJobCompilationError, match="profile 'default'.*does not resolve"):
        await _compile_harbor(async_sdk=_async_platform())


async def test_compile_marks_missing_platform_sdk_as_dependency_unavailable() -> None:
    with pytest.raises(PlatformJobDependencyUnavailableError, match="Jobs service is temporarily unavailable"):
        await _compile_harbor(async_sdk=None)


async def test_compile_marks_profile_transport_failure_as_retryable(mocker: MockerFixture) -> None:
    request = httpx.Request("GET", "http://jobs.test/v2/execution-profiles")
    jobs_client = mocker.Mock()
    jobs_client.get_execution_profiles = mocker.AsyncMock(
        side_effect=NemoTransportError(httpx.ConnectError("profiles unavailable", request=request))
    )
    mocker.patch("nemo_evaluator.jobs.agent_evaluate.client_from_platform", return_value=jobs_client)

    with pytest.raises(PlatformJobDependencyUnavailableError, match="Jobs service is temporarily unavailable"):
        await _compile_harbor(async_sdk=_async_platform())


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
    mocker.patch("nemo_evaluator.jobs.agent_evaluate.client_from_platform", return_value=jobs_client)

    with pytest.raises(PlatformJobDependencyUnavailableError, match="Jobs service is temporarily unavailable"):
        await _compile_harbor(async_sdk=_async_platform())


async def test_compile_does_not_classify_unexpected_profile_failure_as_invalid(mocker: MockerFixture) -> None:
    jobs_client = mocker.Mock()
    jobs_client.get_execution_profiles = mocker.AsyncMock(side_effect=RuntimeError("unexpected lookup bug"))
    mocker.patch("nemo_evaluator.jobs.agent_evaluate.client_from_platform", return_value=jobs_client)

    with pytest.raises(RuntimeError, match="unexpected lookup bug"):
        await _compile_harbor(async_sdk=_async_platform())


async def test_compile_non_harbor_target_does_not_resolve_execution_profiles(mocker: MockerFixture) -> None:
    mocker.patch(
        "nemo_evaluator.jobs.agent_evaluate.client_from_platform",
        side_effect=AssertionError("non-Harbor compilation must not query execution profiles"),
    )
    spec = AgentEvalSpec(tasks=[_task_spec()], target=_runner_target("openai/gpt-5.4"))

    compiled = await AgentEvalJob.compile(
        workspace="default",
        spec=spec,
        entity_client=object(),
        job_name=None,
        async_sdk=None,
    )

    assert cast(dict[str, Any], PlatformJobSpec.model_validate(compiled).steps[0].config)["target"]["kind"] == "fabric"


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
        workspace="default", spec=spec, entity_client=object(), job_name=None, async_sdk=None
    )

    step = PlatformJobSpec.model_validate(compiled).steps[0]
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
            workspace="default", spec=spec, entity_client=object(), job_name=None, async_sdk=None
        )


# --- run_local: the in-process run path, across target types ----------------


@pytest.mark.parametrize(
    "target",
    [
        ModelTarget(
            model=Model(url="http://model.test/v1/chat/completions", name="test-model"), params=RunConfigOnlineModel()
        ),
        AgentTarget(agent=_agent(), params=RunConfigOnline()),
        _runner_target("openai/gpt-5.4"),
        HarborRunnerTarget(agent_name="oracle"),
        GymRunnerTarget(
            agent="simple_agent",
            agent_config="responses_api_agents/simple_agent/configs/simple_agent.yaml",
            resources_server="mcqa",
        ),
    ],
)
def test_run_local_executes_each_target_type(target: Target, mocker: MockerFixture) -> None:
    # run_local drives the full validate -> to_spec -> run path. The evaluator is faked so the target
    # is threaded (a runner resolved to its runtime, an endpoint passed through) without real inference.
    fake = _FakeEvaluator()
    mocker.patch.object(AgentEvalJob, "_build_evaluator", return_value=fake)
    input_spec = AgentEvalInputSpec(
        tasks=[
            AgentEvalTaskInput(
                id="task-1",
                intent="Answer.",
                inputs=_task_inputs(instruction="What is 2+2?"),
                metrics=[_inline_metric()],
            )
        ],
        target=target,
    )

    result = NemoJobScheduler().run_local(AgentEvalJob, input_spec.model_dump(mode="json"))

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


def test_run_local_scores_precomputed_trials_offline(mocker: MockerFixture) -> None:
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

    result = NemoJobScheduler().run_local(AgentEvalJob, input_spec.model_dump(mode="json"))

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

    def test_main_dispatches_agent_eval_job_with_task_sdk(self, mocker: MockerFixture) -> None:
        sdk = NeMoPlatform(base_url="http://platform.test", workspace="default")
        client = NemoClient(
            base_url="http://platform.test", workspace="default", http_client=MagicMock(spec=httpx.Client)
        )
        async_client = AsyncNemoClient(
            base_url="http://platform.test", workspace="default", http_client=AsyncMock(spec=httpx.AsyncClient)
        )
        ctx = MagicMock()
        get_platform_sdk = mocker.patch("nemo_evaluator.tasks.runner.get_task_sdk", return_value=sdk)
        build_ctx = mocker.patch("nemo_evaluator.tasks.runner.build_ctx_from_env", return_value=ctx)
        get_task_client = mocker.patch("nemo_evaluator.tasks.runner.get_task_nemo_client", return_value=client)
        get_async_task_client = mocker.patch(
            "nemo_evaluator.tasks.runner.get_async_task_nemo_client", return_value=async_client
        )
        run_task = mocker.patch("nemo_evaluator.tasks.runner.run_task", return_value=0)

        exit_code = agent_eval_task_main()

        assert exit_code == 0
        get_platform_sdk.assert_called_once_with("evaluator")
        build_ctx.assert_called_once_with(sdk)
        get_task_client.assert_called_once_with("evaluator")
        get_async_task_client.assert_called_once_with("evaluator")
        run_task.assert_called_once_with(AgentEvalJob, sdk=client, async_sdk=async_client, ctx=ctx)

    def test_main_returns_setup_exit_code_when_task_sdk_fails(self, mocker: MockerFixture) -> None:
        get_platform_sdk = mocker.patch("nemo_evaluator.tasks.runner.get_task_sdk", side_effect=RuntimeError("boom"))
        get_task_client = mocker.patch("nemo_evaluator.tasks.runner.get_task_nemo_client")
        run_task = mocker.patch("nemo_evaluator.tasks.runner.run_task")

        exit_code = agent_eval_task_main()

        assert exit_code == SDK_INITIALIZATION_EXIT_CODE
        get_platform_sdk.assert_called_once_with("evaluator")
        get_task_client.assert_not_called()
        run_task.assert_not_called()


async def test_trial_error_survives_the_job_spec_wire_contract() -> None:
    """AALGO-428: ``AgentEvalTrial.error`` is public API, not just an SDK-internal field.

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

    spec = await AgentEvalJob.to_spec(input_spec, workspace="dev", entity_client=None, async_sdk=None, is_local=True)
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
            "tasks": [_task_spec().model_dump(mode="json")],
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
                "tasks": [_task_spec().model_dump(mode="json")],
            }
        )


async def test_compile_resolves_gym_runner_env_secrets() -> None:
    """A Gym environment's model key reaches it through the OS environment, not an endpoint spec.

    Without this the only route was `env_vars`, which stores the credential in plaintext on the spec
    and writes it into whatever persists that spec.
    """
    spec = AgentEvalSpec(
        tasks=[_task_spec()],
        target=GymRunnerTarget(
            agent="simple_agent",
            agent_config="responses_api_agents/simple_agent/configs/simple_agent.yaml",
            resources_server="mcqa",
            env_vars={"WMT_TRANSLATION_COMET_PY_CACHE": "/shared/cache"},
            env_secrets={"OPENAI_API_KEY": SecretRef(root="my-workspace/openai-key")},
        ),
    )

    compiled = await AgentEvalJob.compile(
        workspace="default", spec=spec, entity_client=object(), job_name=None, async_sdk=None
    )

    step = PlatformJobSpec.model_validate(compiled).steps[0]
    secrets = {env.name: env.from_secret.name for env in step.environment or [] if env.from_secret}
    assert secrets == {"OPENAI_API_KEY": "my-workspace/openai-key"}
    # The reference travels; the value never does.
    stored_target = cast(dict[str, Any], step.config)["target"]
    assert stored_target["env_secrets"] == {"OPENAI_API_KEY": "my-workspace/openai-key"}
    assert stored_target["env_vars"] == {"WMT_TRANSLATION_COMET_PY_CACHE": "/shared/cache"}
