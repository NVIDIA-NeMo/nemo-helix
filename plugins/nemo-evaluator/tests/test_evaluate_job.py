# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the evaluator plugin's SDK-backed job."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, cast
from unittest.mock import AsyncMock, MagicMock

import httpx
import nemo_evaluator.cli as evaluator_cli
import pytest
from models import ResolvedModelReference
from nemo_evaluator.cli import EvaluatorPluginCLI
from nemo_evaluator.filesets import FilesetRef
from nemo_evaluator.jobs.evaluate import (
    AGGREGATE_SCORES_RESULT_NAME,
    ARTIFACTS_RESULT_NAME,
    DEFAULT_FILE_NAME,
    DEFAULT_RESULT_NAME,
    ROW_SCORES_RESULT_NAME,
    EvaluateInputSpec,
    EvaluateJob,
    EvaluateSpec,
)
from nemo_evaluator.jobs.metric_resolution import to_runtime_bundle
from nemo_evaluator.shared.metric_bundles.bundles import (
    MetricBundle,
    MetricBundlePackager,
    MetricBundlePayload,
    bundle_metric,
    register_metric_bundle_kind,
    unbundle_metric,
)
from nemo_evaluator.shared.metric_bundles.cloudpickle import CloudpickleMetricBundlePackager
from nemo_evaluator.tasks.evaluate import main as evaluate_task_main
from nemo_evaluator.tasks.runner import SDK_INITIALIZATION_EXIT_CODE
from nemo_evaluator_sdk.enums import AgentFormat
from nemo_evaluator_sdk.metrics.exact_match import ExactMatchMetric
from nemo_evaluator_sdk.metrics.f1 import F1Metric
from nemo_evaluator_sdk.metrics.llm_judge import LLMJudgeMetric
from nemo_evaluator_sdk.metrics.protocol import Metric, MetricInput, MetricOutput, MetricOutputSpec, MetricResult
from nemo_evaluator_sdk.values import (
    Agent,
    AggregatedMetricResult,
    EvaluationResult,
    GenericAgent,
    Model,
    RunConfig,
    RunConfigOnline,
    RunConfigOnlineModel,
    SecretRef,
)
from nemo_evaluator_sdk.values.models import ModelRef
from nemo_evaluator_sdk.values.scores import JSONScoreParser, RangeScore
from nemo_platform_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_platform_plugin.commands import add_job_commands
from nemo_platform_plugin.job_context import JobContext, StoragePaths
from nemo_platform_plugin.job_results import LocalJobResults
from nemo_platform_plugin.jobs.constants import PERSISTENT_JOB_STORAGE_PATH_ENVVAR
from nemo_platform_plugin.jobs.spec import PlatformJobSpec
from nemo_platform_plugin.models.client import AsyncModelsClient
from nemo_platform_plugin.scheduler import NemoJobScheduler
from nemo_platform_plugin.sdk import AsyncNeMoPlatform, NeMoPlatform
from pydantic import BaseModel, ConfigDict
from pytest_mock import MockerFixture
from typer.testing import CliRunner

EXAMPLE_SPEC_PATHS = (
    Path("skills/nemo-evaluator-plugin/assets/specs/exact_match_metric.json"),
    Path("skills/nemo-evaluator-plugin/assets/specs/llm_as_judge.json"),
)


def _exact_match_spec() -> dict:
    return {
        "metrics": [
            _bundle_payload(ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.model_output}}"))
        ],
        "dataset": [
            {"expected": "blue", "model_output": "Blue"},
            {"expected": "Jupiter", "model_output": "Saturn"},
        ],
        "params": {"parallelism": 2},
    }


def _bundle_payload(metric) -> dict[str, Any]:
    return bundle_metric(metric, CloudpickleMetricBundlePackager()).model_dump(mode="json")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _assert_metric_step_entrypoint(job_spec: PlatformJobSpec) -> None:
    step = job_spec.steps[0]
    container = cast(Any, step.executor).container
    assert container.entrypoint == ["python", "-m"]
    assert container.command == ["nemo_evaluator.tasks.evaluate"]


def _load_cli_run_payload(output: str) -> dict[str, Any]:
    """Return the evaluator run JSON payload from CLI stdout."""
    return cast(dict[str, Any], json.loads(output[output.index('{\n  "status"') :]))


def _make_job_context(tmp_path: Path) -> JobContext:
    """Return a local job context with persistent result storage."""
    storage = StoragePaths(ephemeral=tmp_path / "ephemeral", persistent=tmp_path / "persistent")
    storage.ephemeral.mkdir()
    storage.persistent.mkdir()
    return JobContext(
        workspace="dev",
        storage=storage,
        results=LocalJobResults(root=storage.persistent / "results"),
    )


def _empty_evaluation_result() -> EvaluationResult:
    """Return an SDK result object suitable for runner delegation tests."""
    return EvaluationResult(row_scores=[], aggregate_scores=AggregatedMetricResult(scores=[]))


def _assert_saved_result_artifact(
    run_result: dict[str, Any], ctx: JobContext, result_payload: dict[str, object]
) -> None:
    """Assert that the evaluator result was persisted and registered."""
    assert run_result["artifact"] == {
        "name": DEFAULT_RESULT_NAME,
        "artifact_url": f"file://{ctx.storage.persistent / 'results' / DEFAULT_RESULT_NAME}",
    }
    result_path = ctx.storage.persistent / DEFAULT_FILE_NAME
    assert json.loads(result_path.read_text(encoding="utf-8")) == result_payload
    artifact_path = Path(run_result["artifact"]["artifact_url"].removeprefix("file://"))
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == result_payload
    assert (ctx.storage.persistent / "results" / AGGREGATE_SCORES_RESULT_NAME).exists()
    assert (ctx.storage.persistent / "results" / ROW_SCORES_RESULT_NAME).exists()
    assert (ctx.storage.persistent / "results" / ARTIFACTS_RESULT_NAME).is_dir()


def _load_artifact_payload(run_result: dict[str, Any]) -> dict[str, Any]:
    """Load a local artifact payload from a scheduler or CLI run result."""
    artifact_path = Path(run_result["artifact"]["artifact_url"].removeprefix("file://"))
    return cast(dict[str, Any], json.loads(artifact_path.read_text(encoding="utf-8")))


class _StaticMetric:
    def __init__(self, metric_type: str) -> None:
        self._metric_type = metric_type

    @property
    def type(self) -> str:
        return self._metric_type

    def output_spec(self) -> list[MetricOutputSpec]:
        return [MetricOutputSpec.continuous_score("score")]

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        del input
        return MetricResult(outputs=[MetricOutput(name="score", value=1.0)])


class _CountingJobParamsMetric(BaseModel):
    type: str = "params-count"
    applications: int = 0

    def output_spec(self) -> list[MetricOutputSpec]:
        return [MetricOutputSpec.continuous_score("applications")]

    def apply_evaluation_job_params(self, params: RunConfig | RunConfigOnline | RunConfigOnlineModel) -> None:
        del params
        self.applications += 1

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        del input
        return MetricResult(outputs=[MetricOutput(name="applications", value=float(self.applications))])


class _StaticMetricPayload(MetricBundlePayload):
    @property
    def kind(self) -> Literal["test-static"]:
        return "test-static"

    @property
    def digest(self) -> str:
        return "test-static-digest"


class _StrictMetricPayload(MetricBundlePayload):
    model_config = ConfigDict(extra="forbid")

    @property
    def kind(self) -> Literal["test-strict"]:
        return "test-strict"

    @property
    def digest(self) -> str:
        return "test-strict-digest"


class _StaticMetricBundlePackager(MetricBundlePackager):
    def package(self, metric: Metric) -> MetricBundlePayload:
        del metric
        return _StaticMetricPayload()

    def load(self, payload: MetricBundlePayload) -> Metric:
        del payload
        return _StaticMetric("test-static")


register_metric_bundle_kind(
    "test-static",
    payload_type=_StaticMetricPayload,
    packager_factory=_StaticMetricBundlePackager,
)
register_metric_bundle_kind(
    "test-strict",
    payload_type=_StrictMetricPayload,
    packager_factory=_StaticMetricBundlePackager,
)


def _generated_async_sdk() -> AsyncNeMoPlatform:
    return AsyncNeMoPlatform(
        base_url="http://platform.test",
        workspace="default",
        http_client=AsyncMock(spec=httpx.AsyncClient),
    )


def _patch_async_model_reference_resolution(mocker: MockerFixture) -> None:
    async def resolve_model_reference(self: AsyncModelsClient, ref: str) -> ResolvedModelReference:
        del self
        assert ref == "default/judge"
        return ResolvedModelReference(
            url="https://igw.example.test/v1/chat/completions",
            name="judge",
            host_url="http://nim.example.test:8000",
        )

    mocker.patch(
        "nemo_evaluator.jobs.metric_resolution.AsyncModelsClient.resolve_model_reference",
        resolve_model_reference,
    )


def _llm_judge_ref_metric() -> LLMJudgeMetric:
    return LLMJudgeMetric(
        model=ModelRef(root="default/judge"),
        scores=[
            RangeScore(
                name="quality",
                minimum=0,
                maximum=1,
                parser=JSONScoreParser(json_path="quality"),
            )
        ],
    )


@pytest.mark.parametrize(
    "spec_path",
    EXAMPLE_SPEC_PATHS,
)
def test_checked_in_example_spec_uses_inline_metric_bundle_shape(spec_path: Path) -> None:
    payload = json.loads((_repo_root() / spec_path).read_text(encoding="utf-8"))

    spec = EvaluateInputSpec.model_validate(payload)

    assert "metric" not in payload
    assert len(spec.metrics) >= 1
    for metric_payload in payload["metrics"]:
        bundle = MetricBundle.model_validate(metric_payload)
        assert bundle.payload.kind == "inline"
        assert {
            "python_version",
            "cloudpickle_version",
            "pickle_protocol",
            "blob",
        }.isdisjoint(metric_payload["payload"])
        assert bundle.metric_type == metric_payload["metric_type"]
        assert unbundle_metric(bundle).type == bundle.metric_type


@pytest.mark.parametrize(
    "spec_path",
    EXAMPLE_SPEC_PATHS,
)
async def test_checked_in_example_spec_transforms_and_compiles(spec_path: Path) -> None:
    payload = json.loads((_repo_root() / spec_path).read_text(encoding="utf-8"))

    input_spec = EvaluateInputSpec.model_validate(payload)
    spec = await EvaluateJob.to_spec(
        input_spec,
        workspace="default",
        entity_client=None,
        async_sdk=None,
        is_local=False,
    )
    assert isinstance(spec, EvaluateSpec)
    compiled = await EvaluateJob.compile(
        workspace="default",
        spec=spec,
        entity_client=None,
        job_name=None,
        async_sdk=None,
    )

    assert "metric" not in payload
    assert len(spec.metrics) >= 1
    assert PlatformJobSpec.model_validate(compiled).steps[0].config is not None


def test_evaluate_job_runs_inline_exact_match_metric() -> None:
    result = NemoJobScheduler().run_local(EvaluateJob, _exact_match_spec())

    assert result["status"] == "completed"
    assert "result" not in result
    aggregate_scores = _load_artifact_payload(result)["aggregate_scores"]["scores"]
    assert aggregate_scores[0]["name"] == "exact-match.exact-match"
    assert aggregate_scores[0]["mean"] == 0.5


def test_evaluate_job_survives_result_persistence_failure(mocker: MockerFixture) -> None:
    # Mirror of the agent-eval job: the queryable result record is a best-effort convenience index;
    # a persistence failure must not fail an otherwise-successful eval (its artifacts are already saved).
    persist = mocker.patch(
        "nemo_evaluator.jobs.evaluate.persist_evaluate_result",
        side_effect=RuntimeError("entity store unavailable"),
    )

    result = NemoJobScheduler().run_local(EvaluateJob, _exact_match_spec())

    persist.assert_called_once()
    assert result["status"] == "completed"
    aggregate_scores = _load_artifact_payload(result)["aggregate_scores"]["scores"]
    assert aggregate_scores[0]["name"] == "exact-match.exact-match"


def test_evaluate_job_applies_metric_job_params_once() -> None:
    spec = {
        "metrics": [_bundle_payload(_CountingJobParamsMetric())],
        "dataset": [{"value": "ignored"}],
        "params": {"parallelism": 2},
    }

    result = NemoJobScheduler().run_local(EvaluateJob, spec)

    assert result["status"] == "completed"
    aggregate_scores = _load_artifact_payload(result)["aggregate_scores"]["scores"]
    assert aggregate_scores[0]["name"] == "params-count.applications"
    assert aggregate_scores[0]["mean"] == 1.0


def test_cli_explain_uses_registered_evaluator_job_key() -> None:
    app = EvaluatorPluginCLI().get_cli()
    add_job_commands(app, {"evaluator.evaluate": EvaluateJob})

    result = CliRunner().invoke(app, ["evaluate", "explain"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["job_key"] == "evaluator.evaluate"
    assert payload["endpoint"] == "/apis/evaluator/v2/workspaces/{workspace}/evaluate/jobs"
    assert payload["spec_schema"]["title"] == "EvaluateSpec"


def test_cli_info_reports_registered_evaluator_job_keys() -> None:
    app = EvaluatorPluginCLI().get_cli()
    add_job_commands(app, {"evaluator.evaluate": EvaluateJob})

    result = CliRunner().invoke(app, ["info"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["jobs"] == [
        "evaluator.evaluate",
        "evaluator.agent-evaluate",
        "evaluator.retrieve-eval",
    ]


def test_cli_evaluate_still_exposes_run_during_stainless_migration() -> None:
    app = EvaluatorPluginCLI().get_cli()
    add_job_commands(app, {"evaluator.evaluate": EvaluateJob}, cli=EvaluatorPluginCLI())

    result = CliRunner().invoke(app, ["evaluate", "--help"])

    assert result.exit_code == 0
    assert "run" in result.output
    assert "submit" in result.output
    assert "explain" in result.output


def test_cli_metric_types_reports_sdk_metric_union_types() -> None:
    app = EvaluatorPluginCLI().get_cli()

    result = CliRunner().invoke(app, ["metric-types"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    entries = payload["metric_types"]
    metric_names = [entry["name"] for entry in entries]
    metrics = {entry["name"]: entry["description"] for entry in entries}
    assert metrics["exact-match"].startswith("Exact-match metric runtime for evaluator-driven execution.")
    assert metrics["llm-judge"].startswith("Runtime metric implementation for LLM-as-a-judge scoring.")
    assert metrics["remote"].startswith("A metric that computes scores via a remote endpoint.")
    assert metrics["topic_adherence"] == "Metric for measuring topic adherence."
    assert "system" not in metrics
    assert "system-retriever" not in metrics

    ragas_metric_types = {
        "agent_goal_accuracy",
        "answer_accuracy",
        "context_entity_recall",
        "context_precision",
        "context_recall",
        "context_relevance",
        "faithfulness",
        "noise_sensitivity",
        "response_groundedness",
        "response_relevancy",
        "tool_call_accuracy",
        "topic_adherence",
    }
    first_ragas_index = min(metric_names.index(metric_type) for metric_type in ragas_metric_types)
    non_ragas_metric_types = metric_names[:first_ragas_index]
    trailing_ragas_metric_types = metric_names[first_ragas_index:]
    assert not ragas_metric_types.intersection(non_ragas_metric_types)
    assert set(trailing_ragas_metric_types) == ragas_metric_types
    assert non_ragas_metric_types == sorted(non_ragas_metric_types)
    assert trailing_ragas_metric_types == sorted(trailing_ragas_metric_types)


def test_cli_metric_types_rejects_duplicate_metric_type_keys(mocker: MockerFixture) -> None:
    class FirstMetric(BaseModel):
        type: Literal["duplicate-metric"] = "duplicate-metric"

    class SecondMetric(BaseModel):
        type: Literal["duplicate-metric"] = "duplicate-metric"

    mocker.patch.object(
        evaluator_cli,
        "_unwrap_metric_model_classes",
        return_value=[FirstMetric, SecondMetric],
    )

    with pytest.raises(
        ValueError,
        match="Duplicate metric type 'duplicate-metric' mapped to both FirstMetric and SecondMetric",
    ):
        evaluator_cli._metric_type_models()


def test_cli_metric_types_reports_json_schema_for_named_metric_types() -> None:
    app = EvaluatorPluginCLI().get_cli()

    result = CliRunner().invoke(app, ["metric-types", "exact-match"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["title"] == "ExactMatchMetric"
    assert payload["properties"]["type"]["const"] == "exact-match"
    assert "workspace" not in payload["properties"]


def test_cli_metric_types_rejects_unknown_metric_types_name() -> None:
    app = EvaluatorPluginCLI().get_cli()

    result = CliRunner().invoke(app, ["metric-types", "missing-metric"])

    assert result.exit_code != 0
    assert "Unknown metric name 'missing-metric'" in result.output
    assert "nemo evaluator metric-types" in result.output


def test_cli_run_executes_evaluator_job() -> None:
    app = EvaluatorPluginCLI().get_cli()
    add_job_commands(app, {"evaluator.evaluate": EvaluateJob})

    result = CliRunner().invoke(app, ["evaluate", "run", "--spec", json.dumps(_exact_match_spec())])

    assert result.exit_code == 0
    payload = _load_cli_run_payload(result.output)
    assert payload["status"] == "completed"
    assert "result" not in payload
    assert _load_artifact_payload(payload)["aggregate_scores"]["scores"][0]["mean"] == 0.5


def test_unbundle_metric_dispatches_mixed_bundle_kinds_by_payload_kind() -> None:
    """Metric bundle hydration dispatches per bundle instead of assuming one packager."""
    cloudpickle_bundle = bundle_metric(
        ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.output}}"),
        CloudpickleMetricBundlePackager(),
    )
    static_bundle = bundle_metric(_StaticMetric("test-static"), _StaticMetricBundlePackager())

    metrics = [unbundle_metric(bundle) for bundle in [cloudpickle_bundle, static_bundle]]

    assert [metric.type for metric in metrics] == ["exact-match", "test-static"]


def test_metric_bundle_validation_strips_payload_kind_before_payload_validation() -> None:
    """Payload kind is the registry discriminator, not a concrete payload model field."""
    bundle = MetricBundle.model_validate(
        {
            "metric_type": "test-strict",
            "outputs": [{"name": "score", "value_json_schema": {"type": "number"}}],
            "payload": {"kind": "test-strict"},
        }
    )

    assert isinstance(bundle.payload, _StrictMetricPayload)


async def test_evaluate_job_resolves_metric_model_refs_before_sdk_run(
    tmp_path: Path,
    mocker: MockerFixture,
) -> None:
    async def compute_scores(metric: LLMJudgeMetric, input) -> MetricResult:
        del input
        assert isinstance(metric.model, Model)
        assert metric.model.name == "judge"
        assert metric.model.host_url == "http://nim.example.test:8000"
        return MetricResult(outputs=[MetricOutput(name="quality", value=1.0)])

    mocker.patch.object(LLMJudgeMetric, "preflight", mocker.AsyncMock(return_value=None))
    mocker.patch.object(LLMJudgeMetric, "compute_scores", compute_scores)
    _patch_async_model_reference_resolution(mocker)

    ctx = _make_job_context(tmp_path)
    spec = await EvaluateJob.to_spec(
        EvaluateInputSpec.model_validate(
            {
                "metrics": [_bundle_payload(_llm_judge_ref_metric())],
                "dataset": [{"output_text": "hello"}],
            }
        ),
        workspace="default",
        entity_client=object(),
        async_sdk=_generated_async_sdk(),
        is_local=True,
    )
    run_result = EvaluateJob().run(
        spec.model_dump(mode="json"),
        ctx=ctx,
    )

    payload = _load_artifact_payload(run_result)
    assert payload["aggregate_scores"]["scores"][0]["name"] == "llm-judge.quality"
    assert payload["aggregate_scores"]["scores"][0]["mean"] == 1.0


async def test_evaluate_job_compile_produces_cpu_task_step() -> None:
    spec = EvaluateSpec.model_validate(_exact_match_spec())
    compiled = await EvaluateJob.compile(
        workspace="default",
        spec=spec,
        entity_client=object(),
        job_name=None,
        async_sdk=None,
    )
    job_spec = PlatformJobSpec.model_validate(compiled)
    assert len(job_spec.steps) == 1
    step = job_spec.steps[0]
    assert step.name == "evaluate"
    _assert_metric_step_entrypoint(job_spec)
    assert step.config is not None
    config = cast(dict[str, Any], step.config)
    assert config["metrics"][0]["bundle_kind"] == "metric-bundle"
    assert config["metrics"][0]["metric_type"] == "exact-match"
    assert config["dataset"] == _exact_match_spec()["dataset"]


def test_evaluate_spec_rejects_unresolved_bundled_metric_model_refs() -> None:
    with pytest.raises(ValueError, match="EvaluateSpec metric models must be resolved"):
        EvaluateSpec.model_validate(
            {
                "metrics": [_bundle_payload(_llm_judge_ref_metric())],
                "dataset": [{"output_text": "hello"}],
            }
        )


async def test_evaluate_job_to_spec_resolves_bundled_metric_model_refs_before_compile(mocker: MockerFixture) -> None:
    _patch_async_model_reference_resolution(mocker)

    canonical = await EvaluateJob.to_spec(
        EvaluateInputSpec.model_validate(
            {
                "metrics": [_bundle_payload(_llm_judge_ref_metric())],
                "dataset": [{"output_text": "hello"}],
            }
        ),
        workspace="default",
        entity_client=object(),
        async_sdk=_generated_async_sdk(),
        is_local=False,
    )
    assert isinstance(canonical, EvaluateSpec)
    canonical_metric = unbundle_metric(to_runtime_bundle(canonical.metrics[0]))
    assert isinstance(canonical_metric, LLMJudgeMetric)
    assert isinstance(canonical_metric.model, Model)
    assert canonical_metric.model.name == "judge"
    assert canonical_metric.model.url == "https://igw.example.test/v1/chat/completions"
    assert canonical_metric.model.host_url == "http://nim.example.test:8000"

    compiled = await EvaluateJob.compile(
        workspace="default",
        spec=canonical,
        entity_client=object(),
        job_name=None,
        async_sdk=None,
    )

    job_spec = PlatformJobSpec.model_validate(compiled)
    config = cast(dict[str, Any], job_spec.steps[0].config)
    metric_bundle = MetricBundle.model_validate(config["metrics"][0])
    metric = unbundle_metric(metric_bundle)
    assert isinstance(metric, LLMJudgeMetric)
    assert isinstance(metric.model, Model)
    assert metric.model.name == "judge"


async def test_evaluate_job_to_spec_preserves_metric_without_model_refs() -> None:
    canonical = await EvaluateJob.to_spec(
        EvaluateInputSpec.model_validate(
            {
                "metrics": [
                    _bundle_payload(
                        LLMJudgeMetric(
                            model=Model(url="http://judge.test/v1/chat/completions", name="judge"),
                            scores=[
                                RangeScore(
                                    name="quality",
                                    minimum=0,
                                    maximum=1,
                                    parser=JSONScoreParser(json_path="quality"),
                                )
                            ],
                        )
                    )
                ],
                "dataset": [{"output_text": "hello"}],
                "params": RunConfig(),
            }
        ),
        workspace="default",
        entity_client=object(),
        async_sdk=None,
        is_local=False,
    )

    assert isinstance(canonical, EvaluateSpec)
    metric = unbundle_metric(to_runtime_bundle(canonical.metrics[0]))
    assert isinstance(metric, LLMJudgeMetric)
    assert metric.prompt_template is None


async def test_evaluate_job_compile_produces_online_model_job() -> None:
    spec = EvaluateSpec.model_validate(
        {
            **_exact_match_spec(),
            "target": Model(url="http://model.test/v1/chat/completions", name="test-model"),
            "params": RunConfigOnlineModel(parallelism=3),
            "prompt_template": "Question: {{item.question}}",
        }
    )

    compiled = await EvaluateJob.compile(
        workspace="default",
        spec=spec,
        entity_client=object(),
        job_name=None,
        async_sdk=None,
    )

    job_spec = PlatformJobSpec.model_validate(compiled)
    step = job_spec.steps[0]
    config = cast(dict[str, Any], step.config)
    _assert_metric_step_entrypoint(job_spec)
    assert config["target"]["name"] == "test-model"
    assert config["prompt_template"] == "Question: {{item.question}}"
    assert config["params"]["parallelism"] == 3


async def test_evaluate_job_compile_normalizes_generic_online_model_params() -> None:
    spec = EvaluateSpec.model_validate(
        {
            **_exact_match_spec(),
            "target": Model(url="http://model.test/v1/chat/completions", name="test-model"),
            "params": RunConfigOnline(parallelism=3),
            "prompt_template": "Question: {{item.question}}",
        }
    )

    compiled = await EvaluateJob.compile(
        workspace="default",
        spec=spec,
        entity_client=object(),
        job_name=None,
        async_sdk=None,
    )

    job_spec = PlatformJobSpec.model_validate(compiled)
    config = cast(dict[str, Any], job_spec.steps[0].config)
    assert isinstance(spec.params, RunConfigOnlineModel)
    assert config["params"]["parallelism"] == 3


async def test_evaluate_job_compile_produces_online_agent_job() -> None:
    spec = EvaluateSpec.model_validate(
        {
            **_exact_match_spec(),
            "target": GenericAgent(
                url="http://agent.test",
                name="test-agent",
                format=AgentFormat.GENERIC,
                body={"question": "{{item.question}}"},
                response_path="$.answer",
            ),
            "prompt_template": {"question": "{{item.question}}"},
            "params": RunConfigOnline(parallelism=3),
        }
    )

    compiled = await EvaluateJob.compile(
        workspace="default",
        spec=spec,
        entity_client=object(),
        job_name=None,
        async_sdk=None,
    )

    job_spec = PlatformJobSpec.model_validate(compiled)
    step = job_spec.steps[0]
    config = cast(dict[str, Any], step.config)
    _assert_metric_step_entrypoint(job_spec)
    assert config["target"]["name"] == "test-agent"
    assert config["prompt_template"] == {"question": "{{item.question}}"}


async def test_evaluate_job_compile_injects_metric_and_target_secrets() -> None:
    secret_ref = SecretRef(root="NVIDIA_BUILD_API_KEY")
    spec = EvaluateSpec.model_validate(
        {
            **_exact_match_spec(),
            "metrics": [
                _bundle_payload(
                    LLMJudgeMetric(
                        model=Model(
                            url="https://integrate.api.nvidia.com/v1/chat/completions",
                            name="nvidia/nemotron-3-super-120b-a12b",
                            api_key_secret=secret_ref,
                        ),
                        scores=[
                            RangeScore(
                                name="quality",
                                minimum=1,
                                maximum=5,
                                parser=JSONScoreParser(json_path="quality"),
                            ),
                        ],
                    )
                )
            ],
            "target": Model(
                url="https://integrate.api.nvidia.com/v1/chat/completions",
                name="nvidia/nemotron-3-super-120b-a12b",
                api_key_secret=secret_ref,
            ),
            "params": RunConfigOnlineModel(parallelism=3),
            "prompt_template": "Question: {{item.question}}",
        }
    )

    compiled = await EvaluateJob.compile(
        workspace="default",
        spec=spec,
        entity_client=object(),
        job_name=None,
        async_sdk=None,
    )

    step = PlatformJobSpec.model_validate(compiled).steps[0]
    secrets = {env.name: env.from_secret.name for env in step.environment or [] if env.from_secret}
    assert secrets == {"NVIDIA_BUILD_API_KEY": "NVIDIA_BUILD_API_KEY"}


async def test_evaluate_job_compile_rejects_secret_reserved_env_names() -> None:
    metric = LLMJudgeMetric(
        model=Model(
            url="https://integrate.api.nvidia.com/v1/chat/completions",
            name="nvidia/nemotron-3-super-120b-a12b",
            api_key_secret=SecretRef(root=PERSISTENT_JOB_STORAGE_PATH_ENVVAR),
        ),
        scores=[
            RangeScore(
                name="quality",
                minimum=1,
                maximum=5,
                parser=JSONScoreParser(json_path="quality"),
            ),
        ],
    )
    spec = EvaluateSpec.model_validate({**_exact_match_spec(), "metrics": [_bundle_payload(metric)]})

    with pytest.raises(ValueError, match="reserved"):
        await EvaluateJob.compile(
            workspace="default",
            spec=spec,
            entity_client=object(),
            job_name=None,
            async_sdk=None,
        )


class TestEvaluateSpec:
    """Validation coverage for evaluator job specs."""

    def test_rejects_empty_dataset(self) -> None:
        with pytest.raises(ValueError, match="List should have at least 1 item"):
            EvaluateSpec.model_validate(
                {
                    **_exact_match_spec(),
                    "dataset": [],
                }
            )

    def test_rejects_legacy_metric_config(self) -> None:
        with pytest.raises(ValueError, match="metrics|Extra inputs are not permitted"):
            EvaluateSpec.model_validate(
                {
                    "metrics": {
                        "type": "exact-match",
                        "reference": "{{item.expected}}",
                        "candidate": "{{item.model_output}}",
                    },
                    "dataset": _exact_match_spec()["dataset"],
                }
            )

    def test_rejects_singular_metric_field(self) -> None:
        with pytest.raises(ValueError, match="metrics|Extra inputs are not permitted"):
            EvaluateSpec.model_validate(
                {
                    "metric": _exact_match_spec()["metrics"][0],
                    "dataset": _exact_match_spec()["dataset"],
                }
            )

    def test_accepts_metrics_sequence(self) -> None:
        spec = EvaluateSpec.model_validate(
            {
                **_exact_match_spec(),
                "metrics": [
                    _exact_match_spec()["metrics"][0],
                    _bundle_payload(F1Metric(reference="{{item.expected}}", candidate="{{item.model_output}}")),
                ],
            }
        )

        assert [metric.metric_type for metric in spec.metrics] == ["exact-match", "f1"]

    def test_accepts_uppercase_api_key_secret_refs_for_llm_judge_and_target(self) -> None:
        spec = EvaluateSpec.model_validate(
            {
                "metrics": [
                    _bundle_payload(
                        LLMJudgeMetric(
                            model=Model(
                                url="https://integrate.api.nvidia.com/v1/chat/completions",
                                name="nvidia/nemotron-3-super-120b-a12b",
                                api_key_secret=SecretRef(root="NVIDIA_BUILD_API_KEY"),
                            ),
                            scores=[
                                RangeScore(
                                    name="quality",
                                    minimum=1,
                                    maximum=5,
                                    parser=JSONScoreParser(json_path="quality"),
                                ),
                            ],
                        )
                    )
                ],
                "dataset": [{"prompt": "Hello", "model_output": "Hi"}],
                "target": {
                    "url": "https://integrate.api.nvidia.com/v1/chat/completions",
                    "name": "nvidia/nemotron-3-super-120b-a12b",
                    "api_key_secret": "NVIDIA_BUILD_API_KEY",
                    "format": "nim",
                },
                "params": RunConfigOnlineModel(),
            }
        )

        assert isinstance(spec.target, Model)
        assert spec.target.api_key_secret is not None
        assert spec.metrics[0].metric_type == "llm-judge"
        assert spec.target.api_key_secret.root == "NVIDIA_BUILD_API_KEY"

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValueError, match="Extra inputs are not permitted"):
            EvaluateSpec.model_validate(
                {
                    **_exact_match_spec(),
                    "unexpected": "field",
                }
            )

    def test_accepts_serialized_fileset_ref_dataset(self) -> None:
        spec = EvaluateSpec.model_validate(
            {
                **_exact_match_spec(),
                "dataset": "default/helpsteer2#validation/*.jsonl",
            }
        )

        assert spec.dataset == FilesetRef(root="default/helpsteer2#validation/*.jsonl")

    def test_rejects_aggregate_fields_in_params(self) -> None:
        with pytest.raises(ValueError, match="Extra inputs are not permitted"):
            EvaluateSpec.model_validate(
                {
                    **_exact_match_spec(),
                    "params": {"aggregate_fields": ["mean", "max"]},
                }
            )


class TestEvaluateJobCompile:
    """Branch coverage for the evaluator job compiler."""

    async def test_accepts_equivalent_base_model_spec(self) -> None:
        class EquivalentSpec(BaseModel):
            """Spec shape used to verify compile canonicalizes BaseModel inputs."""

            metrics: list[dict[str, object]]
            dataset: list[dict[str, object]]
            params: dict[str, object] | None = None

        compiled = await EvaluateJob.compile(
            workspace="default",
            spec=EquivalentSpec.model_validate(_exact_match_spec()),
            entity_client=object(),
            job_name=None,
            async_sdk=None,
        )

        job_spec = PlatformJobSpec.model_validate(compiled)
        step = job_spec.steps[0]
        config = cast(dict[str, Any], step.config)
        assert config["metrics"][0]["bundle_kind"] == "metric-bundle"
        assert config["metrics"][0]["metric_type"] == "exact-match"
        assert config["dataset"] == _exact_match_spec()["dataset"]
        assert config["params"]["parallelism"] == 2

    async def test_accepts_metrics_sequence(self) -> None:
        spec = EvaluateSpec.model_validate(
            {
                **_exact_match_spec(),
                "metrics": [
                    _exact_match_spec()["metrics"][0],
                    _bundle_payload(F1Metric(reference="{{item.expected}}", candidate="{{item.model_output}}")),
                ],
            }
        )

        compiled = await EvaluateJob.compile(
            workspace="default",
            spec=spec,
            entity_client=object(),
            job_name=None,
            async_sdk=None,
        )

        config = cast(dict[str, Any], PlatformJobSpec.model_validate(compiled).steps[0].config)
        assert [metric["metric_type"] for metric in config["metrics"]] == ["exact-match", "f1"]

    @pytest.mark.parametrize(
        ("target", "expected_message"),
        [
            (
                Model(url="http://model.test/v1/chat/completions", name="test-model"),
                "prompt_template is required when EvaluateSpec.target is a model",
            ),
            (
                GenericAgent(
                    url="http://agent.test",
                    name="test-agent",
                    format=AgentFormat.GENERIC,
                    body={"question": "{{item.question}}"},
                    response_path="$.answer",
                ),
                "prompt_template is required when EvaluateSpec.target is an agent",
            ),
        ],
    )
    async def test_requires_prompt_template_for_online_targets(
        self, target: Model | Agent, expected_message: str
    ) -> None:
        spec = EvaluateSpec.model_validate(
            {
                **_exact_match_spec(),
                "target": target,
                "params": RunConfigOnlineModel() if isinstance(target, Model) else RunConfigOnline(),
            }
        )

        with pytest.raises(ValueError, match=expected_message):
            await EvaluateJob.compile(
                workspace="default",
                spec=spec,
                entity_client=object(),
                job_name=None,
                async_sdk=None,
            )

    @pytest.mark.parametrize(
        ("target", "expected_message"),
        [
            (Model(url="http://model.test/v1/chat/completions", name="test-model"), "model target"),
            (
                GenericAgent(
                    url="http://agent.test",
                    name="test-agent",
                    format=AgentFormat.GENERIC,
                    body={"question": "{{item.question}}"},
                    response_path="$.answer",
                ),
                "agent target",
            ),
        ],
    )
    async def test_rejects_wrong_online_param_type(self, target: Model | Agent, expected_message: str) -> None:
        with pytest.raises(TypeError, match=expected_message):
            EvaluateSpec.model_validate(
                {
                    **_exact_match_spec(),
                    "target": target,
                    "params": RunConfig(),
                    "prompt_template": "Question: {{item.question}}",
                }
            )

    async def test_rejects_missing_online_params(self) -> None:
        with pytest.raises(TypeError, match="model target requires RunConfigOnlineModel"):
            EvaluateSpec.model_validate(
                {
                    **_exact_match_spec(),
                    "target": Model(url="http://model.test/v1/chat/completions", name="test-model"),
                    "prompt_template": "Question: {{item.question}}",
                }
            )

    async def test_fileset_ref_dataset_compiles_into_evaluate_step(self) -> None:
        dataset = FilesetRef(root="default/helpsteer2#validation/*.jsonl")

        compiled = await EvaluateJob.compile(
            workspace="default",
            spec=EvaluateSpec.model_validate({**_exact_match_spec(), "dataset": dataset}),
            entity_client=object(),
            job_name=None,
            async_sdk=None,
        )

        job_spec = PlatformJobSpec.model_validate(compiled)
        assert [step.name for step in job_spec.steps] == ["evaluate"]
        config = cast(dict[str, Any], job_spec.steps[0].config)
        assert config["dataset"] == dataset.root


class TestEvaluateJobRun:
    """Coverage for the local evaluator job runner."""

    @pytest.mark.parametrize(
        ("spec_overrides", "expected_config_type"),
        [
            ({}, RunConfig),
            (
                {
                    "target": Model(url="http://model.test/v1/chat/completions", name="test-model"),
                    "params": RunConfigOnlineModel(),
                    "prompt_template": "Question: {{item.question}}",
                },
                RunConfigOnlineModel,
            ),
            (
                {
                    "target": GenericAgent(
                        url="http://agent.test",
                        name="test-agent",
                        format=AgentFormat.GENERIC,
                        body={"question": "{{item.question}}"},
                        response_path="$.answer",
                    ),
                    "params": RunConfigOnline(),
                    "prompt_template": {"question": "{{item.question}}"},
                },
                RunConfigOnline,
            ),
        ],
    )
    def test_delegates_to_sdk_evaluator(
        self,
        spec_overrides: dict[str, object],
        expected_config_type: type[RunConfig],
        tmp_path: Path,
        mocker: MockerFixture,
    ) -> None:
        result = _empty_evaluation_result()
        result_payload = result.model_dump(mode="json")
        evaluator = mocker.Mock()
        evaluator.run_sync.return_value = result
        evaluator_cls = mocker.patch("nemo_evaluator.jobs.evaluate.Evaluator", return_value=evaluator)
        config = {
            **_exact_match_spec(),
            **spec_overrides,
        }
        expected_spec = EvaluateSpec.model_validate(config)
        expected_config = expected_spec.params
        assert isinstance(expected_config, expected_config_type)
        ctx = _make_job_context(tmp_path)

        run_result = EvaluateJob().run(config, ctx=ctx)

        assert run_result == {
            "status": "completed",
            "artifact": run_result["artifact"],
        }
        assert "result" not in run_result
        _assert_saved_result_artifact(run_result, ctx, result_payload)
        evaluator_cls.assert_called_once_with()
        call_kwargs = evaluator.run_sync.call_args.kwargs
        assert [type(metric) for metric in call_kwargs["metrics"]] == [ExactMatchMetric]
        assert call_kwargs["dataset"] == expected_spec.dataset
        assert call_kwargs["config"] == expected_config
        assert call_kwargs["target"] == expected_spec.target
        assert call_kwargs["prompt_template"] == expected_spec.prompt_template

    def test_delegates_metrics_sequence_to_sdk_evaluator(self, tmp_path: Path, mocker: MockerFixture) -> None:
        result = _empty_evaluation_result()
        result_payload = result.model_dump(mode="json")
        evaluator = mocker.Mock()
        evaluator.run_sync.return_value = result
        evaluator_cls = mocker.patch("nemo_evaluator.jobs.evaluate.Evaluator", return_value=evaluator)
        config = {
            **_exact_match_spec(),
            "metrics": [
                _exact_match_spec()["metrics"][0],
                _bundle_payload(F1Metric(reference="{{item.expected}}", candidate="{{item.model_output}}")),
            ],
        }
        expected_spec = EvaluateSpec.model_validate(config)
        ctx = _make_job_context(tmp_path)

        run_result = EvaluateJob().run(config, ctx=ctx)

        assert run_result == {
            "status": "completed",
            "artifact": run_result["artifact"],
        }
        assert "result" not in run_result
        _assert_saved_result_artifact(run_result, ctx, result_payload)
        evaluator_cls.assert_called_once_with()
        call_kwargs = evaluator.run_sync.call_args.kwargs
        assert [metric.type.value for metric in call_kwargs["metrics"]] == ["exact-match", "f1"]
        assert call_kwargs["dataset"] == expected_spec.dataset
        assert call_kwargs["config"] == expected_spec.params
        assert call_kwargs["target"] == expected_spec.target
        assert call_kwargs["prompt_template"] == expected_spec.prompt_template

    def test_downloads_fileset_ref_dataset_and_passes_path_to_sdk_evaluator(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        result = _empty_evaluation_result()
        result_payload = result.model_dump(mode="json")
        evaluator = mocker.Mock()
        evaluator.run_sync.return_value = result
        mocker.patch("nemo_evaluator.jobs.evaluate.Evaluator", return_value=evaluator)
        downloaded_path = tmp_path / "persistent" / "dataset" / "default" / "helpsteer2" / "validation.jsonl"
        download_dataset = mocker.patch(
            "nemo_evaluator.jobs.evaluate.download_dataset",
            new=mocker.AsyncMock(return_value=downloaded_path),
            create=True,
        )
        download_dataset_sync = mocker.patch("nemo_evaluator.jobs.evaluate.download_dataset_sync", create=True)
        async_sdk = AsyncNemoClient(
            base_url="http://platform.test",
            workspace="default",
            http_client=AsyncMock(spec=httpx.AsyncClient),
        )
        http_client = mocker.AsyncMock(name="http_client")
        http_client.__aenter__.return_value = http_client
        http_client.__aexit__.return_value = None
        mocker.patch("nemo_evaluator.jobs.utils.httpx.AsyncClient", return_value=http_client)
        ctx = _make_job_context(tmp_path)
        dataset = FilesetRef(root="default/helpsteer2#validation.jsonl")
        config = {**_exact_match_spec(), "dataset": dataset}

        run_result = EvaluateJob().run(config, ctx=ctx, async_sdk=async_sdk)

        _assert_saved_result_artifact(run_result, ctx, result_payload)
        assert download_dataset.await_count == 1
        cloned_client = download_dataset.await_args.kwargs["client"]
        assert isinstance(cloned_client, AsyncNemoClient)
        assert cloned_client is not async_sdk
        assert cloned_client.base_url == async_sdk.base_url
        assert cloned_client.workspace == async_sdk.workspace
        assert cloned_client._client is http_client
        assert download_dataset.await_args.kwargs["dataset"] == dataset
        assert download_dataset.await_args.kwargs["destination"] == str(ctx.storage.persistent / "dataset")
        download_dataset_sync.assert_not_called()
        call_kwargs = evaluator.run_sync.call_args.kwargs
        assert [type(metric) for metric in call_kwargs["metrics"]] == [ExactMatchMetric]
        assert call_kwargs["dataset"] == downloaded_path
        assert call_kwargs["config"] == EvaluateSpec.model_validate(config).params
        assert call_kwargs["target"] is None
        assert call_kwargs["prompt_template"] is None

    def test_downloads_fileset_ref_dataset_with_sync_sdk_and_passes_path_to_sdk_evaluator(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        result = _empty_evaluation_result()
        result_payload = result.model_dump(mode="json")
        evaluator = mocker.Mock()
        evaluator.run_sync.return_value = result
        mocker.patch("nemo_evaluator.jobs.evaluate.Evaluator", return_value=evaluator)
        downloaded_path = tmp_path / "persistent" / "dataset" / "default" / "helpsteer2" / "validation.jsonl"
        download_dataset = mocker.patch("nemo_evaluator.jobs.evaluate.download_dataset", create=True)
        download_dataset_sync = mocker.patch(
            "nemo_evaluator.jobs.evaluate.download_dataset_sync",
            return_value=downloaded_path,
            create=True,
        )
        ctx = _make_job_context(tmp_path)
        sync_sdk = mocker.Mock()
        dataset = FilesetRef(root="default/helpsteer2#validation.jsonl")
        config = {**_exact_match_spec(), "dataset": dataset}

        run_result = EvaluateJob().run(config, ctx=ctx, sdk=sync_sdk)

        _assert_saved_result_artifact(run_result, ctx, result_payload)
        download_dataset.assert_not_called()
        download_dataset_sync.assert_called_once_with(
            client=sync_sdk,
            dataset=dataset,
            destination=str(ctx.storage.persistent / "dataset"),
        )
        call_kwargs = evaluator.run_sync.call_args.kwargs
        assert [type(metric) for metric in call_kwargs["metrics"]] == [ExactMatchMetric]
        assert call_kwargs["dataset"] == downloaded_path
        assert call_kwargs["config"] == EvaluateSpec.model_validate(config).params
        assert call_kwargs["target"] is None
        assert call_kwargs["prompt_template"] is None

    def test_run_adapts_the_generated_sdk_the_local_cli_injects(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """``nemo evaluator evaluate run`` injects a generated ``NeMoPlatform``, but resolving a
        ``FilesetRef`` dataset only accepts a typed client."""
        evaluator = mocker.Mock()
        evaluator.run_sync.return_value = _empty_evaluation_result()
        mocker.patch("nemo_evaluator.jobs.evaluate.Evaluator", return_value=evaluator)
        download_dataset_sync = mocker.patch(
            "nemo_evaluator.jobs.evaluate.download_dataset_sync",
            return_value=tmp_path / "persistent" / "dataset" / "default" / "helpsteer2" / "validation.jsonl",
            create=True,
        )
        platform = NeMoPlatform(base_url="http://platform.test", workspace="dev", http_client=httpx.Client())
        config = {**_exact_match_spec(), "dataset": FilesetRef(root="default/helpsteer2#validation.jsonl")}

        EvaluateJob().run(config, ctx=_make_job_context(tmp_path), sdk=platform)

        assert isinstance(download_dataset_sync.call_args.kwargs["client"], NemoClient)

    def test_prefers_sync_sdk_for_fileset_ref_when_both_sdks_injected(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        """Avoid binding async_sdk's httpx client before result-entity persistence."""
        result = _empty_evaluation_result()
        result_payload = result.model_dump(mode="json")
        evaluator = mocker.Mock()
        evaluator.run_sync.return_value = result
        mocker.patch("nemo_evaluator.jobs.evaluate.Evaluator", return_value=evaluator)
        downloaded_path = tmp_path / "persistent" / "dataset" / "default" / "helpsteer2" / "validation.jsonl"
        download_dataset = mocker.patch("nemo_evaluator.jobs.evaluate.download_dataset", create=True)
        download_dataset_sync = mocker.patch(
            "nemo_evaluator.jobs.evaluate.download_dataset_sync",
            return_value=downloaded_path,
            create=True,
        )
        persist = mocker.patch("nemo_evaluator.jobs.evaluate.persist_evaluate_result")
        ctx = _make_job_context(tmp_path)
        sync_sdk = mocker.Mock()
        async_sdk = mocker.Mock()
        dataset = FilesetRef(root="default/helpsteer2#validation.jsonl")
        config = {**_exact_match_spec(), "dataset": dataset}

        run_result = EvaluateJob().run(config, ctx=ctx, sdk=sync_sdk, async_sdk=async_sdk)

        _assert_saved_result_artifact(run_result, ctx, result_payload)
        download_dataset.assert_not_called()
        download_dataset_sync.assert_called_once_with(
            client=sync_sdk,
            dataset=dataset,
            destination=str(ctx.storage.persistent / "dataset"),
        )
        persist.assert_called_once()
        assert persist.call_args.kwargs["async_sdk"] is async_sdk
        assert persist.call_args.kwargs["dataset_ref"] == dataset.root


class TestEvaluateTask:
    """Coverage for the compiled container task entrypoint."""

    def test_main_dispatches_evaluate_job_with_task_sdk(self, mocker: MockerFixture) -> None:
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

        exit_code = evaluate_task_main()

        assert exit_code == 0
        get_platform_sdk.assert_called_once_with("evaluator")
        build_ctx.assert_called_once_with(sdk)
        get_task_client.assert_called_once_with("evaluator")
        get_async_task_client.assert_called_once_with("evaluator")
        run_task.assert_called_once_with(EvaluateJob, sdk=client, async_sdk=async_client, ctx=ctx)

    def test_main_returns_setup_exit_code_when_task_sdk_fails(self, mocker: MockerFixture) -> None:
        get_platform_sdk = mocker.patch("nemo_evaluator.tasks.runner.get_task_sdk", side_effect=RuntimeError("boom"))
        get_task_client = mocker.patch("nemo_evaluator.tasks.runner.get_task_nemo_client")
        run_task = mocker.patch("nemo_evaluator.tasks.runner.run_task")

        exit_code = evaluate_task_main()

        assert exit_code == SDK_INITIALIZATION_EXIT_CODE
        get_platform_sdk.assert_called_once_with("evaluator")
        get_task_client.assert_not_called()
        run_task.assert_not_called()
