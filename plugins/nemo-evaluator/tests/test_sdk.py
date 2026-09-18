# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the evaluator plugin SDK status resource."""

from __future__ import annotations

import json
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from models import ResolvedModelReference
from nemo_evaluator.api.schemas import MetricInline
from nemo_evaluator.filesets import FilesetRef
from nemo_evaluator.jobs.evaluate import EvaluateInputSpec, EvaluateSpec
from nemo_evaluator.metric_refs import MetricRefOrInline
from nemo_evaluator.sdk import _executor as executor_module
from nemo_evaluator.sdk._executor import (
    MetricBundlePackagerPolicyError,
    _AsyncEvaluatorPluginExecutor,
    _build_evaluate_spec,
    _SyncEvaluatorPluginExecutor,
    bundle_metrics_for_spec,
)
from nemo_evaluator.sdk.job_resources import AsyncEvaluatorJobResource, EvaluatorJobResource
from nemo_evaluator.sdk.resources import AsyncEvaluator, Evaluator
from nemo_evaluator.shared.metric_bundles.bundles import (
    MetricBundlePackager,
    MetricBundlePayload,
    bundle_metric,
)
from nemo_evaluator.shared.metric_bundles.cloudpickle import CloudpickleMetricBundlePackager
from nemo_evaluator.shared.metric_bundles.inline import InlineMetricBundlePackager
from nemo_evaluator_sdk.metrics.exact_match import ExactMatchMetric
from nemo_evaluator_sdk.metrics.protocol import Metric, MetricInput, MetricOutput, MetricOutputSpec, MetricResult
from nemo_evaluator_sdk.values import FieldMapping, Model, ModelRef, RunConfig, RunConfigOnline, RunConfigOnlineModel
from nemo_platform_plugin.client.errors import NemoResponseValidationError, NotFoundError
from nemo_platform_plugin.evaluator.client import AsyncEvaluatorClient, EvaluatorClient
from nemo_platform_plugin.jobs.schemas import PlatformJobStatus
from nemo_platform_plugin.models.client import AsyncModelsClient, ModelsClient
from pytest_mock import MockerFixture

_EXACT_MATCH_METRIC = ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.output}}")
_EXACT_MATCH_SPEC = {
    "metrics": [
        bundle_metric(
            _EXACT_MATCH_METRIC,
            CloudpickleMetricBundlePackager(),
        ).model_dump(mode="json")
    ],
    "dataset": [{"expected": "a", "output": "a"}],
}
_LEGACY_EXACT_MATCH_SPEC = {
    "metrics": {
        "type": "exact-match",
        "reference": "{{item.expected}}",
        "candidate": "{{item.output}}",
    },
    "dataset": [{"expected": "a", "output": "a"}],
}
_EXACT_MATCH_EVALUATE_SPEC = EvaluateSpec.model_validate(_EXACT_MATCH_SPEC)
_EXACT_MATCH_EVALUATE_SPEC_JSON = _EXACT_MATCH_EVALUATE_SPEC.model_dump(mode="json")
_EXACT_MATCH_EVALUATE_INPUT_SPEC = EvaluateInputSpec.model_validate(_EXACT_MATCH_SPEC)
_EXACT_MATCH_EVALUATE_INPUT_SPEC_JSON = _EXACT_MATCH_EVALUATE_INPUT_SPEC.model_dump(mode="json")


def _single_metric(spec: EvaluateInputSpec | EvaluateSpec) -> MetricInline:
    """Return the single metric from an evaluator job spec."""
    if len(spec.metrics) != 1:
        raise AssertionError("Expected a single metric spec.")
    metric = spec.metrics[0]
    # `EvaluateInputSpec.metrics` also admits `MetricRef`; every caller here builds inline metrics.
    assert isinstance(metric, MetricInline)
    return metric


def _metric_type(metric: MetricRefOrInline) -> str:
    """Metric type of an inline metric, narrowing away the `MetricRef` arm of the union."""
    assert isinstance(metric, MetricInline)
    return metric.metric_type


class _RecordingMetricBundlePackager(MetricBundlePackager):
    """Test packager that records all runtime metrics selected for packaging."""

    def __init__(self) -> None:
        self.metrics: list[Metric] = []
        self._delegate = CloudpickleMetricBundlePackager()

    def package(self, metric: Metric) -> MetricBundlePayload:
        self.metrics.append(metric)
        return self._delegate.package(metric)

    def load(self, payload: MetricBundlePayload) -> Metric:
        del payload
        raise NotImplementedError("test packager only exercises submission-side packaging")


class _CustomRuntimeMetric:
    """A protocol-satisfying metric that is not part of MetricsUnion (not inline-bundleable)."""

    type = "custom-score"
    description = "custom metric"
    labels: dict[str, str] = {}

    def output_spec(self) -> list[MetricOutputSpec]:
        return [MetricOutputSpec.continuous_score("score")]

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        del input
        return MetricResult(outputs=[MetricOutput(name="score", value=1.0)])


class _SyncPlatform(EvaluatorClient):
    def __init__(self) -> None:
        self.http_client = MagicMock(spec=httpx.Client)
        super().__init__(
            base_url="http://test:8000",
            workspace="platform-ws",
            default_headers={"Authorization": "Bearer sync-platform-token"},
            timeout=httpx.Timeout(42.0),
            http_client=self.http_client,
        )


class _AsyncPlatform(AsyncEvaluatorClient):
    def __init__(self) -> None:
        self.http_client = AsyncMock(spec=httpx.AsyncClient)
        super().__init__(
            base_url="http://test:8000",
            workspace="platform-ws",
            default_headers={"Authorization": "Bearer platform-token"},
            timeout=httpx.Timeout(43.0),
            http_client=self.http_client,
        )


def _json_response(method: str, url: str, payload: object, *, status_code: int = 200) -> httpx.Response:
    return httpx.Response(status_code, request=httpx.Request(method, url), json=payload)


def _empty_response(method: str, url: str, *, status_code: int = 204) -> httpx.Response:
    return httpx.Response(status_code, request=httpx.Request(method, url))


def _request_body(platform: _SyncPlatform | _AsyncPlatform) -> dict[str, Any]:
    return json.loads(platform.http_client.request.call_args.kwargs["content"].decode())


def _last_request_url(platform: _SyncPlatform | _AsyncPlatform) -> str:
    return platform.http_client.request.call_args.args[1]


def test_sync_executor_initializes_without_resource_callbacks() -> None:
    platform = _SyncPlatform()

    executor = _SyncEvaluatorPluginExecutor(client=platform)

    assert executor is not None


def test_async_executor_initializes_without_resource_callbacks() -> None:
    platform = _AsyncPlatform()

    executor = _AsyncEvaluatorPluginExecutor(client=platform)

    assert executor is not None


def test_resolve_workspace_requires_explicit_or_default_workspace() -> None:
    """Remote submission should fail when no workspace can be resolved."""
    client = EvaluatorClient(base_url="http://test:8000", workspace=None)
    executor = _SyncEvaluatorPluginExecutor(client=client)

    with pytest.raises(ValueError, match="workspace must be provided"):
        executor.submit(
            metric=_EXACT_MATCH_METRIC,
            dataset=[{"expected": "a", "output": "a"}],
            params=RunConfig(),
            metric_bundle_packager=CloudpickleMetricBundlePackager(),
        )


def test_bundle_metrics_for_spec_rejects_non_metric_object() -> None:
    """Metrics must satisfy the runtime Metric protocol before plugin execution."""
    bundle_metrics = object.__getattribute__(bundle_metrics_for_spec, "__call__")
    invalid_metric: Any = object()

    with pytest.raises(TypeError, match="metrics must be a Metric or a sequence of Metric objects"):
        bundle_metrics(invalid_metric, metric_bundle_packager=CloudpickleMetricBundlePackager())


def test_build_evaluate_spec_requires_metric_bundle_packager() -> None:
    with pytest.raises(MetricBundlePackagerPolicyError, match="CloudpickleMetricBundlePackager"):
        _build_evaluate_spec(
            metrics=ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.output}}"),
            dataset=[{"expected": "a", "output": "a"}],
            params=RunConfig(),
        )


def test_build_evaluate_spec_includes_target_and_prompt_template() -> None:
    """Online evaluator specs should preserve model targets and prompt templates."""
    model = Model(url="https://model.test/v1", name="model-a")
    spec = _build_evaluate_spec(
        metrics=ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.output}}"),
        metric_bundle_packager=CloudpickleMetricBundlePackager(),
        dataset=[{"expected": "a", "output": "a"}],
        params=RunConfigOnlineModel(),
        target=model,
        prompt_template="Answer: {{item.input}}",
    )

    assert spec.target == model
    assert spec.prompt_template == "Answer: {{item.input}}"


def test_build_evaluate_spec_uses_selected_packager_for_all_runtime_metrics() -> None:
    """Submission packages all outgoing runtime metrics with the caller-selected packager."""
    metric_a = ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.output}}")
    metric_b = ExactMatchMetric(reference="{{item.other_expected}}", candidate="{{item.other_output}}")
    packager = _RecordingMetricBundlePackager()

    spec = _build_evaluate_spec(
        metrics=[metric_a, metric_b],
        metric_bundle_packager=packager,
        dataset=[{"expected": "a", "output": "a"}],
        params=RunConfig(),
    )

    assert packager.metrics == [metric_a, metric_b]
    assert [_metric_type(metric) for metric in spec.metrics] == ["exact-match", "exact-match"]


def test_build_evaluate_spec_excludes_aggregate_fields() -> None:
    """Evaluator specs should not persist result-shaping options."""
    spec = _build_evaluate_spec(
        metrics=ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.output}}"),
        metric_bundle_packager=CloudpickleMetricBundlePackager(),
        dataset=[{"expected": "a", "output": "a"}],
        params=RunConfig(),
    )

    assert spec.params is not None
    assert "aggregate_fields" not in spec.params.model_dump(mode="json")


def test_build_evaluate_spec_preserves_field_mapping() -> None:
    """Evaluator specs should preserve dataset field mappings for local and remote jobs."""
    field_mapping = FieldMapping(output="prediction", reference="expected")

    spec = _build_evaluate_spec(
        metrics=ExactMatchMetric(reference="{{reference}}"),
        metric_bundle_packager=CloudpickleMetricBundlePackager(),
        dataset=[{"expected": "a", "prediction": "a"}],
        params=RunConfig(),
        field_mapping=field_mapping,
    )

    assert spec.field_mapping == field_mapping


def test_build_evaluate_spec_preserves_fileset_ref_dataset() -> None:
    """FilesetRef datasets should be carried to the job spec without eager row materialization."""
    dataset = FilesetRef(root="default/helpsteer2")

    spec = _build_evaluate_spec(
        metrics=ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.output}}"),
        metric_bundle_packager=CloudpickleMetricBundlePackager(),
        dataset=dataset,
        params=RunConfig(),
    )

    assert spec.dataset == dataset


def test_sync_resource_calls_evaluator_plugin_status() -> None:
    platform = _SyncPlatform()
    platform.http_client.request.return_value = _json_response(
        "GET",
        "http://test:8000/apis/evaluator/v1/healthz",
        {"plugin": "evaluator", "status": "ok"},
    )

    resource = Evaluator(platform)

    assert resource.plugin_status() == {"plugin": "evaluator", "status": "ok"}
    assert platform.http_client.request.call_args.args == ("GET", "http://test:8000/apis/evaluator/v1/healthz")
    assert platform.http_client.request.call_args.kwargs["headers"] == {"Authorization": "Bearer sync-platform-token"}
    assert platform.http_client.request.call_args.kwargs["params"] is None
    assert platform.http_client.request.call_args.kwargs["timeout"] == platform._timeout


def test_sync_resource_rejects_non_object_plugin_status() -> None:
    platform = _SyncPlatform()
    platform.http_client.request.return_value = _json_response(
        "GET", "http://test:8000/apis/evaluator/v1/healthz", ["ok"]
    )
    resource = Evaluator(platform)

    with pytest.raises(NemoResponseValidationError):
        resource.plugin_status()


_EVALUATOR_PUBLIC_RESOURCE_NAMES = {
    "agent_eval_results",
    "eval_results",
    "from_sdk",
    "get_job_resource",
    "metrics",
    "plugin_status",
    "submit",
    "tasks",
    "tasksets",
}


def _public_resource_names(resource: object) -> set[str]:
    return {name for name in dir(resource) if not name.startswith("_")}


def test_sync_resource_exposes_only_evaluator_sdk_surface() -> None:
    resource = Evaluator(_SyncPlatform())

    assert _public_resource_names(resource) == _EVALUATOR_PUBLIC_RESOURCE_NAMES


def test_sync_executor_creates_evaluator_job() -> None:
    platform = _SyncPlatform()
    platform.http_client.request.return_value = _json_response(
        "POST",
        "http://test:8000/apis/evaluator/v2/workspaces/ws/evaluate/jobs",
        {"name": "job-123", "status": "created", "spec": _EXACT_MATCH_SPEC},
        status_code=201,
    )
    executor = _SyncEvaluatorPluginExecutor(client=platform)
    spec = _EXACT_MATCH_EVALUATE_INPUT_SPEC

    job = executor.create(spec=spec, workspace="ws")

    assert isinstance(job, EvaluatorJobResource)
    assert job.name == "job-123"
    assert job.job.status == PlatformJobStatus.CREATED
    assert job.job.spec is not None
    assert _single_metric(job.job.spec).metric_type == "exact-match"
    assert platform.http_client.request.call_args.args == (
        "POST",
        "http://test:8000/apis/evaluator/v2/workspaces/ws/evaluate/jobs",
    )
    assert _request_body(platform) == {"spec": _EXACT_MATCH_EVALUATE_INPUT_SPEC_JSON}
    assert platform.http_client.request.call_args.kwargs["headers"] == {
        "Authorization": "Bearer sync-platform-token",
        "Content-Type": "application/json",
    }
    assert platform.http_client.request.call_args.kwargs["timeout"] == platform._timeout


def test_sync_executor_create_does_not_use_asyncio_thread_bridge() -> None:
    platform = _SyncPlatform()
    platform.http_client.request.return_value = _json_response(
        "POST",
        "http://test:8000/apis/evaluator/v2/workspaces/ws/evaluate/jobs",
        {"name": "job-123", "status": "created", "spec": _EXACT_MATCH_SPEC},
        status_code=201,
    )
    executor = _SyncEvaluatorPluginExecutor(client=platform)

    job = executor.create(spec=_EXACT_MATCH_EVALUATE_INPUT_SPEC, workspace="ws")

    assert isinstance(job, EvaluatorJobResource)
    # Remote submission stays on the sync transport path.
    assert not hasattr(executor_module, "asyncio")
    platform.http_client.request.assert_called_once()


def test_sync_executor_create_uses_platform_workspace_by_default() -> None:
    platform = _SyncPlatform()
    platform.http_client.request.return_value = _json_response(
        "POST",
        "http://test:8000/apis/evaluator/v2/workspaces/platform-ws/evaluate/jobs",
        {"name": "job-123", "spec": _EXACT_MATCH_SPEC},
        status_code=201,
    )
    executor = _SyncEvaluatorPluginExecutor(client=platform)

    job = executor.create(spec=_EXACT_MATCH_EVALUATE_INPUT_SPEC)
    assert job.name == "job-123"
    assert job.job.spec is not None
    assert _single_metric(job.job.spec).metric_type == "exact-match"
    assert platform.http_client.request.call_args.args == (
        "POST",
        "http://test:8000/apis/evaluator/v2/workspaces/platform-ws/evaluate/jobs",
    )
    assert _request_body(platform) == {"spec": _EXACT_MATCH_EVALUATE_INPUT_SPEC_JSON}


def test_sync_executor_create_rejects_malformed_response() -> None:
    platform = _SyncPlatform()
    platform.http_client.request.return_value = _json_response(
        "POST",
        "http://test:8000/apis/evaluator/v2/workspaces/ws/evaluate/jobs",
        ["job-123"],
        status_code=201,
    )
    executor = _SyncEvaluatorPluginExecutor(client=platform)

    with pytest.raises(NemoResponseValidationError):
        executor.create(spec=_EXACT_MATCH_EVALUATE_INPUT_SPEC, workspace="ws")
    assert _last_request_url(platform) == "http://test:8000/apis/evaluator/v2/workspaces/ws/evaluate/jobs"
    assert _request_body(platform) == {"spec": _EXACT_MATCH_EVALUATE_INPUT_SPEC_JSON}


def test_sync_executor_waits_when_requested(mocker: MockerFixture) -> None:
    platform = _SyncPlatform()
    platform.http_client.request.return_value = _json_response(
        "POST",
        "http://test:8000/apis/evaluator/v2/workspaces/ws/evaluate/jobs",
        {"name": "job-123", "status": "created", "spec": _EXACT_MATCH_SPEC},
        status_code=201,
    )
    wait = mocker.patch("nemo_evaluator.sdk.job_resources.EvaluatorJobResource.wait_until_done")
    executor = _SyncEvaluatorPluginExecutor(client=platform)

    job = executor.create(spec=_EXACT_MATCH_EVALUATE_INPUT_SPEC, workspace="ws", wait_until_done=True)

    assert isinstance(job, EvaluatorJobResource)
    assert platform.http_client.request.call_args.args == (
        "POST",
        "http://test:8000/apis/evaluator/v2/workspaces/ws/evaluate/jobs",
    )
    assert _request_body(platform) == {"spec": _EXACT_MATCH_EVALUATE_INPUT_SPEC_JSON}
    wait.assert_called_once_with(
        poll_interval_seconds=mocker.ANY,
        job_timeout_seconds=mocker.ANY,
        pending_timeout_seconds=mocker.ANY,
    )


def test_sync_resource_gets_existing_job_resource() -> None:
    platform = _SyncPlatform()
    platform.http_client.request.return_value = _json_response(
        "GET",
        "http://test:8000/apis/evaluator/v2/workspaces/ws/evaluate/jobs/job-123",
        {"name": "job-123", "status": "created", "spec": _EXACT_MATCH_SPEC},
    )
    resource = Evaluator(platform)

    job = resource.get_job_resource("job-123", workspace="ws")

    assert isinstance(job, EvaluatorJobResource)
    assert job.name == "job-123"
    assert platform.http_client.request.call_args.args == (
        "GET",
        "http://test:8000/apis/evaluator/v2/workspaces/ws/evaluate/jobs/job-123",
    )


def test_sync_resource_propagates_missing_job_response() -> None:
    platform = _SyncPlatform()
    platform.http_client.request.return_value = _empty_response(
        "GET",
        "http://test:8000/apis/evaluator/v2/workspaces/ws/evaluate/jobs/missing",
        status_code=404,
    )
    resource = Evaluator(platform)

    with pytest.raises(NotFoundError) as exc_info:
        resource.get_job_resource("missing", workspace="ws")

    assert exc_info.value.status_code == 404


def test_sync_resource_url_encodes_reserved_chars_in_job_name() -> None:
    """Reserved URL characters in ``job_name`` must be percent-encoded so the path stays unambiguous."""
    platform = _SyncPlatform()
    platform.http_client.request.return_value = _json_response(
        "GET",
        "http://test:8000/apis/evaluator/v2/workspaces/ws/evaluate/jobs/job%2F123%3F",
        {"name": "job/123?", "status": "created", "spec": _EXACT_MATCH_SPEC},
    )
    resource = Evaluator(platform)

    resource.get_job_resource("job/123?", workspace="ws")

    assert platform.http_client.request.call_args.args == (
        "GET",
        "http://test:8000/apis/evaluator/v2/workspaces/ws/evaluate/jobs/job%2F123%3F",
    )


class TestEvaluatorSubmit:
    """Tests for ``Evaluator.submit`` request construction."""

    def test_builds_request_from_unpacked_fields(self, mocker: MockerFixture) -> None:
        """Submit should forward public fields to the executor explicitly."""
        platform = _SyncPlatform()
        resource = Evaluator(platform)
        expected_job = mocker.Mock(spec=EvaluatorJobResource)
        submit = mocker.patch.object(resource._executor, "submit", return_value=expected_job)
        metric = ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.output}}")
        dataset = [{"expected": "a", "output": "a"}]
        model = Model(url="https://model.test/v1", name="model-a")
        config = RunConfigOnlineModel(parallelism=3, limit_samples=5)

        packager = CloudpickleMetricBundlePackager()

        job = resource.submit(
            metric=metric,
            dataset=dataset,
            config=config,
            target=model,
            prompt_template={"template": "Answer {{item.input}}"},
            metric_bundle_packager=packager,
        )

        assert job is expected_job
        submit.assert_called_once_with(
            metric=metric,
            dataset=dataset,
            params=config,
            target=model,
            field_mapping=None,
            prompt_template={"template": "Answer {{item.input}}"},
            metric_bundle_packager=packager,
        )

    def test_accepts_fileset_ref_dataset(self, mocker: MockerFixture) -> None:
        """Submit should forward FilesetRef datasets unchanged to the executor."""
        platform = _SyncPlatform()
        resource = Evaluator(platform)
        expected_job = mocker.Mock(spec=EvaluatorJobResource)
        submit = mocker.patch.object(resource._executor, "submit", return_value=expected_job)
        metric = ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.output}}")
        dataset = FilesetRef(root="default/helpsteer2")

        packager = CloudpickleMetricBundlePackager()

        job = resource.submit(metric=metric, dataset=dataset, metric_bundle_packager=packager)

        assert job is expected_job
        submit.assert_called_once_with(
            metric=metric,
            dataset=dataset,
            params=None,
            target=None,
            field_mapping=None,
            prompt_template=None,
            metric_bundle_packager=packager,
        )

    def test_accepts_model_ref_target(self, mocker: MockerFixture) -> None:
        """Submit should forward platform ModelRef targets to the plugin executor."""
        platform = _SyncPlatform()
        resource = Evaluator(platform)
        expected_job = mocker.Mock(spec=EvaluatorJobResource)
        submit = mocker.patch.object(resource._executor, "submit", return_value=expected_job)
        metric = ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.output}}")
        dataset = [{"expected": "a", "output": "a"}]
        model_ref = ModelRef(root="default/model-a")
        packager = CloudpickleMetricBundlePackager()

        job = resource.submit(
            metric=metric,
            dataset=dataset,
            config=RunConfigOnlineModel(),
            target=model_ref,
            field_mapping=None,
            prompt_template="Answer: {{item.input}}",
            metric_bundle_packager=packager,
        )

        assert job is expected_job
        submit.assert_called_once_with(
            metric=metric,
            dataset=dataset,
            params=RunConfigOnlineModel(),
            target=model_ref,
            field_mapping=None,
            prompt_template="Answer: {{item.input}}",
            metric_bundle_packager=packager,
        )

    def test_defaults_to_inline_packager_for_builtin_metric(self, mocker: MockerFixture) -> None:
        """Submit of a built-in metric without an explicit packager defaults to inline bundling."""
        resource = Evaluator(_SyncPlatform())
        expected_job = mocker.Mock(spec=EvaluatorJobResource)
        submit = mocker.patch.object(resource._executor, "submit", return_value=expected_job)

        job = resource.submit(
            metric=ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.output}}"),
            dataset=[{"expected": "a", "output": "a"}],
        )

        assert job is expected_job
        assert isinstance(submit.call_args.kwargs["metric_bundle_packager"], InlineMetricBundlePackager)

    def test_requires_explicit_packager_for_custom_metric(self) -> None:
        """Submit of a custom metric requires an explicit cloudpickle opt-in."""
        resource = Evaluator(_SyncPlatform())

        with pytest.raises(MetricBundlePackagerPolicyError, match="CloudpickleMetricBundlePackager"):
            resource.submit(
                metric=cast(Metric, _CustomRuntimeMetric()),
                dataset=[{"expected": "a", "output": "a"}],
            )


def test_sync_executor_submit_resolves_model_ref_before_creating_job(mocker: MockerFixture) -> None:
    platform = _SyncPlatform()
    models_client = mocker.Mock(spec=ModelsClient)
    models_client.resolve_model_reference.return_value = ResolvedModelReference(
        url="https://igw.example.test/v1/chat/completions",
        name="model-a",
        host_url=None,
    )
    mocker.patch("nemo_evaluator.sdk._executor.ModelsClient.from_client", return_value=models_client)
    executor = _SyncEvaluatorPluginExecutor(client=platform)
    expected_job = mocker.Mock(spec=EvaluatorJobResource)
    create = mocker.patch.object(executor, "create", return_value=expected_job)
    resolved_model = Model(url="https://igw.example.test/v1/chat/completions", name="model-a")
    metric = ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.output}}")
    dataset = [{"expected": "a", "output": "a"}]

    job = executor.submit(
        metric=metric,
        dataset=dataset,
        params=RunConfigOnlineModel(),
        target=ModelRef(root="default/model-a"),
        metric_bundle_packager=CloudpickleMetricBundlePackager(),
    )

    assert job is expected_job
    models_client.resolve_model_reference.assert_called_once_with("default/model-a")
    created_spec = create.call_args.kwargs["spec"]
    assert created_spec.target == resolved_model


def test_sync_executor_submit_requires_online_model_params_for_model_ref() -> None:
    executor = _SyncEvaluatorPluginExecutor(client=_SyncPlatform())

    with pytest.raises(TypeError, match="ModelRef target requires RunConfigOnlineModel"):
        executor.submit(
            metric=ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.output}}"),
            dataset=[{"expected": "a", "output": "a"}],
            params=RunConfig(),
            target=ModelRef(root="default/model-a"),
            metric_bundle_packager=CloudpickleMetricBundlePackager(),
        )


def test_sync_executor_submit_rejects_online_params_without_target() -> None:
    executor = _SyncEvaluatorPluginExecutor(client=_SyncPlatform())

    with pytest.raises(TypeError, match="offline evaluation requires RunConfig"):
        executor.submit(
            metric=ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.output}}"),
            dataset=[{"expected": "a", "output": "a"}],
            params=RunConfigOnline(),
            metric_bundle_packager=CloudpickleMetricBundlePackager(),
        )


@pytest.mark.asyncio
async def test_async_resource_calls_evaluator_plugin_status() -> None:
    platform = _AsyncPlatform()
    platform.http_client.request.return_value = _json_response(
        "GET",
        "http://test:8000/apis/evaluator/v1/healthz",
        {"plugin": "evaluator", "status": "ok"},
    )

    resource = AsyncEvaluator(platform)

    assert await resource.plugin_status() == {"plugin": "evaluator", "status": "ok"}
    platform.http_client.request.assert_awaited_once()
    assert platform.http_client.request.call_args.args == (
        "GET",
        "http://test:8000/apis/evaluator/v1/healthz",
    )
    assert platform.http_client.request.call_args.kwargs["headers"] == {"Authorization": "Bearer platform-token"}
    assert platform.http_client.request.call_args.kwargs["timeout"] == platform._timeout


@pytest.mark.asyncio
async def test_async_resource_rejects_non_object_plugin_status() -> None:
    platform = _AsyncPlatform()
    platform.http_client.request.return_value = _json_response(
        "GET", "http://test:8000/apis/evaluator/v1/healthz", ["ok"]
    )
    resource = AsyncEvaluator(platform)

    with pytest.raises(NemoResponseValidationError):
        await resource.plugin_status()


def test_async_resource_exposes_only_evaluator_sdk_surface() -> None:
    resource = AsyncEvaluator(_AsyncPlatform())

    assert _public_resource_names(resource) == _EVALUATOR_PUBLIC_RESOURCE_NAMES


@pytest.mark.asyncio
async def test_async_executor_creates_evaluator_job() -> None:
    platform = _AsyncPlatform()
    platform.http_client.request.return_value = _json_response(
        "POST",
        "http://test:8000/apis/evaluator/v2/workspaces/ws/evaluate/jobs",
        {"name": "job-123", "status": "created", "spec": _EXACT_MATCH_SPEC},
        status_code=201,
    )
    executor = _AsyncEvaluatorPluginExecutor(client=platform)
    spec = _EXACT_MATCH_EVALUATE_INPUT_SPEC

    job = await executor.create(spec=spec, workspace="ws")

    assert isinstance(job, AsyncEvaluatorJobResource)
    assert job.name == "job-123"
    assert job.job.status == PlatformJobStatus.CREATED
    assert job.job.spec is not None
    assert _single_metric(job.job.spec).metric_type == "exact-match"
    platform.http_client.request.assert_awaited_once()
    assert platform.http_client.request.call_args.args == (
        "POST",
        "http://test:8000/apis/evaluator/v2/workspaces/ws/evaluate/jobs",
    )
    assert _request_body(platform) == {"spec": _EXACT_MATCH_EVALUATE_INPUT_SPEC_JSON}
    assert platform.http_client.request.call_args.kwargs["headers"] == {
        "Authorization": "Bearer platform-token",
        "Content-Type": "application/json",
    }
    assert not hasattr(executor_module, "asyncio")
    assert not hasattr(executor_module, "httpx")


@pytest.mark.asyncio
async def test_async_executor_waits_when_requested(mocker: MockerFixture) -> None:
    platform = _AsyncPlatform()
    platform.http_client.request.return_value = _json_response(
        "POST",
        "http://test:8000/apis/evaluator/v2/workspaces/ws/evaluate/jobs",
        {"name": "job-123", "status": "created", "spec": _EXACT_MATCH_SPEC},
        status_code=201,
    )
    wait = mocker.patch(
        "nemo_evaluator.sdk.job_resources.AsyncEvaluatorJobResource.wait_until_done",
        new=AsyncMock(),
    )
    executor = _AsyncEvaluatorPluginExecutor(client=platform)

    job = await executor.create(spec=_EXACT_MATCH_EVALUATE_INPUT_SPEC, workspace="ws", wait_until_done=True)

    assert isinstance(job, AsyncEvaluatorJobResource)
    platform.http_client.request.assert_awaited_once()
    assert _last_request_url(platform) == "http://test:8000/apis/evaluator/v2/workspaces/ws/evaluate/jobs"
    assert _request_body(platform) == {"spec": _EXACT_MATCH_EVALUATE_INPUT_SPEC_JSON}
    wait.assert_awaited_once_with(
        poll_interval_seconds=mocker.ANY,
        job_timeout_seconds=mocker.ANY,
        pending_timeout_seconds=mocker.ANY,
    )


@pytest.mark.asyncio
async def test_async_resource_gets_existing_job_resource() -> None:
    platform = _AsyncPlatform()
    platform.http_client.request.return_value = _json_response(
        "GET",
        "http://test:8000/apis/evaluator/v2/workspaces/ws/evaluate/jobs/job-123",
        {"name": "job-123", "status": "created", "spec": _EXACT_MATCH_SPEC},
    )
    resource = AsyncEvaluator(platform)

    job = await resource.get_job_resource("job-123", workspace="ws")

    assert isinstance(job, AsyncEvaluatorJobResource)
    assert job.name == "job-123"
    platform.http_client.request.assert_awaited_once()
    assert platform.http_client.request.call_args.args == (
        "GET",
        "http://test:8000/apis/evaluator/v2/workspaces/ws/evaluate/jobs/job-123",
    )


@pytest.mark.asyncio
async def test_async_resource_url_encodes_reserved_chars_in_job_name() -> None:
    """Reserved URL characters in ``job_name`` must be percent-encoded on the async path too."""
    platform = _AsyncPlatform()
    platform.http_client.request.return_value = _json_response(
        "GET",
        "http://test:8000/apis/evaluator/v2/workspaces/ws/evaluate/jobs/job%2F123%3F",
        {"name": "job/123?", "status": "created", "spec": _EXACT_MATCH_SPEC},
    )
    resource = AsyncEvaluator(platform)

    await resource.get_job_resource("job/123?", workspace="ws")

    platform.http_client.request.assert_awaited_once()
    assert platform.http_client.request.call_args.args == (
        "GET",
        "http://test:8000/apis/evaluator/v2/workspaces/ws/evaluate/jobs/job%2F123%3F",
    )


class TestAsyncEvaluatorSubmit:
    """Tests for ``AsyncEvaluator.submit`` request construction."""

    @pytest.mark.asyncio
    async def test_builds_request_from_unpacked_fields(self, mocker: MockerFixture) -> None:
        """Submit should forward public fields to the executor explicitly."""
        platform = _AsyncPlatform()
        resource = AsyncEvaluator(platform)
        expected_job = mocker.Mock(spec=AsyncEvaluatorJobResource)
        submit = mocker.patch.object(resource._executor, "submit", new=AsyncMock(return_value=expected_job))
        metric = ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.output}}")
        dataset = [{"expected": "a", "output": "a"}]
        model = Model(url="https://model.test/v1", name="model-a")
        config = RunConfigOnlineModel(parallelism=3, limit_samples=5)

        packager = CloudpickleMetricBundlePackager()

        job = await resource.submit(
            metric=metric,
            dataset=dataset,
            config=config,
            target=model,
            prompt_template={"template": "Answer {{item.input}}"},
            metric_bundle_packager=packager,
        )

        assert job is expected_job
        submit.assert_awaited_once_with(
            metric=metric,
            dataset=dataset,
            params=config,
            target=model,
            field_mapping=None,
            prompt_template={"template": "Answer {{item.input}}"},
            metric_bundle_packager=packager,
        )

    @pytest.mark.asyncio
    async def test_accepts_fileset_ref_dataset(self, mocker: MockerFixture) -> None:
        """Submit should forward FilesetRef datasets unchanged to the executor."""
        platform = _AsyncPlatform()
        resource = AsyncEvaluator(platform)
        expected_job = mocker.Mock(spec=AsyncEvaluatorJobResource)
        submit = mocker.patch.object(resource._executor, "submit", new=AsyncMock(return_value=expected_job))
        metric = ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.output}}")
        dataset = FilesetRef(root="default/helpsteer2")

        packager = CloudpickleMetricBundlePackager()

        job = await resource.submit(metric=metric, dataset=dataset, metric_bundle_packager=packager)

        assert job is expected_job
        submit.assert_awaited_once_with(
            metric=metric,
            dataset=dataset,
            params=None,
            target=None,
            field_mapping=None,
            prompt_template=None,
            metric_bundle_packager=packager,
        )

    @pytest.mark.asyncio
    async def test_accepts_model_ref_target(self, mocker: MockerFixture) -> None:
        """Submit should forward platform ModelRef targets to the plugin executor."""
        platform = _AsyncPlatform()
        resource = AsyncEvaluator(platform)
        expected_job = mocker.Mock(spec=AsyncEvaluatorJobResource)
        submit = mocker.patch.object(resource._executor, "submit", new=AsyncMock(return_value=expected_job))
        metric = ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.output}}")
        dataset = [{"expected": "a", "output": "a"}]
        model_ref = ModelRef(root="default/model-a")
        packager = CloudpickleMetricBundlePackager()

        job = await resource.submit(
            metric=metric,
            dataset=dataset,
            config=RunConfigOnlineModel(),
            target=model_ref,
            field_mapping=None,
            prompt_template="Answer: {{item.input}}",
            metric_bundle_packager=packager,
        )

        assert job is expected_job
        submit.assert_awaited_once_with(
            metric=metric,
            dataset=dataset,
            params=RunConfigOnlineModel(),
            target=model_ref,
            field_mapping=None,
            prompt_template="Answer: {{item.input}}",
            metric_bundle_packager=packager,
        )

    @pytest.mark.asyncio
    async def test_defaults_to_inline_packager_for_builtin_metric(self, mocker: MockerFixture) -> None:
        """Async submit of a built-in metric defaults to inline bundling."""
        resource = AsyncEvaluator(_AsyncPlatform())
        expected_job = mocker.Mock(spec=AsyncEvaluatorJobResource)
        submit = mocker.patch.object(resource._executor, "submit", new=AsyncMock(return_value=expected_job))

        job = await resource.submit(
            metric=ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.output}}"),
            dataset=[{"expected": "a", "output": "a"}],
        )

        assert job is expected_job
        assert isinstance(submit.call_args.kwargs["metric_bundle_packager"], InlineMetricBundlePackager)

    @pytest.mark.asyncio
    async def test_requires_explicit_packager_for_custom_metric(self) -> None:
        """Async submit of a custom metric requires an explicit cloudpickle opt-in."""
        resource = AsyncEvaluator(_AsyncPlatform())

        with pytest.raises(MetricBundlePackagerPolicyError, match="CloudpickleMetricBundlePackager"):
            await resource.submit(
                metric=cast(Metric, _CustomRuntimeMetric()),
                dataset=[{"expected": "a", "output": "a"}],
            )


@pytest.mark.asyncio
async def test_async_executor_remote_submit_uses_platform_async_client_headers_and_timeout() -> None:
    platform = _AsyncPlatform()
    platform.http_client.request.return_value = _json_response(
        "POST",
        "http://test:8000/apis/evaluator/v2/workspaces/ws/evaluate/jobs",
        {"name": "job-123", "status": "created", "spec": _EXACT_MATCH_SPEC},
        status_code=201,
    )
    executor = _AsyncEvaluatorPluginExecutor(client=platform)

    job = await executor.create(spec=_EXACT_MATCH_EVALUATE_INPUT_SPEC, workspace="ws")

    assert job.name == "job-123"
    platform.http_client.request.assert_awaited_once()
    assert platform.http_client.request.call_args.args == (
        "POST",
        "http://test:8000/apis/evaluator/v2/workspaces/ws/evaluate/jobs",
    )
    assert _request_body(platform) == {"spec": _EXACT_MATCH_EVALUATE_INPUT_SPEC_JSON}
    assert platform.http_client.request.call_args.kwargs["headers"] == {
        "Authorization": "Bearer platform-token",
        "Content-Type": "application/json",
    }
    assert platform.http_client.request.call_args.kwargs["timeout"] == platform._timeout
    assert not hasattr(executor_module, "httpx")


@pytest.mark.asyncio
async def test_async_executor_submit_resolves_model_ref_before_creating_job(mocker: MockerFixture) -> None:
    platform = _AsyncPlatform()
    models_client = mocker.Mock(spec=AsyncModelsClient)
    models_client.resolve_model_reference = AsyncMock(
        return_value=ResolvedModelReference(
            url="https://igw.example.test/v1/chat/completions",
            name="model-a",
            host_url=None,
        )
    )
    mocker.patch("nemo_evaluator.sdk._executor.AsyncModelsClient.from_client", return_value=models_client)
    executor = _AsyncEvaluatorPluginExecutor(client=platform)
    expected_job = mocker.Mock(spec=AsyncEvaluatorJobResource)
    create = mocker.patch.object(executor, "create", new=AsyncMock(return_value=expected_job))
    resolved_model = Model(url="https://igw.example.test/v1/chat/completions", name="model-a")
    metric = ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.output}}")
    dataset = [{"expected": "a", "output": "a"}]

    job = await executor.submit(
        metric=metric,
        dataset=dataset,
        params=RunConfigOnlineModel(),
        target=ModelRef(root="default/model-a"),
        metric_bundle_packager=CloudpickleMetricBundlePackager(),
    )

    assert job is expected_job
    models_client.resolve_model_reference.assert_awaited_once_with("default/model-a")
    created_spec = create.call_args.kwargs["spec"]
    assert created_spec.target == resolved_model


@pytest.mark.asyncio
async def test_async_executor_submit_rejects_online_params_without_target() -> None:
    executor = _AsyncEvaluatorPluginExecutor(client=_AsyncPlatform())

    with pytest.raises(TypeError, match="offline evaluation requires RunConfig"):
        await executor.submit(
            metric=ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.output}}"),
            dataset=[{"expected": "a", "output": "a"}],
            params=RunConfigOnline(),
            metric_bundle_packager=CloudpickleMetricBundlePackager(),
        )
