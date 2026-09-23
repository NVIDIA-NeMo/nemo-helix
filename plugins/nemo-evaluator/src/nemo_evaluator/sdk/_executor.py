# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private evaluator plugin executor implementation shared by SDK resources."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from models import parse_workspace_name_ref
from nemo_evaluator.api.schemas import MetricInline, TasksetRef
from nemo_evaluator.filesets import FilesetRef
from nemo_evaluator.jobs.agent_spec import AgentEvalInputSpec, GymPlacement
from nemo_evaluator.jobs.evaluate import EvaluateInputSpec, TargetSpec
from nemo_evaluator.jobs.runner_targets import runner_to_target
from nemo_evaluator.sdk.job_resources import (
    AgentEvaluatorJob,
    AgentEvaluatorJobResource,
    AsyncEvaluatorJobResource,
    EvaluatorJob,
    EvaluatorJobResource,
)
from nemo_evaluator.sdk.types import PluginDatasetInput
from nemo_evaluator.shared.metric_bundles.bundles import (
    MetricBundle,
    MetricBundlePackager,
    MetricBundlePackagerPolicyError,
    bundle_metric,
)
from nemo_evaluator_sdk.agent_eval.trials import AgentTaskRunner
from nemo_evaluator_sdk.datasets.loader import prepare_dataset_rows
from nemo_evaluator_sdk.execution.config import resolve_params
from nemo_evaluator_sdk.execution.utils import is_metric, is_metric_sequence
from nemo_evaluator_sdk.metrics.protocol import Metric
from nemo_evaluator_sdk.values import (
    Agent,
    FieldMapping,
    Model,
    ModelRef,
    RunConfig,
    RunConfigOnline,
    RunConfigOnlineModel,
)
from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.client.errors import NotFoundError as ClientNotFoundError
from nemo_helix_plugin.evaluator.client import AsyncEvaluatorClient, EvaluatorClient
from nemo_helix_plugin.evaluator.types import SubmitAgentEvalJobRequest, SubmitEvaluateJobRequest
from nemo_helix_plugin.models.client import AsyncModelsClient, ModelsClient

_DEFAULT_POLL_INTERVAL_SECONDS = 10.0
_DEFAULT_JOB_TIMEOUT_SECONDS = 3600.0
_DEFAULT_PENDING_TIMEOUT_SECONDS = 600.0

SubmitTargetSpec = TargetSpec | ModelRef


def create_job_payload(spec: EvaluateInputSpec) -> dict[str, dict[str, Any]]:
    """Serialize an evaluator job creation request body.

    Lives here rather than in the client adapter because it is the only thing there that needed
    ``EvaluateInputSpec``, and that import made the adapter — and so everything importing it,
    including ``intake.publish`` — pull in ``jobs.evaluate``, which cycles for any module that
    ``jobs.evaluate`` itself imports.
    """
    return {"spec": spec.model_dump(mode="json")}


def _require_metric_bundle_packager(metric_bundle_packager: MetricBundlePackager | None) -> MetricBundlePackager:
    if metric_bundle_packager is None:
        raise MetricBundlePackagerPolicyError(
            "Packaging runtime metrics for evaluator plugin submission requires an explicit metric_bundle_packager. "
            "Pass CloudpickleMetricBundlePackager() to opt in to cloudpickle metric bundles."
        )
    return metric_bundle_packager


def _submit_params(
    params: RunConfig | RunConfigOnline | RunConfigOnlineModel | None,
    target: SubmitTargetSpec | None,
) -> RunConfig | RunConfigOnline | RunConfigOnlineModel:
    if isinstance(target, ModelRef):
        if not isinstance(params, RunConfigOnlineModel):
            raise TypeError("ModelRef target requires RunConfigOnlineModel")
        return params
    return resolve_params(params, target)


def _resolve_submit_target(
    models_client: ModelsClient,
    target: SubmitTargetSpec | None,
) -> Model | Agent | None:
    if isinstance(target, ModelRef):
        workspace, name = parse_workspace_name_ref(
            target.root, label="ModelRef", expected_format="workspace/model_name"
        )
        try:
            resolved = models_client.resolve_model_reference(target.root)
        except ClientNotFoundError as exc:
            raise ValueError(
                f"Model reference '{target.root}' not found. "
                f"Ensure the model entity '{name}' exists in workspace '{workspace}', "
                "or use an inline model definition instead."
            ) from exc
        return Model(url=resolved.url, name=resolved.name, host_url=resolved.host_url)
    return target


async def _resolve_submit_target_async(
    models_client: AsyncModelsClient,
    target: SubmitTargetSpec | None,
) -> Model | Agent | None:
    if isinstance(target, ModelRef):
        workspace, name = parse_workspace_name_ref(
            target.root, label="ModelRef", expected_format="workspace/model_name"
        )
        try:
            resolved = await models_client.resolve_model_reference(target.root)
        except ClientNotFoundError as exc:
            raise ValueError(
                f"Model reference '{target.root}' not found. "
                f"Ensure the model entity '{name}' exists in workspace '{workspace}', "
                "or use an inline model definition instead."
            ) from exc
        return Model(url=resolved.url, name=resolved.name, host_url=resolved.host_url)
    return target


def _dataset_config(
    dataset: PluginDatasetInput,
    params: RunConfig | RunConfigOnline | RunConfigOnlineModel,
) -> list[dict[str, Any]] | FilesetRef:
    """Return the dataset payload to store in an evaluator plugin job spec."""
    if isinstance(dataset, FilesetRef):
        return dataset
    return prepare_dataset_rows(
        dataset,
        None,
        params.limit_samples,
    )


def _build_evaluate_spec(
    *,
    metrics: Metric | Sequence[Metric],
    dataset: PluginDatasetInput,
    params: RunConfig | RunConfigOnline | RunConfigOnlineModel,
    target: TargetSpec | None = None,
    field_mapping: FieldMapping | None = None,
    prompt_template: str | dict[str, Any] | None = None,
    metric_bundle_packager: MetricBundlePackager | None = None,
) -> EvaluateInputSpec:
    """Build the evaluator plugin input spec shared by local and remote execution."""
    effective_packager = _require_metric_bundle_packager(metric_bundle_packager)
    runtime_bundles = bundle_metrics_for_spec(metrics, metric_bundle_packager=effective_packager)
    spec = {
        # Carry inline metrics as the wire DTO (matches EvaluateInputSpec.metrics).
        "metrics": [MetricInline.model_validate_json(bundle.model_dump_json()) for bundle in runtime_bundles],
        "dataset": _dataset_config(dataset, params),
        "params": params.model_dump(mode="json"),
    }
    if target is not None:
        spec["target"] = target.model_dump(mode="json")
    if field_mapping is not None:
        spec["field_mapping"] = field_mapping.model_dump(mode="json")
    if prompt_template is not None:
        spec["prompt_template"] = prompt_template
    return EvaluateInputSpec.model_validate(spec)


class _SyncEvaluatorPluginExecutor:
    """Sync evaluator plugin executor used by the sync SDK resource."""

    def __init__(
        self,
        *,
        client: NemoClient,
        evaluator_client: EvaluatorClient | None = None,
        workspace: str | None = None,
        poll_interval_seconds: float = _DEFAULT_POLL_INTERVAL_SECONDS,
        job_timeout_seconds: float = _DEFAULT_JOB_TIMEOUT_SECONDS,
        pending_timeout_seconds: float = _DEFAULT_PENDING_TIMEOUT_SECONDS,
    ) -> None:
        """Store the sync service clients used for evaluator execution."""
        self._client = evaluator_client or EvaluatorClient.from_client(client)
        self._models_client = ModelsClient.from_client(client)
        self._workspace = workspace
        self._poll_interval_seconds = poll_interval_seconds
        self._job_timeout_seconds = job_timeout_seconds
        self._pending_timeout_seconds = pending_timeout_seconds

    def create(
        self,
        *,
        spec: EvaluateInputSpec,
        workspace: str | None = None,
        wait_until_done: bool = False,
    ) -> EvaluatorJobResource:
        """Create an evaluator plugin job with a sync platform client."""
        resolved_workspace = self._client.resolve_workspace(workspace)
        payload = (
            self._client.submit_evaluate_job(
                workspace=resolved_workspace,
                body=SubmitEvaluateJobRequest(spec=spec.model_dump(mode="json")),
            )
            .data()
            .model_dump(mode="json")
        )

        job_resource = EvaluatorJobResource(
            job=EvaluatorJob.model_validate(payload),
            client=self._client,
            workspace=resolved_workspace,
        )

        if wait_until_done:
            job_resource.wait_until_done(
                poll_interval_seconds=self._poll_interval_seconds,
                job_timeout_seconds=self._job_timeout_seconds,
                pending_timeout_seconds=self._pending_timeout_seconds,
            )
        return job_resource

    def create_agent_eval(
        self,
        *,
        spec: AgentEvalInputSpec,
        workspace: str | None = None,
        wait_until_done: bool = False,
    ) -> AgentEvaluatorJobResource:
        """Create an agent-evaluation job with a sync platform client.

        The twin of :meth:`create`, against the ``agent-evaluate`` collection rather than
        ``evaluate``. The two job kinds share a resource type because a submitted job is a submitted
        job — polling, status, and artifacts do not differ by what produced the trials.
        """
        resolved_workspace = self._client.resolve_workspace(workspace)
        payload = (
            self._client.submit_agent_eval_job(
                workspace=resolved_workspace,
                body=SubmitAgentEvalJobRequest(spec=spec.model_dump(mode="json")),
            )
            .data()
            .model_dump(mode="json")
        )

        job_resource = AgentEvaluatorJobResource(
            job=AgentEvaluatorJob.model_validate(payload),
            client=self._client,
            workspace=resolved_workspace,
        )

        if wait_until_done:
            job_resource.wait_until_done(
                poll_interval_seconds=self._poll_interval_seconds,
                job_timeout_seconds=self._job_timeout_seconds,
                pending_timeout_seconds=self._pending_timeout_seconds,
            )
        return job_resource

    def submit_agent_eval(
        self,
        *,
        tasks: TasksetRef,
        target: AgentTaskRunner,
        placement: GymPlacement | None = None,
        wait_until_done: bool = False,
    ) -> AgentEvaluatorJobResource:
        """Submit an agent evaluation over a stored taskset, run by ``target`` and placed by ``placement``.

        ``target`` is a *live* runner — the object someone already ran locally with
        ``AgentEvaluator()``. :func:`runner_to_target` describes it as the spec that reproduces it
        job-side, and refuses runners carrying state the wire cannot express, so a submitted job
        cannot silently run something other than what was tested.
        """
        resolved_workspace = self._client.require_workspace(self._workspace)
        spec = AgentEvalInputSpec(tasks=tasks, target=runner_to_target(target, placement))
        return self.create_agent_eval(
            spec=spec,
            workspace=resolved_workspace,
            wait_until_done=wait_until_done,
        )

    def submit(
        self,
        *,
        metric: Metric,
        dataset: PluginDatasetInput,
        params: RunConfig | RunConfigOnline | RunConfigOnlineModel | None = None,
        target: SubmitTargetSpec | None = None,
        field_mapping: FieldMapping | None = None,
        prompt_template: str | dict[str, Any] | None = None,
        metric_bundle_packager: MetricBundlePackager | None = None,
    ) -> EvaluatorJobResource:
        """Submit a remote evaluator plugin metric job and return the job resource."""
        submit_params = _submit_params(params, target)
        resolved_target = _resolve_submit_target(self._models_client, target)
        spec = _build_evaluate_spec(
            metrics=metric,
            dataset=dataset,
            params=submit_params,
            target=resolved_target,
            field_mapping=field_mapping,
            prompt_template=prompt_template,
            metric_bundle_packager=metric_bundle_packager,
        )

        resolved_workspace = self._client.require_workspace(self._workspace)
        job = self.create(spec=spec, workspace=resolved_workspace)

        return job


class _AsyncEvaluatorPluginExecutor:
    """Async evaluator plugin executor used by the async SDK resource."""

    def __init__(
        self,
        *,
        client: AsyncNemoClient,
        evaluator_client: AsyncEvaluatorClient | None = None,
        workspace: str | None = None,
        poll_interval_seconds: float = _DEFAULT_POLL_INTERVAL_SECONDS,
        job_timeout_seconds: float = _DEFAULT_JOB_TIMEOUT_SECONDS,
        pending_timeout_seconds: float = _DEFAULT_PENDING_TIMEOUT_SECONDS,
    ) -> None:
        """Store the async service clients used for evaluator execution."""
        self._client = evaluator_client or AsyncEvaluatorClient.from_client(client)
        self._models_client = AsyncModelsClient.from_client(client)
        self._workspace = workspace
        self._poll_interval_seconds = poll_interval_seconds
        self._job_timeout_seconds = job_timeout_seconds
        self._pending_timeout_seconds = pending_timeout_seconds

    async def create(
        self,
        *,
        spec: EvaluateInputSpec,
        workspace: str | None = None,
        wait_until_done: bool = False,
    ) -> AsyncEvaluatorJobResource:
        """Create an evaluator plugin job and return a high-level async job resource."""
        resolved_workspace = self._client.resolve_workspace(workspace)
        response = await self._client.submit_evaluate_job(
            workspace=resolved_workspace,
            body=SubmitEvaluateJobRequest(spec=spec.model_dump(mode="json")),
        )
        payload = response.data().model_dump(mode="json")

        job_resource = AsyncEvaluatorJobResource(
            job=EvaluatorJob.model_validate(payload),
            client=self._client,
            workspace=resolved_workspace,
        )

        if wait_until_done:
            await job_resource.wait_until_done(
                poll_interval_seconds=self._poll_interval_seconds,
                job_timeout_seconds=self._job_timeout_seconds,
                pending_timeout_seconds=self._pending_timeout_seconds,
            )
        return job_resource

    async def submit(
        self,
        *,
        metric: Metric,
        dataset: PluginDatasetInput,
        params: RunConfig | RunConfigOnline | RunConfigOnlineModel | None = None,
        target: SubmitTargetSpec | None = None,
        field_mapping: FieldMapping | None = None,
        prompt_template: str | dict[str, Any] | None = None,
        metric_bundle_packager: MetricBundlePackager | None = None,
    ) -> AsyncEvaluatorJobResource:
        """Submit a remote evaluator plugin metric job and return the job resource."""
        submit_params = _submit_params(params, target)
        resolved_target = await _resolve_submit_target_async(self._models_client, target)
        spec = _build_evaluate_spec(
            metrics=metric,
            dataset=dataset,
            params=submit_params,
            target=resolved_target,
            field_mapping=field_mapping,
            prompt_template=prompt_template,
            metric_bundle_packager=metric_bundle_packager,
        )

        resolved_workspace = self._client.require_workspace(self._workspace)
        job = await self.create(spec=spec, workspace=resolved_workspace)

        return job


def bundle_metrics_for_spec(
    metrics: Metric | Sequence[Metric], *, metric_bundle_packager: MetricBundlePackager
) -> list[MetricBundle]:
    """Package one metric or a benchmark metric sequence for an evaluator plugin spec."""
    if is_metric(metrics):
        return [bundle_metric(metrics, metric_bundle_packager)]
    if is_metric_sequence(metrics):
        return [bundle_metric(metric, metric_bundle_packager) for metric in metrics]
    raise TypeError("metrics must be a Metric or a sequence of Metric objects")
