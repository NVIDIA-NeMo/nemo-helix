# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SDK resources for the evaluator plugin scaffold."""

from __future__ import annotations

from typing import Any, overload

from nemo_evaluator.api.schemas import TasksetRef
from nemo_evaluator.sdk._executor import (
    SubmitTargetSpec,
    _AsyncEvaluatorPluginExecutor,
    _SyncEvaluatorPluginExecutor,
)
from nemo_evaluator.sdk.job_resources import (
    AgentEvaluatorJobResource,
    AsyncEvaluatorJobResource,
    EvaluatorJob,
    EvaluatorJobResource,
)
from nemo_evaluator.sdk.metric_resources import (
    AsyncEvaluatorMetricsResource,
    EvaluatorMetricsResource,
)
from nemo_evaluator.sdk.result_resources import (
    AsyncEvaluatorAgentEvalResultsResource,
    AsyncEvaluatorEvalResultsResource,
    EvaluatorAgentEvalResultsResource,
    EvaluatorEvalResultsResource,
)
from nemo_evaluator.sdk.task_resources import (
    AsyncEvaluatorTasksResource,
    EvaluatorTasksResource,
)
from nemo_evaluator.sdk.taskset_resources import (
    AsyncEvaluatorTasksetsResource,
    EvaluatorTasksetsResource,
)
from nemo_evaluator.sdk.types import (
    PluginDatasetInput,
    RunConfig,
    RunConfigOnline,
    RunConfigOnlineModel,
)
from nemo_evaluator.shared.metric_bundles.bundles import MetricBundlePackager
from nemo_evaluator.shared.metric_bundles.defaults import resolve_default_metric_bundle_packager
from nemo_evaluator_sdk.agent_eval.trials import AgentTaskRunner
from nemo_evaluator_sdk.metrics.protocol import Metric
from nemo_evaluator_sdk.values import (
    Agent,
    FieldMapping,
    Model,
    ModelRef,
)
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.evaluator.client import AsyncEvaluatorClient, EvaluatorClient
from nemo_platform_plugin.sdk import AsyncNeMoPlatform, NeMoPlatform, NemoPluginSDKResources


class Evaluator:
    """Sync SDK namespace mounted as ``client.evaluator``."""

    def __init__(self, client: EvaluatorClient) -> None:
        """Store the typed evaluator client used for evaluator plugin HTTP calls."""
        self._client = client
        self._executor = _SyncEvaluatorPluginExecutor(client=self._client, evaluator_client=self._client)
        self.metrics = EvaluatorMetricsResource(self._client)
        self.agent_eval_results = EvaluatorAgentEvalResultsResource(self._client)
        self.eval_results = EvaluatorEvalResultsResource(self._client)
        self.tasks = EvaluatorTasksResource(self._client)
        self.tasksets = EvaluatorTasksetsResource(self._client)

    @classmethod
    def from_sdk(cls, sdk: NeMoPlatform) -> Evaluator:
        return cls(client_from_platform(sdk, EvaluatorClient))

    def plugin_status(self) -> dict[str, object]:
        """Return evaluator plugin health information from the service."""
        payload = self._client.get_health().data().model_dump(mode="json", exclude_none=True, exclude_unset=True)
        return {str(key): value for key, value in payload.items()}

    def get_job_resource(self, job_name: str, workspace: str | None = None) -> EvaluatorJobResource:
        """Get a high-level resource for an existing evaluator plugin job."""
        response = self._client.get_evaluate_job(name=job_name, workspace=workspace)
        return EvaluatorJobResource(
            job=EvaluatorJob.model_validate(response.data().model_dump(mode="json")),
            client=self._client,
            workspace=self._client.resolve_workspace(workspace),
        )

    @overload
    def submit(
        self,
        *,
        metric: Metric,
        dataset: PluginDatasetInput,
        config: RunConfig | None = None,
        target: None = None,
        field_mapping: FieldMapping | None = None,
        prompt_template: None = None,
        metric_bundle_packager: MetricBundlePackager | None = None,
    ) -> EvaluatorJobResource: ...

    @overload
    def submit(
        self,
        *,
        metric: Metric,
        dataset: PluginDatasetInput,
        config: RunConfigOnlineModel,
        target: Model | ModelRef,
        field_mapping: FieldMapping | None = None,
        prompt_template: str | dict[str, Any] | None = None,
        metric_bundle_packager: MetricBundlePackager | None = None,
    ) -> EvaluatorJobResource: ...

    @overload
    def submit(
        self,
        *,
        metric: Metric,
        dataset: PluginDatasetInput,
        config: RunConfigOnline,
        target: Agent,
        field_mapping: FieldMapping | None = None,
        prompt_template: str | dict[str, Any],
        metric_bundle_packager: MetricBundlePackager | None = None,
    ) -> EvaluatorJobResource: ...

    @overload
    def submit(
        self,
        *,
        tasks: TasksetRef,
        target: AgentTaskRunner,
    ) -> AgentEvaluatorJobResource: ...

    def submit(
        self,
        *,
        metric: Metric | None = None,
        dataset: PluginDatasetInput | None = None,
        tasks: TasksetRef | None = None,
        config: RunConfig | RunConfigOnline | RunConfigOnlineModel | None = None,
        target: SubmitTargetSpec | AgentTaskRunner | None = None,
        field_mapping: FieldMapping | None = None,
        prompt_template: str | dict[str, Any] | None = None,
        metric_bundle_packager: MetricBundlePackager | None = None,
    ) -> EvaluatorJobResource | AgentEvaluatorJobResource:
        """Submit an evaluation job through the evaluator plugin executor.

        Two shapes, discriminated by what you supply: ``metric`` + ``dataset`` evaluates rows, and
        ``tasks`` + ``target`` evaluates a stored taskset with a live agent runner. They are one
        method because they are one concept — the split is a property of how the work is described
        today, not of what the caller is asking for.
        """
        if tasks is not None:
            if metric is not None or dataset is not None:
                raise TypeError(
                    "submit() takes either `tasks` (a taskset evaluation) or `metric` + `dataset` "
                    "(a row evaluation), not both. Drop whichever does not describe this run."
                )
            if not isinstance(target, AgentTaskRunner):
                raise TypeError(
                    "submit(tasks=...) evaluates a taskset with an agent runner, so `target` must be "
                    f"an AgentTaskRunner; got {type(target).__name__}. Pass a runner such as "
                    "GymAgentTaskRunner(config=...)."
                )
            # Everything below configures a *row* evaluation. The taskset path cannot honour any of
            # it, so accepting it silently would run a job that ignores what the caller asked for.
            row_only = {
                "config": config,
                "field_mapping": field_mapping,
                "prompt_template": prompt_template,
                "metric_bundle_packager": metric_bundle_packager,
            }
            supplied = sorted(name for name, value in row_only.items() if value is not None)
            if supplied:
                raise TypeError(
                    f"submit(tasks=...) does not use {', '.join(supplied)} — those configure a row "
                    "evaluation. A taskset evaluation is configured by the runner passed as "
                    "`target`, so supplying them here would have no effect."
                )
            return self._executor.submit_agent_eval(tasks=tasks, target=target)
        if metric is None or dataset is None:
            raise TypeError(
                "submit() needs either `tasks` + `target` for a taskset evaluation, or `metric` + "
                "`dataset` for a row evaluation; neither was supplied."
            )
        if isinstance(target, AgentTaskRunner):
            # The mirror of the check above: a runner is only meaningful for a taskset evaluation,
            # so this is a caller who meant to pass `tasks` and passed a dataset instead. Reaching
            # the row path with it would fail much deeper, describing the runner as a model endpoint.
            raise TypeError(
                f"{type(target).__name__} is an agent runner, which runs a taskset rather than "
                "rows. Pass `tasks=TasksetRef(...)` instead of `metric` + `dataset`."
            )
        return self._executor.submit(
            metric=metric,
            dataset=dataset,
            params=config,
            target=target,
            field_mapping=field_mapping,
            prompt_template=prompt_template,
            metric_bundle_packager=resolve_default_metric_bundle_packager(
                metric, metric_bundle_packager, allow_cloudpickle_fallback=False, action="Submitting"
            ),
        )


class AsyncEvaluator:
    """Async SDK namespace mounted as ``client.evaluator``."""

    def __init__(self, client: AsyncEvaluatorClient) -> None:
        """Store the typed async evaluator client used for evaluator plugin HTTP calls."""
        self._client = client
        self._executor = _AsyncEvaluatorPluginExecutor(client=self._client, evaluator_client=self._client)
        self.metrics = AsyncEvaluatorMetricsResource(self._client)
        self.agent_eval_results = AsyncEvaluatorAgentEvalResultsResource(self._client)
        self.eval_results = AsyncEvaluatorEvalResultsResource(self._client)
        self.tasks = AsyncEvaluatorTasksResource(self._client)
        self.tasksets = AsyncEvaluatorTasksetsResource(self._client)

    @classmethod
    def from_sdk(cls, async_sdk: AsyncNeMoPlatform) -> AsyncEvaluator:
        return cls(client_from_platform(async_sdk, AsyncEvaluatorClient))

    async def plugin_status(self) -> dict[str, object]:
        """Return evaluator plugin health information from the service."""
        payload = (
            (await self._client.get_health()).data().model_dump(mode="json", exclude_none=True, exclude_unset=True)
        )
        return {str(key): value for key, value in payload.items()}

    async def get_job_resource(self, job_name: str, workspace: str | None = None) -> AsyncEvaluatorJobResource:
        """Get a high-level async resource for an existing evaluator plugin job."""
        response = await self._client.get_evaluate_job(name=job_name, workspace=workspace)
        return AsyncEvaluatorJobResource(
            job=EvaluatorJob.model_validate(response.data().model_dump(mode="json")),
            client=self._client,
            workspace=self._client.resolve_workspace(workspace),
        )

    @overload
    async def submit(
        self,
        *,
        metric: Metric,
        dataset: PluginDatasetInput,
        config: RunConfig | None = None,
        target: None = None,
        field_mapping: FieldMapping | None = None,
        prompt_template: None = None,
        metric_bundle_packager: MetricBundlePackager | None = None,
    ) -> AsyncEvaluatorJobResource: ...

    @overload
    async def submit(
        self,
        *,
        metric: Metric,
        dataset: PluginDatasetInput,
        config: RunConfigOnlineModel,
        target: Model | ModelRef,
        field_mapping: FieldMapping | None = None,
        prompt_template: str | dict[str, Any] | None = None,
        metric_bundle_packager: MetricBundlePackager | None = None,
    ) -> AsyncEvaluatorJobResource: ...

    @overload
    async def submit(
        self,
        *,
        metric: Metric,
        dataset: PluginDatasetInput,
        config: RunConfigOnline,
        target: Agent,
        field_mapping: FieldMapping | None = None,
        prompt_template: str | dict[str, Any],
        metric_bundle_packager: MetricBundlePackager | None = None,
    ) -> AsyncEvaluatorJobResource: ...

    async def submit(
        self,
        *,
        metric: Metric,
        dataset: PluginDatasetInput,
        config: RunConfig | RunConfigOnline | RunConfigOnlineModel | None = None,
        target: SubmitTargetSpec | None = None,
        field_mapping: FieldMapping | None = None,
        prompt_template: str | dict[str, Any] | None = None,
        metric_bundle_packager: MetricBundlePackager | None = None,
    ) -> AsyncEvaluatorJobResource:
        """Submit a metric job through the evaluator plugin executor."""
        return await self._executor.submit(
            metric=metric,
            dataset=dataset,
            params=config,
            target=target,
            field_mapping=field_mapping,
            prompt_template=prompt_template,
            metric_bundle_packager=resolve_default_metric_bundle_packager(
                metric, metric_bundle_packager, allow_cloudpickle_fallback=False, action="Submitting"
            ),
        )


evaluator_sdk_resources = NemoPluginSDKResources(
    sync_resource=Evaluator.from_sdk,
    async_resource=AsyncEvaluator.from_sdk,
)
