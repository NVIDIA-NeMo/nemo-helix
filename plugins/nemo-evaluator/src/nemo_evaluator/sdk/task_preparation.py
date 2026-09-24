# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare publication inputs without registering task or metric entities."""

from nemo_evaluator.api.schemas import (
    EvaluatorTaskDefinition,
    MetadataItem,
    MetricInline,
    MetricRefOrInline,
    TaskInput,
    TaskInputs,
)
from nemo_evaluator.harbor.publication import publish_harbor_task_archive, publish_harbor_task_archive_async
from nemo_evaluator.shared.metric_bundles.bundles import MetricBundlePackager, bundle_metric
from nemo_evaluator.shared.metric_bundles.defaults import resolve_default_metric_bundle_packager
from nemo_evaluator_sdk.agent_eval.runtimes.harbor_tasks import (
    HARBOR_DATASET_PATH_KEY,
    HARBOR_TASK_DIR_KEY,
    HarborAgentEvalTask,
)
from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalTask
from nemo_evaluator_sdk.metrics.runner_rewards import HarborRewardMetric
from nemo_helix_plugin.files.client import AsyncFilesClient, FilesClient

_HARBOR_LOCATION_KEYS = frozenset({HARBOR_TASK_DIR_KEY, HARBOR_DATASET_PATH_KEY})
_RUNTIME_LOCATION_KEYS = _HARBOR_LOCATION_KEYS | {"gym_dataset_path"}


class TaskPublicationError(RuntimeError):
    """The task write failed after preparation; retry using prepared_task without uploading again."""

    def __init__(self, prepared_task: TaskInput):
        super().__init__("Task publication failed after preparation; inspect prepared_task and the chained error")
        self.prepared_task = prepared_task


def _scoring(
    task: AgentEvalTask, packager: MetricBundlePackager | None
) -> tuple[AgentEvalTask, list[MetricRefOrInline], list[MetadataItem]]:
    """Revalidate a local task and bundle the metrics it publishes.

    A discovered Harbor task publishes every metric except its mandatory reward, which the
    runner supplies. Plain tasks publish all metrics.
    """
    if isinstance(task, HarborAgentEvalTask):
        # Check the reward first so a duplicate gets this Harbor-specific message, not the generic
        # duplicate-metric-type error from revalidation.
        rewards = [metric for metric in task.metrics if isinstance(metric, HarborRewardMetric)]
        if len(rewards) != 1:
            raise ValueError(
                f"A Harbor task must have exactly one HarborRewardMetric. Provided: {len(rewards)}. "
                "HarborRewardMetric is attached to every task by default in the discover_harbor_tasks() function."
            )
        if type(rewards[0]) is not HarborRewardMetric or rewards[0].model_dump() != HarborRewardMetric().model_dump():
            raise ValueError("Customized mandatory HarborRewardMetric cannot be published")
        # Revalidate fields that model_copy or mutation may have changed. Harbor reward output names
        # depend on the submission target's reward_key, so views are checked without reward signals
        # here. Submission validates the complete views.
        HarborAgentEvalTask.model_validate(
            {
                **dict(task),
                "views": {
                    name: view.model_copy(
                        update={"signals": [signal for signal in view.signals if signal.metric != "harbor_reward"]}
                    )
                    for name, view in task.views.items()
                },
            }
        )
        metrics = [metric for metric in task.metrics if not isinstance(metric, HarborRewardMetric)]
    elif _HARBOR_LOCATION_KEYS.intersection(task.metadata):
        # Handle a potential and unlikely edge case where a Harbor task is downcast (e.g. rebuilt as AgentEvalTask).
        # Harbor discovery stamps these keys; seeing them on a plain task means a Harbor task was
        # downcast (e.g. rebuilt as AgentEvalTask)
        raise ValueError(
            f"Harbor tasks must be of type HarborAgentEvalTask to be published. Current type: {type(task)}. Use discover_harbor_tasks() to get a HarborAgentEvalTask or rebuild the task as a HarborAgentEvalTask."
        )
    else:
        task = AgentEvalTask.model_validate(dict(task))
        metrics = task.metrics
    annotations = [
        MetadataItem(key=key, value=value) for key, value in task.metadata.items() if key not in _RUNTIME_LOCATION_KEYS
    ]
    selected = resolve_default_metric_bundle_packager(
        metrics, packager, allow_cloudpickle_fallback=False, action="Storing"
    )
    return (
        task,
        [MetricInline.model_validate_json(bundle_metric(metric, selected).model_dump_json()) for metric in metrics],
        annotations,
    )


def _evaluator_input(
    task: AgentEvalTask, metrics: list[MetricRefOrInline], metadata: list[MetadataItem], *, archived: bool
) -> TaskInput:
    """Build the evaluator task input for a non-Harbor task.

    Raises:
        ValueError: If archive options were supplied, since only discovered Harbor tasks are archived.
    """
    if archived:
        raise ValueError("Archive options require a discovered Harbor task")
    return TaskInput(
        spec=EvaluatorTaskDefinition(
            kind="evaluator",
            intent=task.intent,
            inputs=TaskInputs.model_validate(task.inputs),
            reference=task.reference,
            views=task.views,
            metrics=metrics,
        ),
        metadata=metadata,
    )


def prepare_task(
    task: AgentEvalTask,
    *,
    files_client: FilesClient,
    workspace: str,
    fileset_ref: str | None = None,
    path_prefix: str | None = None,
    metric_bundle_packager: MetricBundlePackager | None = None,
) -> TaskInput:
    """Prepare one local task for storage without creating a task entity.

    Algorithm:
        - Validate and bundle the task's metrics and convert metadata to API annotations.
        - Convert plain tasks directly to an evaluator definition.
        - Capture, upload, and verify discovered Harbor tasks before attaching their metrics and views.

    Args:
        task: Plain evaluator task, or a Harbor task from ``discover_harbor_tasks``.
        files_client: Authenticated synchronous Files client used for Harbor archives.
        workspace: Workspace used by the default Harbor fileset.
        fileset_ref: Optional ``workspace/fileset`` destination for a Harbor archive.
        path_prefix: Optional path prefix within the Harbor fileset.
        metric_bundle_packager: Optional explicit packager for executable metrics.

    Returns:
        API-ready task input containing bundled metrics and any uploaded Harbor source.

    Raises:
        ValueError: Archive options are used with a plain task or task validation fails.
    """
    validated, metrics, metadata = _scoring(task, metric_bundle_packager)
    if not isinstance(task, HarborAgentEvalTask):
        return _evaluator_input(
            validated, metrics, metadata, archived=fileset_ref is not None or path_prefix is not None
        )
    definition = publish_harbor_task_archive(
        task.source_dir,
        files_client=files_client,
        fileset_ref=fileset_ref if fileset_ref is not None else f"{workspace}/harbor-tasks",
        path_prefix=path_prefix or "",
        verify_against=task,
    )
    definition.metrics = metrics
    definition.views = task.views.copy()
    return TaskInput(spec=definition, metadata=metadata)


async def prepare_task_async(
    task: AgentEvalTask,
    *,
    files_client: AsyncFilesClient,
    workspace: str,
    fileset_ref: str | None = None,
    path_prefix: str | None = None,
    metric_bundle_packager: MetricBundlePackager | None = None,
) -> TaskInput:
    """Asynchronously prepare one local task without creating a task entity.

    Algorithm:
        - Validate and bundle the task's metrics and convert metadata to API annotations.
        - Convert plain tasks directly without performing Files operations.
        - Asynchronously upload and verify discovered Harbor tasks, then attach their metrics and views.

    Args:
        task: Plain evaluator task, or a Harbor task from ``discover_harbor_tasks``.
        files_client: Authenticated asynchronous Files client used for Harbor archives.
        workspace: Workspace used by the default Harbor fileset.
        fileset_ref: Optional ``workspace/fileset`` destination for a Harbor archive.
        path_prefix: Optional path prefix within the Harbor fileset.
        metric_bundle_packager: Optional explicit packager for executable metrics.

    Returns:
        API-ready task input containing bundled metrics and any uploaded Harbor source.

    Raises:
        ValueError: Archive options are used with a plain task or task validation fails.
    """
    validated, metrics, metadata = _scoring(task, metric_bundle_packager)
    if not isinstance(task, HarborAgentEvalTask):
        return _evaluator_input(
            validated, metrics, metadata, archived=fileset_ref is not None or path_prefix is not None
        )
    definition = await publish_harbor_task_archive_async(
        task.source_dir,
        files_client=files_client,
        fileset_ref=fileset_ref if fileset_ref is not None else f"{workspace}/harbor-tasks",
        path_prefix=path_prefix or "",
        verify_against=task,
    )
    definition.metrics = metrics
    definition.views = task.views.copy()
    return TaskInput(spec=definition, metadata=metadata)
