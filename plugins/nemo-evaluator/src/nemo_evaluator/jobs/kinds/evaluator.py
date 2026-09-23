# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve and prepare evaluator tasks, including Gym rows and inline datasets."""

from collections.abc import Sequence

from nemo_evaluator.api.task_definitions.evaluator import ResolvedEvaluatorTaskDefinition
from nemo_evaluator.api.task_definitions.provenance import TaskProvenance
from nemo_evaluator.jobs.agent_spec import (
    AgentTarget,
    FabricRunnerTarget,
    GymRunnerTarget,
    HarborRunnerTarget,
    ModelTarget,
    ResolvedTask,
    Target,
)
from nemo_evaluator.jobs.kinds.types import LoadedTask, PrepareContext, SubmitContext, TaskKindAdapter
from nemo_evaluator.jobs.metric_resolution import (
    require_resolved_model_refs,
    resolve_metrics_to_inline,
    to_runtime_metrics,
)
from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalTask
from nemo_evaluator_sdk.agent_eval.trials import AgentEvalTrial


def _to_runtime_task(task: ResolvedTask) -> AgentEvalTask:
    """Convert an evaluator snapshot to SDK task fields, unpacking metrics and flattening metadata."""
    definition = task.spec
    if not isinstance(definition, ResolvedEvaluatorTaskDefinition):
        raise ValueError("Expected an evaluator task")
    return AgentEvalTask(
        id=task.id,
        intent=definition.intent,
        inputs=definition.inputs.model_dump(exclude_none=True),
        reference=definition.reference,
        metrics=to_runtime_metrics(definition.metrics),
        views=definition.views,
        metadata={item.key: item.value for item in task.metadata},
    )


class EvaluatorTaskAdapter(TaskKindAdapter):
    kind: str = "evaluator"

    def runtime_id(self, item: LoadedTask) -> str:
        """Use the inline task ID or stored entity name while snapshotting on the API server."""
        if item.inline is not None:
            return item.inline.id
        assert item.stored is not None
        return item.stored[0].name

    async def resolve(self, item: LoadedTask, ctx: SubmitContext) -> ResolvedEvaluatorTaskDefinition:
        """Resolve an evaluator definition and its metrics on the API server.

        Inline metrics resolve in the submission workspace; stored metrics resolve in the task's
        workspace. Stored definitions retain the selected revision's provenance; the caller copies task metadata.
        """
        if item.inline is not None:
            definition = item.inline
            workspace = ctx.workspace
            provenance = None
        else:
            assert item.stored is not None
            head, revision = item.stored
            if revision.spec.kind != "evaluator":
                raise ValueError("Expected an evaluator definition")
            definition = revision.spec
            workspace = head.workspace
            provenance = TaskProvenance(
                entity_name=f"{head.workspace}/{head.name}", revision_digest=revision.content_hash
            )
        metrics = await resolve_metrics_to_inline(
            definition.metrics, workspace=workspace, entity_client=ctx.entity_client, async_sdk=ctx.async_sdk
        )
        return ResolvedEvaluatorTaskDefinition(
            kind="evaluator",
            provenance=provenance,
            intent=definition.intent,
            inputs=definition.inputs.model_copy(deep=True),
            reference=definition.model_dump(include={"reference"})["reference"],
            metrics=metrics,
            views={key: view.model_copy(deep=True) for key, view in definition.views.items()},
        )

    def accepts_target(self, target: Target | None, tasks: Sequence[ResolvedTask]) -> bool:
        """Accept known evaluator targets, restricting Harbor to inline tasks."""
        if target is None:
            return True
        if isinstance(target, (ModelTarget, AgentTarget, FabricRunnerTarget, GymRunnerTarget)):
            return True
        if isinstance(target, HarborRunnerTarget):
            return all(task.spec.provenance is None for task in tasks)
        return False

    def validate_scoring(
        self, tasks: Sequence[ResolvedTask], *, target: Target | None, trials: Sequence[AgentEvalTrial] | None
    ) -> None:
        """Reject unresolved metric model references during API and worker validation."""
        for task in tasks:
            runtime = _to_runtime_task(task)
            require_resolved_model_refs(runtime.metrics, subject=f"AgentEvalSpec task {task.id!r}")

    def prepare(self, tasks: Sequence[ResolvedTask], ctx: PrepareContext) -> list[AgentEvalTask]:
        """Convert evaluator snapshots to SDK tasks on the worker without external materialization."""
        return [_to_runtime_task(task) for task in tasks]
