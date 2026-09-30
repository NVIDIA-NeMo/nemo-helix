# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harbor definition snapshots, reward validation, and worker preparation."""

from collections.abc import Sequence

from nemo_evaluator.api.task_definitions.harbor import ResolvedHarborTaskDefinition
from nemo_evaluator.api.task_definitions.provenance import TaskProvenance
from nemo_evaluator.harbor.preparation import prepare_stored_harbor_tasks
from nemo_evaluator.harbor.resolution import harbor_member
from nemo_evaluator.jobs.agent_spec import HarborRunnerTarget, ResolvedTask, Target
from nemo_evaluator.jobs.harbor_scoring import harbor_scoring_task
from nemo_evaluator.jobs.kinds.types import LoadedTask, PrepareContext, SubmitContext, TaskKindAdapter
from nemo_evaluator.jobs.metric_resolution import resolve_metrics_to_inline
from nemo_evaluator_sdk.agent_eval.runtimes.harbor_scoring import saved_harbor_reward_key
from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalTask
from nemo_evaluator_sdk.agent_eval.trials import AgentEvalTrial


class HarborTaskAdapter(TaskKindAdapter):
    kind: str = "harbor"

    def runtime_id(self, item: LoadedTask) -> str:
        """Use the stored definition's native Harbor task ID while snapshotting on the API server."""
        if item.stored is None or item.stored[1].spec.kind != "harbor":
            raise ValueError("Expected a stored Harbor task")
        return item.stored[1].spec.native_task_id

    async def resolve(self, item: LoadedTask, ctx: SubmitContext) -> ResolvedHarborTaskDefinition:
        """Resolve a stored Harbor definition and its metrics on the API server.

        Metric references resolve in the stored task's workspace. The definition retains the selected
        revision's provenance; the caller copies task metadata into the submission snapshot.
        """
        assert item.stored is not None
        head, revision = item.stored
        member = harbor_member(head, revision)
        metrics = await resolve_metrics_to_inline(
            member.definition.metrics,
            workspace=head.workspace,
            entity_client=ctx.entity_client,
            async_sdk=ctx.async_sdk,
        )
        return ResolvedHarborTaskDefinition(
            **member.definition.model_dump(exclude={"metrics"}),
            metrics=metrics,
            provenance=TaskProvenance(entity_name=member.entity_name, revision_digest=member.revision_digest),
        )

    def accepts_target(self, target: Target | None, tasks: Sequence[ResolvedTask]) -> bool:
        """Allow Harbor targets or trials-only scoring during API and worker validation."""
        return target is None or isinstance(target, HarborRunnerTarget)

    def validate_scoring(
        self, tasks: Sequence[ResolvedTask], *, target: Target | None, trials: Sequence[AgentEvalTrial] | None
    ) -> None:
        """Validate metrics and views against the selected reward key on the API and worker."""
        reward_key = self._reward_key(target, trials)
        for task in tasks:
            if not isinstance(task.spec, ResolvedHarborTaskDefinition):
                raise ValueError("Expected a Harbor task")
            harbor_scoring_task(definition=task.spec, reward_key=reward_key)

    def prepare(self, tasks: Sequence[ResolvedTask], ctx: PrepareContext) -> list[AgentEvalTask]:
        """Materialize Harbor archives or reconstruct offline scoring tasks on the worker."""
        return prepare_stored_harbor_tasks(
            tasks,
            destination_root=ctx.storage_root / "harbor-inputs",
            client=ctx.client,
            async_client=ctx.async_client,
            reward_key=self._reward_key(ctx.target, ctx.trials),
            trials=ctx.trials,
        )

    @staticmethod
    def _reward_key(target: Target | None, trials: Sequence[AgentEvalTrial] | None) -> str:
        return target.reward_key if isinstance(target, HarborRunnerTarget) else saved_harbor_reward_key(trials or [])
