# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare stored Harbor scoring; materialize archives only for online execution."""

import logging
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from nemo_evaluator.api.task_definitions.harbor import HarborTaskDefinition, ResolvedHarborTaskDefinition
from nemo_evaluator.harbor.tasks import StoredHarborTask
from nemo_evaluator.jobs.agent_spec import ResolvedTask
from nemo_evaluator.jobs.harbor_scoring import harbor_scoring_task
from nemo_evaluator.jobs.utils import run_with_isolated_async_client
from nemo_evaluator_sdk.agent_eval.runtimes.harbor_archive import normalize_harbor_instruction
from nemo_evaluator_sdk.agent_eval.runtimes.harbor_scoring import harbor_scoring_metrics
from nemo_evaluator_sdk.agent_eval.runtimes.harbor_tasks import HARBOR_DATASET_PATH_KEY, HARBOR_TASK_DIR_KEY
from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalTask
from nemo_evaluator_sdk.agent_eval.trials import AgentEvalTrial
from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.files.client import AsyncFilesClient, FilesClient

logger = logging.getLogger(__name__)


def prepare_stored_harbor_tasks(
    snapshots: Sequence[ResolvedTask],
    *,
    destination_root: Path,
    client: NemoClient | None,
    async_client: AsyncNemoClient | None,
    reward_key: str,
    trials: Sequence[AgentEvalTrial] | None = None,
) -> list[AgentEvalTask]:
    """Build runnable or offline-scoring tasks from resolved Harbor snapshots.

    Algorithm:
        - Reconstruct and validate scoring from the immutable submission definitions.
        - For saved trials, discover secondary rewards without downloading task archives.
        - For new trials, materialize archives and reuse their verified native metadata.
        - Match native identities and scoring inputs, then attach final materialized paths.

    Args:
        snapshots: Ordered resolved Harbor task snapshots from the submitted job.
        destination_root: Parent directory for invocation-owned materialized archives.
        client: Optional authenticated synchronous platform client.
        async_client: Optional authenticated asynchronous platform client.
        reward_key: Primary Harbor reward output name.
        trials: Saved trials for offline scoring, or ``None`` to prepare new execution.

    Returns:
        Runtime tasks in snapshot order, ready for execution or offline scoring.

    Raises:
        ValueError: Snapshots, client availability, materialized tasks, or scoring inputs are invalid.
    """
    definitions: list[ResolvedHarborTaskDefinition] = []
    for task in snapshots:
        if not isinstance(task.spec, ResolvedHarborTaskDefinition):
            raise ValueError("Expected a resolved Harbor task")
        definitions.append(task.spec)
    tasks = [harbor_scoring_task(definition, reward_key=reward_key) for definition in definitions]
    if trials is not None:
        return _prepare_offline_tasks(tasks, trials, reward_key=reward_key)

    if client is None and async_client is None:
        raise ValueError("Stored Harbor archives require an authenticated platform client")

    # The revision digest is provenance; the expanded definitions are already in the job.
    members = [
        StoredHarborTask(
            **definition.provenance.model_dump(),
            definition=HarborTaskDefinition.model_validate(definition.model_dump(exclude={"provenance"})),
        )
        for definition in definitions
    ]
    logger.info("Materializing resolved Harbor task archives")
    if async_client is not None:

        async def prepare(isolated_client: AsyncNemoClient):
            from nemo_evaluator.harbor.materialization import materialize_harbor_tasks

            return await materialize_harbor_tasks(
                members, files_client=AsyncFilesClient.from_client(isolated_client), destination_root=destination_root
            )

        materialized = run_with_isolated_async_client(async_client, prepare)
    else:
        assert client is not None
        from nemo_evaluator.harbor.materialization import materialize_harbor_tasks_sync

        materialized = materialize_harbor_tasks_sync(
            members, files_client=FilesClient.from_client(client), destination_root=destination_root
        )

    prepared = []
    for task, member in zip(tasks, materialized.members, strict=True):
        inputs = {"instruction": normalize_harbor_instruction(member.native.instruction, task_id=member.native.task_id)}
        if inputs != task.inputs or member.native.task_id != task.intent:
            raise ValueError("Harbor archive scoring inputs do not match the stored definition")
        prepared.append(
            AgentEvalTask(
                **task.model_dump(exclude={"metadata", "metrics"}),
                metrics=task.metrics,
                metadata={
                    HARBOR_TASK_DIR_KEY: str(member.task_dir),
                    HARBOR_DATASET_PATH_KEY: str(materialized.dataset_root),
                },
            )
        )
    return prepared


def _prepare_offline_tasks(
    tasks: Sequence[AgentEvalTask], trials: Sequence[AgentEvalTrial], *, reward_key: str
) -> list[AgentEvalTask]:
    """Return scoring tasks with secondary rewards discovered from the supplied trials.

    Args:
        tasks: Ordered scoring tasks to reconstruct through validation.
        trials: Saved trials grouped by task identity for reward discovery.
        reward_key: Primary Harbor reward output name.

    Returns:
        Validated tasks in input order, with expanded reward metrics.
    """
    by_task: dict[str, list[AgentEvalTrial]] = defaultdict(list)
    for trial in trials:
        by_task[trial.task_id].append(trial)
    return [
        AgentEvalTask(
            **task.model_dump(exclude={"metrics"}),
            metrics=harbor_scoring_metrics(task, by_task[task.id], reward_key=reward_key),
        )
        for task in tasks
    ]
