# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare stored Harbor scoring; materialize archives only for online execution."""

import logging
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from nemo_evaluator.harbor.resolution import resolve_harbor_source, resolve_harbor_source_sync, validate_native_task_ids
from nemo_evaluator.harbor.tasks import PinnedHarborSource, StoredHarborTask
from nemo_evaluator.jobs.harbor_scoring import apply_harbor_scoring
from nemo_evaluator.jobs.utils import run_with_isolated_async_client
from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import HarborTasksetLoader, normalize_harbor_instruction
from nemo_evaluator_sdk.agent_eval.runtimes.harbor_scoring import harbor_scoring_metrics
from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalTask
from nemo_evaluator_sdk.agent_eval.trials import AgentEvalTrial
from nemo_platform_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_platform_plugin.entities import EntityClient, SyncEntityClient
from nemo_platform_plugin.entities.client import AsyncEntitiesClient, EntitiesClient
from nemo_platform_plugin.files.client import AsyncFilesClient, FilesClient

logger = logging.getLogger(__name__)


def _scoring_tasks(
    source: PinnedHarborSource, members: list[StoredHarborTask], *, reward_key: str
) -> list[AgentEvalTask]:
    validate_native_task_ids(members)
    refs = [f"{member.entity_name}#{member.revision_digest}" for member in members]
    by_ref = {entry.task_ref.root: entry for entry in source.scoring}
    if len(by_ref) != len(source.scoring) or set(by_ref) != set(refs):
        raise ValueError("Harbor scoring references must match the pinned tasks")
    return [
        apply_harbor_scoring(
            by_ref[ref],
            AgentEvalTask(
                id=member.definition.native_task_id,
                intent=member.definition.native_task_id,
                inputs={
                    "instruction": normalize_harbor_instruction(
                        member.definition.instruction, task_id=member.definition.native_task_id
                    )
                },
            ),
            reward_key=reward_key,
        )
        for ref, member in zip(refs, members, strict=True)
    ]


def prepare_stored_harbor_tasks(
    source: PinnedHarborSource,
    *,
    destination_root: Path,
    client: NemoClient | None,
    async_client: AsyncNemoClient | None,
    reward_key: str,
    trials: Sequence[AgentEvalTrial] | None = None,
) -> list[AgentEvalTask]:
    if client is None and async_client is None:
        raise ValueError("Stored Harbor tasks require an authenticated platform client")

    logger.info("Resolving stored Harbor task revisions")
    if async_client is not None:

        async def prepare(isolated_client: AsyncNemoClient):
            members = await resolve_harbor_source(
                source, entity_client=EntityClient(AsyncEntitiesClient.from_client(isolated_client))
            )
            tasks = _scoring_tasks(source, members, reward_key=reward_key)
            if trials is not None:
                return tasks, None
            from nemo_evaluator.harbor.materialization import materialize_harbor_tasks

            materialized = await materialize_harbor_tasks(
                members, files_client=AsyncFilesClient.from_client(isolated_client), destination_root=destination_root
            )
            return tasks, materialized

        tasks, materialized = run_with_isolated_async_client(async_client, prepare)
    else:
        assert client is not None
        members = resolve_harbor_source_sync(source, entity_client=SyncEntityClient(EntitiesClient.from_client(client)))
        tasks = _scoring_tasks(source, members, reward_key=reward_key)
        materialized = None
        if trials is None:
            from nemo_evaluator.harbor.materialization import materialize_harbor_tasks_sync

            materialized = materialize_harbor_tasks_sync(
                members, files_client=FilesClient.from_client(client), destination_root=destination_root
            )

    if trials is not None:
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

    assert materialized is not None
    loaded = HarborTasksetLoader(materialized.dataset_root).load().tasks
    expected = {member.task_dir.resolve(): member.task_id for member in materialized.members}
    actual: dict[Path, AgentEvalTask] = {}
    for task in loaded:
        path = Path(str(task.metadata["harbor_task_dir"])).resolve()
        if path in actual or expected.get(path) != task.id:
            raise ValueError(f"Harbor loader returned an unexpected task: {task.id!r} at {path}")
        if Path(str(task.metadata["harbor_dataset_path"])).resolve() != materialized.dataset_root.resolve():
            raise ValueError("Harbor loader returned an unexpected dataset root")
        actual[path] = task
    if len(loaded) != len(materialized.members) or actual.keys() != expected.keys():
        raise ValueError("Harbor loader omitted or replaced verified tasks")
    prepared = []
    for task, member in zip(tasks, materialized.members, strict=True):
        loaded_task = actual[member.task_dir.resolve()]
        if loaded_task.inputs != task.inputs or loaded_task.intent != task.intent:
            raise ValueError("Harbor archive scoring inputs do not match the stored definition")
        prepared.append(
            AgentEvalTask(
                **task.model_dump(exclude={"metadata", "metrics"}), metrics=task.metrics, metadata=loaded_task.metadata
            )
        )
    return prepared
