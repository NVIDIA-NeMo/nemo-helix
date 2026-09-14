# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare stored Harbor sources with the worker's authenticated clients."""

import logging
from pathlib import Path

from nemo_evaluator.harbor.resolution import resolve_harbor_source, resolve_harbor_source_sync
from nemo_evaluator.harbor.tasks import PinnedHarborSource
from nemo_evaluator.jobs.utils import run_with_isolated_async_client
from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import HarborTasksetLoader
from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalTask
from nemo_platform_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_platform_plugin.entities import EntityClient, SyncEntityClient
from nemo_platform_plugin.entities.client import AsyncEntitiesClient, EntitiesClient
from nemo_platform_plugin.files.client import AsyncFilesClient, FilesClient

logger = logging.getLogger(__name__)


def prepare_stored_harbor_tasks(
    source: PinnedHarborSource,
    *,
    destination_root: Path,
    sdk: NemoClient | None,
    async_sdk: AsyncNemoClient | None,
) -> list[AgentEvalTask]:
    if sdk is None and async_sdk is None:
        raise ValueError("Stored Harbor tasks require an authenticated platform client")

    from nemo_evaluator.harbor.materialization import materialize_harbor_tasks, materialize_harbor_tasks_sync

    logger.info("Resolving stored Harbor task revisions")
    if async_sdk is not None:

        async def prepare(client: AsyncNemoClient):
            members = await resolve_harbor_source(
                source, entity_client=EntityClient(AsyncEntitiesClient.from_client(client))
            )
            logger.info("Downloading and verifying %s Harbor packages", len(members))
            return await materialize_harbor_tasks(
                members, files_client=AsyncFilesClient.from_client(client), destination_root=destination_root
            )

        materialized = run_with_isolated_async_client(async_sdk, prepare)
    else:
        assert sdk is not None
        members = resolve_harbor_source_sync(source, entity_client=SyncEntityClient(EntitiesClient.from_client(sdk)))
        logger.info("Downloading and verifying %s Harbor packages", len(members))
        materialized = materialize_harbor_tasks_sync(
            members, files_client=FilesClient.from_client(sdk), destination_root=destination_root
        )

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
    return [actual[member.task_dir.resolve()] for member in materialized.members]
