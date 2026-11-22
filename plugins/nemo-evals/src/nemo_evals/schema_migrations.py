# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed online upgrades; callers persist or delete the returned record."""

from nemo_evals._fileset_receipt import FilesetReceipt
from nemo_evals.entities import MetricBundleEntity
from nemo_evals.jobs.agent_spec import (
    FabricRunnerTarget,
    GymRunnerTarget,
    HarborRunnerTarget,
    Target,
    registered_agent_source,
)
from nemo_evals.taskset_runs.entities import TasksetRunEntity
from nemo_helix_plugin.files.client import AsyncFilesClient, FilesClient
from nemo_helix_plugin.files.schema_migrations import migrate_schema_fileset_id, migrate_schema_fileset_id_async


async def migrate_schema_metric(metric: MetricBundleEntity, *, files_client: AsyncFilesClient) -> MetricBundleEntity:
    """Return the metric with its associated fileset ID, preserving its entity version."""
    fileset_id = await migrate_schema_fileset_id_async(
        files_client, ref=metric.bundle_ref, fileset_id=metric.bundle_fileset_id, workspace=metric.workspace
    )
    return metric.model_copy(update={"bundle_fileset_id": fileset_id})


async def migrate_schema_taskset_run(run: TasksetRunEntity, *, files_client: AsyncFilesClient) -> TasksetRunEntity:
    """Return the run with its plan allocation ID; leave plan bytes and digest unchanged."""
    fileset_id = await migrate_schema_fileset_id_async(
        files_client, ref=run.plan_ref, fileset_id=run.plan_fileset_id, workspace=run.workspace
    )
    return run.model_copy(update={"plan_fileset_id": fileset_id})


async def migrate_schema_agent_target(target: Target | None, *, files_client: AsyncFilesClient) -> Target | None:
    """Return an upgraded local target; never rewrite a stored attempt."""
    if not isinstance(target, (FabricRunnerTarget, HarborRunnerTarget, GymRunnerTarget)):
        return target
    source = registered_agent_source(target)
    if source is None or source.files is None:
        return target
    fileset_id = await migrate_schema_fileset_id_async(
        files_client, ref=source.files.root, fileset_id=source.fileset_id
    )
    return target.model_copy(update={"source": source.model_copy(update={"fileset_id": fileset_id})})


def migrate_schema_agent_target_sync(target: Target | None, *, files_client: FilesClient) -> Target | None:
    """Apply the same local target upgrade with a synchronous Files client."""
    if not isinstance(target, (FabricRunnerTarget, HarborRunnerTarget, GymRunnerTarget)):
        return target
    source = registered_agent_source(target)
    if source is None or source.files is None:
        return target
    fileset_id = migrate_schema_fileset_id(files_client, ref=source.files.root, fileset_id=source.fileset_id)
    return target.model_copy(update={"source": source.model_copy(update={"fileset_id": fileset_id})})


async def migrate_schema_fileset_receipt(
    ref: str, *, fileset_id: str | None, files_client: AsyncFilesClient
) -> FilesetReceipt | None:
    """Return an ID-backed receipt for cleanup, or None when the fileset is already absent."""
    fileset_id = await migrate_schema_fileset_id_async(files_client, ref=ref, fileset_id=fileset_id)
    return FilesetReceipt(ref=ref, fileset_id=fileset_id) if fileset_id is not None else None
