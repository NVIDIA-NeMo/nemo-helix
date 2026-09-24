# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task revision lookup and bounded concurrency for generic loading and Harbor resolution.

These helpers read stored entities without downloading packages or starting jobs."""

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import cast

from nemo_evaluator.api.schemas import HarborTaskDefinition, TaskRef, TasksetRef, parse_subentity_ref
from nemo_evaluator.entities import TaskEntity, TaskRevisionEntity, TasksetEntity, TasksetRevisionEntity
from nemo_evaluator.harbor.tasks import StoredHarborTask
from nemo_evaluator.revisions import get_revision
from nemo_evaluator.task_identity import UnsupportedTaskKindError, qualified_task_refs
from nemo_helix_plugin.entities import EntityClientProtocol

RESOLUTION_CONCURRENCY = 16


def validate_native_task_ids(members: Sequence[StoredHarborTask]) -> None:
    """Require case-insensitively unique native Harbor task IDs.

    Args:
        members: Stored tasks whose archive-defined identities will share one dataset.

    Raises:
        ValueError: Two members have the same native task ID, ignoring case.
    """
    ids = [member.definition.native_task_id.casefold() for member in members]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate Harbor task IDs")


async def map_with_limited_concurrency[T, R](
    fn: Callable[[T], Awaitable[R]],
    items: Sequence[T],
    *,
    concurrency: int = RESOLUTION_CONCURRENCY,
) -> list[R]:
    """Apply an async function to each item with bounded concurrency.

    Algorithm:
        - Start at most ``concurrency`` workers over one shared input iterator.
        - Record each result by its input index so completion order cannot reorder output.
        - Cancel and await unfinished workers before propagating failure or cancellation.

    Args:
        fn: Async function to apply to each item.
        items: Input items to process.
        concurrency: Maximum number of calls to run at once.

    Returns:
        One result per input item, in the same order as ``items``.
    """
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")

    iterator = iter(enumerate(items))
    results: dict[int, R] = {}

    async def worker() -> None:
        for index, item in iterator:
            results[index] = await fn(item)

    workers = [asyncio.create_task(worker()) for _ in range(min(len(items), concurrency))]
    try:
        await asyncio.gather(*workers)
    finally:
        for task in workers:
            if not task.done():
                task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
    return [results[index] for index in range(len(items))]


async def task_revision(
    ref: TaskRef, entity_client: EntityClientProtocol[TasksetEntity]
) -> tuple[TaskEntity, TaskRevisionEntity]:
    """Resolve a task reference to its head and selected revision, using latest when no selector is given."""
    workspace, name, fragment = parse_subentity_ref(ref.root, "default")
    head = await cast(EntityClientProtocol[TaskEntity], entity_client).get(TaskEntity, name=name, workspace=workspace)
    revision = await get_revision(
        cast(EntityClientProtocol[TaskRevisionEntity], entity_client), TaskRevisionEntity, head, fragment
    )
    return head, revision


def pinned_members(revision: TasksetRevisionEntity) -> list[TaskRef]:
    """Return fully qualified member refs from a taskset revision.

    Args:
        revision: Immutable taskset revision containing member refs and its workspace.

    Returns:
        Member refs qualified with workspace and revision selector.

    Raises:
        ValueError: The taskset uses shared files, which stored Harbor execution does not support.
    """
    if revision.files_ref is not None:
        raise ValueError("Stored Harbor tasksets do not support shared files_ref")
    return qualified_task_refs(revision.tasks, revision.workspace)


def harbor_member(head: TaskEntity, revision: TaskRevisionEntity) -> StoredHarborTask:
    """Require a Harbor definition and attach the stored entity name and selected revision digest."""
    if not isinstance(revision.spec, HarborTaskDefinition):
        raise UnsupportedTaskKindError(f"Task {head.workspace}/{head.name!s} must have stored kind harbor")
    return StoredHarborTask(
        entity_name=f"{head.workspace}/{head.name}", revision_digest=revision.content_hash, definition=revision.spec
    )


async def resolve_harbor_taskset(
    ref: TasksetRef, *, entity_client: EntityClientProtocol[TasksetEntity], workspace: str = "default"
) -> list[StoredHarborTask]:
    """Resolve a Harbor taskset into ordered, validated materialization descriptors.

    Algorithm:
        - Resolve the selected taskset revision and qualify its member references.
        - Resolve member task revisions concurrently while preserving taskset order.
        - Require Harbor definitions and unique native task IDs before returning.

    Args:
        ref: Taskset selector to resolve.
        entity_client: Authenticated client for taskset and member revisions.
        workspace: Default workspace for an unqualified selector.

    Returns:
        Stored Harbor members in taskset order.

    Raises:
        ValueError: Membership, task kind, or native task identities are invalid.
    """
    workspace, name, fragment = parse_subentity_ref(ref.root, workspace)
    head = await entity_client.get(TasksetEntity, name=name, workspace=workspace)
    revision = await get_revision(
        cast(EntityClientProtocol[TasksetRevisionEntity], entity_client), TasksetRevisionEntity, head, fragment
    )
    refs = pinned_members(revision)

    async def resolve(ref: TaskRef) -> StoredHarborTask:
        """Return a validated Harbor descriptor for the supplied member reference."""
        head, revision = await task_revision(ref, entity_client)
        return harbor_member(head, revision)

    members = await map_with_limited_concurrency(resolve, refs)
    validate_native_task_ids(members)
    return members
