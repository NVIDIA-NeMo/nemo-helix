# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve pinned Harbor sources without downloading packages or starting jobs."""

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import cast

from nemo_evaluator.api.schemas import HarborTaskDefinition, TaskRef, TasksetRef, parse_subentity_ref
from nemo_evaluator.entities import TaskEntity, TaskRevisionEntity, TasksetEntity, TasksetRevisionEntity
from nemo_evaluator.harbor.tasks import (
    PinnedHarborSource,
    PinnedHarborTaskList,
    PinnedHarborTaskset,
    StoredHarborTask,
    qualified_task_refs,
    require_pin,
    require_task_kind,
)
from nemo_evaluator.revisions import RevisionNotFoundError, _verify_content, get_revision
from nemo_platform_plugin.entities import EntityClientProtocol, SyncEntityClient
from nemo_platform_plugin.filter_ops import ComparisonOperation, FilterOperator, LogicalOperation

RESOLUTION_CONCURRENCY = 16


@dataclass(frozen=True)
class ResolvedHarborSelection:
    """Resolved selectors before metric resolution and canonical job construction."""

    members: list[StoredHarborTask]
    taskset_ref: TasksetRef | None = None

    def __post_init__(self) -> None:
        validate_native_task_ids(self.members)


def validate_native_task_ids(members: Sequence[StoredHarborTask]) -> None:
    ids = [member.definition.native_task_id.casefold() for member in members]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate Harbor task IDs")


async def bounded_ordered_map[T, R](fn: Callable[[T], Awaitable[R]], items: Sequence[T]) -> list[R]:
    """Bound pending tasks as well as active requests; drain on failure/cancellation."""
    iterator = iter(enumerate(items))
    results: dict[int, R] = {}

    async def worker() -> None:
        for index, item in iterator:
            results[index] = await fn(item)

    workers = [asyncio.create_task(worker()) for _ in range(min(len(items), RESOLUTION_CONCURRENCY))]
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
    workspace, name, fragment = parse_subentity_ref(ref.root, "default")
    head = await cast(EntityClientProtocol[TaskEntity], entity_client).get(TaskEntity, name=name, workspace=workspace)
    revision = await get_revision(
        cast(EntityClientProtocol[TaskRevisionEntity], entity_client), TaskRevisionEntity, head, fragment
    )
    return head, revision


async def taskset_revision(
    ref: TasksetRef, entity_client: EntityClientProtocol[TasksetEntity], workspace: str = "default"
) -> tuple[TasksetEntity, TasksetRevisionEntity]:
    workspace, name, fragment = parse_subentity_ref(ref.root, workspace)
    head = await entity_client.get(TasksetEntity, name=name, workspace=workspace)
    revision = await get_revision(
        cast(EntityClientProtocol[TasksetRevisionEntity], entity_client), TasksetRevisionEntity, head, fragment
    )
    pinned_members(revision)
    return head, revision


def pinned_members(revision: TasksetRevisionEntity) -> list[TaskRef]:
    if revision.files_ref is not None:
        raise ValueError("Stored Harbor tasksets do not support shared files_ref")
    for ref in revision.tasks:
        require_pin(ref)
    return qualified_task_refs(revision.tasks, revision.workspace)


def harbor_member(head: TaskEntity, revision: TaskRevisionEntity) -> StoredHarborTask:
    require_task_kind(f"{head.workspace}/{head.name}", revision.spec.kind, "harbor")
    assert isinstance(revision.spec, HarborTaskDefinition)
    return StoredHarborTask(
        entity_name=f"{head.workspace}/{head.name}", revision_digest=revision.content_hash, definition=revision.spec
    )


async def resolve_harbor_source(
    source: PinnedHarborSource, *, entity_client: EntityClientProtocol[TasksetEntity]
) -> list[StoredHarborTask]:
    # Revalidate mutable model instances at this trust boundary.
    if isinstance(source, PinnedHarborTaskset):
        source = PinnedHarborTaskset.model_validate(source.model_dump())
        _, revision = await taskset_revision(source.taskset_ref, entity_client)
        refs = pinned_members(revision)
    else:
        source = PinnedHarborTaskList.model_validate(source.model_dump())
        refs = source.task_refs

    return await resolve_harbor_members(refs, entity_client=entity_client)


async def resolve_harbor_members(
    refs: Sequence[TaskRef], *, entity_client: EntityClientProtocol[TasksetEntity]
) -> list[StoredHarborTask]:
    """Resolve task references without requiring a compiled scoring configuration."""

    async def resolve(ref: TaskRef) -> StoredHarborTask:
        head, revision = await task_revision(ref, entity_client)
        return harbor_member(head, revision)

    return await bounded_ordered_map(resolve, refs)


async def resolve_harbor_taskset(
    ref: TasksetRef, *, entity_client: EntityClientProtocol[TasksetEntity], workspace: str = "default"
) -> ResolvedHarborSelection:
    head, revision = await taskset_revision(ref, entity_client, workspace)
    members = await resolve_harbor_members(pinned_members(revision), entity_client=entity_client)
    return ResolvedHarborSelection(
        members=members, taskset_ref=TasksetRef(f"{head.workspace}/{head.name}#{revision.content_hash}")
    )


def _pinned_revision_sync[R: (TaskRevisionEntity, TasksetRevisionEntity)](
    client: SyncEntityClient,
    head: TaskEntity | TasksetEntity,
    revision_type: type[R],
    digest: str,
) -> R:
    # Sync transport uses the same digest query and content verifier as get_revision.
    result = client.list(
        revision_type,
        workspace=head.workspace,
        filter_operation=LogicalOperation(
            operator=FilterOperator.AND,
            operations=[
                ComparisonOperation(field="parent", operator=FilterOperator.EQ, value=head.id),
                ComparisonOperation(field="data.content_hash", operator=FilterOperator.EQ, value=digest),
            ],
        ),
        sort="-created_at",
        page_size=1,
    )
    if not result.data:
        raise RevisionNotFoundError(f"'{head.workspace}/{head.name}' has no revision with digest {digest!r}")
    revision = revision_type.model_validate(result.data[0].model_dump())
    _verify_content(head, revision)
    return revision


def resolve_harbor_source_sync(
    source: PinnedHarborSource, *, entity_client: SyncEntityClient
) -> list[StoredHarborTask]:
    if isinstance(source, PinnedHarborTaskset):
        source = PinnedHarborTaskset.model_validate(source.model_dump())
        workspace, name, digest = parse_subentity_ref(source.taskset_ref.root, "default")
        head = cast(TasksetEntity, entity_client.get(TasksetEntity, name=name, workspace=workspace))
        refs = pinned_members(_pinned_revision_sync(entity_client, head, TasksetRevisionEntity, digest))
    else:
        refs = PinnedHarborTaskList.model_validate(source.model_dump()).task_refs
    members = []
    for ref in refs:
        workspace, name, digest = parse_subentity_ref(ref.root, "default")
        task = cast(TaskEntity, entity_client.get(TaskEntity, name=name, workspace=workspace))
        members.append(harbor_member(task, _pinned_revision_sync(entity_client, task, TaskRevisionEntity, digest)))
    return members
