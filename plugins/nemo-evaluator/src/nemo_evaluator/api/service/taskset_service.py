# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CRUD service for persisted taskset entities.

A taskset is a flexible grouping of stored tasks: it holds references to its members
(``workspace/name``) plus free-form annotations, stored whole in the entity store. Stored
``TasksetEntity`` rows are mapped to the :class:`Taskset` API DTO — the same DTO/entity split
``TaskService`` uses — so the wire contract round-trips cleanly (an ``EntityBase``'s ``id``/
``created_at`` are computed and don't deserialize from the entity's own serialized form).

Unlike ``TaskService`` there are no inline members to normalize. Instead, every member reference is
resolved on write to the exact revision digest it names, which is also where a missing member is
caught — resolution has to fetch the task anyway, so existence falls out of it rather than costing
a separate round trip per member.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol, cast

from nemo_evaluator.api.schemas import (
    Revision,
    TaskRef,
    Taskset,
    TasksetInput,
    parse_subentity_ref,
)
from nemo_evaluator.entities import TasksetEntity, TasksetRevisionEntity
from nemo_evaluator.revisions import (
    RevisionNotFoundError,
    apply_tag,
    get_revision,
    list_revisions,
    publish_revision,
)
from nemo_platform_plugin.entities import (
    EntityClientProtocol,
    EntityUpdateClientProtocol,
    PaginationInfo,
)
from nemo_platform_plugin.entity_client import (
    NemoEntityConflictError,
    NemoEntityNotFoundError,
)
from nemo_platform_plugin.filter_ops import FilterOperation
from nemo_platform_plugin.log_utils import sanitize_for_log
from nemo_platform_plugin.schema import Page, PaginationData

logger = logging.getLogger(__name__)

#: How many member refs resolve at once when publishing a taskset. Bounded so a large grouping
#: cannot flood the entity store with simultaneous requests, but high enough that a hundred-member
#: dataset does not publish at one round trip per member.
_MEMBER_RESOLUTION_CONCURRENCY = 10


class _TaskService(Protocol):
    async def get_task(self, workspace: str, name: str) -> object | None: ...

    async def resolve_revision(self, workspace: str, name: str, fragment: str = "latest") -> str: ...


class TaskRefNotFoundError(ValueError):
    """A taskset references a task that does not exist.

    Subclasses ``ValueError`` so existing callers still catch it, while letting the route distinguish
    a missing-member reference (a 422 on the submitted body) from other validation errors.
    """


class DuplicateTaskRefError(ValueError):
    """A taskset lists two references that resolve to the same task.

    The field validator already rejects byte-identical refs; this catches refs that differ in form
    but resolve to the same ``(workspace, name)`` (e.g. ``task-a`` and ``default/task-a`` when the
    taskset lives in ``default``). Subclasses ``ValueError`` so the route can map it to a 422.
    """


class TasksetExistsError(ValueError):
    """A taskset with the given workspace/name already exists.

    Subclasses ``ValueError`` so existing callers still catch it, while letting the route map a
    name collision to a 409 without inspecting the message text.
    """


def _entity_to_taskset(entity: TasksetEntity) -> Taskset:
    """Map a stored taskset entity to its API DTO, guarding the persistence timestamps."""
    created_at = entity.created_at
    updated_at = entity.updated_at
    if created_at is None or updated_at is None:
        raise ValueError(f"Stored taskset '{entity.workspace}/{entity.name}' is missing persistence timestamps")
    return Taskset(
        id=entity.id,
        name=entity.name,
        workspace=entity.workspace,
        project=entity.project,
        description=entity.description,
        tasks=entity.tasks,
        files_ref=entity.files_ref,
        revision=entity.latest_revision,
        tags=entity.tags,
        metadata=entity.metadata,
        created_at=created_at,
        updated_at=updated_at,
    )


def _revision_to_taskset(head: TasksetEntity, revision: TasksetRevisionEntity) -> Taskset:
    """Present a published revision as the ``Taskset`` DTO — identity from the head, content from
    the revision. Tags are the head's, since "what points here now" is a question about the present."""
    created_at = revision.created_at
    if created_at is None:
        raise ValueError(f"Revision {revision.revision} of '{head.workspace}/{head.name}' has no timestamp")
    return Taskset(
        id=head.id,
        name=head.name,
        workspace=head.workspace,
        project=head.project,
        description=revision.description,
        tasks=revision.tasks,
        files_ref=revision.files_ref,
        metadata=revision.metadata,
        revision=revision.revision,
        tags={tag: ordinal for tag, ordinal in head.tags.items() if ordinal == revision.revision},
        created_at=created_at,
        updated_at=revision.updated_at or created_at,
    )


def _entity_to_revision(revision: TasksetRevisionEntity, tags: dict[str, int]) -> Revision:
    """Map a stored revision to its API DTO, attaching the tags that currently point at it."""
    created_at = revision.created_at
    if created_at is None:
        raise ValueError(f"Revision {revision.revision} is missing its creation timestamp")
    return Revision(
        revision=revision.revision,
        content_hash=revision.content_hash,
        tags=sorted(tag for tag, ordinal in tags.items() if ordinal == revision.revision),
        created_at=created_at,
    )


def _pagination(src: PaginationInfo, current_page_size: int) -> PaginationData:
    """Carry the entity-store pagination counts into the API ``Page`` envelope."""
    return PaginationData(
        page=src.page,
        page_size=src.page_size,
        current_page_size=current_page_size,
        total_pages=src.total_pages,
        total_results=src.total_results,
    )


class TasksetEntityStoreProtocol(
    EntityClientProtocol[TasksetEntity], EntityUpdateClientProtocol[TasksetEntity], Protocol
):
    """The store surface this service needs for its own taskset records.

    Composed from the shared entity-client protocols rather than a bespoke one, so the surface a
    service depends on cannot drift from the client that satisfies it.
    """


class TasksetService:
    """Create/get/list/delete for persisted taskset entities, exposed as the ``Taskset`` DTO."""

    def __init__(self, entity_client: TasksetEntityStoreProtocol, task_service: _TaskService):
        self.entity_client = entity_client
        #: The same client, viewed at the revision type. Python has no intersection types, so a
        #: single annotation cannot say "serves TasksetEntity *and* TasksetRevisionEntity" — but the
        #: concrete client's methods are generic over the entity type and genuinely satisfy both.
        self.revision_client: EntityClientProtocol[TasksetRevisionEntity] = cast(
            EntityClientProtocol[TasksetRevisionEntity], entity_client
        )
        self.task_service = task_service

    def _reject_duplicate_members(self, tasks: list[TaskRef], *, workspace: str) -> None:
        """Reject two refs resolving to the same task. Pure in-memory, so it runs before any I/O.

        The field validator only catches byte-identical refs; this catches refs that differ in form
        but resolve to the same ``(workspace, name)`` — e.g. ``task-a`` and ``default/task-a`` in
        the ``default`` workspace.

        The revision fragment is deliberately discarded: two refs naming the same task at different
        revisions are still the same member, and a taskset holding both would expand that task twice.
        """
        seen: set[tuple[str, str]] = set()
        for ref in tasks:
            ref_workspace, name, _ = parse_subentity_ref(ref.root, workspace)
            resolved = (ref_workspace, name)
            if resolved in seen:
                raise DuplicateTaskRefError(
                    f"Task reference '{ref.root}' resolves to '{resolved[0]}/{resolved[1]}', already in this taskset"
                )
            seen.add(resolved)

    async def _pin_member(self, ref: TaskRef, *, workspace: str) -> TaskRef:
        """Resolve one member ref to ``workspace/name#<digest>``.

        Existence and revision resolution are one operation: ``resolve_revision`` already fetches
        the task, so a separate existence check would just re-read the same record.
        """
        ref_workspace, name, fragment = parse_subentity_ref(ref.root, workspace)
        try:
            digest = await self.task_service.resolve_revision(ref_workspace, name, fragment)
        except NemoEntityNotFoundError as exc:
            raise TaskRefNotFoundError(f"Task reference '{ref.root}' not found in workspace '{ref_workspace}'") from exc
        except RevisionNotFoundError as exc:
            raise TaskRefNotFoundError(f"Task reference '{ref.root}' names no published revision: {exc}") from exc
        return TaskRef(f"{ref_workspace}/{name}#{digest}")

    async def _resolved_content(self, taskset_input: TasksetInput, *, workspace: str) -> list[TaskRef]:
        """Validate membership and resolve it to digest-pinned refs.

        Members resolve **concurrently**, bounded by :data:`_MEMBER_RESOLUTION_CONCURRENCY`. A
        Harbor-scale dataset can name hundreds of tasks, and resolving them one at a time made
        publish latency linear in membership. Bounded rather than unbounded so a large taskset
        cannot open hundreds of simultaneous connections to the entity store.

        Resolution happens on write because tags move: a stored ``#latest`` would silently re-point
        this taskset's membership the next time that task published, and a published grouping that
        changes underneath you is not a grouping. Same reason a lockfile records resolved versions,
        not ranges.

        The resolved refs are then sorted into a **canonical order**. Membership is a set — the
        field documents it as such and duplicates are rejected — but the digest covers the stored
        list, so without this ``[a, b]`` and ``[b, a]`` would be the same grouping under two
        different digests, and re-submitting a reordered request would cut a revision that changed
        nothing. Sorting the stored content rather than only the hash input keeps the digest a hash
        of exactly what is stored, which is what verification on read re-checks.
        """
        self._reject_duplicate_members(taskset_input.tasks, workspace=workspace)
        limit = asyncio.Semaphore(_MEMBER_RESOLUTION_CONCURRENCY)

        async def _bounded(ref: TaskRef) -> TaskRef:
            async with limit:
                return await self._pin_member(ref, workspace=workspace)

        pending = [asyncio.create_task(_bounded(ref)) for ref in taskset_input.tasks]
        try:
            return sorted(await asyncio.gather(*pending), key=lambda ref: ref.root)
        except BaseException:
            # ``gather`` propagates the first failure but leaves its siblings running. One bad
            # member in a large grouping would otherwise keep issuing reads long after the request
            # failed, and their own errors would surface as unretrieved-task warnings.
            for task in pending:
                task.cancel()
            raise

    async def create_taskset(
        self, name: str, taskset_input: TasksetInput, *, workspace: str, project: str | None = None
    ) -> tuple[Taskset, bool]:
        """Store a new taskset and publish it as revision 1.

        Strict create: raises :class:`TasksetExistsError` if the name is taken, and
        :class:`TaskRefNotFoundError` if a member does not exist or has no such revision. Returns
        ``(taskset, published)``; ``published`` is always ``True`` here.
        """
        entity = TasksetEntity(
            name=name,
            workspace=workspace,
            project=project,
            description=taskset_input.description,
            tasks=await self._resolved_content(taskset_input, workspace=workspace),
            files_ref=taskset_input.files_ref,
            metadata=taskset_input.metadata,
        )
        try:
            created = await self.entity_client.create(entity)
        except NemoEntityConflictError as exc:
            raise TasksetExistsError(f"Taskset '{workspace}/{name}' already exists") from exc
        try:
            revision, head, published = await self._publish(created, tags=set(taskset_input.tags))
        except Exception:
            # A head with no revision would break the invariant consumers rely on — `#latest`
            # always resolves and `revision` is never 0. No cross-entity transaction exists, so
            # roll the head back rather than leave a half-created taskset behind.
            logger.exception("Publishing revision 1 failed; rolling back the taskset record")
            try:
                await self.entity_client.delete(TasksetEntity, name=name, workspace=workspace)
            except Exception:
                # Report the rollback failure, but re-raise the *original* error: replacing it
                # would hide why the publish failed and leave the caller debugging the cleanup.
                logger.exception("Rollback of the orphaned taskset record also failed")
            raise
        logger.info(
            "Taskset created",
            extra={"workspace": sanitize_for_log(workspace), "taskset_name": sanitize_for_log(name)},
        )
        return _revision_to_taskset(head, revision), published

    async def replace_taskset(
        self, name: str, taskset_input: TasksetInput, *, workspace: str, project: str | None = None
    ) -> tuple[Taskset, bool]:
        """Replace a taskset's content and publish the result, creating it if absent.

        Note that re-submitting *identical* membership can still publish a new revision: members are
        re-resolved on every write, so if a member task published since last time, ``#latest`` now
        names a different digest and this grouping genuinely differs. That is the intended
        behavior — the taskset's content is the exact revisions it names, not the names alone.
        """
        try:
            head = await self.entity_client.get(TasksetEntity, name=name, workspace=workspace)
        except NemoEntityNotFoundError:
            return await self.create_taskset(name, taskset_input, workspace=workspace, project=project)

        if project is not None:
            # Applied rather than ignored. ``project`` is a query parameter, so an omitted value
            # means "leave it alone" — only an explicit value moves the record.
            head.project = project
        head.description = taskset_input.description
        head.tasks = await self._resolved_content(taskset_input, workspace=workspace)
        head.files_ref = taskset_input.files_ref
        head.metadata = taskset_input.metadata
        # Publish the staged content *without* committing the head first — see the matching comment
        # in ``TaskService.replace_task``. Publishing writes the head itself, so a pre-write would
        # only open a window where a failed publish leaves the head serving uncovered content.
        revision, published_head, published = await self._publish(head, tags=set(taskset_input.tags))
        if not published:
            # Publishing wrote nothing (content already published and tagged as requested), so
            # persist what sits outside the digest — ``project``.
            published_head = await self.entity_client.update(published_head)
        logger.info(
            "Taskset replaced",
            extra={
                "workspace": sanitize_for_log(workspace),
                "taskset_name": sanitize_for_log(name),
                "published": published,
            },
        )
        return _revision_to_taskset(published_head, revision), published

    async def _publish(
        self, head: TasksetEntity, *, tags: set[str]
    ) -> tuple[TasksetRevisionEntity, TasksetEntity, bool]:
        """Freeze the head as a revision. The returned head already carries the new pointers."""
        revision, published_head, created = await publish_revision(
            self.entity_client, self.revision_client, head, TasksetRevisionEntity, tags=tags
        )
        return revision, published_head, created

    async def list_revisions(
        self, workspace: str, name: str, *, page: int = 1, page_size: int = 100
    ) -> Page[Revision] | None:
        """List a taskset's published revisions, newest first; ``None`` if the taskset is absent.

        Paged rather than returning every revision: a bare list would silently truncate at the
        store's page size, making a capped history indistinguishable from a complete one.
        """
        try:
            head = await self.entity_client.get(TasksetEntity, name=name, workspace=workspace)
        except NemoEntityNotFoundError:
            return None
        result = await list_revisions(self.revision_client, TasksetRevisionEntity, head, page=page, page_size=page_size)
        data = [_entity_to_revision(revision, head.tags) for revision in result.data]
        return Page(data=data, pagination=_pagination(result.pagination, len(data)), sort=None, filter=None)

    async def tag_revision(self, workspace: str, name: str, tag: str, fragment: str) -> Taskset | None:
        """Point a tag at an existing revision; ``None`` if the taskset is absent."""
        try:
            head = await self.entity_client.get(TasksetEntity, name=name, workspace=workspace)
        except NemoEntityNotFoundError:
            return None
        updated = await apply_tag(self.entity_client, self.revision_client, TasksetRevisionEntity, head, tag, fragment)
        logger.info(
            "Taskset revision tagged",
            extra={"workspace": sanitize_for_log(workspace), "taskset_name": sanitize_for_log(name)},
        )
        return _entity_to_taskset(updated)

    async def get_taskset(self, workspace: str, name: str, revision: str | None = None) -> Taskset | None:
        """Get a stored taskset; ``None`` if absent.

        ``revision`` is a tag or content digest — what a ref's ``#fragment`` carries — so a consumer
        holding a pinned reference reads the membership that was published, not what is current.
        """
        try:
            head = await self.entity_client.get(TasksetEntity, name=name, workspace=workspace)
        except NemoEntityNotFoundError:
            return None
        if revision is None:
            return _entity_to_taskset(head)
        return _revision_to_taskset(
            head, await get_revision(self.revision_client, TasksetRevisionEntity, head, revision)
        )

    async def list_tasksets(
        self,
        *,
        workspace: str,
        page: int = 1,
        page_size: int = 100,
        sort: str | None = None,
        filter_operation: FilterOperation | None = None,
    ) -> Page[Taskset]:
        result = await self.entity_client.list(
            TasksetEntity,
            workspace=workspace,
            filter_operation=filter_operation,
            sort=sort,
            page=page,
            page_size=page_size,
        )
        data = [_entity_to_taskset(entity) for entity in result.data]
        return Page(data=data, pagination=_pagination(result.pagination, len(data)), sort=sort, filter=None)

    async def delete_taskset(self, workspace: str, name: str) -> bool:
        """Delete a stored taskset; ``False`` if absent."""
        try:
            await self.entity_client.delete(TasksetEntity, name=name, workspace=workspace)
        except NemoEntityNotFoundError:
            return False
        logger.info(
            "Taskset deleted",
            extra={"workspace": sanitize_for_log(workspace), "taskset_name": sanitize_for_log(name)},
        )
        return True
