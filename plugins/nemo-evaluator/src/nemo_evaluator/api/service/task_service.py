# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CRUD service for persisted agent-eval task entities.

A task is stored whole in the entity store (its metrics — inline bundles and/or stored-metric refs —
live in the entity record; there's no separate Files payload, unlike a metric bundle). Stored
``TaskEntity`` rows are mapped to the :class:`Task` API DTO — like ``MetricService`` maps
``MetricBundleEntity`` to ``Metric`` — so the wire contract round-trips cleanly (an ``EntityBase``'s
``id``/``created_at`` are computed and don't deserialize from the entity's own serialized form).
"""

from __future__ import annotations

import logging
from typing import Protocol, cast

from nemo_evaluator.api.schemas import (
    LATEST_TAG,
    HarborTaskDefinition,
    MetricInline,
    MetricRef,
    Revision,
    Task,
    TaskDefinition,
    TaskInput,
)
from nemo_evaluator.entities import TaskEntity, TaskRevisionEntity
from nemo_evaluator.harbor.materialization import verify_definition
from nemo_evaluator.metric_refs import parse_metric_ref
from nemo_evaluator.revisions import (
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
from nemo_platform_plugin.files.client import AsyncFilesClient
from nemo_platform_plugin.filter_ops import FilterOperation
from nemo_platform_plugin.log_utils import sanitize_for_log
from nemo_platform_plugin.schema import Page, PaginationData

logger = logging.getLogger(__name__)


class _MetricService(Protocol):
    async def store_derived_metric(self, metric: MetricInline, *, workspace: str) -> MetricRef: ...

    async def get_metric(self, workspace: str, name: str) -> object | None: ...


class MetricRefNotFoundError(ValueError):
    """A task references a stored metric that does not exist."""


def _entity_to_task(entity: TaskEntity) -> Task:
    """Map a stored task entity to its API DTO, guarding the persistence timestamps."""
    created_at = entity.created_at
    updated_at = entity.updated_at
    if created_at is None or updated_at is None:
        raise ValueError(f"Stored task '{entity.workspace}/{entity.name}' is missing persistence timestamps")
    return Task(
        id=entity.id,
        name=entity.name,
        workspace=entity.workspace,
        project=entity.project,
        spec=entity.spec,
        metadata=entity.metadata,
        revision=entity.latest_revision,
        tags=entity.tags,
        created_at=created_at,
        updated_at=updated_at,
    )


def _revision_to_task(head: TaskEntity, revision: TaskRevisionEntity) -> Task:
    """Present a published revision as the ``Task`` DTO.

    Identity (id, name, workspace, project) comes from the head — it is the same task — while every
    content field comes from the revision. ``tags`` are the head's, because tags are current state:
    "which tags point here now" is a question about the present, not about what was frozen.
    """
    created_at = revision.created_at
    if created_at is None:
        raise ValueError(f"Revision {revision.revision} of '{head.workspace}/{head.name}' has no timestamp")
    return Task(
        id=head.id,
        name=head.name,
        workspace=head.workspace,
        project=head.project,
        spec=revision.spec,
        metadata=revision.metadata,
        revision=revision.revision,
        tags={tag: ordinal for tag, ordinal in head.tags.items() if ordinal == revision.revision},
        created_at=created_at,
        updated_at=revision.updated_at or created_at,
    )


def _entity_to_revision(revision: TaskRevisionEntity, tags: dict[str, int]) -> Revision:
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


class TaskEntityStoreProtocol(EntityClientProtocol[TaskEntity], EntityUpdateClientProtocol[TaskEntity], Protocol):
    """The store surface this service needs for its own task records.

    Composed from the shared entity-client protocols rather than a bespoke one, so the surface a
    service depends on cannot drift from the client that satisfies it.
    """


class TaskService:
    """Create/get/list/delete for persisted agent-eval task entities, exposed as the ``Task`` DTO."""

    def __init__(
        self,
        entity_client: TaskEntityStoreProtocol,
        metric_service: _MetricService,
        files_client: AsyncFilesClient | None = None,
    ):
        self.entity_client = entity_client
        #: The same client, viewed at the revision type. Python has no intersection types, so a
        #: single annotation cannot say "serves TaskEntity *and* TaskRevisionEntity" — but the
        #: concrete client's methods are generic over the entity type and genuinely satisfy both.
        self.revision_client: EntityClientProtocol[TaskRevisionEntity] = cast(
            EntityClientProtocol[TaskRevisionEntity], entity_client
        )
        self.metric_service = metric_service
        self.files_client = files_client

    async def _normalize_metrics(self, metrics: list[MetricRef | MetricInline], *, workspace: str) -> list[MetricRef]:
        """Resolve a task's submitted metrics to references — inline metrics are stored as derived
        metrics (content-addressed, hidden from the listing) so a persisted task only ever holds refs."""
        refs: list[MetricRef] = []
        for metric in metrics:
            if isinstance(metric, MetricRef):
                ref_workspace, name = parse_metric_ref(metric.root, workspace)
                if await self.metric_service.get_metric(ref_workspace, name) is None:
                    raise MetricRefNotFoundError(
                        f"Metric reference '{metric.root}' not found. "
                        f"Ensure a stored metric named '{name}' exists in workspace '{ref_workspace}', "
                        "or pass an inline metric instead."
                    )
                refs.append(MetricRef(f"{ref_workspace}/{name}"))
            else:
                refs.append(await self.metric_service.store_derived_metric(metric, workspace=workspace))
        return refs

    async def _normalize_spec(self, spec: TaskDefinition, *, workspace: str) -> TaskDefinition:
        """Narrow a submitted spec to its stored form.

        Inline metrics are offloaded to derived stored metrics for both task kinds.
        Harbor archives are independently verified
        with request credentials before their projections are persisted.
        """
        if isinstance(spec, HarborTaskDefinition):
            if self.files_client is None:
                raise ValueError("Harbor registration requires an authenticated Files client")
            native = await verify_definition(spec, self.files_client)
            if spec.native_task_id != native.task_id:
                raise ValueError("Harbor native_task_id does not match the verified archive")
            return spec.model_copy(
                update={
                    "config": native.config,
                    "instruction": native.instruction,
                    "metrics": await self._normalize_metrics(spec.metrics, workspace=workspace),
                }
            )
        # Same model in and out — only ``metrics`` narrows, from possibly-inline to references.
        return spec.model_copy(update={"metrics": await self._normalize_metrics(spec.metrics, workspace=workspace)})

    async def _apply_content(self, entity: TaskEntity, task_input: TaskInput, *, workspace: str) -> TaskEntity:
        """Overwrite a head record's content from a request body (leaving revision pointers alone)."""
        entity.spec = await self._normalize_spec(task_input.spec, workspace=workspace)
        entity.metadata = task_input.metadata
        return entity

    async def create_task(
        self, name: str, task_input: TaskInput, *, workspace: str, project: str | None = None
    ) -> tuple[Task, bool]:
        """Store a new task and publish it as revision 1.

        Strict create: raises ``ValueError`` if the name is taken. Returns ``(task, published)``,
        where ``published`` is always ``True`` here — a fresh task always cuts a revision. Use
        :meth:`replace_task` to publish a further revision of an existing task.
        """
        # Normalize once: ``_apply_content`` would offload the same inline metrics a second time.
        entity = TaskEntity(
            name=name,
            workspace=workspace,
            project=project,
            spec=await self._normalize_spec(task_input.spec, workspace=workspace),
            metadata=task_input.metadata,
        )
        try:
            created = await self.entity_client.create(entity)
        except NemoEntityConflictError as exc:
            raise ValueError(f"Task '{workspace}/{name}' already exists") from exc
        try:
            revision, head, published = await self._publish(created, tags=set(task_input.tags))
        except Exception:
            # A head with no revision would violate the invariant every consumer relies on — that
            # `#latest` always resolves and `revision` is never 0. There is no cross-entity
            # transaction, so roll the head back by hand rather than leaving a half-created task.
            logger.exception("Publishing revision 1 failed; rolling back the task record")
            try:
                await self.entity_client.delete(TaskEntity, name=name, workspace=workspace)
            except Exception:
                # Report the rollback failure, but re-raise the *original* error: replacing it
                # would hide why the publish failed and leave the caller debugging the cleanup.
                logger.exception("Rollback of the orphaned task record also failed")
            raise
        logger.info(
            "Task created", extra={"workspace": sanitize_for_log(workspace), "task_name": sanitize_for_log(name)}
        )
        return _revision_to_task(head, revision), published

    async def replace_task(
        self, name: str, task_input: TaskInput, *, workspace: str, project: str | None = None
    ) -> tuple[Task, bool]:
        """Replace a task's content and publish the result, creating the task if absent.

        Upsert rather than 404-on-missing so a publisher can issue one idempotent call without
        first checking existence — checking then creating is both an extra round trip and a race
        between two publishers of the same task.

        Returns ``(task, published)``. ``published`` is ``False`` when the submitted content matches
        the current revision: the request is then a no-op that still applies any new tags, which is
        what makes repeated PUTs of the same content cheap and genuinely idempotent.
        """
        try:
            head = await self.entity_client.get(TaskEntity, name=name, workspace=workspace)
        except NemoEntityNotFoundError:
            return await self.create_task(name, task_input, workspace=workspace, project=project)

        await self._apply_content(head, task_input, workspace=workspace)
        if project is not None:
            # Applied rather than ignored. ``project`` is a query parameter, so an omitted value
            # means "leave it alone" — only an explicit value moves the record.
            head.project = project
        # Publish the staged content *without* committing the head first. Publishing already writes
        # the head (pointers and content together), so a pre-write would be a second round trip
        # whose only distinct effect is a window: if publishing then failed, the head would hold
        # content no revision covers and a plain GET would serve it.
        revision, published_head, published = await self._publish(head, tags=set(task_input.tags))
        if not published:
            # Content matched a revision that is already tagged as requested, so publishing wrote
            # nothing. Anything outside the digest — ``project`` — still has to be persisted.
            published_head = await self.entity_client.update(published_head)
        logger.info(
            "Task replaced",
            extra={
                "workspace": sanitize_for_log(workspace),
                "task_name": sanitize_for_log(name),
                "published": published,
            },
        )
        return _revision_to_task(published_head, revision), published

    async def _publish(self, head: TaskEntity, *, tags: set[str]) -> tuple[TaskRevisionEntity, TaskEntity, bool]:
        """Freeze the head as a revision. The returned head already carries the new pointers."""
        revision, published_head, created = await publish_revision(
            self.entity_client, self.revision_client, head, TaskRevisionEntity, tags=tags
        )
        return revision, published_head, created

    async def resolve_revision(self, workspace: str, name: str, fragment: str = LATEST_TAG) -> str:
        """Return the content digest of the revision a ref fragment names.

        This is what turns a *tag*-pinned member reference into a *digest*-pinned one at taskset
        publish time. Raises :class:`RevisionNotFoundError` if the task has no such revision, and
        ``NemoEntityNotFoundError`` if the task itself is missing.
        """
        head = await self.entity_client.get(TaskEntity, name=name, workspace=workspace)
        revision = await get_revision(self.revision_client, TaskRevisionEntity, head, fragment)
        return revision.content_hash

    async def list_revisions(
        self, workspace: str, name: str, *, page: int = 1, page_size: int = 100
    ) -> Page[Revision] | None:
        """List a task's published revisions, newest first; ``None`` if the task is absent.

        Paged rather than returning every revision: a bare list would silently truncate at the
        store's page size, making a capped history indistinguishable from a complete one.
        """
        try:
            head = await self.entity_client.get(TaskEntity, name=name, workspace=workspace)
        except NemoEntityNotFoundError:
            return None
        result = await list_revisions(self.revision_client, TaskRevisionEntity, head, page=page, page_size=page_size)
        data = [_entity_to_revision(revision, head.tags) for revision in result.data]
        return Page(data=data, pagination=_pagination(result.pagination, len(data)), sort=None, filter=None)

    async def tag_revision(self, workspace: str, name: str, tag: str, fragment: str) -> Task | None:
        """Point a tag at an existing revision; ``None`` if the task is absent."""
        try:
            head = await self.entity_client.get(TaskEntity, name=name, workspace=workspace)
        except NemoEntityNotFoundError:
            return None
        updated = await apply_tag(self.entity_client, self.revision_client, TaskRevisionEntity, head, tag, fragment)
        logger.info(
            "Task revision tagged",
            extra={"workspace": sanitize_for_log(workspace), "task_name": sanitize_for_log(name)},
        )
        return _entity_to_task(updated)

    async def get_task(self, workspace: str, name: str, revision: str | None = None) -> Task | None:
        """Get a stored task; ``None`` if absent.

        ``revision`` is a tag or a content digest — the same thing a ref's ``#fragment`` carries.
        Omitted, the current content is returned. Supplied, the task is returned *as of* that
        revision, which is how a consumer holding a pinned reference reads what was published
        rather than what happens to be current.
        """
        try:
            head = await self.entity_client.get(TaskEntity, name=name, workspace=workspace)
        except NemoEntityNotFoundError:
            return None
        if revision is None:
            return _entity_to_task(head)
        return _revision_to_task(head, await get_revision(self.revision_client, TaskRevisionEntity, head, revision))

    async def list_tasks(
        self,
        *,
        workspace: str,
        page: int = 1,
        page_size: int = 100,
        sort: str | None = None,
        filter_operation: FilterOperation | None = None,
    ) -> Page[Task]:
        result = await self.entity_client.list(
            TaskEntity,
            workspace=workspace,
            filter_operation=filter_operation,
            sort=sort,
            page=page,
            page_size=page_size,
        )
        data = [_entity_to_task(entity) for entity in result.data]
        return Page(data=data, pagination=_pagination(result.pagination, len(data)), sort=sort, filter=None)

    async def delete_task(self, workspace: str, name: str) -> bool:
        """Delete a stored task; ``False`` if absent."""
        try:
            await self.entity_client.delete(TaskEntity, name=name, workspace=workspace)
        except NemoEntityNotFoundError:
            return False
        logger.info(
            "Task deleted", extra={"workspace": sanitize_for_log(workspace), "task_name": sanitize_for_log(name)}
        )
        return True
