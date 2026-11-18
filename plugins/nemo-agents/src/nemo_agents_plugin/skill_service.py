# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish, resolve and manage stored skills.

A skill is a :class:`~nemo_agents_plugin.entities.SkillEntity` head that holds revision pointers
only, plus immutable :class:`~nemo_agents_plugin.entities.SkillRevisionEntity` children named
``rev.<n>``. This follows the evals task/taskset revision design, except that the head never holds
content: the current skill is whatever ``tags['latest']`` names, verified on read.

Publish order, each step recoverable by retrying the same request:

1. Authorize the caller's read of the source fileset (and the linked taskset, if any).
2. Validate tags; verify the linked taskset revision exists.
3. Download the source into private staging, build the archive, upload it to the service-owned
   fileset and verify the stored bytes (:class:`~nemo_agents_plugin.skill_storage.SkillArchiveStore`).
4. Create ``rev.<n>``. A conflict means another publisher took ``n``: refresh the head pointers and
   retry with ``n + 1``. If a revision at or after ``latest`` already has this content digest (an
   identical concurrent publish, or a crash before step 5), verify and reuse it instead.
5. Move the tags under the head's optimistic lock. ``latest`` only moves forward.
"""

from __future__ import annotations

import asyncio
import logging
import re
import tempfile
from pathlib import Path
from typing import Protocol, cast
from urllib.parse import quote

from nemo_agents_plugin.entities import SKILL_REVISION_SELF_FIELDS, SkillEntity, SkillRevisionEntity
from nemo_agents_plugin.skill_storage import SkillArchiveStore, SkillSourceError
from nemo_helix_plugin.agents.skills import (
    SKILL_NAME_MAX_LENGTH,
    SKILL_NAME_PATTERN,
    Skill,
    SkillFrontmatter,
    SkillInput,
)
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.content_hash import content_hash
from nemo_helix_plugin.dependencies import RequestAuthorizer
from nemo_helix_plugin.entities import (
    EntityClientProtocol,
    EntityUpdateClientProtocol,
    PaginationInfo,
)
from nemo_helix_plugin.entity_client import NemoEntityConflictError, NemoEntityNotFoundError
from nemo_helix_plugin.evals.client import AsyncEvaluatorClient
from nemo_helix_plugin.filter_ops import ComparisonOperation, FilterOperation, FilterOperator, LogicalOperation
from nemo_helix_plugin.log_utils import sanitize_for_log
from nemo_helix_plugin.revision_refs import (
    LATEST_TAG,
    is_digest,
    parse_subentity_ref,
    validate_movable_tag,
    validate_tag_name,
)
from nemo_helix_plugin.schema import Page, PaginationData
from nemo_helix_plugin.skill_archive import DEFAULT_LIMITS, SkillArchiveError, SkillArchiveLimits, build_skill_archive

logger = logging.getLogger(__name__)

#: Ordinal allocation and pointer-write attempts before giving up on a contended skill.
_MAX_ATTEMPTS = 5

_REVISION_NAME_PREFIX = "rev."

_SKILL_NAME_RE = re.compile(SKILL_NAME_PATTERN)


class SkillStoreProtocol(EntityClientProtocol[SkillEntity], EntityUpdateClientProtocol[SkillEntity], Protocol):
    """Entity-store surface the service needs for skill heads (and, viewed at the revision type, revisions)."""


class SkillExistsError(ValueError):
    """Create was asked for a skill name that already exists."""


class SkillValidationError(ValueError):
    """The request is invalid: bad name, bad tag, or frontmatter name not matching the skill name."""


class LinkedTasksetNotFoundError(ValueError):
    """The linked taskset revision does not exist (or the evals service is not available to check it)."""


class SkillRevisionNotFoundError(LookupError):
    """A tag or digest names no revision of this skill."""


class SkillRevisionConflictError(RuntimeError):
    """Ordinal allocation or the pointer write lost too many races."""


class SkillRevisionContentMismatchError(RuntimeError):
    """A stored revision no longer hashes to its recorded content digest."""


def revision_name(ordinal: int) -> str:
    """Entity name for a revision ordinal."""
    return f"{_REVISION_NAME_PREFIX}{ordinal}"


def revision_digest(revision: SkillRevisionEntity) -> str:
    """Content digest of a revision: every content field, excluding its own digest and ordinal."""
    return content_hash(revision, exclude=SKILL_REVISION_SELF_FIELDS)


def verify_revision(head: SkillEntity, revision: SkillRevisionEntity) -> SkillRevisionEntity:
    """Re-hash a revision and refuse it when it no longer matches its recorded digest."""
    actual = revision_digest(revision)
    if actual != revision.content_hash:
        raise SkillRevisionContentMismatchError(
            f"revision {revision.revision} of skill '{head.workspace}/{head.name}' does not match its recorded "
            f"digest (recorded {revision.content_hash}, actual {actual}); the stored content was modified"
        )
    return revision


def _to_skill(head: SkillEntity, revision: SkillRevisionEntity, *, all_tags: bool) -> Skill:
    """Identity from the head, content from the revision."""
    tags = dict(head.tags) if all_tags else {t: o for t, o in head.tags.items() if o == revision.revision}
    return Skill(
        id=head.id,
        entity_id=head.id,
        name=head.name,
        workspace=head.workspace,
        project=head.project,
        created_at=head.created_at,
        created_by=head.created_by,
        updated_at=head.updated_at,
        updated_by=head.updated_by,
        db_version=head.db_version,
        revision=revision.revision,
        content_hash=revision.content_hash,
        tags=tags,
        frontmatter=revision.frontmatter,
        archive=revision.archive,
        taskset=revision.taskset,
        published_at=revision.created_at,
    )


def _pagination(src: PaginationInfo, current_page_size: int) -> PaginationData:
    return PaginationData(
        page=src.page,
        page_size=src.page_size,
        current_page_size=current_page_size,
        total_pages=src.total_pages,
        total_results=src.total_results,
    )


class SkillService:
    """Create, publish, resolve, tag, list and delete stored skills.

    Args:
        entity_client: Entity store client for skill heads and revisions.
        archive_store: Stages sources and stores verified archives.
        authorize: The caller's request authorizer. The entity and files clients may carry service
            authority, so every source the publish reads is checked as the caller first.
        evals_client: Typed evals client used to verify a linked taskset revision. Optional so an
            unlinked publish needs no evals service; a linked publish without it is refused.
        limits: Archive limits applied when building.
    """

    def __init__(
        self,
        entity_client: SkillStoreProtocol,
        archive_store: SkillArchiveStore,
        *,
        authorize: RequestAuthorizer,
        evals_client: AsyncEvaluatorClient | None = None,
        limits: SkillArchiveLimits = DEFAULT_LIMITS,
    ) -> None:
        self.entity_client = entity_client
        #: The same client viewed at the revision type (no intersection types in Python).
        self.revision_client = cast(EntityClientProtocol[SkillRevisionEntity], entity_client)
        self.archive_store = archive_store
        self.authorize = authorize
        self.evals_client = evals_client
        self.limits = limits

    # ------------------------------------------------------------------ writes

    async def create_skill(
        self, workspace: str, name: str, skill_input: SkillInput, *, project: str | None = None
    ) -> Skill:
        """Create a skill and publish revision 1. Strict: a published name raises :class:`SkillExistsError`.

        The archive is built and stored before the head exists, so a bad source never leaves a
        head behind. If publishing revision 1 fails after this call created the head, the head is
        deleted. A head with no published revision (a create that crashed between the two writes)
        is resumed rather than refused, so retrying the same create recovers.
        """
        _validate_skill_name(name)
        existing = await self._get_head(workspace, name)
        if existing is not None and LATEST_TAG in existing.tags:
            raise SkillExistsError(f"Skill '{workspace}/{name}' already exists")
        staged, tags = await self._prepare(workspace, name, skill_input)
        if existing is None:
            try:
                existing = await self.entity_client.create(SkillEntity(name=name, workspace=workspace, project=project))
            except NemoEntityConflictError as exc:
                existing = await self._get_head(workspace, name)
                if existing is None or LATEST_TAG in existing.tags:
                    raise SkillExistsError(f"Skill '{workspace}/{name}' already exists") from exc
            else:
                return await self._publish_first(existing, staged, tags)
        revision, head, _ = await self._commit(existing, staged, tags)
        logger.info(
            "Skill created", extra={"workspace": sanitize_for_log(workspace), "skill_name": sanitize_for_log(name)}
        )
        return _to_skill(head, revision, all_tags=False)

    async def _publish_first(self, created: SkillEntity, staged: SkillRevisionEntity, tags: set[str]) -> Skill:
        """Publish revision 1 onto a head this call created; delete the head if that fails."""
        created_db_version = created.db_version
        try:
            revision, head, _ = await self._commit(created, staged, tags)
        except Exception:
            logger.exception("Publishing skill revision 1 failed; rolling back the skill head")
            try:
                await self.entity_client.delete_by_id(SkillEntity, created.id, expected_db_version=created_db_version)
            except Exception:
                logger.exception("Rollback of the orphaned skill head also failed")
            raise
        logger.info(
            "Skill created",
            extra={"workspace": sanitize_for_log(head.workspace), "skill_name": sanitize_for_log(head.name)},
        )
        return _to_skill(head, revision, all_tags=False)

    async def publish_skill(
        self, workspace: str, name: str, skill_input: SkillInput, *, project: str | None = None
    ) -> tuple[Skill, bool]:
        """Publish the next revision, creating the skill if absent. Returns ``(skill, created)``.

        ``created`` is ``True`` only when this call created the revision. It is ``False`` when the
        content was already the latest revision or an identical concurrent publish created it first
        (the same outcome evals reports); the requested tags are still applied.
        """
        _validate_skill_name(name)
        head = await self._get_head(workspace, name)
        if head is None:
            try:
                return await self.create_skill(workspace, name, skill_input, project=project), True
            except SkillExistsError:
                head = await self._require_head(workspace, name)
        if LATEST_TAG not in head.tags:
            # An unpublished head left by a crashed create: finishing it is a create.
            return await self.create_skill(workspace, name, skill_input, project=project), True
        staged, tags = await self._prepare(workspace, name, skill_input)
        revision, head, created = await self._commit(head, staged, tags)
        return _to_skill(head, revision, all_tags=False), created

    async def tag_revision(self, workspace: str, name: str, tag: str, fragment: str) -> Skill | None:
        """Point a user tag at the revision ``fragment`` names; ``None`` if the skill is absent.

        ``latest`` is refused (it is machine-managed), as are digest-shaped tags.
        """
        try:
            validate_movable_tag(tag)
        except ValueError as exc:
            raise SkillValidationError(str(exc)) from exc
        head = await self._get_head(workspace, name)
        if head is None:
            return None
        revision = await self._resolve(head, fragment)
        head = await self._point_tags(head, {tag}, revision.revision)
        return _to_skill(head, revision, all_tags=False)

    async def delete_skill(self, workspace: str, name: str) -> bool:
        """Delete a skill head; its revisions go with it (the store cascades children).

        Archives stay in the skills fileset: an accepted eval run may hold a pin to one.
        """
        head = await self._get_head(workspace, name)
        if head is None:
            return False
        try:
            await self.entity_client.delete_by_id(SkillEntity, head.id, expected_db_version=head.db_version)
        except NemoEntityNotFoundError:
            return False
        logger.info(
            "Skill deleted", extra={"workspace": sanitize_for_log(workspace), "skill_name": sanitize_for_log(name)}
        )
        return True

    # ------------------------------------------------------------------- reads

    async def get_skill(self, workspace: str, name: str, revision: str | None = None) -> Skill | None:
        """The skill at ``revision`` (a tag or content digest; default ``latest``); ``None`` if absent."""
        head = await self._get_head(workspace, name)
        if head is None:
            return None
        resolved = await self._resolve(head, revision or LATEST_TAG)
        return _to_skill(head, resolved, all_tags=revision is None)

    async def resolve_pinned_ref(self, ref: str, default_workspace: str) -> Skill:
        """Resolve ``workspace/name[#tag-or-digest]`` to a verified revision.

        Raises:
            NemoEntityNotFoundError: The skill does not exist.
            SkillRevisionNotFoundError: The fragment names no revision.
        """
        workspace, name, fragment = parse_subentity_ref(ref, default_workspace)
        head = await self._require_head(workspace, name)
        return _to_skill(head, await self._resolve(head, fragment), all_tags=False)

    async def list_skills(
        self,
        workspace: str,
        *,
        page: int = 1,
        page_size: int = 100,
        sort: str | None = None,
        filter_operation: FilterOperation | None = None,
    ) -> Page[Skill]:
        """A page of skills at their latest revision. A head with no published revision is skipped."""
        result = await self.entity_client.list(
            SkillEntity,
            workspace=workspace,
            filter_operation=filter_operation,
            sort=sort,
            page=page,
            page_size=page_size,
        )

        async def latest(head: SkillEntity) -> Skill | None:
            if LATEST_TAG not in head.tags:
                return None
            return _to_skill(head, await self._resolve(head, LATEST_TAG), all_tags=True)

        skills = [skill for skill in await asyncio.gather(*(latest(h) for h in result.data)) if skill is not None]
        return Page(data=skills, pagination=_pagination(result.pagination, len(skills)), sort=sort, filter=None)

    async def list_revisions(
        self, workspace: str, name: str, *, page: int = 1, page_size: int = 100
    ) -> Page[Skill] | None:
        """A page of a skill's revisions, newest first, each verified; ``None`` if the skill is absent."""
        head = await self._get_head(workspace, name)
        if head is None:
            return None
        result = await self.revision_client.list(
            SkillRevisionEntity,
            workspace=workspace,
            filter_operation=ComparisonOperation(field="parent", operator=FilterOperator.EQ, value=head.id),
            sort="-created_at",
            page=page,
            page_size=page_size,
        )
        data = [_to_skill(head, verify_revision(head, rev), all_tags=False) for rev in result.data]
        return Page(data=data, pagination=_pagination(result.pagination, len(data)), sort="-created_at", filter=None)

    # ---------------------------------------------------------------- internals

    async def _get_head(self, workspace: str, name: str) -> SkillEntity | None:
        try:
            return await self.entity_client.get(SkillEntity, name=name, workspace=workspace)
        except NemoEntityNotFoundError:
            return None

    async def _require_head(self, workspace: str, name: str) -> SkillEntity:
        return await self.entity_client.get(SkillEntity, name=name, workspace=workspace)

    async def _prepare(
        self, workspace: str, name: str, skill_input: SkillInput
    ) -> tuple[SkillRevisionEntity, set[str]]:
        """Authorize, validate, build, store and verify. Returns unsaved revision content and the tags to apply."""
        source_workspace, source_fileset = _fileset_identity(skill_input.source.fileset)
        await self.authorize(
            "GET",
            f"/apis/files/v2/workspaces/{quote(source_workspace, safe='')}/filesets/{quote(source_fileset, safe='')}",
        )
        if skill_input.taskset is not None:
            await self._verify_taskset(skill_input.taskset)
        try:
            tags = {LATEST_TAG} | {validate_tag_name(t) for t in skill_input.tags if t != LATEST_TAG}
        except ValueError as exc:
            raise SkillValidationError(str(exc)) from exc

        with tempfile.TemporaryDirectory(prefix=".skill-publish-") as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir(mode=0o700)
            await self.archive_store.stage_source(skill_input.source.fileset, source)
            archive_path = root / "skill_archive"
            try:
                info = await asyncio.to_thread(build_skill_archive, source, archive_path, limits=self.limits)
            except SkillArchiveError as exc:
                raise SkillSourceError(f"Skill source {skill_input.source.fileset!r} is invalid: {exc}") from exc
            if info.skill_name != name:
                raise SkillValidationError(
                    f"SKILL.md names the skill {info.skill_name!r}, but it is being published as {name!r}; "
                    "the frontmatter name must equal the skill name"
                )
            descriptor = await self.archive_store.store(workspace, archive_path, info)

        revision = SkillRevisionEntity(
            name=revision_name(1),
            workspace=workspace,
            content_hash="0" * 64,
            revision=1,
            frontmatter=info.frontmatter,
            archive=descriptor,
            taskset=skill_input.taskset,
        )
        revision.content_hash = revision_digest(revision)
        return revision, tags

    async def _verify_taskset(self, taskset_ref: str) -> None:
        """Check the caller may read the linked taskset and that its pinned revision exists."""
        base, _, digest = taskset_ref.partition("#")
        workspace, name = base.split("/", 1)
        await self.authorize(
            "GET", f"/apis/evals/v2/workspaces/{quote(workspace, safe='')}/tasksets/{quote(name, safe='')}"
        )
        if self.evals_client is None:
            raise LinkedTasksetNotFoundError(
                f"Cannot verify linked taskset {taskset_ref!r}: the evals service is not available"
            )
        try:
            (await self.evals_client.get_taskset_revision(workspace=workspace, name=name, revision=digest)).data()
        except NotFoundError as exc:
            raise LinkedTasksetNotFoundError(f"Linked taskset revision {taskset_ref!r} does not exist") from exc

    async def _reload_head(self, head: SkillEntity) -> SkillEntity:
        """Re-read ``head`` for a retry, refusing a different skill that now has the same name.

        A retry must keep writing to the skill it started on: if that head was deleted (and maybe
        recreated under the same name) its ordinals mean nothing for the new one.
        """
        fresh = await self._get_head(head.workspace, head.name)
        if fresh is None or fresh.id != head.id:
            raise SkillRevisionConflictError(
                f"Skill '{head.workspace}/{head.name}' was deleted while this publish was in progress"
            )
        return fresh

    async def _highest_ordinal(self, head: SkillEntity) -> int:
        """Highest stored revision ordinal of ``head`` (0 if none). Creation order matches ordinal order."""
        result = await self.revision_client.list(
            SkillRevisionEntity,
            workspace=head.workspace,
            filter_operation=ComparisonOperation(field="parent", operator=FilterOperator.EQ, value=head.id),
            sort="-created_at",
            page_size=1,
        )
        return result.data[0].revision if result.data else 0

    async def _revision_at(self, head: SkillEntity, ordinal: int) -> SkillRevisionEntity | None:
        try:
            return await self.revision_client.get(
                SkillRevisionEntity, name=revision_name(ordinal), workspace=head.workspace, parent=head.id
            )
        except NemoEntityNotFoundError:
            return None

    async def _find_by_digest(self, head: SkillEntity, digest: str) -> SkillRevisionEntity | None:
        """Newest revision of this skill with ``digest``, or ``None``."""
        result = await self.revision_client.list(
            SkillRevisionEntity,
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
        return result.data[0] if result.data else None

    async def _resolve(self, head: SkillEntity, fragment: str) -> SkillRevisionEntity:
        """The verified revision a tag or digest names."""
        if is_digest(fragment):
            found = await self._find_by_digest(head, fragment)
            if found is None:
                raise SkillRevisionNotFoundError(
                    f"Skill '{head.workspace}/{head.name}' has no revision with digest {fragment!r}"
                )
            return verify_revision(head, found)
        ordinal = head.tags.get(fragment)
        if ordinal is None:
            known = ", ".join(sorted(head.tags)) or "none"
            raise SkillRevisionNotFoundError(
                f"Skill '{head.workspace}/{head.name}' has no revision tagged {fragment!r} (known tags: {known})"
            )
        found = await self._revision_at(head, ordinal)
        if found is None:
            raise SkillRevisionNotFoundError(
                f"Skill '{head.workspace}/{head.name}' tags {fragment!r} as revision {ordinal}, but that revision "
                "is missing"
            )
        return verify_revision(head, found)

    async def _commit(
        self, head: SkillEntity, staged: SkillRevisionEntity, tags: set[str]
    ) -> tuple[SkillRevisionEntity, SkillEntity, bool]:
        """Create (or reuse) the revision for ``staged`` and point ``tags`` at it.

        Returns the revision, the updated head, and whether this call created the revision.
        """
        digest = staged.content_hash
        for attempt in range(_MAX_ATTEMPTS):
            # Reuse an identical revision at or after ``latest``: the current one (a no-op republish), or
            # one a crashed or concurrent identical publish created without moving pointers yet. An
            # identical revision *older* than latest is a revert, which publishes anew so latest stays forward-only.
            existing = await self._find_by_digest(head, digest)
            if existing is not None and existing.revision >= head.tags.get(LATEST_TAG, 0):
                verify_revision(head, existing)
                return (
                    existing,
                    await self._point_tags(head, tags, existing.revision, latest_frontmatter=existing.frontmatter),
                    False,
                )

            ordinal = head.latest_revision + 1
            candidate = staged.model_copy(update={"name": revision_name(ordinal), "revision": ordinal})
            candidate._parent = head.id
            try:
                created = await self.revision_client.create(candidate)
            except NemoEntityConflictError:
                logger.info(
                    "Skill revision ordinal contended, retrying",
                    extra={"skill": f"{head.workspace}/{head.name}", "ordinal": ordinal, "attempt": attempt + 1},
                )
                fresh = await self._reload_head(head)
                # The failed create proves ``ordinal`` is taken even if the winner's pointer write never
                # landed; children whose pointer writes never landed may occupy more, so allocate above the
                # highest stored one rather than rediscovering them one attempt at a time.
                fresh.latest_revision = max(fresh.latest_revision, ordinal, await self._highest_ordinal(fresh))
                head = fresh
                continue
            return (
                created,
                await self._point_tags(head, tags, ordinal, latest_frontmatter=created.frontmatter),
                True,
            )
        raise SkillRevisionConflictError(
            f"could not allocate a revision for skill '{head.workspace}/{head.name}' after {_MAX_ATTEMPTS} attempts"
        )

    async def _point_tags(
        self,
        head: SkillEntity,
        tags: set[str],
        ordinal: int,
        *,
        latest_frontmatter: SkillFrontmatter | None = None,
    ) -> SkillEntity:
        """Point ``tags`` at ``ordinal`` under the head's optimistic lock; ``latest`` only moves forward.

        When ``latest`` moves, the head's search copy of the frontmatter moves with it in the same write.
        """
        for attempt in range(_MAX_ATTEMPTS):
            moves_forward = ordinal >= head.tags.get(LATEST_TAG, 0)
            applied = {tag: ordinal for tag in tags if tag != LATEST_TAG or moves_forward}
            new_latest = max(head.latest_revision, ordinal)
            new_frontmatter = (
                latest_frontmatter if latest_frontmatter is not None and LATEST_TAG in applied else head.frontmatter
            )
            if (
                all(head.tags.get(t) == o for t, o in applied.items())
                and new_latest == head.latest_revision
                and new_frontmatter == head.frontmatter
            ):
                return head
            update = head.model_copy(
                update={"tags": {**head.tags, **applied}, "latest_revision": new_latest, "frontmatter": new_frontmatter}
            )
            update._parent, update._id, update._db_version = head._parent, head._id, head._db_version
            try:
                return await self.entity_client.update(update)
            except NemoEntityConflictError:
                logger.info(
                    "Skill pointer update contended, re-reading",
                    extra={"skill": f"{head.workspace}/{head.name}", "attempt": attempt + 1},
                )
                head = await self._reload_head(head)
        raise SkillRevisionConflictError(
            f"could not update revision pointers for skill '{head.workspace}/{head.name}' after {_MAX_ATTEMPTS} attempts"
        )


def _validate_skill_name(name: str) -> None:
    if not _SKILL_NAME_RE.fullmatch(name) or len(name) > SKILL_NAME_MAX_LENGTH:
        raise SkillValidationError(
            f"Skill name {name!r} must start with a letter and use lowercase alphanumerics separated by single hyphens "
            f"(2-{SKILL_NAME_MAX_LENGTH} characters)"
        )


def _fileset_identity(ref: str) -> tuple[str, str]:
    base = ref.partition("#")[0]
    workspace, _, fileset = base.partition("/")
    return workspace, fileset
