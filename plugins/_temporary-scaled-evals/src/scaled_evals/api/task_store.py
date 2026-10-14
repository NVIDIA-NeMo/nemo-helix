# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tasks and their revisions, stored in Entity Store as one entity per task."""

from __future__ import annotations

# `TaskStore.list` shadows the builtin inside the class body.
import builtins
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, TypeVar

from nemo_helix_plugin.entities.base import EntityConflictError, EntityNotFoundError, SyncEntityClient
from nemo_scaled_evals_plugin.entities import ScaledEvalsTask, ScaledEvalsTaskRevision

from scaled_evals.api import entity_store
from scaled_evals.api.repositories.base_repository import Conflict

T = TypeVar("T")

_MAX_WRITE_ATTEMPTS = 5
_CLAIMABLE_BACKENDS = frozenset({"image_builder_service", "cloudbuild", "buildkit", "prebuilt"})
_PLATFORM_JOB_PREFIX = "scaled-evals-build-"
_EPOCH = datetime.min.replace(tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class FinalizeRevision:
    revision: int
    status: str
    object_key: str
    quota_exceeded: bool = False
    previous_storage_bytes: int = 0
    uploaded_bytes: int = 0


@dataclass(frozen=True, slots=True)
class NewRevision:
    revision: int
    object_key: str


@dataclass(frozen=True, slots=True)
class PrebuiltFinalizeResult:
    revision: int
    status: str
    object_key: str
    finalized: bool
    quota_exceeded: bool = False
    previous_storage_bytes: int = 0
    uploaded_bytes: int = 0


@dataclass(frozen=True, slots=True)
class UploadingRevision:
    revision: int
    status: str
    object_key: str


@dataclass(frozen=True, slots=True)
class TaskPackRevision:
    task_id: str
    revision: int
    status: str
    object_key: str


@dataclass(frozen=True, slots=True)
class TaskBuildJob:
    task_id: str
    revision: int
    backend: str
    payload: dict[str, Any]
    credentials: dict[str, str]
    object_key: str
    attempt: int


def _now() -> datetime:
    return datetime.now(tz=UTC)


def _task_row(entity: ScaledEvalsTask) -> dict[str, Any]:
    return {
        "id": entity.name,
        "owner_id": entity.owner_id,
        "name": entity.display_name,
        "slug": entity.slug,
        "description": entity.description,
        "visibility": entity.visibility,
        "current_revision": entity.current_revision,
        "created_at": entity.created_at,
        "updated_at": entity.updated_at,
    }


def _latest(entity: ScaledEvalsTask) -> ScaledEvalsTaskRevision | None:
    return max(entity.revisions, key=lambda rev: rev.revision, default=None)


def _revision(entity: ScaledEvalsTask, revision: int | None) -> ScaledEvalsTaskRevision | None:
    """Return the exact revision, or the latest one when `revision` is None."""
    if revision is None:
        return _latest(entity)
    return next((rev for rev in entity.revisions if rev.revision == revision), None)


def _advance_current(entity: ScaledEvalsTask, revision: int) -> None:
    if entity.current_revision is None or entity.current_revision <= revision:
        entity.current_revision = revision


def _clear_claim(rev: ScaledEvalsTaskRevision) -> None:
    rev.build_claimed_at = None
    rev.build_claimed_by = None
    rev.build_next_attempt_at = None


def _mark_ready(
    rev: ScaledEvalsTaskRevision,
    *,
    image_ref: str,
    image_digest: str,
    verifier_image_ref: str | None,
    verifier_image_digest: str | None,
) -> None:
    rev.status = "ready"
    rev.image_ref = image_ref
    rev.image_digest = image_digest
    rev.verifier_image_ref = verifier_image_ref
    rev.verifier_image_digest = verifier_image_digest
    rev.build_error = None
    rev.build_completed_at = _now()
    _clear_claim(rev)


def _mark_failed(rev: ScaledEvalsTaskRevision, build_error: str) -> None:
    rev.status = "failed"
    rev.build_error = build_error
    rev.build_completed_at = _now()
    _clear_claim(rev)


def _lease_expired(rev: ScaledEvalsTaskRevision, claim_timeout: float, now: datetime) -> bool:
    return rev.build_claimed_at is not None and rev.build_claimed_at < now - timedelta(seconds=claim_timeout)


def _claimable(rev: ScaledEvalsTaskRevision, *, claim_timeout: float, max_attempts: int, now: datetime) -> bool:
    return (
        rev.status == "building"
        and rev.build_backend in _CLAIMABLE_BACKENDS
        and rev.build_attempts < max_attempts
        and (rev.build_next_attempt_at is None or rev.build_next_attempt_at <= now)
        and (rev.build_claimed_at is None or _lease_expired(rev, claim_timeout, now))
    )


def _held_by(rev: ScaledEvalsTaskRevision | None, worker_id: str) -> bool:
    return rev is not None and rev.status == "building" and rev.build_claimed_by == worker_id


class TaskStore:
    def __init__(self, client: SyncEntityClient | None = None) -> None:
        self._client = client or entity_store.entity_client()

    def _get(self, task_id: str, *, include_deleted: bool = False) -> ScaledEvalsTask | None:
        try:
            entity = self._client.get(ScaledEvalsTask, task_id, workspace=entity_store.private_workspace())
        except EntityNotFoundError:
            return None
        return entity if include_deleted or entity.deleted_at is None else None

    def _all(self) -> builtins.list[ScaledEvalsTask]:
        return entity_store.list_all(self._client, ScaledEvalsTask)

    def _live(self) -> builtins.list[ScaledEvalsTask]:
        return [entity for entity in self._all() if entity.deleted_at is None]

    def _mutate(
        self,
        task_id: str,
        change: Callable[[ScaledEvalsTask], tuple[T, bool]],
        *,
        include_deleted: bool = False,
    ) -> T | None:
        """Apply `change` to a fresh read and save it when it reports a write.

        Returns None when the task is absent. A concurrent write reruns
        `change` against the newer version, so it must re-check its own
        preconditions rather than trust an earlier read.
        """
        for _ in range(_MAX_WRITE_ATTEMPTS):
            entity = self._get(task_id, include_deleted=include_deleted)
            if entity is None:
                return None
            result, write = change(entity)
            if not write:
                return result
            try:
                self._client.update(entity)
            except EntityConflictError:
                continue
            return result
        raise Conflict("task_changed", "task changed concurrently; retry")

    def _ensure_slug_free(self, slug: str, *, task_id: str | None = None) -> None:
        # ponytail: check-then-write, so two concurrent writes claiming the
        # same slug can both land. Naming the entity by slug would close it,
        # but slugs are mutable and task ids are what everything references.
        if any(entity.slug == slug and entity.name != task_id for entity in self._live()):
            raise Conflict("slug_conflict", f"slug already exists: {slug}")

    def _owner_storage_bytes(self, entity: ScaledEvalsTask, revision: int) -> int:
        """Sum the owner's stored pack bytes, excluding this revision."""
        # ponytail: unlocked, so concurrent finalizes by one owner can each pass
        # the quota and overshoot it together. O(tasks) per finalize.
        others = [task for task in self._live() if task.owner_id == entity.owner_id and task.name != entity.name]
        return sum(
            rev.tarball_size_bytes or 0
            for task in [*others, entity]
            for rev in task.revisions
            if not (task.name == entity.name and rev.revision == revision)
        )

    def _quota_check(
        self, entity: ScaledEvalsTask, rev: ScaledEvalsTaskRevision, size: int | None, quota: int | None
    ) -> tuple[bool, int]:
        """Record the upload size; return whether it exceeds the quota, and the prior usage."""
        if size is None or quota is None:
            return False, 0
        previous = self._owner_storage_bytes(entity, rev.revision)
        if previous + size > quota:
            return True, previous
        rev.tarball_size_bytes = size
        return False, previous

    def create_with_initial_revision(
        self,
        task_id: str,
        *,
        name: str,
        slug: str,
        description: str | None,
        visibility: str,
        object_key: str,
        owner_id: str | None = None,
    ) -> dict[str, Any]:
        self._ensure_slug_free(slug)
        entity = ScaledEvalsTask(
            name=task_id,
            workspace=entity_store.private_workspace(),
            owner_id=owner_id,
            display_name=name,
            slug=slug,
            description=description,
            visibility=visibility,
            revisions=[ScaledEvalsTaskRevision(revision=1, tarball_object_key=object_key, created_at=_now())],
        )
        return _task_row(self._client.create(entity))

    def mark_latest_revision_building(
        self,
        task_id: str,
        *,
        build_backend: str | None = None,
        build_payload: dict[str, Any] | None = None,
        build_credentials: dict[str, str] | None = None,
        tarball_sha256: str | None = None,
        expected_revision: int | None = None,
        exact_revision: bool = False,
        tarball_size_bytes: int | None = None,
        tenant_storage_quota_bytes: int | None = None,
    ) -> FinalizeRevision | None:
        def change(entity: ScaledEvalsTask) -> tuple[FinalizeRevision | None, bool]:
            rev = _revision(entity, expected_revision if exact_revision else None)
            if rev is None:
                return None, False
            found = FinalizeRevision(revision=rev.revision, status=rev.status, object_key=rev.tarball_object_key)
            if (expected_revision is not None and rev.revision != expected_revision) or rev.status != "uploading":
                return found, False
            exceeded, previous = self._quota_check(entity, rev, tarball_size_bytes, tenant_storage_quota_bytes)
            sized = FinalizeRevision(
                revision=rev.revision,
                status=rev.status,
                object_key=rev.tarball_object_key,
                quota_exceeded=exceeded,
                previous_storage_bytes=previous,
                uploaded_bytes=tarball_size_bytes or 0,
            )
            if exceeded:
                return sized, False
            rev.status = "building"
            rev.build_started_at = rev.build_next_attempt_at = _now()
            rev.build_backend = build_backend
            rev.build_payload = dict(build_payload or {})
            rev.build_credentials = dict(build_credentials or {})
            rev.tarball_sha256 = tarball_sha256 or rev.tarball_sha256
            rev.build_error = rev.build_completed_at = None
            rev.build_claimed_at = rev.build_first_claimed_at = rev.build_claimed_by = None
            rev.build_attempts = 0
            return sized, True

        return self._mutate(task_id, change)

    def revision_for_finalize(self, task_id: str, *, expected_revision: int | None = None) -> UploadingRevision | None:
        """Return exact or latest revision metadata for object-store finalize guards."""
        entity = self._get(task_id)
        rev = _revision(entity, expected_revision) if entity is not None else None
        if rev is None:
            return None
        return UploadingRevision(revision=rev.revision, status=rev.status, object_key=rev.tarball_object_key)

    def finalize_latest_revision_prebuilt(
        self,
        task_id: str,
        *,
        image_ref: str,
        image_digest: str,
        tarball_sha256: str | None,
        expected_revision: int | None = None,
        exact_revision: bool = False,
        tarball_size_bytes: int | None = None,
        tenant_storage_quota_bytes: int | None = None,
        verifier_image_ref: str | None = None,
        verifier_image_digest: str | None = None,
    ) -> PrebuiltFinalizeResult | None:
        """Finalize the latest uploading revision with a prebuilt image.

        ``verifier_image_ref``/``verifier_image_digest`` name the separate verifier image, for tasks
        whose verifier runs in its own sandbox; leave them unset otherwise.
        """

        def change(entity: ScaledEvalsTask) -> tuple[PrebuiltFinalizeResult | None, bool]:
            rev = _revision(entity, expected_revision if exact_revision else None)
            if rev is None:
                return None, False
            if (expected_revision is not None and rev.revision != expected_revision) or rev.status != "uploading":
                return PrebuiltFinalizeResult(
                    revision=rev.revision, status=rev.status, object_key=rev.tarball_object_key, finalized=False
                ), False
            exceeded, previous = self._quota_check(entity, rev, tarball_size_bytes, tenant_storage_quota_bytes)
            if exceeded:
                return PrebuiltFinalizeResult(
                    revision=rev.revision,
                    status=rev.status,
                    object_key=rev.tarball_object_key,
                    finalized=False,
                    quota_exceeded=True,
                    previous_storage_bytes=previous,
                    uploaded_bytes=tarball_size_bytes or 0,
                ), False
            _mark_ready(
                rev,
                image_ref=image_ref,
                image_digest=image_digest,
                verifier_image_ref=verifier_image_ref,
                verifier_image_digest=verifier_image_digest,
            )
            rev.tarball_sha256 = tarball_sha256 or rev.tarball_sha256
            rev.build_backend = None
            rev.build_payload = {}
            rev.build_credentials = {}
            _advance_current(entity, rev.revision)
            return PrebuiltFinalizeResult(
                revision=rev.revision,
                status="ready",
                object_key=rev.tarball_object_key,
                finalized=True,
                previous_storage_bytes=previous,
                uploaded_bytes=tarball_size_bytes or 0,
            ), True

        return self._mutate(task_id, change)

    def create_next_revision(self, task_id: str) -> NewRevision | None:
        def change(entity: ScaledEvalsTask) -> tuple[NewRevision, bool]:
            latest = _latest(entity)
            number = (latest.revision if latest else 0) + 1
            object_key = f"{task_id}/rev/{number}/tarball.tar.gz"
            entity.revisions.append(
                ScaledEvalsTaskRevision(revision=number, tarball_object_key=object_key, created_at=_now())
            )
            entity.current_revision = number
            return NewRevision(revision=number, object_key=object_key), True

        return self._mutate(task_id, change)

    def task_pack_revisions_for_owner(
        self,
        *,
        owner_id: str,
        statuses: tuple[str, ...] = ("ready", "building"),
        limit: int = 500,
    ) -> builtins.list[TaskPackRevision]:
        tasks = sorted(
            (entity for entity in self._live() if entity.owner_id == owner_id),
            key=lambda entity: (-entity.updated_at.timestamp(), entity.name),
        )
        return [
            TaskPackRevision(
                task_id=entity.name, revision=rev.revision, status=rev.status, object_key=rev.tarball_object_key
            )
            for entity in tasks
            for rev in sorted(entity.revisions, key=lambda rev: -rev.revision)
            if rev.status in statuses
        ][:limit]

    def mark_task_pack_missing(
        self,
        *,
        owner_id: str,
        task_id: str,
        revision: int,
        object_key: str,
        previous_status: str,
    ) -> bool:
        def change(entity: ScaledEvalsTask) -> tuple[bool, bool]:
            rev = _revision(entity, revision)
            if (
                entity.owner_id != owner_id
                or rev is None
                or rev.status != previous_status
                or rev.tarball_object_key != object_key
            ):
                return False, False
            _mark_failed(rev, f"task_object_missing: task pack object is missing: {object_key}")
            ready = [other.revision for other in entity.revisions if other.status == "ready"]
            if entity.current_revision == revision and ready:
                entity.current_revision = max(ready)
            return True, True

        return bool(self._mutate(task_id, change))

    def list(
        self,
        *,
        limit: int,
        cursor: str | None,
        order: str,
        owner_id: str | None = None,
        q: str | None = None,
    ) -> builtins.list[dict[str, Any]]:
        # ponytail: in-process filtering over every task, as for config
        # profiles; O(n) per list.
        rows = [
            _task_row(entity)
            for entity in self._live()
            if (owner_id is None or entity.owner_id == owner_id)
            and entity_store.contains(q, entity.name, entity.display_name, entity.slug, entity.description)
        ]
        return entity_store.keyset_page(rows, cursor=cursor, order=order, limit=limit)

    def get(self, task_id: str) -> dict[str, Any] | None:
        entity = self._get(task_id)
        return _task_row(entity) if entity is not None else None

    def get_by_slug(self, slug: str) -> dict[str, Any] | None:
        entity = next((entity for entity in self._live() if entity.slug == slug), None)
        return _task_row(entity) if entity is not None else None

    def get_detail(self, task_id: str) -> dict[str, Any] | None:
        """Return the task with its latest revision's fields, which are None before any revision."""
        entity = self._get(task_id)
        if entity is None:
            return None
        rev = _latest(entity)
        return {
            **_task_row(entity),
            **{
                field: getattr(rev, field, None)
                for field in (
                    "revision",
                    "status",
                    "image_ref",
                    "image_digest",
                    "verifier_image_ref",
                    "verifier_image_digest",
                    "build_error",
                    "tarball_size_bytes",
                    "tarball_sha256",
                    "tarball_object_key",
                )
            },
            "revision_created_at": rev.created_at if rev else None,
        }

    def revision(self, task_id: str, revision: int, *, include_deleted: bool = True) -> dict[str, Any] | None:
        """Return one revision's fields with the task's `task_slug`, `task_name` and `task_deleted`."""
        entity = self._get(task_id, include_deleted=include_deleted)
        rev = _revision(entity, revision) if entity is not None else None
        if entity is None or rev is None:
            return None
        return self._view(entity, rev)

    def revisions(self, keys: Iterable[tuple[str, int]]) -> dict[tuple[str, int], dict[str, Any]]:
        """Return `revision()` for each `(task_id, revision)`, reading every task once."""
        wanted = set(keys)
        found: dict[tuple[str, int], dict[str, Any]] = {}
        for task_id in {task_id for task_id, _ in wanted}:
            entity = self._get(task_id, include_deleted=True)
            if entity is None:
                continue
            for rev in entity.revisions:
                if (task_id, rev.revision) in wanted:
                    found[(task_id, rev.revision)] = self._view(entity, rev)
        return found

    @staticmethod
    def _view(entity: ScaledEvalsTask, rev: ScaledEvalsTaskRevision) -> dict[str, Any]:
        return {
            **rev.model_dump(),
            "task_id": entity.name,
            "task_slug": entity.slug,
            "task_name": entity.display_name,
            "task_deleted": entity.deleted_at is not None,
        }

    def resolve_members(
        self, members: Iterable[dict[str, Any]], *, include_deleted: bool
    ) -> builtins.list[tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]]:
        """Pair each member row with its task and the revision it runs.

        A member's None `task_revision` floats to the task's current revision; the
        revision is None when it does not exist. Members whose task is missing,
        or soft-deleted unless `include_deleted`, are dropped.
        """
        members = builtins.list(members)
        entities = {
            task_id: self._get(task_id, include_deleted=include_deleted) for task_id in {m["task_id"] for m in members}
        }
        resolved = []
        for member in members:
            entity = entities[member["task_id"]]
            if entity is None:
                continue
            number = member["task_revision"] if member["task_revision"] is not None else entity.current_revision
            rev = _revision(entity, number) if number is not None else None
            resolved.append((member, _task_row(entity), self._view(entity, rev) if rev else None))
        return resolved

    def tasks(self, task_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
        """Return the task rows that exist, including soft-deleted ones, by id."""
        found = {}
        for task_id in set(task_ids):
            entity = self._get(task_id, include_deleted=True)
            if entity is not None:
                found[task_id] = {**_task_row(entity), "deleted": entity.deleted_at is not None}
        return found

    def update(
        self,
        task_id: str,
        *,
        name: str | None = None,
        slug: str | None = None,
        description: str | None = None,
        visibility: str | None = None,
    ) -> dict[str, Any] | None:
        if slug is not None:
            self._ensure_slug_free(slug, task_id=task_id)
        changes = {
            "display_name": name,
            "slug": slug,
            "description": description,
            "visibility": visibility,
        }

        def change(entity: ScaledEvalsTask) -> tuple[bool, bool]:
            for field, value in changes.items():
                if value is not None:
                    setattr(entity, field, value)
            return True, any(value is not None for value in changes.values())

        if self._mutate(task_id, change) is None:
            return None
        return self.get(task_id)

    def soft_delete(self, task_id: str) -> bool:
        def change(entity: ScaledEvalsTask) -> tuple[bool, bool]:
            entity.deleted_at = _now()
            return True, True

        return bool(self._mutate(task_id, change))

    def owner_summary(self, owner_id: str) -> dict[str, int]:
        """Count the owner's live tasks whose latest revision is ready, or still building."""
        statuses = [rev.status for entity in self._live() if entity.owner_id == owner_id if (rev := _latest(entity))]
        return {
            "ready": statuses.count("ready"),
            "building": sum(status in ("pending", "uploading", "building") for status in statuses),
        }

    def count(self, *, owner_id: str | None = None, include_deleted: bool = False) -> int:
        tasks = self._all() if include_deleted else self._live()
        return sum(owner_id is None or entity.owner_id == owner_id for entity in tasks)

    def owner_activity(self, owner_id: str) -> builtins.list[dict[str, Any]]:
        """Return the owner's live tasks as activity rows with their latest revision status."""
        return [
            {
                "id": entity.name,
                "kind": "task",
                "name": entity.display_name,
                "status": rev.status if (rev := _latest(entity)) else None,
                "created_at": entity.created_at,
            }
            for entity in self._live()
            if entity.owner_id == owner_id
        ]

    def ready_task_pack_revisions(self, *, limit: int) -> builtins.list[dict[str, Any]]:
        tasks = sorted(self._live(), key=lambda entity: (-entity.updated_at.timestamp(), entity.name))
        return [
            {
                "owner_id": entity.owner_id,
                "task_id": entity.name,
                "revision": rev.revision,
                "tarball_object_key": rev.tarball_object_key,
            }
            for entity in tasks
            for rev in sorted(entity.revisions, key=lambda rev: -rev.revision)
            if rev.status == "ready"
        ][:limit]

    def record_success(self, task_id: str, revision: int, *, image_ref: str, image_digest: str) -> None:
        def change(entity: ScaledEvalsTask) -> tuple[None, bool]:
            rev = _revision(entity, revision)
            if rev is None or rev.status != "building":
                return None, False
            rev.status = "ready"
            rev.image_ref = image_ref
            rev.image_digest = image_digest
            rev.build_completed_at = _now()
            _advance_current(entity, revision)
            return None, True

        self._mutate(task_id, change, include_deleted=True)

    def record_failure(self, task_id: str, revision: int, build_error: str) -> None:
        def change(entity: ScaledEvalsTask) -> tuple[None, bool]:
            rev = _revision(entity, revision)
            if rev is None or rev.status != "building":
                return None, False
            rev.status = "failed"
            rev.build_error = build_error
            rev.build_completed_at = _now()
            return None, True

        self._mutate(task_id, change, include_deleted=True)

    def claim_next(self, *, worker_id: str, claim_timeout: float, max_attempts: int) -> TaskBuildJob | None:
        # ponytail: scans every task per claim, O(tasks); a status-indexed
        # build entity would make this a filtered list.
        now = _now()
        building = [
            (entity.name, rev)
            for entity in self._all()
            for rev in entity.revisions
            if rev.status == "building" and rev.build_backend in _CLAIMABLE_BACKENDS
        ]
        for task_id, rev in building:
            # A worker can die after claiming its final allowed attempt. Once
            # that lease expires there is no attempt left to claim, so the
            # revision becomes terminal rather than staying `building` forever.
            if rev.build_attempts >= max_attempts and _lease_expired(rev, claim_timeout, now):
                self._expire_final_attempt(
                    task_id, rev.revision, claim_timeout=claim_timeout, max_attempts=max_attempts
                )
        candidates = sorted(
            (
                (task_id, rev)
                for task_id, rev in building
                if _claimable(rev, claim_timeout=claim_timeout, max_attempts=max_attempts, now=now)
            ),
            key=lambda item: (item[1].build_started_at or _EPOCH, item[0], item[1].revision),
        )
        for task_id, candidate in candidates:
            job = self._claim(
                task_id, candidate.revision, worker_id=worker_id, claim_timeout=claim_timeout, max_attempts=max_attempts
            )
            if job is not None:
                return job
        return None

    def _expire_final_attempt(self, task_id: str, revision: int, *, claim_timeout: float, max_attempts: int) -> None:
        def change(entity: ScaledEvalsTask) -> tuple[None, bool]:
            rev = _revision(entity, revision)
            if (
                rev is None
                or rev.status != "building"
                or rev.build_attempts < max_attempts
                or not _lease_expired(rev, claim_timeout, _now())
            ):
                return None, False
            _mark_failed(rev, rev.build_error or "build worker lease expired after final attempt")
            return None, True

        self._mutate(task_id, change, include_deleted=True)

    def _claim(
        self, task_id: str, revision: int, *, worker_id: str, claim_timeout: float, max_attempts: int
    ) -> TaskBuildJob | None:
        def change(entity: ScaledEvalsTask) -> tuple[TaskBuildJob | None, bool]:
            rev = _revision(entity, revision)
            now = _now()
            if rev is None or not _claimable(rev, claim_timeout=claim_timeout, max_attempts=max_attempts, now=now):
                return None, False
            rev.build_claimed_at = now
            rev.build_first_claimed_at = rev.build_first_claimed_at or now
            rev.build_claimed_by = worker_id
            rev.build_attempts += 1
            return TaskBuildJob(
                task_id=task_id,
                revision=revision,
                backend=str(rev.build_backend),
                payload=dict(rev.build_payload),
                credentials=dict(rev.build_credentials),
                object_key=rev.tarball_object_key,
                attempt=rev.build_attempts,
            ), True

        return self._mutate(task_id, change, include_deleted=True)

    def bind_platform_job(self, task_id: str, revision: int, *, worker_id: str, job_name: str) -> bool:
        """Transfer a short-lived controller claim to a durable Platform Job."""

        def change(entity: ScaledEvalsTask) -> tuple[bool, bool]:
            rev = _revision(entity, revision)
            if rev is None or not _held_by(rev, worker_id):
                return False, False
            rev.build_claimed_by = job_name
            rev.build_claimed_at = _now()
            return True, True

        return bool(self._mutate(task_id, change, include_deleted=True))

    def list_platform_jobs(self, *, limit: int = 100) -> builtins.list[dict[str, Any]]:
        """List active task builds currently owned by Platform Jobs."""
        rows = [
            (rev.build_claimed_at or _EPOCH, entity.name, rev)
            for entity in self._all()
            for rev in entity.revisions
            if rev.status == "building" and (rev.build_claimed_by or "").startswith(_PLATFORM_JOB_PREFIX)
        ]
        rows.sort(key=lambda row: (row[0], row[1], row[2].revision))
        return [
            {
                "task_id": task_id,
                "revision": rev.revision,
                "build_attempts": rev.build_attempts,
                "job_name": rev.build_claimed_by,
            }
            for _, task_id, rev in rows[:limit]
        ]

    def heartbeat(self, task_id: str, revision: int, *, worker_id: str) -> bool:
        def change(entity: ScaledEvalsTask) -> tuple[bool, bool]:
            rev = _revision(entity, revision)
            if rev is None or not _held_by(rev, worker_id):
                return False, False
            rev.build_claimed_at = _now()
            rev.build_first_claimed_at = rev.build_first_claimed_at or rev.build_claimed_at
            return True, True

        return bool(self._mutate(task_id, change, include_deleted=True))

    def complete(
        self,
        task_id: str,
        revision: int,
        *,
        worker_id: str,
        image_ref: str,
        image_digest: str,
        verifier_image_ref: str | None = None,
        verifier_image_digest: str | None = None,
    ) -> bool:
        def change(entity: ScaledEvalsTask) -> tuple[bool, bool]:
            rev = _revision(entity, revision)
            if rev is None or not _held_by(rev, worker_id):
                return False, False
            _mark_ready(
                rev,
                image_ref=image_ref,
                image_digest=image_digest,
                verifier_image_ref=verifier_image_ref,
                verifier_image_digest=verifier_image_digest,
            )
            _advance_current(entity, revision)
            return True, True

        return bool(self._mutate(task_id, change, include_deleted=True))

    def retry_or_fail(
        self,
        task_id: str,
        revision: int,
        *,
        worker_id: str,
        build_error: str,
        attempt: int,
        max_attempts: int,
        retry_delay: float,
    ) -> bool:
        def change(entity: ScaledEvalsTask) -> tuple[bool, bool]:
            rev = _revision(entity, revision)
            if rev is None or not _held_by(rev, worker_id):
                return False, False
            if attempt >= max_attempts:
                _mark_failed(rev, build_error)
            else:
                rev.build_error = build_error
                rev.build_claimed_at = rev.build_claimed_by = None
                rev.build_next_attempt_at = _now() + timedelta(seconds=retry_delay)
            return True, True

        return bool(self._mutate(task_id, change, include_deleted=True))
