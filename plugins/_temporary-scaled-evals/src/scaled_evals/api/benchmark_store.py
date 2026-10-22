# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmarks and their revisions, stored in Entity Store as one entity per benchmark."""

from __future__ import annotations

# `BenchmarkStore.list` shadows the builtin inside the class body.
import builtins
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, TypeVar

from nemo_helix_plugin.entities.base import EntityConflictError, EntityNotFoundError, SyncEntityClient
from nemo_scaled_evals_plugin.entities import (
    ScaledEvalsBenchmark,
    ScaledEvalsBenchmarkMember,
    ScaledEvalsBenchmarkRevision,
)

from scaled_evals.api import entity_store
from scaled_evals.api.repositories.base_repository import Conflict, InvalidReference
from scaled_evals.api.task_store import TaskStore

T = TypeVar("T")

_MAX_WRITE_ATTEMPTS = 5


@dataclass(frozen=True, slots=True)
class TaskRef:
    task_id: str
    task_revision: int | None = None


def _now() -> datetime:
    return datetime.now(tz=UTC)


def _row(entity: ScaledEvalsBenchmark) -> dict[str, Any]:
    return {
        "id": entity.name,
        "owner_id": entity.owner_id,
        "name": entity.display_name,
        "slug": entity.slug,
        "description": entity.description,
        "visibility": entity.visibility,
        "qualification_status": entity.qualification_status,
        "qualification_evidence": entity.qualification_evidence,
        "qualified_at": entity.qualified_at,
        "qualified_by": entity.qualified_by,
        "current_revision": entity.current_revision,
        "created_at": entity.created_at,
        "updated_at": entity.updated_at,
    }


def _revision(entity: ScaledEvalsBenchmark, revision: int) -> ScaledEvalsBenchmarkRevision | None:
    return next((rev for rev in entity.revisions if rev.revision == revision), None)


def _meta(rev: ScaledEvalsBenchmarkRevision) -> dict[str, Any]:
    return {
        "derived_from_benchmark_id": rev.derived_from_benchmark_id,
        "derived_from_revision": rev.derived_from_revision,
        "operational_policy": rev.operational_policy,
    }


def _validated_members(tasks: builtins.list[TaskRef]) -> builtins.list[ScaledEvalsBenchmarkMember]:
    """Return the member records for `tasks`.

    Each task must be live and a pinned ``task_revision`` must exist.

    Raises:
        InvalidReference: on a missing or duplicated member (router → 422).

    """
    store = TaskStore()
    live = {task_id for task_id, task in store.tasks(ref.task_id for ref in tasks).items() if not task["deleted"]}
    pinned = store.revisions((ref.task_id, ref.task_revision) for ref in tasks if ref.task_revision is not None)
    seen: set[str] = set()
    members = []
    for position, ref in enumerate(tasks):
        if ref.task_id in seen:
            raise InvalidReference(f"duplicate task in benchmark: {ref.task_id}")
        seen.add(ref.task_id)
        if ref.task_id not in live:
            raise InvalidReference(f"task not found: {ref.task_id}")
        if ref.task_revision is not None and (ref.task_id, ref.task_revision) not in pinned:
            raise InvalidReference(f"task revision not found: {ref.task_id} rev {ref.task_revision}")
        members.append(
            ScaledEvalsBenchmarkMember(task_id=ref.task_id, task_revision=ref.task_revision, position=position)
        )
    return members


class BenchmarkStore:
    def __init__(self, client: SyncEntityClient | None = None) -> None:
        self._client = client or entity_store.entity_client()

    def _get(self, benchmark_id: str, *, include_deleted: bool = False) -> ScaledEvalsBenchmark | None:
        try:
            entity = self._client.get(ScaledEvalsBenchmark, benchmark_id, workspace=entity_store.private_workspace())
        except EntityNotFoundError:
            return None
        return entity if include_deleted or entity.deleted_at is None else None

    def _live(self) -> builtins.list[ScaledEvalsBenchmark]:
        return [
            entity for entity in entity_store.list_all(self._client, ScaledEvalsBenchmark) if entity.deleted_at is None
        ]

    def _mutate(self, benchmark_id: str, change: Callable[[ScaledEvalsBenchmark], tuple[T, bool]]) -> T | None:
        """Apply `change` to a fresh read of a live benchmark and save it when it reports a write.

        Returns None when the benchmark is absent. A concurrent write reruns
        `change` against the newer version.
        """
        for _ in range(_MAX_WRITE_ATTEMPTS):
            entity = self._get(benchmark_id)
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
        raise Conflict("benchmark_changed", "benchmark changed concurrently; retry")

    def _ensure_slug_free(self, slug: str, *, benchmark_id: str | None = None) -> None:
        # ponytail: check-then-write, as for tasks. Concurrent creates are settled
        # afterwards by `_slug_owner`; a concurrent rename onto a taken slug can
        # still land. Entity Store has no secondary unique index, and naming the
        # entity by slug would break the ids benchmark runs reference.
        if any(entity.slug == slug and entity.name != benchmark_id for entity in self._live()):
            raise Conflict("slug_conflict", f"slug already exists: {slug}")

    def _slug_owner(self, slug: str) -> ScaledEvalsBenchmark | None:
        """Return the oldest live benchmark with `slug`, so every reader agrees on one."""
        holders = [entity for entity in self._live() if entity.slug == slug]
        return min(holders, key=lambda entity: (entity.created_at, entity.name), default=None)

    def _create(
        self,
        benchmark_id: str,
        *,
        name: str,
        slug: str,
        description: str | None,
        visibility: str,
        revision: ScaledEvalsBenchmarkRevision,
        owner_id: str | None,
    ) -> dict[str, Any]:
        self._ensure_slug_free(slug)
        entity = ScaledEvalsBenchmark(
            name=benchmark_id,
            workspace=entity_store.private_workspace(),
            owner_id=owner_id,
            display_name=name,
            slug=slug,
            description=description,
            visibility=visibility,
            current_revision=revision.revision,
            revisions=[revision],
        )
        created = self._client.create(entity)
        owner = self._slug_owner(slug)
        if owner is not None and owner.name != benchmark_id:
            # A concurrent create took the slug first; withdraw this one.
            self.soft_delete(benchmark_id)
            raise Conflict("slug_conflict", f"slug already exists: {slug}")
        return _row(created)

    def create_with_initial_revision(
        self,
        benchmark_id: str,
        *,
        name: str,
        slug: str,
        description: str | None,
        visibility: str,
        tasks: builtins.list[TaskRef],
        owner_id: str | None = None,
    ) -> dict[str, Any]:
        revision = ScaledEvalsBenchmarkRevision(
            revision=1, description=description, members=_validated_members(tasks), created_at=_now()
        )
        return self._create(
            benchmark_id,
            name=name,
            slug=slug,
            description=description,
            visibility=visibility,
            revision=revision,
            owner_id=owner_id,
        )

    def create_variant(
        self,
        variant_id: str,
        *,
        base_benchmark_id: str,
        from_revision: int | None,
        name: str,
        slug: str,
        description: str | None,
        visibility: str,
        operational_policy: dict[str, Any],
        owner_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Derive a new benchmark that copies base member pins and stores policy.

        Returns the new benchmark row plus ``derived_from``, ``operational_policy``
        and ``revision``, or None if the base benchmark is missing.

        Raises:
            InvalidReference: if the base has no such revision.

        """
        base = self._get(base_benchmark_id)
        if base is None:
            return None
        source_revision = from_revision or base.current_revision
        if source_revision is None:
            raise InvalidReference("base benchmark has no revision to derive from")
        source = _revision(base, source_revision)
        if source is None:
            raise InvalidReference(f"benchmark revision not found: {base_benchmark_id} rev {source_revision}")
        refs = [
            TaskRef(m.task_id, m.task_revision) for m in sorted(source.members, key=lambda m: (m.position, m.task_id))
        ]
        revision = ScaledEvalsBenchmarkRevision(
            revision=1,
            description=description,
            derived_from_benchmark_id=base_benchmark_id,
            derived_from_revision=source_revision,
            operational_policy=operational_policy,
            members=_validated_members(refs),
            created_at=_now(),
        )
        row = self._create(
            variant_id,
            name=name,
            slug=slug,
            description=description,
            visibility=visibility,
            revision=revision,
            owner_id=owner_id,
        )
        return {
            **row,
            "revision": 1,
            "derived_from": {"benchmark_id": base_benchmark_id, "revision": source_revision},
            "operational_policy": operational_policy,
        }

    def get_revision_meta(self, benchmark_id: str, revision: int) -> dict[str, Any] | None:
        """Return lineage and policy for one benchmark revision, deleted benchmarks included."""
        entity = self._get(benchmark_id, include_deleted=True)
        rev = _revision(entity, revision) if entity is not None else None
        return _meta(rev) if rev is not None else None

    def create_next_revision(
        self, benchmark_id: str, *, description: str | None, tasks: builtins.list[TaskRef]
    ) -> int | None:
        members = _validated_members(tasks)

        def change(entity: ScaledEvalsBenchmark) -> tuple[int, bool]:
            number = max((rev.revision for rev in entity.revisions), default=0) + 1
            entity.revisions.append(
                ScaledEvalsBenchmarkRevision(
                    revision=number, description=description, members=members, created_at=_now()
                )
            )
            entity.current_revision = number
            return number, True

        return self._mutate(benchmark_id, change)

    def list(
        self, *, limit: int, cursor: str | None, order: str, q: str | None = None
    ) -> builtins.list[dict[str, Any]]:
        # ponytail: in-process filtering over every benchmark, as for tasks; O(n) per list.
        rows = [
            _row(entity)
            for entity in self._live()
            if entity_store.contains(q, entity.name, entity.display_name, entity.slug, entity.description)
        ]
        return entity_store.keyset_page(rows, cursor=cursor, order=order, limit=limit)

    def get(self, benchmark_id: str) -> dict[str, Any] | None:
        entity = self._get(benchmark_id)
        return _row(entity) if entity is not None else None

    def get_by_slug(self, slug: str) -> dict[str, Any] | None:
        entity = self._slug_owner(slug)
        return _row(entity) if entity is not None else None

    def slug(self, benchmark_id: str) -> str | None:
        """Return the slug of a benchmark, deleted ones included."""
        entity = self._get(benchmark_id, include_deleted=True)
        return entity.slug if entity is not None else None

    def members(self, benchmark_id: str, revision: int) -> builtins.list[dict[str, Any]]:
        """Return `(task_id, task_revision, position)` rows of one revision in benchmark order."""
        entity = self._get(benchmark_id, include_deleted=True)
        rev = _revision(entity, revision) if entity is not None else None
        if rev is None:
            return []
        return [member.model_dump() for member in sorted(rev.members, key=lambda m: (m.position, m.task_id))]

    def revision_for_run(self, benchmark_id: str, revision: int | None = None) -> dict[str, Any] | None:
        """Resolve a live benchmark revision and its advisory qualification state."""
        entity = self._get(benchmark_id)
        if entity is None or (revision is not None and _revision(entity, revision) is None):
            return None
        return {
            "revision": revision if revision is not None else entity.current_revision,
            "qualification_status": entity.qualification_status,
            "qualification_evidence": entity.qualification_evidence,
            "qualified_at": entity.qualified_at,
        }

    def list_tasks(
        self,
        benchmark_id: str,
        revision: int,
        *,
        limit: int | None = None,
        cursor: int | None = None,
    ) -> builtins.list[dict[str, Any]]:
        """Return a revision's members with their task's slug and name, from position `cursor`."""
        rows = [row for row in self.members(benchmark_id, revision) if cursor is None or row["position"] >= cursor]
        if limit is not None:
            rows = rows[: limit + 1]
        tasks = TaskStore().tasks(row["task_id"] for row in rows)
        return [
            {**row, "task_slug": task["slug"], "task_name": task["name"]}
            for row in rows
            if (task := tasks.get(row["task_id"]))
        ]

    def list_resolved_tasks(self, benchmark_id: str, revision: int) -> builtins.list[dict[str, Any]]:
        """List members with their effective immutable task revision and build state."""
        fields = ("status", "image_ref", "image_digest", "tarball_object_key", "tarball_size_bytes")
        members = self.members(benchmark_id, revision)
        return [
            {
                **member,
                "task_slug": task["slug"],
                "task_name": task["name"],
                "resolved_task_revision": rev["revision"] if rev else task["current_revision"],
                **{field: rev[field] if rev else None for field in fields},
            }
            for member, task, rev in TaskStore().resolve_members(members, include_deleted=False)
        ]

    def get_detail(self, benchmark_id: str) -> dict[str, Any] | None:
        entity = self._get(benchmark_id)
        if entity is None:
            return None
        number = entity.current_revision
        rev = _revision(entity, number) if number is not None else None
        derived_from = None
        if rev is not None and rev.derived_from_benchmark_id is not None:
            derived_from = {"benchmark_id": rev.derived_from_benchmark_id, "revision": rev.derived_from_revision}
        return {
            **_row(entity),
            "revision": number,
            "tasks": self.list_tasks(benchmark_id, number) if number is not None else [],
            "derived_from": derived_from,
            "operational_policy": dict(rev.operational_policy) if rev is not None else {},
        }

    def update(
        self,
        benchmark_id: str,
        *,
        name: str | None = None,
        slug: str | None = None,
        description: str | None = None,
        visibility: str | None = None,
    ) -> dict[str, Any] | None:
        if slug is not None:
            self._ensure_slug_free(slug, benchmark_id=benchmark_id)
        fields = {"display_name": name, "slug": slug, "description": description, "visibility": visibility}
        changes = {field: value for field, value in fields.items() if value is not None}

        def change(entity: ScaledEvalsBenchmark) -> tuple[bool, bool]:
            for field, value in changes.items():
                setattr(entity, field, value)
            return True, bool(changes)

        return self.get(benchmark_id) if self._mutate(benchmark_id, change) else None

    def soft_delete(self, benchmark_id: str) -> bool:
        def change(entity: ScaledEvalsBenchmark) -> tuple[bool, bool]:
            entity.deleted_at = _now()
            return True, True

        return bool(self._mutate(benchmark_id, change))

    def set_qualification(
        self,
        benchmark_id: str,
        *,
        status: str,
        evidence: dict[str, Any],
        qualified_by: str,
    ) -> dict[str, Any] | None:
        def change(entity: ScaledEvalsBenchmark) -> tuple[bool, bool]:
            entity.qualification_status = status
            entity.qualification_evidence = evidence
            entity.qualified_at = _now()
            entity.qualified_by = qualified_by
            if status == "rejected":
                entity.visibility = "private"
            return True, True

        return self.get(benchmark_id) if self._mutate(benchmark_id, change) else None

    def promote(self, benchmark_id: str, *, qualified_by: str) -> dict[str, Any] | None:
        def change(entity: ScaledEvalsBenchmark) -> tuple[bool, bool]:
            if entity.qualification_status != "qualified":
                return False, False
            entity.visibility = "public"
            entity.qualified_by = qualified_by
            return True, True

        return self.get(benchmark_id) if self._mutate(benchmark_id, change) else None
