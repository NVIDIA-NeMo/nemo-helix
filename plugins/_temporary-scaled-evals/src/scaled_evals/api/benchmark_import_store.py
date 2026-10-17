# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark imports, stored in Entity Store as one entity per import."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any, TypeVar

from nemo_helix_plugin.entities.base import EntityConflictError, EntityNotFoundError, SyncEntityClient
from nemo_scaled_evals_plugin.entities import (
    ScaledEvalsBenchmarkImport,
    ScaledEvalsBenchmarkImportBenchmark,
    ScaledEvalsBenchmarkImportTask,
)

from scaled_evals.api import entity_store
from scaled_evals.api.repositories.base_repository import Conflict
from scaled_evals.api.task_store import TaskStore

T = TypeVar("T")

_MAX_WRITE_ATTEMPTS = 5


def import_id_for(owner_id: str, manifest_sha256: str, visibility: str) -> str:
    """Return the id of the one import an owner may have for a manifest and visibility."""
    identity = hashlib.sha256(f"{owner_id}\0{manifest_sha256}\0{visibility}".encode()).hexdigest()
    return f"bmi_{identity[:26]}"


def _row(entity: ScaledEvalsBenchmarkImport) -> dict[str, Any]:
    return {
        "id": entity.name,
        "owner_id": entity.owner_id,
        "manifest_sha256": entity.manifest_sha256,
        "manifest": entity.manifest,
        "visibility": entity.visibility,
        "description": entity.description,
        "created_at": entity.created_at,
        "updated_at": entity.updated_at,
    }


class BenchmarkImportStore:
    def __init__(self, client: SyncEntityClient | None = None) -> None:
        self._client = client or entity_store.entity_client()

    def _get(self, import_id: str) -> ScaledEvalsBenchmarkImport | None:
        try:
            return self._client.get(ScaledEvalsBenchmarkImport, import_id, workspace=entity_store.private_workspace())
        except EntityNotFoundError:
            return None

    def _mutate(self, import_id: str, change: Callable[[ScaledEvalsBenchmarkImport], tuple[T, bool]]) -> T | None:
        """Apply `change` to a fresh read and save it when it reports a write; None if absent."""
        for _ in range(_MAX_WRITE_ATTEMPTS):
            entity = self._get(import_id)
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
        raise Conflict("benchmark_import_changed", "benchmark import changed concurrently; retry")

    def create(
        self,
        import_id: str,
        *,
        owner_id: str,
        manifest_sha256: str,
        manifest: dict[str, Any],
        visibility: str,
        description: str | None,
    ) -> dict[str, Any] | None:
        """Create the import; None if one with this id already exists."""
        entity = ScaledEvalsBenchmarkImport(
            name=import_id,
            workspace=entity_store.private_workspace(),
            owner_id=owner_id,
            manifest_sha256=manifest_sha256,
            manifest=manifest,
            visibility=visibility,
            description=description,
        )
        try:
            return _row(self._client.create(entity))
        except EntityConflictError:
            return None

    def delete(self, import_id: str) -> None:
        try:
            self._client.delete(ScaledEvalsBenchmarkImport, import_id, workspace=entity_store.private_workspace())
        except EntityNotFoundError:
            pass

    def get(self, import_id: str, *, owner_id: str) -> dict[str, Any] | None:
        entity = self._get(import_id)
        return _row(entity) if entity is not None and entity.owner_id == owner_id else None

    def add_task(
        self,
        import_id: str,
        *,
        position: int,
        slug: str,
        task_id: str,
        task_revision: int,
        pack_path: str,
        pack_sha256: str,
        image_ref: str | None,
        image_digest: str | None,
        image_metadata: dict[str, Any] | None = None,
    ) -> None:
        task = ScaledEvalsBenchmarkImportTask(
            position=position,
            slug=slug,
            task_id=task_id,
            task_revision=task_revision,
            pack_path=pack_path,
            pack_sha256=pack_sha256,
            image_ref=image_ref,
            image_digest=image_digest,
            image_metadata=image_metadata or {},
        )

        def change(entity: ScaledEvalsBenchmarkImport) -> tuple[None, bool]:
            if any(existing.slug == slug for existing in entity.tasks):
                raise Conflict("duplicate_import_task", f"import already has task {slug!r}")
            entity.tasks.append(task)
            return None, True

        self._mutate(import_id, change)

    def attach_task_image(
        self,
        import_id: str,
        slug: str,
        *,
        image_ref: str,
        image_digest: str,
        image_metadata: dict[str, Any],
    ) -> bool:
        """Attach one immutable image identity, accepting exact idempotent replays."""

        def change(entity: ScaledEvalsBenchmarkImport) -> tuple[bool, bool]:
            task = next((task for task in entity.tasks if task.slug == slug), None)
            if task is None:
                return False, False
            # ponytail: the uploading check and the write are not atomic; a
            # finalize landing in between leaves an unused requested image.
            revision = TaskStore().revision(task.task_id, task.task_revision)
            if not revision or revision["status"] != "uploading":
                return False, False
            if task.image_ref is not None and (task.image_ref, task.image_digest) != (image_ref, image_digest):
                return False, False
            task.image_ref, task.image_digest, task.image_metadata = image_ref, image_digest, image_metadata
            return True, True

        return bool(self._mutate(import_id, change))

    def add_benchmark(
        self,
        import_id: str,
        *,
        position: int,
        slug: str,
        name: str,
        task_slugs: list[str],
    ) -> None:
        benchmark = ScaledEvalsBenchmarkImportBenchmark(position=position, slug=slug, name=name, task_slugs=task_slugs)

        def change(entity: ScaledEvalsBenchmarkImport) -> tuple[None, bool]:
            if any(existing.slug == slug for existing in entity.benchmarks):
                raise Conflict("duplicate_import_benchmark", f"import already has benchmark {slug!r}")
            entity.benchmarks.append(benchmark)
            return None, True

        self._mutate(import_id, change)

    def tasks(self, import_id: str) -> list[dict[str, Any]]:
        """Return the import's tasks in manifest order, merged with their task revision's state.

        A task whose revision no longer exists is skipped.
        """
        entity = self._get(import_id)
        if entity is None:
            return []
        rows = sorted(entity.tasks, key=lambda task: task.position)
        revisions = TaskStore().revisions((task.task_id, task.task_revision) for task in rows)
        fields = ("status", "image_ref", "image_digest", "build_error", "tarball_object_key")
        return [
            {
                "position": task.position,
                "slug": task.slug,
                "task_id": task.task_id,
                "task_revision": task.task_revision,
                "pack_path": task.pack_path,
                "pack_sha256": task.pack_sha256,
                "requested_image_ref": task.image_ref,
                "requested_image_digest": task.image_digest,
                "requested_image_metadata": task.image_metadata,
                **{field: revision[field] for field in fields},
            }
            for task in rows
            if (revision := revisions.get((task.task_id, task.task_revision)))
        ]

    def benchmarks(self, import_id: str) -> list[dict[str, Any]]:
        entity = self._get(import_id)
        if entity is None:
            return []
        return [benchmark.model_dump() for benchmark in sorted(entity.benchmarks, key=lambda item: item.position)]

    def set_task_revision(self, import_id: str, slug: str, *, task_revision: int) -> None:
        def change(entity: ScaledEvalsBenchmarkImport) -> tuple[None, bool]:
            for task in entity.tasks:
                if task.slug == slug:
                    task.task_revision = task_revision
            return None, True

        self._mutate(import_id, change)

    def set_benchmark_revision(
        self,
        import_id: str,
        slug: str,
        *,
        benchmark_id: str,
        benchmark_revision: int,
    ) -> None:
        def change(entity: ScaledEvalsBenchmarkImport) -> tuple[None, bool]:
            for benchmark in entity.benchmarks:
                if benchmark.slug == slug:
                    benchmark.benchmark_id, benchmark.benchmark_revision = benchmark_id, benchmark_revision
            return None, True

        self._mutate(import_id, change)
