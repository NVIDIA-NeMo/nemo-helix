# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import psycopg
from psycopg.types.json import Jsonb


class BenchmarkImportRepository:
    def __init__(self, conn: psycopg.Connection) -> None:
        self.conn = conn

    def lock_identity(self, owner_id: str, manifest_sha256: str, visibility: str) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT pg_advisory_xact_lock(hashtext(%s), hashtext(%s))",
                (owner_id, f"{manifest_sha256}:{visibility}"),
            )

    def create(
        self,
        import_id: str,
        *,
        owner_id: str,
        manifest_sha256: str,
        manifest: dict[str, Any],
        visibility: str,
        description: str | None,
    ) -> dict[str, Any]:
        with self.conn.cursor() as cur:
            cur.execute("INSERT INTO users (id) VALUES (%s) ON CONFLICT (id) DO NOTHING", (owner_id,))
            cur.execute(
                """
                INSERT INTO benchmark_imports
                    (id, owner_id, manifest_sha256, manifest, visibility, description)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id, owner_id, manifest_sha256, visibility, description,
                          created_at, updated_at
                """,
                (import_id, owner_id, manifest_sha256, Jsonb(manifest), visibility, description),
            )
            return cur.fetchone()

    def get_by_identity(self, *, owner_id: str, manifest_sha256: str, visibility: str) -> dict[str, Any] | None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, owner_id, manifest_sha256, manifest, visibility, description,
                       created_at, updated_at
                FROM benchmark_imports
                WHERE owner_id = %s AND manifest_sha256 = %s AND visibility = %s
                """,
                (owner_id, manifest_sha256, visibility),
            )
            return cur.fetchone()

    def get(self, import_id: str, *, owner_id: str) -> dict[str, Any] | None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, owner_id, manifest_sha256, manifest, visibility, description,
                       created_at, updated_at
                FROM benchmark_imports
                WHERE id = %s AND owner_id = %s
                """,
                (import_id, owner_id),
            )
            return cur.fetchone()

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
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO benchmark_import_tasks
                    (import_id, position, slug, task_id, task_revision, pack_path,
                     pack_sha256, image_ref, image_digest, image_metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    import_id,
                    position,
                    slug,
                    task_id,
                    task_revision,
                    pack_path,
                    pack_sha256,
                    image_ref,
                    image_digest,
                    Jsonb(image_metadata or {}),
                ),
            )

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
        with self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE benchmark_import_tasks i
                SET image_ref = %s, image_digest = %s, image_metadata = %s
                FROM task_revisions r
                WHERE i.import_id = %s AND i.slug = %s
                  AND r.task_id = i.task_id AND r.revision = i.task_revision
                  AND r.status = 'uploading'
                  AND (i.image_ref IS NULL OR (i.image_ref = %s AND i.image_digest = %s))
                RETURNING i.slug
                """,
                (
                    image_ref,
                    image_digest,
                    Jsonb(image_metadata),
                    import_id,
                    slug,
                    image_ref,
                    image_digest,
                ),
            )
            return cur.fetchone() is not None

    def add_benchmark(
        self,
        import_id: str,
        *,
        position: int,
        slug: str,
        name: str,
        task_slugs: list[str],
    ) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO benchmark_import_benchmarks
                    (import_id, position, slug, name, task_slugs)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (import_id, position, slug, name, Jsonb(task_slugs)),
            )

    def tasks(self, import_id: str) -> list[dict[str, Any]]:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT i.position, i.slug, i.task_id, i.task_revision, i.pack_path,
                       i.pack_sha256, i.image_ref AS requested_image_ref,
                       i.image_digest AS requested_image_digest,
                       i.image_metadata AS requested_image_metadata,
                       r.status, r.image_ref, r.image_digest, r.build_error,
                       r.tarball_object_key
                FROM benchmark_import_tasks i
                JOIN task_revisions r
                  ON r.task_id = i.task_id AND r.revision = i.task_revision
                WHERE i.import_id = %s
                ORDER BY i.position
                """,
                (import_id,),
            )
            return cur.fetchall()

    def lock_benchmark_slug(self, slug: str) -> None:
        with self.conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"benchmark:{slug}",))

    def benchmarks(self, import_id: str) -> list[dict[str, Any]]:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT position, slug, name, task_slugs, benchmark_id, benchmark_revision
                FROM benchmark_import_benchmarks
                WHERE import_id = %s
                ORDER BY position
                """,
                (import_id,),
            )
            return cur.fetchall()

    def set_task_revision(self, import_id: str, slug: str, *, task_revision: int) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE benchmark_import_tasks
                SET task_revision = %s
                WHERE import_id = %s AND slug = %s
                """,
                (task_revision, import_id, slug),
            )

    def set_benchmark_revision(
        self,
        import_id: str,
        slug: str,
        *,
        benchmark_id: str,
        benchmark_revision: int,
    ) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE benchmark_import_benchmarks
                SET benchmark_id = %s, benchmark_revision = %s
                WHERE import_id = %s AND slug = %s
                """,
                (benchmark_id, benchmark_revision, import_id, slug),
            )
            cur.execute(
                "UPDATE benchmark_imports SET updated_at = NOW() WHERE id = %s",
                (import_id,),
            )
