# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

# The repository below defines a ``list`` method, which shadows the builtin for
# annotations in the same class body, so those spell the type ``builtins.list``.
import builtins
from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg.errors import UniqueViolation
from psycopg.types.json import Json

from scaled_evals.api.repositories.base_repository import (
    Conflict,
    created_at_cursor_clause,
    join_where,
    normalize_order,
    order_by_clause,
    patch_set_clause,
    substring_search_pattern,
)

TASK_COLUMNS = "id, owner_id, name, slug, description, visibility, current_revision, created_at, updated_at"
_PATCHABLE_COLUMNS = frozenset({"name", "slug", "description", "visibility"})
_TASK_PACK_QUOTA_LOCK_NAMESPACE = 1936024439
_UNOWNED_QUOTA_LOCK_KEY = "__scaled_evals_unowned_tasks__"


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


class TaskRepository:
    def __init__(self, conn: psycopg.Connection) -> None:
        self.conn = conn

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
    ) -> dict:
        try:
            with self.conn.transaction(), self.conn.cursor() as cur:
                if owner_id is not None:
                    cur.execute(
                        "INSERT INTO users (id) VALUES (%s) ON CONFLICT (id) DO NOTHING",
                        (owner_id,),
                    )
                if owner_id is None:
                    cur.execute(
                        f"""
                        INSERT INTO tasks (id, name, slug, description, visibility)
                        VALUES (%s, %s, %s, %s, %s)
                        RETURNING {TASK_COLUMNS}
                        """,
                        (task_id, name, slug, description, visibility),
                    )
                else:
                    cur.execute(
                        f"""
                        INSERT INTO tasks (id, owner_id, name, slug, description, visibility)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        RETURNING {TASK_COLUMNS}
                        """,
                        (task_id, owner_id, name, slug, description, visibility),
                    )
                row = cur.fetchone()
                cur.execute(
                    """
                    INSERT INTO task_revisions
                        (task_id, revision, status, tarball_object_key)
                    VALUES (%s, 1, 'uploading', %s)
                    """,
                    (task_id, object_key),
                )
        except UniqueViolation as exc:
            raise Conflict("slug_conflict", f"slug already exists: {slug}") from exc
        return row

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
        revision: int | None = None
        object_key = ""
        status = ""
        with self.conn.transaction(), self.conn.cursor() as cur:
            cur.execute(
                "SELECT id, owner_id FROM tasks WHERE id = %s AND deleted_at IS NULL FOR UPDATE",
                (task_id,),
            )
            task_row = cur.fetchone()
            quota_exceeded = False
            previous_storage_bytes = 0
            uploaded_bytes = tarball_size_bytes or 0
            if task_row is not None:
                revision_filter = "AND revision = %s" if exact_revision and expected_revision is not None else ""
                params: tuple[object, ...] = (
                    (task_id, expected_revision) if exact_revision and expected_revision is not None else (task_id,)
                )
                cur.execute(
                    f"""
                    SELECT revision, status, tarball_object_key
                    FROM task_revisions
                    WHERE task_id = %s
                    {revision_filter}
                    ORDER BY revision DESC
                    LIMIT 1
                    FOR UPDATE
                    """,
                    params,
                )
                rev_row = cur.fetchone()
                if rev_row is not None:
                    revision = rev_row["revision"]
                    status = rev_row["status"]
                    object_key = rev_row["tarball_object_key"]
                    if expected_revision is not None and revision != expected_revision:
                        return FinalizeRevision(
                            revision=revision,
                            status=status,
                            object_key=object_key,
                        )
                    if status == "uploading":
                        if tarball_size_bytes is not None and tenant_storage_quota_bytes is not None:
                            owner_id = task_row.get("owner_id")
                            self._lock_owner_task_pack_quota(cur, owner_id)
                            previous_storage_bytes = self._owner_task_pack_storage_bytes(
                                cur, owner_id, task_id, revision
                            )
                            if previous_storage_bytes + tarball_size_bytes > tenant_storage_quota_bytes:
                                quota_exceeded = True
                                return FinalizeRevision(
                                    revision=revision,
                                    status=status,
                                    object_key=object_key,
                                    quota_exceeded=quota_exceeded,
                                    previous_storage_bytes=previous_storage_bytes,
                                    uploaded_bytes=uploaded_bytes,
                                )
                            cur.execute(
                                """
                                UPDATE task_revisions
                                SET tarball_size_bytes = %s
                                WHERE task_id = %s AND revision = %s AND status = 'uploading'
                                """,
                                (tarball_size_bytes, task_id, revision),
                            )
                        cur.execute(
                            """
                            UPDATE task_revisions
                            SET status = 'building',
                                build_started_at = NOW(),
                                build_backend = %s,
                                build_payload = %s,
                                build_credentials = %s,
                                tarball_sha256 = COALESCE(%s, tarball_sha256),
                                build_error = NULL,
                                build_completed_at = NULL,
                                build_claimed_at = NULL,
                                build_first_claimed_at = NULL,
                                build_claimed_by = NULL,
                                build_attempts = 0,
                                build_next_attempt_at = NOW()
                            WHERE task_id = %s AND revision = %s
                            """,
                            (
                                build_backend,
                                Json(build_payload or {}),
                                Json(build_credentials or {}),
                                tarball_sha256,
                                task_id,
                                revision,
                            ),
                        )
        if revision is None:
            return None
        return FinalizeRevision(
            revision=revision,
            status=status,
            object_key=object_key,
            quota_exceeded=quota_exceeded,
            previous_storage_bytes=previous_storage_bytes,
            uploaded_bytes=uploaded_bytes,
        )

    def revision_for_finalize(self, task_id: str, *, expected_revision: int | None = None) -> UploadingRevision | None:
        """Return exact or latest revision metadata for object-store finalize guards."""
        with self.conn.cursor() as cur:
            revision_filter = "AND revision = %s" if expected_revision is not None else ""
            params: tuple[object, ...] = (expected_revision, task_id) if expected_revision is not None else (task_id,)
            cur.execute(
                f"""
                SELECT r.revision, r.status, r.tarball_object_key
                FROM tasks t
                JOIN LATERAL (
                    SELECT revision, status, tarball_object_key
                    FROM task_revisions
                    WHERE task_id = t.id
                    {revision_filter}
                    ORDER BY revision DESC
                    LIMIT 1
                ) r ON TRUE
                WHERE t.id = %s AND t.deleted_at IS NULL
                """,
                params,
            )
            row = cur.fetchone()
        if row is None:
            return None
        return UploadingRevision(
            revision=row["revision"],
            status=row["status"],
            object_key=row["tarball_object_key"],
        )

    def latest_revision_for_finalize(self, task_id: str) -> UploadingRevision | None:
        """Backward-compatible latest revision lookup."""
        return self.revision_for_finalize(task_id)

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
        """Atomically finalize the latest uploading revision with a prebuilt image.

        ``verifier_image_ref``/``verifier_image_digest`` name the separate verifier image, for tasks
        whose verifier runs in its own sandbox; leave them unset otherwise.
        """
        result: PrebuiltFinalizeResult | None = None
        with self.conn.transaction(), self.conn.cursor() as cur:
            cur.execute(
                "SELECT id, owner_id FROM tasks WHERE id = %s AND deleted_at IS NULL FOR UPDATE",
                (task_id,),
            )
            task_row = cur.fetchone()
            if task_row is None:
                return None
            revision_filter = "AND revision = %s" if exact_revision and expected_revision is not None else ""
            params: tuple[object, ...] = (
                (task_id, expected_revision) if exact_revision and expected_revision is not None else (task_id,)
            )
            cur.execute(
                f"""
                SELECT revision, status, tarball_object_key
                FROM task_revisions
                WHERE task_id = %s
                {revision_filter}
                ORDER BY revision DESC
                LIMIT 1
                FOR UPDATE
                """,
                params,
            )
            row = cur.fetchone()
            if row is None:
                return None
            if expected_revision is not None and row["revision"] != expected_revision:
                return PrebuiltFinalizeResult(
                    revision=row["revision"],
                    status=row["status"],
                    object_key=row["tarball_object_key"],
                    finalized=False,
                )
            finalized = row["status"] == "uploading"
            previous_storage_bytes = 0
            uploaded_bytes = tarball_size_bytes or 0
            if finalized:
                if tarball_size_bytes is not None and tenant_storage_quota_bytes is not None:
                    owner_id = task_row.get("owner_id")
                    self._lock_owner_task_pack_quota(cur, owner_id)
                    previous_storage_bytes = self._owner_task_pack_storage_bytes(
                        cur, owner_id, task_id, row["revision"]
                    )
                    if previous_storage_bytes + tarball_size_bytes > tenant_storage_quota_bytes:
                        return PrebuiltFinalizeResult(
                            revision=row["revision"],
                            status=row["status"],
                            object_key=row["tarball_object_key"],
                            finalized=False,
                            quota_exceeded=True,
                            previous_storage_bytes=previous_storage_bytes,
                            uploaded_bytes=uploaded_bytes,
                        )
                    cur.execute(
                        """
                        UPDATE task_revisions
                        SET tarball_size_bytes = %s
                        WHERE task_id = %s AND revision = %s AND status = 'uploading'
                        """,
                        (tarball_size_bytes, task_id, row["revision"]),
                    )
                cur.execute(
                    """
                    UPDATE task_revisions
                    SET status = 'ready', image_ref = %s, image_digest = %s,
                        verifier_image_ref = %s, verifier_image_digest = %s,
                        tarball_sha256 = COALESCE(%s, tarball_sha256), build_backend = NULL,
                        build_payload = '{}'::jsonb, build_credentials = '{}'::jsonb,
                        build_error = NULL, build_completed_at = NOW(),
                        build_claimed_at = NULL, build_claimed_by = NULL,
                        build_next_attempt_at = NULL
                    WHERE task_id = %s AND revision = %s AND status = 'uploading'
                    """,
                    (
                        image_ref,
                        image_digest,
                        verifier_image_ref,
                        verifier_image_digest,
                        tarball_sha256,
                        task_id,
                        row["revision"],
                    ),
                )
                finalized = cur.rowcount == 1
                if finalized:
                    cur.execute(
                        """
                        UPDATE tasks SET current_revision = %s, updated_at = NOW()
                        WHERE id = %s
                          AND (current_revision IS NULL OR current_revision <= %s)
                        """,
                        (row["revision"], task_id, row["revision"]),
                    )
            result = PrebuiltFinalizeResult(
                revision=row["revision"],
                status="ready" if finalized else row["status"],
                object_key=row["tarball_object_key"],
                finalized=finalized,
                previous_storage_bytes=previous_storage_bytes,
                uploaded_bytes=uploaded_bytes,
            )
        return result

    def _lock_owner_task_pack_quota(self, cur: psycopg.Cursor, owner_id: str | None) -> None:
        cur.execute(
            "SELECT pg_advisory_xact_lock(%s, hashtext(%s))",
            (_TASK_PACK_QUOTA_LOCK_NAMESPACE, owner_id or _UNOWNED_QUOTA_LOCK_KEY),
        )

    def _owner_task_pack_storage_bytes(
        self,
        cur: psycopg.Cursor,
        owner_id: str | None,
        task_id: str,
        revision: int,
    ) -> int:
        cur.execute(
            """
            SELECT COALESCE(SUM(r2.tarball_size_bytes), 0) AS previous_storage_bytes
            FROM tasks t2
            JOIN task_revisions r2 ON r2.task_id = t2.id
            WHERE t2.deleted_at IS NULL
              AND t2.owner_id IS NOT DISTINCT FROM %s
              AND NOT (r2.task_id = %s AND r2.revision = %s)
            """,
            (owner_id, task_id, revision),
        )
        row = cur.fetchone()
        return int(row["previous_storage_bytes"] if row else 0)

    def create_next_revision(self, task_id: str) -> NewRevision | None:
        new_rev: int | None = None
        object_key = ""
        with self.conn.transaction(), self.conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM tasks WHERE id = %s AND deleted_at IS NULL FOR UPDATE",
                (task_id,),
            )
            if cur.fetchone() is not None:
                cur.execute(
                    """
                    SELECT MAX(revision) AS max_rev
                    FROM task_revisions
                    WHERE task_id = %s
                    """,
                    (task_id,),
                )
                new_rev = (cur.fetchone()["max_rev"] or 0) + 1
                object_key = f"{task_id}/rev/{new_rev}/tarball.tar.gz"
                cur.execute(
                    """
                    INSERT INTO task_revisions
                        (task_id, revision, status, tarball_object_key)
                    VALUES (%s, %s, 'uploading', %s)
                    """,
                    (task_id, new_rev, object_key),
                )
                cur.execute(
                    "UPDATE tasks SET current_revision = %s, updated_at = NOW() WHERE id = %s",
                    (new_rev, task_id),
                )
        if new_rev is None:
            return None
        return NewRevision(revision=new_rev, object_key=object_key)

    def task_pack_revisions_for_owner(
        self,
        *,
        owner_id: str,
        statuses: tuple[str, ...] = ("ready", "building"),
        limit: int = 500,
    ) -> builtins.list[TaskPackRevision]:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT r.task_id, r.revision, r.status, r.tarball_object_key
                FROM task_revisions r
                JOIN tasks t ON t.id = r.task_id
                WHERE t.deleted_at IS NULL
                  AND t.owner_id = %s
                  AND r.status = ANY(%s)
                ORDER BY t.updated_at DESC, r.task_id, r.revision DESC
                LIMIT %s
                """,
                (owner_id, list(statuses), limit),
            )
            rows = cur.fetchall()
        return [
            TaskPackRevision(
                task_id=row["task_id"],
                revision=row["revision"],
                status=row["status"],
                object_key=row["tarball_object_key"],
            )
            for row in rows
        ]

    def mark_task_pack_missing(
        self,
        *,
        owner_id: str,
        task_id: str,
        revision: int,
        object_key: str,
        previous_status: str,
    ) -> bool:
        build_error = f"task_object_missing: task pack object is missing: {object_key}"
        with self.conn.transaction(), self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE task_revisions r
                SET status = 'failed',
                    build_error = %s,
                    build_completed_at = NOW(),
                    build_claimed_at = NULL,
                    build_claimed_by = NULL,
                    build_next_attempt_at = NULL
                FROM tasks t
                WHERE t.id = r.task_id
                  AND t.deleted_at IS NULL
                  AND t.owner_id = %s
                  AND r.task_id = %s
                  AND r.revision = %s
                  AND r.status = %s
                  AND r.tarball_object_key = %s
                """,
                (build_error, owner_id, task_id, revision, previous_status, object_key),
            )
            marked = cur.rowcount == 1
            if marked:
                cur.execute(
                    """
                    UPDATE tasks t
                    SET current_revision = latest.revision,
                        updated_at = NOW()
                    FROM LATERAL (
                        SELECT MAX(r2.revision) AS revision
                        FROM task_revisions r2
                        WHERE r2.task_id = t.id
                          AND r2.status = 'ready'
                    ) latest
                    WHERE t.id = %s
                      AND t.owner_id = %s
                      AND t.current_revision = %s
                      AND latest.revision IS NOT NULL
                    """,
                    (task_id, owner_id, revision),
                )
        return marked

    def list(
        self,
        *,
        limit: int,
        cursor: str | None,
        order: str,
        owner_id: str | None = None,
        q: str | None = None,
    ) -> builtins.list[dict]:
        direction = normalize_order(order)
        ordering = order_by_clause(("created_at", "id"), direction)
        filters = ["deleted_at IS NULL"]
        params: list[Any] = []
        if owner_id is not None:
            filters.append("owner_id = %s")
            params.append(owner_id)
        if search := substring_search_pattern(q):
            filters.append(
                "(id ILIKE %s ESCAPE '\\' OR name ILIKE %s ESCAPE '\\' "
                "OR slug ILIKE %s ESCAPE '\\' "
                "OR COALESCE(description, '') ILIKE %s ESCAPE '\\')"
            )
            params.extend([search] * 4)
        cursor_filter, cursor_params = created_at_cursor_clause(cursor, direction)
        if cursor_filter:
            filters.append(cursor_filter)
            params.extend(cursor_params)
        params.append(limit + 1)
        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT {TASK_COLUMNS}
                FROM tasks
                WHERE {join_where(filters)}
                ORDER BY {ordering}
                LIMIT %s
                """,
                params,
            )
            return cur.fetchall()

    def get(self, task_id: str) -> dict | None:
        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT {TASK_COLUMNS}
                FROM tasks
                WHERE id = %s AND deleted_at IS NULL
                """,
                (task_id,),
            )
            return cur.fetchone()

    def get_by_slug(self, slug: str) -> dict | None:
        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT {TASK_COLUMNS}
                FROM tasks
                WHERE slug = %s AND deleted_at IS NULL
                """,
                (slug,),
            )
            return cur.fetchone()

    def get_detail(self, task_id: str) -> dict | None:
        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT {", ".join(f"b.{c}" for c in TASK_COLUMNS.split(", "))},
                       r.revision, r.status, r.image_ref, r.image_digest,
                       r.verifier_image_ref, r.verifier_image_digest, r.build_error,
                       r.tarball_size_bytes, r.tarball_sha256, r.tarball_object_key,
                       r.created_at AS revision_created_at
                FROM tasks b
                LEFT JOIN LATERAL (
                    SELECT revision, status, image_ref, image_digest,
                           verifier_image_ref, verifier_image_digest, build_error,
                           tarball_size_bytes, tarball_sha256, tarball_object_key, created_at
                    FROM task_revisions
                    WHERE task_id = b.id
                    ORDER BY revision DESC
                    LIMIT 1
                ) r ON TRUE
                WHERE b.id = %s AND b.deleted_at IS NULL
                """,
                (task_id,),
            )
            return cur.fetchone()

    def count_failed_revisions(self, task_id: str, tarball_sha256: str) -> int:
        """Count terminal build failures for one immutable package digest."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*) AS count
                FROM task_revisions
                WHERE task_id = %s AND tarball_sha256 = %s AND status = 'failed'
                """,
                (task_id, tarball_sha256),
            )
            row = cur.fetchone()
        return int(row["count"] if row else 0)

    def update(
        self,
        task_id: str,
        *,
        name: str | None = None,
        slug: str | None = None,
        description: str | None = None,
        visibility: str | None = None,
    ) -> dict | None:
        sets, params = patch_set_clause(
            (
                ("name", name),
                ("slug", slug),
                ("description", description),
                ("visibility", visibility),
            ),
            _PATCHABLE_COLUMNS,
        )
        with self.conn.cursor() as cur:
            if not sets:
                cur.execute(
                    f"""
                    SELECT {TASK_COLUMNS}
                    FROM tasks
                    WHERE id = %s AND deleted_at IS NULL
                    """,
                    (task_id,),
                )
            else:
                sets.append("updated_at = NOW()")
                params.append(task_id)
                try:
                    cur.execute(
                        f"""
                        UPDATE tasks
                        SET {", ".join(sets)}
                        WHERE id = %s AND deleted_at IS NULL
                        RETURNING {TASK_COLUMNS}
                        """,
                        params,
                    )
                except UniqueViolation as exc:
                    raise Conflict("slug_conflict", f"slug already exists: {slug}") from exc
            return cur.fetchone()

    def active_evaluation_reference_exists(self, task_id: str) -> bool:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT id
                FROM evaluations
                WHERE deleted_at IS NULL
                  AND status NOT IN ('succeeded', 'failed', 'cancelled')
                  AND task_id = %s
                LIMIT 1
                """,
                (task_id,),
            )
            return cur.fetchone() is not None

    def soft_delete(self, task_id: str) -> bool:
        if self.active_evaluation_reference_exists(task_id):
            raise Conflict(
                "task_in_use",
                "task is referenced by an active evaluation",
            )
        with self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE tasks
                SET deleted_at = NOW()
                WHERE id = %s AND deleted_at IS NULL
                RETURNING id
                """,
                (task_id,),
            )
            return cur.fetchone() is not None
