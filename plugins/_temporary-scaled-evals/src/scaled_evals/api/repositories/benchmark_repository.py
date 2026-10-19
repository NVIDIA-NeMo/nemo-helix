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
from psycopg.types.json import Jsonb

from scaled_evals.api.repositories.base_repository import (
    Conflict,
    InvalidReference,
    created_at_cursor_clause,
    join_where,
    normalize_order,
    order_by_clause,
    patch_set_clause,
    substring_search_pattern,
)

BENCHMARK_COLUMNS = (
    "id, owner_id, name, slug, description, visibility, qualification_status, "
    "qualification_evidence, qualified_at, qualified_by, current_revision, created_at, updated_at"
)
_PATCHABLE_COLUMNS = frozenset({"name", "slug", "description", "visibility"})


@dataclass(frozen=True, slots=True)
class TaskRef:
    task_id: str
    task_revision: int | None = None


class BenchmarkRepository:
    def __init__(self, conn: psycopg.Connection) -> None:
        self.conn = conn

    def _insert_members(
        self, cur: psycopg.Cursor, benchmark_id: str, revision: int, tasks: builtins.list[TaskRef]
    ) -> None:
        """Validate and insert the member tasks of one benchmark revision.

        Each task must be a live task; a pinned ``task_revision`` must exist.
        Duplicates within a revision are rejected. Raises ``InvalidReference``
        (router → 422) on any bad/duplicate member.
        """
        seen: set[str] = set()
        for position, ref in enumerate(tasks):
            if ref.task_id in seen:
                raise InvalidReference(f"duplicate task in benchmark: {ref.task_id}")
            seen.add(ref.task_id)
            cur.execute(
                "SELECT 1 FROM tasks WHERE id = %s AND deleted_at IS NULL",
                (ref.task_id,),
            )
            if cur.fetchone() is None:
                raise InvalidReference(f"task not found: {ref.task_id}")
            if ref.task_revision is not None:
                cur.execute(
                    "SELECT 1 FROM task_revisions WHERE task_id = %s AND revision = %s",
                    (ref.task_id, ref.task_revision),
                )
                if cur.fetchone() is None:
                    raise InvalidReference(f"task revision not found: {ref.task_id} rev {ref.task_revision}")
            cur.execute(
                """
                INSERT INTO benchmark_revision_tasks
                    (benchmark_id, revision, task_id, task_revision, position)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (benchmark_id, revision, ref.task_id, ref.task_revision, position),
            )

    def create_with_initial_revision(
        self,
        benchmark_id: str,
        *,
        name: str,
        slug: str,
        description: str | None,
        visibility: str,
        tasks: builtins.list[TaskRef],
    ) -> dict:
        try:
            with self.conn.transaction(), self.conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO benchmarks
                        (id, name, slug, description, visibility, current_revision)
                    VALUES (%s, %s, %s, %s, %s, 1)
                    RETURNING {BENCHMARK_COLUMNS}
                    """,
                    (benchmark_id, name, slug, description, visibility),
                )
                row = cur.fetchone()
                cur.execute(
                    """
                    INSERT INTO benchmark_revisions (benchmark_id, revision, description)
                    VALUES (%s, 1, %s)
                    """,
                    (benchmark_id, description),
                )
                self._insert_members(cur, benchmark_id, 1, tasks)
        except UniqueViolation as exc:
            raise Conflict("slug_conflict", f"slug already exists: {slug}") from exc
        return row

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
    ) -> dict | None:
        """Derive a new benchmark that copies base member pins + stores policy.

        Returns the new benchmark row plus ``derived_from`` / ``operational_policy``
        / ``revision``. ``None`` if the base benchmark (or requested revision) is
        missing.
        """
        try:
            with self.conn.transaction(), self.conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT current_revision
                    FROM benchmarks
                    WHERE id = %s AND deleted_at IS NULL
                    FOR SHARE
                    """,
                    (base_benchmark_id,),
                )
                base = cur.fetchone()
                if base is None:
                    return None
                source_revision = from_revision or base["current_revision"]
                if source_revision is None:
                    raise InvalidReference("base benchmark has no revision to derive from")
                cur.execute(
                    """
                    SELECT 1 FROM benchmark_revisions
                    WHERE benchmark_id = %s AND revision = %s
                    """,
                    (base_benchmark_id, source_revision),
                )
                if cur.fetchone() is None:
                    raise InvalidReference(f"benchmark revision not found: {base_benchmark_id} rev {source_revision}")
                cur.execute(
                    """
                    SELECT task_id, task_revision, position
                    FROM benchmark_revision_tasks
                    WHERE benchmark_id = %s AND revision = %s
                    ORDER BY position, task_id
                    """,
                    (base_benchmark_id, source_revision),
                )
                members = cur.fetchall()
                tasks = [TaskRef(task_id=row["task_id"], task_revision=row["task_revision"]) for row in members]
                cur.execute(
                    f"""
                    INSERT INTO benchmarks
                        (id, name, slug, description, visibility, current_revision)
                    VALUES (%s, %s, %s, %s, %s, 1)
                    RETURNING {BENCHMARK_COLUMNS}
                    """,
                    (variant_id, name, slug, description, visibility),
                )
                row = cur.fetchone()
                cur.execute(
                    """
                    INSERT INTO benchmark_revisions (
                        benchmark_id, revision, description,
                        derived_from_benchmark_id, derived_from_revision,
                        operational_policy
                    )
                    VALUES (%s, 1, %s, %s, %s, %s)
                    """,
                    (
                        variant_id,
                        description,
                        base_benchmark_id,
                        source_revision,
                        Jsonb(operational_policy),
                    ),
                )
                self._insert_members(cur, variant_id, 1, tasks)
        except UniqueViolation as exc:
            raise Conflict("slug_conflict", f"slug already exists: {slug}") from exc
        return {
            **row,
            "revision": 1,
            "derived_from": {
                "benchmark_id": base_benchmark_id,
                "revision": source_revision,
            },
            "operational_policy": operational_policy,
        }

    def get_revision_meta(self, benchmark_id: str, revision: int) -> dict | None:
        """Return lineage + policy for one benchmark revision."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT derived_from_benchmark_id, derived_from_revision, operational_policy
                FROM benchmark_revisions
                WHERE benchmark_id = %s AND revision = %s
                """,
                (benchmark_id, revision),
            )
            return cur.fetchone()

    def create_next_revision(
        self, benchmark_id: str, *, description: str | None, tasks: builtins.list[TaskRef]
    ) -> int | None:
        new_rev: int | None = None
        with self.conn.transaction(), self.conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM benchmarks WHERE id = %s AND deleted_at IS NULL FOR UPDATE",
                (benchmark_id,),
            )
            if cur.fetchone() is None:
                return None
            cur.execute(
                "SELECT MAX(revision) AS max_rev FROM benchmark_revisions WHERE benchmark_id = %s",
                (benchmark_id,),
            )
            new_rev = (cur.fetchone()["max_rev"] or 0) + 1
            cur.execute(
                """
                INSERT INTO benchmark_revisions (benchmark_id, revision, description)
                VALUES (%s, %s, %s)
                """,
                (benchmark_id, new_rev, description),
            )
            self._insert_members(cur, benchmark_id, new_rev, tasks)
            cur.execute(
                "UPDATE benchmarks SET current_revision = %s, updated_at = NOW() WHERE id = %s",
                (new_rev, benchmark_id),
            )
        return new_rev

    def list(self, *, limit: int, cursor: str | None, order: str, q: str | None = None) -> builtins.list[dict]:
        direction = normalize_order(order)
        ordering = order_by_clause(("created_at", "id"), direction)
        filters = ["deleted_at IS NULL"]
        params: list[Any] = []
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
                SELECT {BENCHMARK_COLUMNS}
                FROM benchmarks
                WHERE {join_where(filters)}
                ORDER BY {ordering}
                LIMIT %s
                """,
                params,
            )
            return cur.fetchall()

    def get(self, benchmark_id: str) -> dict | None:
        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT {BENCHMARK_COLUMNS}
                FROM benchmarks
                WHERE id = %s AND deleted_at IS NULL
                """,
                (benchmark_id,),
            )
            return cur.fetchone()

    def get_by_slug(self, slug: str) -> dict | None:
        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT {BENCHMARK_COLUMNS}
                FROM benchmarks
                WHERE slug = %s AND deleted_at IS NULL
                """,
                (slug,),
            )
            return cur.fetchone()

    def list_tasks(
        self,
        benchmark_id: str,
        revision: int,
        *,
        limit: int | None = None,
        cursor: int | None = None,
    ) -> builtins.list[dict]:
        # Join tasks so members carry their human-readable slug/name, not just
        # the opaque task_id (used by the benchmark detail + /tasks listing).
        position_filter = "AND m.position >= %s" if cursor is not None else ""
        pagination = "" if limit is None else "LIMIT %s"
        params: list[Any] = [benchmark_id, revision]
        if cursor is not None:
            params.append(cursor)
        if limit is not None:
            params.append(limit + 1)
        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT m.task_id, m.task_revision, m.position,
                       t.slug AS task_slug, t.name AS task_name
                FROM benchmark_revision_tasks m
                JOIN tasks t ON t.id = m.task_id
                WHERE m.benchmark_id = %s AND m.revision = %s
                  {position_filter}
                ORDER BY m.position, m.task_id
                {pagination}
                """,
                params,
            )
            return cur.fetchall()

    def list_resolved_tasks(self, benchmark_id: str, revision: int) -> builtins.list[dict]:
        """List members with their effective immutable task revision and build state."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT m.task_id, m.task_revision, m.position,
                       t.slug AS task_slug, t.name AS task_name,
                       COALESCE(m.task_revision, t.current_revision) AS resolved_task_revision,
                       r.status, r.image_ref, r.image_digest,
                       r.tarball_object_key, r.tarball_size_bytes
                FROM benchmark_revision_tasks m
                JOIN tasks t ON t.id = m.task_id AND t.deleted_at IS NULL
                LEFT JOIN task_revisions r
                  ON r.task_id = m.task_id
                 AND r.revision = COALESCE(m.task_revision, t.current_revision)
                WHERE m.benchmark_id = %s AND m.revision = %s
                ORDER BY m.position, m.task_id
                """,
                (benchmark_id, revision),
            )
            return cur.fetchall()

    def get_detail(self, benchmark_id: str) -> dict | None:
        row = self.get(benchmark_id)
        if row is None:
            return None
        revision = row["current_revision"]
        tasks = self.list_tasks(benchmark_id, revision) if revision is not None else []
        derived_from = None
        operational_policy: dict[str, Any] = {}
        if revision is not None:
            meta = self.get_revision_meta(benchmark_id, revision)
            if meta is not None:
                policy = meta.get("operational_policy")
                operational_policy = dict(policy) if isinstance(policy, dict) else {}
                if meta.get("derived_from_benchmark_id") is not None:
                    derived_from = {
                        "benchmark_id": meta["derived_from_benchmark_id"],
                        "revision": meta["derived_from_revision"],
                    }
        return {
            **row,
            "revision": revision,
            "tasks": tasks,
            "derived_from": derived_from,
            "operational_policy": operational_policy,
        }

    def update(
        self,
        benchmark_id: str,
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
                    SELECT {BENCHMARK_COLUMNS}
                    FROM benchmarks
                    WHERE id = %s AND deleted_at IS NULL
                    """,
                    (benchmark_id,),
                )
            else:
                sets.append("updated_at = NOW()")
                params.append(benchmark_id)
                try:
                    cur.execute(
                        f"""
                        UPDATE benchmarks
                        SET {", ".join(sets)}
                        WHERE id = %s AND deleted_at IS NULL
                        RETURNING {BENCHMARK_COLUMNS}
                        """,
                        params,
                    )
                except UniqueViolation as exc:
                    raise Conflict("slug_conflict", f"slug already exists: {slug}") from exc
            return cur.fetchone()

    def soft_delete(self, benchmark_id: str) -> bool:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE benchmarks
                SET deleted_at = NOW()
                WHERE id = %s AND deleted_at IS NULL
                RETURNING id
                """,
                (benchmark_id,),
            )
            return cur.fetchone() is not None

    def set_qualification(
        self,
        benchmark_id: str,
        *,
        status: str,
        evidence: dict[str, Any],
        qualified_by: str,
    ) -> dict | None:
        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE benchmarks
                SET qualification_status = %s,
                    qualification_evidence = %s,
                    qualified_at = NOW(),
                    qualified_by = %s,
                    visibility = CASE WHEN %s = 'rejected' THEN 'private'::benchmark_visibility
                                      ELSE visibility END,
                    updated_at = NOW()
                WHERE id = %s AND deleted_at IS NULL
                RETURNING {BENCHMARK_COLUMNS}
                """,
                (status, Jsonb(evidence), qualified_by, status, benchmark_id),
            )
            return cur.fetchone()

    def promote(self, benchmark_id: str, *, qualified_by: str) -> dict | None:
        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE benchmarks
                SET visibility = 'public', qualified_by = %s, updated_at = NOW()
                WHERE id = %s
                  AND deleted_at IS NULL
                  AND qualification_status = 'qualified'
                RETURNING {BENCHMARK_COLUMNS}
                """,
                (qualified_by, benchmark_id),
            )
            return cur.fetchone()
