# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

# The repository below defines a ``list`` method, which shadows the builtin for
# annotations in the same class body, so those spell the type ``builtins.list``.
import builtins
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from scaled_evals.api.repositories.base_repository import (
    Conflict,
    created_at_cursor_clause,
    join_where,
    normalize_order,
    order_by_clause,
    patch_set_clause,
    substring_search_pattern,
)

CONFIG_PROFILE_COLUMNS = "id, name, type, config, created_at, updated_at"
_PATCHABLE_COLUMNS = frozenset({"name", "config"})


class ConfigProfileRepository:
    def __init__(self, conn: psycopg.Connection) -> None:
        self.conn = conn

    def create(
        self,
        profile_id: str,
        *,
        name: str,
        type: str,
        config: dict[str, Any],
        owner_id: str | None = None,
    ) -> dict:
        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO config_profiles (id, name, type, config, owner_id)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING {CONFIG_PROFILE_COLUMNS}
                """,
                (profile_id, name, type, Jsonb(config), owner_id),
            )
            return cur.fetchone()

    def list(
        self,
        *,
        type: str | None,
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
        if type is not None:
            filters.append("type = %s")
            params.append(type)
        if search := substring_search_pattern(q):
            filters.append("(id ILIKE %s ESCAPE '\\' OR name ILIKE %s ESCAPE '\\' OR type::text ILIKE %s ESCAPE '\\')")
            params.extend([search] * 3)
        cursor_filter, cursor_params = created_at_cursor_clause(cursor, direction)
        if cursor_filter:
            filters.append(cursor_filter)
            params.extend(cursor_params)
        params.append(limit + 1)
        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT {CONFIG_PROFILE_COLUMNS}
                FROM config_profiles
                WHERE {join_where(filters)}
                ORDER BY {ordering}
                LIMIT %s
                """,
                params,
            )
            return cur.fetchall()

    def get(self, profile_id: str) -> dict | None:
        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT {CONFIG_PROFILE_COLUMNS}
                FROM config_profiles
                WHERE id = %s AND deleted_at IS NULL
                """,
                (profile_id,),
            )
            return cur.fetchone()

    def update(
        self,
        profile_id: str,
        *,
        name: str | None = None,
        config: dict[str, Any] | None = None,
        owner_id: str | None = None,
    ) -> dict | None:
        updates: list[tuple[str, Any]] = [("name", name)]
        if config is not None:
            updates.append(("config", Jsonb(config)))
        sets, params = patch_set_clause(updates, _PATCHABLE_COLUMNS)
        # owner_id=None keeps writes unrestricted (auth-disabled dev, admins);
        # otherwise the row must belong to the caller.
        owner_filter = "" if owner_id is None else " AND owner_id = %s"
        owner_params: list[Any] = [] if owner_id is None else [owner_id]

        with self.conn.transaction(), self.conn.cursor() as cur:
            if config is not None:
                cur.execute(
                    f"""
                    SELECT id FROM config_profiles
                    WHERE id = %s AND deleted_at IS NULL{owner_filter}
                    FOR UPDATE
                    """,
                    (profile_id, *owner_params),
                )
                if cur.fetchone() is None:
                    return None
                if self.active_evaluation_reference_exists(profile_id):
                    raise Conflict(
                        "profile_in_use",
                        "config profile is referenced by an active evaluation",
                    )
            if not sets:
                cur.execute(
                    f"""
                    SELECT {CONFIG_PROFILE_COLUMNS}
                    FROM config_profiles
                    WHERE id = %s AND deleted_at IS NULL{owner_filter}
                    """,
                    (profile_id, *owner_params),
                )
            else:
                sets.append("updated_at = NOW()")
                params.append(profile_id)
                params.extend(owner_params)
                cur.execute(
                    f"""
                    UPDATE config_profiles
                    SET {", ".join(sets)}
                    WHERE id = %s AND deleted_at IS NULL{owner_filter}
                    RETURNING {CONFIG_PROFILE_COLUMNS}
                    """,
                    params,
                )
            return cur.fetchone()

    def active_evaluation_reference_exists(self, profile_id: str) -> bool:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT id
                FROM evaluations
                WHERE deleted_at IS NULL
                  AND status NOT IN ('succeeded', 'failed', 'cancelled')
                  AND (
                      framework_profile_id = %s
                      OR harbor_profile_id = %s
                      OR intake_profile_id = %s
                  )
                LIMIT 1
                """,
                (profile_id, profile_id, profile_id),
            )
            return cur.fetchone() is not None

    def soft_delete(self, profile_id: str, *, owner_id: str | None = None) -> bool:
        if self.active_evaluation_reference_exists(profile_id):
            raise Conflict(
                "profile_in_use",
                "config profile is referenced by an active evaluation",
            )
        # owner_id=None keeps deletes unrestricted (auth-disabled dev, admins).
        owner_filter = "" if owner_id is None else " AND owner_id = %s"
        owner_params: list[Any] = [] if owner_id is None else [owner_id]
        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE config_profiles
                SET deleted_at = NOW()
                WHERE id = %s AND deleted_at IS NULL{owner_filter}
                RETURNING id
                """,
                (profile_id, *owner_params),
            )
            return cur.fetchone() is not None
