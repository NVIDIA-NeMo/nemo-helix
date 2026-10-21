# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-memory stand-in for SyncEntityClient, with the store's version checks."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

from nemo_helix_plugin.entities.base import EntityConflictError, EntityNotFoundError

_EPOCH = datetime(2026, 1, 1, tzinfo=UTC)


class FakeEntityStore:
    def __init__(self) -> None:
        self.stored: dict[tuple[str, str, str], Any] = {}
        self._ticks = 0

    def _now(self) -> datetime:
        # Strictly increasing, so ordering by creation time is deterministic.
        self._ticks += 1
        return _EPOCH + timedelta(seconds=self._ticks)

    @staticmethod
    def _key(entity_type: Any, workspace: str | None, name: str) -> tuple[str, str, str]:
        return (entity_type.__entity_type__, workspace or "default", name)

    def _copy(self, entity: Any) -> Any:
        clone = entity.model_copy(deep=True)
        for attr in ("_id", "_created_at", "_updated_at", "_db_version"):
            setattr(clone, attr, getattr(entity, attr))
        return clone

    def create(self, entity: Any) -> Any:
        key = self._key(type(entity), entity.workspace, entity.name)
        if key in self.stored:
            raise EntityConflictError(f"{entity.name} exists")
        stored = self._copy(entity)
        stored._id = f"entity-{len(self.stored) + 1}"
        stored._created_at = stored._updated_at = self._now()
        stored._db_version = 1
        self.stored[key] = stored
        return self._copy(stored)

    def get(self, entity_type: Any, name: str, *, workspace: str | None = None, **_: Any) -> Any:
        key = self._key(entity_type, workspace, name)
        if key not in self.stored:
            raise EntityNotFoundError(f"{name} not found")
        return self._copy(self.stored[key])

    def update(self, entity: Any, *, original_name: str | None = None) -> Any:
        key = self._key(type(entity), entity.workspace, entity.name)
        current = self.stored.get(key)
        if current is None:
            raise EntityNotFoundError(f"{entity.name} not found")
        if current.db_version != entity.db_version:
            raise EntityConflictError("stale db_version")
        stored = self._copy(entity)
        stored._updated_at = self._now()
        stored._db_version = current.db_version + 1
        self.stored[key] = stored
        return self._copy(stored)

    def delete(
        self,
        entity_type: Any,
        name: str,
        *,
        workspace: str | None = None,
        expected_db_version: int | None = None,
        **_: Any,
    ) -> None:
        key = self._key(entity_type, workspace, name)
        current = self.stored.get(key)
        if current is None:
            raise EntityNotFoundError(f"{name} not found")
        if expected_db_version is not None and current.db_version != expected_db_version:
            raise EntityConflictError("stale db_version")
        del self.stored[key]

    def list(
        self, entity_type: Any, *, workspace: str = "default", page: int = 1, page_size: int = 100, **_: Any
    ) -> Any:
        matches = [
            self._copy(entity)
            for (kind, entity_workspace, _name), entity in self.stored.items()
            if kind == entity_type.__entity_type__ and entity_workspace == workspace
        ]
        total_pages = max(1, -(-len(matches) // page_size))
        data = matches[(page - 1) * page_size : page * page_size]
        return SimpleNamespace(
            data=data, pagination=SimpleNamespace(total_pages=total_pages, total_results=len(matches))
        )
