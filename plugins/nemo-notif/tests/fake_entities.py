# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-memory stand-in for NemoEntitiesClient, covering the calls the notif store makes."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from nemo_helix_plugin.entity_client import NemoEntityConflictError, NemoEntityNotFoundError
from nemo_notif_plugin.entities import NotifEventEntity

T0 = datetime(2026, 11, 18, tzinfo=timezone.utc)


class FakeEntitiesClient:
    def __init__(self) -> None:
        self.rows: dict[tuple[str, str], NotifEventEntity] = {}
        self.conflicts_to_raise = 0

    def _copy(self, entity: NotifEventEntity) -> NotifEventEntity:
        copy = entity.model_copy(deep=True)
        copy._created_at = entity._created_at
        copy._created_by = entity._created_by
        copy._db_version = entity._db_version
        return copy

    async def create(self, entity: NotifEventEntity) -> NotifEventEntity:
        stored = self._copy(entity)
        stored._created_at = T0 + timedelta(seconds=len(self.rows))
        stored._created_by = "alice"
        self.rows[(entity.workspace, entity.name)] = stored
        return self._copy(stored)

    async def get(self, entity_type: type, *, name: str, workspace: str) -> NotifEventEntity:
        try:
            return self._copy(self.rows[(workspace, name)])
        except KeyError:
            raise NemoEntityNotFoundError(f"{name} not found") from None

    async def update(self, entity: NotifEventEntity) -> NotifEventEntity:
        current = self.rows[(entity.workspace, entity.name)]
        if self.conflicts_to_raise or entity._db_version != current._db_version:
            self.conflicts_to_raise = max(0, self.conflicts_to_raise - 1)
            current._db_version += 1  # simulate a concurrent writer
            raise NemoEntityConflictError("db_version mismatch")
        stored = self._copy(entity)
        stored._db_version = current._db_version + 1
        self.rows[(entity.workspace, entity.name)] = stored
        return self._copy(stored)

    async def list(self, entity_type: type, *, workspace: str, sort: str, page: int, page_size: int):
        assert sort == "-created_at"
        rows = sorted(
            (e for (ws, _), e in self.rows.items() if ws == workspace),
            key=lambda e: e._created_at,
            reverse=True,
        )
        return SimpleNamespace(data=[self._copy(e) for e in rows[:page_size]])
