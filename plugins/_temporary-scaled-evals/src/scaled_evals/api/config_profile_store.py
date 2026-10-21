# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config profiles, stored in Entity Store as one entity per profile id."""

from __future__ import annotations

import builtins
from collections.abc import Iterable
from typing import Any

from nemo_helix_plugin.entities.base import EntityConflictError, EntityNotFoundError, SyncEntityClient
from nemo_scaled_evals_plugin.entities import ScaledEvalsConfigProfile

from scaled_evals.api import entity_store
from scaled_evals.api.repositories.base_repository import Conflict, InvalidReference


def _row(entity: ScaledEvalsConfigProfile) -> dict[str, Any]:
    return {
        "id": entity.name,
        "owner_id": entity.owner_id,
        "name": entity.display_name,
        "type": entity.type,
        "config": entity.config,
        "created_at": entity.created_at,
        "updated_at": entity.updated_at,
    }


class ConfigProfileStore:
    def __init__(self, client: SyncEntityClient | None = None) -> None:
        self._client = client or entity_store.entity_client()

    def _get(self, profile_id: str) -> ScaledEvalsConfigProfile | None:
        try:
            return self._client.get(ScaledEvalsConfigProfile, profile_id, workspace=entity_store.private_workspace())
        except EntityNotFoundError:
            return None

    def create(
        self,
        profile_id: str,
        *,
        name: str,
        type: str,
        config: dict[str, Any],
        owner_id: str | None = None,
    ) -> dict[str, Any]:
        entity = ScaledEvalsConfigProfile(
            name=profile_id,
            workspace=entity_store.private_workspace(),
            owner_id=owner_id,
            display_name=name,
            type=type,
            config=config,
        )
        return _row(self._client.create(entity))

    def list(
        self,
        *,
        type: str | None,
        limit: int,
        cursor: str | None,
        order: str,
        owner_id: str | None = None,
        q: str | None = None,
    ) -> builtins.list[dict[str, Any]]:
        # ponytail: reads every profile and filters in process, which keeps the
        # exact keyset cursor. O(n) per list; move the filters into the store
        # query once profiles number in the thousands.
        rows = [
            _row(entity)
            for entity in entity_store.list_all(self._client, ScaledEvalsConfigProfile)
            if (owner_id is None or entity.owner_id == owner_id)
            and (type is None or entity.type == type)
            and entity_store.contains(q, entity.name, entity.display_name, entity.type)
        ]
        return entity_store.keyset_page(rows, cursor=cursor, order=order, limit=limit)

    def get(self, profile_id: str) -> dict[str, Any] | None:
        entity = self._get(profile_id)
        return None if entity is None else _row(entity)

    def get_typed(self, profile_id: str, types: Iterable[str]) -> dict[str, Any] | None:
        """Return the profile when it exists and has one of `types`."""
        row = self.get(profile_id)
        return row if row is not None and row["type"] in set(types) else None

    def get_many(self, profile_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
        """Return the existing profiles among `profile_ids`, keyed by id."""
        rows = {profile_id: self.get(profile_id) for profile_id in set(profile_ids)}
        return {profile_id: row for profile_id, row in rows.items() if row is not None}

    def validate_references(self, profile_slots: builtins.list[tuple[str, str]]) -> None:
        """Raise `InvalidReference` unless each `(id, type)` names a profile of that type."""
        found = self.get_many(profile_id for profile_id, _ in profile_slots)
        for profile_id, expected_type in profile_slots:
            if profile_id not in found:
                raise InvalidReference(f"config profile not found: {profile_id}")
            actual_type = found[profile_id]["type"]
            if actual_type != expected_type:
                raise InvalidReference(f"{profile_id} is a '{actual_type}' profile, expected '{expected_type}'")

    def update(
        self,
        profile_id: str,
        *,
        name: str | None = None,
        config: dict[str, Any] | None = None,
        owner_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Apply the given fields; None when absent or, with `owner_id`, not owned by it."""
        entity = self._get(profile_id)
        if entity is None or (owner_id is not None and entity.owner_id != owner_id):
            return None
        if name is None and config is None:
            return _row(entity)
        if name is not None:
            entity.display_name = name
        if config is not None:
            entity.config = config
        try:
            return _row(self._client.update(entity))
        except EntityConflictError as exc:
            raise Conflict("profile_changed", "config profile changed concurrently; retry") from exc

    def delete(self, profile_id: str, *, owner_id: str | None = None) -> bool:
        """Delete the profile; False when absent or, with `owner_id`, not owned by it."""
        entity = self._get(profile_id)
        if entity is None or (owner_id is not None and entity.owner_id != owner_id):
            return False
        try:
            self._client.delete(
                ScaledEvalsConfigProfile,
                profile_id,
                workspace=entity_store.private_workspace(),
                expected_db_version=entity.db_version,
            )
        except EntityNotFoundError:
            return False
        except EntityConflictError as exc:
            raise Conflict("profile_changed", "config profile changed concurrently; retry") from exc
        return True
