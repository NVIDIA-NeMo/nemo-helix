# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent bundles, stored in Entity Store as one entity per bundle id."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from nemo_helix_plugin.entities.base import EntityConflictError, EntityNotFoundError, SyncEntityClient
from nemo_scaled_evals_plugin.entities import ScaledEvalsAgentBundle

from scaled_evals.api import entity_store
from scaled_evals.api.repositories.base_repository import Conflict

_DATA_FIELDS = frozenset(ScaledEvalsAgentBundle.model_fields) - {"name", "workspace", "project"}


def _row(entity: ScaledEvalsAgentBundle) -> dict[str, Any]:
    return {
        "id": entity.name,
        **{field: getattr(entity, field) for field in _DATA_FIELDS},
        "created_at": entity.created_at,
        "updated_at": entity.updated_at,
    }


def _accessible(entity: ScaledEvalsAgentBundle, owner_id: str) -> bool:
    return entity.owner_id == owner_id or entity.visibility == "public"


def _changed_concurrently() -> Conflict:
    return Conflict("agent_bundle_changed", "agent bundle changed concurrently; retry")


class AgentBundleStore:
    def __init__(self, client: SyncEntityClient | None = None) -> None:
        self._client = client or entity_store.entity_client()

    def _get(self, bundle_id: str) -> ScaledEvalsAgentBundle | None:
        try:
            return self._client.get(ScaledEvalsAgentBundle, bundle_id, workspace=entity_store.private_workspace())
        except EntityNotFoundError:
            return None

    def _save(self, entity: ScaledEvalsAgentBundle) -> dict[str, Any]:
        try:
            return _row(self._client.update(entity))
        except EntityConflictError as exc:
            raise _changed_concurrently() from exc

    def create(self, bundle_id: str, *, owner_id: str, bundle_name: str, **fields: Any) -> dict[str, Any]:
        """Register a private bundle; `Conflict` if the owner already has one by that name."""
        # ponytail: check-then-create, so two concurrent registrations of the
        # same name can both land. A deterministic entity name per
        # (owner, bundle_name) would close it, at the cost of the opaque id.
        if any(
            entity.owner_id == owner_id and entity.bundle_name == bundle_name
            for entity in entity_store.list_all(self._client, ScaledEvalsAgentBundle)
        ):
            raise Conflict("agent_bundle_exists", f"a live bundle named {bundle_name!r} already exists for this owner")
        entity = ScaledEvalsAgentBundle(
            name=bundle_id,
            workspace=entity_store.private_workspace(),
            owner_id=owner_id,
            bundle_name=bundle_name,
            **fields,
        )
        return _row(self._client.create(entity))

    def get_accessible(self, bundle_id: str, *, owner_id: str, include_all: bool = False) -> dict[str, Any] | None:
        entity = self._get(bundle_id)
        if entity is None or not (include_all or _accessible(entity, owner_id)):
            return None
        return _row(entity)

    def list_accessible(
        self,
        *,
        owner_id: str,
        include_all: bool,
        mine: bool,
        visibility: str | None,
        limit: int,
        cursor: str | None,
        order: str,
    ) -> list[dict[str, Any]]:
        # ponytail: in-process filtering over every bundle, as for config
        # profiles; O(n) per list.
        rows = [
            _row(entity)
            for entity in entity_store.list_all(self._client, ScaledEvalsAgentBundle)
            if (entity.owner_id == owner_id if mine else include_all or _accessible(entity, owner_id))
            and (visibility is None or entity.visibility == visibility)
        ]
        return entity_store.keyset_page(rows, cursor=cursor, order=order, limit=limit)

    def set_qualification(
        self, bundle_id: str, *, status: str, evidence: dict[str, Any], qualified_by: str
    ) -> dict[str, Any] | None:
        entity = self._get(bundle_id)
        if entity is None:
            return None
        entity.qualification_status = status
        entity.qualification_evidence = evidence
        entity.qualified_at = datetime.now(tz=UTC)
        entity.qualified_by = qualified_by
        if status == "rejected":
            entity.visibility = "private"
        return self._save(entity)

    def promote(self, bundle_id: str, *, qualified_by: str) -> dict[str, Any] | None:
        """Make a qualified bundle public; None when absent or not qualified."""
        entity = self._get(bundle_id)
        if entity is None or entity.qualification_status != "qualified":
            return None
        entity.visibility = "public"
        entity.qualified_by = qualified_by
        return self._save(entity)

    def delete(self, bundle_id: str, *, owner_id: str, include_all: bool) -> bool:
        """Delete the bundle; non-admins may delete only their own private bundles."""
        entity = self._get(bundle_id)
        if entity is None or not (include_all or (entity.owner_id == owner_id and entity.visibility == "private")):
            return False
        try:
            self._client.delete(
                ScaledEvalsAgentBundle,
                bundle_id,
                workspace=entity_store.private_workspace(),
                expected_db_version=entity.db_version,
            )
        except EntityNotFoundError:
            return False
        except EntityConflictError as exc:
            raise _changed_concurrently() from exc
        return True
