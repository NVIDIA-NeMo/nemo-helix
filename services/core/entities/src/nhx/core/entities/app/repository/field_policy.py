# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable request-bound policy for one top-level Entity.data field."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nhx.common.api.filter import ComparisonOperation, FilterOperation, LogicalOperation
from nhx.core.entities.entities import Entity
from nhx.core.entities.utils.filter import RelationshipFilterOperation


@dataclass(frozen=True)
class ProtectedEntityField:
    key: str

    def __post_init__(self) -> None:
        if not self.key or "." in self.key:
            raise ValueError("Protected field must be a nonempty top-level Entity.data key")

    def bind(self, *, policy_enabled: bool, can_read: bool, can_write: bool) -> BoundEntityFieldPolicy:
        return BoundEntityFieldPolicy(self, policy_enabled, can_read, can_write)


@dataclass(frozen=True)
class BoundEntityFieldPolicy:
    field: ProtectedEntityField
    policy_enabled: bool
    can_read: bool
    can_write: bool

    @property
    def path(self) -> str:
        return f"data.{self.field.key}"

    def _protected_field(self, field: str) -> bool:
        return field == "data" or field == self.path or field.startswith(f"{self.path}.")

    def _validate_filter(self, operation: FilterOperation) -> None:
        if isinstance(operation, ComparisonOperation):
            if self._protected_field(operation.field):
                raise ValueError(f"Field '{self.path}' is not queryable")
            return
        if isinstance(operation, LogicalOperation):
            for child in operation.operations:
                self._validate_filter(child)
            return
        if isinstance(operation, RelationshipFilterOperation):
            if operation.condition is not None:
                self._validate_filter(operation.condition)
            return
        raise ValueError("Filter operation is not permitted for this reader")

    def create_data(self, data: dict[str, Any]) -> dict[str, Any]:
        if not self.policy_enabled or self.can_write:
            return data
        return {key: value for key, value in data.items() if key != self.field.key}

    def update_data(self, data: dict[str, Any], stored_data: dict[str, Any]) -> dict[str, Any]:
        if not self.policy_enabled or self.can_write and data.get(self.field.key) is not None:
            return data
        updated = data.copy()
        if self.field.key in stored_data:
            updated[self.field.key] = stored_data[self.field.key]
        else:
            updated.pop(self.field.key, None)
        return updated

    def project(self, entity: Entity) -> Entity:
        if not self.policy_enabled or self.can_read or self.field.key not in entity.data:
            return entity
        visible_data = {key: value for key, value in entity.data.items() if key != self.field.key}
        return entity.model_copy(update={"data": visible_data})

    def validate_query(
        self,
        filter_op: FilterOperation | None,
        sort: str | None = None,
        group_by: str | None = None,
    ) -> None:
        if not self.policy_enabled or self.can_read:
            return
        if filter_op is not None:
            self._validate_filter(filter_op)
        for field in (sort, group_by):
            if field is not None and self._protected_field(field.removeprefix("-")):
                raise ValueError(f"Field '{self.path}' is not queryable")
