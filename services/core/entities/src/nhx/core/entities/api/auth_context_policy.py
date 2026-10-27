# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request-bound auth-context field policy for Entity Store."""

from dataclasses import dataclass
from typing import Any

from nemo_helix_plugin.auth.access import AuthContextAccessRule
from nhx.common.api.filter import ComparisonOperation, FilterOperation, LogicalOperation
from nhx.core.entities.entities import Entity
from nhx.core.entities.utils.filter import RelationshipFilterOperation

AUTH_CONTEXT_KEY = "_auth_context"
AUTH_CONTEXT_PATH = f"data.{AUTH_CONTEXT_KEY}"


def _protected_field(field: str) -> bool:
    return field == "data" or field == AUTH_CONTEXT_PATH or field.startswith(f"{AUTH_CONTEXT_PATH}.")


def _validate_filter(operation: FilterOperation) -> None:
    if isinstance(operation, ComparisonOperation):
        if _protected_field(operation.field):
            raise ValueError(f"Field '{AUTH_CONTEXT_PATH}' is not queryable")
        return
    if isinstance(operation, LogicalOperation):
        for child in operation.operations:
            _validate_filter(child)
        return
    if isinstance(operation, RelationshipFilterOperation):
        if operation.condition is not None:
            _validate_filter(operation.condition)
        return
    raise ValueError("Filter operation is not permitted for this reader")


@dataclass(frozen=True)
class AuthContextEntityPolicy:
    """Protect stored auth context using the trusted request decision."""

    access_rule: AuthContextAccessRule

    def create_data(self, data: dict[str, Any]) -> dict[str, Any]:
        if not self.access_rule.policy_enabled or self.access_rule.can_write:
            return data
        return {key: value for key, value in data.items() if key != AUTH_CONTEXT_KEY}

    def update_data(self, data: dict[str, Any], stored_data: dict[str, Any]) -> dict[str, Any]:
        if not self.access_rule.policy_enabled:
            return data
        if self.access_rule.can_write and data.get(AUTH_CONTEXT_KEY) is not None:
            return data
        updated = data.copy()
        if AUTH_CONTEXT_KEY in stored_data:
            updated[AUTH_CONTEXT_KEY] = stored_data[AUTH_CONTEXT_KEY]
        else:
            updated.pop(AUTH_CONTEXT_KEY, None)
        return updated

    def project(self, entity: Entity) -> Entity:
        if self.access_rule.can_read or AUTH_CONTEXT_KEY not in entity.data:
            return entity
        visible_data = {key: value for key, value in entity.data.items() if key != AUTH_CONTEXT_KEY}
        return entity.model_copy(update={"data": visible_data})

    def validate_query(
        self,
        filter_op: FilterOperation | None,
        sort: str | None = None,
        group_by: str | None = None,
    ) -> None:
        if self.access_rule.can_read:
            return
        if filter_op is not None:
            _validate_filter(filter_op)
        for field in (sort, group_by):
            if field is not None and _protected_field(field.removeprefix("-")):
                raise ValueError(f"Field '{AUTH_CONTEXT_PATH}' is not queryable")
