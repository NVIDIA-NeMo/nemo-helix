# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Policy seam for repository data and query operations."""

from typing import Any, Protocol

from nhx.common.api.filter import FilterOperation
from nhx.core.entities.entities import Entity


class EntityFieldPolicy(Protocol):
    def create_data(self, data: dict[str, Any]) -> dict[str, Any]: ...

    def update_data(self, data: dict[str, Any], stored_data: dict[str, Any]) -> dict[str, Any]: ...

    def project(self, entity: Entity) -> Entity: ...

    def validate_query(
        self,
        filter_op: FilterOperation | None,
        sort: str | None = None,
        group_by: str | None = None,
    ) -> None: ...


class NoOpEntityFieldPolicy:
    """Default behavior for repositories outside a request."""

    def create_data(self, data: dict[str, Any]) -> dict[str, Any]:
        return data

    def update_data(self, data: dict[str, Any], stored_data: dict[str, Any]) -> dict[str, Any]:
        return data

    def project(self, entity: Entity) -> Entity:
        return entity

    def validate_query(
        self,
        filter_op: FilterOperation | None,
        sort: str | None = None,
        group_by: str | None = None,
    ) -> None:
        return None


NO_OP_ENTITY_FIELD_POLICY = NoOpEntityFieldPolicy()
