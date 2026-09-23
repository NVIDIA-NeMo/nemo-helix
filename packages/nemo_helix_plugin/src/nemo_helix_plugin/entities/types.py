# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared request/response types for the Entities service.

These types define the HTTP contract for the generic entity-store routes.
Both the server (FastAPI routes) and the client (NemoClient endpoints) can
import from here — one source of truth, no Stainless-generated duplicates.

This module must stay free of ``nhx_common`` (server-only) imports, so any
server constant it needs is inlined with a pointer back to the original.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, NotRequired, TypedDict

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Response types
# ---------------------------------------------------------------------------


class Entity(BaseModel):
    """Response DTO for a generic entity-store entity."""

    entity_type: str = Field(description="Entity type identifier")
    id: str = Field(description="UUID identifier")
    workspace: str = Field(description="Workspace identifier")
    parent: str | None = Field(default=None, description="Parent entity ID for nested entities")
    project: str | None = Field(default=None, description="The name of the project associated with this entity")
    name: str = Field(description="Entity name")
    data: dict[str, Any] = Field(description="Entity data")
    created_at: datetime = Field(description="Timestamp of entity creation")
    created_by: str | None = Field(default=None, description="Principal id for entity creator")
    updated_at: datetime = Field(description="Timestamp of last entity update")
    updated_by: str | None = Field(default=None, description="Principal id for last entity update")
    db_version: int = Field(description="Database version of the entity for optimistic locking")

    # Lenient on unknown fields (client-side forward compatibility). from_attributes
    # lets the server build this from an ORM row when it adopts these as its schema.
    model_config = ConfigDict(from_attributes=True)


class DeleteResponse(BaseModel):
    """Response for successful delete operations."""

    message: str = Field(default="Resource deleted successfully")
    id: str = Field(description="ID of the deleted resource")
    deleted_count: int = Field(default=1, description="Number of items deleted")


# ---------------------------------------------------------------------------
# Request types
# ---------------------------------------------------------------------------


class EntityCreateInput(BaseModel):
    """Request body for creating an entity (name-based routes).

    ``name`` is optional — the server auto-generates one when omitted.
    ``workspace`` and ``entity_type`` come from the URL path.
    """

    # Name format is validated server-side (the entities service returns 422 on a
    # bad name). The client DTO stays permissive so an invalid name surfaces as the
    # server's 422, not a client-side pydantic error that a caller would see as a 500.
    name: str | None = Field(
        default=None,
        description="Entity name (optional, auto-generated if not provided)",
    )
    parent: str | None = Field(default=None, description="Parent entity ID for nested entities")
    project: str | None = Field(default=None, description="The name of the project associated with this entity")
    data: dict[str, Any] = Field(description="Entity-specific data (opaque to the entity store)")


class EntityUpdate(BaseModel):
    """Request body for updating an entity."""

    # Name format is validated server-side; see EntityCreateInput.name.
    new_name: str | None = Field(default=None, description="Updated entity name (optional)")
    project: str | None = Field(default=None, description="The name of the project associated with this entity")
    data: dict[str, Any] = Field(description="Updated entity-specific data")
    expected_db_version: int | None = Field(
        default=None,
        description="Optional database version for optimistic locking. Update only succeeds if the version matches.",
    )


# ---------------------------------------------------------------------------
# Query parameter types
# ---------------------------------------------------------------------------


class ListEntitiesQueryParams(TypedDict, total=False):
    page: NotRequired[int]
    page_size: NotRequired[int]
    sort: NotRequired[str]
    filter: NotRequired[str]
    # Group matching entities by a data field; the server returns the tallies in
    # the ``group_counts`` envelope field instead of the usual item payload.
    count_by: NotRequired[str]


class EntityByNameQueryParams(TypedDict, total=False):
    parent: NotRequired[str]


class EntityDeleteQueryParams(TypedDict, total=False):
    parent: NotRequired[str]
    # Optimistic locking: the delete only succeeds if the entity still has this version.
    expected_db_version: NotRequired[int]
