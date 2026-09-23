# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed endpoint definitions for the Entities service.

These are the single source of truth for the HTTP contract of the generic
entity-store routes (``services/core/entities`` → ``api/v2/entities``).
"""

from __future__ import annotations

from abc import abstractmethod

from nemo_helix_plugin.client.endpoint import delete, get, post, put
from nemo_helix_plugin.client.types import Paginated
from nemo_helix_plugin.entities.types import (
    DeleteResponse,
    Entity,
    EntityByNameQueryParams,
    EntityCreateInput,
    EntityDeleteQueryParams,
    EntityUpdate,
    ListEntitiesQueryParams,
)

# ---------------------------------------------------------------------------
# Name-based operations (primary)
# ---------------------------------------------------------------------------


@post("/apis/entities/v2/workspaces/{workspace}/entities/{entity_type}")
@abstractmethod
def create_entity(*, workspace: str | None = None, entity_type: str, body: EntityCreateInput) -> Entity: ...


@get("/apis/entities/v2/workspaces/{workspace}/entities/{entity_type}")
@abstractmethod
def list_entities(
    *, workspace: str | None = None, entity_type: str, query_params: ListEntitiesQueryParams | None = None
) -> Paginated[Entity]: ...


@get("/apis/entities/v2/workspaces/{workspace}/entities/{entity_type}/{name}")
@abstractmethod
def get_entity_by_name(
    *, workspace: str | None = None, entity_type: str, name: str, query_params: EntityByNameQueryParams | None = None
) -> Entity: ...


@put("/apis/entities/v2/workspaces/{workspace}/entities/{entity_type}/{name}")
@abstractmethod
def update_entity_by_name(
    *,
    workspace: str | None = None,
    entity_type: str,
    name: str,
    body: EntityUpdate,
    query_params: EntityByNameQueryParams | None = None,
) -> Entity: ...


@delete("/apis/entities/v2/workspaces/{workspace}/entities/{entity_type}/{name}")
@abstractmethod
def delete_entity_by_name(
    *, workspace: str | None = None, entity_type: str, name: str, query_params: EntityDeleteQueryParams | None = None
) -> DeleteResponse: ...


# ---------------------------------------------------------------------------
# ID-based operations (debug/internal) — no workspace in the path
# ---------------------------------------------------------------------------


@get("/apis/entities/v2/entities/{entity_id}")
@abstractmethod
def get_entity_by_id(*, entity_id: str) -> Entity: ...
