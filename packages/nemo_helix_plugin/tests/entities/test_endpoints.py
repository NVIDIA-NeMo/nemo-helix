# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Entities service endpoint definitions."""

from __future__ import annotations

import json
from typing import get_origin

from nemo_helix_plugin.client.types import Paginated, PreparedRequest
from nemo_helix_plugin.entities import endpoints
from nemo_helix_plugin.entities.types import DeleteResponse, Entity, EntityCreateInput, EntityUpdate


def test_create_entity() -> None:
    body = EntityCreateInput(data={"colour": "blue"})
    prepared = endpoints.create_entity(workspace="default", entity_type="widget", body=body)

    assert isinstance(prepared, PreparedRequest)
    assert prepared.method == "POST"
    assert prepared.path_template == "/apis/entities/v2/workspaces/{workspace}/entities/{entity_type}"
    assert prepared.path_params == {"workspace": "default", "entity_type": "widget"}
    assert prepared.content == body.model_dump_json(exclude_unset=True).encode()
    assert prepared.content_type == "application/json"
    assert prepared.response_type is Entity


def test_create_entity_workspace_optional() -> None:
    body = EntityCreateInput(data={"colour": "blue"})
    prepared = endpoints.create_entity(entity_type="widget", body=body)

    assert prepared.path_params == {"entity_type": "widget"}


def test_create_entity_excludes_unset_fields() -> None:
    body = EntityCreateInput(data={"colour": "blue"})
    prepared = endpoints.create_entity(workspace="default", entity_type="widget", body=body)

    content = json.loads(prepared.content)
    assert content == {"data": {"colour": "blue"}}
    assert "name" not in content
    assert "parent" not in content


def test_list_entities() -> None:
    prepared = endpoints.list_entities(workspace="default", entity_type="widget")

    assert prepared.method == "GET"
    assert prepared.path_template == "/apis/entities/v2/workspaces/{workspace}/entities/{entity_type}"
    assert prepared.path_params == {"workspace": "default", "entity_type": "widget"}
    assert prepared.content is None
    assert get_origin(prepared.response_type) is Paginated


def test_list_entities_with_query_params() -> None:
    prepared = endpoints.list_entities(
        workspace="default", entity_type="widget", query_params={"page": 2, "page_size": 10, "sort": "-created_at"}
    )

    assert prepared.query_params == {"page": 2, "page_size": 10, "sort": "-created_at"}


def test_list_entities_cross_workspace() -> None:
    """workspace='-' is a normal path value the caller supplies for cross-workspace listing."""
    prepared = endpoints.list_entities(workspace="-", entity_type="widget")

    assert prepared.path_params == {"workspace": "-", "entity_type": "widget"}


def test_get_entity_by_name() -> None:
    prepared = endpoints.get_entity_by_name(workspace="default", entity_type="widget", name="my-widget")

    assert prepared.method == "GET"
    assert prepared.path_template == "/apis/entities/v2/workspaces/{workspace}/entities/{entity_type}/{name}"
    assert prepared.path_params == {"workspace": "default", "entity_type": "widget", "name": "my-widget"}
    assert prepared.response_type is Entity


def test_get_entity_by_name_with_parent() -> None:
    prepared = endpoints.get_entity_by_name(
        workspace="default", entity_type="widget", name="my-widget", query_params={"parent": "parent-id"}
    )

    assert prepared.query_params == {"parent": "parent-id"}


def test_update_entity_by_name() -> None:
    body = EntityUpdate(data={"colour": "red"})
    prepared = endpoints.update_entity_by_name(workspace="default", entity_type="widget", name="my-widget", body=body)

    assert prepared.method == "PUT"
    assert prepared.path_params == {"workspace": "default", "entity_type": "widget", "name": "my-widget"}
    assert prepared.content == body.model_dump_json(exclude_unset=True).encode()
    assert prepared.response_type is Entity


def test_update_entity_by_name_excludes_unset_fields() -> None:
    body = EntityUpdate(data={"colour": "red"})
    prepared = endpoints.update_entity_by_name(workspace="default", entity_type="widget", name="my-widget", body=body)

    content = json.loads(prepared.content)
    assert content == {"data": {"colour": "red"}}
    assert "new_name" not in content
    assert "expected_db_version" not in content


def test_delete_entity_by_name() -> None:
    prepared = endpoints.delete_entity_by_name(workspace="default", entity_type="widget", name="my-widget")

    assert prepared.method == "DELETE"
    assert prepared.path_params == {"workspace": "default", "entity_type": "widget", "name": "my-widget"}
    assert prepared.content is None
    assert prepared.query_params is None
    assert prepared.response_type is DeleteResponse


def test_delete_entity_by_name_with_version_guard() -> None:
    """``expected_db_version`` is the optimistic-locking guard and must ride as a query param.

    The server declares it as ``Query(...)`` on the delete route, so the key name here is a
    wire contract: a rename on either side silently turns a guarded delete into an
    unconditional one. See the integration test in the entities service for the paired
    server-side assertion.
    """
    prepared = endpoints.delete_entity_by_name(
        workspace="default",
        entity_type="widget",
        name="my-widget",
        query_params={"parent": "parent-id", "expected_db_version": 7},
    )

    assert prepared.query_params == {"parent": "parent-id", "expected_db_version": 7}


def test_get_entity_by_id() -> None:
    prepared = endpoints.get_entity_by_id(entity_id="widget-5Q2LoF8z8M9JZxZsHwJKNn")

    assert prepared.method == "GET"
    assert prepared.path_template == "/apis/entities/v2/entities/{entity_id}"
    assert prepared.path_params == {"entity_id": "widget-5Q2LoF8z8M9JZxZsHwJKNn"}
    assert prepared.content is None
    assert prepared.response_type is Entity
