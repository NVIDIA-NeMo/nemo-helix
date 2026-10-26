# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict versioned token validation and effective-query binding."""

import json
from datetime import datetime

import base58
import pytest
from nhx.common.api.filter import ComparisonOperation, FilterOperation, FilterOperator
from nhx.common.api.utils import tweak_spec
from nhx.core.entities.api.v2.entities.endpoints import EntitiesPage
from nhx.core.entities.api.v2.utils import add_workspace_filtering
from nhx.core.entities.app.repository.pagination import EntityPageToken, entity_page_scope


def scope_hash(
    *,
    workspace: str = "default",
    entity_type: str | None = "widget",
    sort: str = "-created_at",
    filter_op: FilterOperation | None = None,
    relationship_child_workspaces: set[str] | frozenset[str] | None = frozenset({"a", "b"}),
) -> str:
    return entity_page_scope(
        workspace=workspace,
        entity_type=entity_type,
        sort=sort,
        filter_op=filter_op,
        relationship_child_workspaces=(
            set(relationship_child_workspaces) if relationship_child_workspaces is not None else None
        ),
    )


def payload():
    return {"v": 1, "t": "2026-01-01T00:00:00.123456", "i": "widget-1", "q": scope_hash()}


def encode_payload(data):
    return base58.b58encode(json.dumps(data).encode()).decode()


def test_token_round_trip_preserves_microseconds():
    position = EntityPageToken(v=1, t=datetime(2026, 1, 1, microsecond=123456), i="widget-1", q=scope_hash())
    assert EntityPageToken.decode(position.encode(), scope_hash=scope_hash()) == position


@pytest.mark.parametrize(
    "changes",
    [
        {"v": 2},
        {"v": "1"},
        {"t": "not-a-date"},
        {"t": 123},
        {"t": "2026-01-01T00:00:00Z"},
        {"i": ""},
        {"i": 1},
        {"i": "i" * 256},
        {"q": "not-a-hash"},
        {"extra": "ignored?"},
    ],
)
def test_token_rejects_invalid_payload(changes):
    data = payload() | changes
    with pytest.raises(ValueError, match="Invalid page_token"):
        EntityPageToken.decode(encode_payload(data), scope_hash=scope_hash())


@pytest.mark.parametrize("field", ["v", "t", "i", "q"])
def test_token_requires_every_field(field):
    data = payload()
    del data[field]
    with pytest.raises(ValueError, match="Invalid page_token"):
        EntityPageToken.decode(encode_payload(data), scope_hash=scope_hash())


@pytest.mark.parametrize("data", [[], None, 1, "text"])
def test_token_rejects_other_json_shapes(data):
    with pytest.raises(ValueError, match="Invalid page_token"):
        EntityPageToken.decode(encode_payload(data), scope_hash=scope_hash())


@pytest.mark.parametrize(
    "changes",
    [
        {"workspace": "other"},
        {"entity_type": "other"},
        {"sort": "created_at"},
        {"relationship_child_workspaces": {"a"}},
        {"relationship_child_workspaces": None},
        {"filter_op": ComparisonOperation(field="name", operator=FilterOperator.EQ, value="one")},
    ],
)
def test_token_rejects_changed_query_or_authorization_scope(changes):
    with pytest.raises(ValueError, match="does not match"):
        EntityPageToken.decode(encode_payload(payload()), scope_hash=scope_hash(**changes))


def test_terminal_token_remains_nullable_in_normalized_openapi():
    spec = tweak_spec({"components": {"schemas": {"EntitiesPage": EntitiesPage.model_json_schema()}}})
    token_schema = spec["components"]["schemas"]["EntitiesPage"]["properties"]["next_page_token"]
    assert token_schema["type"] == ["string", "null"]


def test_cross_workspace_filter_uses_canonical_workspace_order():
    operation = add_workspace_filtering({"z", "a", "m"}, None, field="workspace")
    assert operation is not None
    assert operation.to_dict() == {"workspace": {"$in": ["a", "m", "z"]}}


def test_scope_is_independent_of_workspace_set_iteration_order():
    assert scope_hash(relationship_child_workspaces=set(["a", "b"])) == scope_hash(
        relationship_child_workspaces=set(["b", "a"])
    )
