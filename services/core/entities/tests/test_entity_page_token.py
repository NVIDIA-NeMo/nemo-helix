# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict versioned token validation and effective-query binding."""

import json
from collections.abc import Iterable, Iterator
from datetime import datetime

import base58
import pytest
from nhx.common.api.filter import FilterOperation
from nhx.common.api.utils import tweak_spec
from nhx.core.entities.api.v2.entities.endpoints import EntitiesPage
from nhx.core.entities.api.v2.utils import add_workspace_filtering
from nhx.core.entities.app.repository.pagination import MAX_PAGE_TOKEN_LENGTH, EntityPageToken, entity_page_scope


class IterationOrderedSet(set[str]):
    """Simulate differing set iteration orders across server workers."""

    def __init__(self, values: Iterable[str], *, reverse: bool = False):
        super().__init__(values)
        self._reverse = reverse

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(super().__iter__(), reverse=self._reverse))


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
            set(relationship_child_workspaces)
            if isinstance(relationship_child_workspaces, frozenset)
            else relationship_child_workspaces
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
        pytest.param({"v": 2}, id="unsupported-version"),
        pytest.param({"v": "1"}, id="string-version"),
        pytest.param({"t": "not-a-date"}, id="invalid-timestamp"),
        pytest.param({"t": 123}, id="numeric-timestamp"),
        pytest.param({"t": "2026-01-01T00:00:00Z"}, id="aware-timestamp"),
        pytest.param({"i": ""}, id="empty-id"),
        pytest.param({"i": 1}, id="numeric-id"),
        pytest.param({"i": "i" * 256}, id="oversize-id"),
        pytest.param({"q": "not-a-hash"}, id="malformed-scope-hash"),
        pytest.param({"extra": "ignored?"}, id="unexpected-field"),
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


def test_token_rejects_non_object_payload():
    with pytest.raises(ValueError, match="Invalid page_token"):
        EntityPageToken.decode(encode_payload([]), scope_hash=scope_hash())


@pytest.mark.parametrize(
    "changes",
    [
        pytest.param({"relationship_child_workspaces": {"a"}}, id="changed-child-workspaces"),
        pytest.param({"relationship_child_workspaces": None}, id="unrestricted-child-workspaces"),
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
    workspaces = IterationOrderedSet({"z", "a", "m"}, reverse=True)
    operation = add_workspace_filtering(workspaces, None, field="workspace")
    assert operation is not None
    assert operation.to_dict() == {"workspace": {"$in": ["a", "m", "z"]}}


def test_scope_is_independent_of_workspace_set_iteration_order():
    # Constructing two ordinary sets with the same elements does not reliably
    # change their iteration order. Simulate the order difference across workers.
    workspaces = {"a", "b"}
    assert scope_hash(relationship_child_workspaces=IterationOrderedSet(workspaces)) == scope_hash(
        relationship_child_workspaces=IterationOrderedSet(workspaces, reverse=True)
    )


def test_decoder_rejects_oversize_token_before_decoding():
    # Valid JSON with leading whitespace would decode successfully without the
    # size guard. A malformed oversized token would not prove that guard exists.
    raw = b" " * MAX_PAGE_TOKEN_LENGTH + json.dumps(payload()).encode()
    token = base58.b58encode(raw).decode()
    assert len(token) > MAX_PAGE_TOKEN_LENGTH
    with pytest.raises(ValueError, match="Invalid page_token"):
        EntityPageToken.decode(token, scope_hash=scope_hash())
