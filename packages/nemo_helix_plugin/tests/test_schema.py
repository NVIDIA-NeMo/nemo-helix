# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for nemo_helix_plugin.schema — NemoListResponse and NemoFilter."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from nemo_helix_plugin.entity import NemoEntity
from nemo_helix_plugin.schema import NemoFilter, NemoListResponse, PaginationData
from pydantic import ValidationError

NOW = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


class _Widget(NemoEntity, entity_type="test_schema_widget_v2"):
    colour: str = "red"
    weight_kg: float = 0.0


def _make_widget(name: str = "w1") -> _Widget:
    w = _Widget(name=name, workspace="default")
    w._id = f"id-{name}"
    w._created_at = NOW
    return w


class _WidgetFilter(NemoFilter):
    colour: str | None = None


# ---------------------------------------------------------------------------
# NemoListResponse — used with entity objects directly
# ---------------------------------------------------------------------------


def test_list_response_with_entities() -> None:
    """NemoListResponse accepts entity objects in data list."""
    page = NemoListResponse[_Widget](
        data=[_make_widget("w1"), _make_widget("w2")],
    )
    assert len(page.data) == 2
    assert page.data[0].name == "w1"
    assert page.data[0].id == "id-w1"
    assert page.data[0].created_at == NOW


def test_list_response_basic_structure() -> None:
    """NemoListResponse has the expected Page-compatible wire format."""
    page = NemoListResponse[_Widget](data=[_make_widget()])
    assert page.pagination is None
    assert page.sort is None
    assert page.filter is None


def test_list_response_with_pagination() -> None:
    """NemoListResponse accepts and echoes PaginationData."""
    pagination = PaginationData(
        page=1,
        page_size=20,
        current_page_size=1,
        total_pages=1,
        total_results=1,
    )
    page = NemoListResponse[_Widget](
        data=[_make_widget()],
        pagination=pagination,
        sort="-created_at",
        filter={"colour": "red"},
    )
    assert page.pagination is not None
    assert page.pagination.total_results == 1
    assert page.sort == "-created_at"
    assert page.filter == {"colour": "red"}


def test_list_response_serializes_to_expected_keys() -> None:
    """Serialized NemoListResponse has data/pagination/sort/filter keys."""
    page = NemoListResponse[_Widget](data=[_make_widget()])
    payload = page.model_dump()
    assert set(payload.keys()) == {"data", "pagination", "sort", "filter"}


def test_list_response_data_includes_entity_computed_fields() -> None:
    """Serialized entity items include id and created_at from computed fields."""
    page = NemoListResponse[_Widget](data=[_make_widget("w1")])
    payload = page.model_dump()
    item = payload["data"][0]
    assert item["id"] == "id-w1"
    assert item["name"] == "w1"
    assert item["created_at"] == NOW


def test_list_response_empty_data() -> None:
    """NemoListResponse accepts an empty data list."""
    page = NemoListResponse[_Widget](data=[])
    assert page.data == []


# ---------------------------------------------------------------------------
# NemoFilter
# ---------------------------------------------------------------------------


def test_filter_extra_field_raises() -> None:
    """NemoFilter subclass rejects unknown fields (extra='forbid')."""
    with pytest.raises(ValidationError, match="Extra inputs"):
        _WidgetFilter.model_validate({"colour": "red", "unknown_field": "oops"})


def test_filter_declared_field_accepted() -> None:
    """NemoFilter subclass accepts declared fields without error."""
    f = _WidgetFilter(colour="blue")
    assert f.colour == "blue"


def test_filter_all_optional_fields_omitted() -> None:
    """NemoFilter subclass with all-optional fields can be constructed empty."""
    f = _WidgetFilter()
    assert f.colour is None


def test_nemo_filter_is_exported() -> None:
    """NemoFilter is importable from nemo_helix_plugin.schema."""
    from nemo_helix_plugin.schema import NemoFilter as NF

    assert NF is NemoFilter
