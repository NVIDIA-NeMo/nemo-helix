# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for nhx.common.api.generic — generic_get response shape."""

from typing import Optional
from unittest.mock import MagicMock

import pytest
from nhx.common.api.generic import generic_get
from nhx.common.entity.values import Filter, Value


class _Item(Value):
    name: str
    type: str


class _ItemFilter(Filter):
    type: Optional[str] = None


def _request_with_query(query_string: str) -> MagicMock:
    """Build a minimal fake Request with a query_params dict the parser accepts."""
    from starlette.datastructures import QueryParams

    request = MagicMock()
    request.query_params = QueryParams(query_string)
    return request


@pytest.mark.asyncio
async def test_generic_get_filter_response_is_dict_not_model():
    """Page.filter is Optional[dict]; generic_get must serialize the filter model
    before populating the Page, otherwise Pydantic raises ValidationError on response.
    """
    items = [_Item(name="a", type="probes"), _Item(name="b", type="buffs")]

    page = await generic_get(
        all_items=items,
        request=_request_with_query("filter[type]=probes"),
        page=1,
        page_size=10,
        sort="",
        filter_cls=_ItemFilter,
    )

    assert page.filter == {"type": "probes"}
    assert isinstance(page.filter, dict)
    assert [item.name for item in page.data] == ["a"]


@pytest.mark.asyncio
async def test_generic_get_empty_filter_returns_none():
    items = [_Item(name="a", type="probes")]

    page = await generic_get(
        all_items=items,
        request=_request_with_query(""),
        page=1,
        page_size=10,
        sort="",
        filter_cls=_ItemFilter,
    )

    assert page.filter is None
    assert len(page.data) == 1
