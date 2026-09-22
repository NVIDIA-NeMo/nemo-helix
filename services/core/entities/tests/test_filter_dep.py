# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for FilterDep auto-detection between object and text search syntax."""

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException, Request
from nhx.common.api.filter import (
    ComparisonOperation,
    FilterOperator,
    LogicalOperation,
)
from nhx.core.entities.utils.filter import make_filter_dep
from starlette.datastructures import QueryParams


def create_mock_request(query_string: str, entity_type: str | None = None) -> Request:
    """Create a mock FastAPI request with the given query string."""
    request = MagicMock(spec=Request)

    params = {}
    if query_string:
        for param in query_string.split("&"):
            if "=" in param:
                key, value = param.split("=", 1)
                params[key] = value

    request.query_params = QueryParams(params)
    request.path_params = {"entity_type": entity_type} if entity_type else {}
    return request


async def call_filter_dep(request: Request, filter_value: str | None = None):
    """Call the filter dependency function directly."""
    dep = make_filter_dep()
    return await dep(request, filter_value)


class TestFilterDep:
    """Test the filter dependency function."""

    @pytest.mark.asyncio
    async def test_json_filter(self):
        """Test parsing JSON object filter via dependency."""
        request = create_mock_request('filter={"name":"llama"}')

        result = await call_filter_dep(request, filter_value='{"name":"llama"}')

        assert isinstance(result, ComparisonOperation)
        assert result.field == "name"
        assert result.value == "llama"

    @pytest.mark.asyncio
    async def test_text_search(self):
        """Test parsing text search via dependency (auto-detection)."""
        request = create_mock_request('filter=name:"llama"')

        result = await call_filter_dep(request, filter_value='name:"llama"')

        assert isinstance(result, ComparisonOperation)
        assert result.field == "name"
        assert result.value == "llama"

    @pytest.mark.asyncio
    async def test_text_search_with_operators(self):
        """Test text search with AND/OR operators."""
        request = create_mock_request("")

        result = await call_filter_dep(request, filter_value='status:"active" AND amount>500')

        assert isinstance(result, LogicalOperation)
        assert result.operator == FilterOperator.AND
        assert len(result.operations) == 2

    @pytest.mark.asyncio
    async def test_bracket_search(self):
        """Test parsing bracket search via dependency."""
        request = create_mock_request("filter[name][$like]=llama")

        result = await call_filter_dep(request, filter_value=None)

        assert isinstance(result, ComparisonOperation)
        assert result.operator == FilterOperator.LIKE
        assert result.field == "name"
        assert result.value == "llama"

    @pytest.mark.asyncio
    async def test_no_search(self):
        """Test no search params returns None."""
        request = create_mock_request("")

        result = await call_filter_dep(request, filter_value=None)

        assert result is None

    @pytest.mark.asyncio
    async def test_invalid_json_raises_http_exception(self):
        request = create_mock_request("")

        with pytest.raises(HTTPException) as exc_info:
            await call_filter_dep(request, filter_value="{invalid}")

        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_invalid_text_raises_http_exception(self):
        request = create_mock_request("")

        with pytest.raises(HTTPException) as exc_info:
            await call_filter_dep(request, filter_value="!!invalid!!")

        assert exc_info.value.status_code == 400


class TestAutoDetection:
    """Test auto-detection between object and text search syntax."""

    @pytest.mark.asyncio
    async def test_detects_json_by_curly_brace(self):
        """Strings starting with { are parsed as JSON."""
        request = create_mock_request("")

        result = await call_filter_dep(request, filter_value='{"name":"llama"}')
        assert isinstance(result, ComparisonOperation)
        assert result.value == "llama"

    @pytest.mark.asyncio
    async def test_detects_text_without_curly_brace(self):
        """Strings not starting with { are parsed as text syntax."""
        request = create_mock_request("")

        result = await call_filter_dep(request, filter_value='name:"llama"')
        assert isinstance(result, ComparisonOperation)
        assert result.value == "llama"

    @pytest.mark.asyncio
    async def test_whitespace_before_json(self):
        """Leading whitespace is stripped before detection."""
        request = create_mock_request("")

        result = await call_filter_dep(request, filter_value='  {"name":"test"}')
        assert isinstance(result, ComparisonOperation)
        assert result.value == "test"

    @pytest.mark.asyncio
    async def test_whitespace_before_text(self):
        """Leading whitespace is stripped before text parsing."""
        request = create_mock_request("")

        result = await call_filter_dep(request, filter_value='  name:"test"')
        assert isinstance(result, ComparisonOperation)
        assert result.value == "test"


class TestFilterOnly:
    """Test that filter param is primary, search is accepted for backward compat."""

    @pytest.mark.asyncio
    async def test_search_param_accepted_as_fallback(self):
        """When only search param is present (no filter), it's accepted as backward compat."""
        request = create_mock_request('search=name:"llama"')
        result = await call_filter_dep(request, filter_value=None)
        assert result is not None
        assert isinstance(result, ComparisonOperation)
        assert result.field == "name"
        assert result.value == "llama"

    @pytest.mark.asyncio
    async def test_filter_param_works(self):
        """Filter param is used."""
        request = MagicMock(spec=Request)
        request.query_params = QueryParams({"filter": 'name:"primary"'})
        request.path_params = {}
        result = await call_filter_dep(request, filter_value='name:"primary"')
        assert isinstance(result, ComparisonOperation)
        assert result.value == "primary"
