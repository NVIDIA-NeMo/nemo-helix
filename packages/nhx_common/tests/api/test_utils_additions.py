# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for newly added utility functions in nhx.common.api.utils."""

import logging
from datetime import datetime

import pytest
from nhx.common.api.utils import (
    HealthCheckFilter,
    _apply_comparison_operators,
    filter_by_created_at,
    filter_health_checks,
    filter_match,
    parse_deep_object,
    split_named_entity_urn,
)
from starlette.datastructures import QueryParams


class TestSplitNamedEntityUrn:
    """Test cases for split_named_entity_urn function."""

    def test_valid_urn_simple(self):
        """Test splitting a valid URN with namespace and name."""
        result = split_named_entity_urn("my-namespace/my-name")
        assert result == ["my-namespace", "my-name"]

    def test_valid_urn_with_slashes_in_name(self):
        """Test URN where name contains additional slashes."""
        result = split_named_entity_urn("namespace/name/with/slashes")
        assert result == ["namespace", "name/with/slashes"]

    def test_valid_urn_special_characters(self):
        """Test URN with special characters."""
        result = split_named_entity_urn("my-ns_123/my.name-v2")
        assert result == ["my-ns_123", "my.name-v2"]

    def test_invalid_urn_no_slash(self):
        """Test that ValueError is raised for URN without slash."""
        with pytest.raises(ValueError) as exc_info:
            split_named_entity_urn("invalid-urn")
        assert "Required format is <namespace>/<name>" in str(exc_info.value)
        assert "invalid-urn" in str(exc_info.value)

    def test_invalid_urn_empty_string(self):
        """Test that ValueError is raised for empty string."""
        with pytest.raises(ValueError) as exc_info:
            split_named_entity_urn("")
        assert "Required format is <namespace>/<name>" in str(exc_info.value)

    def test_urn_with_empty_namespace(self):
        """Test URN where namespace is empty but slash exists."""
        result = split_named_entity_urn("/name")
        assert result == ["", "name"]

    def test_urn_with_empty_name(self):
        """Test URN where name is empty but slash exists."""
        result = split_named_entity_urn("namespace/")
        assert result == ["namespace", ""]


class TestFilterByCreatedAt:
    """Test cases for filter_by_created_at function."""

    def test_greater_than(self):
        """Test gt (greater than) comparison."""
        item_date = datetime(2024, 1, 15, 12, 0, 0)
        filter_value = {"gt": datetime(2024, 1, 10, 12, 0, 0)}
        assert filter_by_created_at(filter_value, item_date) is True

        filter_value = {"gt": datetime(2024, 1, 20, 12, 0, 0)}
        assert filter_by_created_at(filter_value, item_date) is False

    def test_greater_than_or_equal(self):
        """Test gte (greater than or equal) comparison."""
        item_date = datetime(2024, 1, 15, 12, 0, 0)

        # Equal case - should pass
        filter_value = {"gte": datetime(2024, 1, 15, 12, 0, 0)}
        assert filter_by_created_at(filter_value, item_date) is True

        # Greater case - should pass
        filter_value = {"gte": datetime(2024, 1, 10, 12, 0, 0)}
        assert filter_by_created_at(filter_value, item_date) is True

        # Less case - should fail
        filter_value = {"gte": datetime(2024, 1, 20, 12, 0, 0)}
        assert filter_by_created_at(filter_value, item_date) is False

    def test_less_than(self):
        """Test lt (less than) comparison."""
        item_date = datetime(2024, 1, 15, 12, 0, 0)
        filter_value = {"lt": datetime(2024, 1, 20, 12, 0, 0)}
        assert filter_by_created_at(filter_value, item_date) is True

        filter_value = {"lt": datetime(2024, 1, 10, 12, 0, 0)}
        assert filter_by_created_at(filter_value, item_date) is False

    def test_less_than_or_equal(self):
        """Test lte (less than or equal) comparison."""
        item_date = datetime(2024, 1, 15, 12, 0, 0)

        filter_value = {"lte": datetime(2024, 1, 15, 12, 0, 0)}
        assert filter_by_created_at(filter_value, item_date) is True

        filter_value = {"lte": datetime(2024, 1, 20, 12, 0, 0)}
        assert filter_by_created_at(filter_value, item_date) is True

        filter_value = {"lte": datetime(2024, 1, 10, 12, 0, 0)}
        assert filter_by_created_at(filter_value, item_date) is False

    def test_equal(self):
        """Test eq (equal) comparison."""
        item_date = datetime(2024, 1, 15, 12, 0, 0)
        filter_value = {"eq": datetime(2024, 1, 15, 12, 0, 0)}
        assert filter_by_created_at(filter_value, item_date) is True

        filter_value = {"eq": datetime(2024, 1, 16, 12, 0, 0)}
        assert filter_by_created_at(filter_value, item_date) is False

    def test_not_equal(self):
        """Test neq (not equal) comparison."""
        item_date = datetime(2024, 1, 15, 12, 0, 0)
        filter_value = {"neq": datetime(2024, 1, 16, 12, 0, 0)}
        assert filter_by_created_at(filter_value, item_date) is True

        filter_value = {"neq": datetime(2024, 1, 15, 12, 0, 0)}
        assert filter_by_created_at(filter_value, item_date) is False

    def test_multiple_conditions(self):
        """Test multiple conditions in one filter."""
        item_date = datetime(2024, 1, 15, 12, 0, 0)

        # Date is between bounds
        filter_value = {"gte": datetime(2024, 1, 10, 12, 0, 0), "lte": datetime(2024, 1, 20, 12, 0, 0)}
        assert filter_by_created_at(filter_value, item_date) is True

        # Date is outside bounds
        filter_value = {"gte": datetime(2024, 1, 16, 12, 0, 0), "lte": datetime(2024, 1, 20, 12, 0, 0)}
        assert filter_by_created_at(filter_value, item_date) is False

    def test_invalid_operator(self):
        """Test with invalid operator returns False."""
        item_date = datetime(2024, 1, 15, 12, 0, 0)
        filter_value = {"invalid_op": datetime(2024, 1, 10, 12, 0, 0)}
        assert filter_by_created_at(filter_value, item_date) is False

    def test_none_value(self):
        """Test with None value in filter."""
        item_date = datetime(2024, 1, 15, 12, 0, 0)
        filter_value = {"gt": None}
        # None values should be skipped (truthy condition)
        assert filter_by_created_at(filter_value, item_date) is True


class TestApplyComparisonOperators:
    """Test cases for _apply_comparison_operators function."""

    def test_numeric_comparisons(self):
        """Test comparison operators with numeric values."""
        assert _apply_comparison_operators(10, {"gt": 5}) is True
        assert _apply_comparison_operators(10, {"gt": 15}) is False
        assert _apply_comparison_operators(10, {"gte": 10}) is True
        assert _apply_comparison_operators(10, {"lt": 15}) is True
        assert _apply_comparison_operators(10, {"lt": 5}) is False
        assert _apply_comparison_operators(10, {"lte": 10}) is True
        assert _apply_comparison_operators(10, {"eq": 10}) is True
        assert _apply_comparison_operators(10, {"eq": 5}) is False
        assert _apply_comparison_operators(10, {"neq": 5}) is True
        assert _apply_comparison_operators(10, {"neq": 10}) is False

    def test_string_comparisons(self):
        """Test comparison operators with string values."""
        assert _apply_comparison_operators("banana", {"gt": "apple"}) is True
        assert _apply_comparison_operators("apple", {"lt": "banana"}) is True
        assert _apply_comparison_operators("apple", {"eq": "apple"}) is True
        assert _apply_comparison_operators("apple", {"neq": "banana"}) is True

    def test_datetime_comparisons(self):
        """Test comparison operators with datetime values."""
        date1 = datetime(2024, 1, 15)
        date2 = datetime(2024, 1, 20)

        assert _apply_comparison_operators(date1, {"lt": date2}) is True
        assert _apply_comparison_operators(date2, {"gt": date1}) is True
        assert _apply_comparison_operators(date1, {"eq": date1}) is True

    def test_datetime_from_iso_string(self):
        """Test that ISO datetime strings are converted for comparison."""
        item_value = "2024-01-15T12:00:00Z"
        filter_value = "2024-01-10T12:00:00Z"

        assert _apply_comparison_operators(item_value, {"gt": filter_value}) is True
        assert _apply_comparison_operators(item_value, {"lt": filter_value}) is False

    def test_in_operator(self):
        """Test 'in' operator."""
        assert _apply_comparison_operators("apple", {"in": ["apple", "banana", "cherry"]}) is True
        assert _apply_comparison_operators("grape", {"in": ["apple", "banana", "cherry"]}) is False
        assert _apply_comparison_operators(5, {"in": [1, 2, 3, 4, 5]}) is True

    def test_multiple_operators(self):
        """Test multiple operators combined."""
        # Value must be between 5 and 15
        assert _apply_comparison_operators(10, {"gt": 5, "lt": 15}) is True
        assert _apply_comparison_operators(10, {"gte": 10, "lte": 10}) is True
        assert _apply_comparison_operators(10, {"gt": 15, "lt": 20}) is False

    def test_non_datetime_strings_unchanged(self):
        """Test that non-datetime strings are compared as strings."""
        assert _apply_comparison_operators("hello", {"eq": "hello"}) is True
        assert _apply_comparison_operators("hello", {"neq": "world"}) is True


class TestFilterMatchEnhancements:
    """Test cases for enhanced filter_match functionality."""

    def test_comparison_operators_in_filter(self):
        """Test filter_match with comparison operators."""
        item = {"age": 25, "score": 85}

        # Greater than
        assert filter_match(item, {"age": {"gt": 20}}) is True
        assert filter_match(item, {"age": {"gt": 30}}) is False

        # Range
        assert filter_match(item, {"age": {"gte": 25, "lte": 30}}) is True
        assert filter_match(item, {"score": {"gt": 80, "lt": 90}}) is True

    def test_date_range_filters(self):
        """Test filter_match with start/end date ranges."""
        item = {"created": "2024-01-15"}

        # Within range
        assert filter_match(item, {"created": {"start": "2024-01-10", "end": "2024-01-20"}}) is True

        # Before range
        assert filter_match(item, {"created": {"start": "2024-01-20", "end": "2024-01-30"}}) is False

        # After range
        assert filter_match(item, {"created": {"start": "2024-01-01", "end": "2024-01-10"}}) is False

    def test_list_or_logic(self):
        """Test filter_match with list values (OR logic)."""
        item = {"status": "active"}

        # Match one of multiple values
        assert filter_match(item, {"status": ["active", "pending"]}) is True
        assert filter_match(item, {"status": ["inactive", "suspended"]}) is False

        # Empty list
        assert filter_match(item, {"status": []}) is False

    def test_in_operator(self):
        """Test filter_match with 'in' operator."""
        item = {"category": "electronics"}

        assert filter_match(item, {"category": {"in": ["electronics", "computers", "phones"]}}) is True
        assert filter_match(item, {"category": {"in": ["books", "toys"]}}) is False

    def test_nested_dict_recursive(self):
        """Test filter_match with nested dictionaries."""
        item = {"user": {"profile": {"age": 25}}}

        assert filter_match(item, {"user": {"profile": {"age": 25}}}) is True
        assert filter_match(item, {"user": {"profile": {"age": 30}}}) is False

    def test_string_parsed_as_json(self):
        """Test filter_match when item value is JSON string."""
        item = {"metadata": '{"key": "value"}'}

        assert filter_match(item, {"metadata": {"key": "value"}}) is True
        assert filter_match(item, {"metadata": {"key": "other"}}) is False

    def test_non_strict_mode(self):
        """Test filter_match in non-strict mode (substring matching)."""
        item = {"name": "Hello World"}

        # Substring match (case-insensitive)
        assert filter_match(item, {"name": "hello"}, strict=False) is True
        assert filter_match(item, {"name": "WORLD"}, strict=False) is True
        assert filter_match(item, {"name": "xyz"}, strict=False) is False

    def test_skip_asterisk_marker(self):
        """Test that asterisk key is skipped in filtering."""
        item = {"name": "test"}

        # Asterisk should be ignored
        assert filter_match(item, {"*": "raw", "name": "test"}) is True


class TestParseDeepObject:
    """Test cases for parse_deep_object function."""

    def test_simple_flat_params(self):
        """Test parsing simple flat query parameters."""
        params = QueryParams("filter[name]=test&filter[status]=active")
        result = parse_deep_object("filter", params)

        assert result == {"name": "test", "status": "active"}

    def test_nested_params(self):
        """Test parsing nested query parameters."""
        params = QueryParams("filter[user][name]=john&filter[user][age]=30")
        result = parse_deep_object("filter", params)

        assert result == {"user": {"name": "john", "age": "30"}}

    def test_json_encoded_values(self):
        """Test parsing JSON-encoded values."""
        params = QueryParams('filter[config]={"key":"value"}')
        result = parse_deep_object("filter", params)

        assert result == {"config": {"key": "value"}}

    def test_array_encoded_values(self):
        """Test parsing JSON array values."""
        params = QueryParams('filter[items]=["a","b","c"]')
        result = parse_deep_object("filter", params)

        assert result == {"items": ["a", "b", "c"]}

    def test_empty_value_as_none(self):
        """Test that empty string becomes None."""
        params = QueryParams("filter[optional]=")
        result = parse_deep_object("filter", params)

        assert result == {"optional": None}

    def test_multiple_values_same_key(self):
        """Test handling multiple values for same key (OR logic)."""
        params = QueryParams("filter[status]=active&filter[status]=pending")
        result = parse_deep_object("filter", params)

        # Should create a list for OR logic
        assert result == {"status": ["active", "pending"]}

    def test_invalid_json_raises_error(self):
        """Test that invalid JSON raises ValueError."""
        params = QueryParams("filter[config]={invalid json}")

        with pytest.raises(ValueError) as exc_info:
            parse_deep_object("filter", params)
        assert "Invalid filter value" in str(exc_info.value)

    def test_invalid_json_array_includes_helpful_error(self):
        """Test that invalid JSON array raises ValueError with helpful example."""
        params = QueryParams("filter[id][in]=[item1,item2]")

        with pytest.raises(ValueError) as exc_info:
            parse_deep_object("filter", params)
        error_message = str(exc_info.value)
        assert "Invalid filter value" in error_message
        assert "valid JSON with proper quoting" in error_message
        assert '["item1","item2"]' in error_message

    def test_nonexistent_top_level_name(self):
        """Test parsing when top-level name doesn't exist."""
        params = QueryParams("other[name]=test")
        result = parse_deep_object("filter", params)

        assert result is None

    def test_deeply_nested_params(self):
        """Test deeply nested query parameters."""
        params = QueryParams("filter[a][b][c][d]=deep")
        result = parse_deep_object("filter", params)

        assert result == {"a": {"b": {"c": {"d": "deep"}}}}

    def test_comma_delimited_values(self):
        """Test parsing comma-delimited values into a list."""
        params = QueryParams("filter[status]=pending,active")
        result = parse_deep_object("filter", params)

        # Should split comma-delimited values into a list
        assert result == {"status": ["pending", "active"]}

    def test_comma_delimited_with_multiple_values(self):
        """Test parsing multiple comma-delimited values."""
        params = QueryParams("filter[status]=pending,active,completed")
        result = parse_deep_object("filter", params)

        assert result == {"status": ["pending", "active", "completed"]}

    def test_single_value_no_comma(self):
        """Test that single values without commas remain as single values."""
        params = QueryParams("filter[status]=active")
        result = parse_deep_object("filter", params)

        # Should remain a single value, not a list
        assert result == {"status": "active"}


class TestHealthCheckFilter:
    """Test cases for HealthCheckFilter class."""

    def test_filters_get_health_requests(self):
        """Test that GET /health requests are filtered."""
        health_filter = HealthCheckFilter()

        # Create a mock LogRecord for GET /health
        record = logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="",
            args=("127.0.0.1", "GET", "/health"),
            exc_info=None,
        )

        assert health_filter.filter(record) is False

    def test_allows_non_health_requests(self):
        """Test that non-health requests pass through."""
        health_filter = HealthCheckFilter()

        # Create a mock LogRecord for GET /api/users
        record = logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="",
            args=("127.0.0.1", "GET", "/api/users"),
            exc_info=None,
        )

        assert health_filter.filter(record) is True

    def test_allows_post_health_requests(self):
        """Test that POST /health requests pass through (only filters GET)."""
        health_filter = HealthCheckFilter()

        record = logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="",
            args=("127.0.0.1", "POST", "/health"),
            exc_info=None,
        )

        assert health_filter.filter(record) is True

    def test_allows_non_uvicorn_access_logs(self):
        """Test that non-uvicorn.access logs pass through."""
        health_filter = HealthCheckFilter()

        record = logging.LogRecord(
            name="my.logger",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="",
            args=("127.0.0.1", "GET", "/health"),
            exc_info=None,
        )

        assert health_filter.filter(record) is True

    def test_handles_insufficient_args(self):
        """Test that records with insufficient args pass through."""
        health_filter = HealthCheckFilter()

        record = logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="",
            args=("127.0.0.1",),  # Only 1 arg instead of 3
            exc_info=None,
        )

        assert health_filter.filter(record) is True

    def test_health_endpoint_variations(self):
        """Test various health endpoint patterns."""
        health_filter = HealthCheckFilter()

        # /health with query params
        record = logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="",
            args=("127.0.0.1", "GET", "/health?check=full"),
            exc_info=None,
        )
        assert health_filter.filter(record) is False

        # /api/health
        record.args = ("127.0.0.1", "GET", "/api/health")
        assert health_filter.filter(record) is False

        # /healthz (also contains /health substring, so filtered)
        record.args = ("127.0.0.1", "GET", "/healthz")
        assert health_filter.filter(record) is False  # Also filtered

        # /status (filtered like health endpoints)
        record.args = ("127.0.0.1", "GET", "/status")
        assert health_filter.filter(record) is False


class TestFilterHealthChecks:
    """Test cases for filter_health_checks function."""

    def test_adds_filter_when_disabled(self, monkeypatch):
        """Test that filter is added when health logging is disabled."""
        monkeypatch.setenv("LOG_HEALTH_ENDPOINTS", "")

        logger = logging.getLogger("uvicorn.access")
        # Clear any existing filters
        logger.filters = []

        filter_health_checks(logger, log_enabled=False)

        # Should have added the HealthCheckFilter
        assert len(logger.filters) > 0
        assert isinstance(logger.filters[-1], HealthCheckFilter)

    def test_no_filter_when_enabled(self, monkeypatch):
        """Test that no filter is added when health logging is enabled."""
        monkeypatch.setenv("LOG_HEALTH_ENDPOINTS", "true")

        logger = logging.getLogger("uvicorn.access")
        initial_filter_count = len(logger.filters)

        filter_health_checks(logger, log_enabled=True)

        # Should not have added any new filters
        assert len(logger.filters) == initial_filter_count
