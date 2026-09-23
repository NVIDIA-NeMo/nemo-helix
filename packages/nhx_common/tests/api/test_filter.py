# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for nhx.common.api.filter — parsing and serialization."""

import pytest
from nhx.common.api.filter import (
    ComparisonOperation,
    FilterOperator,
    LogicalOperation,
    parse_json_filter,
)


class TestLogicalOperationToDict:
    def test_and_produces_list(self):
        op = LogicalOperation(
            operator=FilterOperator.AND,
            operations=[
                ComparisonOperation(operator=FilterOperator.EQ, field="a", value=1),
                ComparisonOperation(operator=FilterOperator.EQ, field="b", value=2),
            ],
        )
        assert op.to_dict() == {"$and": [{"a": {"$eq": 1}}, {"b": {"$eq": 2}}]}

    def test_or_produces_list(self):
        op = LogicalOperation(
            operator=FilterOperator.OR,
            operations=[
                ComparisonOperation(operator=FilterOperator.EQ, field="a", value=1),
                ComparisonOperation(operator=FilterOperator.EQ, field="b", value=2),
            ],
        )
        assert op.to_dict() == {"$or": [{"a": {"$eq": 1}}, {"b": {"$eq": 2}}]}

    def test_not_produces_dict_not_list(self):
        """$not must serialize to a dict, not a list, so the parser can re-read it."""
        op = LogicalOperation(
            operator=FilterOperator.NOT,
            operations=[
                ComparisonOperation(operator=FilterOperator.EQ, field="data.api_endpoint", value=None),
            ],
        )
        result = op.to_dict()
        assert result == {"$not": {"data.api_endpoint": {"$eq": None}}}
        assert isinstance(result["$not"], dict), "$not value must be a dict, not a list"


class TestParseJsonFilterRoundtrip:
    def test_not_eq_null_roundtrip(self):
        """Regression: parse -> to_dict must produce re-parseable output for $not."""
        filter_json = '{"data.api_endpoint":{"$not":{"$eq":null}}}'
        op = parse_json_filter(filter_json)

        serialized = op.to_dict()
        assert serialized == {"$not": {"data.api_endpoint": {"$eq": None}}}

        # Re-parse the serialized form -- must not raise
        re_parsed = parse_json_filter(str(serialized).replace("None", "null").replace("'", '"'))
        assert re_parsed.to_dict() == serialized

    def test_not_nested_in_and_roundtrip(self):
        """Regression: when $not is nested inside $and the re-serialized form must be parseable."""
        import json

        op = parse_json_filter('{"data.api_endpoint":{"$not":{"$eq":null}}}')
        combined = {"$and": [{"data.project": "proj"}, op.to_dict()]}
        combined_json = json.dumps(combined)

        # Must not raise AttributeError: 'list' object has no attribute 'items'
        re_parsed = parse_json_filter(combined_json)
        assert re_parsed is not None

    def test_not_eq_roundtrip(self):
        filter_json = '{"name":{"$not":{"$eq":"llama"}}}'
        op = parse_json_filter(filter_json)
        assert op.to_dict() == {"$not": {"name": {"$eq": "llama"}}}

    def test_and_roundtrip(self):
        filter_json = '{"$and":[{"name":{"$like":"llama"}},{"data.description":{"$like":"llama"}}]}'
        op = parse_json_filter(filter_json)
        assert op.to_dict() == {
            "$and": [
                {"name": {"$like": "llama"}},
                {"data.description": {"$like": "llama"}},
            ]
        }

    def test_or_roundtrip(self):
        filter_json = '{"$or":[{"name":{"$eq":"a"}},{"name":{"$eq":"b"}}]}'
        op = parse_json_filter(filter_json)
        assert op.to_dict() == {"$or": [{"name": {"$eq": "a"}}, {"name": {"$eq": "b"}}]}

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="Invalid JSON"):
            parse_json_filter("not-valid-json")
