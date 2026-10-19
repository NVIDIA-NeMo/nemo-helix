# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for text-based search query parser."""

import pytest
from lark.exceptions import UnexpectedCharacters, UnexpectedToken
from nemo_helix_plugin.api.filter import (
    ComparisonOperation,
    FilterOperator,
    LogicalOperation,
)
from nemo_helix_plugin.api.text_filter import (
    TextFilterParser,
    parse_text_filter,
)


def as_comparison(op: object) -> ComparisonOperation:
    """Type helper to cast to ComparisonOperation."""
    assert isinstance(op, ComparisonOperation)
    return op


def as_logical(op: object) -> LogicalOperation:
    """Type helper to cast to LogicalOperation."""
    assert isinstance(op, LogicalOperation)
    return op


class TestSimpleComparisons:
    """Test basic field:value comparisons."""

    def test_exact_match_quoted_string(self):
        """Test: name:"llama" """
        op = parse_text_filter('name:"llama"')
        assert isinstance(op, ComparisonOperation)
        assert op.operator == FilterOperator.EQ
        assert op.field == "name"
        assert op.value == "llama"

    def test_unquoted_value_rejected(self):
        """Test that unquoted string values are rejected."""
        with pytest.raises((UnexpectedCharacters, UnexpectedToken)):
            parse_text_filter("status:active")

    def test_like_match(self):
        """Test: email~"amy" """
        op = parse_text_filter('email~"amy"')
        assert isinstance(op, ComparisonOperation)
        assert op.operator == FilterOperator.LIKE
        assert op.field == "email"
        assert op.value == "amy"

    def test_greater_than(self):
        """Test: created>1620310503"""
        op = parse_text_filter("created>1620310503")
        assert isinstance(op, ComparisonOperation)
        assert op.operator == FilterOperator.GT
        assert op.field == "created"
        assert op.value == 1620310503

    def test_greater_than_or_equal(self):
        """Test: amount>=500"""
        op = parse_text_filter("amount>=500")
        assert isinstance(op, ComparisonOperation)
        assert op.operator == FilterOperator.GTE
        assert op.field == "amount"
        assert op.value == 500

    def test_less_than(self):
        """Test: count<10"""
        op = parse_text_filter("count<10")
        assert isinstance(op, ComparisonOperation)
        assert op.operator == FilterOperator.LT
        assert op.field == "count"
        assert op.value == 10

    def test_less_than_or_equal(self):
        """Test: priority<=5"""
        op = parse_text_filter("priority<=5")
        assert isinstance(op, ComparisonOperation)
        assert op.operator == FilterOperator.LTE
        assert op.field == "priority"
        assert op.value == 5

    def test_null_value(self):
        """Test: url:null"""
        op = parse_text_filter("url:null")
        assert isinstance(op, ComparisonOperation)
        assert op.operator == FilterOperator.EQ
        assert op.field == "url"
        assert op.value is None

    def test_null_case_insensitive(self):
        """Test: url:NULL"""
        op = parse_text_filter("url:NULL")
        assert isinstance(op, ComparisonOperation)
        assert op.value is None

    def test_float_value(self):
        """Test: price>19.99"""
        op = parse_text_filter("price>19.99")
        assert isinstance(op, ComparisonOperation)
        assert op.value == 19.99

    def test_negative_number(self):
        """Test: balance>-100"""
        op = parse_text_filter("balance>-100")
        assert isinstance(op, ComparisonOperation)
        assert op.value == -100


class TestFieldSyntax:
    """Test different field syntax patterns."""

    def test_dotted_field(self):
        """Test: data.nested.key:"value" """
        op = parse_text_filter('data.nested.key:"value"')
        assert isinstance(op, ComparisonOperation)
        assert op.field == "data.nested.key"
        assert op.value == "value"

    def test_metadata_field(self):
        """Test: metadata["key"]:"value" """
        op = parse_text_filter('metadata["key"]:"value"')
        assert isinstance(op, ComparisonOperation)
        assert op.field == "metadata.key"
        assert op.value == "value"

    def test_metadata_field_with_dots_in_key(self):
        """Test: metadata["nested.key"]:"value" """
        op = parse_text_filter('metadata["nested.key"]:"value"')
        assert isinstance(op, ComparisonOperation)
        assert op.field == "metadata.nested.key"

    def test_underscore_in_field(self):
        """Test: created_at>1000"""
        op = parse_text_filter("created_at>1000")
        assert isinstance(op, ComparisonOperation)
        assert op.field == "created_at"


class TestLogicalOperators:
    """Test AND and OR combinations."""

    def test_explicit_and(self):
        """Test: status:"active" AND amount>500"""
        op = as_logical(parse_text_filter('status:"active" AND amount>500'))
        assert op.operator == FilterOperator.AND
        assert len(op.operations) == 2

        assert as_comparison(op.operations[0]).field == "status"
        assert as_comparison(op.operations[0]).value == "active"
        assert as_comparison(op.operations[1]).field == "amount"
        assert as_comparison(op.operations[1]).value == 500

    def test_implicit_and(self):
        """Test: status:"active" amount>500 (implicit AND)"""
        op = parse_text_filter('status:"active" amount>500')
        assert isinstance(op, LogicalOperation)
        assert op.operator == FilterOperator.AND
        assert len(op.operations) == 2

    def test_or_operator(self):
        """Test: currency:"usd" OR currency:"eur" """
        op = as_logical(parse_text_filter('currency:"usd" OR currency:"eur"'))
        assert op.operator == FilterOperator.OR
        assert len(op.operations) == 2

        assert as_comparison(op.operations[0]).value == "usd"
        assert as_comparison(op.operations[1]).value == "eur"

    def test_and_case_insensitive(self):
        """Test: name:"a" and name:"b" """
        op = parse_text_filter('name:"a" and name:"b"')
        assert isinstance(op, LogicalOperation)
        assert op.operator == FilterOperator.AND

    def test_or_case_insensitive(self):
        """Test: name:"a" or name:"b" """
        op = parse_text_filter('name:"a" or name:"b"')
        assert isinstance(op, LogicalOperation)
        assert op.operator == FilterOperator.OR

    def test_multiple_and_clauses(self):
        """Test: a:1 AND b:2 AND c:3"""
        op = parse_text_filter("a:1 AND b:2 AND c:3")
        assert isinstance(op, LogicalOperation)
        assert op.operator == FilterOperator.AND
        assert len(op.operations) == 3

    def test_multiple_or_clauses(self):
        """Test: a:1 OR b:2 OR c:3"""
        op = parse_text_filter("a:1 OR b:2 OR c:3")
        assert isinstance(op, LogicalOperation)
        assert op.operator == FilterOperator.OR
        assert len(op.operations) == 3


class TestNegation:
    """Test negation operator."""

    def test_negated_comparison(self):
        """Test: -currency:"jpy" """
        op = parse_text_filter('-currency:"jpy"')
        assert isinstance(op, LogicalOperation)
        assert op.operator == FilterOperator.NOT
        assert len(op.operations) == 1

        inner = op.operations[0]
        assert isinstance(inner, ComparisonOperation)
        assert inner.field == "currency"
        assert inner.value == "jpy"

    def test_negated_group(self):
        """Test: -(name:"llama" AND status:"draft")"""
        op = parse_text_filter('-(name:"llama" AND status:"draft")')
        assert isinstance(op, LogicalOperation)
        assert op.operator == FilterOperator.NOT

        inner = op.operations[0]
        assert isinstance(inner, LogicalOperation)
        assert inner.operator == FilterOperator.AND


class TestGrouping:
    """Test parentheses for nested expressions."""

    def test_grouped_or_with_and(self):
        """Test: (name:"a" OR name:"b") AND status:"active" """
        op = parse_text_filter('(name:"a" OR name:"b") AND status:"active"')
        assert isinstance(op, LogicalOperation)
        assert op.operator == FilterOperator.AND
        assert len(op.operations) == 2

        # First operand should be the OR group
        or_group = op.operations[0]
        assert isinstance(or_group, LogicalOperation)
        assert or_group.operator == FilterOperator.OR

        # Second operand should be status comparison
        status_op = op.operations[1]
        assert isinstance(status_op, ComparisonOperation)
        assert status_op.field == "status"

    def test_nested_parentheses(self):
        """Test: ((a:1 OR b:2) AND c:3)"""
        op = parse_text_filter("((a:1 OR b:2) AND c:3)")
        assert isinstance(op, LogicalOperation)
        assert op.operator == FilterOperator.AND

    def test_operator_precedence_without_parens(self):
        """Test that AND has higher precedence than OR: a:1 OR b:2 AND c:3"""
        op = parse_text_filter("a:1 OR b:2 AND c:3")
        # Should parse as: a:1 OR (b:2 AND c:3)
        assert isinstance(op, LogicalOperation)
        assert op.operator == FilterOperator.OR
        assert len(op.operations) == 2

        # First operand is simple comparison
        assert isinstance(op.operations[0], ComparisonOperation)
        assert op.operations[0].field == "a"

        # Second operand is AND group
        and_group = op.operations[1]
        assert isinstance(and_group, LogicalOperation)
        assert and_group.operator == FilterOperator.AND


class TestStringEscaping:
    """Test string value handling."""

    def test_escaped_quote(self):
        """Test: name:"say \\"hello\\"" """
        op = parse_text_filter('name:"say \\"hello\\""')
        assert isinstance(op, ComparisonOperation)
        assert op.value == 'say "hello"'

    def test_string_with_spaces(self):
        """Test: name:"hello world" """
        op = parse_text_filter('name:"hello world"')
        assert isinstance(op, ComparisonOperation)
        assert op.value == "hello world"

    def test_quoted_email(self):
        """Test: email:"user@example.com" """
        op = parse_text_filter('email:"user@example.com"')
        assert isinstance(op, ComparisonOperation)
        assert op.value == "user@example.com"


class TestInOperator:
    """Test IN and NOT IN operators."""

    def test_in_with_strings(self):
        """Test: status IN ["active", "pending"]"""
        op = parse_text_filter('status IN ["active", "pending"]')
        assert isinstance(op, ComparisonOperation)
        assert op.operator == FilterOperator.IN
        assert op.field == "status"
        assert op.value == ["active", "pending"]

    def test_in_with_numbers(self):
        """Test: priority IN [1, 2, 3]"""
        op = parse_text_filter("priority IN [1, 2, 3]")
        assert isinstance(op, ComparisonOperation)
        assert op.operator == FilterOperator.IN
        assert op.field == "priority"
        assert op.value == [1, 2, 3]

    def test_in_with_mixed_types(self):
        """Test: field IN ["a", 1, 2.5]"""
        op = parse_text_filter('field IN ["a", 1, 2.5]')
        assert isinstance(op, ComparisonOperation)
        assert op.operator == FilterOperator.IN
        assert op.value == ["a", 1, 2.5]

    def test_in_with_single_value(self):
        """Test: status IN ["active"]"""
        op = parse_text_filter('status IN ["active"]')
        assert isinstance(op, ComparisonOperation)
        assert op.operator == FilterOperator.IN
        assert op.value == ["active"]

    def test_in_with_empty_list(self):
        """Test: status IN []"""
        op = parse_text_filter("status IN []")
        assert isinstance(op, ComparisonOperation)
        assert op.operator == FilterOperator.IN
        assert op.value == []

    def test_in_case_insensitive(self):
        """Test: status in ["active"]"""
        op = parse_text_filter('status in ["active"]')
        assert isinstance(op, ComparisonOperation)
        assert op.operator == FilterOperator.IN

    def test_not_in_with_strings(self):
        """Test: status NOT IN ["deleted", "archived"]"""
        op = parse_text_filter('status NOT IN ["deleted", "archived"]')
        assert isinstance(op, ComparisonOperation)
        assert op.operator == FilterOperator.NIN
        assert op.field == "status"
        assert op.value == ["deleted", "archived"]

    def test_not_in_case_insensitive(self):
        """Test: status not in ["deleted"]"""
        op = parse_text_filter('status not in ["deleted"]')
        assert isinstance(op, ComparisonOperation)
        assert op.operator == FilterOperator.NIN

    def test_in_with_null_values(self):
        """Test: field IN [null, "value"]"""
        op = parse_text_filter('field IN [null, "value"]')
        assert isinstance(op, ComparisonOperation)
        assert op.operator == FilterOperator.IN
        assert op.value == [None, "value"]

    def test_in_with_dotted_field(self):
        """Test: data.status IN ["a", "b"]"""
        op = parse_text_filter('data.status IN ["a", "b"]')
        assert isinstance(op, ComparisonOperation)
        assert op.field == "data.status"
        assert op.value == ["a", "b"]

    def test_in_combined_with_and(self):
        """Test: status IN ["active"] AND type:"user" """
        op = parse_text_filter('status IN ["active"] AND type:"user"')
        assert isinstance(op, LogicalOperation)
        assert op.operator == FilterOperator.AND
        assert len(op.operations) == 2

        in_op = as_comparison(op.operations[0])
        assert in_op.operator == FilterOperator.IN

        eq_op = as_comparison(op.operations[1])
        assert eq_op.operator == FilterOperator.EQ

    def test_in_combined_with_or(self):
        """Test: status IN ["active"] OR status:"legacy" """
        op = parse_text_filter('status IN ["active"] OR status:"legacy"')
        assert isinstance(op, LogicalOperation)
        assert op.operator == FilterOperator.OR

    def test_negated_in(self):
        """Test: -(status IN ["active"])"""
        op = parse_text_filter('-(status IN ["active"])')
        assert isinstance(op, LogicalOperation)
        assert op.operator == FilterOperator.NOT

        inner = as_comparison(op.operations[0])
        assert inner.operator == FilterOperator.IN


class TestDateComparisons:
    """Test date and datetime comparisons using ISO format strings."""

    def test_date_greater_than(self):
        """Test: created>"2024-01-01" """
        op = parse_text_filter('created>"2024-01-01"')
        assert isinstance(op, ComparisonOperation)
        assert op.operator == FilterOperator.GT
        assert op.field == "created"
        assert op.value == "2024-01-01"

    def test_date_greater_than_or_equal(self):
        """Test: created>="2024-01-01" """
        op = parse_text_filter('created>="2024-01-01"')
        assert isinstance(op, ComparisonOperation)
        assert op.operator == FilterOperator.GTE
        assert op.value == "2024-01-01"

    def test_date_less_than(self):
        """Test: updated<"2024-12-31" """
        op = parse_text_filter('updated<"2024-12-31"')
        assert isinstance(op, ComparisonOperation)
        assert op.operator == FilterOperator.LT
        assert op.field == "updated"
        assert op.value == "2024-12-31"

    def test_date_less_than_or_equal(self):
        """Test: expires<="2025-06-30" """
        op = parse_text_filter('expires<="2025-06-30"')
        assert isinstance(op, ComparisonOperation)
        assert op.operator == FilterOperator.LTE
        assert op.value == "2025-06-30"

    def test_datetime_with_timezone(self):
        """Test: created>="2024-01-01T10:00:00Z" """
        op = parse_text_filter('created>="2024-01-01T10:00:00Z"')
        assert isinstance(op, ComparisonOperation)
        assert op.operator == FilterOperator.GTE
        assert op.field == "created"
        assert op.value == "2024-01-01T10:00:00Z"

    def test_datetime_with_offset(self):
        """Test: created<"2024-06-15T14:30:00+05:00" """
        op = parse_text_filter('created<"2024-06-15T14:30:00+05:00"')
        assert isinstance(op, ComparisonOperation)
        assert op.operator == FilterOperator.LT
        assert op.value == "2024-06-15T14:30:00+05:00"

    def test_datetime_with_milliseconds(self):
        """Test: timestamp>"2024-01-01T00:00:00.123Z" """
        op = parse_text_filter('timestamp>"2024-01-01T00:00:00.123Z"')
        assert isinstance(op, ComparisonOperation)
        assert op.operator == FilterOperator.GT
        assert op.value == "2024-01-01T00:00:00.123Z"

    def test_date_range_with_and(self):
        """Test: created>"2024-01-01" AND created<"2024-12-31" """
        op = parse_text_filter('created>"2024-01-01" AND created<"2024-12-31"')
        assert isinstance(op, LogicalOperation)
        assert op.operator == FilterOperator.AND
        assert len(op.operations) == 2

        start = as_comparison(op.operations[0])
        assert start.operator == FilterOperator.GT
        assert start.field == "created"
        assert start.value == "2024-01-01"

        end = as_comparison(op.operations[1])
        assert end.operator == FilterOperator.LT
        assert end.field == "created"
        assert end.value == "2024-12-31"

    def test_date_exact_match(self):
        """Test: date:"2024-07-04" (exact date match)."""
        op = parse_text_filter('date:"2024-07-04"')
        assert isinstance(op, ComparisonOperation)
        assert op.operator == FilterOperator.EQ
        assert op.value == "2024-07-04"

    def test_unix_timestamp_comparison(self):
        """Test: created>1620310503 (Unix timestamp as integer)."""
        op = parse_text_filter("created>1620310503")
        assert isinstance(op, ComparisonOperation)
        assert op.operator == FilterOperator.GT
        assert op.field == "created"
        assert op.value == 1620310503

    def test_date_combined_with_other_filters(self):
        """Test: status:"active" AND created>"2024-01-01" """
        op = parse_text_filter('status:"active" AND created>"2024-01-01"')
        assert isinstance(op, LogicalOperation)
        assert op.operator == FilterOperator.AND

        status = as_comparison(op.operations[0])
        assert status.field == "status"
        assert status.value == "active"

        created = as_comparison(op.operations[1])
        assert created.field == "created"
        assert created.value == "2024-01-01"

    def test_date_in_list(self):
        """Test: date IN ["2024-01-01", "2024-07-04", "2024-12-25"]."""
        op = parse_text_filter('date IN ["2024-01-01", "2024-07-04", "2024-12-25"]')
        assert isinstance(op, ComparisonOperation)
        assert op.operator == FilterOperator.IN
        assert op.value == ["2024-01-01", "2024-07-04", "2024-12-25"]


class TestErrorHandling:
    """Test error cases."""

    def test_empty_query(self):
        """Test empty query raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            parse_text_filter("")

    def test_whitespace_only(self):
        """Test whitespace-only query raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            parse_text_filter("   ")

    def test_invalid_syntax(self):
        """Test invalid syntax raises parser error."""
        with pytest.raises((UnexpectedCharacters, UnexpectedToken)):
            parse_text_filter("!!invalid!!")


class TestParserInstance:
    """Test TextFilterParser class directly."""

    def test_parser_reuse(self):
        """Test parser can be reused for multiple queries."""
        parser = TextFilterParser()

        op1 = as_comparison(parser.parse('name:"a"'))
        op2 = as_comparison(parser.parse('name:"b"'))

        assert op1.value == "a"
        assert op2.value == "b"

    def test_module_parser_reuse(self):
        """Test module-level parser is shared."""
        op1 = as_comparison(parse_text_filter('x:"1"'))
        op2 = as_comparison(parse_text_filter('y:"2"'))

        assert op1.field == "x"
        assert op2.field == "y"
