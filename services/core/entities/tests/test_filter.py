# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for search parser.

Validates all examples from the search.py documentation.
"""

from unittest.mock import MagicMock

import pytest
from fastapi import Request
from nhx.common.api.filter import (
    ComparisonOperation,
    FilterOperator,
    FilterRepository,
    LogicalOperation,
)
from nhx.core.entities.utils.filter import (
    RelationshipFilterOperation,
    _parse_bracket_params,
    _parse_json_filter,
    make_filter_dep,
)
from starlette.datastructures import QueryParams


def create_mock_request(query_string: str) -> Request:
    """Create a mock FastAPI request with the given query string."""
    request = MagicMock(spec=Request)

    params = {}
    if query_string:
        for param in query_string.split("&"):
            if "=" in param:
                key, value = param.split("=", 1)
                params[key] = value

    request.query_params = QueryParams(params)
    return request


class TestParseJsonSearch:
    """Test JSON search parameter parsing."""

    def test_simple_equality(self):
        """Test: {"name":"llama"}"""
        result = _parse_json_filter('{"name":"llama"}')

        assert isinstance(result, ComparisonOperation)
        assert result.operator == FilterOperator.EQ
        assert result.field == "name"
        assert result.value == "llama"

    def test_multiple_fields(self):
        """Test: {"name":"llama","created_at":{"$lte":"2024-01-01"}}"""
        result = _parse_json_filter('{"name":"llama","created_at":{"$lte":"2024-01-01"}}')

        assert isinstance(result, LogicalOperation)
        assert result.operator == FilterOperator.AND
        assert len(result.operations) == 2

    def test_complex_not_and(self):
        """Test: {"$not": {"$and": [{"name":"llama"},{"name":"llama2"}]}}"""
        result = _parse_json_filter('{"$not": {"$and": [{"name":"llama"},{"name":"llama2"}]}}')

        assert isinstance(result, LogicalOperation)
        assert result.operator == FilterOperator.NOT
        assert len(result.operations) == 1

        and_op = result.operations[0]
        assert isinstance(and_op, LogicalOperation)
        assert and_op.operator == FilterOperator.AND
        assert len(and_op.operations) == 2

    def test_or_operator(self):
        """Test: {"$or":[{"name":"llama"},{"name":"llama2"}]}"""
        result = _parse_json_filter('{"$or":[{"name":"llama"},{"name":"llama2"}]}')

        assert isinstance(result, LogicalOperation)
        assert result.operator == FilterOperator.OR
        assert len(result.operations) == 2

    def test_all_comparison_operators(self):
        """Test all comparison operators."""
        operators = [
            ("$eq", FilterOperator.EQ),
            ("$like", FilterOperator.LIKE),
            ("$lt", FilterOperator.LT),
            ("$lte", FilterOperator.LTE),
            ("$gt", FilterOperator.GT),
            ("$gte", FilterOperator.GTE),
        ]

        for op_str, op_enum in operators:
            result = _parse_json_filter('{"field":{"' + op_str + '":"value"}}')
            assert isinstance(result, ComparisonOperation)
            assert result.operator == op_enum
            assert result.field == "field"
            assert result.value == "value"

    def test_in_nin_operators(self):
        """Test $in and $nin operators."""
        result = _parse_json_filter('{"field":{"$in":"val1,val2"}}')
        assert isinstance(result, ComparisonOperation)
        assert result.operator == FilterOperator.IN
        assert result.value == ["val1", "val2"]

        result = _parse_json_filter('{"field":{"$nin":"val1,val2"}}')
        assert isinstance(result, ComparisonOperation)
        assert result.operator == FilterOperator.NIN
        assert result.value == ["val1", "val2"]

    def test_invalid_json(self):
        """Test invalid JSON raises ValueError."""
        with pytest.raises(ValueError, match="Invalid JSON"):
            _parse_json_filter("{invalid json}")

    def test_empty_dict(self):
        """Test empty dictionary raises ValueError."""
        with pytest.raises(ValueError, match="No valid filter criteria"):
            _parse_json_filter("{}")


class TestParseBracketParams:
    """Test bracket-style search parameter parsing."""

    def test_simple_equality(self):
        """Test: filter[name]=llama"""
        result = _parse_bracket_params({"filter[name]": "llama"})

        assert isinstance(result, ComparisonOperation)
        assert result.operator == FilterOperator.EQ
        assert result.field == "name"
        assert result.value == "llama"

    def test_with_operator(self):
        """Test: filter[created_at][$lte]=2024-01-01"""
        result = _parse_bracket_params({"filter[created_at][$lte]": "2024-01-01"})

        assert isinstance(result, ComparisonOperation)
        assert result.operator == FilterOperator.LTE
        assert result.field == "created_at"
        assert result.value == "2024-01-01"

    def test_multiple_fields(self):
        """Test: filter[name]=llama&filter[created_at][$lte]=2024-01-01"""
        result = _parse_bracket_params(
            {
                "filter[name]": "llama",
                "filter[created_at][$lte]": "2024-01-01",
            }
        )

        assert isinstance(result, LogicalOperation)
        assert result.operator == FilterOperator.AND
        assert len(result.operations) == 2

    def test_in_operator(self):
        """Test: filter[name][$in]=llama,llama2"""
        result = _parse_bracket_params({"filter[name][$in]": "llama,llama2"})

        assert isinstance(result, ComparisonOperation)
        assert result.operator == FilterOperator.IN
        assert result.field == "name"
        assert result.value == ["llama", "llama2"]

    def test_or_operator(self):
        """Test: filter[$or]=[{"name":"llama"},{"name":"llama2"}]"""
        result = _parse_bracket_params({"filter[$or]": '[{"name":"llama"},{"name":"llama2"}]'})

        assert isinstance(result, LogicalOperation)
        assert result.operator == FilterOperator.OR
        assert len(result.operations) == 2

    def test_empty_params(self):
        """Test empty params raises ValueError."""
        with pytest.raises(ValueError, match="No filter parameters"):
            _parse_bracket_params({})


class TestMakeSearchDep:
    """Test the make_filter_dep dependency factory (backward compat alias for make_filter_dep)."""

    @pytest.mark.asyncio
    async def test_json_filter(self):
        """Test parsing JSON filter via dependency."""
        dep = make_filter_dep()
        request = create_mock_request('filter={"name":"llama"}')

        result = await dep(request, filter='{"name":"llama"}')

        assert isinstance(result, ComparisonOperation)
        assert result.field == "name"
        assert result.value == "llama"

    @pytest.mark.asyncio
    async def test_bracket_search(self):
        """Test parsing bracket search via dependency."""
        dep = make_filter_dep()
        request = create_mock_request("filter[name][$like]=llama")

        result = await dep(request, filter=None)

        assert isinstance(result, ComparisonOperation)
        assert result.operator == FilterOperator.LIKE
        assert result.field == "name"
        assert result.value == "llama"

    @pytest.mark.asyncio
    async def test_no_search(self):
        """Test no search params returns None."""
        dep = make_filter_dep()
        request = create_mock_request("")

        result = await dep(request, filter=None)

        assert result is None

    @pytest.mark.asyncio
    async def test_invalid_json_raises_http_exception(self):
        """Test invalid JSON raises HTTPException."""
        from fastapi import HTTPException

        dep = make_filter_dep()
        request = create_mock_request("")

        with pytest.raises(HTTPException) as exc_info:
            await dep(request, filter="{invalid}")

        assert exc_info.value.status_code == 400


class TestRepositoryApply:
    """Test that operations can apply to a repository."""

    def test_comparison_operations(self):
        """Test comparison operations apply correctly."""

        class MockRepository(FilterRepository):
            def eq(self, field, value):
                return f"eq({field}, {value})"

            def like(self, field, value):
                return f"like({field}, {value})"

            def lt(self, field, value):
                return f"lt({field}, {value})"

            def lte(self, field, value):
                return f"lte({field}, {value})"

            def gt(self, field, value):
                return f"gt({field}, {value})"

            def gte(self, field, value):
                return f"gte({field}, {value})"

            def in_op(self, field, values):
                return f"in({field}, {values})"

            def nin(self, field, values):
                return f"nin({field}, {values})"

            def and_op(self, operations):
                return f"and({operations})"

            def or_op(self, operations):
                return f"or({operations})"

            def not_op(self, operation):
                return f"not({operation})"

        repo = MockRepository()

        op = ComparisonOperation(operator=FilterOperator.EQ, field="name", value="test")
        assert op.apply(repo) == "eq(name, test)"

        op = ComparisonOperation(operator=FilterOperator.LIKE, field="name", value="test")
        assert op.apply(repo) == "like(name, test)"

    def test_relationship_operation_calls_repository(self):
        """Test RelationshipFilterOperation delegates to repository.relationship_exists."""
        from nhx.core.entities.utils.relationships import Relationship

        class MockRepository(FilterRepository):
            def eq(self, field, value):
                return f"eq({field}, {value})"

            def like(self, field, value):
                return f"like({field}, {value})"

            def lt(self, field, value):
                return f"lt({field}, {value})"

            def lte(self, field, value):
                return f"lte({field}, {value})"

            def gt(self, field, value):
                return f"gt({field}, {value})"

            def gte(self, field, value):
                return f"gte({field}, {value})"

            def in_op(self, field, values):
                return f"in({field}, {values})"

            def nin(self, field, values):
                return f"nin({field}, {values})"

            def and_op(self, operations):
                return f"and({operations})"

            def or_op(self, operations):
                return f"or({operations})"

            def not_op(self, operation):
                return f"not({operation})"

            def relationship_exists(self, target_entity_type, join_field, child_condition, negate):
                return f"rel_exists({target_entity_type}, {join_field}, cond={child_condition is not None}, negate={negate})"

        repo = MockRepository()
        rel = Relationship(kind="one_to_many", target_entity_type="adapter", via="parent")

        op = RelationshipFilterOperation(
            relationship_name="adapters",
            relationship=rel,
            condition=None,
            exists=True,
        )
        assert op.apply(repo) == "rel_exists(adapter, parent, cond=False, negate=False)"

        op_neg = RelationshipFilterOperation(
            relationship_name="adapters",
            relationship=rel,
            condition=None,
            exists=False,
        )
        assert op_neg.apply(repo) == "rel_exists(adapter, parent, cond=False, negate=True)"

        op_cond = RelationshipFilterOperation(
            relationship_name="adapters",
            relationship=rel,
            condition=ComparisonOperation(operator=FilterOperator.EQ, field="data.finetuning_type", value="LoRA"),
            exists=True,
        )
        assert op_cond.apply(repo) == "rel_exists(adapter, parent, cond=True, negate=False)"

    def test_logical_operations(self):
        """Test logical operations apply correctly."""

        class MockRepository(FilterRepository):
            def eq(self, field, value):
                return f"eq({field}, {value})"

            def like(self, field, value):
                return f"like({field}, {value})"

            def lt(self, field, value):
                return f"lt({field}, {value})"

            def lte(self, field, value):
                return f"lte({field}, {value})"

            def gt(self, field, value):
                return f"gt({field}, {value})"

            def gte(self, field, value):
                return f"gte({field}, {value})"

            def in_op(self, field, values):
                return f"in({field}, {values})"

            def nin(self, field, values):
                return f"nin({field}, {values})"

            def and_op(self, operations):
                return f"and({operations})"

            def or_op(self, operations):
                return f"or({operations})"

            def not_op(self, operation):
                return f"not({operation})"

        repo = MockRepository()

        op1 = ComparisonOperation(operator=FilterOperator.EQ, field="name", value="test1")
        op2 = ComparisonOperation(operator=FilterOperator.EQ, field="age", value=25)

        and_op = LogicalOperation(operator=FilterOperator.AND, operations=[op1, op2])
        assert and_op.apply(repo) == "and(['eq(name, test1)', 'eq(age, 25)'])"

        or_op = LogicalOperation(operator=FilterOperator.OR, operations=[op1, op2])
        assert or_op.apply(repo) == "or(['eq(name, test1)', 'eq(age, 25)'])"

        not_op = LogicalOperation(operator=FilterOperator.NOT, operations=[op1])
        assert not_op.apply(repo) == "not(eq(name, test1))"


class TestRelationshipJsonParsing:
    """Test relationship-aware JSON search parsing."""

    def test_exists_true(self):
        result = _parse_json_filter('{"adapters":{"$exists":true}}', entity_type="model")
        assert isinstance(result, RelationshipFilterOperation)
        assert result.relationship_name == "adapters"
        assert result.exists is True
        assert result.condition is None

    def test_exists_false(self):
        result = _parse_json_filter('{"adapters":{"$exists":false}}', entity_type="model")
        assert isinstance(result, RelationshipFilterOperation)
        assert result.exists is False
        assert result.condition is None

    def test_child_field_eq(self):
        result = _parse_json_filter('{"adapters":{"finetuning_type":"LoRA"}}', entity_type="model")
        assert isinstance(result, RelationshipFilterOperation)
        assert result.exists is True
        assert isinstance(result.condition, ComparisonOperation)
        assert result.condition.field == "data.finetuning_type"
        assert result.condition.value == "LoRA"

    def test_child_field_with_operator(self):
        result = _parse_json_filter(
            '{"adapters":{"finetuning_type":{"$in":["LoRA","P_TUNING"]}}}',
            entity_type="model",
        )
        assert isinstance(result, RelationshipFilterOperation)
        assert isinstance(result.condition, ComparisonOperation)
        assert result.condition.operator == FilterOperator.IN
        assert result.condition.field == "data.finetuning_type"

    def test_child_base_field_not_prefixed(self):
        """Base fields like 'name' should not get data. prefix."""
        result = _parse_json_filter('{"adapters":{"name":"my-adapter"}}', entity_type="model")
        assert isinstance(result, RelationshipFilterOperation)
        assert isinstance(result.condition, ComparisonOperation)
        assert result.condition.field == "name"

    def test_combined_with_regular_field(self):
        result = _parse_json_filter(
            '{"name":{"$like":"llama"},"adapters":{"$exists":true}}',
            entity_type="model",
        )
        assert isinstance(result, LogicalOperation)
        assert result.operator == FilterOperator.AND
        assert len(result.operations) == 2
        types = {type(op) for op in result.operations}
        assert ComparisonOperation in types
        assert RelationshipFilterOperation in types

    def test_unregistered_relationship_falls_through(self):
        """When entity_type has no 'foo' relationship, 'foo' is treated as a field path."""
        result = _parse_json_filter('{"foo":"bar"}', entity_type="model")
        assert isinstance(result, ComparisonOperation)
        assert result.field == "foo"
        assert result.value == "bar"

    def test_no_entity_type_skips_relationships(self):
        """Without entity_type, 'adapters' is treated as a regular field."""
        result = _parse_json_filter('{"adapters":"something"}', entity_type=None)
        assert isinstance(result, ComparisonOperation)
        assert result.field == "adapters"

    def test_exists_on_non_relationship_field_passes_through(self):
        """$exists on a non-relationship field passes through as a ComparisonOperation.

        Validation is deferred to the downstream service (entities) that
        understands relationships, rather than failing eagerly in the parser.
        """
        result = _parse_json_filter('{"name":{"$exists":true}}', entity_type="model")
        assert isinstance(result, ComparisonOperation)
        assert result.operator == FilterOperator.EXISTS
        assert result.field == "name"
        assert result.value is True

    def test_to_dict_exists(self):
        from nhx.core.entities.utils.relationships import Relationship

        op = RelationshipFilterOperation(
            relationship_name="adapters",
            relationship=Relationship(kind="one_to_many", target_entity_type="adapter", via="parent"),
            exists=False,
        )
        assert op.to_dict() == {"adapters": {"$exists": False}}

    def test_to_dict_condition(self):
        from nhx.core.entities.utils.relationships import Relationship

        op = RelationshipFilterOperation(
            relationship_name="adapters",
            relationship=Relationship(kind="one_to_many", target_entity_type="adapter", via="parent"),
            condition=ComparisonOperation(operator=FilterOperator.EQ, field="data.finetuning_type", value="LoRA"),
            exists=True,
        )
        assert op.to_dict() == {"adapters": {"data.finetuning_type": {"$eq": "LoRA"}}}


class TestRelationshipBracketParsing:
    """Test relationship-aware bracket search parsing."""

    def test_exists_true(self):
        result = _parse_bracket_params(
            {"filter[adapters][$exists]": "true"},
            entity_type="model",
        )
        assert isinstance(result, RelationshipFilterOperation)
        assert result.exists is True
        assert result.condition is None

    def test_exists_false(self):
        result = _parse_bracket_params(
            {"filter[adapters][$exists]": "false"},
            entity_type="model",
        )
        assert isinstance(result, RelationshipFilterOperation)
        assert result.exists is False

    def test_child_field_implicit_eq(self):
        result = _parse_bracket_params(
            {"filter[adapters][finetuning_type]": "LoRA"},
            entity_type="model",
        )
        assert isinstance(result, RelationshipFilterOperation)
        assert isinstance(result.condition, ComparisonOperation)
        assert result.condition.operator == FilterOperator.EQ
        assert result.condition.field == "data.finetuning_type"
        assert result.condition.value == "LoRA"

    def test_child_field_with_operator(self):
        result = _parse_bracket_params(
            {"filter[adapters][finetuning_type][$in]": "LoRA,P_TUNING"},
            entity_type="model",
        )
        assert isinstance(result, RelationshipFilterOperation)
        assert isinstance(result.condition, ComparisonOperation)
        assert result.condition.operator == FilterOperator.IN
        assert result.condition.field == "data.finetuning_type"
        assert result.condition.value == ["LoRA", "P_TUNING"]

    def test_child_base_field(self):
        result = _parse_bracket_params(
            {"filter[adapters][name][$like]": "my-adapter"},
            entity_type="model",
        )
        assert isinstance(result, RelationshipFilterOperation)
        assert isinstance(result.condition, ComparisonOperation)
        assert result.condition.field == "name"

    def test_combined_with_regular_field(self):
        result = _parse_bracket_params(
            {
                "filter[name][$like]": "llama",
                "filter[adapters][$exists]": "true",
            },
            entity_type="model",
        )
        assert isinstance(result, LogicalOperation)
        assert result.operator == FilterOperator.AND

    def test_no_entity_type_treats_as_field(self):
        """Without entity_type, 'adapters' is a regular field name."""
        result = _parse_bracket_params(
            {"filter[adapters]": "something"},
            entity_type=None,
        )
        assert isinstance(result, ComparisonOperation)
        assert result.field == "adapters"

    def test_missing_sub_bracket_errors(self):
        with pytest.raises(ValueError, match="requires at least one sub-bracket"):
            _parse_bracket_params(
                {"filter[adapters]": "something"},
                entity_type="model",
            )
