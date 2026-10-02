# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nhx.common.api.filter import ComparisonOperation, FilterOperator, LogicalOperation
from nhx.common.entities.values import _apply_field_map


def test_apply_field_map_simple():
    op = ComparisonOperation(operator=FilterOperator.EQ, field="status", value="active")
    result = _apply_field_map(op, {"status": "data.status"})
    assert isinstance(result, ComparisonOperation)
    assert result.field == "data.status"
    assert result.value == "active"
    assert result.operator == FilterOperator.EQ


def test_apply_field_map_unmapped_field():
    op = ComparisonOperation(operator=FilterOperator.EQ, field="name", value="llama")
    result = _apply_field_map(op, {"status": "data.status"})
    assert isinstance(result, ComparisonOperation)
    assert result.field == "name"
    assert result is op  # should return same object when no mapping needed


def test_apply_field_map_logical():
    op = LogicalOperation(
        operator=FilterOperator.AND,
        operations=[
            ComparisonOperation(operator=FilterOperator.EQ, field="status", value="active"),
            ComparisonOperation(operator=FilterOperator.LIKE, field="name", value="llama"),
        ],
    )
    result = _apply_field_map(op, {"status": "data.status", "name": "data.name"})
    assert isinstance(result, LogicalOperation)
    assert result.operator == FilterOperator.AND
    assert result.operations[0].field == "data.status"
    assert result.operations[1].field == "data.name"


def test_apply_field_map_nested_logical():
    op = LogicalOperation(
        operator=FilterOperator.OR,
        operations=[
            LogicalOperation(
                operator=FilterOperator.AND,
                operations=[
                    ComparisonOperation(operator=FilterOperator.EQ, field="status", value="active"),
                    ComparisonOperation(operator=FilterOperator.GTE, field="score", value=0.9),
                ],
            ),
            ComparisonOperation(operator=FilterOperator.EQ, field="name", value="default"),
        ],
    )
    result = _apply_field_map(op, {"status": "data.status", "score": "data.score"})
    assert isinstance(result, LogicalOperation)
    inner = result.operations[0]
    assert isinstance(inner, LogicalOperation)
    assert inner.operations[0].field == "data.status"
    assert inner.operations[1].field == "data.score"
    # "name" not in field_map, should remain unchanged
    assert result.operations[1].field == "name"


def test_apply_field_map_empty_map():
    op = ComparisonOperation(operator=FilterOperator.EQ, field="status", value="active")
    result = _apply_field_map(op, {})
    assert result is op  # no mapping, same object returned


def test_apply_field_map_preserves_operator_and_value():
    op = ComparisonOperation(operator=FilterOperator.IN, field="status", value=["a", "b"])
    result = _apply_field_map(op, {"status": "data.status"})
    assert isinstance(result, ComparisonOperation)
    assert result.operator == FilterOperator.IN
    assert result.value == ["a", "b"]
    assert result.field == "data.status"
