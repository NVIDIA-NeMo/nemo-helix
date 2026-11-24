# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest
from nhx.common.api.filter import ComparisonOperation, FilterOperator, LogicalOperation
from nhx.core.models.schemas import ModelEntityFilter

CAPABILITIES = "data.metadata.capabilities"
CONTEXT_LENGTH = "data.metadata.context_length"


def _translate(field: str, operator: FilterOperator, value: Any):
    return ModelEntityFilter.translate_operation(ComparisonOperation(operator=operator, field=field, value=value))


def _contains(value: str) -> ComparisonOperation:
    return ComparisonOperation(operator=FilterOperator.CONTAINS, field=CAPABILITIES, value=value)


def test_list_field_eq_scalar_becomes_contains():
    assert _translate("capabilities", FilterOperator.EQ, "reasoning") == _contains("reasoning")


def test_list_field_eq_list_requires_every_value():
    assert _translate("capabilities", FilterOperator.EQ, ["tool_calling", "reasoning"]) == LogicalOperation(
        operator=FilterOperator.AND, operations=[_contains("tool_calling"), _contains("reasoning")]
    )


def test_list_field_in_matches_any_value():
    assert _translate("capabilities", FilterOperator.IN, ["tool_calling", "reasoning"]) == LogicalOperation(
        operator=FilterOperator.OR, operations=[_contains("tool_calling"), _contains("reasoning")]
    )


def _present_and_not(operation: ComparisonOperation | LogicalOperation) -> LogicalOperation:
    is_null = ComparisonOperation(operator=FilterOperator.EQ, field=CAPABILITIES, value=None)
    return LogicalOperation(
        operator=FilterOperator.AND,
        operations=[
            LogicalOperation(operator=FilterOperator.NOT, operations=[is_null]),
            LogicalOperation(operator=FilterOperator.NOT, operations=[operation]),
        ],
    )


def test_list_field_nin_requires_the_field_and_negates_any_match():
    assert _translate("capabilities", FilterOperator.NIN, ["embedding", "reasoning"]) == _present_and_not(
        LogicalOperation(operator=FilterOperator.OR, operations=[_contains("embedding"), _contains("reasoning")])
    )


def test_list_field_nin_single_value():
    assert _translate("capabilities", FilterOperator.NIN, ["embedding"]) == _present_and_not(_contains("embedding"))


def test_list_field_rewrites_inside_logical_operations():
    operation = LogicalOperation(
        operator=FilterOperator.NOT,
        operations=[ComparisonOperation(operator=FilterOperator.EQ, field="capabilities", value=["reasoning"])],
    )
    assert ModelEntityFilter.translate_operation(operation) == LogicalOperation(
        operator=FilterOperator.NOT, operations=[_contains("reasoning")]
    )


@pytest.mark.parametrize(
    "operator", [FilterOperator.LIKE, FilterOperator.GTE, FilterOperator.STARTS_WITH, FilterOperator.ELEM_MATCH]
)
def test_list_field_rejects_other_operators(operator: FilterOperator):
    with pytest.raises(ValueError, match=rf"\{operator.value} is not supported for filter field 'input_modalities'"):
        _translate("input_modalities", operator, "text")


@pytest.mark.parametrize("value", [[], ["reasoning", "telepathy"], None])
def test_list_field_rejects_invalid_values(value: Any):
    with pytest.raises(ValueError, match="for filter field 'capabilities'"):
        _translate("capabilities", FilterOperator.IN, value)


@pytest.mark.parametrize(
    ("operator", "value", "expected"),
    [
        (FilterOperator.GTE, 8191.5, 8191.5),
        (FilterOperator.LT, "8192.5", 8192.5),
        (FilterOperator.GT, "8192.0", 8192),
        (FilterOperator.LTE, "1e5", 100000),
        (FilterOperator.EQ, "8192", 8192),
        (FilterOperator.EQ, 8192.0, 8192),
        (FilterOperator.EQ, "123456789012345678901", 123456789012345678901),
    ],
)
def test_number_field_keeps_value(operator: FilterOperator, value: Any, expected: int | float):
    translated = _translate("context_length", operator, value)
    assert translated == ComparisonOperation(operator=operator, field=CONTEXT_LENGTH, value=expected)
    assert isinstance(translated, ComparisonOperation)
    assert type(translated.value) is type(expected)


def test_number_field_in_coerces_each_value():
    assert _translate("context_length", FilterOperator.IN, ["8192", "1e5"]) == ComparisonOperation(
        operator=FilterOperator.IN, field=CONTEXT_LENGTH, value=[8192, 100000]
    )


def test_number_field_eq_list_becomes_in():
    assert _translate("context_length", FilterOperator.EQ, ["8192", "131072"]) == ComparisonOperation(
        operator=FilterOperator.IN, field=CONTEXT_LENGTH, value=[8192, 131072]
    )


@pytest.mark.parametrize(
    ("operator", "value"),
    [
        (FilterOperator.EQ, "8192.5"),
        (FilterOperator.IN, ["8192", 8192.5]),
        (FilterOperator.NIN, ["0.5"]),
        (FilterOperator.EQ, None),
        (FilterOperator.EQ, ""),
        (FilterOperator.EQ, True),
        (FilterOperator.GTE, None),
        (FilterOperator.GTE, float("inf")),
        (FilterOperator.GTE, "-inf"),
        (FilterOperator.LT, "nan"),
        (FilterOperator.LT, ["1"]),
        (FilterOperator.IN, []),
    ],
)
def test_number_field_rejects_invalid_values(operator: FilterOperator, value: Any):
    with pytest.raises(ValueError, match=r"Invalid value .* for filter field 'context_length'"):
        _translate("context_length", operator, value)


def test_number_field_rejects_other_operators():
    with pytest.raises(ValueError, match=r"\$like is not supported for filter field 'parameter_count'"):
        _translate("parameter_count", FilterOperator.LIKE, "8")
