# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest
from nhx.common.api.filter import ComparisonOperation, FilterOperator, LogicalOperation
from nhx.core.models.schemas import ModelEntityFilter

CAPABILITIES = "data.metadata.capabilities"
CONTEXT_LENGTH = "data.metadata.context_length"
MODEL_PROVIDERS = "data.model_providers"


def _translate(field: str, operator: FilterOperator, value: Any):
    return ModelEntityFilter.translate_operation(ComparisonOperation(operator=operator, field=field, value=value))


def _split(value: Any) -> Any:
    if isinstance(value, str):
        return value.split(",")
    return value


def _contains(value: str, field: str = CAPABILITIES) -> ComparisonOperation:
    return ComparisonOperation(operator=FilterOperator.CONTAINS, field=field, value=value)


def _contains_any(values: list[str], field: str = CAPABILITIES) -> ComparisonOperation:
    return ComparisonOperation(operator=FilterOperator.ELEM_MATCH, field=field, value={"$in": values})


def test_list_field_eq_scalar_becomes_contains():
    assert _translate("capabilities", FilterOperator.EQ, "reasoning") == _contains("reasoning")


def test_list_field_eq_list_requires_every_value():
    assert _translate("capabilities", FilterOperator.EQ, ["tool_calling", "reasoning"]) == LogicalOperation(
        operator=FilterOperator.AND, operations=[_contains("tool_calling"), _contains("reasoning")]
    )


def test_list_field_in_matches_any_value():
    assert _translate("capabilities", FilterOperator.IN, ["tool_calling", "reasoning"]) == _contains_any(
        ["tool_calling", "reasoning"]
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
        _contains_any(["embedding", "reasoning"])
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


@pytest.mark.parametrize("field", ["model_providers", MODEL_PROVIDERS])
def test_provider_field_matches_by_membership_under_either_name(field: str):
    assert _translate(field, FilterOperator.EQ, "default/a") == _contains("default/a", MODEL_PROVIDERS)
    assert _translate(field, FilterOperator.IN, ["default/a", "default/b"]) == _contains_any(
        ["default/a", "default/b"], MODEL_PROVIDERS
    )


def test_list_field_matches_by_membership_under_its_entity_path():
    assert _translate(CAPABILITIES, FilterOperator.EQ, "reasoning") == _contains("reasoning")


def test_provider_field_nin_keeps_models_without_a_provider_list():
    assert _translate("model_providers", FilterOperator.NIN, ["default/a", "default/b"]) == LogicalOperation(
        operator=FilterOperator.NOT, operations=[_contains_any(["default/a", "default/b"], MODEL_PROVIDERS)]
    )


@pytest.mark.parametrize("value", ["", 7, None, [], ["default/a", ""], "default/a,"])
def test_provider_field_rejects_invalid_values(value: Any):
    with pytest.raises(ValueError, match="for filter field 'model_providers'"):
        _translate("model_providers", FilterOperator.IN, value if isinstance(value, list) else _split(value))


@pytest.mark.parametrize("field", ["model_providers", MODEL_PROVIDERS])
def test_provider_field_rejects_other_operators(field: str):
    with pytest.raises(ValueError, match=rf"\$like is not supported for filter field '{field}'"):
        _translate(field, FilterOperator.LIKE, "default/a")


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
        (FilterOperator.GTE, 8191.5, 8192),
        (FilterOperator.GT, 8191.5, 8191),
        (FilterOperator.LT, "8192.5", 8193),
        (FilterOperator.LTE, "8192.5", 8192),
        (FilterOperator.LT, "8192.0000000000001", 8193),
        (FilterOperator.GTE, "-0.5", 0),
        (FilterOperator.GT, "8192.0", 8192),
        (FilterOperator.LTE, "1e5", 100000),
        (FilterOperator.EQ, "8192", 8192),
        (FilterOperator.EQ, 8192.0, 8192),
        (FilterOperator.EQ, "123456789012345678901", 123456789012345678901),
    ],
)
def test_number_field_sends_the_store_an_equivalent_integer(operator: FilterOperator, value: Any, expected: int):
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
        (FilterOperator.EQ, "8192.0000000000001"),
        (FilterOperator.GTE, "1e999999999"),
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
