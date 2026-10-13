# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared ClickHouse expressions for span-derived token and cost rollups."""

from typing import Any

from nhx.intake.spans.span_attribute_catalog import COST_SCALE, SpanAttributeField, spec_for_field

METRIC_ATTRIBUTE_FIELDS = {
    "input_tokens": SpanAttributeField.INPUT_TOKENS,
    "output_tokens": SpanAttributeField.OUTPUT_TOKENS,
    "cached_tokens": SpanAttributeField.CACHED_TOKENS,
    "total_tokens": SpanAttributeField.TOTAL_TOKENS,
    "cost_usd": SpanAttributeField.COST_TOTAL_USD,
    "cost_input_usd": SpanAttributeField.COST_INPUT_USD,
    "cost_output_usd": SpanAttributeField.COST_OUTPUT_USD,
}


def metric_aggregate_columns(source_alias: str) -> tuple[str, dict[str, Any]]:
    """Build nullable sum expressions for canonical numeric span metrics."""

    parameters: dict[str, Any] = {}
    columns: list[str] = []
    for alias, field in METRIC_ATTRIBUTE_FIELDS.items():
        spec = spec_for_field(field)
        key_param = f"{alias}_key"
        parameters[key_param] = spec.bag_key
        number_bag = f"{source_alias}.attributes_number"
        has_expr = f"has(mapKeys({number_bag}), %({key_param})s)"
        sum_expr = f"sumIf({number_bag}[%({key_param})s], {has_expr})"
        value_expr = f"{sum_expr} / {COST_SCALE}" if spec.scale is not None else sum_expr
        columns.append(f"if(countIf({has_expr}) = 0, NULL, {value_expr}) AS {alias}")
    return ",\n            ".join(columns), parameters
