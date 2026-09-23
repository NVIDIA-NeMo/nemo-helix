# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Span attribute catalog tests."""

import json
from decimal import Decimal

import pytest
from nhx.intake.spans.span_attribute_bags import SpanAttributeBags
from nhx.intake.spans.span_attribute_catalog import (
    ATTRIBUTE_SPECS,
    SPECS_BY_FIELD_VALUE,
    from_bag,
    to_bag,
    where_clause,
)
from nhx.intake.spans.span_semantic_attributes import SpanSemanticAttributes


@pytest.mark.parametrize("spec", ATTRIBUTE_SPECS)
def test_span_attribute_catalog_round_trips_entries(spec):
    value = Decimal("1.234567") if spec.scale is not None else _sample_value(spec.bag.value)

    assert from_bag(to_bag(value, spec), spec) == _expected_value(value, spec)


def test_span_attribute_catalog_scales_cost_to_micros():
    spec = SPECS_BY_FIELD_VALUE["cost_total_usd"]

    bag_value = to_bag(Decimal("1.234567"), spec)

    assert bag_value == 1_234_567
    assert from_bag(bag_value, spec) == Decimal("1.234567")


def test_span_attribute_catalog_preserves_fractional_scaled_bag_values():
    spec = SPECS_BY_FIELD_VALUE["cost_total_usd"]

    assert from_bag(1_234_567.8, spec) == Decimal("1.2345678")


def test_span_attribute_catalog_extracts_source_aliases_and_consumed_keys():
    semantic, consumed_keys = SpanSemanticAttributes.from_source_attributes(
        {
            "gen_ai.provider.name": "nvidia",
            "gen_ai.response.model": "model-a",
            "llm.provider": "legacy-provider",
            "llm.model_name": "legacy-model",
        }
    )

    assert semantic.provider == "nvidia"
    assert semantic.model == "model-a"
    assert consumed_keys == {
        "gen_ai.provider.name",
        "gen_ai.response.model",
        "llm.provider",
        "llm.model_name",
    }


def test_span_semantic_attributes_use_name_based_evaluation_fields():
    semantic, consumed_keys = SpanSemanticAttributes.from_source_attributes(
        {
            "nemo.evaluation.name": "evaluation-a",
            "nemo.test_case.name": "case-a",
        }
    )

    assert semantic.evaluation_name == "evaluation-a"
    assert semantic.test_case_name == "case-a"
    assert "evaluation_id" not in SpanSemanticAttributes.model_fields
    assert "test_case_id" not in SpanSemanticAttributes.model_fields
    assert consumed_keys == {"nemo.evaluation.name", "nemo.test_case.name"}

    bags = semantic.to_bags()
    assert bags.string["nemo.test_case.name"] == "case-a"
    assert "nemo.test_case.id" not in bags.string


def test_span_semantic_attributes_read_deprecated_test_case_bag_key():
    semantic, consumed_keys = SpanSemanticAttributes.from_source_attributes({"nemo.test_case.id": "case-a"})
    restored = SpanSemanticAttributes.from_bags(SpanAttributeBags(string={"nemo.test_case.id": "case-a"}))

    assert semantic.test_case_name == "case-a"
    assert restored.test_case_name == "case-a"
    assert consumed_keys == {"nemo.test_case.id"}


def test_span_semantic_attributes_normalizes_source_layers_with_span_precedence():
    normalized = SpanSemanticAttributes.from_source_attribute_layers(
        resource_attributes={
            "gen_ai.system": "resource-provider",
            "llm.model_name": "resource-model",
            "resource.custom": "resource-value",
        },
        span_attributes={
            "gen_ai.request.model": "span-model",
            "gen_ai.system": "span-provider",
            "span.custom": "span-value",
        },
    )

    assert normalized.semantic.model == "span-model"
    assert normalized.semantic.provider == "span-provider"
    assert normalized.source_attributes == {
        "gen_ai.system": "span-provider",
        "llm.model_name": "resource-model",
        "resource.custom": "resource-value",
        "gen_ai.request.model": "span-model",
        "span.custom": "span-value",
    }
    assert normalized.consumed_keys == {"gen_ai.request.model", "gen_ai.system", "llm.model_name"}


def test_span_attribute_bags_skip_consumed_source_aliases():
    bags = SpanAttributeBags()

    bags.put_unhandled_source_attributes(
        {
            "gen_ai.response.model": "model-a",
            "llm.model_name": "legacy-model",
            "custom": "value",
        },
        consumed_keys={"gen_ai.response.model", "llm.model_name"},
    )

    assert bags.string == {"custom": "value"}


def test_span_attribute_catalog_maps_bounded_openinference_token_details():
    semantic, consumed_keys = SpanSemanticAttributes.from_source_attributes(
        {
            "llm.token_count.prompt": 10,
            "llm.token_count.completion": 15,
            "llm.token_count.total": 25,
            "llm.token_count.prompt_details.cache_read": 5,
            "llm.token_count.prompt_details.cache_write": 7,
            "llm.token_count.prompt_details.audio": 3,
            "llm.token_count.completion_details.reasoning": 11,
            "llm.token_count.completion_details.audio": 2,
        }
    )

    assert semantic.input_tokens == 10
    assert semantic.output_tokens == 15
    assert semantic.total_tokens == 25
    assert semantic.cached_tokens == 5
    assert semantic.prompt_cache_write_tokens == 7
    assert semantic.prompt_audio_tokens == 3
    assert semantic.completion_reasoning_tokens == 11
    assert semantic.completion_audio_tokens == 2
    assert consumed_keys == {
        "llm.token_count.prompt",
        "llm.token_count.completion",
        "llm.token_count.total",
        "llm.token_count.prompt_details.cache_read",
        "llm.token_count.prompt_details.cache_write",
        "llm.token_count.prompt_details.audio",
        "llm.token_count.completion_details.reasoning",
        "llm.token_count.completion_details.audio",
    }

    bags = semantic.to_bags()
    restored_semantic = SpanSemanticAttributes.from_bags(bags)

    assert restored_semantic.prompt_cache_write_tokens == 7
    assert restored_semantic.prompt_audio_tokens == 3
    assert restored_semantic.completion_reasoning_tokens == 11
    assert restored_semantic.completion_audio_tokens == 2
    assert bags.raw_attributes_json() is None


def test_span_attribute_catalog_preserves_unknown_token_details_as_raw():
    source_attributes = {"llm.token_count.prompt_details.vendor_specific": 9}
    semantic, consumed_keys = SpanSemanticAttributes.from_source_attributes(source_attributes)
    bags = semantic.to_bags()

    bags.put_unhandled_source_attributes(source_attributes, consumed_keys=consumed_keys)

    assert consumed_keys == set()
    assert json.loads(bags.raw_attributes_json() or "{}") == {"llm.token_count.prompt_details.vendor_specific": 9.0}


@pytest.mark.parametrize("operator", ["=", ">", "<", ">=", "<="])
def test_span_attribute_catalog_predicates_include_existence_guard(operator: str):
    sql, params = where_clause("cost_total_usd", operator, Decimal("1.0"))

    assert "has(mapKeys(attributes_number)" in sql
    assert "attributes_number[" in sql
    assert operator in sql
    assert params["cost_total_usd_key"] == "cost.total"
    assert params["cost_total_usd_value"] == 1_000_000


def test_span_attribute_catalog_rejects_ordering_on_string_fields():
    with pytest.raises(ValueError, match="only supports equality"):
        where_clause("model", ">", "gpt-4")


def test_test_case_filter_queries_canonical_and_deprecated_bag_keys():
    sql, params = where_clause("test_case_name", "=", "case-a")

    assert " OR " in sql
    assert params["test_case_name_key"] == "nemo.test_case.name"
    assert params["test_case_name_key_alias_1"] == "nemo.test_case.id"
    assert params["test_case_name_value"] == "case-a"


def test_span_attribute_catalog_rejects_non_numeric_numeric_filters():
    with pytest.raises(ValueError):
        where_clause("total_tokens", ">", "not-a-number")


def _sample_value(bag: str) -> str | int | bool:
    if bag == "attributes_string":
        return "sample"
    if bag == "attributes_bool":
        return True
    return 42


def _expected_value(value, entry):
    if entry.scale is not None:
        return value
    if entry.bag == "attributes_number":
        return int(value)
    return value
