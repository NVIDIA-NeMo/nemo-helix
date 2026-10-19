# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Known Intake span attributes and their storage mapping.

This module is the source of truth for three related questions:

* which semantic span attributes Intake understands;
* which OTEL/OpenInference source keys can populate each attribute, in priority order;
* where each attribute is stored in ClickHouse attribute bags.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from enum import Enum, StrEnum
from typing import Any


# Storage bag names: the three ClickHouse Map columns used for span attributes.
class AttributeBag(StrEnum):
    STRING = "attributes_string"
    NUMBER = "attributes_number"
    BOOL = "attributes_bool"


# Semantic field names: the typed internal surface used by ingest and read APIs.
class SpanAttributeField(StrEnum):
    MODEL = "model"
    PROVIDER = "provider"
    PROMPT_ID = "prompt_id"
    AGENT_ID = "agent_id"
    AGENT_NAME = "agent_name"
    AGENT_VERSION = "agent_version"
    TOOL_NAME = "tool_name"
    PROJECT = "project"
    EVALUATION_NAME = "evaluation_name"
    TEST_CASE_NAME = "test_case_name"
    ERROR_TYPE = "error_type"
    ERROR_MESSAGE = "error_message"
    INPUT_TOKENS = "input_tokens"
    OUTPUT_TOKENS = "output_tokens"
    CACHED_TOKENS = "cached_tokens"
    TOTAL_TOKENS = "total_tokens"
    PROMPT_CACHE_WRITE_TOKENS = "prompt_cache_write_tokens"
    PROMPT_AUDIO_TOKENS = "prompt_audio_tokens"
    COMPLETION_REASONING_TOKENS = "completion_reasoning_tokens"
    COMPLETION_AUDIO_TOKENS = "completion_audio_tokens"
    COST_TOTAL_USD = "cost_total_usd"
    COST_INPUT_USD = "cost_input_usd"
    COST_OUTPUT_USD = "cost_output_usd"


# Catalog entry: source aliases -> typed semantic field -> canonical storage bag/key.
@dataclass(frozen=True)
class AttributeSpec:
    field: SpanAttributeField
    bag: AttributeBag
    bag_key: str
    source_keys: tuple[str, ...]
    bag_aliases: tuple[str, ...] = ()
    scale: int | None = None


# Cost is stored as scaled integer micros to preserve fractional precision through ClickHouse Map(Float64).
COST_SCALE = 1_000_000


# Source keys are ordered by precedence: GenAI first, then OpenInference/llm,
# then generic or legacy aliases.
ATTRIBUTE_SPECS = (
    AttributeSpec(
        field=SpanAttributeField.MODEL,
        bag=AttributeBag.STRING,
        bag_key="gen_ai.request.model",
        source_keys=(
            "gen_ai.request.model",
            "gen_ai.response.model",
            "llm.model_name",
        ),
    ),
    AttributeSpec(
        field=SpanAttributeField.PROVIDER,
        bag=AttributeBag.STRING,
        bag_key="gen_ai.system",
        source_keys=(
            "gen_ai.system",
            "gen_ai.provider.name",
            "llm.provider",
            "llm.system",
        ),
    ),
    AttributeSpec(
        field=SpanAttributeField.PROMPT_ID,
        bag=AttributeBag.STRING,
        bag_key="prompt.id",
        source_keys=(
            "gen_ai.prompt.id",
            "prompt.id",
        ),
    ),
    AttributeSpec(
        field=SpanAttributeField.AGENT_ID,
        bag=AttributeBag.STRING,
        bag_key="gen_ai.agent.id",
        source_keys=(
            "gen_ai.agent.id",
            "agent.id",
        ),
    ),
    AttributeSpec(
        field=SpanAttributeField.AGENT_NAME,
        bag=AttributeBag.STRING,
        bag_key="gen_ai.agent.name",
        source_keys=(
            "gen_ai.agent.name",
            "llm.agent.name",
            "agent.name",
        ),
    ),
    AttributeSpec(
        field=SpanAttributeField.AGENT_VERSION,
        bag=AttributeBag.STRING,
        bag_key="agent.version",
        source_keys=(
            "gen_ai.agent.version",
            "agent.version",
        ),
    ),
    AttributeSpec(
        field=SpanAttributeField.TOOL_NAME,
        bag=AttributeBag.STRING,
        bag_key="gen_ai.tool.name",
        source_keys=(
            "gen_ai.tool.name",
            "tool.name",
        ),
    ),
    AttributeSpec(
        field=SpanAttributeField.PROJECT,
        bag=AttributeBag.STRING,
        bag_key="project.name",
        source_keys=(
            "gen_ai.project",
            "project.name",
            "project",
        ),
    ),
    AttributeSpec(
        field=SpanAttributeField.EVALUATION_NAME,
        bag=AttributeBag.STRING,
        # Canonical key is ``nemo.evaluation.name`` (renamed from ``nemo.experiment.id`` in the
        # Experiment->Evaluation rename). The legacy ``nemo.experiment.id`` is still accepted on ingest
        # and normalized to the canonical key, so producers can migrate independently without dropping
        # their evaluation association.
        bag_key="nemo.evaluation.name",
        source_keys=("nemo.evaluation.name", "nemo.experiment.id"),
        bag_aliases=("nemo.experiment.id",),
    ),
    AttributeSpec(
        field=SpanAttributeField.TEST_CASE_NAME,
        bag=AttributeBag.STRING,
        bag_key="nemo.test_case.name",
        source_keys=("nemo.test_case.name", "nemo.test_case.id"),
        bag_aliases=("nemo.test_case.id",),
    ),
    AttributeSpec(
        field=SpanAttributeField.ERROR_TYPE,
        bag=AttributeBag.STRING,
        bag_key="exception.type",
        source_keys=("exception.type",),
    ),
    AttributeSpec(
        field=SpanAttributeField.ERROR_MESSAGE,
        bag=AttributeBag.STRING,
        bag_key="exception.message",
        source_keys=("exception.message",),
    ),
    AttributeSpec(
        field=SpanAttributeField.INPUT_TOKENS,
        bag=AttributeBag.NUMBER,
        bag_key="llm.token_count.prompt",
        source_keys=(
            "gen_ai.usage.input_tokens",
            "llm.token_count.prompt",
        ),
    ),
    AttributeSpec(
        field=SpanAttributeField.OUTPUT_TOKENS,
        bag=AttributeBag.NUMBER,
        bag_key="llm.token_count.completion",
        source_keys=(
            "gen_ai.usage.output_tokens",
            "llm.token_count.completion",
        ),
    ),
    AttributeSpec(
        field=SpanAttributeField.CACHED_TOKENS,
        bag=AttributeBag.NUMBER,
        bag_key="llm.token_count.cached",
        source_keys=(
            "gen_ai.usage.cached_tokens",
            "gen_ai.usage.input_cache_tokens",
            "llm.token_count.prompt_details.cache_read",
            "llm.token_count.cached",
        ),
    ),
    AttributeSpec(
        field=SpanAttributeField.TOTAL_TOKENS,
        bag=AttributeBag.NUMBER,
        bag_key="llm.token_count.total",
        source_keys=(
            "gen_ai.usage.total_tokens",
            "llm.token_count.total",
        ),
    ),
    AttributeSpec(
        field=SpanAttributeField.PROMPT_CACHE_WRITE_TOKENS,
        bag=AttributeBag.NUMBER,
        bag_key="llm.token_count.prompt_details.cache_write",
        source_keys=("llm.token_count.prompt_details.cache_write",),
    ),
    AttributeSpec(
        field=SpanAttributeField.PROMPT_AUDIO_TOKENS,
        bag=AttributeBag.NUMBER,
        bag_key="llm.token_count.prompt_details.audio",
        source_keys=("llm.token_count.prompt_details.audio",),
    ),
    AttributeSpec(
        field=SpanAttributeField.COMPLETION_REASONING_TOKENS,
        bag=AttributeBag.NUMBER,
        bag_key="llm.token_count.completion_details.reasoning",
        source_keys=("llm.token_count.completion_details.reasoning",),
    ),
    AttributeSpec(
        field=SpanAttributeField.COMPLETION_AUDIO_TOKENS,
        bag=AttributeBag.NUMBER,
        bag_key="llm.token_count.completion_details.audio",
        source_keys=("llm.token_count.completion_details.audio",),
    ),
    AttributeSpec(
        field=SpanAttributeField.COST_TOTAL_USD,
        bag=AttributeBag.NUMBER,
        bag_key="cost.total",
        source_keys=(
            "gen_ai.usage.cost",
            "llm.cost.total",
        ),
        scale=COST_SCALE,
    ),
    AttributeSpec(
        field=SpanAttributeField.COST_INPUT_USD,
        bag=AttributeBag.NUMBER,
        bag_key="cost.input",
        source_keys=("llm.cost.prompt",),
        scale=COST_SCALE,
    ),
    AttributeSpec(
        field=SpanAttributeField.COST_OUTPUT_USD,
        bag=AttributeBag.NUMBER,
        bag_key="cost.output",
        source_keys=("llm.cost.completion",),
        scale=COST_SCALE,
    ),
)

SPECS_BY_FIELD = {spec.field: spec for spec in ATTRIBUTE_SPECS}
SPECS_BY_FIELD_VALUE = {spec.field.value: spec for spec in ATTRIBUTE_SPECS}
SPECS_BY_BAG_KEY = {key: spec for spec in ATTRIBUTE_SPECS for key in (spec.bag_key, *spec.bag_aliases)}
KNOWN_BAG_KEYS = frozenset(SPECS_BY_BAG_KEY)
QUERYABLE_FIELDS = frozenset(SPECS_BY_FIELD_VALUE)

# Source keys that populate top-level span fields or payloads rather than
# semantic attribute bags.
SOURCE_ONLY_KEYS = frozenset(
    {
        "gen_ai.conversation.id",
        "input.value",
        "otel.status_code",
        "output.value",
        "session.id",
        "status",
    }
)
SOURCE_ONLY_PREFIXES = (
    "llm.input_messages.",
    "llm.output_messages.",
    "llm.tools.",
)
_COMPARISON_SQL = {
    "$eq": "=",
    "eq": "=",
    "=": "=",
    "$gt": ">",
    "gt": ">",
    ">": ">",
    "$lt": "<",
    "lt": "<",
    "<": "<",
    "$gte": ">=",
    "gte": ">=",
    ">=": ">=",
    "$lte": "<=",
    "lte": "<=",
    "<=": "<=",
}
_EQUALITY_OPERATORS = {"$eq", "eq", "="}


def spec_for_field(field: SpanAttributeField | str) -> AttributeSpec:
    if isinstance(field, SpanAttributeField):
        return SPECS_BY_FIELD[field]
    try:
        return SPECS_BY_FIELD[SpanAttributeField(field)]
    except ValueError as exc:
        raise ValueError(f"Unsupported span attribute field: {field}") from exc


def bag_keys(spec: AttributeSpec) -> tuple[str, ...]:
    """Return the canonical storage key followed by historical storage aliases."""

    return (spec.bag_key, *spec.bag_aliases)


def to_bag(typed_value: Any, spec: AttributeSpec) -> str | int | float | bool | None:
    if typed_value is None:
        return None
    value = to_semantic_value(typed_value, spec)
    if value is None:
        return None
    if spec.bag == AttributeBag.STRING:
        return str(value)
    if spec.bag == AttributeBag.BOOL:
        return bool(value)
    if spec.scale is not None:
        return scaled_decimal_to_int(value, scale=spec.scale)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    return float(value)


def from_bag(bag_value: Any, spec: AttributeSpec) -> str | int | float | bool | Decimal | None:
    if bag_value is None:
        return None
    if spec.bag == AttributeBag.STRING:
        value = str(bag_value)
        return value or None
    if spec.bag == AttributeBag.BOOL:
        return bool(bag_value)
    if spec.scale is not None:
        return Decimal(str(bag_value)) / Decimal(spec.scale)
    numeric = float(bag_value)
    return int(numeric) if numeric.is_integer() else numeric


def where_clause(
    field: str,
    operator: str | Enum,
    value: Any,
    *,
    param_prefix: str | None = None,
) -> tuple[str, dict[str, Any]]:
    spec = spec_for_field(field)

    operator_value = operator.value if isinstance(operator, Enum) else str(operator)
    sql_operator = _COMPARISON_SQL.get(operator_value)
    if sql_operator is None:
        raise ValueError(f"Unsupported span attribute filter operator: {operator_value}")
    if spec.bag != AttributeBag.NUMBER and operator_value not in _EQUALITY_OPERATORS:
        raise ValueError(f"Span attribute filter {field!r} only supports equality comparisons")

    param_root = param_prefix or field
    value_param = f"{param_root}_value"
    parsed_value = _parse_filter_value(value) if spec.bag == AttributeBag.NUMBER else value
    bag_value = to_bag(parsed_value, spec)
    if bag_value is None:
        raise ValueError(f"Span attribute filter {field!r} does not support null values")

    conditions: list[str] = []
    params = {value_param: bag_value}
    for index, key in enumerate(bag_keys(spec)):
        key_param = f"{param_root}_key" if index == 0 else f"{param_root}_key_alias_{index}"
        params[key_param] = key
        conditions.append(
            f"has(mapKeys({spec.bag.value}), %({key_param})s) "
            f"AND {spec.bag.value}[%({key_param})s] {sql_operator} %({value_param})s"
        )
    sql = conditions[0] if len(conditions) == 1 else "(" + ") OR (".join(conditions) + ")"
    return sql, params


def to_semantic_value(value: Any, spec: AttributeSpec) -> str | int | float | bool | Decimal | None:
    if value is None:
        return None
    if spec.bag == AttributeBag.STRING:
        text = str(value)
        return text or None
    if spec.bag == AttributeBag.BOOL:
        return bool(value)
    if spec.scale is not None:
        return _decimal(value)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid numeric value for {spec.field.value}: {value!r}") from exc
    return int(numeric) if numeric.is_integer() else numeric


def _parse_filter_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return Decimal(value)
        except InvalidOperation:
            return value
    return value


def _decimal(value: Any) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"Invalid decimal value: {value!r}") from exc


def scaled_decimal_to_int(value: Any, *, scale: int) -> int:
    decimal_value = _decimal(value)
    return int((decimal_value * Decimal(scale)).to_integral_exact())
