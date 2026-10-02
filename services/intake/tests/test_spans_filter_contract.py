# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every span filter Intake publishes must reach SQL.

``SpanFilter`` is the published contract, so a field in it that cannot be served is a
promise the API does not keep. Five of them once resolved to span attributes with no
catalog entry, so ``spec_for_field`` raised ``ValueError`` deep in the repository and the
request failed with HTTP 500 and the body "An unexpected error occurred."

These tests walk the whole chain that broke: filter JSON, then the API filter builder,
then the SQL builder.
"""

from typing import Any

import pytest
from nhx.common.api.filter import parse_json_filter
from nhx.common.api.parsed_filter import ParsedFilter
from nhx.intake.repository.clickhouse.span import _span_where
from nhx.intake.spans.api.spans import ATTRIBUTE_EQ_FILTER_FIELD_MAP, ATTRIBUTE_EQ_FILTER_FIELDS, _span_filter
from nhx.intake.spans.api.spans_schemas import SpanFilter
from nhx.intake.spans.span_attribute_catalog import spec_for_field

# A value each field accepts, so a failure means the field and never the value.
FILTER_VALUES: dict[str, Any] = {
    "kind": "LLM",
    "status": "success",
    "started_at": {"$gte": "2026-01-01T00:00:00"},
}
DEFAULT_VALUE = "value-a"

# Fields that were published for months and answered with HTTP 500, because Intake never
# stores them: SpanSemanticAttributes has no dataset or prompt-name field.
NEVER_STORED = frozenset({"dataset_id", "dataset_name", "dataset_version", "prompt_name", "prompt_version"})


def build_where(field: str) -> tuple[str, dict[str, Any]]:
    operation = parse_json_filter(f'{{"{field}": {_json_value(field)}}}')
    filters = _span_filter("workspace-a", ParsedFilter(operation=operation))
    return _span_where(filters)


def _json_value(field: str) -> str:
    value = FILTER_VALUES.get(field, DEFAULT_VALUE)
    if isinstance(value, dict):
        inner = ", ".join(f'"{key}": "{item}"' for key, item in value.items())
        return f"{{{inner}}}"
    return f'"{value}"'


@pytest.mark.parametrize("field", sorted(SpanFilter.model_fields))
def test_every_published_filter_reaches_sql(field: str) -> None:
    # Without this, a field can pass schema validation and still raise while the SQL is
    # built, which reaches the caller as HTTP 500 instead of a rejected request.
    clause, parameters = build_where(field)

    assert clause
    assert parameters


@pytest.mark.parametrize("field", sorted(ATTRIBUTE_EQ_FILTER_FIELDS))
def test_every_attribute_filter_has_a_catalog_entry(field: str) -> None:
    # The repository resolves attribute filters through the catalog, so a field routed
    # there without an entry raises rather than returning a clean rejection.
    assert spec_for_field(ATTRIBUTE_EQ_FILTER_FIELD_MAP[field]) is not None


@pytest.mark.parametrize(
    ("canonical", "deprecated"),
    [("evaluation_name", "evaluation_id"), ("test_case_name", "test_case_id")],
)
def test_deprecated_identifier_filters_build_the_same_query(canonical: str, deprecated: str) -> None:
    assert build_where(canonical) == build_where(deprecated)


def test_identifier_filter_schema_marks_only_old_fields_deprecated() -> None:
    properties = SpanFilter.model_json_schema()["properties"]

    assert properties["evaluation_name"].get("deprecated") is not True
    assert properties["test_case_name"].get("deprecated") is not True
    assert properties["evaluation_id"]["deprecated"] is True
    assert properties["test_case_id"]["deprecated"] is True


def test_no_filter_field_is_published_without_a_way_to_serve_it() -> None:
    explicit = {"session_id", "trace_id", "parent_span_id", "source", "kind", "status", "started_at"}

    unserviceable = set(SpanFilter.model_fields) - explicit - set(ATTRIBUTE_EQ_FILTER_FIELDS)

    assert not unserviceable, (
        f"SpanFilter publishes {sorted(unserviceable)}, which _span_filter cannot route. "
        "Either handle the field or remove it from the published schema."
    )


def test_no_attribute_filter_is_routed_that_the_schema_does_not_publish() -> None:
    # The other direction of the same contract. ATTRIBUTE_EQ_FILTER_FIELDS is hand-kept,
    # so a name left behind after a schema change is routing that nothing can reach.
    orphaned = set(ATTRIBUTE_EQ_FILTER_FIELDS) - set(SpanFilter.model_fields)

    assert not orphaned, (
        f"ATTRIBUTE_EQ_FILTER_FIELDS routes {sorted(orphaned)}, which SpanFilter no longer publishes. "
        "Drop the name, or publish the field again."
    )


def test_fields_intake_never_stores_are_not_published() -> None:
    assert not NEVER_STORED & set(SpanFilter.model_fields)
    assert not NEVER_STORED & set(ATTRIBUTE_EQ_FILTER_FIELDS)
