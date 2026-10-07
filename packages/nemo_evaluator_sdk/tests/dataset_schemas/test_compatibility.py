# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from textwrap import dedent

import pytest
from nemo_evaluator_sdk.dataset_schemas.common import SchemaCompatibilityError
from nemo_evaluator_sdk.dataset_schemas.compatibility import (
    apply_column_mapping_to_row,
    check_dataset_schema_compatibility,
    merge_metric_required_schemas,
    project_dataset_schema_for_column_mapping,
    prune_schema_properties,
    validate_dataset_schema_requirement,
    validate_prompt_template_against_dataset_schema,
)
from nemo_evaluator_sdk.values.dataset_schemas import FieldMapping


def test_validate_prompt_template_against_dataset_schema_uses_column_mapping():
    errors = validate_prompt_template_against_dataset_schema(
        {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        "Question: {{ input | trim }}",
        FieldMapping(input="text"),
    )

    assert errors == []


def test_validate_prompt_template_against_dataset_schema_can_ignore_runtime_roots():
    errors = validate_prompt_template_against_dataset_schema(
        {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        dedent(
            """
            {% for score_name, score in scores.items() %}
            {{ score_name }}
            {% endfor %}
            Question: {{ input }}
            """
        ).lstrip(),
        FieldMapping(input="text"),
        ignored_roots={"scores"},
    )

    assert errors == []


def test_validate_dataset_schema_requirement_reports_missing_projected_required_field():
    errors = validate_dataset_schema_requirement(
        {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        {
            "type": "object",
            "properties": {"input": {"type": "string"}, "output": {"type": "string"}},
            "required": ["input", "output"],
        },
        FieldMapping(input="text", output="answer"),
    )

    assert "output" in errors[0]


def test_validate_dataset_schema_requirement_reports_missing_mapped_dataset_field_explicitly():
    errors = validate_dataset_schema_requirement(
        {
            "type": "object",
            "properties": {"question": {"type": "string"}, "gold": {"type": "string"}},
            "required": ["question", "gold"],
        },
        {
            "type": "object",
            "properties": {"input": {"type": "string"}, "output": {"type": "string"}, "reference": {"type": "string"}},
            "required": ["input", "output", "reference"],
        },
        FieldMapping(input="question", output="missing_output", reference="gold"),
    )

    assert errors[0] == (
        "field_mapping.output refers to dataset field 'missing_output', but that field is not present in the dataset schema"
    )
    assert "dataset schema missing required field 'output'" not in errors


def test_dataset_schema_compatibility_allows_superset_nullable_and_integer_for_number():
    dataset_schema = {
        "type": "object",
        "properties": {
            "a": {"type": "integer"},
            "b": {"type": "string"},
            "d": {"type": "number"},
        },
        "required": ["a", "b", "d"],
        "additionalProperties": False,
    }
    required_schema = {
        "type": "object",
        "properties": {
            "a": {"type": "number"},
            "b": {"type": ["string", "null"]},
        },
        "required": ["a", "b"],
    }

    assert check_dataset_schema_compatibility(dataset_schema, required_schema) == []


def test_dataset_schema_compatibility_allows_missing_optional_property_definition():
    dataset_schema = {
        "type": "object",
        "properties": {
            "input": {"type": "string"},
        },
        "required": ["input"],
    }
    required_schema = {
        "type": "object",
        "properties": {
            "input": {"type": "string"},
            "reference": {"type": "string"},
        },
        "required": ["input"],
    }

    assert check_dataset_schema_compatibility(dataset_schema, required_schema) == []


def test_dataset_schema_compatibility_reports_missing_nested_field():
    dataset_schema = {
        "type": "object",
        "properties": {
            "messages": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"content": {"type": "string"}},
                    "required": ["content"],
                },
            }
        },
        "required": ["messages"],
    }
    required_schema = {
        "type": "object",
        "properties": {
            "messages": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "tool_calls": {"type": "array"},
                    },
                    "required": ["content", "tool_calls"],
                },
            }
        },
        "required": ["messages"],
    }

    errors = check_dataset_schema_compatibility(dataset_schema, required_schema)

    assert "messages[].tool_calls" in errors[0]


def test_dataset_schema_compatibility_rejects_nullable_object_for_required_object():
    dataset_schema = {
        "type": "object",
        "properties": {
            "metadata": {
                "type": ["object", "null"],
                "properties": {"request_id": {"type": "string"}},
                "required": ["request_id"],
            }
        },
        "required": ["metadata"],
    }
    required_schema = {
        "type": "object",
        "properties": {
            "metadata": {
                "type": "object",
                "properties": {"request_id": {"type": "string"}},
                "required": ["request_id"],
            }
        },
        "required": ["metadata"],
    }

    errors = check_dataset_schema_compatibility(dataset_schema, required_schema)

    assert errors == ["dataset field 'metadata' is incompatible: expected ['object'], found ['null', 'object']"]


def test_prune_schema_properties_removes_runtime_output():
    pruned = prune_schema_properties(
        {
            "type": "object",
            "properties": {"input": {}, "output": {}},
            "required": ["input", "output"],
        },
        {"output"},
    )

    assert pruned["required"] == ["input"]
    assert "output" not in pruned["properties"]


def test_project_dataset_schema_for_column_mapping_maps_dataset_columns_to_canonical_fields():
    projected = project_dataset_schema_for_column_mapping(
        {
            "type": "object",
            "properties": {"text": {"type": "string"}, "answer": {"type": "string"}},
            "required": ["text", "answer"],
        },
        {
            "type": "object",
            "properties": {"input": {"type": "string"}, "output": {"type": "string"}},
            "required": ["input", "output"],
        },
        FieldMapping(input="text", output="answer"),
    )

    assert projected["required"] == ["input", "output"]
    assert projected["properties"]["input"]["type"] == "string"
    assert projected["properties"]["output"]["type"] == "string"


def test_apply_column_mapping_to_row_adds_canonical_fields_for_nested_paths():
    row = {"payload": {"prompt": "Question?"}, "answer": "42"}

    mapped = apply_column_mapping_to_row(
        row,
        FieldMapping(input="payload.prompt", output="answer"),
    )

    assert mapped["input"] == "Question?"
    assert mapped["output"] == "42"


def test_merge_metric_required_schemas_widens_integer_and_number():
    merged = merge_metric_required_schemas(
        [
            ("m1", {"type": "object", "properties": {"score": {"type": "integer"}}, "required": ["score"]}),
            ("m2", {"type": "object", "properties": {"score": {"type": "number"}}, "required": ["score"]}),
        ]
    )

    assert merged["properties"]["score"]["type"] == "number"


def test_merge_metric_required_schemas_preserves_nullability_and_widens_numbers():
    merged = merge_metric_required_schemas(
        [
            (
                "m1",
                {"type": "object", "properties": {"score": {"type": ["integer", "null"]}}, "required": ["score"]},
            ),
            (
                "m2",
                {"type": "object", "properties": {"score": {"type": ["number", "null"]}}, "required": ["score"]},
            ),
        ]
    )

    assert merged["properties"]["score"]["type"] == ["null", "number"]


def test_merge_metric_required_schemas_rejects_incompatible_types():
    with pytest.raises(SchemaCompatibilityError, match="incompatible primitive schema types"):
        merge_metric_required_schemas(
            [
                ("m1", {"type": "object", "properties": {"score": {"type": "string"}}, "required": ["score"]}),
                ("m2", {"type": "object", "properties": {"score": {"type": "integer"}}, "required": ["score"]}),
            ]
        )


class TestColumnMappingOwnsItsField:
    """Canonical fields and dataset columns share one namespace, so a mapping must own its slot."""

    _MAPPING = FieldMapping(reference="messages[role=assistant].content")
    _NO_MATCH = [{"role": "user", "content": "q"}]
    _MATCH = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "answer"}]

    def test_a_miss_clears_a_dataset_column_of_the_same_name(self) -> None:
        """The case that silently scored the wrong value: the file supplies its own `reference`."""
        row = {"messages": self._NO_MATCH, "reference": "FROM THE FILE"}

        assert "reference" not in apply_column_mapping_to_row(row, self._MAPPING)

    def test_a_miss_leaves_the_field_unset_when_no_such_column_exists(self) -> None:
        row = {"messages": self._NO_MATCH}

        assert "reference" not in apply_column_mapping_to_row(row, self._MAPPING)

    def test_a_hit_replaces_a_dataset_column_of_the_same_name(self) -> None:
        """The mapping wins when it resolves, too -- not only when it fails."""
        row = {"messages": self._MATCH, "reference": "FROM THE FILE"}

        assert apply_column_mapping_to_row(row, self._MAPPING)["reference"] == "answer"

    def test_unmapped_dataset_columns_are_untouched(self) -> None:
        """Only the mapped canonical field is governed; the rest of the row passes through."""
        row = {"messages": self._NO_MATCH, "reference": "FROM THE FILE", "notes": "keep me"}

        mapped = apply_column_mapping_to_row(row, self._MAPPING)

        assert mapped["notes"] == "keep me"
        assert mapped["messages"] == self._NO_MATCH
