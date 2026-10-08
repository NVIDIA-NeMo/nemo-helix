# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from dataset_schemas.test_common import PATH_CASES
from nhx_evals_sdk.values.dataset_schemas import (
    _FIELD_MAPPING_PATH_PATTERN,
    FieldMapping,
    InputSchema,
)


def test_column_mapping_rejects_unknown_fields():
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        FieldMapping.model_validate({"input": "text", "unexpected": "value"})


def test_column_mapping_rejects_array_paths():
    with pytest.raises(ValueError):
        FieldMapping.model_validate({"messages": "conversations[].messages"})


def test_column_mapping_rejects_empty_paths():
    with pytest.raises(ValueError):
        FieldMapping.model_validate({"input": ""})

    with pytest.raises(ValueError):
        FieldMapping.model_validate({"custom": {"rubric": ""}})


def test_column_mapping_supports_trajectory_binding():
    mapping = FieldMapping.model_validate({"trajectory": "steps"})

    assert mapping.mapping()["trajectory"] == "steps"


def test_column_mapping_schema_constrains_every_path_field():
    """Every mapping field, including custom ones, carries the same path grammar."""
    schema = FieldMapping.model_json_schema()
    expected_pattern = _FIELD_MAPPING_PATH_PATTERN

    for field_name in ("input", "output", "context", "reference", "trajectory", "messages", "tool_calls", "tools"):
        any_of = schema["properties"][field_name]["anyOf"]
        constrained_branch = next(branch for branch in any_of if branch.get("type") == "string")
        assert constrained_branch["minLength"] == 1
        assert constrained_branch["pattern"] == expected_pattern

    assert schema["properties"]["custom"]["additionalProperties"]["minLength"] == 1
    assert schema["properties"]["custom"]["additionalProperties"]["pattern"] == expected_pattern


def test_required_input_schema_rejects_invalid_json_schema():
    with pytest.raises(ValueError, match="invalid JSON Schema"):
        InputSchema.model_validate({"schema": {"type": "definitely-not-a-valid-json-schema-type"}})


@pytest.mark.parametrize("path", [p for p, segments in PATH_CASES if segments is not None])
def test_column_mapping_accepts_a_positional_path(path: str) -> None:
    assert FieldMapping(input=path).mapping()["input"] == path


@pytest.mark.parametrize("path", [p for p, segments in PATH_CASES if segments is None])
def test_column_mapping_rejects_a_path_that_does_not_tokenize(path: str) -> None:
    with pytest.raises(ValueError):
        FieldMapping(input=path)


def test_column_mapping_explains_why_the_wildcard_is_rejected() -> None:
    """The wildcard parses but names no single element, so the validator -- not the pattern -- owns it."""
    with pytest.raises(ValueError, match=r"wildcard array segments"):
        FieldMapping(input="messages[].content")


def test_column_mapping_explains_why_a_predicate_cannot_be_chained() -> None:
    """`messages[role=assistant][1]` parses but reads like "the second assistant turn"; it is not."""
    with pytest.raises(ValueError, match=r"cannot be followed by another array segment"):
        FieldMapping(input="messages[role=assistant][1].text")
