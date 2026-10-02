# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nemo_evaluator_sdk.dataset_schemas.common import (
    _MISSING,
    ARRAY_TOKEN,
    allowed_types,
    display_path,
    empty_object_schema,
    encode_type,
    get_schema_at_path,
    get_value_at_path,
    primitive_type_name,
    schema_kind,
    split_path,
    validate_json_schema,
    value_kind,
)


def test_validate_json_schema_rejects_invalid_schema():
    with pytest.raises(ValueError, match="invalid JSON Schema"):
        validate_json_schema({"type": "definitely-not-a-valid-json-schema-type"})


def test_empty_object_schema_uses_expected_shape():
    assert empty_object_schema() == {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {},
        "required": [],
    }


def test_split_path_expands_array_segments():
    assert split_path("messages[].content") == ["messages", ARRAY_TOKEN, "content"]


def test_split_path_preserves_nested_array_segments():
    assert split_path("messages[][].content") == ["messages", ARRAY_TOKEN, ARRAY_TOKEN, "content"]


def test_get_value_at_path_returns_missing_for_array_segments():
    assert get_value_at_path({"messages": [{"content": "hi"}]}, "messages[].content") is _MISSING


def test_get_schema_at_path_tracks_nested_required_state():
    schema, is_required = get_schema_at_path(
        {
            "type": "object",
            "properties": {
                "payload": {
                    "type": "object",
                    "properties": {"prompt": {"type": "string"}},
                    "required": ["prompt"],
                }
            },
            "required": ["payload"],
        },
        "payload.prompt",
    )

    assert schema == {"type": "string"}
    assert is_required is True


def test_schema_helpers_classify_common_types():
    assert encode_type(["string"]) == "string"
    assert encode_type(["null", "string"]) == ["null", "string"]
    assert primitive_type_name(True) == "boolean"
    assert primitive_type_name(3) == "integer"
    assert primitive_type_name(3.5) == "number"
    assert primitive_type_name(None) == "null"
    assert value_kind({"a": 1}) == "object"
    assert value_kind([1, 2]) == "array"
    assert allowed_types({"type": ["string", "null"]}) == {"string", "null"}
    assert schema_kind({"properties": {}}) == "object"
    assert schema_kind({"type": "null"}) == "primitive"
    assert schema_kind({"type": ["object", "null"]}) == "object"
    assert schema_kind({"type": ["array", "null"]}) == "array"
    assert display_path("") == "<root>"


#: One table drives both the tokenizer and the FieldMapping grammar, so the two cannot drift apart.
#: ``segments`` is ``None`` where the path is not a legal column mapping.
PATH_CASES: list[tuple[str, list[str] | None]] = [
    ("question", ["question"]),
    ("a.b.c", ["a", "b", "c"]),
    ("messages[1].content", ["messages", "[1]", "content"]),
    ("messages[0]", ["messages", "[0]"]),
    ("turns[0][2].text", ["turns", "[0]", "[2]", "text"]),
    ("messages[].content", None),
    ("messages[-1].content", None),
    ("messages[x].content", None),
    ("messages[1]content", None),
    ("messages[", None),
]


@pytest.mark.parametrize(("path", "segments"), [(p, s) for p, s in PATH_CASES if s is not None])
def test_split_path_tokenizes_positional_segments(path: str, segments: list[str]) -> None:
    assert split_path(path) == segments


def test_split_path_leaves_a_malformed_bracket_group_in_the_key() -> None:
    """``[x]`` is not an index, so it stays part of the key rather than becoming a token."""
    assert split_path("a[x].b") == ["a[x]", "b"]


_ROW = {
    "messages": [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ],
    "name": "flat",
    "nested": {"turns": [["a", "b"]]},
}


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("messages[0].content", "sys"),
        ("messages[1].content", "hi"),
        ("nested.turns[0][1]", "b"),
        ("name", "flat"),
    ],
)
def test_get_value_at_path_resolves_an_index(path: str, expected: str) -> None:
    assert get_value_at_path(_ROW, path) == expected


@pytest.mark.parametrize(
    "path",
    [
        "messages[5].content",  # out of range
        "name[0]",  # a string is not indexed character-wise
        "messages[0].absent",  # missing key past an index
        "messages[].content",  # the wildcard names no single element
    ],
)
def test_get_value_at_path_returns_missing_when_an_index_does_not_resolve(path: str) -> None:
    assert get_value_at_path(_ROW, path) is _MISSING


def test_get_schema_at_path_resolves_an_index_to_the_item_schema() -> None:
    """A positional segment describes one element, so it yields the same schema the wildcard does."""
    schema = {
        "type": "object",
        "properties": {
            "messages": {
                "type": "array",
                "items": {"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"]},
            }
        },
        "required": ["messages"],
    }

    indexed, indexed_required = get_schema_at_path(schema, "messages[1].content")
    wildcard, wildcard_required = get_schema_at_path(schema, "messages[].content")

    assert indexed == {"type": "string"}
    assert (indexed, indexed_required) == (wildcard, wildcard_required)
