# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared schema, path, and validation primitives for evaluator dataset-schema helpers."""

from __future__ import annotations

import re
from typing import Any, Literal

from jsonschema.exceptions import SchemaError
from jsonschema.validators import validator_for

JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"
ARRAY_TOKEN = "[]"

#: A positional array segment such as ``[2]``. Distinct from :data:`ARRAY_TOKEN`, which is a
#: wildcard over an array of unknown length and resolves no concrete value.
_INDEX_TOKEN = re.compile(r"^\[([0-9]+)\]$")

#: Trailing bracket group on one dot segment, wildcard or positional.
_BRACKET_SUFFIX = re.compile(r"\[[0-9]*\]$")


class TemplateSchemaInferenceError(ValueError):
    """Raised when a template cannot be mapped to a canonical evaluator schema safely."""


class SchemaCompatibilityError(ValueError):
    """Raised when metric schemas cannot be merged or validated."""


class _MissingType:
    pass


_MISSING = _MissingType()


def validate_json_schema(schema: dict | None) -> None:
    """Validate a JSON Schema document if present.

    Raises `ValueError` when the schema is malformed according to the declared JSON Schema
    dialect.
    """
    if schema is None:
        return
    validator = validator_for(schema)
    try:
        validator.check_schema(schema)
    except SchemaError as e:
        raise ValueError(f"invalid JSON Schema: {e.message}") from e


def empty_object_schema() -> dict:
    return {"$schema": JSON_SCHEMA_DIALECT, "type": "object", "properties": {}, "required": []}


def encode_type(types: list[str]) -> str | list[str]:
    return types[0] if len(types) == 1 else types


def primitive_type_name(value: Any) -> Literal["boolean", "integer", "number", "string", "null"]:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    return "string"


def value_kind(value: Any) -> str:
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    return primitive_type_name(value)


def allowed_types(schema: dict) -> set[str]:
    """Return the set of JSON Schema types implied by a schema fragment.

    This includes inferred object/array kinds when ``type`` is omitted but
    ``properties`` or ``items`` are present.
    """
    schema_type = schema.get("type")
    if isinstance(schema_type, str):
        return {schema_type}
    if isinstance(schema_type, list):
        return {item for item in schema_type if isinstance(item, str)}
    if "properties" in schema:
        return {"object"}
    if "items" in schema:
        return {"array"}
    return set()


def schema_kind(schema: dict) -> Literal["any", "object", "array", "primitive"]:
    """Classify a schema fragment into the subset used by path traversal helpers.

    ``object`` and ``array`` include nullable unions (for example ``["object",
    "null"]``), while empty/untyped fragments are classified as ``any``.
    """
    allowed = allowed_types(schema)
    if not allowed:
        return "any"
    if "object" in allowed and allowed <= {"object", "null"}:
        return "object"
    if "array" in allowed and allowed <= {"array", "null"}:
        return "array"
    return "primitive"


def display_path(path: str) -> str:
    return path or "<root>"


def split_path(path: str) -> list[str]:
    """Split a dot path into segments, expanding array suffixes.

    Examples:
        "messages[].content" -> ["messages", "[]", "content"]
        "messages[][].content" -> ["messages", "[]", "[]", "content"]
        "messages[1].content" -> ["messages", "[1]", "content"]

    A bracket group that is neither a wildcard nor a non-negative integer, such as ``a[x]``, is
    left in the segment and treated as part of the key name.
    """
    if not path:
        return []

    parts: list[str] = []
    for segment in path.split("."):
        brackets: list[str] = []
        while (match := _BRACKET_SUFFIX.search(segment)) is not None:
            brackets.append(match.group(0))
            segment = segment[: match.start()]
        if segment:
            parts.append(segment)
        parts.extend(reversed(brackets))
    return parts


def get_value_at_path(data: dict[str, Any], path: str) -> Any:
    """Resolve a dotted path from an input row, returning ``_MISSING`` if absent.

    A positional segment such as ``messages[1]`` indexes a list. Anything that does not resolve --
    an out-of-range index, a non-list where an index was asked for, a missing key, or the ``[]``
    wildcard, which names no single element -- yields ``_MISSING``.
    """
    current: Any = data
    for segment in split_path(path):
        if segment == ARRAY_TOKEN:
            return _MISSING
        index_match = _INDEX_TOKEN.match(segment)
        if index_match is not None:
            # A list specifically, so that a string is never indexed character-wise.
            if not isinstance(current, list):
                return _MISSING
            index = int(index_match.group(1))
            if index >= len(current):
                return _MISSING
            current = current[index]
            continue
        if not isinstance(current, dict) or segment not in current:
            return _MISSING
        current = current[segment]
    return current


def get_schema_at_path(schema: dict, path: str) -> tuple[dict | None, bool]:
    """Resolve a schema fragment for a dotted path and whether it is required.

    Requiredness accumulates as traversal descends through object properties.
    Returns ``(None, False)`` when the path cannot be resolved from the schema.
    """
    current = schema
    required = True
    for segment in split_path(path):
        current_kind = schema_kind(current)
        # A positional segment describes one element, so it resolves to the same item schema the
        # wildcard does.
        if segment == ARRAY_TOKEN or _INDEX_TOKEN.match(segment) is not None:
            if current_kind != "array":
                return None, False
            items = current.get("items")
            if not isinstance(items, dict):
                return None, False
            current = items
            continue

        if current_kind != "object":
            return None, False
        properties = current.get("properties", {})
        if segment not in properties:
            return None, False
        required = required and segment in current.get("required", [])
        current = properties[segment]
    return current, required
