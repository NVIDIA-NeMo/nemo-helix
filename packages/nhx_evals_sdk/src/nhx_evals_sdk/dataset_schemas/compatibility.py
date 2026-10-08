# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schema projection, compatibility checking, and merge helpers for evaluator inputs."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from nhx_evals_sdk.dataset_schemas.common import (
    _MISSING,
    JSON_SCHEMA_DIALECT,
    SchemaCompatibilityError,
    allowed_types,
    display_path,
    empty_object_schema,
    encode_type,
    get_schema_at_path,
    get_value_at_path,
    schema_kind,
    validate_json_schema,
)
from nhx_evals_sdk.dataset_schemas.templates import (
    infer_required_schema_from_template,
)
from nhx_evals_sdk.values.dataset_schemas import FieldMapping


def validate_dataset_schema_requirement(
    dataset_schema: dict,
    required_schema: dict,
    field_mapping: FieldMapping | None = None,
) -> list[str]:
    """Project a dataset schema into canonical fields and check compatibility.

    This is the main adapter entrypoint for comparing raw dataset schemas against the canonical
    evaluator field schema expected by a metric.
    """
    mapped_path_errors = _validate_column_mapping_paths(dataset_schema, required_schema, field_mapping)
    projected_dataset_schema = project_dataset_schema_for_column_mapping(dataset_schema, required_schema, field_mapping)
    compatibility_errors = check_dataset_schema_compatibility(projected_dataset_schema, required_schema)
    if not mapped_path_errors:
        return compatibility_errors

    assert field_mapping is not None
    missing_canonical_fields = {
        canonical_name
        for canonical_name, dataset_path in field_mapping.mapping().items()
        if get_schema_at_path(dataset_schema, dataset_path)[0] is None
    }
    filtered_errors = [
        error
        for error in compatibility_errors
        if not _is_redundant_missing_field_error(error, missing_canonical_fields)
    ]
    return [*mapped_path_errors, *filtered_errors]


def validate_prompt_template_against_dataset_schema(
    dataset_schema: dict,
    prompt_template: str | dict,
    field_mapping: FieldMapping | None = None,
    *,
    ignored_roots: set[str] | None = None,
    optional_fields: set[str] | None = None,
) -> list[str]:
    """Infer a schema from a prompt template and validate it against a dataset schema."""
    prompt_schema = infer_required_schema_from_template(
        prompt_template,
        ignored_roots=ignored_roots,
        optional_fields=optional_fields,
    )
    return validate_dataset_schema_requirement(dataset_schema, prompt_schema, field_mapping)


def merge_metric_required_schemas(named_schemas: Iterable[tuple[str, dict]]) -> dict:
    """Merge metric-required schemas for benchmark validation.

    Primitive `integer` and `number` requirements are widened to `number`. Other incompatible
    primitive type combinations raise `SchemaCompatibilityError`.
    """
    merged: dict | None = None
    for metric_name, schema in named_schemas:
        validate_json_schema(schema)
        merged = schema if merged is None else _merge_schema_nodes(merged, schema, metric_name, path="")

    if merged is None:
        merged = empty_object_schema()
    validate_json_schema(merged)
    return merged


def check_dataset_schema_compatibility(dataset_schema: dict, required_schema: dict) -> list[str]:
    """Return compatibility errors between a dataset schema and a metric-required schema."""
    validate_json_schema(dataset_schema)
    validate_json_schema(required_schema)
    return _check_schema_node(dataset_schema, required_schema, path="")


def prune_schema_properties(schema: dict, excluded_fields: set[str]) -> dict:
    """Drop top-level fields that are supplied at runtime rather than by the dataset."""
    if not excluded_fields:
        return schema
    if schema_kind(schema) != "object":
        return schema

    properties = schema.get("properties", {})
    required = schema.get("required", [])
    pruned = {
        "$schema": schema.get("$schema", JSON_SCHEMA_DIALECT),
        "type": schema.get("type", "object"),
        "properties": {name: value for name, value in properties.items() if name not in excluded_fields},
        "required": [name for name in required if name not in excluded_fields],
    }
    if "additionalProperties" in schema:
        pruned["additionalProperties"] = schema["additionalProperties"]
    return pruned


def project_dataset_schema_for_column_mapping(
    dataset_schema: dict,
    required_schema: dict,
    column_mapping: FieldMapping | None = None,
) -> dict:
    """Project a raw dataset schema into canonical evaluator field names.

    Each required evaluator field is resolved either directly by name or through the provided
    column mapping. Missing dataset paths are omitted from the projected schema so that the
    compatibility checker can produce the final user-facing errors.
    """
    validate_json_schema(dataset_schema)
    validate_json_schema(required_schema)

    if schema_kind(required_schema) != "object":
        return dataset_schema

    mapping = column_mapping.mapping() if column_mapping is not None else {}
    projected = empty_object_schema()
    for property_name in required_schema.get("properties", {}):
        dataset_path = mapping.get(property_name, property_name)
        dataset_property_schema, is_required = get_schema_at_path(dataset_schema, dataset_path)
        if dataset_property_schema is None:
            continue
        projected["properties"][property_name] = dataset_property_schema
        if property_name in required_schema.get("required", []) and is_required:
            projected["required"].append(property_name)

    if dataset_schema.get("additionalProperties") is False:
        projected["additionalProperties"] = False
    return projected


def _validate_column_mapping_paths(
    dataset_schema: dict,
    required_schema: dict,
    column_mapping: FieldMapping | None = None,
) -> list[str]:
    if schema_kind(required_schema) != "object" or column_mapping is None:
        return []

    errors: list[str] = []
    for canonical_name in required_schema.get("properties", {}):
        dataset_path = column_mapping.mapping().get(canonical_name)
        if dataset_path is None:
            continue
        dataset_property_schema, _ = get_schema_at_path(dataset_schema, dataset_path)
        if dataset_property_schema is None:
            errors.append(
                f"field_mapping.{canonical_name} refers to dataset field '{dataset_path}', but that field is not present in the dataset schema"
            )
    return errors


def _is_redundant_missing_field_error(error: str, missing_canonical_fields: set[str]) -> bool:
    for field_name in missing_canonical_fields:
        if error == f"dataset schema missing required field '{field_name}'":
            return True
        if error == f"dataset schema missing field definition '{field_name}'":
            return True
    return False


def apply_column_mapping_to_row(row: dict[str, Any], column_mapping: FieldMapping | None = None) -> dict[str, Any]:
    """Augment a dataset row with canonical evaluator fields.

    A mapped canonical field is set from its path, or left unset when the path resolves to nothing.
    It never falls back to a dataset column that happens to share the name: canonical fields and
    dataset columns occupy one namespace, so without this the mapping would silently fail to take
    effect and the column's own value would be scored in its place.
    """
    if column_mapping is None:
        return dict(row)

    mapped = dict(row)
    for canonical_name, dataset_path in column_mapping.mapping().items():
        value = get_value_at_path(row, dataset_path)
        if value is _MISSING:
            mapped.pop(canonical_name, None)
        else:
            mapped[canonical_name] = value
    return mapped


def _merge_schema_nodes(left: dict, right: dict, metric_name: str, path: str) -> dict:
    left_kind = schema_kind(left)
    right_kind = schema_kind(right)
    left_nullable = "null" in allowed_types(left)
    right_nullable = "null" in allowed_types(right)

    if left_kind == "any":
        return right
    if right_kind == "any":
        return left
    if left_kind != right_kind:
        raise SchemaCompatibilityError(
            f"benchmark metrics require incompatible schemas at '{display_path(path)}' for metric '{metric_name}': {left_kind} vs {right_kind}"
        )

    if left_kind == "object":
        merged_properties: dict[str, dict] = {}
        left_properties = left.get("properties", {})
        right_properties = right.get("properties", {})
        property_names = sorted(set(left_properties) | set(right_properties))
        for property_name in property_names:
            prop_path = f"{path}.{property_name}" if path else property_name
            if property_name in left_properties and property_name in right_properties:
                merged_properties[property_name] = _merge_schema_nodes(
                    left_properties[property_name],
                    right_properties[property_name],
                    metric_name,
                    prop_path,
                )
            else:
                if property_name in left_properties:
                    merged_properties[property_name] = left_properties[property_name]
                else:
                    merged_properties[property_name] = right_properties[property_name]

        merged_required = sorted(set(left.get("required", [])) | set(right.get("required", [])))
        merged = {
            "$schema": JSON_SCHEMA_DIALECT,
            "type": _merge_container_types("object", left_nullable, right_nullable),
            "properties": merged_properties,
            "required": merged_required,
        }
        if left.get("additionalProperties") is False and right.get("additionalProperties") is False:
            merged["additionalProperties"] = False
        return merged

    if left_kind == "array":
        return {
            "$schema": JSON_SCHEMA_DIALECT,
            "type": _merge_container_types("array", left_nullable, right_nullable),
            "items": _merge_schema_nodes(left.get("items", {}), right.get("items", {}), metric_name, f"{path}[]"),
        }

    allowed = _merge_primitive_types(allowed_types(left), allowed_types(right))
    return {"$schema": JSON_SCHEMA_DIALECT, "type": encode_type(allowed)}


def _merge_primitive_types(left: set[str], right: set[str]) -> list[str]:
    left_non_null = left - {"null"}
    right_non_null = right - {"null"}
    combined_non_null = left_non_null | right_non_null

    if combined_non_null <= {"integer", "number"}:
        merged = {"number"}
    elif len(combined_non_null) == 1:
        merged = set(combined_non_null)
    else:
        raise SchemaCompatibilityError(f"incompatible primitive schema types: {sorted(left)} vs {sorted(right)}")

    if "null" in left and "null" in right:
        merged.add("null")
    if len(merged) == 1:
        return sorted(merged)
    if merged == {"number", "null"}:
        return ["null", "number"]
    if merged == {"integer", "null"}:
        return ["integer", "null"]
    raise SchemaCompatibilityError(f"incompatible primitive schema types: {sorted(left)} vs {sorted(right)}")


def _check_schema_node(dataset: dict, required: dict, path: str) -> list[str]:
    dataset_allowed_types = allowed_types(dataset)
    required_allowed_types = allowed_types(required)
    required_kind = schema_kind(required)
    dataset_kind = schema_kind(dataset)
    errors: list[str] = []

    if required_kind == "any":
        return []
    if dataset_kind == "any":
        return [
            f"dataset field '{display_path(path)}' has unconstrained schema and cannot satisfy required schema {required!r}"
        ]
    if "null" in dataset_allowed_types and "null" not in required_allowed_types:
        return [
            f"dataset field '{display_path(path)}' is incompatible: expected {sorted(required_allowed_types)}, found {sorted(dataset_allowed_types)}"
        ]
    if required_kind == "primitive":
        if not _dataset_types_fit_requirement(dataset_allowed_types, required_allowed_types):
            return [
                f"dataset field '{display_path(path)}' is incompatible: expected {sorted(required_allowed_types)}, found {sorted(dataset_allowed_types)}"
            ]
        return []
    if dataset_kind != required_kind:
        return [f"dataset field '{display_path(path)}' is incompatible: expected {required_kind}, found {dataset_kind}"]

    if required_kind == "array":
        dataset_items = dataset.get("items")
        required_items = required.get("items")
        if required_items and dataset_items is None:
            return [f"dataset field '{display_path(path)}' is missing array item schema"]
        if dataset_items is None or required_items is None:
            return []
        return _check_schema_node(dataset_items, required_items, f"{path}[]")

    if required_kind == "object":
        dataset_required = set(dataset.get("required", []))
        dataset_properties = dataset.get("properties", {})
        required_properties = required.get("properties", {})

        for property_name in required.get("required", []):
            if property_name not in dataset_required:
                child_path = f"{path}.{property_name}" if path else property_name
                errors.append(f"dataset schema missing required field '{display_path(child_path)}'")

        for property_name, property_schema in required_properties.items():
            child_path = f"{path}.{property_name}" if path else property_name
            dataset_property_schema = dataset_properties.get(property_name)
            if dataset_property_schema is None:
                if property_name in required.get("required", []):
                    errors.append(f"dataset schema missing field definition '{display_path(child_path)}'")
                continue
            errors.extend(_check_schema_node(dataset_property_schema, property_schema, child_path))
        return errors

    return []


def _dataset_types_fit_requirement(dataset_types: set[str], required_types: set[str]) -> bool:
    dataset_non_null = dataset_types - {"null"}
    required_non_null = required_types - {"null"}
    return all(
        any(_type_is_compatible(dataset_type, required_type) for required_type in required_non_null)
        for dataset_type in dataset_non_null
    )


def _type_is_compatible(dataset_type: str, required_type: str) -> bool:
    return dataset_type == required_type or (dataset_type == "integer" and required_type == "number")


def _merge_container_types(
    base_type: str,
    left_nullable: bool,
    right_nullable: bool,
) -> str | list[str]:
    return [base_type, "null"] if left_nullable and right_nullable else base_type
