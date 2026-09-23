# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Table column configurations for different resource types."""

from __future__ import annotations

import re
from typing import Any, Literal

import typer

from nemo_helix_ext.cli.core.formatters import Column, _extract_items_from_response


def validate_output_columns(columns: Literal["all"] | str | list[Column]) -> list[Column] | Literal["all"]:
    """Validate and process output column specifications.

    Args:
        columns: Column specification - can be "all", or a comma-separated string of column names

    Returns:
        List of (display_name, field_path) tuples, or the string "all"
    """
    # Handle "all" keyword
    if columns == "all":
        return columns

    # Parse and resolve comma-separated column names
    resolved_columns = []
    if isinstance(columns, str):
        for column in columns.split(","):
            column = column.strip()
            resolved_columns.append(Column(column))

    return resolved_columns


def get_available_nested_fields(obj: Any, prefix: str = "", max_depth: int = 5) -> list[str]:
    """Recursively extract all field paths from an object (e.g., "name", "config.type")."""
    if max_depth <= 0:
        return []

    fields = []

    # Handle list - introspect first element
    if isinstance(obj, list):
        if len(obj) > 0:
            return get_available_nested_fields(obj[0], prefix, max_depth)
        return []

    # Handle dict
    if isinstance(obj, dict):
        for key, value in obj.items():
            current_path = f"{prefix}.{key}" if prefix else key
            fields.append(current_path)

            # Recursively get nested fields for dicts and lists
            if isinstance(value, (dict, list)):
                nested = get_available_nested_fields(value, current_path, max_depth - 1)
                fields.extend(nested)

    return fields


def _is_field_invalid(field_path: str, first_item_dict: dict, valid_fields: set) -> bool:
    """Check if a field path is invalid by validating top-level and nested paths."""
    top_level_field = field_path.split(".")[0]

    # Check if top-level field exists
    if top_level_field not in valid_fields:
        return True

    # For nested fields, validate the full path
    if "." in field_path:
        value = get_nested_value(first_item_dict, field_path)
        top_value = first_item_dict.get(top_level_field)
        # If top-level has data but nested path returns empty, the field is invalid
        if value == "" and top_value is not None and top_value != []:
            return True

    return False


def _print_shell_expansion_hint() -> None:
    """Print hint about quoting field names with special characters."""
    typer.echo("", err=True)
    typer.echo("Note: Field names with special characters like '$' must be quoted to prevent", err=True)
    typer.echo("      shell variable expansion. Use quotes like:", err=True)
    typer.echo("      --output-columns 'dataset_schemas.$defs.SourceTypeEnum'", err=True)


def _print_nested_columns(error_parents: set[str], first_item_dict: dict) -> None:
    """Print nested columns grouped by parent for the given error parents."""
    all_fields = get_available_nested_fields(first_item_dict)
    nested_by_parent = {}
    for field in all_fields:
        if "." in field:
            parent = field.split(".")[0]
            nested_by_parent.setdefault(parent, []).append(field)

    typer.echo("Available columns:", err=True)
    for parent in sorted(error_parents):
        typer.echo(f"  {parent}", err=True)
        for nested in sorted(nested_by_parent.get(parent, [])):
            typer.echo(f"    {nested}", err=True)


def _print_top_level_columns(valid_fields: set) -> None:
    """Print top-level columns."""
    typer.echo("Valid columns:", err=True)
    for field in sorted(valid_fields):
        typer.echo(f"  {field}", err=True)


def _print_validation_errors(invalid_fields: list[str], first_item_dict: dict, valid_fields: set) -> None:
    """Print validation errors with shell expansion hints and available columns."""
    typer.echo(f"Error: Invalid column names: {', '.join(invalid_fields)}", err=True)

    if any(".." in field for field in invalid_fields):
        _print_shell_expansion_hint()

    typer.echo("", err=True)

    if any("." in field for field in invalid_fields):
        error_parents = {field.split(".")[0] for field in invalid_fields if "." in field}
        _print_nested_columns(error_parents, first_item_dict)
    else:
        _print_top_level_columns(valid_fields)


def resolve_and_validate_columns(
    output_columns: list[Column] | Literal["all"],
    items: Any,
) -> list[Column]:
    """
    Resolve 'all' keyword and validate column names against actual data fields.

    Args:
        output_columns: Column specification - can be "all" or list of (display_name, field_path) tuples
        items: Response data containing items to extract field names from

    Returns:
        List of (display_name, field_path) tuples with validated columns

    Raises:
        typer.Exit: If invalid column names are detected
    """
    # Extract data from response (handles .data, .files, .items, etc.)
    data = _extract_items_from_response(items)
    if not data:
        return []

    # Convert first item to dict if it's a Pydantic model, otherwise use as-is
    first_item = data[0]
    if hasattr(first_item, "model_dump"):
        first_item_dict = first_item.model_dump()
    elif isinstance(first_item, dict):
        first_item_dict = first_item
    else:
        first_item_dict = {}
    valid_fields = set(first_item_dict.keys())
    # Preserve key order from the response (matches JSON output order)
    field_order = list(first_item_dict.keys())

    # Handle "all" keyword - return all available fields in same order as JSON
    if output_columns == "all":
        return [Column(field) for field in field_order]

    # Validate all field paths
    invalid_fields = []
    for column in output_columns:
        field_path = column.field
        # For computed fields, extract and validate inner field paths
        if "{" in field_path and "}" in field_path:
            for field in re.findall(r"\{([^}]+)\}", field_path):
                if _is_field_invalid(field, first_item_dict, valid_fields):
                    invalid_fields.append(field)
        # For regular fields, validate directly
        elif field_path not in valid_fields:
            if _is_field_invalid(field_path, first_item_dict, valid_fields):
                invalid_fields.append(field_path)

    if invalid_fields:
        _print_validation_errors(invalid_fields, first_item_dict, valid_fields)
        # Return valid columns only, filtering out invalid ones
        result = [col for col in output_columns if col.field not in invalid_fields]
        if result == []:
            typer.echo("Note: No valid columns found, returning all valid fields", err=True)
            return [Column(field) for field in field_order]
        return result

    return output_columns


# Timestamp fields that should be formatted
TIMESTAMP_FIELDS = {
    "created_at",
    "updated_at",
    "started_at",
    "completed_at",
    "finished_at",
}

# Fields that should never be truncated
NO_TRUNCATE_FIELDS = {
    "name",
    "{namespace}/{name}",  # Computed Resource ID field
}


def get_nested_value(obj: Any, path: str) -> str:
    """
    Get a nested value from an object using dot notation.

    Args:
        obj: Object to extract value from (dict or object with attributes)
        path: Dot-notation path (e.g., "metadata.name") or computed path (e.g., "{namespace}/{name}")
              Path segments can be quoted to handle special characters (e.g., "dataset_schemas.'$defs'.field")

    Returns:
        String representation of the value, or empty string if not found
    """
    # Handle computed fields (e.g., "{namespace}/{name}")
    if "{" in path and "}" in path:
        # Extract field names from the pattern
        field_pattern = re.findall(r"\{([^}]+)\}", path)
        result = path
        for field_name in field_pattern:
            # Get the field value recursively
            field_value = get_nested_value(obj, field_name)
            # Replace the placeholder with the value
            result = result.replace(f"{{{field_name}}}", field_value)
        return result

    parts = path.split(".")
    current = obj

    for part in parts:
        if current is None:
            return ""

        # Strip quotes from part (handles both single and double quotes)
        # This allows paths like "dataset_schemas.'$defs'.field" to work
        if (part.startswith("'") and part.endswith("'")) or (part.startswith('"') and part.endswith('"')):
            part = part[1:-1]

        # If current is a list, get first element before trying to access the field
        if isinstance(current, list):
            current = current[0] if current else None
            if current is None:
                return ""

        # Try dict access
        if isinstance(current, dict):
            current = current.get(part)
        # Try attribute access
        elif hasattr(current, part):
            current = getattr(current, part)
        else:
            return ""

    # Handle list result - get first element or empty
    if isinstance(current, list):
        current = current[0] if current else None

    # Convert to string, handling None
    return "" if current is None else str(current)


def is_timestamp_field(field_path: str) -> bool:
    """
    Check if a field path represents a timestamp field.

    Args:
        field_path: Field path (e.g., "created_at", "metadata.updated_at")

    Returns:
        True if this is a timestamp field
    """
    # Get the last part of the path (e.g., "updated_at" from "metadata.updated_at")
    field_name = field_path.split(".")[-1]
    return field_name in TIMESTAMP_FIELDS


def should_truncate_field(field_path: str) -> bool:
    """
    Check if a field should be truncated in table output.

    Args:
        field_path: Field path (e.g., "name", "description", "{namespace}/{name}")

    Returns:
        False if this field should never be truncated, True otherwise
    """
    # Check exact match
    if field_path in NO_TRUNCATE_FIELDS:
        return False

    # Get the last part of the path (e.g., "name" from "metadata.name")
    field_name = field_path.split(".")[-1]
    return field_name not in NO_TRUNCATE_FIELDS
