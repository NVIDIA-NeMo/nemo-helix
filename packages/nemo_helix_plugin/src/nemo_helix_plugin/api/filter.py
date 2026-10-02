# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Filter parsing utilities (canonical home for parsers).

Adds JSON / bracket-notation parsers that turn raw query input into
``FilterOperation`` trees. Base operation types live in
``nemo_helix_plugin.filter_ops`` and are re-exported here so callers using the
parsers can pull ops + parsers from a single module. Callers that need
only the ops should import from ``nemo_helix_plugin.filter_ops`` directly.

Supported operators: ``$eq``, ``$like``, ``$lt``, ``$lte``, ``$gt``, ``$gte``,
``$in``, ``$nin``, ``$and``, ``$or``, ``$not``.
"""

import json
from typing import Any, Dict, Sequence

# Re-export base types so callers can import everything from one place.
from nemo_helix_plugin.filter_ops import ComparisonOperation as ComparisonOperation
from nemo_helix_plugin.filter_ops import FilterOperation as FilterOperation
from nemo_helix_plugin.filter_ops import FilterOperator as FilterOperator
from nemo_helix_plugin.filter_ops import FilterRepository as FilterRepository
from nemo_helix_plugin.filter_ops import LogicalOperation as LogicalOperation


def _normalize_value(operator: FilterOperator, value: Any) -> Any:
    """Normalize value based on operator (e.g., split comma-separated strings for $in/$nin)."""
    if operator in (FilterOperator.IN, FilterOperator.NIN) and isinstance(value, str):
        return value.split(",")
    return value


def _wrap_operations(
    operations: Sequence[FilterOperation], empty_error: str = "No valid filter criteria found"
) -> FilterOperation:
    """Wrap multiple operations in AND, or return single operation as-is."""
    if len(operations) == 1:
        return operations[0]
    elif len(operations) > 1:
        return LogicalOperation(operator=FilterOperator.AND, operations=list(operations))
    else:
        raise ValueError(empty_error)


_LOGICAL_KEYS = frozenset({"$and", "$or", "$not"})


def _reject_mixed_logical_root(d: Dict[str, Any], scope: str) -> None:
    """Reject filters that mix a logical operator with sibling keys.

    A shape like ``{"$or": [...], "project": "p"}`` is ambiguous: should
    ``project`` be ANDed with the ``$or``, or is the ``$or`` meant to scope
    a different concept? Previously the parser short-circuited on the first
    logical operator and silently dropped the siblings — overbroad results,
    hard to debug. Force the caller to disambiguate by wrapping siblings
    under an explicit ``$and``.
    """
    if any(k in d for k in _LOGICAL_KEYS) and len(d) > 1:
        present = sorted(d.keys())
        raise ValueError(
            f"{scope} filter mixes a logical operator with sibling keys ({present}); "
            "wrap the siblings inside the logical operator or under an explicit $and"
        )


def _parse_field_operation(field: str, op_dict: Dict[str, Any]) -> FilterOperation:
    """Parse an operation dict in the context of a specific field.

    Handles:
    - Comparison operators: {"$eq": "value"} -> ComparisonOperation
    - Logical operators: {"$or": [{"$eq": "v1"}, {"$eq": "v2"}]} -> LogicalOperation
    """
    if not isinstance(op_dict, dict):
        raise ValueError(f"Expected dict for field operation, got {type(op_dict)}")

    _reject_mixed_logical_root(op_dict, scope=f"Field '{field}'")

    # Check for logical operators at field level
    for logical_op in (FilterOperator.OR, FilterOperator.AND):
        op_key = logical_op.value
        if op_key in op_dict:
            sub_operations = [_parse_field_operation(field, item) for item in op_dict[op_key]]
            return LogicalOperation(operator=logical_op, operations=sub_operations)

    if "$not" in op_dict:
        sub_operation = _parse_field_operation(field, op_dict["$not"])
        return LogicalOperation(operator=FilterOperator.NOT, operations=[sub_operation])

    # Handle comparison operators
    operations = [
        ComparisonOperation(
            operator=FilterOperator(op_str), field=field, value=_normalize_value(FilterOperator(op_str), op_value)
        )
        for op_str, op_value in op_dict.items()
        if op_str.startswith("$")
    ]

    return _wrap_operations(operations, f"No valid operation found for field {field}")


def _parse_dict_to_operation(filter_dict: Dict[str, Any]) -> FilterOperation:
    """Convert dictionary to FilterOperation (without relationship support)."""
    _reject_mixed_logical_root(filter_dict, scope="Top-level")

    # Check for top-level logical operators
    for logical_op in (FilterOperator.AND, FilterOperator.OR):
        op_key = logical_op.value
        if op_key in filter_dict:
            operations = [_parse_dict_to_operation(item) for item in filter_dict[op_key]]
            return LogicalOperation(operator=logical_op, operations=operations)

    if "$not" in filter_dict:
        operation = _parse_dict_to_operation(filter_dict["$not"])
        return LogicalOperation(operator=FilterOperator.NOT, operations=[operation])

    operations: list[FilterOperation] = []
    for field, value in filter_dict.items():
        if isinstance(value, dict) and any(k.startswith("$") for k in value):
            operations.append(_parse_field_operation(field, value))
        else:
            operations.append(ComparisonOperation(operator=FilterOperator.EQ, field=field, value=value))

    return _wrap_operations(operations)


def parse_json_filter(filter_json: str) -> FilterOperation:
    """Parse JSON filter parameter into a FilterOperation tree.

    Args:
        filter_json: JSON string representing the filter query.

    Returns:
        Parsed FilterOperation.

    Raises:
        ValueError: If the JSON is invalid or cannot be parsed.
    """
    try:
        filter_dict = json.loads(filter_json)
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in filter parameter: {filter_json}") from None
    return _parse_dict_to_operation(filter_dict)


# Comparison operator names accepted without $ prefix in bracket notation
# (e.g., filter[created_at][gte]=... is equivalent to filter[created_at][$gte]=...).
_BRACKET_OPERATOR_ALIASES = frozenset({"gte", "lte", "gt", "lt", "like", "eq", "in", "nin", "not", "exists"})


def _normalize_operator_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    """Add ``$`` prefix to operator keys that are missing it.

    Bracket notation ``filter[field][gte]=value`` produces ``{"gte": "value"}``,
    but the filter parser expects ``{"$gte": "value"}``.
    """
    result: Dict[str, Any] = {}
    for key, value in d.items():
        new_key = f"${key}" if key in _BRACKET_OPERATOR_ALIASES else key
        if isinstance(value, dict):
            value = _normalize_operator_keys(value)
        result[new_key] = value
    return result


def _apply_implicit_eq(d: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap bare values in ``{"$eq": value}`` so bracket notation defaults to exact matching.

    Dicts (already containing an explicit operator like ``{"$like": ...}``) are
    passed through unchanged, with operator keys normalized (``gte`` -> ``$gte``).
    All other values (strings, numbers, booleans) get wrapped in ``$eq``.
    """
    result: Dict[str, Any] = {}
    for key, value in d.items():
        if isinstance(value, dict):
            result[key] = _normalize_operator_keys(value)
        else:
            result[key] = {"$eq": value}
    return result


def parse_bracket_filter(bracket_dict: Dict[str, Any]) -> FilterOperation:
    """Convert a bracket-notation dict (from ``parse_deep_object``) into a FilterOperation.

    Bare string values are implicitly treated as ``$eq`` (exact match).
    Use explicit operators for other behavior, e.g. ``filter[name][$like]=llama``.
    Values that are already dicts (e.g. ``{"$like": "..."}``) are passed through as-is.
    """
    return parse_json_filter(json.dumps(_apply_implicit_eq(bracket_dict)))
