# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for converting platform filter operations into span query filters."""

from __future__ import annotations

import math
from datetime import datetime
from enum import Enum
from typing import Any, TypeVar

from fastapi import HTTPException, status
from nhx.common.api.filter import ComparisonOperation, FilterOperation, FilterOperator, LogicalOperation
from nhx.common.api.parsed_filter import ParsedFilter
from pydantic import TypeAdapter, ValidationError

_DATETIME_ADAPTER = TypeAdapter(datetime)
_TEnum = TypeVar("_TEnum", bound=Enum)


def filter_comparisons(parsed: ParsedFilter) -> list[ComparisonOperation]:
    return _comparisons(parsed.operation)


def require_eq_value(comparison: ComparisonOperation) -> Any:
    if comparison.operator != FilterOperator.EQ:
        raise _bad_filter(f"Filter field {comparison.field!r} only supports equality.")
    if comparison.value is None:
        raise _bad_filter(f"Filter field {comparison.field!r} does not support null values.")
    return comparison.value


def require_string_value(comparison: ComparisonOperation) -> str:
    value = require_eq_value(comparison)
    if not isinstance(value, str):
        raise _bad_filter(f"Filter field {comparison.field!r} must be a string.")
    return value


def require_string_or_list_value(comparison: ComparisonOperation) -> list[str]:
    """Accept an equality string or an `$in` list of strings, normalized to a list."""
    if comparison.operator != FilterOperator.IN:
        return [require_string_value(comparison)]
    value = comparison.value
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise _bad_filter(f"Filter field {comparison.field!r} with $in must be a list of strings.")
    return value


def require_enum_value(comparison: ComparisonOperation, enum_type: type[_TEnum]) -> _TEnum:
    value = require_eq_value(comparison)
    try:
        return enum_type(value)
    except ValueError:
        allowed = ", ".join(str(item.value) for item in enum_type)
        raise _bad_filter(f"Filter field {comparison.field!r} must be one of: {allowed}") from None


def require_datetime_value(comparison: ComparisonOperation) -> datetime:
    try:
        return _DATETIME_ADAPTER.validate_python(comparison.value)
    except ValidationError as exc:
        raise _bad_filter(f"Filter field {comparison.field!r} must be a valid datetime.") from exc


def require_float_value(comparison: ComparisonOperation) -> float:
    value = comparison.value
    if isinstance(value, bool):
        raise _bad_filter(f"Filter field {comparison.field!r} must be a number.")
    parsed: float | None = None
    if isinstance(value, (int, float)):
        parsed = float(value)
    elif isinstance(value, str):
        try:
            parsed = float(value)
        except ValueError:
            parsed = None
    if parsed is None or not math.isfinite(parsed):
        raise _bad_filter(f"Filter field {comparison.field!r} must be a number.")
    return parsed


def _comparisons(operation: FilterOperation | None) -> list[ComparisonOperation]:
    if operation is None:
        return []
    if isinstance(operation, ComparisonOperation):
        return [operation]
    if isinstance(operation, LogicalOperation) and operation.operator == FilterOperator.AND:
        comparisons: list[ComparisonOperation] = []
        for child in operation.operations:
            comparisons.extend(_comparisons(child))
        return comparisons
    raise _bad_filter("Only AND-combined field filters are supported for Intake span storage.")


def _bad_filter(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)
