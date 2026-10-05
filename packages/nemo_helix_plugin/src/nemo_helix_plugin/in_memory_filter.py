# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-memory FilterRepository: evaluate a filter tree against Python objects.

A peer of ``SQLAlchemyFilterRepository`` — same filter tree, same ``apply``
front door, different executor. Bind an entity and run ``operation.apply(repo)``
to get a ``bool`` instead of a SQL expression.
"""

import operator
from datetime import datetime
from typing import Any, Callable, Dict, List

from nemo_helix_plugin.filter_ops import FilterOperator, FilterRepository

# Sentinel distinguishing "field/key is absent" from an explicit None value.
_MISSING = object()


class InMemoryFilterRepository(FilterRepository):
    """Evaluate a filter tree against one in-memory entity, returning ``bool``.

    Bind an entity (mirrors ``SQLAlchemyFilterRepository`` binding a model), then
    run ``operation.apply(repo)``. The entity may be a mapping or an object: base
    fields resolve as attributes/keys, and ``data.<dotted>`` fields walk the
    nested ``data`` mapping.

    Semantics are plain Python, NOT a byte-for-byte mirror of the SQL backends:

    - Values are compared by native Python type. There is no JSON-to-text
      coercion, so ``data.score`` (stored int ``5``) does not match the string
      ``"5"``, and ``$like``/``$in``/``$nin`` see the real value.
    - Absent fields (missing key/attr) and explicit ``None`` are equivalent and
      satisfy only ``$eq None`` (standard SQL three-valued NULL logic, which both
      SQLite and PostgreSQL share).
    - ``$like`` is a case-insensitive substring test; ``%`` and ``_`` are literal
      characters, NOT SQL wildcards.

    Where the SQL backends disagree with each other (JSON boolean text is
    ``"1"/"0"`` on SQLite but ``"true"/"false"`` on PostgreSQL; SQLite casts
    non-numeric text to ``0.0`` while PostgreSQL errors), this path follows
    neither. Realistic callers (e.g. job-status filtering in the jobs dispatcher)
    compare strings, numbers, and datetimes, where native and SQL semantics agree.
    """

    _ORDERED: Dict[FilterOperator, Callable[[Any, Any], bool]] = {
        FilterOperator.LT: operator.lt,
        FilterOperator.LTE: operator.le,
        FilterOperator.GT: operator.gt,
        FilterOperator.GTE: operator.ge,
    }

    def __init__(self, entity: Any):
        self.entity = entity

    def _value(self, field: str) -> Any:
        """Resolve ``field`` on the bound entity (``_MISSING`` if absent).

        Mirrors ``SQLAlchemyFilterRepository._get_column``: a plain attribute/key,
        or a ``data.<dotted>`` walk through the nested ``data`` mapping.
        """
        if _has_attr(self.entity, field):
            return _get_attr(self.entity, field, _MISSING)
        if field.startswith("data."):
            data = _get_attr(self.entity, "data", _MISSING)
            return _walk_json_path(data, field.split(".")[1:])
        raise ValueError(f"Field '{field}' does not exist on {type(self.entity).__name__}")

    def eq(self, field: str, value: Any) -> bool:
        field_value = self._value(field)
        # Absent collapses to None, so only `$eq None` matches a missing/null field.
        return (None if field_value is _MISSING else field_value) == value

    def like(self, field: str, value: str) -> bool:
        field_value = self._value(field)
        if field_value is _MISSING or field_value is None:
            return False
        return str(value).lower() in str(field_value).lower()

    def lt(self, field: str, value: Any) -> bool:
        return self._ordered_compare(field, value, FilterOperator.LT)

    def lte(self, field: str, value: Any) -> bool:
        return self._ordered_compare(field, value, FilterOperator.LTE)

    def gt(self, field: str, value: Any) -> bool:
        return self._ordered_compare(field, value, FilterOperator.GT)

    def gte(self, field: str, value: Any) -> bool:
        return self._ordered_compare(field, value, FilterOperator.GTE)

    def _ordered_compare(self, field: str, value: Any, op: FilterOperator) -> bool:
        field_value = self._value(field)
        # Absent/None never satisfies an ordered comparison (SQL `NULL < x` is NULL).
        if field_value is _MISSING or field_value is None:
            return False
        if isinstance(field_value, datetime) and isinstance(value, str):
            value = datetime.fromisoformat(value).replace(tzinfo=None)
        try:
            return self._ORDERED[op](field_value, value)
        except TypeError:
            # Incomparable operands (e.g. text vs number) — no match.
            return False

    def in_op(self, field: str, values: List[Any]) -> bool:
        field_value = self._value(field)
        if field_value is _MISSING or field_value is None:
            return False
        return field_value in values

    def nin(self, field: str, values: List[Any]) -> bool:
        field_value = self._value(field)
        # Absent/None matches nothing (`NULL NOT IN (...)` is NULL, i.e. not true)
        # — this is NOT simply `not in_op`.
        if field_value is _MISSING or field_value is None:
            return False
        return field_value not in values

    def contains(self, field: str, value: Any) -> bool:
        """Array membership: true when the list at ``field`` contains ``value``.

        Absent/None/non-list fields match nothing (mirrors the SQL path, where a missing
        or scalar element can't contain the token). Elements are compared by native value.
        """
        field_value = self._value(field)
        if field_value is _MISSING or not isinstance(field_value, (list, tuple)):
            return False
        return value in field_value

    def and_op(self, operations: List[Any]) -> bool:
        return all(operations)

    def or_op(self, operations: List[Any]) -> bool:
        return any(operations)

    def not_op(self, operation: Any) -> bool:
        return not operation


def _has_attr(entity: Any, field: str) -> bool:
    """Whether `field` is a key (mapping) or attribute (object) of `entity`."""
    if isinstance(entity, dict):
        return field in entity
    return hasattr(entity, field)


def _get_attr(entity: Any, field: str, default: Any) -> Any:
    """Read `field` as a key (mapping) or attribute (object), else `default`."""
    if isinstance(entity, dict):
        return entity.get(field, default)
    return getattr(entity, field, default)


def _walk_json_path(data: Any, path: List[str]) -> Any:
    """Navigate a dotted `data.<a>.<b>` path through nested mappings.

    Mirrors the SQL JSON subscript walk. A missing key (or descending into a
    non-mapping) yields ``_MISSING`` so callers apply SQL null semantics.
    """
    current = data
    for key in path:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return _MISSING
    return current
