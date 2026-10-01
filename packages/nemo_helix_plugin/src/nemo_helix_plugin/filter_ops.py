# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Filter operation base types for the entity store.

These are the minimal types needed by EntityClient and Filter. The full parsing
engine (parse_json_filter, parse_bracket_filter, etc.) lives in nhx.common.api.filter.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List

from pydantic import BaseModel

#: A value an ``$elemMatch`` criterion can compare against.
ElemMatchScalar = str | int | float | bool | None


def validate_elem_match_criteria(value: Any) -> Dict[str, ElemMatchScalar]:
    """Return ``value`` as ``$elemMatch`` criteria, or raise ``ValueError`` if it isn't a non-empty scalar map."""
    if not isinstance(value, dict) or not value:
        raise ValueError("$elemMatch requires a non-empty object of element field -> value")
    for key, criterion in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError("$elemMatch element field names must be non-empty strings")
        if criterion is not None and not isinstance(criterion, (str, int, float, bool)):
            raise ValueError(f"$elemMatch value for '{key}' must be a string, number, boolean, or null")
    return value


def validate_string_operand(operator: "FilterOperator", value: Any) -> str:
    """Return ``value`` as the non-empty, quote-free string operand ``operator`` needs, or raise ``ValueError``."""
    if not isinstance(value, str) or not value or '"' in value:
        raise ValueError(f"{operator.value} requires a non-empty string without double quotes")
    return value


class FilterOperator(str, Enum):
    """Filter operator."""

    # Comparison operators
    EQ = "$eq"
    LIKE = "$like"
    LT = "$lt"
    LTE = "$lte"
    GT = "$gt"
    GTE = "$gte"
    IN = "$in"
    NIN = "$nin"
    CONTAINS = "$contains"
    ELEM_MATCH = "$elemMatch"
    CONTAINS_PREFIX = "$containsPrefix"
    HAS_KEY = "$hasKey"

    # Logical operators
    OR = "$or"
    AND = "$and"
    NOT = "$not"

    # Relationship operators
    EXISTS = "$exists"


class FilterRepository(ABC):
    """Abstract base class for repository implementations that execute filter operations."""

    @abstractmethod
    def eq(self, field: str, value: Any) -> Any:
        pass

    @abstractmethod
    def like(self, field: str, value: str) -> Any:
        pass

    @abstractmethod
    def lt(self, field: str, value: Any) -> Any:
        pass

    @abstractmethod
    def lte(self, field: str, value: Any) -> Any:
        pass

    @abstractmethod
    def gt(self, field: str, value: Any) -> Any:
        pass

    @abstractmethod
    def gte(self, field: str, value: Any) -> Any:
        pass

    @abstractmethod
    def in_op(self, field: str, values: List[Any]) -> Any:
        pass

    @abstractmethod
    def nin(self, field: str, values: List[Any]) -> Any:
        pass

    def contains(self, field: str, value: Any) -> Any:
        """Array membership: match rows where the array at ``field`` contains scalar ``value``.

        Optional — additive to the base contract, so repositories that don't support
        array-valued fields may leave it unimplemented.
        """
        raise NotImplementedError("$contains not supported by this repository")

    def elem_match(self, field: str, criteria: Dict[str, ElemMatchScalar]) -> Any:
        """Match rows where some object in the array at ``field`` has every ``criteria`` key ``$eq`` its value.

        Optional — repositories that don't support arrays of objects may leave it unimplemented.
        """
        raise NotImplementedError("$elemMatch not supported by this repository")

    def contains_prefix(self, field: str, prefix: str) -> Any:
        """Match rows where the array at ``field`` has a string element starting with ``prefix``.

        Optional — repositories that don't support array-valued fields may leave it unimplemented.
        """
        raise NotImplementedError("$containsPrefix not supported by this repository")

    def has_key(self, field: str, key: str) -> Any:
        """Match rows where the object at ``field`` has ``key`` with a non-null value.

        Optional — repositories that don't support object-valued fields may leave it unimplemented.
        """
        raise NotImplementedError("$hasKey not supported by this repository")

    @abstractmethod
    def and_op(self, operations: List[Any]) -> Any:
        pass

    @abstractmethod
    def or_op(self, operations: List[Any]) -> Any:
        pass

    @abstractmethod
    def not_op(self, operation: Any) -> Any:
        pass

    def relationship_exists(
        self,
        target_entity_type: str,
        join_field: str,
        child_condition: "FilterOperation | None",
        negate: bool,
    ) -> Any:
        raise NotImplementedError("Relationship queries not supported by this repository")


class FilterOperation(BaseModel, ABC):
    """Abstract base class for filter operations."""

    operator: FilterOperator

    @abstractmethod
    def apply(self, repository: FilterRepository) -> Any:
        """Apply this operation using the given repository."""
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        pass


class ComparisonOperation(FilterOperation):
    """Comparison operation (e.g., eq, lt, gte, like)."""

    operator: FilterOperator
    field: str
    value: Any

    def apply(self, repository: FilterRepository) -> Any:
        if self.operator == FilterOperator.EQ:
            return repository.eq(self.field, self.value)
        elif self.operator == FilterOperator.LIKE:
            return repository.like(self.field, self.value)
        elif self.operator == FilterOperator.LT:
            return repository.lt(self.field, self.value)
        elif self.operator == FilterOperator.LTE:
            return repository.lte(self.field, self.value)
        elif self.operator == FilterOperator.GT:
            return repository.gt(self.field, self.value)
        elif self.operator == FilterOperator.GTE:
            return repository.gte(self.field, self.value)
        elif self.operator == FilterOperator.IN:
            return repository.in_op(self.field, self.value)
        elif self.operator == FilterOperator.NIN:
            return repository.nin(self.field, self.value)
        elif self.operator == FilterOperator.CONTAINS:
            return repository.contains(self.field, self.value)
        elif self.operator == FilterOperator.ELEM_MATCH:
            return repository.elem_match(self.field, validate_elem_match_criteria(self.value))
        elif self.operator == FilterOperator.CONTAINS_PREFIX:
            return repository.contains_prefix(self.field, validate_string_operand(self.operator, self.value))
        elif self.operator == FilterOperator.HAS_KEY:
            return repository.has_key(self.field, validate_string_operand(self.operator, self.value))
        elif self.operator == FilterOperator.EXISTS:
            raise NotImplementedError(
                "$exists requires a relationship-aware repository (use the entities service parser)"
            )
        else:
            raise ValueError(f"Unknown comparison operator: {self.operator}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {self.field: {self.operator.value: self.value}}


class LogicalOperation(FilterOperation):
    """Logical operation (and, or, not)."""

    operator: FilterOperator
    operations: List[FilterOperation]

    def apply(self, repository: FilterRepository) -> Any:
        if self.operator == FilterOperator.AND:
            return repository.and_op([op.apply(repository) for op in self.operations])
        elif self.operator == FilterOperator.OR:
            return repository.or_op([op.apply(repository) for op in self.operations])
        elif self.operator == FilterOperator.NOT:
            if len(self.operations) != 1:
                raise ValueError("NOT operation must have exactly one operand")
            return repository.not_op(self.operations[0].apply(repository))
        else:
            raise ValueError(f"Unknown logical operator: {self.operator}")

    def to_dict(self) -> Dict[str, Any]:
        if self.operator == FilterOperator.NOT:
            return {self.operator.value: self.operations[0].to_dict()}
        return {self.operator.value: [op.to_dict() for op in self.operations]}
