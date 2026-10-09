# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Filter operation base types for the entity store.

These are the minimal types needed by EntityClient and Filter. The full parsing
engine (parse_json_filter, parse_bracket_filter, etc.) lives in nhx.common.api.filter.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, NamedTuple

from pydantic import BaseModel

#: A value an ``$elemMatch`` criterion can compare against.
ElemMatchScalar = str | int | float | bool | None


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
    HAS_KEY = "$hasKey"
    STARTS_WITH = "$startsWith"
    ENDS_WITH = "$endsWith"

    # Logical operators
    OR = "$or"
    AND = "$and"
    NOT = "$not"

    # Relationship operators
    EXISTS = "$exists"


class ElemMatchCondition(NamedTuple):
    """One comparison inside ``$elemMatch``: element field ``key`` (``None`` for the element itself)."""

    key: str | None
    operator: FilterOperator
    value: Any


#: Comparisons allowed inside ``$elemMatch``; several on one target combine with AND.
ELEM_MATCH_OPERATORS = frozenset(
    {
        FilterOperator.EQ,
        FilterOperator.IN,
        FilterOperator.NIN,
        FilterOperator.LIKE,
        FilterOperator.LT,
        FilterOperator.LTE,
        FilterOperator.GT,
        FilterOperator.GTE,
        FilterOperator.STARTS_WITH,
        FilterOperator.ENDS_WITH,
    }
)

_STRING_OPERATORS = frozenset({FilterOperator.LIKE, FilterOperator.STARTS_WITH, FilterOperator.ENDS_WITH})


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def validate_string_operand(operator: FilterOperator, value: Any, *, allow_quotes: bool = True) -> str:
    """Return ``value`` as the non-empty string operand ``operator`` needs, or raise ``ValueError``."""
    if not isinstance(value, str) or not value or (not allow_quotes and '"' in value):
        quotes = "" if allow_quotes else " without double quotes"
        raise ValueError(f"{operator.value} requires a non-empty string{quotes}")
    return value


def _operator_conditions(key: str | None, operators: Dict[str, Any]) -> List[ElemMatchCondition]:
    target = f"'{key}'" if key is not None else "the element"
    if not operators:
        raise ValueError(f"$elemMatch needs at least one operator for {target}")
    conditions = []
    for name, operand in operators.items():
        try:
            operator = FilterOperator(name)
        except ValueError:
            raise ValueError(f"Unknown operator {name!r} in $elemMatch") from None
        if operator not in ELEM_MATCH_OPERATORS:
            raise ValueError(f"{name} is not supported inside $elemMatch")
        if operator in (FilterOperator.IN, FilterOperator.NIN):
            if not isinstance(operand, list) or not operand or not all(_is_scalar(item) for item in operand):
                raise ValueError(f"{name} for {target} in $elemMatch requires a non-empty list of scalars")
        elif operator in _STRING_OPERATORS:
            validate_string_operand(operator, operand)
        elif operator == FilterOperator.EQ:
            if not _is_scalar(operand):
                raise ValueError(f"{name} for {target} in $elemMatch requires a string, number, boolean, or null")
        elif not isinstance(operand, (str, int, float)) or isinstance(operand, bool):
            raise ValueError(f"{name} for {target} in $elemMatch requires a string or number")
        conditions.append(ElemMatchCondition(key, operator, operand))
    return conditions


def parse_elem_match_criteria(value: Any) -> List[ElemMatchCondition]:
    """Parse ``$elemMatch`` criteria into conditions that one array element must all satisfy.

    Two forms: ``{field: value-or-operators}`` matches object elements, a bare value meaning ``$eq``;
    ``{operator: operand}`` matches scalar elements. Raises ``ValueError`` on anything else.
    """
    if not isinstance(value, dict) or not value:
        raise ValueError("$elemMatch requires a non-empty object of element fields or operators")
    keys_are_operators = [isinstance(key, str) and key.startswith("$") for key in value]
    if all(keys_are_operators):
        return _operator_conditions(None, value)
    if any(keys_are_operators):
        raise ValueError("$elemMatch takes either element field names or operators, not both")
    conditions: List[ElemMatchCondition] = []
    for key, criterion in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError("$elemMatch element field names must be non-empty strings")
        if isinstance(criterion, dict):
            conditions.extend(_operator_conditions(key, criterion))
        elif _is_scalar(criterion):
            conditions.append(ElemMatchCondition(key, FilterOperator.EQ, criterion))
        else:
            raise ValueError(f"$elemMatch value for '{key}' must be a string, number, boolean, null, or operators")
    return conditions


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

    def elem_match(self, field: str, conditions: List[ElemMatchCondition]) -> Any:
        """Match rows where one element of the array at ``field`` satisfies every condition.

        Optional — repositories that don't support array-valued fields may leave it unimplemented.
        """
        raise NotImplementedError("$elemMatch not supported by this repository")

    def starts_with(self, field: str, prefix: str) -> Any:
        """Match rows where ``field`` starts with ``prefix`` (case-sensitive).

        Optional — repositories may leave it unimplemented.
        """
        raise NotImplementedError("$startsWith not supported by this repository")

    def ends_with(self, field: str, suffix: str) -> Any:
        """Match rows where ``field`` ends with ``suffix`` (case-sensitive).

        Optional — repositories may leave it unimplemented.
        """
        raise NotImplementedError("$endsWith not supported by this repository")

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
            return repository.elem_match(self.field, parse_elem_match_criteria(self.value))
        elif self.operator == FilterOperator.HAS_KEY:
            return repository.has_key(
                self.field, validate_string_operand(self.operator, self.value, allow_quotes=False)
            )
        elif self.operator == FilterOperator.STARTS_WITH:
            return repository.starts_with(self.field, validate_string_operand(self.operator, self.value))
        elif self.operator == FilterOperator.ENDS_WITH:
            return repository.ends_with(self.field, validate_string_operand(self.operator, self.value))
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
