# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TypedDict introspection utilities for CLI generation."""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass
from typing import Any, ForwardRef, Literal, Union, get_args, get_origin, get_type_hints

from nemo_helix_sdk_tools.sdk.cli_generator.type_formatter import format_type


@dataclass
class TypedDictField:
    """Represents a field from a TypedDict type."""

    name: str
    type_annotation: Any  # String representation (legacy, prefer evaluated_type)
    is_required: bool
    evaluated_type: Any = None  # Actual resolved type (use this!)

    @property
    def python_type_name(self) -> str:
        """Get string representation of the Python type."""
        return format_type(self.type_annotation)

    @property
    def is_simple_cli_type(self) -> bool:
        """Check if field is simple CLI type using actual type objects.

        Simple types: str, int, bool, float, Literal, Enum
        Complex types: dict, TypedDict, nested objects

        Uses evaluated_type (actual Type object) instead of string matching.
        """
        if self.evaluated_type is None:
            # No evaluated type - fall back to string heuristic
            return self._is_simple_from_string()

        # Use actual type checking
        tp = self.evaluated_type
        origin = get_origin(tp)

        # Handle unions - check if any member is simple
        if origin is Union:
            for arg in get_args(tp):
                if arg is not type(None) and self._is_simple_type(arg):
                    return True
            return False

        return self._is_simple_type(tp)

    def _is_simple_type(self, tp: Any) -> bool:
        """Check if a type is simple (str, int, bool, float, Literal, Enum)."""
        # Primitives
        if tp in (str, int, bool, float):
            return True

        # Literal
        if get_origin(tp) is Literal:
            return True

        # Enum
        try:
            if isinstance(tp, type) and issubclass(tp, enum.Enum):
                return True
        except TypeError:
            pass

        return False

    def _is_simple_from_string(self) -> bool:
        """Fallback: string-based heuristic when evaluated_type unavailable."""
        type_str = self.type_annotation
        if not isinstance(type_str, str):
            return False

        type_str_lower = type_str.lower()
        simple_indicators = ["str", "int", "bool", "float"]
        complex_indicators = ["dict", "range", "ownership", "iterable[dict"]

        for indicator in complex_indicators:
            if indicator in type_str_lower:
                return False

        for indicator in simple_indicators:
            if indicator in type_str_lower:
                return True

        # Check for enum-like types: single PascalCase identifier
        if re.match(r"^[A-Z][a-zA-Z0-9]*$", type_str):
            return True

        return False

    @property
    def is_list_type(self) -> bool:
        """Check if field accepts multiple values (list/sequence).

        Uses evaluated_type to check if any Union member is a Sequence or Iterable.
        """
        import collections.abc

        if self.evaluated_type is not None:
            origin = get_origin(self.evaluated_type)
            if origin is Union:
                args = get_args(self.evaluated_type)
                for arg in args:
                    arg_origin = get_origin(arg)
                    if arg_origin is not None:
                        try:
                            if issubclass(
                                arg_origin,
                                (collections.abc.Sequence, collections.abc.Iterable),
                            ):
                                # Exclude str (technically a Sequence)
                                if arg_origin is not str:
                                    return True
                        except TypeError:
                            pass
            return False

        # Fallback to string-based detection
        type_str = self.type_annotation
        if isinstance(type_str, str):
            type_str_lower = type_str.lower()
            if "sequence" in type_str_lower or "iterable" in type_str_lower:
                return True
        return False

    @property
    def cli_type(self) -> str:
        """Get CLI type for this field (simplified).

        Returns full type including list[] wrapper if it's a list type.
        """
        type_str = self.python_type_name
        # Determine base type
        if "str" in type_str.lower():
            base_type = "str"
        elif "int" in type_str.lower():
            base_type = "int"
        elif "bool" in type_str.lower():
            base_type = "bool"
        elif "float" in type_str.lower():
            base_type = "float"
        else:
            # Default to string for complex types (enums, etc.)
            base_type = "str"

        # Wrap in list[] if it's a list type
        if self.is_list_type:
            return f"list[{base_type}]"
        return base_type


def is_explodable_typed_dict(type_annotation: Any) -> bool:
    """Check if type annotation is a TypedDict that should be exploded.

    Detects filter/search TypedDict params like CustomizationJobListFilterParam.

    Args:
        type_annotation: Type to check

    Returns:
        True if it's a TypedDict that should be exploded into individual CLI options
    """
    from nemo_helix_sdk_tools.sdk.cli_generator.typing_utils import get_union_args

    # Handle Union types (e.g., FilterParam | Omit)
    union_args = get_union_args(type_annotation)
    if union_args:
        for arg in union_args:
            if is_typed_dict(arg):
                return True
        return False

    return is_typed_dict(type_annotation)


def is_typed_dict(type_annotation: Any) -> bool:
    """Check if a type is a TypedDict.

    Args:
        type_annotation: Type to check

    Returns:
        True if it's a TypedDict
    """
    if type_annotation is None:
        return False

    if isinstance(type_annotation, type):
        try:
            if issubclass(type_annotation, dict) and hasattr(type_annotation, "__required_keys__"):
                return True
        except TypeError:
            pass

    return False


def get_typed_dict_class(type_annotation: Any) -> type | None:
    """Extract TypedDict class from a type annotation (handles Union with Omit).

    Args:
        type_annotation: Type annotation that may contain a TypedDict

    Returns:
        TypedDict class if found, None otherwise
    """
    from nemo_helix_sdk_tools.sdk.cli_generator.typing_utils import get_union_args

    args = get_union_args(type_annotation)
    if args:
        for arg in args:
            if is_typed_dict(arg):
                return arg
        return None

    if is_typed_dict(type_annotation):
        return type_annotation

    return None


def introspect_typed_dict(typed_dict_class: type) -> list[TypedDictField]:
    """Extract fields from a TypedDict class.

    Args:
        typed_dict_class: A TypedDict class to introspect

    Returns:
        List of TypedDictField objects representing the fields
    """
    if not is_typed_dict(typed_dict_class):
        return []

    fields = []
    required_keys = getattr(typed_dict_class, "__required_keys__", frozenset())
    annotations = getattr(typed_dict_class, "__annotations__", {})

    # Try to get evaluated type hints for better type detection
    evaluated_hints: dict[str, Any] = {}
    try:
        evaluated_hints = get_type_hints(typed_dict_class)
    except (NameError, TypeError):
        # Forward ref or type resolution issue - use raw annotations
        evaluated_hints = {}

    for field_name, field_type in annotations.items():
        # Get evaluated type if available
        evaluated_type = evaluated_hints.get(field_name)

        # Get string representation of the type for display (legacy)
        type_str = field_type
        if isinstance(field_type, ForwardRef):
            try:
                type_str = field_type.__forward_arg__
            except AttributeError:
                type_str = str(field_type)

        is_required = field_name in required_keys

        fields.append(
            TypedDictField(
                name=field_name,
                type_annotation=type_str,
                is_required=is_required,
                evaluated_type=evaluated_type,
            )
        )

    return fields
