# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Format type annotations as strings for CLI generation.

This module consolidates all type formatting logic into a single place.
Work with actual Type objects, not string representations.
"""

from __future__ import annotations

import collections.abc
import inspect
from typing import Any, ForwardRef, Literal, Type, Union, get_args, get_origin

from nemo_helix_sdk_tools.sdk.cli_generator.typing_utils import get_union_args


def format_type(tp: Type) -> str:
    """Format type for CLI signatures.

    Handles all type objects including primitives, generics, unions, etc.
    This is the main function for formatting types in generated CLI code.

    Args:
        tp: Type object to format

    Returns:
        String representation suitable for CLI signatures
    """
    if tp == inspect.Parameter.empty:
        return "Any"

    if isinstance(tp, str):
        return tp

    origin = get_origin(tp)

    # Handle Literal types
    if origin is Literal:
        args = get_args(tp)
        args_str = ", ".join(repr(arg) for arg in args)
        return f"Literal[{args_str}]"

    # Handle generic types (List, Dict, Optional, etc.)
    if origin is not None:
        args = get_args(tp)
        if args:
            args_str = ", ".join(format_type(arg) for arg in args)
            origin_name = getattr(origin, "__name__", repr(origin))
            return f"{origin_name}[{args_str}]"

    # Handle NoneType specially
    if tp is type(None):
        return "None"

    # Handle object type - convert to str for CLI (can't handle arbitrary objects)
    if tp is object:
        return "str"

    # Handle built-in types and classes
    if hasattr(tp, "__name__"):
        return tp.__name__

    return repr(tp)


def format_type_for_help(tp: Type) -> str:
    """Format type for help text display (simplified).

    Used in help messages and schemas where we want a simpler representation.

    Args:
        tp: Type object to format

    Returns:
        Simplified string representation for help text
    """
    if isinstance(tp, str):
        # Extract base type from strings
        if "str" in tp.lower():
            return "str"
        if "int" in tp.lower():
            return "int"
        if "bool" in tp.lower():
            return "bool"
        if "dict" in tp.lower():
            return "dict"
        if "datetime" in tp.lower():
            return "str"  # datetimes are typically str in JSON
        return tp

    # Handle ForwardRef
    if isinstance(tp, ForwardRef):
        try:
            arg = tp.__forward_arg__
            return format_type_for_help(arg)
        except AttributeError:
            return str(tp)

    origin = get_origin(tp)
    args = get_args(tp)

    # Handle Literal types as user-facing allowed values (without "Literal[...]"
    # type-system syntax) so JSON examples are easier to follow.
    if origin is Literal:
        return " | ".join(repr(arg) for arg in args)

    # Handle Annotated types (strip the annotations)
    if hasattr(tp, "__metadata__") or (origin and getattr(origin, "__name__", "") == "Annotated"):
        if args:
            return format_type_for_help(args[0])

    # Handle Union types
    if origin is Union:
        non_none_args = [a for a in args if a is not type(None)]
        if len(non_none_args) == 1:
            return format_type_for_help(non_none_args[0])
        if non_none_args:
            return format_type_for_help(non_none_args[0])
        return "Any"

    # Handle dict
    if origin is dict:
        if args and len(args) >= 2:
            key_type = format_type_for_help(args[0])
            value_type = format_type_for_help(args[1])
            if key_type == "str" and value_type == "str":
                return "dict[str, str]"
            elif key_type == "str":
                return f"dict[str, {value_type}]"
            return f"dict[{key_type}, {value_type}]"
        return "dict"

    # Handle list/sequence types
    is_sequence = origin in (list, tuple)
    if not is_sequence and origin:
        try:
            is_sequence = issubclass(origin, (collections.abc.Sequence, collections.abc.Iterable))
        except TypeError:
            is_sequence = False

    if is_sequence:
        if args:
            inner = format_type_for_help(args[0])
            return f"[{inner}]"
        return "list"

    # Simple type with __name__
    if hasattr(tp, "__name__"):
        name = tp.__name__
        # Convert datetime to str for JSON context
        if name == "datetime":
            return "str"
        return name

    return str(tp)


def get_type_schema(tp: Type) -> str | None:
    """Get schema for complex types like TypedDict.

    For TypedDicts, returns something like: {start: str, end: str}
    For Dict types, returns: dict[key, value]
    For simple types, returns None (no schema needed).

    Args:
        tp: Type to get schema for

    Returns:
        Compact schema string, or None for simple types
    """
    # Handle Union types - find the concrete type
    origin = get_origin(tp)
    if origin is Union:
        args = get_args(tp)
        for arg in args:
            if arg is type(None):
                continue
            schema = get_type_schema(arg)
            if schema:
                return schema
        return None

    # Handle TypedDict
    if _is_typed_dict(tp):
        return _format_typed_dict_schema(tp)

    # Handle Dict types
    if origin is dict:
        args = get_args(tp)
        if args:
            key_type = format_type_for_help(args[0])
            value_type = format_type_for_help(args[1])
            return f"dict[{key_type}, {value_type}]"
        return "dict"

    # Handle Iterable/Sequence of TypedDicts
    is_iterable = origin in (list, tuple)
    if not is_iterable and origin:
        try:
            is_iterable = issubclass(origin, collections.abc.Iterable)
        except TypeError:
            is_iterable = False

    if is_iterable:
        args = get_args(tp)
        if args:
            inner_type = args[0]
            inner_schema = get_type_schema(args[0])
            if inner_schema:
                return f"[{inner_schema}, ...]"
            # Fall back to a simple inner type so list[str] and similar
            # annotations still produce actionable JSON-only field hints.
            inner_type_schema = format_type_for_help(inner_type)
            inner_origin = get_origin(inner_type)
            if inner_origin is Literal:
                return f"[{inner_type_schema}]"
            if inner_origin and getattr(inner_origin, "__name__", "") == "Annotated":
                inner_args = get_args(inner_type)
                if inner_args and get_origin(inner_args[0]) is Literal:
                    return f"[{inner_type_schema}]"
            return f"[{inner_type_schema}, ...]"
        return None

    return None


def _is_typed_dict(tp: Any) -> bool:
    """Check if a type is a TypedDict."""
    if tp is None:
        return False
    if isinstance(tp, type):
        if issubclass(tp, dict) and hasattr(tp, "__required_keys__"):
            return True
    return False


def _format_typed_dict_schema(typed_dict_class: type) -> str:
    """Format a TypedDict as a compact schema string.

    Returns something like: {start: str, end: str}

    Args:
        typed_dict_class: TypedDict class to format

    Returns:
        Compact schema representation
    """
    if not _is_typed_dict(typed_dict_class):
        return str(typed_dict_class.__name__)

    # Use get_type_hints to resolve ForwardRefs and Annotated types
    try:
        from typing import get_type_hints

        resolved_hints = get_type_hints(typed_dict_class, include_extras=False)
    except (NameError, TypeError):
        # Fall back to raw annotations if resolution fails
        resolved_hints = getattr(typed_dict_class, "__annotations__", {})

    parts = []
    for field_name, field_type in resolved_hints.items():
        type_name = format_type_for_help(field_type)
        parts.append(f"{field_name}: {type_name}")

    return "{" + ", ".join(parts) + "}"


def extract_imports_from_type(tp: Type) -> set[str]:
    """Extract import statements needed for a type annotation.

    Args:
        tp: Type to extract imports from

    Returns:
        Set of import lines needed (e.g., "from typing import Sequence")
    """
    imports = set()

    if tp == inspect.Parameter.empty or tp is None:
        return imports

    # Get the origin of generic types
    union_args = get_union_args(tp)
    if union_args is not None:
        for arg in union_args:
            imports.update(extract_imports_from_type(arg))

    # Handle datetime
    if hasattr(tp, "__name__") and tp.__name__ == "datetime":
        imports.add("from datetime import datetime")
        return imports

    # Don't import builtins
    if tp.__module__ == "builtins":
        return imports

    # Main import
    imports.add(f"from {tp.__module__} import {tp.__name__}")

    return imports
