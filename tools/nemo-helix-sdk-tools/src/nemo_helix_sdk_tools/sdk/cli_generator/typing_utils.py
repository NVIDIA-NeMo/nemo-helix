# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import UnionType
from typing import Literal, Type, Union, get_args, get_origin


def get_union_args(tp: Type) -> tuple[Type, ...] | None:
    """
    Returns the arguments of a Union type or None if `type` is not a Union.
    Works with both Union[...] and X | Y syntax.
    """
    # Check if it's either type of union
    if isinstance(tp, UnionType) or get_origin(tp) is Union:
        return get_args(tp)

    return None


def format_type(tp: Type) -> str:
    """Format a type annotation as a string for code generation."""
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

    # Handle built-in types and classes
    if hasattr(tp, "__name__"):
        return tp.__name__

    # Fallback for other typing constructs
    return repr(tp)
