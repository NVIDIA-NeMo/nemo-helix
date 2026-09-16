# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared data models and types for CLI generation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Type

from nemo_helix_sdk_tools.sdk.cli_generator.type_formatter import format_type
from nemo_helix_sdk_tools.sdk.cli_generator.typing_utils import get_union_args


@dataclass
class PathParam:
    """Path parameter (positional argument) for CLI command."""

    var_name: str
    type: str
    help: str | None

    def to_typer_argument(self) -> str:
        """Generate Typer argument signature line."""
        if self.help:
            return f'{self.var_name}: Annotated[{self.type}, typer.Argument(help="{self.help}")],'
        return f"{self.var_name}: Annotated[{self.type}, typer.Argument()],"


@dataclass
class Parameter:
    """Optional parameter (option) for CLI command."""

    var_name: str
    type: str
    option_args: str
    default: str | None = None
    needs_json_parse: bool = False  # If True, parse CLI value as JSON before using
    is_required: bool = False  # If True, field is required by SDK (validated after JSON merge)
    is_list_type: bool = False  # If True, this is a repeatable list option (e.g., --roles Admin --roles Editor)
    help: str | None = None  # Help text for this parameter
    is_positional: bool = False  # If True, render as typer.Argument instead of typer.Option

    def to_typer_option(self) -> str:
        """Generate Typer option signature line."""
        option = f"{self.var_name}: Annotated[{self.type}, typer.Option({self.option_args})]"
        if self.default is not None:
            option += f" = {self.default}"
        return option + ","

    def to_typer_argument(self) -> str:
        """Generate Typer argument (positional) signature line."""
        if self.help:
            arg = (
                f'{self.var_name}: Annotated[{self.type}, typer.Argument(help="{escape_for_python_string(self.help)}")]'
            )
        else:
            arg = f"{self.var_name}: Annotated[{self.type}, typer.Argument()]"
        if self.default is not None:
            arg += f" = {self.default}"
        return arg + ","


@dataclass
class KwargsEntry:
    """Entry in the build_kwargs call."""

    sdk_name: str
    value_expr: str

    def to_kwargs_line(self) -> str:
        """Generate kwargs assignment line."""
        return f"{self.sdk_name}={self.value_expr},"


@dataclass
class ExplodedField:
    """Simple field from an exploded TypedDict parameter."""

    var_name: str
    field_name: str
    cli_type: str
    cli_option: str


_QUERY_PARAM_LINE = re.compile(r"^-\s+.*\?(?:filter|search)\[", re.MULTILINE)


def strip_api_only_lines(text: str) -> str:
    """Strip lines documenting HTTP query parameter syntax from help text.

    Lines like ``- Bracket notation: ?filter[name][$like]=value`` describe
    API query parameter conventions that don't apply to CLI users.  This
    function removes any bullet line containing ``?filter[`` or ``?search[``
    patterns, which are inherently API-level documentation.
    """
    lines = text.split("\n")
    kept = [line for line in lines if not _QUERY_PARAM_LINE.search(line)]
    return "\n".join(kept).rstrip()


def escape_for_python_string(text: str | None) -> str | None:
    """Escape text for embedding in a Python string literal."""
    if text is None:
        return None

    text = strip_api_only_lines(text)

    return (
        text.replace("\\", "\\\\")  # Escape backslashes first
        .replace('"', '\\"')  # Escape double quotes
        .replace("'", "\\'")  # Escape single quotes
        .replace("\n", "\\n")  # Escape newlines
    )


def sanitize_help_text(text: str | None) -> str | None:
    """Sanitize help text for use in docstrings.

    Fixes escape sequences that cause Python warnings (e.g. \\* -> *).
    Strips API-only lines (HTTP query parameter documentation).
    Preserves newlines and other formatting.
    """
    if text is None:
        return None

    text = strip_api_only_lines(text)

    # SDK generates \** from OpenAPI * (double-escaping issue)
    # Replace \** with * first, then handle remaining \* -> *
    text = text.replace("\\**", "*").replace("\\*", "*")

    # Fix Stainless-generated HTTP method examples with extra spaces around / and -
    def _fix_http_path(m: re.Match) -> str:
        method = m.group(1)
        path = m.group(2)
        path = re.sub(r"\s*/\s*", "/", path)
        path = re.sub(r"\s+-\s+", "-", path)
        path = re.sub(r"\s+@\s+", "@", path)
        return f"{method} {path}"

    text = re.sub(
        r"\b(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\s+(/.+)$",
        _fix_http_path,
        text,
        flags=re.MULTILINE,
    )

    return text


def _simplify_list_type(tp: Type) -> str | None:
    """Simplify list types with complex sub-types for Typer compatibility.

    Typer doesn't support list[Literal[...]] or other complex list element types.
    Simplify these to list[str].

    Returns simplified type string, or None if not a list type.
    """
    from typing import Literal, get_args, get_origin

    origin = get_origin(tp)
    if origin is not list:
        return None

    args = get_args(tp)
    if not args:
        return "list"

    element_type = args[0]
    element_origin = get_origin(element_type)

    # Simple element types that Typer can handle
    if element_type in (str, int, float, bool):
        return f"list[{element_type.__name__}]"

    # Complex element types (Literal, unions, etc.) - simplify to list[str]
    if element_origin is Literal:
        return "list[str]"

    # Default: simplify to list[str]
    return "list[str]"


def _collapse_typer_union_types(type_names: list[str]) -> list[str]:
    """Collapse multi-type unions into annotations Typer can build.

    Example: Literal["positive", "negative"] | str | float | None
    becomes str | None because Typer cannot build one option from multiple
    concrete runtime types.
    """
    concrete_types = [type_name for type_name in type_names if type_name != "None"]
    if len(concrete_types) <= 1:
        return type_names

    has_none = len(concrete_types) != len(type_names)
    if set(concrete_types) <= {"int", "float"}:
        collapsed = ["float"]
    else:
        collapsed = ["str"]

    if has_none:
        collapsed.append("None")
    return collapsed


def clean_type_annotation(tp: Type) -> str:
    """Clean up a type annotation string for CLI use.

    - Replace '| Omit' with '| None'
    - Simplify 'Literal[False] | Literal[True]' to 'bool'
    - Merge multiple Literal types into a single Literal
    - Simplify list[Literal[...]] to list[str]
    - Simplify dict[...] to str (Typer doesn't support dict options)
    - Deduplicate type names
    """
    from typing import Literal, get_args, get_origin

    # Handle dict types - Typer doesn't support dict options, use str (JSON)
    origin = get_origin(tp)
    if origin is dict:
        return "str"

    # Handle list types with complex sub-types
    simplified_list = _simplify_list_type(tp)
    if simplified_list is not None:
        return simplified_list

    # Replace Omit with None
    union_types = get_union_args(tp)
    if union_types is not None:
        has_none = False
        bool_literal_values: set[bool] = set()
        string_literal_values: list[str] = []
        other_types: list[str] = []

        for t in union_types:
            if t is type(None):
                has_none = True
            elif hasattr(t, "__name__") and t.__name__ == "Omit":
                has_none = True
            elif get_origin(t) is Literal:
                args = get_args(t)
                for arg in args:
                    if isinstance(arg, bool):
                        bool_literal_values.add(arg)
                    elif isinstance(arg, str):
                        string_literal_values.append(arg)
                    else:
                        # Other literal values (int, etc.) - add as string
                        string_literal_values.append(repr(arg))
            elif get_origin(t) is list:
                # Handle list types in unions
                simplified = _simplify_list_type(t)
                if simplified:
                    other_types.append(simplified)
                else:
                    other_types.append(format_type(t))
            elif get_origin(t) is dict:
                # Handle dict types in unions - Typer doesn't support dict, use str
                other_types.append("str")
            else:
                other_types.append(format_type(t))

        result_types: list[str] = []

        # If we have exactly Literal[True] and Literal[False], simplify to bool
        if bool_literal_values == {True, False}:
            result_types.append("bool")
        elif bool_literal_values:
            # Incomplete bool set - add as literals
            bool_args = ", ".join(repr(v) for v in sorted(bool_literal_values))
            result_types.append(f"Literal[{bool_args}]")

        # Merge all string literals into a single Literal type
        if string_literal_values:
            # Deduplicate while preserving order
            seen_literals: set[str] = set()
            unique_literals: list[str] = []
            for val in string_literal_values:
                if val not in seen_literals:
                    seen_literals.add(val)
                    unique_literals.append(val)
            literal_args = ", ".join(repr(v) for v in unique_literals)
            result_types.append(f"Literal[{literal_args}]")

        result_types.extend(other_types)

        if has_none:
            result_types.append("None")

        # Deduplicate while preserving order
        seen: set[str] = set()
        deduped: list[str] = []
        for t in result_types:
            if t not in seen:
                seen.add(t)
                deduped.append(t)

        deduped = _collapse_typer_union_types(deduped)
        return " | ".join(deduped)

    return format_type(tp)
