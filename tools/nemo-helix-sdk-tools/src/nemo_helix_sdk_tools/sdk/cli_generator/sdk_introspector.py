# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SDK introspection for CLI generation.

This module introspects the SDK at generation time to extract method signatures,
parameter types, and docstrings for generating CLI commands.
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from typing import Any, Type, cast, get_origin, get_type_hints

from caseutil import to_pascal
from nemo_helix_sdk_tools.sdk.cli_generator.docstring_parser import ParsedDocstring
from nemo_helix_sdk_tools.sdk.cli_generator.type_formatter import format_type
from nemo_helix_sdk_tools.sdk.cli_generator.typed_dict_utils import (
    TypedDictField,
    get_typed_dict_class,
    introspect_typed_dict,
    is_explodable_typed_dict,
)


@dataclass
class SDKParameter:
    """Represents a parameter from an SDK method."""

    name: str
    type_annotation: Type
    default: Any
    is_required: bool
    is_positional: bool
    description: str | None = None

    @property
    def is_path_param(self) -> bool:
        """Check if this is a path parameter (positional argument)."""
        return self.is_positional and self.is_required

    @property
    def python_type_name(self) -> str:
        """Get a string representation of the Python type."""
        if self.type_annotation == inspect.Parameter.empty:
            return "Any"
        return format_type(self.type_annotation)

    @property
    def is_dict_type(self) -> bool:
        origin = get_origin(self.type_annotation)
        return origin is dict or (isinstance(self.type_annotation, type) and issubclass(self.type_annotation, dict))

    @property
    def is_list_type(self) -> bool:
        origin = get_origin(self.type_annotation)
        return origin is list or (isinstance(self.type_annotation, type) and issubclass(self.type_annotation, list))

    @property
    def is_simple_string_list(self) -> bool:
        """Check if this is a simple list of strings (list[str], Sequence[str], SequenceNotStr[str]).

        These can be represented as repeatable CLI options instead of JSON strings.
        """
        import types
        from collections.abc import Sequence
        from typing import Union, get_args

        type_ann = self.type_annotation
        origin = get_origin(type_ann)

        # Handle Union types (including X | Y syntax via types.UnionType)
        if origin is Union or origin is types.UnionType:
            args = get_args(type_ann)
            # Filter out None and Omit
            non_none_args = [
                arg
                for arg in args
                if arg is not type(None) and not (hasattr(arg, "__name__") and arg.__name__ == "Omit")
            ]
            # If we have exactly one non-None arg, check that
            if len(non_none_args) == 1:
                type_ann = non_none_args[0]
                origin = get_origin(type_ann)

        # Check if it's a list/Sequence type
        if origin is list:
            inner_args = get_args(type_ann)
            if inner_args and inner_args[0] is str:
                return True
        elif origin is Sequence or (hasattr(origin, "__name__") and "Sequence" in origin.__name__):
            inner_args = get_args(type_ann)
            if inner_args and inner_args[0] is str:
                return True
        # Handle SequenceNotStr (it's a Protocol, so check by name)
        elif hasattr(type_ann, "__origin__") and hasattr(type_ann.__origin__, "__name__"):
            if "SequenceNotStr" in type_ann.__origin__.__name__:
                inner_args = get_args(type_ann)
                if inner_args and inner_args[0] is str:
                    return True

        return False

    @property
    def is_explodable_typed_dict(self) -> bool:
        """Check if this parameter should be exploded into individual options."""
        return is_explodable_typed_dict(self.type_annotation)

    @property
    def typed_dict_fields(self) -> list[TypedDictField]:
        """Get the fields if this is an explodable TypedDict."""
        typed_dict_class = get_typed_dict_class(self.type_annotation)
        if typed_dict_class:
            return introspect_typed_dict(typed_dict_class)
        return []

    @property
    def typed_dict_class_name(self) -> str | None:
        """Get the class name if this is an explodable TypedDict."""
        typed_dict_class = get_typed_dict_class(self.type_annotation)
        if typed_dict_class:
            return typed_dict_class.__name__
        return None


@dataclass
class SDKMethod:
    """Represents a method from an SDK resource."""

    name: str
    resource_path: list[str]
    parameters: list[SDKParameter]
    return_type: Type
    docstring: str | None = None
    _parsed_docstring: ParsedDocstring | None = None

    @property
    def parsed_docstring(self) -> ParsedDocstring:
        """Get parsed docstring with description and param descriptions."""
        if self._parsed_docstring is None:
            self._parsed_docstring = ParsedDocstring.parse(self.docstring)
        return self._parsed_docstring

    @property
    def description(self) -> str:
        """Get the method description from docstring."""
        return self.parsed_docstring.description

    def get_param_description(self, param_name: str) -> str | None:
        """Get the description for a specific parameter."""
        return self.parsed_docstring.param_descriptions.get(param_name)

    @property
    def path_parameters(self) -> list[SDKParameter]:
        """Get path parameters (required positional args)."""
        return [p for p in self.parameters if p.is_path_param]

    @property
    def optional_parameters(self) -> list[SDKParameter]:
        """Get optional keyword parameters."""
        return [p for p in self.parameters if not p.is_path_param]


class SDKIntrospector:
    """Introspects SDK resources to extract method metadata."""

    def __init__(self, sdk_package_path: str = "nemo_helix"):
        self.sdk_package_path = sdk_package_path

    def introspect_resource(self, resource_path: list[str]) -> dict[str, SDKMethod]:
        """Introspect an SDK resource and return its methods.

        Args:
            resource_path: Path to the resource (e.g., ["customization", "configs"])

        Returns:
            Dictionary mapping method names to SDKMethod objects
        """
        try:
            module_path = f"{self.sdk_package_path}.resources"
            for part in resource_path:
                module_path += f".{part}"

            module = importlib.import_module(module_path)

            # Find the resource class (e.g., DatasetsResource)
            resource_class_name = to_pascal(resource_path[-1]) + "Resource"
            if not hasattr(module, resource_class_name):
                return {}

            resource_class = getattr(module, resource_class_name)

            methods = {}
            for name, method in inspect.getmembers(resource_class, predicate=inspect.isfunction):
                if name.startswith("_"):
                    continue

                # FIXME: Methods should be coming from the Stainless config, not from the code.
                if name in [
                    "with_raw_response",
                    "with_streaming_response",
                    "create_from_dict",
                ]:
                    continue

                sdk_method = self._introspect_method(method, resource_path)
                if sdk_method:
                    methods[name] = sdk_method

            return methods

        except (ImportError, AttributeError) as e:
            raise RuntimeError(f"Failed to import {resource_path}: {e}")

    def _introspect_method(self, method: Any, resource_path: list[str]) -> SDKMethod | None:
        """Introspect a single method and extract its metadata."""
        try:
            type_hints = get_type_hints(method)
            sig = inspect.signature(method)

            parameters = []
            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue

                type_annotation = type_hints.get(param_name, param.annotation)
                # Parameters before `*` in signature are positional (POSITIONAL_OR_KEYWORD)
                # Parameters before `/` are positional-only (POSITIONAL_ONLY)
                # Both should map to CLI positional arguments when required
                is_positional = param.kind in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
                sdk_param = SDKParameter(
                    name=param_name,
                    type_annotation=type_annotation,
                    default=param.default,
                    is_positional=is_positional,
                    is_required=param.default == inspect.Parameter.empty,
                )
                parameters.append(sdk_param)

            # Extract docstring
            docstring = inspect.getdoc(method)

            return SDKMethod(
                name=method.__name__,
                resource_path=resource_path,
                parameters=parameters,
                return_type=cast(Type, type_hints.get("return")),
                docstring=docstring,
            )

        except (TypeError, NameError):
            # Type resolution failed - skip this method
            # TODO: We should actually fail here
            return None
