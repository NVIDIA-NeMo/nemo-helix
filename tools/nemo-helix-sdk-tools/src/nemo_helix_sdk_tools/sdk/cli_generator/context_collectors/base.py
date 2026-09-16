# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base context collector with shared utilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from nemo_helix_ext.cli.core.help_formatter import _REQUIRED_SUFFIX
from nemo_helix_sdk_tools.sdk.cli_generator.config import CLIConfig
from nemo_helix_sdk_tools.sdk.cli_generator.docstring_parser import transform_query_to_cli
from nemo_helix_sdk_tools.sdk.cli_generator.models import (
    ExplodedField,
    KwargsEntry,
    Parameter,
    PathParam,
    clean_type_annotation,
    escape_for_python_string,
)
from nemo_helix_sdk_tools.sdk.cli_generator.sdk_introspector import (
    SDKMethod,
    SDKParameter,
)
from nemo_helix_sdk_tools.sdk.cli_generator.type_formatter import get_type_schema

_UNSET = object()


def build_required_fields_example(params: list[Parameter], required_fields: list[str]) -> str:
    """Build a JSON example string for required fields.

    Uses ``{}`` for fields that need JSON parsing (objects) and ``"value"`` for
    simple string fields, so the example is syntactically valid for the API.

    Returns a string like ``"name": "value", "plugins": {}, "run": {}``
    (without outer braces — the template wraps it in ``{...}``).
    """
    json_fields: dict[str, bool] = {p.var_name: p.needs_json_parse for p in params}
    parts = []
    for f in required_fields:
        value = "{}" if json_fields.get(f) else '"value"'
        parts.append(f'"{f}": {value}')
    return ", ".join(parts)


def promote_name_to_positional(
    params: list[Parameter],
    resource_path: list[str],
    method_name: str,
    cli_config: CLIConfig,
    *,
    type_str: str | None = None,
    default: object = _UNSET,
) -> bool:
    """Promote the 'name' parameter to positional if config allows.

    Reads ``name_positional`` from the method config (default ``True``). When
    ``False``, ``name`` is left as a regular CLI option — used for gateway proxy
    commands where ``name`` is a workspace filter, not a resource identifier.

    Args:
        params: List of Parameters to search through.
        resource_path: Resource path for config lookup.
        method_name: Method name for config lookup.
        cli_config: CLI configuration object.
        type_str: If not None, override ``param.type`` with this value.
        default: If not ``_UNSET``, override ``param.default`` with this value.
            Pass ``None`` (Python None) for required (no default emitted),
            or ``"None"`` (the string) for optional (emits ``= None``).

    Returns:
        ``True`` if the name parameter was found and promoted.
    """
    method_config = cli_config.get_method_config(resource_path, method_name) or {}
    if not method_config.get("name_positional", True):
        return False
    for param in params:
        if param.var_name == "name":
            param.is_positional = True
            if type_str is not None:
                param.type = type_str
            if default is not _UNSET:
                param.default = default  # type: ignore[assignment]
            return True
    return False


def is_simple_cli_type(param: SDKParameter) -> bool:
    """Check if a parameter is a simple type that should get a CLI flag.

    Simple types: str, int, float, bool, Literal (and their Optional/Union variants)

    Args:
        param: SDK parameter to check

    Returns:
        True if parameter should get a CLI flag
    """
    import types
    from datetime import datetime
    from typing import Literal, Union, get_args, get_origin

    # Skip if it's a dict or list
    if param.is_dict_type or param.is_list_type:
        return False

    # Skip if it's a TypedDict
    if param.is_explodable_typed_dict:
        return False

    type_ann = param.type_annotation
    origin = get_origin(type_ann)

    # Literal types are simple (string enums)
    if origin is Literal:
        return True

    # Handle Union types (both Union[...] and X | Y syntax)
    # Union[...] has origin = Union, X | Y has origin = types.UnionType
    if origin is Union or origin is types.UnionType:
        args = get_args(type_ann)
        # Check all types in the union
        has_simple_type = False
        for arg in args:
            # Skip None and Omit (these are OK to have in the union)
            if arg is type(None):
                continue
            if hasattr(arg, "__name__") and arg.__name__ == "Omit":
                continue
            # If any type is datetime, skip this parameter
            if arg is datetime or (
                hasattr(arg, "__module__")
                and hasattr(arg, "__name__")
                and arg.__module__ == "datetime"
                and arg.__name__ == "datetime"
            ):
                return False
            # Check if this type is simple (including Literal)
            arg_origin = get_origin(arg)
            if arg in (str, int, float, bool) or arg_origin is Literal:
                has_simple_type = True
            else:
                # Found a non-simple, non-datetime type - skip this parameter
                return False
        # All non-None/Omit types are simple, and at least one exists
        return has_simple_type

    # Direct simple types
    return type_ann in (str, int, float, bool)


def build_body_params(
    sdk_method: SDKMethod,
    skip_params: list[str] | None = None,
) -> tuple[list[Parameter], list[str]]:
    """Extract body parameters as CLI options (for create/update).

    Args:
        sdk_method: SDK method metadata
        skip_params: Parameter names to skip

    Returns:
        Tuple of (body_params, required_field_names)
    """
    if skip_params is None:
        skip_params = ["extra_headers", "extra_query", "extra_body", "timeout"]

    body_params: list[Parameter] = []
    required_fields: list[str] = []

    for param in sdk_method.optional_parameters:
        if param.name in skip_params:
            continue

        help_text = sdk_method.get_param_description(param.name)
        cli_option = f"--{param.name}".replace("_", "-")

        # Check if this is a simple type (no JSON parsing needed)
        is_simple = is_simple_cli_type(param)
        # Check if this is a simple string list (repeatable option)
        is_string_list = param.is_simple_string_list

        if is_string_list:
            # Simple list of strings: use list[str] with repeatable option
            clean_type = "list[str] | None"
            needs_json = False
            # Add hint about repeating the option
            if help_text:
                help_text = f"{help_text} (can be repeated)"
            else:
                help_text = "Can be repeated for multiple values"
        elif is_simple:
            # Simple types: use the cleaned type annotation.
            # Always add | None because default is always "None" at Typer level
            # (required fields are validated post-merge, not at the Typer signature).
            clean_type = clean_type_annotation(param.type_annotation)
            if "None" not in clean_type:
                clean_type = f"{clean_type} | None"
            needs_json = False
        else:
            # Complex types: accept as string, parse as JSON
            clean_type = "str | None"
            needs_json = True
            # Add hint to help text
            if help_text:
                help_text = f"{help_text} (JSON string)"
            else:
                help_text = "JSON string"

        # Track required fields for post-merge validation
        if param.is_required:
            required_fields.append(param.name)
            help_text = f"{help_text}{_REQUIRED_SUFFIX}" if help_text else _REQUIRED_SUFFIX.lstrip()

        if help_text:
            option_args = f'"{cli_option}", help="{escape_for_python_string(help_text)}"'
        else:
            option_args = f'"{cli_option}"'

        # Always use None at Typer level - validate after JSON merge
        default = "None"
        body_params.append(
            Parameter(
                var_name=param.name,
                type=clean_type,
                option_args=option_args,
                default=default,
                needs_json_parse=needs_json,
                is_required=param.is_required,
                is_list_type=is_string_list,
                help=help_text,
            )
        )

    return body_params, required_fields


def build_path_params(
    sdk_method: SDKMethod,
    skip_params: list[str] | None = None,
    param_help_overrides: dict[str, str] | None = None,
) -> list[PathParam]:
    """Extract path parameters (required positional args).

    Args:
        sdk_method: SDK method metadata
        skip_params: Parameter names to skip
        param_help_overrides: Optional dict mapping param name to help text, overriding
            the SDK description for that parameter.

    Returns:
        List of path parameters
    """
    if skip_params is None:
        skip_params = ["extra_headers", "extra_query", "extra_body", "timeout"]

    path_params: list[PathParam] = []

    for param in sdk_method.path_parameters:
        if param.name in skip_params:
            continue

        if param_help_overrides and param.name in param_help_overrides:
            help_text = param_help_overrides[param.name]
        else:
            help_text = sdk_method.get_param_description(param.name)
        clean_type = clean_type_annotation(param.type_annotation)

        path_params.append(
            PathParam(
                var_name=param.name,
                type=clean_type,
                help=escape_for_python_string(help_text),
            )
        )

    return path_params


class BaseContextCollector(ABC):
    """Abstract base for all context collectors.

    Context collectors extract metadata from SDK methods and prepare
    template variables for code generation.
    """

    def __init__(self, cli_config: CLIConfig):
        """Initialize collector.

        Args:
            cli_config: CLI configuration for column definitions, etc.
        """
        self._cli_config = cli_config

    @abstractmethod
    def collect(
        self,
        resource_path: list[str],
        sdk_method: SDKMethod,
        method_name: str,
    ) -> dict[str, Any]:
        """Collect template context for this operation type.

        Args:
            resource_path: Resource path (e.g., ["customization", "jobs"])
            sdk_method: SDK method metadata
            method_name: Method name (e.g., "list")

        Returns:
            Dictionary with template variables
        """
        pass

    # Shared utilities all collectors need

    def _build_optional_params(
        self,
        sdk_method: SDKMethod,
        skip_params: list[str] | None = None,
    ) -> tuple[list[Parameter], list[KwargsEntry], bool]:
        """Extract optional parameters (CLI options).

        Args:
            sdk_method: SDK method metadata
            skip_params: Parameter names to skip

        Returns:
            Tuple of (parameters, kwargs_entries, has_exploded_params)
        """
        if skip_params is None:
            skip_params = ["extra_headers", "extra_query", "extra_body", "timeout"]

        parameters: list[Parameter] = []
        kwargs_entries: list[KwargsEntry] = []
        has_exploded_params = False

        for param in sdk_method.optional_parameters:
            if param.name in skip_params:
                continue

            if param.is_explodable_typed_dict:
                has_exploded_params = True
                self._add_exploded_param(param, sdk_method, parameters, kwargs_entries)
            else:
                self._add_regular_param(param, sdk_method, parameters, kwargs_entries)

        return parameters, kwargs_entries, has_exploded_params

    def _add_regular_param(
        self,
        param: SDKParameter,
        sdk_method: SDKMethod,
        parameters: list[Parameter],
        kwargs_entries: list[KwargsEntry],
    ) -> None:
        """Add a regular (non-exploded) parameter.

        Args:
            param: SDK parameter
            sdk_method: SDK method (for description lookup)
            parameters: List to append Parameter to
            kwargs_entries: List to append KwargsEntry to
        """
        help_text = sdk_method.get_param_description(param.name)
        clean_type = clean_type_annotation(param.type_annotation)
        cli_option = f"--{param.name}".replace("_", "-")

        if help_text:
            option_args = f'"{cli_option}", help="{escape_for_python_string(help_text)}"'
        else:
            option_args = f'"{cli_option}"'

        # Required keyword params use ... as default, optional use None
        default = "..." if param.is_required else "None"

        parameters.append(
            Parameter(
                var_name=param.name,
                type=clean_type,
                option_args=option_args,
                default=default,
            )
        )
        kwargs_entries.append(KwargsEntry(sdk_name=param.name, value_expr=param.name))

    def _add_exploded_param(
        self,
        param: SDKParameter,
        sdk_method: SDKMethod,
        parameters: list[Parameter],
        kwargs_entries: list[KwargsEntry],
    ) -> None:
        """Add an exploded TypedDict parameter (filter/search).

        Args:
            param: SDK parameter with TypedDict type
            sdk_method: SDK method (for description lookup)
            parameters: List to append Parameters to
            kwargs_entries: List to append KwargsEntry to
        """
        param_dash = param.name.replace("_", "-")
        panel_name = f"{param.name.capitalize()} Options"

        simple_fields: list[ExplodedField] = []
        complex_fields: list[dict[str, str]] = []

        for field in param.typed_dict_fields:
            if not field.is_simple_cli_type:
                schema = get_type_schema(field.evaluated_type or field.type_annotation)
                complex_fields.append({"name": field.name, "schema": schema or "object"})
                continue

            cli_option = f"--{param.name}.{field.name}".replace("_", "-")
            simple_fields.append(
                ExplodedField(
                    var_name=f"{param.name}_{field.name}",
                    field_name=field.name,
                    cli_type=field.cli_type,
                    cli_option=cli_option,
                )
            )

        help_text = self._build_exploded_help_text(param, sdk_method, param_dash, complex_fields)

        # Add the main JSON parameter
        parameters.append(
            Parameter(
                var_name=param.name,
                type="str | None",
                option_args=(
                    f'"--{param_dash}", '
                    f'metavar="{param.name.upper()}_JSON", '
                    f'help="{escape_for_python_string(help_text)}", '
                    f'rich_help_panel="{panel_name}"'
                ),
                default="None",
            )
        )

        # Add individual field parameters
        for field in simple_fields:
            parameters.append(
                Parameter(
                    var_name=field.var_name,
                    type=f"{field.cli_type} | None",
                    option_args=f'"{field.cli_option}", rich_help_panel="{panel_name}"',
                    default="None",
                )
            )

        # Add kwargs entry that merges both
        field_args = ", ".join(f"{f.field_name}={f.var_name}" for f in simple_fields)
        kwargs_entries.append(
            KwargsEntry(
                sdk_name=param.name,
                value_expr=f"merge_filter_dict({param.name}, {field_args})",
            )
        )

    def _build_exploded_help_text(
        self,
        param: SDKParameter,
        sdk_method: SDKMethod,
        param_dash: str,
        complex_fields: list[dict[str, str]],
    ) -> str:
        """Build help text for an exploded parameter's JSON option.

        Args:
            param: SDK parameter
            sdk_method: SDK method (for description lookup)
            param_dash: Parameter name with dashes
            complex_fields: List of complex fields that need JSON

        Returns:
            Help text string
        """
        usage_prefix = (
            f"Use --{param_dash} with JSON for complex/nested queries, "
            f"or --{param_dash}.FIELD options for simple fields. "
            f"Both can be combined, with field options taking precedence."
        )

        if complex_fields:
            field_lines = [f"  {f['name']}: {f['schema']}" for f in complex_fields]
            usage_prefix += "\nJSON-only fields:\n" + "\n".join(field_lines)

        sdk_help = sdk_method.get_param_description(param.name) or ""
        if sdk_help:
            help_text = transform_query_to_cli(sdk_help, param.name)
            help_text = f"{usage_prefix}\n\n{help_text}"
        else:
            help_text = usage_prefix

        return help_text
