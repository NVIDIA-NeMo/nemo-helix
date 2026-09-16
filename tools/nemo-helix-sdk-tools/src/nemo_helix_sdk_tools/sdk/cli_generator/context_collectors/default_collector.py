# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Context collector for default/unclassified operations."""

from __future__ import annotations

from typing import Any

from caseutil.cases import to_kebab
from nemo_helix_sdk_tools.sdk.cli_generator.context_collectors.base import BaseContextCollector, build_path_params
from nemo_helix_sdk_tools.sdk.cli_generator.models import sanitize_help_text
from nemo_helix_sdk_tools.sdk.cli_generator.sdk_introspector import SDKMethod


class DefaultContextCollector(BaseContextCollector):
    """Collect context for default/unclassified operations (stub implementation).

    Default collector handles any operation that doesn't fit other categories.
    It provides a basic command structure that:
    - Accepts path and optional parameters
    - Calls SDK method
    - Returns formatted output
    """

    def collect(
        self,
        resource_path: list[str],
        sdk_method: SDKMethod,
        method_name: str,
    ) -> dict[str, Any]:
        """Collect template context for default command.

        Args:
            resource_path: Resource path (e.g., ["customization", "jobs"])
            sdk_method: SDK method metadata
            method_name: Method name

        Returns:
            Dictionary with template variables for default_command.py.j2
        """
        resource_name = resource_path[-1]

        # Extract parameters
        path_params = build_path_params(sdk_method)
        parameters, kwargs_entries, has_exploded_params = self._build_optional_params(sdk_method)

        # Build command description
        command_description = f"Execute {method_name} on {resource_name}."
        if sdk_method and sdk_method.description:
            command_description = sdk_method.description

        # Build CLI names
        cli_command_name = method_name
        if cli_command_name == "retrieve":
            # TODO: Better handling for this. Maybe add "get" as a operation type.
            cli_command_name = "get"
        cli_command_name = to_kebab(cli_command_name)
        function_name = f"{method_name}_{resource_name}"

        return {
            "operation_type": "default",
            "resource_name": resource_name,
            "resource_path": resource_path,
            "resource_path_quoted": ", ".join(f'"{p}"' for p in resource_path),
            "sdk_accessor": ".".join(resource_path),
            "method_name": method_name,
            "cli_command_name": cli_command_name,
            "function_name": function_name,
            "path_params": path_params,
            "parameters": parameters,
            "kwargs_entries": kwargs_entries,
            "help_text": sanitize_help_text(command_description),
        }
