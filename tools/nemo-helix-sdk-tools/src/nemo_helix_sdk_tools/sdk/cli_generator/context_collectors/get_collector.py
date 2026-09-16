# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from caseutil.cases import to_kebab
from nemo_helix_sdk_tools.sdk.cli_generator.context_collectors.base import (
    BaseContextCollector,
    build_path_params,
    promote_name_to_positional,
)
from nemo_helix_sdk_tools.sdk.cli_generator.models import sanitize_help_text
from nemo_helix_sdk_tools.sdk.cli_generator.sdk_introspector import SDKMethod


class GetContextCollector(BaseContextCollector):
    """Collect context for get/retrieve operations.

    Get operations retrieve a single resource by ID.
    They typically:
    - Accept an ID parameter (required)
    - May accept additional query parameters
    - Return a single model object
    - Show detailed information about one item
    """

    def collect(
        self,
        resource_path: list[str],
        sdk_method: SDKMethod,
        method_name: str,
    ) -> dict[str, Any]:
        """Collect template context for get command.

        Args:
            resource_path: Resource path (e.g., ["customization", "jobs"])
            sdk_method: SDK method metadata
            method_name: Method name

        Returns:
            Dictionary with template variables for get_command.py.j2
        """
        resource_name = resource_path[-1]

        # Extract parameters
        path_params = build_path_params(sdk_method)
        parameters, kwargs_entries, has_exploded_params = self._build_optional_params(sdk_method)

        promote_name_to_positional(parameters, resource_path, method_name, self._cli_config)

        # Build command description
        command_description = f"Get details of a specific {resource_name}."
        if sdk_method and sdk_method.description:
            command_description = sdk_method.description

        # Build CLI names
        # Always use "get" as the CLI command name for consistency
        cli_command_name = to_kebab(method_name.replace("retrieve", "get"))
        function_name = method_name + "_" + resource_name

        return {
            "operation_type": "get",
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
