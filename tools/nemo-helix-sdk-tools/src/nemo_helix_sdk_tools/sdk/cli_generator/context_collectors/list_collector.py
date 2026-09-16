# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Context collector for list operations."""

from __future__ import annotations

from typing import Any

from nemo_helix.pagination import SyncDefaultPagination, SyncLogsPagination
from nemo_helix_sdk_tools.sdk.cli_generator.context_collectors.base import BaseContextCollector, build_path_params
from nemo_helix_sdk_tools.sdk.cli_generator.models import sanitize_help_text
from nemo_helix_sdk_tools.sdk.cli_generator.sdk_introspector import SDKMethod


class ListContextCollector(BaseContextCollector):
    """Collect context for list operations (paginated).

    List operations return BasePage subclasses and support:
    - Pagination (page number or cursor-based)
    - Output formatting and column customization
    - Exploded filter/search parameters
    """

    def collect(
        self,
        resource_path: list[str],
        sdk_method: SDKMethod,
        method_name: str,
    ) -> dict[str, Any]:
        """Collect template context for list command.

        Args:
            resource_path: Resource path (e.g., ["customization", "jobs"])
            sdk_method: SDK method metadata
            method_name: Method name (e.g., "list" or "list_namespace")

        Returns:
            Dictionary with template variables for list_command.py.j2
        """
        resource_name = resource_path[-1]

        # Extract parameters
        param_help = self._cli_config.get_param_help(resource_path, method_name)
        path_params = build_path_params(sdk_method, param_help_overrides=param_help)
        parameters, kwargs_entries, has_exploded_params = self._build_optional_params(sdk_method)

        # Detect pagination type
        pagination_type = self._detect_pagination_type(sdk_method)

        # Get column configuration
        columns = self._cli_config.get_columns(resource_path, method_name)

        # Build command description
        command_description = f"List all {resource_name}."
        if sdk_method and sdk_method.description:
            command_description = sdk_method.description

        # Build CLI names
        cli_command_name = method_name.replace("_", "-")
        function_name = f"{method_name}_{resource_name}"

        return {
            "operation_type": "list",
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
            "pagination_type": pagination_type,
            "columns": columns,
            "help_text": sanitize_help_text(command_description),
        }

    def _detect_pagination_type(self, sdk_method: SDKMethod) -> str:
        """Detect pagination type from return type.

        Args:
            sdk_method: SDK method metadata

        Returns:
            Pagination type string: "PAGE_NUMBER", "CURSOR", or "NOT_PAGINATED"
        """
        if not sdk_method.return_type or not isinstance(sdk_method.return_type, type):
            return "NOT_PAGINATED"

        if issubclass(sdk_method.return_type, SyncLogsPagination):
            return "CURSOR"

        if issubclass(sdk_method.return_type, SyncDefaultPagination):
            return "PAGE_NUMBER"

        # Return type is not a pagination class (e.g., ListFilesetFilesResponse)
        return "NOT_PAGINATED"
