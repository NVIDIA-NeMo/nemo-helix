# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Context collector for download operations."""

from __future__ import annotations

from typing import Any

from nemo_helix_sdk_tools.sdk.cli_generator.context_collectors.base import BaseContextCollector, build_path_params
from nemo_helix_sdk_tools.sdk.cli_generator.models import sanitize_help_text
from nemo_helix_sdk_tools.sdk.cli_generator.sdk_introspector import SDKMethod


class DownloadContextCollector(BaseContextCollector):
    """Collect context for download operations (stub implementation).

    Download operations typically:
    - Return binary content (HttpxBinaryResponseContent)
    - Accept output path parameter
    - Stream large files
    - Show progress bar
    """

    def collect(
        self,
        resource_path: list[str],
        sdk_method: SDKMethod,
        method_name: str,
    ) -> dict[str, Any]:
        """Collect template context for download command.

        Args:
            resource_path: Resource path (e.g., ["models"])
            sdk_method: SDK method metadata
            method_name: Method name (e.g., "download")

        Returns:
            Dictionary with template variables for download_command.py.j2
        """
        resource_name = resource_path[-1]

        # Extract parameters
        path_params = build_path_params(sdk_method)
        parameters, kwargs_entries, has_exploded_params = self._build_optional_params(sdk_method)

        # Build command description
        command_description = f"Download {resource_name}."
        if sdk_method and sdk_method.description:
            command_description = sdk_method.description

        # Build CLI names
        cli_command_name = method_name.replace("_", "-")
        function_name = f"{method_name}_{resource_name}"

        return {
            "operation_type": "download",
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
