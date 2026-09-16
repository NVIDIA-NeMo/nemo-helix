# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Context collector for create operations."""

from __future__ import annotations

from typing import Any

from caseutil import to_kebab
from nemo_platform_sdk_tools.sdk.cli_generator.context_collectors.base import (
    BaseContextCollector,
    build_required_fields_example,
    promote_name_to_positional,
)
from nemo_platform_sdk_tools.sdk.cli_generator.models import sanitize_help_text
from nemo_platform_sdk_tools.sdk.cli_generator.sdk_introspector import SDKMethod


class CreateContextCollector(BaseContextCollector):
    """Collect context for create operations (stub implementation).

    Create operations typically:
    - Accept parameters to configure the new resource
    - Return a single model instance
    - May accept JSON from stdin for complex payloads
    """

    def collect(
        self,
        resource_path: list[str],
        sdk_method: SDKMethod,
        method_name: str,
    ) -> dict[str, Any]:
        """Collect template context for create command.

        Args:
            resource_path: Resource path (e.g., ["customization", "jobs"])
            sdk_method: SDK method metadata
            method_name: Method name (e.g., "create")

        Returns:
            Dictionary with template variables for create_command.py.j2
        """
        resource_name = resource_path[-1]

        # Extract path parameters (usually none for create, but handle if present)
        from nemo_platform_sdk_tools.sdk.cli_generator.context_collectors.base import (
            build_body_params,
            build_path_params,
        )

        path_params = build_path_params(sdk_method)

        # Generate CLI options for simple body parameters
        body_params, required_fields = build_body_params(sdk_method)

        # `name` stays optional ([NAME] in help) so callers can still use --input-file / --input-data.
        promote_name_to_positional(
            body_params,
            resource_path,
            method_name,
            self._cli_config,
            type_str="str | None",
            default="None",
        )

        parameters = body_params

        # Build command description
        command_description = f"Create {resource_name}."
        if sdk_method and sdk_method.description:
            command_description = sdk_method.description

        # Build CLI names
        cli_command_name = to_kebab(method_name)
        function_name = f"{method_name}_{resource_name}"

        return {
            "operation_type": "create",
            "resource_name": resource_name,
            "resource_path": resource_path,
            "resource_path_quoted": ", ".join(f'"{p}"' for p in resource_path),
            "sdk_accessor": ".".join(resource_path),
            "method_name": method_name,
            "cli_command_name": cli_command_name,
            "function_name": function_name,
            "path_params": path_params,
            "parameters": parameters,
            "kwargs_entries": [],
            "help_text": sanitize_help_text(command_description),
            "required_fields": required_fields,
            "required_fields_example": build_required_fields_example(parameters, required_fields),
            "examples": self._cli_config.get_examples(resource_path, method_name),
            "wait_config": self._cli_config.get_wait_config(resource_path, method_name),
            "watch_config": self._cli_config.get_watch_config(resource_path, method_name),
            "stamp_job_telemetry": resource_path == ["jobs"] and method_name == "create",
        }
