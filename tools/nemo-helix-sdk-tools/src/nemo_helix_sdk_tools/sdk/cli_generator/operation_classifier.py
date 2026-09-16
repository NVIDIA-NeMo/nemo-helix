# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Classification of SDK methods into operation types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from nemo_helix._base_client import BasePage
from nemo_helix._response import BinaryAPIResponse
from nemo_helix_sdk_tools.sdk.cli_generator.config import CLIConfig
from nemo_helix_sdk_tools.sdk.cli_generator.sdk_introspector import SDKMethod

OperationType = Literal["list", "create", "update", "delete", "download", "get", "default"]


@dataclass
class OperationInfo:
    """Information about a classified SDK method operation."""

    operation_type: OperationType
    sdk_method: SDKMethod
    method_name: str
    resource_path: list[str]


class OperationClassifier:
    """Classify SDK methods by operation type.

    Classification rules (priority order):
    1. Config override - Explicit operation_type in cli_config.yaml
    2. Return type - Most reliable signal
    3. Method name patterns - Heuristic fallback
    4. Default - Catch-all for unclassified
    """

    def __init__(self, cli_config: CLIConfig):
        self._cli_config = cli_config

    def classify(self, sdk_method: SDKMethod) -> OperationType:
        """Classify SDK method into operation type.

        Args:
            sdk_method: SDK method metadata

        Returns:
            Operation type
        """
        # Rule 1: Config override (not implemented yet)
        override = self._get_config_override(sdk_method)
        if override:
            return override

        # Rule 2: Return type inspection (most reliable)
        op_type = self._classify_by_return_type(sdk_method)
        if op_type:
            return op_type

        # Rule 3: Method name patterns (heuristic fallback)
        op_type = self._classify_by_method_name(sdk_method.name)
        if op_type:
            return op_type

        # Rule 4: Default fallback
        return "default"

    def _get_config_override(self, sdk_method: SDKMethod) -> str | None:
        """Get explicit operation type from config (future feature).

        Args:
            method_name: Name of the SDK method

        Returns:
            Operation type from config, or None if not specified
        """
        config = self._cli_config.get_method_config(sdk_method.resource_path, sdk_method.name)
        if config and "operation_type" in config:
            return config["operation_type"]
        return None

    def _classify_by_return_type(self, sdk_method: SDKMethod) -> OperationType | None:
        """Classify by return type inspection.

        Args:
            sdk_method: SDK method metadata

        Returns:
            Operation type based on return type, or None if can't determine
        """
        return_type = sdk_method.return_type

        if return_type is None:
            return None

        try:
            # List operations return BasePage subclasses
            if isinstance(return_type, type) and issubclass(return_type, BasePage):
                return "list"

            if isinstance(return_type, type) and issubclass(return_type, BinaryAPIResponse):
                return "download"

        except (TypeError, AttributeError):
            # Return type might not be a class (Union, etc)
            pass

        return None

    def _classify_by_method_name(self, method_name: str) -> OperationType | None:
        """Classify by method name pattern matching.

        Args:
            method_name: Name of the SDK method

        Returns:
            Operation type based on name pattern, or None if can't determine
        """
        method_name_mappings = {
            "list": ["list"],
            "create": ["create", "post", "put"],  # HTTP methods with body use create pattern
            "update": ["update", "patch"],
            "delete": ["delete", "remove"],
            "download": ["download"],
            "get": ["get", "retrieve"],
        }
        for op_type, name_variants in method_name_mappings.items():
            for variant in name_variants:
                if method_name == variant or method_name.startswith(f"{variant}_"):
                    return op_type

        return None
