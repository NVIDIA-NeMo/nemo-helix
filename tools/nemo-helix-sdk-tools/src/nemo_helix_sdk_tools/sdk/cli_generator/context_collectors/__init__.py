# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Context collector registry and factory."""

from __future__ import annotations

from typing import Type

from nemo_helix_sdk_tools.sdk.cli_generator.config import CLIConfig
from nemo_helix_sdk_tools.sdk.cli_generator.context_collectors.base import BaseContextCollector
from nemo_helix_sdk_tools.sdk.cli_generator.context_collectors.create_collector import (
    CreateContextCollector,
)
from nemo_helix_sdk_tools.sdk.cli_generator.context_collectors.default_collector import (
    DefaultContextCollector,
)
from nemo_helix_sdk_tools.sdk.cli_generator.context_collectors.delete_collector import (
    DeleteContextCollector,
)
from nemo_helix_sdk_tools.sdk.cli_generator.context_collectors.download_collector import (
    DownloadContextCollector,
)
from nemo_helix_sdk_tools.sdk.cli_generator.context_collectors.get_collector import (
    GetContextCollector,
)
from nemo_helix_sdk_tools.sdk.cli_generator.context_collectors.list_collector import (
    ListContextCollector,
)
from nemo_helix_sdk_tools.sdk.cli_generator.context_collectors.update_collector import (
    UpdateContextCollector,
)
from nemo_helix_sdk_tools.sdk.cli_generator.operation_classifier import OperationType


class ContextCollectorRegistry:
    """Registry mapping operation types to collector classes."""

    _collectors: dict[OperationType, Type[BaseContextCollector]] = {}

    @classmethod
    def register(cls, operation_type: OperationType, collector_class: Type[BaseContextCollector]):
        """Register a collector for an operation type.

        Args:
            operation_type: Operation type string (e.g., "list", "create")
            collector_class: Collector class to register
        """
        cls._collectors[operation_type] = collector_class

    @classmethod
    def get_collector(cls, operation_type: OperationType, cli_config: CLIConfig) -> BaseContextCollector:
        """Get a collector instance for an operation type.

        Args:
            operation_type: Operation type string
            cli_config: CLI configuration

        Returns:
            Collector instance

        Raises:
            ValueError: If no collector registered for operation type
        """
        collector_class = cls._collectors.get(operation_type)
        if not collector_class:
            raise ValueError(f"No collector registered for operation type: {operation_type}")
        return collector_class(cli_config)

    @classmethod
    def list_registered(cls) -> list[OperationType]:
        """List all registered operation types.

        Returns:
            List of operation type strings
        """
        return list(cls._collectors.keys())


# Register all collectors
ContextCollectorRegistry.register("list", ListContextCollector)
ContextCollectorRegistry.register("create", CreateContextCollector)
ContextCollectorRegistry.register("update", UpdateContextCollector)
ContextCollectorRegistry.register("delete", DeleteContextCollector)
ContextCollectorRegistry.register("download", DownloadContextCollector)
ContextCollectorRegistry.register("get", GetContextCollector)
ContextCollectorRegistry.register("default", DefaultContextCollector)
