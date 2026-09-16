# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Quickstart cluster management for NeMo Helix.

This module provides a high-level API for managing NeMo Helix quickstart clusters
using the Python Docker SDK.

Examples:
    Quick Start::

        from nemo_helix_ext.quickstart import QuickstartCluster

        cluster = QuickstartCluster()
        cluster.start()
        print(cluster.status())
        cluster.stop()

    Configuration::

        from nemo_helix_ext.quickstart import QuickstartConfig

        config = QuickstartConfig.load()
        config.host_port = 9090
        config.save()

    Pre-flight Checks::

        from nemo_helix_ext.quickstart import QuickstartCluster, CheckStatus

        cluster = QuickstartCluster()
        results = cluster.preflight()
        for result in results:
            if result.status == CheckStatus.FAIL:
                print(f"FAIL: {result.name}: {result.message}")
"""

from nemo_helix_ext.ui.prompts import is_interactive, prompt_confirm, prompt_select

from .cluster import PreflightError, QuickstartCluster
from .config import QuickstartConfig
from .container import ContainerManager, PullProgress
from .platform_config import PlatformConfig
from .preflight import CheckStatus, PreflightChecker, PreflightResult
from .prompts import (
    RegistryCredentials,
    detect_registry_auth_type,
    prompt_for_configuration,
    prompt_for_optional_registry_credentials,
    prompt_for_registry_credentials,
)
from .storage import StorageManager
from .validators import (
    ValidationResult,
    validate_config,
    validate_docker_available,
    validate_docker_socket,
    validate_image_registry_access,
    validate_ngc_credentials,
    validate_port_available,
    validate_registry_credentials,
    validate_storage_path,
)

__all__ = [
    # High-level API
    "QuickstartCluster",
    "PreflightError",
    # Configuration
    "QuickstartConfig",
    "PlatformConfig",
    # Pre-flight checks
    "PreflightChecker",
    "PreflightResult",
    "CheckStatus",
    # Lower-level managers
    "ContainerManager",
    "PullProgress",
    "StorageManager",
    # Interactive prompts
    "RegistryCredentials",
    "prompt_for_configuration",
    "prompt_for_optional_registry_credentials",
    "prompt_for_registry_credentials",
    "prompt_confirm",
    "prompt_select",
    "detect_registry_auth_type",
    "is_interactive",
    # Validators
    "ValidationResult",
    "validate_config",
    "validate_docker_available",
    "validate_image_registry_access",
    "validate_ngc_credentials",
    "validate_registry_credentials",
    "validate_docker_socket",
    "validate_storage_path",
    "validate_port_available",
]
