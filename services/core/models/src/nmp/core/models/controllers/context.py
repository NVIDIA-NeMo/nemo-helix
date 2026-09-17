# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Context dataclass for bundling related entities in the Models Controller."""

from dataclasses import dataclass
from typing import Optional

from nemo_platform_plugin.models.types import (
    ModelDeployment,
    ModelDeploymentConfig,
    ModelEntity,
    ModelProvider,
    ServedModelMapping,
)


@dataclass
class ModelContext:
    """Bundles related model entities together.

    This context object pre-fetches and bundles entities related to model deployments
    and providers to avoid redundant API calls during reconciliation.

    All fields are optional - populate based on what's available and relevant for the
    current reconciliation operation.

    Attributes:
        model_deployment: The ModelDeployment being reconciled, if applicable
        model_deployment_config: Associated deployment configuration, if it exists
        model_provider: Associated model provider, if it exists
        model_entity: Associated model entity, if it exists
        served_models: The served-model mappings computed during the current
            provider reconciliation cycle, when available.
    """

    model_deployment: Optional[ModelDeployment] = None
    model_deployment_config: Optional[ModelDeploymentConfig] = None
    model_provider: Optional[ModelProvider] = None
    model_entity: Optional[ModelEntity] = None
    served_models: Optional[list[ServedModelMapping]] = None
