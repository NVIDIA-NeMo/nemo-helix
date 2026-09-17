# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastAPI dependencies for the Models API."""

from fastapi import Depends
from nemo_platform import AsyncNeMoPlatform
from nmp.common.entities.client import EntityClient
from nmp.common.service.dependencies import get_entity_client, get_sdk_client
from nmp.core.models.api.service.adapter_entity_service import AdapterEntityService
from nmp.core.models.api.service.model_deployment_config_service import ModelDeploymentConfigService
from nmp.core.models.api.service.model_deployment_service import ModelDeploymentService
from nmp.core.models.api.service.model_entity_service import ModelEntityService
from nmp.core.models.api.service.model_provider_service import ModelProviderService
from nmp.core.models.api.service.prompt_service import PromptService


def get_model_entity_service(
    entity_client: EntityClient = Depends(get_entity_client),
    nmp_sdk: AsyncNeMoPlatform = Depends(get_sdk_client),
) -> ModelEntityService:
    """Dependency to get ModelEntityService instance."""
    return ModelEntityService(entity_client, sdk=nmp_sdk)


def get_adapter_entity_service(
    entity_client: EntityClient = Depends(get_entity_client),
    nmp_sdk: AsyncNeMoPlatform = Depends(get_sdk_client),
) -> AdapterEntityService:
    """Dependency to get AdapterEntityService instance."""
    return AdapterEntityService(entity_client, sdk=nmp_sdk)


def get_model_provider_service(
    entity_client: EntityClient = Depends(get_entity_client),
) -> ModelProviderService:
    """Dependency to get ModelProviderService instance."""
    return ModelProviderService(entity_client)


def get_prompt_service(
    entity_client: EntityClient = Depends(get_entity_client),
) -> PromptService:
    """Dependency to get PromptService instance."""
    return PromptService(entity_client)


def get_model_deployment_config_service(
    entity_client: EntityClient = Depends(get_entity_client),
) -> ModelDeploymentConfigService:
    """Dependency to get ModelDeploymentConfigService instance."""
    return ModelDeploymentConfigService(entity_client)


def get_model_deployment_service(
    entity_client: EntityClient = Depends(get_entity_client),
    nmp_sdk: AsyncNeMoPlatform = Depends(get_sdk_client),
) -> ModelDeploymentService:
    """Dependency to get ModelDeploymentService instance."""
    return ModelDeploymentService(entity_client, nmp_sdk=nmp_sdk)
