# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastAPI dependencies for the Models API."""

from fastapi import Depends
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.secrets.client import AsyncSecretsClient
from nhx.common.entities.client import EntityClient
from nhx.common.service.dependencies import get_entity_client, get_nemo_client
from nhx.core.models.api.service.adapter_entity_service import AdapterEntityService
from nhx.core.models.api.service.model_deployment_config_service import ModelDeploymentConfigService
from nhx.core.models.api.service.model_deployment_service import ModelDeploymentService
from nhx.core.models.api.service.model_entity_service import ModelEntityService
from nhx.core.models.api.service.model_provider_service import ModelProviderService
from nhx.core.models.api.service.prompt_service import PromptService


def get_files_client(client: AsyncNemoClient = Depends(get_nemo_client)) -> AsyncFilesClient:
    """Dependency to get a request-scoped Files service client."""
    return AsyncFilesClient.from_client(client)


def get_secrets_client(client: AsyncNemoClient = Depends(get_nemo_client)) -> AsyncSecretsClient:
    """Dependency to get a request-scoped Secrets service client."""
    return AsyncSecretsClient.from_client(client)


def get_model_entity_service(
    entity_client: EntityClient = Depends(get_entity_client),
    files_client: AsyncFilesClient = Depends(get_files_client),
) -> ModelEntityService:
    """Dependency to get ModelEntityService instance."""
    return ModelEntityService(entity_client, files=files_client)


def get_adapter_entity_service(
    entity_client: EntityClient = Depends(get_entity_client),
    files_client: AsyncFilesClient = Depends(get_files_client),
) -> AdapterEntityService:
    """Dependency to get AdapterEntityService instance."""
    return AdapterEntityService(entity_client, files=files_client)


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
) -> ModelDeploymentService:
    """Dependency to get ModelDeploymentService instance."""
    return ModelDeploymentService(entity_client)
