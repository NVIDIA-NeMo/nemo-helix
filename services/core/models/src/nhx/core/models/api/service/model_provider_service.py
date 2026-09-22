# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Service layer for ModelProvider operations using EntityClient."""

import logging

from nhx.common.api.common import Page, PaginationData
from nhx.common.api.filter import FilterOperation
from nhx.common.auth import AuthContext
from nhx.common.entities.client import EntityClient, EntityConflictError, EntityNotFoundError
from nhx.common.entities.utils import parse_entity_ref
from nhx.core.models.entities import Model
from nhx.core.models.entities import ModelProvider as ModelProviderEntity
from nhx.core.models.schemas import (
    CreateModelProviderRequest,
    DeleteModelProviderRequest,
    GetModelProviderRequest,
    ModelProvider,
    ModelProviderStatus,
    UpdateModelProviderStatusRequest,
    UpsertModelProviderRequest,
)

logger = logging.getLogger(__name__)


class ModelProviderValidationError(Exception):
    """Raised when model provider validation fails."""

    pass


# Mapping of rejected header names (lowercase) to their error messages.
# The {field_name} placeholder will be replaced with the actual field name.
REJECTED_HEADERS: dict[str, str] = {
    "authorization": "Authorization header is not allowed in {field_name}. Use api_key_secret_name for authentication.",
    "cookie": "Cookie header is not allowed in {field_name}.",
}


def _validate_extra_headers(headers: dict[str, str] | None, field_name: str) -> None:
    """Validate extra_headers doesn't contain reserved headers.

    Args:
        headers: The headers dict to validate
        field_name: Name of the field being validated (for error message)

    Raises:
        ModelProviderValidationError: If reserved headers are found
    """
    if headers is None:
        return

    for header_key in headers.keys():
        lowercase_key = header_key.lower()
        if lowercase_key in REJECTED_HEADERS:
            error_message = REJECTED_HEADERS[lowercase_key].format(field_name=field_name)
            raise ModelProviderValidationError(error_message)


def _entity_to_schema(entity: ModelProviderEntity) -> ModelProvider:
    """Convert an EntityBase ModelProvider to the API schema."""
    return ModelProvider(
        id=entity.id,
        name=entity.name,
        workspace=entity.workspace,
        project=entity.project,
        description=entity.description,
        host_url=entity.host_url,
        api_key_secret_name=entity.api_key_secret_name,
        served_models=entity.served_models or [],
        enabled_models=entity.enabled_models,
        status=entity.status,
        status_message=entity.status_message or "",
        default_extra_body=entity.default_extra_body,
        default_extra_headers=entity.default_extra_headers,
        required_extra_body=entity.required_extra_body,
        required_extra_headers=entity.required_extra_headers,
        model_deployment_id=entity.model_deployment_id,
        auth_context=entity.auth_context,
        auth_header_format=entity.auth_header_format,
        created_at=entity.created_at,
        updated_at=entity.updated_at,
    )


class ModelProviderService:
    """Service layer for ModelProvider operations."""

    def __init__(self, entity_client: EntityClient):
        self.entity_client = entity_client

    async def create_model_provider(
        self,
        request: CreateModelProviderRequest,
        workspace: str,
        auth_context: AuthContext | None = None,
    ) -> ModelProvider:
        """Create a new model provider."""
        logger.debug("Creating model provider", extra={"workspace": workspace, "provider_name": request.name})

        _validate_extra_headers(request.default_extra_headers, "default_extra_headers")
        _validate_extra_headers(request.required_extra_headers, "required_extra_headers")

        try:
            existing = await self.entity_client.get(ModelProviderEntity, name=request.name, workspace=workspace)
            if existing:
                logger.warning(
                    "Model provider already exists", extra={"workspace": workspace, "provider_name": request.name}
                )
                raise ValueError(f"Model provider with name '{request.name}' already exists in workspace '{workspace}'")
        except EntityNotFoundError:
            pass  # Expected - provider doesn't exist, proceed with creation

        # api_key_secret_name is a reference to a secret created by the user via the Secrets API
        entity = ModelProviderEntity(
            name=request.name,
            workspace=workspace,
            project=request.project,
            description=request.description,
            host_url=request.host_url,
            api_key_secret_name=request.api_key_secret_name,
            enabled_models=request.enabled_models,
            default_extra_body=request.default_extra_body,
            default_extra_headers=request.default_extra_headers,
            required_extra_body=request.required_extra_body,
            required_extra_headers=request.required_extra_headers,
            model_deployment_id=request.model_deployment_id,
            auth_header_format=request.auth_header_format,
            status=request.status if request.status is not None else ModelProviderStatus.CREATED,
            status_message=request.status_message if request.status_message is not None else "Model provider created",
        ).with_auth_context(auth_context)

        try:
            created = await self.entity_client.create(entity)
            logger.info("Model provider created", extra={"workspace": created.workspace, "provider_name": created.name})
            return _entity_to_schema(created)
        except EntityConflictError as e:
            logger.warning(
                "Model provider already exists (conflict)",
                extra={"workspace": workspace, "provider_name": request.name},
            )
            raise ValueError(
                f"Model provider with name '{request.name}' already exists in workspace '{workspace}'"
            ) from e

    async def get_model_provider(self, request: GetModelProviderRequest) -> ModelProvider | None:
        """Get a model provider by workspace and name."""
        logger.debug("Getting model provider", extra={"workspace": request.workspace, "provider_name": request.name})

        try:
            entity = await self.entity_client.get(
                ModelProviderEntity,
                workspace=request.workspace,
                name=request.name,
            )
            logger.debug("Found model provider", extra={"workspace": entity.workspace, "provider_name": entity.name})
            return _entity_to_schema(entity)
        except EntityNotFoundError:
            logger.debug(
                "Model provider not found", extra={"workspace": request.workspace, "provider_name": request.name}
            )
            return None

    async def list_model_providers(
        self,
        workspace: str,
        page: int = 1,
        page_size: int = 100,
        sort: str | None = None,
        filter_operation: FilterOperation | None = None,
    ) -> Page[ModelProvider]:
        """List model providers with filtering and pagination."""
        logger.debug("Listing model providers", extra={"page": page, "page_size": page_size, "sort": sort})

        result = await self.entity_client.list(
            ModelProviderEntity,
            workspace=workspace,
            filter_operation=filter_operation,
            sort=sort,
            page=page,
            page_size=page_size,
        )

        logger.debug("Listed model providers", extra={"count": len(result.data)})

        providers = [_entity_to_schema(entity) for entity in result.data]

        return Page(
            data=providers,
            pagination=PaginationData(
                page=result.pagination.page,
                page_size=result.pagination.page_size,
                current_page_size=len(providers),
                total_pages=result.pagination.total_pages,
                total_results=result.pagination.total_results,
            ),
            sort=sort,
            filter=None,
        )

    async def upsert_model_provider(
        self,
        workspace: str,
        name: str,
        request: UpsertModelProviderRequest,
        auth_context: AuthContext | None = None,
    ) -> ModelProvider:
        """Create or update a model provider."""
        logger.debug("Upserting model provider", extra={"workspace": workspace, "provider_name": name})

        _validate_extra_headers(request.default_extra_headers, "default_extra_headers")
        _validate_extra_headers(request.required_extra_headers, "required_extra_headers")

        existing_entity: ModelProviderEntity | None = None
        try:
            existing_entity = await self.entity_client.get(
                ModelProviderEntity,
                workspace=workspace,
                name=name,
            )
        except EntityNotFoundError:
            pass

        if request.status is not None:
            status = request.status
            status_message = request.status_message if request.status_message is not None else ""
        elif existing_entity:
            status = existing_entity.status
            status_message = existing_entity.status_message
        else:
            status = ModelProviderStatus.CREATED
            status_message = "Model provider created"

        if existing_entity:
            existing_entity.project = request.project
            existing_entity.description = request.description
            existing_entity.host_url = request.host_url
            existing_entity.api_key_secret_name = request.api_key_secret_name
            existing_entity.enabled_models = request.enabled_models
            existing_entity.default_extra_body = request.default_extra_body
            existing_entity.default_extra_headers = request.default_extra_headers
            existing_entity.required_extra_body = request.required_extra_body
            existing_entity.required_extra_headers = request.required_extra_headers
            existing_entity.model_deployment_id = request.model_deployment_id
            existing_entity.auth_header_format = request.auth_header_format
            existing_entity.status = status
            existing_entity.status_message = status_message
            if auth_context:
                existing_entity.with_auth_context(auth_context)

            updated = await self.entity_client.update(existing_entity)
            logger.info("Model provider updated", extra={"workspace": updated.workspace, "provider_name": updated.name})
            return _entity_to_schema(updated)
        else:
            entity = ModelProviderEntity(
                name=name,
                workspace=workspace,
                project=request.project,
                description=request.description,
                host_url=request.host_url,
                api_key_secret_name=request.api_key_secret_name,
                enabled_models=request.enabled_models,
                default_extra_body=request.default_extra_body,
                default_extra_headers=request.default_extra_headers,
                required_extra_body=request.required_extra_body,
                required_extra_headers=request.required_extra_headers,
                model_deployment_id=request.model_deployment_id,
                auth_header_format=request.auth_header_format,
                status=status,
                status_message=status_message,
            ).with_auth_context(auth_context)

            created = await self.entity_client.create(entity)
            logger.info("Model provider created", extra={"workspace": created.workspace, "provider_name": created.name})
            return _entity_to_schema(created)

    async def delete_model_provider(self, request: DeleteModelProviderRequest) -> bool:
        """Delete a model provider and clean up references from linked model entities."""
        logger.debug("Deleting model provider", extra={"workspace": request.workspace, "provider_name": request.name})

        try:
            provider = await self.entity_client.get(
                ModelProviderEntity,
                workspace=request.workspace,
                name=request.name,
            )
        except EntityNotFoundError:
            logger.warning(
                "Model provider not found for deletion",
                extra={"workspace": request.workspace, "provider_name": request.name},
            )
            return False

        provider_id = f"{request.workspace}/{request.name}"
        await self.entity_client.delete(
            ModelProviderEntity,
            provider.name,
            workspace=request.workspace,
            expected_db_version=provider.db_version,
        )

        await self._cleanup_model_entity_references(provider, provider_id)
        logger.info("Model provider deleted", extra={"workspace": request.workspace, "provider_name": request.name})
        return True

    async def _cleanup_model_entity_references(self, provider: ModelProviderEntity, provider_id: str) -> None:
        """Remove this provider from the model_providers list of all linked model entities."""
        if not provider.served_models:
            logger.debug("No served_models, skipping model entity cleanup", extra={"provider_id": provider_id})
            return

        for served_model in provider.served_models:
            try:
                try:
                    ref = parse_entity_ref(served_model.model_entity_id)
                except ValueError:
                    logger.warning(
                        "Invalid model_entity_id format, skipping cleanup",
                        extra={"model_entity_id": served_model.model_entity_id, "provider_id": provider_id},
                    )
                    continue
                model_workspace, model_name = ref.workspace, ref.name
                model = await self.entity_client.get(Model, workspace=model_workspace, name=model_name)

                if provider_id in model.model_providers:
                    model.model_providers = [p for p in model.model_providers if p != provider_id]
                    await self.entity_client.update(model)
                    logger.info(
                        "Removed provider from model entity",
                        extra={"provider_id": provider_id, "model_entity_id": served_model.model_entity_id},
                    )
                else:
                    logger.debug(
                        "Provider not in model entity model_providers",
                        extra={"provider_id": provider_id, "model_entity_id": served_model.model_entity_id},
                    )
            except EntityNotFoundError:
                logger.debug(
                    "Model entity not found during provider cleanup",
                    extra={"model_entity_id": served_model.model_entity_id},
                )
            except Exception as e:
                logger.warning(
                    "Failed to clean up model entity for provider",
                    extra={
                        "model_entity_id": served_model.model_entity_id,
                        "provider_id": provider_id,
                        "error": str(e),
                    },
                )

    async def update_model_provider_status(
        self, workspace: str, name: str, request: UpdateModelProviderStatusRequest
    ) -> ModelProvider | None:
        """Update status-related fields of a model provider (partial update).

        Used by Models Controller to update served_models and status during autodiscovery.
        """
        logger.debug("Updating model provider status fields", extra={"workspace": workspace, "provider_name": name})

        try:
            entity = await self.entity_client.get(
                ModelProviderEntity,
                workspace=workspace,
                name=name,
            )
        except EntityNotFoundError:
            logger.warning(
                "Model provider not found for status update", extra={"workspace": workspace, "provider_name": name}
            )
            return None

        if request.model_deployment_id is not None:
            entity.model_deployment_id = request.model_deployment_id
        if request.served_models is not None:
            entity.served_models = request.served_models
        if request.status is not None:
            entity.status = request.status
            entity.status_message = request.status_message if request.status_message is not None else ""

        updated = await self.entity_client.update(entity)
        logger.debug(
            "Model provider status fields updated",
            extra={"workspace": updated.workspace, "provider_name": updated.name},
        )
        return _entity_to_schema(updated)
