# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import math

from fastapi import APIRouter, Depends, HTTPException, Query, status
from nhx.common.api.common import Page, PaginationData
from nhx.common.auth import AuthClient, get_auth_client
from nhx.common.config import HelixConfig
from nhx.common.entities.client import EntityClient, EntityConflictError, EntityNotFoundError
from nhx.common.observability import scoped_app_ctx
from nhx.common.secrets.encryption import envelope_decrypt, envelope_encrypt
from nhx.common.secrets.exceptions import EncryptionError
from nhx.common.service.dependencies import get_entity_client, get_platform_config
from nhx.core.secrets.api.v2.secrets import schemas
from nhx.core.secrets.api.v2.secrets.ngc_api_key import get_default_ngc_api_key, is_default_ngc_api_key
from nhx.core.secrets.app.ctx import SecretsContext
from nhx.core.secrets.app.encryptor import get_current_encryptor, get_encryptor_by_name
from nhx.core.secrets.entities import HelixSecret

router = APIRouter()

logger = logging.getLogger(__name__)

## CRUD Endpoints for Platform Secrets ##


@router.post("/v2/workspaces/{workspace}/secrets", status_code=status.HTTP_201_CREATED)
async def create_secret(
    workspace: str,
    create_request: schemas.HelixSecretCreateRequest,
    entity_store: EntityClient = Depends(get_entity_client),
    platform_config: HelixConfig = Depends(get_platform_config),
) -> schemas.HelixSecretResponse:
    """Create a new secret."""
    with scoped_app_ctx(SecretsContext(name=create_request.name, namespace=workspace)):
        # Disallow creation of the default NGC API key secret
        if is_default_ngc_api_key(platform_config, workspace, create_request.name):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cannot overwrite the default NGC API key secret",
            )

        # Check for existing secret
        try:
            await entity_store.get(HelixSecret, workspace=workspace, name=create_request.name)
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Secret '{workspace}/{create_request.name}' already exists",
            )
        except EntityNotFoundError:
            pass
        secret = HelixSecret(
            workspace=workspace,
            name=create_request.name,
            description=create_request.description,
        )
        # Data is attached to stored entity as private attribute, not returned in API responses
        try:
            # Get the current secret provider.
            encryptor = get_current_encryptor()
            secret._data, secret._encrypted_dek, secret._secret_provider = envelope_encrypt(
                encryptor, create_request.value.get_secret_value()
            )
        except EncryptionError:
            logger.exception("Failed to encrypt secret value")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unexpected error creating secret"
            )
        try:
            created_secret = await entity_store.create(secret)
            return schemas.HelixSecretResponse.from_entity(created_secret)
        except EntityConflictError:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Secret '{workspace}/{create_request.name}' already exists",
            )


@router.get(
    "/v2/workspaces/{workspace}/secrets",
    response_model=Page[schemas.HelixSecretResponse],
    response_model_exclude_none=True,
)
async def list_secrets(
    workspace: str,
    page: int = Query(default=1, description="Page number.", gt=0),
    page_size: int = Query(default=10, description="Page size.", gt=0),
    entity_store: EntityClient = Depends(get_entity_client),
    platform_config: HelixConfig = Depends(get_platform_config),
) -> Page[schemas.HelixSecretResponse]:
    """List available secrets"""
    with scoped_app_ctx(SecretsContext(namespace=workspace)):
        result = await entity_store.list(
            entity_type=HelixSecret,
            workspace=workspace,
            page=page,
            page_size=page_size,
        )
        total_results = result.pagination.total_results
        total_pages = result.pagination.total_pages
        ngc_api_key_secret = platform_config.ngc_api_key_secret.split("/")
        if is_default_ngc_api_key(platform_config, workspace, ngc_api_key_secret[1]):
            total_results = result.pagination.total_results + 1
            total_pages = math.ceil(total_results / page_size) if page_size else 1
            if page == total_pages:
                result.data.append(get_default_ngc_api_key(platform_config))

        return Page(
            data=[schemas.HelixSecretResponse.from_entity(secret) for secret in result.data],
            pagination=PaginationData(
                page=result.pagination.page,
                page_size=result.pagination.page_size,
                current_page_size=len(result.data),
                total_pages=total_pages,
                total_results=total_results,
            ),
        )


@router.get("/v2/workspaces/{workspace}/secrets/{name}", status_code=status.HTTP_200_OK)
async def get_secret(
    name: str,
    workspace: str,
    entity_store: EntityClient = Depends(get_entity_client),
    platform_config: HelixConfig = Depends(get_platform_config),
) -> schemas.HelixSecretResponse:
    """Retrieve a secret by its name."""
    with scoped_app_ctx(SecretsContext(name=name, namespace=workspace)):
        # If the secret is the default NGC API key secret, return the default NGC API key secret metadata
        if is_default_ngc_api_key(platform_config, workspace, name):
            return schemas.HelixSecretResponse.from_entity(get_default_ngc_api_key(platform_config))

        try:
            secret = await entity_store.get(HelixSecret, workspace=workspace, name=name)
        except EntityNotFoundError:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Secret {workspace}/{name} not found")
        return schemas.HelixSecretResponse.from_entity(secret)


@router.patch("/v2/workspaces/{workspace}/secrets/{name}", status_code=status.HTTP_200_OK)
async def update_secret(
    name: str,
    workspace: str,
    patch_request: schemas.HelixSecretUpdateRequest,
    entity_store: EntityClient = Depends(get_entity_client),
    platform_config: HelixConfig = Depends(get_platform_config),
) -> schemas.HelixSecretResponse:
    """Update a secret's metadata."""
    with scoped_app_ctx(SecretsContext(name=name, namespace=workspace)):
        # Disallow updating the default NGC API key secret
        if is_default_ngc_api_key(platform_config, workspace, name):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cannot update the default NGC API key secret",
            )

        try:
            secret = await entity_store.get(HelixSecret, workspace=workspace, name=name)
        except EntityNotFoundError:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Secret {workspace}/{name} not found")

        if patch_request.description is not None:
            secret.description = patch_request.description
        if patch_request.value is not None:
            encryptor = get_encryptor_by_name(secret._secret_provider)
            secret._data, secret._encrypted_dek, secret._secret_provider = envelope_encrypt(
                encryptor, patch_request.value.get_secret_value()
            )

        updated_secret = await entity_store.update(secret)
        return schemas.HelixSecretResponse.from_entity(updated_secret)


@router.delete("/v2/workspaces/{workspace}/secrets/{name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_secret(
    name: str,
    workspace: str,
    entity_store: EntityClient = Depends(get_entity_client),
    platform_config: HelixConfig = Depends(get_platform_config),
) -> None:
    """Delete a secret."""
    with scoped_app_ctx(SecretsContext(name=name, namespace=workspace)):
        # Disallow deletion of the default NGC API key secret
        if is_default_ngc_api_key(platform_config, workspace, name):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cannot delete the default NGC API key secret",
            )
        try:
            await entity_store.get(HelixSecret, workspace=workspace, name=name)
        except EntityNotFoundError:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Secret {workspace}/{name} not found")
        await entity_store.delete(HelixSecret, name, workspace=workspace)


@router.get("/v2/workspaces/{workspace}/secrets/{name}/access", status_code=status.HTTP_200_OK)
async def access_secret(
    name: str,
    workspace: str,
    entity_store: EntityClient = Depends(get_entity_client),
    auth_client: AuthClient = Depends(get_auth_client),
    platform_config: HelixConfig = Depends(get_platform_config),
) -> schemas.HelixSecretAccessResponse:
    """Access the value of a secret."""
    with scoped_app_ctx(SecretsContext(name=name, namespace=workspace)):
        # Additional authorization check.
        # we need to check if we are validating a delegated user here, or the caller is the actual accessor.
        # Create auth client with the on-behalf-of principal for authorization check
        if auth_client.principal.on_behalf_of is not None and not await auth_client.on_behalf_of_has_permissions(
            workspace, ["secrets.read"]
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden",
            )

        # If the secret is the default NGC API key secret, return its value directly
        if is_default_ngc_api_key(platform_config, workspace, name):
            return schemas.HelixSecretAccessResponse(
                name=name,
                workspace=workspace,
                value=get_default_ngc_api_key(platform_config)._data,
            )

        try:
            secret = await entity_store.get(HelixSecret, workspace=workspace, name=name)
        except EntityNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Secret {workspace}/{name} not found",
            )

        try:
            encryptor = get_encryptor_by_name(secret._secret_provider)
            secret_data = envelope_decrypt(encryptor, secret._data, secret._encrypted_dek, secret._secret_provider)
        except EncryptionError:
            logger.exception("Failed to decrypt secret value")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unexpected error retrieving secret"
            )
        return schemas.HelixSecretAccessResponse(
            name=secret.name,
            workspace=secret.workspace,
            value=secret_data,
        )
