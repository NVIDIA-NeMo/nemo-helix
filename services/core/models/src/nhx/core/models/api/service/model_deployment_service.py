# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Service layer for ModelDeployment operations using EntityClient."""

import json
import logging
from datetime import datetime, timezone
from typing import Any

from nemo_helix import AsyncNeMoHelix
from nhx.common.api.common import Page, PaginationData
from nhx.common.api.filter import FilterOperation
from nhx.common.auth import AuthContext
from nhx.common.entities.client import EntityClient, EntityConflictError, EntityNotFoundError
from nhx.core.models.entities import ModelDeployment as ModelDeploymentEntity
from nhx.core.models.entities import ModelDeploymentConfig as ModelDeploymentConfigEntity
from nhx.core.models.schemas import (
    CreateModelDeploymentRequest,
    ModelDeployment,
    ModelDeploymentStatus,
    ModelDeploymentStatusHistoryItem,
    UpdateModelDeploymentRequest,
    UpdateModelDeploymentStatusRequest,
)

logger = logging.getLogger(__name__)


class DeploymentStatusConflictError(Exception):
    """Raised when a status transition conflicts with deployment lifecycle constraints."""

    pass


def _status_history_entry(
    timestamp: datetime,
    status: ModelDeploymentStatus,
    status_message: str = "",
) -> dict[str, Any]:
    """Build a status history entry dict for entity storage."""
    return {
        "timestamp": timestamp.isoformat(),
        "status": status.value,
        "status_message": status_message,
    }


def _compact_adjacent_status_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge adjacent duplicate (status, status_message) entries, keeping the first timestamp."""
    if not history:
        return history
    result = [history[0]]
    for entry in history[1:]:
        prev = result[-1]
        if prev.get("status") == entry.get("status") and (prev.get("status_message") or "") == (
            entry.get("status_message") or ""
        ):
            continue
        result.append(entry)
    return result


def _entity_to_schema(entity: ModelDeploymentEntity) -> ModelDeployment:
    """Convert an EntityBase ModelDeployment to the API schema."""
    status_history_items: list[ModelDeploymentStatusHistoryItem] = []
    for entry in entity.status_history:
        ts = entry["timestamp"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        status_history_items.append(
            ModelDeploymentStatusHistoryItem(
                timestamp=ts,
                status=ModelDeploymentStatus(entry["status"]),
                status_message=entry.get("status_message", ""),
            )
        )
    return ModelDeployment(
        id=entity.id,
        name=entity.base_name,  # Use base_name as the logical name
        workspace=entity.workspace,
        project=entity.project,
        config=entity.config,
        config_version=entity.config_version,
        status=entity.status,
        status_message=entity.status_message or "",
        status_history=status_history_items,
        model_provider_id=entity.model_provider_id,
        entity_version=entity.entity_version,
        created_at=entity.created_at,
        updated_at=entity.updated_at,
        auth_context=entity.auth_context,
    )


class ModelDeploymentService:
    """Service layer for ModelDeployment operations."""

    def __init__(self, entity_client: EntityClient, nhx_sdk: AsyncNeMoHelix):
        self.entity_client = entity_client
        self.nhx_sdk = nhx_sdk

    async def _get_latest_version(self, workspace: str, base_name: str) -> int | None:
        """Get the highest version number for a deployment."""
        filter_str = json.dumps({"data.base_name": base_name})
        result = await self.entity_client.list(
            ModelDeploymentEntity,
            workspace=workspace,
            filter_str=filter_str,
        )
        if not result.data:
            return None
        return max(dep.entity_version for dep in result.data)

    async def _get_config_latest_version(self, workspace: str, config_name: str) -> int | None:
        """Get the highest version number for a deployment config."""
        filter_str = json.dumps({"data.base_name": config_name})
        result = await self.entity_client.list(
            ModelDeploymentConfigEntity,
            workspace=workspace,
            filter_str=filter_str,
        )
        if not result.data:
            return None
        return max(config.entity_version for config in result.data)

    async def _get_config_by_version(
        self, workspace: str, config_name: str, version: int
    ) -> ModelDeploymentConfigEntity | None:
        """Get a deployment config by name and version."""
        entity_name = f"{config_name}-v{version}"
        try:
            return await self.entity_client.get(
                ModelDeploymentConfigEntity,
                workspace=workspace,
                name=entity_name,
            )
        except EntityNotFoundError:
            return None

    async def _get_by_base_name_and_version(
        self, workspace: str, base_name: str, version: int | None = None
    ) -> ModelDeploymentEntity | None:
        """Get a deployment by base name and optional version."""
        if version is None:
            version = await self._get_latest_version(workspace, base_name)
            if version is None:
                return None

        entity_name = f"{base_name}-v{version}"
        try:
            return await self.entity_client.get(
                ModelDeploymentEntity,
                workspace=workspace,
                name=entity_name,
            )
        except EntityNotFoundError:
            return None

    async def create_deployment(
        self,
        request: CreateModelDeploymentRequest,
        workspace: str,
        auth_context: AuthContext | None = None,
    ) -> ModelDeployment:
        """Create a new deployment (version 1).

        Args:
            request: The deployment creation request
            workspace: The workspace to create the deployment in
            auth_context: Optional auth context from the request creator. Used for delegated access in the controller.
        """
        logger.info(f"Creating deployment: {workspace}/{request.name}")

        existing = await self._get_latest_version(workspace, request.name)
        if existing is not None:
            raise ValueError(f"Deployment with workspace '{workspace}' and name '{request.name}' already exists")

        # If config_version not specified, get the latest version
        config_version = request.config_version
        if not config_version:
            config_version = await self._get_config_latest_version(workspace, request.config)
            if not config_version:
                raise ValueError(f"Deployment config '{workspace}/{request.config}' does not exist")
            logger.debug(f"Using latest config version: {config_version}")

        # Verify the deployment config exists
        config = await self._get_config_by_version(workspace, request.config, config_version)
        if not config:
            raise ValueError(
                f"Deployment config '{workspace}/{request.config}' version '{config_version}' does not exist"
            )

        # Create the entity with versioned name
        entity = (
            ModelDeploymentEntity(
                name=f"{request.name}-v1",
                workspace=workspace,
                base_name=request.name,
                entity_version=1,
                project=request.project,
                config=request.config,
                config_version=config_version,
                status=ModelDeploymentStatus.CREATED,
                status_message="Deployment created",
            )
            # Set auth context for delegated access by the controller
            .with_auth_context(auth_context)
        )

        try:
            created = await self.entity_client.create(entity)
            logger.info(
                f"Successfully created deployment: {workspace}/{created.base_name} version {created.entity_version}"
            )
            return _entity_to_schema(created)
        except EntityConflictError as e:
            logger.warning(f"Deployment already exists: {workspace}/{request.name}")
            raise ValueError(f"Deployment with name '{request.name}' already exists in workspace '{workspace}'") from e

    async def get_deployment(self, workspace: str, name: str, version: int | None = None) -> ModelDeployment | None:
        """Get a deployment by workspace, name, and optionally version."""
        logger.debug(f"Getting deployment: {workspace}/{name}" + (f" version {version}" if version else " (latest)"))

        entity = await self._get_by_base_name_and_version(workspace, name, version)
        if entity:
            logger.debug(f"Found deployment: {entity.workspace}/{entity.base_name} version {entity.entity_version}")
            return _entity_to_schema(entity)

        logger.debug(f"Deployment not found: {workspace}/{name}" + (f" version {version}" if version else ""))
        return None

    async def list_deployments(
        self,
        workspace: str,
        page: int = 1,
        page_size: int = 100,
        sort: str = "created_at",
        filter_operation: FilterOperation | None = None,
        all_versions: bool = False,
    ) -> Page[ModelDeployment]:
        """List deployments with filtering and pagination.

        Args:
            workspace: Workspace to query
            page: Page number
            page_size: Items per page
            sort: Field to sort by
            filter_operation: Structured filter operation tree
            all_versions: If True, return all versions. If False (default), return only latest
                version of each deployment.
        """
        logger.debug(
            f"Listing deployments: page={page}, page_size={page_size}, sort={sort}, all_versions={all_versions}"
        )

        if all_versions:
            # Return all versions with server-side pagination
            result = await self.entity_client.list(
                ModelDeploymentEntity,
                workspace=workspace,
                filter_operation=filter_operation,
                sort=sort,
                page=page,
                page_size=page_size,
            )

            deployments = [_entity_to_schema(entity) for entity in result.data]

            return Page(
                data=deployments,
                pagination=PaginationData(
                    page=result.pagination.page,
                    page_size=result.pagination.page_size,
                    current_page_size=len(deployments),
                    total_pages=result.pagination.total_pages,
                    total_results=result.pagination.total_results,
                ),
                sort=sort,
                filter=None,
            )
        else:
            # Get all deployments and filter to latest versions client-side
            # Note: For large datasets, this should use server-side aggregation
            result = await self.entity_client.list(
                ModelDeploymentEntity,
                workspace=workspace,
                filter_operation=filter_operation,
                sort=sort,
                page=1,
                page_size=1000,  # Get all to filter latest versions (max allowed)
            )

            # Group by base_name and keep highest version
            latest_by_name: dict[str, ModelDeploymentEntity] = {}
            for dep in result.data:
                existing = latest_by_name.get(dep.base_name)
                if existing is None or dep.entity_version > existing.entity_version:
                    latest_by_name[dep.base_name] = dep

            all_latest = list(latest_by_name.values())

            # Apply pagination
            total_results = len(all_latest)
            total_pages = (total_results + page_size - 1) // page_size if total_results > 0 else 1
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            paginated = all_latest[start_idx:end_idx]

            logger.debug(f"Found {len(paginated)} deployments (latest versions)")

            deployments = [_entity_to_schema(entity) for entity in paginated]

            return Page(
                data=deployments,
                pagination=PaginationData(
                    page=page,
                    page_size=page_size,
                    current_page_size=len(deployments),
                    total_pages=total_pages,
                    total_results=total_results,
                ),
                sort=sort,
                filter=None,
            )

    async def list_deployment_versions(self, workspace: str, name: str) -> list[ModelDeployment]:
        """List all versions of a specific deployment."""
        logger.debug(f"Listing deployment versions: {workspace}/{name}")

        filter_str = json.dumps({"data.base_name": name})
        result = await self.entity_client.list(
            ModelDeploymentEntity,
            workspace=workspace,
            filter_str=filter_str,
        )

        # Sort by version descending
        versions = sorted(result.data, key=lambda d: d.entity_version, reverse=True)
        logger.debug(f"Found {len(versions)} versions for deployment {workspace}/{name}")

        return [_entity_to_schema(entity) for entity in versions]

    async def update_deployment(
        self,
        workspace: str,
        name: str,
        request: UpdateModelDeploymentRequest,
        auth_context: AuthContext | None = None,
    ) -> ModelDeployment:
        """Update a deployment (creates a new version)."""
        logger.info(f"Updating deployment: {workspace}/{name}")

        current = await self._get_by_base_name_and_version(workspace, name)
        if not current:
            raise ValueError(f"Deployment with workspace '{workspace}' and name '{name}' does not exist")

        # Determine config_version: use from request or preserve current
        config_version = request.config_version
        if not config_version:
            config_version = await self._get_config_latest_version(workspace, request.config)
            if not config_version:
                raise ValueError(f"Deployment config '{workspace}/{request.config}' does not exist")
            logger.debug(f"Using latest config version: {config_version}")

        # Verify the deployment config exists
        config = await self._get_config_by_version(workspace, request.config, config_version)
        if not config:
            raise ValueError(
                f"Deployment config '{workspace}/{request.config}' version '{config_version}' does not exist"
            )

        new_version = current.entity_version + 1

        # Create new version entity
        entity = ModelDeploymentEntity(
            name=f"{name}-v{new_version}",
            workspace=workspace,
            base_name=name,
            entity_version=new_version,
            project=current.project,  # Preserve project
            config=request.config,
            config_version=config_version,
            status=ModelDeploymentStatus.PENDING,
            status_message="Deployment update pending",
        ).with_auth_context(auth_context)

        try:
            created = await self.entity_client.create(entity)
            logger.info(
                f"Successfully updated deployment: {created.workspace}/{created.base_name} "
                f"to version {created.entity_version}"
            )
            return _entity_to_schema(created)
        except EntityConflictError as e:
            logger.exception(f"Failed to update deployment {workspace}/{name}")
            raise ValueError(f"Failed to create new version of deployment: {e}") from e

    async def update_deployment_status(
        self,
        workspace: str,
        name: str,
        request: UpdateModelDeploymentStatusRequest,
        version: int | None = None,
        timestamp: datetime | None = None,
    ) -> ModelDeployment | None:
        """Update the status of a deployment."""
        logger.info(
            f"Updating deployment status: {workspace}/{name}" + (f" version {version}" if version else " (latest)")
        )

        entity = await self._get_by_base_name_and_version(workspace, name, version)
        if not entity:
            logger.warning(
                f"Deployment not found for status update: {workspace}/{name}"
                + (f" version {version}" if version else "")
            )
            return None

        if entity.status == ModelDeploymentStatus.DELETING and request.status != ModelDeploymentStatus.DELETED:
            raise DeploymentStatusConflictError(
                "Deployment is marked for deletion (DELETING). Only transition to DELETED is allowed."
            )

        # Deduplication: only update if status or status_message changed
        request_msg = request.status_message if request.status_message is not None else ""
        status_changed = entity.status != request.status
        message_changed = (entity.status_message or "") != request_msg
        if not status_changed and not message_changed:
            logger.debug(f"No status change for {workspace}/{name}, skipping update")
            return _entity_to_schema(entity)

        # Record previous status in history, then apply update, then record new status so current is always in the list
        current_time = timestamp or datetime.now(timezone.utc)
        entity.status_history.append(_status_history_entry(current_time, entity.status, entity.status_message or ""))

        # Apply status updates
        entity.status = request.status
        entity.status_message = request_msg
        if request.model_provider_id is not None:
            entity.model_provider_id = request.model_provider_id

        # Append current status to history so consumers can use history alone (last entry = current)
        entity.status_history.append(_status_history_entry(current_time, entity.status, entity.status_message or ""))

        # Compact adjacent duplicates (same status + message), keeping first-seen timestamp
        entity.status_history = _compact_adjacent_status_history(entity.status_history)

        # Keep only most recent 100 entries
        if len(entity.status_history) > 100:
            entity.status_history = entity.status_history[-100:]

        updated = await self.entity_client.update(entity)
        logger.info(
            f"Successfully updated deployment status: {updated.workspace}/{updated.base_name} "
            f"version {updated.entity_version} to {updated.status}"
        )
        return _entity_to_schema(updated)

    async def delete_deployment(self, workspace: str, name: str, version: int | None = None) -> ModelDeployment | None:
        """Mark a deployment or specific version for deletion.

        Sets the deployment status to DELETING. The controller will pick this up,
        delete the infrastructure, then update the status to DELETED.

        If the deployment is already DELETED, actually delete it from the database.

        Returns:
            The updated deployment with DELETING status, or None if not found or hard-deleted
        """
        if version:
            logger.info(f"Marking deployment version for deletion: {workspace}/{name} version {version}")
        else:
            logger.info(f"Marking all versions of deployment for deletion: {workspace}/{name}")

        if version:
            # Handle specific version
            entity = await self._get_by_base_name_and_version(workspace, name, version)
            if not entity:
                logger.warning(f"Deployment not found: {workspace}/{name} version {version}")
                return None

            # If already DELETED, hard delete from database
            if entity.status == ModelDeploymentStatus.DELETED:
                logger.info(f"Deployment already DELETED, removing from database: {workspace}/{name} version {version}")
                await self.entity_client.delete(
                    ModelDeploymentEntity,
                    entity.name,
                    workspace=workspace,
                    expected_db_version=entity.db_version,
                )
                return None

            # Otherwise, mark for deletion
            entity.status = ModelDeploymentStatus.DELETING
            entity.status_message = "Deployment deletion requested"
            # Append to history so consumers reading status_history[-1] see the
            # DELETING state (the history is the client's source of current status).
            entity.status_history.append(
                _status_history_entry(datetime.now(timezone.utc), entity.status, entity.status_message)
            )
            updated = await self.entity_client.update(entity)
            logger.info(f"Marked deployment for deletion: {workspace}/{name} version {version}")
            return _entity_to_schema(updated)
        else:
            # For deleting all versions, get all versions
            filter_str = json.dumps({"data.base_name": name})
            result = await self.entity_client.list(
                ModelDeploymentEntity,
                workspace=workspace,
                filter_str=filter_str,
            )

            if not result.data:
                logger.warning(f"No deployments found: {workspace}/{name}")
                return None

            # Check if all versions are already DELETED
            all_deleted = all(v.status == ModelDeploymentStatus.DELETED for v in result.data)

            if all_deleted:
                logger.info(f"All versions already DELETED, removing from database: {workspace}/{name}")
                for entity in result.data:
                    await self.entity_client.delete(
                        ModelDeploymentEntity,
                        entity.name,
                        workspace=workspace,
                        expected_db_version=entity.db_version,
                    )
                return None

            # Mark all non-DELETED versions for deletion
            latest_updated = None
            for entity in result.data:
                if entity.status != ModelDeploymentStatus.DELETED:
                    entity.status = ModelDeploymentStatus.DELETING
                    entity.status_message = "Deployment deletion requested"
                    entity.status_history.append(
                        _status_history_entry(datetime.now(timezone.utc), entity.status, entity.status_message)
                    )
                    updated = await self.entity_client.update(entity)
                    latest_updated = updated

            logger.info(f"Marked all non-DELETED versions for deletion: {workspace}/{name}")
            return _entity_to_schema(latest_updated) if latest_updated else None
