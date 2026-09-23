# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Service layer for ModelDeploymentConfig operations using EntityClient."""

import json
import logging

from nemo_platform import PermissionDeniedError
from nmp.common.api.common import Page, PaginationData
from nmp.common.api.filter import FilterOperation
from nmp.common.entities.client import EntityClient, EntityConflictError, EntityNotFoundError
from nmp.common.entities.global_workspace import is_global_workspace
from nmp.core.models.app.utils import parse_model_name_revision as parse_model_name_revision
from nmp.core.models.entities import Model as ModelEntity
from nmp.core.models.entities import ModelDeployment as ModelDeploymentEntity
from nmp.core.models.entities import ModelDeploymentConfig as ModelDeploymentConfigEntity
from nmp.core.models.schemas import (
    ContainerExecutorConfig,
    CreateModelDeploymentConfigRequest,
    Engine,
    ModelDeploymentConfig,
    ModelDeploymentConfigModelSpec,
    ModelDeploymentStatus,
    UpdateModelDeploymentConfigRequest,
)

logger = logging.getLogger(__name__)


def _validate_engine_config(
    engine: Engine,
    executor_config: ContainerExecutorConfig,
    model_spec: ModelDeploymentConfigModelSpec,
) -> None:
    """Validate engine-specific requirements on the deployment config.

    The ``generic`` engine runs an arbitrary container with no inference-engine
    compiler, so it has no platform-default image and no canonical health
    endpoint. Both ``image_name`` and ``health_check_path`` must therefore be
    supplied explicitly; the other engines fall back to their configured
    defaults when these are unset.

    Values are also rejected when they contain surrounding whitespace: an
    image reference or probe path is used verbatim downstream, where a
    leading/trailing space would silently produce an invalid value.

    LoRA is rejected for ``generic``: there is no engine compiler to wire the
    adapter sidecar against, so ``lora_enabled`` would otherwise be silently
    ignored. Reject it up front rather than accept a config that can't be honored.
    """
    if engine != Engine.GENERIC:
        return
    missing: list[str] = []
    padded: list[str] = []
    for field in ("image_name", "health_check_path"):
        value = getattr(executor_config, field)
        if not (value and value.strip()):
            missing.append(field)
        elif value != value.strip():
            padded.append(field)
    if missing:
        raise ValueError(
            "The 'generic' engine requires executor_config."
            + " and executor_config.".join(missing)
            + " to be set (no platform default exists for a generic container)."
        )
    if padded:
        raise ValueError(
            "executor_config." + " and executor_config.".join(padded) + " must not have leading or trailing whitespace."
        )
    if model_spec.lora_enabled:
        raise ValueError(
            "The 'generic' engine does not support LoRA (model_spec.lora_enabled); "
            "there is no engine compiler to wire the adapter sidecar against."
        )


class ReferentialIntegrityError(Exception):
    """Exception raised when trying to delete a resource that has dependencies."""

    def __init__(self, resource_type: str, resource_id: str, dependent_count: int, message: str | None = None):
        self.resource_type = resource_type
        self.resource_id = resource_id
        self.dependent_count = dependent_count
        self.message = message or (
            f"Cannot delete {resource_type} '{resource_id}' because {dependent_count} "
            f"dependent resource(s) still reference it"
        )
        super().__init__(self.message)


def _entity_to_schema(entity: ModelDeploymentConfigEntity) -> ModelDeploymentConfig:
    """Convert an EntityBase ModelDeploymentConfig to the API schema."""
    return ModelDeploymentConfig(
        id=entity.id,
        name=entity.base_name,  # Use base_name as the logical name
        workspace=entity.workspace,
        project=entity.project,
        description=entity.description,
        engine=entity.engine,
        model_spec=entity.model_spec,
        executor_config=entity.executor_config,
        model_entity_id=entity.model_entity_id,
        entity_version=entity.entity_version,
        created_at=entity.created_at,
        updated_at=entity.updated_at,
    )


class ShadowedSharedModelError(ValueError):
    """Raised when deploying a global model into a workspace that has its own model of that name."""


class ModelDeploymentConfigService:
    """Service layer for ModelDeploymentConfig operations."""

    def __init__(self, entity_client: EntityClient):
        self.entity_client = entity_client

    async def _refuse_shadowed_shared_model(
        self,
        workspace: str,
        model_entity_id: str | None,
        model_spec: ModelDeploymentConfigModelSpec,
    ) -> None:
        """A global model deployed elsewhere is routed under the deploying workspace's name for it."""
        if model_entity_id:
            model_workspace, model_name, _ = parse_model_name_revision(model_name=model_entity_id)
        else:
            model_workspace, model_name, _ = parse_model_name_revision(
                model_namespace=model_spec.model_namespace,
                model_name=model_spec.model_name,
                model_revision=model_spec.model_revision,
            )
        if not model_workspace or not model_name:
            return
        if not is_global_workspace(model_workspace) or is_global_workspace(workspace):
            return
        try:
            await self.entity_client.get(ModelEntity, workspace=workspace, name=model_name, local_only=True)
        except EntityNotFoundError:
            return
        raise ShadowedSharedModelError(
            f"Workspace '{workspace}' has its own model named '{model_name}', so a deployment of "
            f"'{model_workspace}/{model_name}' here would be routed as that model. "
            f"Rename the local model or deploy from a workspace without one."
        )

    async def _get_latest_version(self, workspace: str, base_name: str) -> int | None:
        """Get the highest version number for a deployment config."""
        filter_str = json.dumps({"data.base_name": base_name})
        result = await self.entity_client.list(
            ModelDeploymentConfigEntity,
            workspace=workspace,
            filter_str=filter_str,
        )
        if not result.data:
            return None
        return max(config.entity_version for config in result.data)

    async def _get_by_base_name_and_version(
        self, workspace: str, base_name: str, version: int | None = None
    ) -> ModelDeploymentConfigEntity | None:
        """Get a deployment config by base name and optional version."""
        if version is None:
            version = await self._get_latest_version(workspace, base_name)
            if version is None:
                return None

        entity_name = f"{base_name}-v{version}"
        try:
            return await self.entity_client.get(
                ModelDeploymentConfigEntity,
                workspace=workspace,
                name=entity_name,
            )
        except EntityNotFoundError:
            return None

    async def create_deployment_config(
        self, request: CreateModelDeploymentConfigRequest, workspace: str
    ) -> ModelDeploymentConfig:
        """Create a new deployment config (version 1)."""
        logger.info(f"Creating deployment config: {workspace}/{request.name}")

        # Check if config with this name already exists
        existing = await self._get_latest_version(workspace, request.name)
        if existing is not None:
            raise ValueError(f"Deployment config with workspace '{workspace}' and name '{request.name}' already exists")

        _validate_engine_config(request.engine, request.executor_config, request.model_spec)

        if not request.model_entity_id:
            try:
                model_workspace, model_name, _ = parse_model_name_revision(
                    model_namespace=request.model_spec.model_namespace,
                    model_name=request.model_spec.model_name,
                    model_revision=request.model_spec.model_revision,
                )
                if not model_name:
                    logger.warning("No model name found in the model_spec, skipping the model entity id set")
                else:
                    me: ModelEntity = await self.entity_client.get(
                        workspace=model_workspace,
                        name=model_name,
                        entity_type=ModelEntity,
                    )
                    request.model_entity_id = f"{me.workspace}/{me.name}"

            except (EntityNotFoundError, PermissionDeniedError) as err:
                logger.warning(f"Failed to fetch the model entity referenced in the model_spec {err}")

        await self._refuse_shadowed_shared_model(workspace, request.model_entity_id, request.model_spec)

        # Create the entity with versioned name
        entity = ModelDeploymentConfigEntity(
            name=f"{request.name}-v1",
            workspace=workspace,
            base_name=request.name,
            entity_version=1,
            project=request.project,
            description=request.description,
            engine=request.engine,
            model_spec=request.model_spec,
            executor_config=request.executor_config,
            model_entity_id=request.model_entity_id,
        )

        try:
            created = await self.entity_client.create(entity)
            logger.info(
                f"Successfully created deployment config: {created.workspace}/{created.base_name} version {created.entity_version}"
            )
            return _entity_to_schema(created)
        except EntityConflictError as e:
            logger.warning(f"Deployment config already exists: {workspace}/{request.name}")
            raise ValueError(
                f"Deployment config with name '{request.name}' already exists in workspace '{workspace}'"
            ) from e

    async def get_deployment_config(
        self, workspace: str, name: str, version: int | None = None
    ) -> ModelDeploymentConfig | None:
        """Get a deployment config by workspace, name, and optionally version."""
        logger.debug(
            f"Getting deployment config: {workspace}/{name}" + (f" version {version}" if version else " (latest)")
        )

        entity = await self._get_by_base_name_and_version(workspace, name, version)
        if entity:
            logger.debug(
                f"Found deployment config: {entity.workspace}/{entity.base_name} version {entity.entity_version}"
            )
            return _entity_to_schema(entity)

        logger.debug(f"Deployment config not found: {workspace}/{name}" + (f" version {version}" if version else ""))
        return None

    async def list_deployment_configs(
        self,
        workspace: str,
        page: int = 1,
        page_size: int = 100,
        sort: str = "created_at",
        filter_operation: FilterOperation | None = None,
    ) -> Page[ModelDeploymentConfig]:
        """List deployment configs with filtering and pagination.

        Returns only the latest version of each config.
        """
        logger.debug(f"Listing deployment configs: page={page}, page_size={page_size}, sort={sort}")

        # Get all configs (we'll filter to latest versions client-side)
        # Note: For large datasets, this should use server-side aggregation
        result = await self.entity_client.list(
            ModelDeploymentConfigEntity,
            workspace=workspace,
            filter_operation=filter_operation,
            sort=sort,
            page=1,
            page_size=1000,  # Get all to filter latest versions (max allowed)
        )

        # Group by base_name and keep highest version
        latest_by_name: dict[str, ModelDeploymentConfigEntity] = {}
        for config in result.data:
            existing = latest_by_name.get(config.base_name)
            if existing is None or config.entity_version > existing.entity_version:
                latest_by_name[config.base_name] = config

        all_latest = list(latest_by_name.values())

        # Apply pagination
        total_results = len(all_latest)
        total_pages = (total_results + page_size - 1) // page_size if total_results > 0 else 1
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated = all_latest[start_idx:end_idx]

        logger.debug(f"Found {len(paginated)} deployment configs (latest versions)")

        # Convert to schema
        configs = [_entity_to_schema(entity) for entity in paginated]

        return Page(
            data=configs,
            pagination=PaginationData(
                page=page,
                page_size=page_size,
                current_page_size=len(configs),
                total_pages=total_pages,
                total_results=total_results,
            ),
            sort=sort,
            filter=None,
        )

    async def list_deployment_config_versions(self, workspace: str, name: str) -> list[ModelDeploymentConfig]:
        """List all versions of a specific deployment config."""
        logger.debug(f"Listing deployment config versions: {workspace}/{name}")

        filter_str = json.dumps({"data.base_name": name})
        result = await self.entity_client.list(
            ModelDeploymentConfigEntity,
            workspace=workspace,
            filter_str=filter_str,
        )

        # Sort by version descending
        versions = sorted(result.data, key=lambda c: c.entity_version, reverse=True)
        logger.debug(f"Found {len(versions)} versions for deployment config {workspace}/{name}")

        return [_entity_to_schema(entity) for entity in versions]

    async def update_deployment_config(
        self, workspace: str, name: str, request: UpdateModelDeploymentConfigRequest
    ) -> ModelDeploymentConfig:
        """Update a deployment config (creates a new version)."""
        logger.info(f"Updating deployment config: {workspace}/{name}")

        # Get the current config to preserve fields and get version
        current = await self._get_by_base_name_and_version(workspace, name)
        if not current:
            raise ValueError(f"Deployment config with workspace '{workspace}' and name '{name}' does not exist")

        _validate_engine_config(request.engine, request.executor_config, request.model_spec)
        await self._refuse_shadowed_shared_model(workspace, request.model_entity_id, request.model_spec)

        new_version = current.entity_version + 1

        # Create new version entity
        entity = ModelDeploymentConfigEntity(
            name=f"{name}-v{new_version}",
            workspace=workspace,
            base_name=name,
            entity_version=new_version,
            project=current.project,  # Preserve project
            description=request.description,
            engine=request.engine,
            model_spec=request.model_spec,
            executor_config=request.executor_config,
            model_entity_id=request.model_entity_id,
        )

        try:
            created = await self.entity_client.create(entity)
            logger.info(
                f"Successfully updated deployment config: {created.workspace}/{created.base_name} to version {created.entity_version}"
            )
            return _entity_to_schema(created)
        except EntityConflictError as e:
            logger.exception(f"Failed to update deployment config {workspace}/{name}")
            raise ValueError(f"Failed to create new version of deployment config: {e}") from e

    async def delete_deployment_config(self, workspace: str, name: str, version: int | None = None) -> bool:
        """Delete a deployment config or specific version.

        Raises:
            ReferentialIntegrityError: If there are ModelDeployments that reference this config
                                    and are not in DELETED status
        """
        if version:
            logger.info(f"Deleting deployment config version: {workspace}/{name} version {version}")
        else:
            logger.info(f"Deleting all versions of deployment config: {workspace}/{name}")

        # Check for dependent deployments before allowing deletion
        filter_str = json.dumps({"data.config": name})
        dep_result = await self.entity_client.list(
            ModelDeploymentEntity,
            workspace=workspace,
            filter_str=filter_str,
            page_size=1000,
        )

        # Group by base_name and keep only latest version of each deployment
        latest_by_name: dict[str, ModelDeploymentEntity] = {}
        for dep in dep_result.data:
            existing = latest_by_name.get(dep.base_name)
            if existing is None or dep.entity_version > existing.entity_version:
                latest_by_name[dep.base_name] = dep

        referencing_deployments = list(latest_by_name.values())

        if referencing_deployments:
            config_id = f"{workspace}/{name}"
            if version:
                # For specific version, check if any deployments reference that exact version
                version_refs = [d for d in referencing_deployments if d.config_version == version]
                non_deleted_refs = [d for d in version_refs if d.status != ModelDeploymentStatus.DELETED]
                if non_deleted_refs:
                    raise ReferentialIntegrityError(
                        resource_type="ModelDeploymentConfig",
                        resource_id=f"{config_id} version {version}",
                        dependent_count=len(non_deleted_refs),
                        message=(
                            f"Cannot delete ModelDeploymentConfig '{config_id}' version {version} "
                            f"because {len(non_deleted_refs)} ModelDeployment(s) still reference it "
                            f"and are not in DELETED status"
                        ),
                    )
            else:
                # For all versions, check if any deployments reference any version
                non_deleted_deployments = [
                    d for d in referencing_deployments if d.status != ModelDeploymentStatus.DELETED
                ]
                if non_deleted_deployments:
                    raise ReferentialIntegrityError(
                        resource_type="ModelDeploymentConfig",
                        resource_id=config_id,
                        dependent_count=len(non_deleted_deployments),
                        message=(
                            f"Cannot delete ModelDeploymentConfig '{config_id}' "
                            f"because {len(non_deleted_deployments)} ModelDeployment(s) still reference it "
                            f"and are not in DELETED status. Delete the dependent deployments first."
                        ),
                    )

        if version:
            # Delete specific version
            entity = await self._get_by_base_name_and_version(workspace, name, version)
            if not entity:
                logger.warning(f"Deployment config not found for deletion: {workspace}/{name} version {version}")
                return False

            await self.entity_client.delete(
                ModelDeploymentConfigEntity,
                entity.name,
                workspace=workspace,
            )
            logger.info(f"Successfully deleted deployment config: {workspace}/{name} version {version}")
            return True
        else:
            # Delete all versions
            filter_str = json.dumps({"data.base_name": name})
            result = await self.entity_client.list(
                ModelDeploymentConfigEntity,
                workspace=workspace,
                filter_str=filter_str,
            )

            if not result.data:
                logger.warning(f"Deployment config not found for deletion: {workspace}/{name}")
                return False

            for entity in result.data:
                await self.entity_client.delete(
                    ModelDeploymentConfigEntity,
                    entity.name,
                    workspace=workspace,
                )

            logger.info(f"Successfully deleted all versions of deployment config: {workspace}/{name}")
            return True

    async def get_latest_version(self, workspace: str, name: str) -> int | None:
        """Get the latest version number for a deployment config.

        This is a public method used by ModelDeploymentService.
        """
        return await self._get_latest_version(workspace, name)
