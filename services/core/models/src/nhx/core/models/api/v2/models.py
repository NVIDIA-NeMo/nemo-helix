# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status
from nemo_helix import AsyncNeMoHelix
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.client.errors import NemoClientError
from nemo_helix_plugin.jobs.api_factory import (
    ContainerSpec,
    CPUExecutionProviderSpec,
    EnvironmentVariable,
    HelixJobSpec,
    HelixJobStep,
    ResourcesLimitsSpec,
    ResourcesRequestsSpec,
    ResourcesSpec,
)
from nemo_helix_plugin.jobs.client import AsyncJobsClient
from nemo_helix_plugin.jobs.image import get_qualified_image
from nemo_helix_plugin.jobs.spec import HelixJobSpec as HelixJobSpecModel
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest
from nhx.common.api.common import Page
from nhx.common.api.parsed_filter import ParsedFilter, make_filter_dep
from nhx.common.api.utils import generate_openapi_extra_params
from nhx.common.auth import AuthClient, get_auth_client
from nhx.common.entities.client import EntityConflictError, EntityNotFoundError, EntityValidationError
from nhx.common.sdk_factory import get_async_platform_sdk
from nhx.common.service.dependencies import get_sdk_client
from nhx.core.models.api.dependencies import get_adapter_entity_service, get_model_entity_service
from nhx.core.models.api.permissions import check_fileset_access
from nhx.core.models.api.service.adapter_entity_service import AdapterEntityService
from nhx.core.models.api.service.model_entity_service import (
    FilesetValidationError,
    InvalidFilterError,
    ModelEntityService,
    _has_tool_call_plugin,
    fileset_has_tool_call_plugin,
    is_trusted_repo_id,
    validate_tool_call_plugin_allowed,
)
from nhx.core.models.config import config as models_config
from nhx.core.models.entities import Model
from nhx.core.models.schemas import (
    Adapter,
    CreateModelAdapterRequest,
    CreateModelEntityRequest,
    ModelEntity,
    ModelEntityFilter,
    ModelEntitySortField,
    UpdateAdapterRequest,
    UpdateModelEntityRequest,
)
from nhx.core.models.tasks.model_spec.schemas import (
    DEFAULT_TASK_STORAGE_PATH,
    EPHEMERAL_TASK_STORAGE_PATH_ENVVAR,
    ModelSpecTaskConfig,
)

logger = logging.getLogger(__name__)

router = APIRouter()


async def _check_tool_call_plugin_permission(
    request,
    auth_client: AuthClient,
    workspace: str,
    fileset=None,
) -> None:
    """Enforce the tool_call_plugin permission gate for a model create/update.

    Checks two sources for tool_call_plugin usage:
    1. The request body itself (spec.tool_call_config.tool_call_plugin or
       model_spec.tool_call_config.tool_call_plugin).
    2. The already-retrieved fileset's metadata
       (metadata.model.tool_calling.tool_call_plugin), if provided.

    If either source specifies a plugin, validates that the platform-level flag
    is enabled and the caller holds the models.tool-call-plugin.set permission.
    Raises PermissionError on failure.
    """
    needs_perm = _has_tool_call_plugin(request)
    if not needs_perm and fileset is not None:
        needs_perm = fileset_has_tool_call_plugin(fileset)

    if needs_perm:
        await validate_tool_call_plugin_allowed(auth_client, workspace)


@router.post(
    "/v2/workspaces/{workspace}/models",
    summary="Create Model",
    response_description="Create a new model entity",
    status_code=status.HTTP_201_CREATED,
    response_model=ModelEntity,
)
async def create_model(
    workspace: str,
    model_input: CreateModelEntityRequest,
    service: ModelEntityService = Depends(get_model_entity_service),
    nhx_sdk: AsyncNeMoHelix = Depends(get_sdk_client),
    auth_client: AuthClient = Depends(get_auth_client),
) -> ModelEntity:
    """
    Create a new model entity.

    This endpoint creates a new Model Entity in the Models service database.
    The Model Entity will be registered for use within the platform.
    """
    logger.debug(f"Creating model entity: {workspace}/{model_input.name}")

    try:
        fs = None
        if model_input.fileset:
            fs = await check_fileset_access(nhx_sdk, model_input.fileset, workspace)

        await _check_tool_call_plugin_permission(model_input, auth_client, workspace, fileset=fs)

        model_input.trust_remote_code = await set_trust_remote_code(
            nhx_sdk,
            model_input.trust_remote_code,
            model_input.fileset,
            auth_client,
            workspace,
        )

        # Create the model using service
        created_model = await service.create_model_entity(model_input, workspace)

    except PermissionError as e:
        logger.warning(f"Permission denied during model creation: {e}")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except FilesetValidationError as e:
        logger.warning(f"Fileset validation error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except EntityValidationError as e:
        logger.warning(f"Entity store validation error during model creation: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(e),
        )
    except ValueError as e:
        if "already exists" in str(e).lower():
            logger.warning(f"Model already exists: {workspace}/{model_input.name}")
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Model with workspace '{workspace}' and name '{model_input.name}' already exists",
            )
        logger.warning(f"Model creation validation error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception(f"Failed to create model entity - {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create model entity")

    # add sdk job creation here for checkpoint metadata
    if created_model.fileset:
        await start_update_model_spec_job(created_model)
    return created_model


@router.get(
    "/v2/workspaces/{workspace}/models",
    summary="List Models",
    response_description="Return a list of models",
    status_code=status.HTTP_200_OK,
    response_model=Page[ModelEntity],
    response_model_exclude_none=True,
    openapi_extra=generate_openapi_extra_params(
        filter_schema=ModelEntityFilter,
        filter_description=(
            "Filter models by name, project, workspace, base_model, adapters, "
            "finetuning_type, prompt, lora_enabled, description, created_at, and updated_at."
        ),
    ),
)
async def list_models(
    workspace: str,
    page: int = Query(default=1, description="Page number."),
    page_size: int = Query(default=100, description="Page size."),
    sort: ModelEntitySortField = Query(
        default="created_at",
        description="The field to sort by. To sort in decreasing order, use `-` in front of the field name.",
    ),
    parsed_filter: ParsedFilter = Depends(make_filter_dep(ModelEntityFilter)),
    verbose: bool = Query(
        default=False,
        description="Whether to include full spec details",
    ),
    service: ModelEntityService = Depends(get_model_entity_service),
) -> Page[ModelEntity]:
    """
    List Models endpoint with filtering, pagination, and sorting.

    Supports filter parameters for various criteria (including peft, custom fields),
    pagination (page, page_size), sorting, and workspace filtering via query parameter.
    """
    try:
        # Extract workspace — inject from path param if not in filter
        filter_workspace = parsed_filter.remove("workspace") or workspace

        result = await service.list_model_entities(
            workspace=filter_workspace,
            parsed_filter=parsed_filter,
            page=page,
            page_size=page_size,
            sort=sort,
            verbose=verbose,
        )

        return result

    except HTTPException:
        raise
    except InvalidFilterError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception:
        logger.exception("Failed to list model entities")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list model entities")


@router.get(
    "/v2/workspaces/{workspace}/models/{name}",
    summary="Get Model by Workspace and Name",
    response_description="Return model details",
    status_code=status.HTTP_200_OK,
    response_model=ModelEntity,
)
async def get_model(
    workspace: str,
    name: str,
    verbose: bool = Query(
        default=False,
        description="Whether to include full spec details",
    ),
    service: ModelEntityService = Depends(get_model_entity_service),
) -> ModelEntity:
    """
    Get Model by Workspace and Name.

    Returns the details of a specific model entity identified by its workspace and name.
    """
    model_name = name
    logger.debug(f"Getting model entity: {workspace}/{model_name}")

    try:
        # Get the model using service
        model_entity = await service.get_model_entity(workspace, model_name, verbose=verbose)

        if model_entity is None:
            logger.debug(f"Model not found: {workspace}/{model_name}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model with workspace '{workspace}' and name '{model_name}' not found",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to get model entity - {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get model entity")

    return model_entity


async def start_update_model_spec_job(model_entity: ModelEntity):
    sdk = get_async_platform_sdk(as_service="models", internal=True)
    model_spec_task_config = ModelSpecTaskConfig(workspace=model_entity.workspace, name=model_entity.name)
    task_spec = HelixJobSpec(
        steps=[
            # Step 1: Download model and dataset files from Files service
            HelixJobStep(
                name="model-spec-analysis",
                executor=CPUExecutionProviderSpec(
                    provider="cpu",
                    # Customizer uses the ``gpu`` profile for both CPU preparation
                    # and GPU training steps. The provider selects CPU resources here.
                    profile="gpu",
                    container=ContainerSpec(
                        image=get_qualified_image("nhx-customizer-tasks"),
                        entrypoint=["/opt/venv/bin/python"],
                        command=["-m", "nhx.core.models.tasks.model_spec"],
                    ),
                    resources=ResourcesSpec(
                        requests=ResourcesRequestsSpec(
                            cpu="1",
                            memory="8Gi",
                        ),
                        limits=ResourcesLimitsSpec(
                            cpu="4",
                            memory="32Gi",
                        ),
                    ),
                ),
                environment=[
                    EnvironmentVariable(
                        name=EPHEMERAL_TASK_STORAGE_PATH_ENVVAR,
                        value=DEFAULT_TASK_STORAGE_PATH,
                    ),
                    EnvironmentVariable(
                        name="GPU_MEM_GB",
                        value=str(models_config.parallelism.gpu_memory_gb_default),
                    ),
                ],
                config=model_spec_task_config.model_dump(mode="json"),
            ),
        ]
    )
    try:
        jobs = client_from_platform(sdk, AsyncJobsClient)
        job_resp = (
            await jobs.create_job(
                workspace=model_entity.workspace,
                body=CreateHelixJobRequest(
                    source="models-system",
                    # ``task_spec`` is built from the api_factory ``*Param`` TypedDict
                    # aliases (see AIRCORE-922); validate it into the plugin pydantic
                    # ``HelixJobSpec`` the request model expects.
                    platform_spec=HelixJobSpecModel.model_validate(task_spec),
                    spec={},
                    description=f"Model Spec Analyzer for model {model_entity.workspace}/{model_entity.name}",
                    ownership=model_entity.ownership,
                    project=model_entity.project,
                ),
            )
        ).data()
        logger.info("Job Created - %s", job_resp.name)
    except NemoClientError as err:
        logger.warning(f"Failed to create model spec job. {err}")


@router.patch(
    "/v2/workspaces/{workspace}/models/{name}",
    summary="Update Model",
    response_description="Update model metadata",
    status_code=status.HTTP_200_OK,
    response_model=ModelEntity,
)
async def update_model(
    workspace: str,
    name: str,
    model_update: UpdateModelEntityRequest,
    verbose: bool = Query(
        default=False,
        description="Whether to include full spec details",
    ),
    service: ModelEntityService = Depends(get_model_entity_service),
    nhx_sdk: AsyncNeMoHelix = Depends(get_sdk_client),
    auth_client: AuthClient = Depends(get_auth_client),
) -> ModelEntity:
    """
    Update Model metadata.

    Updates the metadata of an existing model entity. If the request body has an empty field,
    the old value is kept.
    """
    model_name = name
    logger.info(f"Updating model entity: {workspace}/{model_name}")

    try:
        # Get existing model
        model: Model = await service.entity_client.get(Model, workspace=workspace, name=name)
    except EntityNotFoundError:
        logger.warning(f"Model entity not found for update: {workspace}/{name}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model with workspace '{workspace}' and name '{model_name}' not found",
        )

    try:
        fs = None
        if model_update.fileset:
            fs = await check_fileset_access(nhx_sdk, model_update.fileset, workspace)

        await _check_tool_call_plugin_permission(model_update, auth_client, workspace, fileset=fs)

        if model_update.trust_remote_code or model_update.fileset:
            model_update.trust_remote_code = await set_trust_remote_code(
                nhx_sdk,
                model_update.trust_remote_code or model.trust_remote_code,
                model_update.fileset or model.fileset,
                auth_client,
                workspace,
            )
        # Update the model using service
        model_entity = await service.get_model_entity(workspace, model_name, verbose=False)

        if model_entity is None:
            logger.warning(f"Model not found: {workspace}/{model_name}")

        original_fileset = model_entity.fileset

        updated_model = await service.update_model_entity(model, workspace, model_name, model_update, verbose=verbose)

        if updated_model is None:
            logger.warning(f"Model not found for update: {workspace}/{model_name}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model with workspace '{workspace}' and name '{model_name}' not found",
            )
    except EntityValidationError as e:
        logger.warning(f"Entity store validation error during model update: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(e),
        )
    except PermissionError as e:
        logger.warning(f"Permission denied during model update: {e}")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except FilesetValidationError as e:
        logger.warning(f"Fileset validation error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except ValueError as e:
        logger.warning(f"Model update validation error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to update model entity - {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update model entity")

    if updated_model.fileset and (updated_model.fileset != original_fileset or not updated_model.spec):
        await start_update_model_spec_job(updated_model)

    return updated_model


@router.delete(
    "/v2/workspaces/{workspace}/models/{name}",
    summary="Delete Model",
    response_description="Delete model entity",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_model(
    workspace: str,
    name: str,
    service: ModelEntityService = Depends(get_model_entity_service),
):
    """
    Delete Model entity.

    Permanently deletes a model entity from the platform.
    """
    model_name = name
    logger.info(f"Deleting model entity: {workspace}/{model_name}")

    try:
        # Delete the model using service
        deleted = await service.delete_model_entity(workspace, model_name)

        if not deleted:
            logger.warning(f"Model not found for deletion: {workspace}/{model_name}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model with workspace '{workspace}' and name '{model_name}' not found",
            )

        return None

    except HTTPException:
        raise
    except EntityConflictError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="Concurrent modification - please retry."
        ) from e
    except Exception:
        logger.exception("Failed to delete model entity")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete model entity")


@router.post(
    "/v2/workspaces/{workspace}/models/{model_name}/adapters",
    summary="Add Model Adapter",
    response_description="Register a new adapter to the model",
    status_code=status.HTTP_201_CREATED,
    response_model=Adapter,
)
async def create_model_adapter(
    workspace: str,
    model_name: str,
    adapter_create: CreateModelAdapterRequest,
    adapter_service: AdapterEntityService = Depends(get_adapter_entity_service),
    nhx_sdk: AsyncNeMoHelix = Depends(get_sdk_client),
) -> Adapter:
    """
    Adds an Adapter to the Model
    """
    logger.info(f"Creating model adapter entity: {workspace}/{model_name}")

    try:
        await check_fileset_access(nhx_sdk, adapter_create.fileset, workspace)
        # create the adapter using service
        created_adapter = await adapter_service.create_adapter(
            workspace,
            adapter_create,
            base_model=model_name,
        )

        if created_adapter is None:
            logger.warning(f"Model not found for update: {workspace}/{model_name}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model with workspace '{workspace}' and name '{model_name}' not found",
            )
    except EntityValidationError as e:
        logger.warning(f"Entity store validation error during adapter creation: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(e),
        )
    except PermissionError as e:
        logger.warning(f"Permission denied during adapter creation: {e}")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except FilesetValidationError as e:
        logger.warning(f"Fileset validation error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except ValueError as err:
        if "already exists" in str(err).lower():
            logger.warning(f"Model adapter exists: {workspace}/{model_name}/{adapter_create.name} - {err}")
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Model adapter already exists: {workspace}/{model_name}/{adapter_create.name}",
            )
        logger.warning(f"Model adapter creation validation error: {err}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to create model adapter - {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create model adapter")

    return created_adapter


@router.delete(
    "/v2/workspaces/{workspace}/models/{model_name}/adapters/{adapter}",
    summary="Delete Model Adapter",
    response_description="Delete model adapter by name",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_model_adapter(
    workspace: str,
    model_name: str,
    adapter: str,
    adapter_service: AdapterEntityService = Depends(get_adapter_entity_service),
):
    """
    Delete Adapter from Model entity.

    Permanently deletes an adapter from a model entity, if it was deployed, it will be cleaned up automatically.
    """
    logger.info(f"Deleting model adapter: {workspace}/{model_name}/{adapter}")

    try:
        # Delete the model using service
        deleted = await adapter_service.delete_adapter(workspace, model_name, adapter)

        if deleted == -1:
            logger.warning(f"Model not found for adapter deletion: {workspace}/{model_name}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model with workspace '{workspace}' and model name '{model_name}' not found",
            )

        if deleted == -2:
            logger.warning(f"Adapter not found for deletion: {workspace}/{model_name}/{adapter}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Adapter {adapter} with workspace '{workspace}' and model name '{model_name}' not found",
            )

        return None

    except HTTPException:
        raise
    except EntityConflictError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="Concurrent modification - please retry."
        ) from e
    except Exception as e:
        logger.exception(f"Failed to delete model entity - {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete model entity")


@router.patch(
    "/v2/workspaces/{workspace}/models/{model_name}/adapters/{adapter}",
    summary="Update Adapter",
    response_description="Update adapter metadata",
    status_code=status.HTTP_200_OK,
    response_model=Adapter,
)
async def update_model_adapter(
    workspace: str,
    model_name: str,
    adapter: str,
    adapter_update: UpdateAdapterRequest,
    adapter_service: AdapterEntityService = Depends(get_adapter_entity_service),
    nhx_sdk: AsyncNeMoHelix = Depends(get_sdk_client),
) -> Adapter:
    """
    Update Adapter deployment or description.
    """
    logger.info(f"Updating model adapter entity: {workspace}/{model_name}")

    try:
        if adapter_update.fileset:
            await check_fileset_access(nhx_sdk, adapter_update.fileset, workspace)
        # Update the model using service
        updated_adapter = await adapter_service.update_adapter(workspace, model_name, adapter, adapter_update)

        if updated_adapter == -1:
            logger.warning(f"Model not found for adapter modification: {workspace}/{model_name}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model with workspace '{workspace}' and model name '{model_name}' not found",
            )

        if updated_adapter == -2:
            logger.warning(f"Adapter not found for modification: {workspace}/{model_name}/{adapter}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Adapter {adapter} with workspace '{workspace}' and model name '{model_name}' not found",
            )

        return updated_adapter

    except EntityValidationError as e:
        logger.warning(f"Entity store validation error during adapter update: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(e),
        )
    except PermissionError as e:
        logger.warning(f"Permission denied during adapter update: {e}")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except FilesetValidationError as e:
        logger.warning(f"Fileset validation error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except ValueError as e:
        logger.warning(f"Adapter update validation error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to update model adapter entity - {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update model adapter entity"
        )


async def set_trust_remote_code(
    sdk: AsyncNeMoHelix,
    desired_trust_remote_code: bool | None,
    fileset: str | None,
    auth_client: AuthClient,
    workspace: str,
) -> bool:
    if not desired_trust_remote_code:
        return False

    is_trusted_repo = await is_trusted_repo_id(sdk, workspace, fileset)
    if is_trusted_repo:
        return True

    if await auth_client.has_permissions(workspace, ["models.trust-remote-code.set"]):
        return True

    raise PermissionError("Insufficient permissions to set the trust_remote_code field") from None
