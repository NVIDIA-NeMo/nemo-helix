# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guardrails config endpoints."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from nemo_helix_plugin.inference_middleware_models import VirtualModel, guardrail_config_membership_filter
from nhx.common.api import ParsedFilter, make_filter_dep
from nhx.common.api.common import DeleteResponse, GenericSortField, Page, PaginationData
from nhx.common.api.utils import generate_openapi_extra_params
from nhx.common.entities import EntityClient, EntityConflictError, EntityNotFoundError
from nhx.common.service.dependencies import get_entity_client
from nhx.guardrails.api.dependencies import ConfigRegistryDep, RailsRegistryDep
from nhx.guardrails.api.v2.configs.schemas import GuardrailConfigFilter, GuardrailConfigInput, GuardrailConfigUpdate
from nhx.guardrails.app.utils.config_utils import enrich_config_with_data, invalidate_and_reload_config_cache
from nhx.guardrails.entities import GuardrailConfig

router = APIRouter()

logger = logging.getLogger(__name__)


@router.get(
    "/v2/workspaces/{workspace}/configs",
    response_model=Page[GuardrailConfig],
    response_model_exclude_none=True,
    openapi_extra=generate_openapi_extra_params(
        filter_schema=GuardrailConfigFilter,
        filter_description=("Filter guardrail configs by name, description, project, created_at, and updated_at."),
    ),
)
async def list_guardrail_configs(
    workspace: str,
    entities_client: EntityClient = Depends(get_entity_client),
    page: int = Query(default=1, description="Page number."),
    page_size: int = Query(default=10, description="Page size."),
    sort: GenericSortField = Query(
        default="created_at",
        description="""The field to sort by. To sort in decreasing order, use `-` in front of the field name.""",
    ),
    parsed: ParsedFilter = Depends(make_filter_dep(GuardrailConfigFilter)),
) -> Page[GuardrailConfig]:
    """List available guardrail configs.

    Lists guardrail configs for a specific workspace.
    """
    res = await entities_client.list(
        GuardrailConfig,
        page=page,
        page_size=page_size,
        sort=sort.value,
        workspace=workspace,
        filter_operation=parsed.operation,
    )

    enriched_configs = [enrich_config_with_data(config) for config in res.data]

    return Page(
        data=enriched_configs,
        pagination=PaginationData(
            page=res.pagination.page,
            page_size=res.pagination.page_size,
            current_page_size=len(enriched_configs),
            total_pages=res.pagination.total_pages,
            total_results=res.pagination.total_results,
        ),
        sort=sort,
    )


@router.post(
    "/v2/workspaces/{workspace}/configs",
    status_code=201,
    responses={
        201: {"description": "Config created successfully."},
        422: {"description": "Validation Error"},
        409: {"description": "Config already exists."},
    },
    response_model=GuardrailConfig,
)
async def create_config(
    workspace: str,
    config: GuardrailConfigInput,
    entities_client: EntityClient = Depends(get_entity_client),
) -> GuardrailConfig:
    """Create a new guardrail config."""
    # Check if config already exists
    try:
        existing = await entities_client.get(GuardrailConfig, name=config.name, workspace=workspace)
        if existing:
            raise HTTPException(status_code=409, detail=f"Config '{config.name}' already exists.")
    except EntityNotFoundError:
        pass  # Config doesn't exist, we can create it

    entity = GuardrailConfig(
        name=config.name,
        workspace=workspace,
        description=config.description,
        data=config.data,
    )

    try:
        return await entities_client.create(entity)
    except EntityConflictError:
        raise HTTPException(status_code=409, detail=f"Config '{config.name}' already exists.")


@router.get(
    "/v2/workspaces/{workspace}/configs/{name}",
    responses={
        200: {"description": "Successful Response"},
        404: {"description": "Config does not exist."},
        422: {"description": "Validation Error"},
    },
    response_model=GuardrailConfig,
    response_model_exclude_none=True,
)
async def get_guardrail_config(
    workspace: str,
    name: str,
    entities_client: EntityClient = Depends(get_entity_client),
) -> GuardrailConfig:
    """Get info about a guardrail configuration."""
    try:
        config = await entities_client.get(GuardrailConfig, name=name, workspace=workspace)
    except EntityNotFoundError:
        raise HTTPException(status_code=404, detail="Guardrail config not found.")

    return enrich_config_with_data(config)


@router.patch(
    "/v2/workspaces/{workspace}/configs/{name}",
    responses={
        200: {"description": "Successful Response"},
        404: {"description": "Config does not exist."},
        422: {"description": "Validation Error"},
    },
    response_model=GuardrailConfig,
)
async def update_config(
    workspace: str,
    name: str,
    config_data: GuardrailConfigUpdate,
    entities_client: EntityClient = Depends(get_entity_client),
    rails_registry: RailsRegistryDep = None,
    config_registry: ConfigRegistryDep = None,
) -> GuardrailConfig:
    """
    Update model metadata. If the request body has an empty field,
    keep the old value.
    """
    try:
        existing_config = await entities_client.get(GuardrailConfig, name=name, workspace=workspace)
    except EntityNotFoundError:
        raise HTTPException(status_code=404, detail="Guardrail config not found.")

    # Apply updates
    diff = config_data.model_dump(include=config_data.model_fields_set, exclude_unset=True)
    config_to_update = existing_config.model_copy(update=diff)

    updated_config = await entities_client.update(config_to_update)

    full_config_id = f"{workspace}/{existing_config.name}" if workspace else existing_config.name

    if config_registry:
        await config_registry.refresh_all()

    if rails_registry:
        invalidate_and_reload_config_cache(
            rails_registry=rails_registry,
            config_id=full_config_id,
        )

    return updated_config


# A guardrail config is referenced by fully qualified "workspace/name", so a VirtualModel in any
# workspace can apply one from another — the in-use scan is deliberately cross-workspace ("-").
_IN_USE_SCAN_PAGE_SIZE = 200
_IN_USE_NAMES_IN_ERROR = 5


async def _virtual_models_using_config(entities_client: EntityClient, config_ref: str) -> list[str]:
    """VirtualModels that apply guardrail config ``config_ref``, as ``"workspace/name"`` refs.

    Two passes, because neither alone is both cheap and correct.  The entity-store filter narrows
    to candidates in SQL, but its legacy arm matches serialized middleware as raw text and so can
    over-match (see :func:`guardrail_config_membership_filter`).  Each candidate is therefore
    re-checked against its parsed middleware calls, which compares the ``config_type``
    discriminator as well as the reference.  The result is exact: a config is never held hostage by
    a VirtualModel that merely happens to contain the same string.

    Collection stops once there are enough names for the error message — the caller only needs to
    know *whether* the config is in use, plus a few examples.  A config with no references is the
    only case that scans every candidate page, and that set is small by construction.
    """
    in_use: list[str] = []
    page = 1
    while True:
        result = await entities_client.list(
            VirtualModel,
            workspace="-",
            page=page,
            page_size=_IN_USE_SCAN_PAGE_SIZE,
            filter_operation=guardrail_config_membership_filter(config_ref),
        )
        for vm in result.data:
            if vm.references_guardrail_config(config_ref):
                in_use.append(f"{vm.workspace}/{vm.name}")
                if len(in_use) >= _IN_USE_NAMES_IN_ERROR:
                    return in_use
        if page >= result.pagination.total_pages or not result.data:
            return in_use
        page += 1


@router.delete(
    "/v2/workspaces/{workspace}/configs/{name}",
    responses={
        200: {"description": "Successful model deletion."},
        404: {"description": "Config does not exist."},
        409: {"description": "Config is applied by one or more VirtualModels and cannot be deleted."},
        422: {"description": ("Unable to delete config due to validation error while processing request.")},
    },
)
async def delete_config(
    workspace: str,
    name: str,
    entities_client: EntityClient = Depends(get_entity_client),
    rails_registry: RailsRegistryDep = None,
    config_registry: ConfigRegistryDep = None,
) -> DeleteResponse:
    """Delete a guardrail config."""
    try:
        config = await entities_client.get(GuardrailConfig, name=name, workspace=workspace)
    except EntityNotFoundError:
        raise HTTPException(status_code=404, detail="Guardrail config not found.")

    full_config_id = f"{workspace}/{config.name}" if workspace else config.name

    # Deleting a config out from under a VirtualModel that applies it would leave that route
    # pointing at a config the Inference Gateway can no longer resolve — the same dangling
    # reference the VirtualModel endpoints reject at attach time. Refuse it here too, so the
    # reference is validated symmetrically on both sides.
    in_use_by = await _virtual_models_using_config(entities_client, full_config_id)
    if in_use_by:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Guardrail config '{full_config_id}' is applied by "
                f"{', '.join(sorted(in_use_by))}. Detach it from those virtual models before deleting it."
            ),
        )

    try:
        await entities_client.delete(GuardrailConfig, config.name, workspace=workspace)
    except EntityNotFoundError:
        raise HTTPException(status_code=404, detail="Guardrail config not found.")
    except EntityConflictError as exc:
        raise HTTPException(status_code=409, detail="Concurrent modification - please retry.") from exc

    if config_registry:
        await config_registry.refresh_all()

    if rails_registry:
        invalidate_and_reload_config_cache(
            rails_registry=rails_registry,
            config_id=full_config_id,
        )

    return DeleteResponse(id=config.id)
