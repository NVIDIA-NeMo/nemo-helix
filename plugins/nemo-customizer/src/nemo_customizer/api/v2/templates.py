# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Saved customization job templates.

A template is a named job input, stored so it can be replayed into a new job — the same
artifact cloning a job produces. The entity store's generic routes are service-and-admin
only, so these routes hold the entities on the caller's behalf.

``config`` is stored as given. The submitter-facing schema belongs to whichever backend
plugin is installed, and the router does not import them, so the config is checked when
it is submitted rather than when it is saved.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from nemo_customizer.api.v2.schemas import (
    CreateCustomizationJobTemplateRequest,
    CustomizationJobTemplatePage,
    UpdateCustomizationJobTemplateRequest,
)
from nemo_customizer.entities import CustomizationJobTemplate
from nemo_platform_plugin.authz import AuthzScope, CallerKind, PermissionSet, path_rule, perm
from nemo_platform_plugin.entity_client import (
    NemoEntitiesClient,
    NemoEntityConflictError,
    NemoEntityNotFoundError,
    get_entity_client,
)
from nemo_platform_plugin.log_utils import sanitize_for_log
from nemo_platform_plugin.schema import PaginationData

router = APIRouter()

logger = logging.getLogger(__name__)

scope = AuthzScope("customization")


class CustomizationJobTemplatePerms(PermissionSet, namespace="customization.job-templates"):
    """Referenced in ``@path_rule``; the platform derives the catalog from the routes."""

    CREATE = perm("Create customization job templates")
    LIST = perm("List customization job templates")
    READ = perm("Read a customization job template")
    UPDATE = perm("Update a customization job template")
    DELETE = perm("Delete a customization job template")


async def _get_template_or_404(
    entity_client: NemoEntitiesClient, workspace: str, name: str
) -> CustomizationJobTemplate:
    try:
        return await entity_client.get(CustomizationJobTemplate, name=name, workspace=workspace)
    except NemoEntityNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Job template '{name}' not found in workspace '{workspace}'.",
        ) from exc


@router.post(
    "/job-templates", response_model=CustomizationJobTemplate, status_code=201, tags=["Customization Job Templates"]
)
@scope.write
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[CustomizationJobTemplatePerms.CREATE])
async def create_job_template(
    workspace: str,
    body: CreateCustomizationJobTemplateRequest,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> CustomizationJobTemplate:
    """Save a job input for reuse."""
    template = CustomizationJobTemplate(
        name=body.name,
        workspace=workspace,
        backend=body.backend,
        config=body.config,
        description=body.description,
    )
    try:
        return await entity_client.create(template)
    except NemoEntityConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail=f"Job template '{body.name}' already exists in workspace '{workspace}'.",
        ) from exc
    except Exception as exc:
        logger.exception("Failed to create job template '%s'", sanitize_for_log(body.name))
        raise HTTPException(status_code=500, detail="Failed to create job template.") from exc


@router.get("/job-templates", response_model=CustomizationJobTemplatePage, tags=["Customization Job Templates"])
@scope.read
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[CustomizationJobTemplatePerms.LIST])
async def list_job_templates(
    workspace: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    sort: str = Query(default="-created_at"),
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> CustomizationJobTemplatePage:
    """List saved job templates in the workspace."""
    try:
        result = await entity_client.list(
            CustomizationJobTemplate,
            workspace=workspace,
            page=page,
            page_size=page_size,
            sort=sort,
        )
    except Exception as exc:
        logger.exception("Failed to list job templates in workspace '%s'", sanitize_for_log(workspace))
        raise HTTPException(status_code=500, detail="Failed to list job templates.") from exc
    pagination = PaginationData.model_validate(result.pagination.model_dump()) if result.pagination else None
    return CustomizationJobTemplatePage(data=result.data, pagination=pagination, sort=sort)


@router.get("/job-templates/{name}", response_model=CustomizationJobTemplate, tags=["Customization Job Templates"])
@scope.read
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[CustomizationJobTemplatePerms.READ])
async def get_job_template(
    workspace: str,
    name: str,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> CustomizationJobTemplate:
    """Get a single job template."""
    return await _get_template_or_404(entity_client, workspace, name)


@router.patch("/job-templates/{name}", response_model=CustomizationJobTemplate, tags=["Customization Job Templates"])
@scope.write
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[CustomizationJobTemplatePerms.UPDATE])
async def update_job_template(
    workspace: str,
    name: str,
    body: UpdateCustomizationJobTemplateRequest,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> CustomizationJobTemplate:
    """Edit a job template's config, target backend or description."""
    existing = await _get_template_or_404(entity_client, workspace, name)
    if body.description is not None:
        existing.description = body.description
    if body.backend is not None:
        existing.backend = body.backend
    if body.config is not None:
        existing.config = body.config
    try:
        return await entity_client.update(existing)
    except NemoEntityNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Job template '{name}' not found in workspace '{workspace}'.",
        ) from exc
    except NemoEntityConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Job template '{name}' was modified by another request in workspace '{workspace}'. "
                "Refresh the template and try again."
            ),
        ) from exc
    except Exception as exc:
        logger.exception("Failed to update job template '%s'", sanitize_for_log(name))
        raise HTTPException(status_code=500, detail="Failed to update job template.") from exc


@router.delete("/job-templates/{name}", status_code=204, tags=["Customization Job Templates"])
@scope.write
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[CustomizationJobTemplatePerms.DELETE])
async def delete_job_template(
    workspace: str,
    name: str,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> None:
    """Delete a job template."""
    await _get_template_or_404(entity_client, workspace, name)
    try:
        await entity_client.delete(CustomizationJobTemplate, name=name, workspace=workspace)
    except NemoEntityNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Job template '{name}' not found in workspace '{workspace}'.",
        ) from exc
    except Exception as exc:
        logger.exception("Failed to delete job template '%s'", sanitize_for_log(name))
        raise HTTPException(status_code=500, detail="Failed to delete job template.") from exc
