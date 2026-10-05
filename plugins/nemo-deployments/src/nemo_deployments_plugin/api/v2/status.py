# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Controller-only status update routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from nemo_deployments_plugin.api.v2._perms import DeploymentPerms, VolumePerms
from nemo_deployments_plugin.api.v2.dependencies import get_entity_client, require_service_principal
from nemo_deployments_plugin.authz import scope
from nemo_deployments_plugin.entities import Deployment, Volume
from nemo_deployments_plugin.schema import UpdateDeploymentStatusRequest, UpdateVolumeStatusRequest
from nemo_helix_plugin.authz import CallerKind, path_rule
from nemo_helix_plugin.entity_client import NemoEntitiesClient, NemoEntityConflictError, NemoEntityNotFoundError

router = APIRouter()


@router.put("/deployments/{name}/status", response_model=Deployment, tags=["Deployment Status"])
@scope.write
@path_rule(callers=[CallerKind.SERVICE_PRINCIPAL], permissions=[DeploymentPerms.STATUS_UPDATE])
async def update_deployment_status(
    workspace: str,
    name: str,
    body: UpdateDeploymentStatusRequest,
    _: None = Depends(require_service_principal),
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> Deployment:
    try:
        deployment = await entity_client.get(Deployment, name=name, workspace=workspace)
    except NemoEntityNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Deployment '{name}' not found in workspace '{workspace}'.",
        ) from exc

    deployment.status = body.status
    deployment.status_message = body.status_message
    deployment.exit_code = body.exit_code
    deployment.error_details = body.error_details

    try:
        return await entity_client.update(deployment)
    except NemoEntityConflictError as exc:
        raise HTTPException(status_code=409, detail="Concurrent modification.") from exc
    except NemoEntityNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Deployment '{name}' not found in workspace '{workspace}'.",
        ) from exc


@router.put("/volumes/{name}/status", response_model=Volume, tags=["Volume Status"])
@scope.write
@path_rule(callers=[CallerKind.SERVICE_PRINCIPAL], permissions=[VolumePerms.STATUS_UPDATE])
async def update_volume_status(
    workspace: str,
    name: str,
    body: UpdateVolumeStatusRequest,
    _: None = Depends(require_service_principal),
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> Volume:
    try:
        volume = await entity_client.get(Volume, name=name, workspace=workspace)
    except NemoEntityNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Volume '{name}' not found in workspace '{workspace}'.",
        ) from exc

    volume.status = body.status
    volume.status_message = body.status_message
    volume.error_details = body.error_details

    try:
        return await entity_client.update(volume)
    except NemoEntityConflictError as exc:
        raise HTTPException(status_code=409, detail="Concurrent modification.") from exc
    except NemoEntityNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Volume '{name}' not found in workspace '{workspace}'.",
        ) from exc
