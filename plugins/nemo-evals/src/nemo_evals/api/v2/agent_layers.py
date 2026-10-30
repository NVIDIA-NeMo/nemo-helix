# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent layers of a task revision: its main image with an agent already installed.

/apis/evals/v2/workspaces/{workspace}/tasks/{name}/revisions/{revision}/agent-layers. A layer is an image build of the
task revision, so it takes the same permissions as one.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from nemo_evals.agent_layers.models import AgentLayer, AgentLayerRequest
from nemo_evals.agent_layers.service import AgentLayerService, AgentNotLayerableError
from nemo_evals.api.dependencies import get_agent_layer_service
from nemo_evals.api.v2.image_builds import RevisionSelector, TaskImageBuildPerms, TaskName
from nemo_evals.api.v2.tasks import TaskPerms
from nemo_evals.authz import scope
from nemo_evals.image_builds.planning import ImagePlanError
from nemo_evals.image_builds.service import ImageBuildForbiddenError, ImageBuildJobConflictError
from nemo_evals.revisions import RevisionNotFoundError
from nemo_evals.task_identity import UnsupportedTaskKindError
from nemo_helix_plugin.authz import CallerKind, path_rule
from nemo_helix_plugin.client.errors import NemoHTTPError, PermissionDeniedError

logger = logging.getLogger(__name__)

_LAYERS = "/tasks/{name}/revisions/{revision}/agent-layers"

router = APIRouter()


@router.post(
    _LAYERS,
    summary="Request Task Agent Layer",
    response_description="The layer started, or the one already running or ready",
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        status.HTTP_403_FORBIDDEN: {"description": "The caller may not submit container image builds"},
        status.HTTP_404_NOT_FOUND: {"description": "Task or revision not found"},
        status.HTTP_409_CONFLICT: {"description": "The layer's Job name is held by another Job"},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"description": "The task or agent can't be layered"},
    },
)
@scope.write
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[TaskPerms.READ, TaskImageBuildPerms.CREATE])
async def request_task_agent_layer(
    workspace: str,
    name: TaskName,
    revision: RevisionSelector,
    request: AgentLayerRequest,
    service: AgentLayerService = Depends(get_agent_layer_service),
) -> AgentLayer:
    """Build the task revision's main image with an agent already installed, so the run's trials needn't install it.

    Repeating the request while a layer is running, or after it is ready, returns that layer. After a failed one, it
    starts the next attempt. Tasks whose environments hold the same files share one Builder build.
    """
    try:
        layer = await service.request_layer(workspace, name, revision, request)
    except RevisionNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    except (UnsupportedTaskKindError, ImagePlanError, AgentNotLayerableError) as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(e)) from e
    except ImageBuildForbiddenError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e)) from e
    except ImageBuildJobConflictError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e)) from e
    except PermissionDeniedError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"Could not start the layer: {e}") from e
    except NemoHTTPError as e:
        logger.warning("Jobs refused the agent layer job: %s", e)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Could not start the layer: {e}") from e
    if layer is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Task not found: {workspace}/{name}")
    return layer


@router.get(
    _LAYERS,
    summary="List Task Agent Layers",
    response_description="The revision's agent layers, newest first",
    responses={status.HTTP_404_NOT_FOUND: {"description": "Task or revision not found"}},
)
@scope.read
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[TaskPerms.READ, TaskImageBuildPerms.LIST])
async def list_task_agent_layers(
    workspace: str,
    name: TaskName,
    revision: RevisionSelector,
    service: AgentLayerService = Depends(get_agent_layer_service),
) -> list[AgentLayer]:
    """List the agent layers of a task revision, for every agent."""
    try:
        layers = await service.list_layers(workspace, name, revision)
    except RevisionNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    if layers is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Task not found: {workspace}/{name}")
    return layers
