# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Routes for agent pipeline configs under ``/apis/agents/v2/workspaces/{workspace}/pipeline-configs``.

Create and update run the static checks only; ``POST .../validate`` also checks agents in the workspace.
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from contextlib import contextmanager
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from nemo_agents_plugin.api.v2._perms import PipelineConfigPerms
from nemo_agents_plugin.api.v2.dependencies import get_entity_client
from nemo_agents_plugin.api.v2.environments import _delete_entity, _get_entity, _list_entities
from nemo_agents_plugin.authz import scope
from nemo_agents_plugin.entities import AgentPipelineConfig
from nemo_agents_plugin.pipeline_validation import check_pipeline_config
from nemo_agents_plugin.schema import (
    CreatePipelineConfigRequest,
    PipelineConfigFilter,
    PipelineConfigPage,
    UpdatePipelineConfigRequest,
)
from nemo_helix_plugin.agents.pipelines import PipelineInline, PipelineValidationResult, ValidatePipelineConfigRequest
from nemo_helix_plugin.api.filters import make_filter_obj_dep
from nemo_helix_plugin.api.pagination import PageParams, make_page_dep
from nemo_helix_plugin.authz import CallerKind, path_rule
from nemo_helix_plugin.entities import CLIENT_ERRORS
from nemo_helix_plugin.entities.base import ID_PATTERN
from nemo_helix_plugin.entity_client import (
    NemoEntitiesClient,
    NemoEntityConflictError,
    NemoEntityNotFoundError,
    NemoEntityValidationError,
)

logger = logging.getLogger(__name__)

router = APIRouter()

_KIND = "pipeline config"
_TAG = "Agent Pipeline Configs"
_pipeline_config_filter_dep = make_filter_obj_dep(PipelineConfigFilter)


@router.post("/pipeline-configs", response_model=AgentPipelineConfig, status_code=201, tags=[_TAG])
@scope.write
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[PipelineConfigPerms.CREATE])
async def create_pipeline_config(
    workspace: Annotated[str, Path(pattern=ID_PATTERN)],
    body: CreatePipelineConfigRequest,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> AgentPipelineConfig:
    """Create a pipeline config from a complete pipeline.yaml. Agent references are not resolved here."""
    pipeline = AgentPipelineConfig.model_validate({**body.model_dump(), "workspace": workspace})
    with _entity_store_errors(body.name, "create", conflict=f"already exists in workspace '{workspace}'"):
        return await entity_client.create(pipeline)


@router.post("/pipeline-configs/validate", response_model=PipelineValidationResult, tags=[_TAG])
@scope.write
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[PipelineConfigPerms.CREATE])
async def validate_pipeline_config(
    workspace: Annotated[str, Path(pattern=ID_PATTERN)],
    body: ValidatePipelineConfigRequest,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> PipelineValidationResult:
    """Check a pipeline.yaml without saving it: static checks, then its agents and context in this workspace.

    Problems with the config, including agents that are missing or cannot be read, are reported in
    ``issues`` with status 200.
    """
    return await check_pipeline_config(body.root, workspace=workspace, entity_client=entity_client)


@router.get("/pipeline-configs", response_model=PipelineConfigPage, tags=[_TAG])
@scope.read
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[PipelineConfigPerms.LIST])
async def list_pipeline_configs(
    workspace: str,
    page_params: PageParams = Depends(make_page_dep(default_size=20, max_size=100)),
    sort: str = Query(default="-created_at"),
    filter: PipelineConfigFilter = Depends(_pipeline_config_filter_dep),
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> PipelineConfigPage:
    """List pipeline configs in the workspace."""
    return await _list_entities(
        entity_client,
        AgentPipelineConfig,
        PipelineConfigPage,
        workspace=workspace,
        page=page_params.page,
        page_size=page_params.page_size,
        sort=sort,
        filter=filter,
        kind="pipeline configs",
    )


@router.get("/pipeline-configs/{name}", response_model=AgentPipelineConfig, tags=[_TAG])
@scope.read
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[PipelineConfigPerms.READ])
async def get_pipeline_config(
    workspace: str,
    name: str,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> AgentPipelineConfig:
    """Get a pipeline config by name."""
    return await _get_entity(entity_client, AgentPipelineConfig, name=name, workspace=workspace, kind=_KIND)


@router.put("/pipeline-configs/{name}", response_model=AgentPipelineConfig, tags=[_TAG])
@scope.write
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[PipelineConfigPerms.UPDATE])
async def update_pipeline_config(
    workspace: str,
    name: str,
    body: UpdatePipelineConfigRequest,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> AgentPipelineConfig:
    """Replace a pipeline config. The name and workspace cannot change; no revisions are kept."""
    if body.name is not None and body.name != name:
        raise HTTPException(
            status_code=422,
            detail=f"Body name '{body.name}' does not match pipeline config '{name}'; renaming is not supported.",
        )
    existing = await _get_entity(entity_client, AgentPipelineConfig, name=name, workspace=workspace, kind=_KIND)
    for field in PipelineInline.model_fields:
        setattr(existing, field, getattr(body, field))
    with _entity_store_errors(name, "update", conflict="was modified by another request; retry"):
        return await entity_client.update(existing)


@router.delete("/pipeline-configs/{name}", status_code=204, tags=[_TAG])
@scope.write
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[PipelineConfigPerms.DELETE])
async def delete_pipeline_config(
    workspace: str,
    name: str,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> None:
    """Delete a pipeline config by name."""
    await _delete_entity(entity_client, AgentPipelineConfig, name=name, workspace=workspace, kind=_KIND)


@contextmanager
def _entity_store_errors(name: str, action: str, *, conflict: str) -> Generator[None]:
    """Map entity-store errors to HTTP errors without returning exception text."""
    try:
        yield
    except NemoEntityNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Pipeline config '{name}' not found.") from exc
    except NemoEntityConflictError as exc:
        raise HTTPException(status_code=409, detail=f"Pipeline config '{name}' {conflict}.") from exc
    except NemoEntityValidationError as exc:
        logger.warning("Entity store rejected pipeline config '%s': %s", name, exc)
        raise HTTPException(status_code=422, detail="The entity store rejected the pipeline config.") from exc
    except CLIENT_ERRORS:
        raise
    except Exception as exc:
        logger.exception("Failed to %s pipeline config '%s'", action, name)
        raise HTTPException(status_code=500, detail=f"Failed to {action} pipeline config.") from exc
