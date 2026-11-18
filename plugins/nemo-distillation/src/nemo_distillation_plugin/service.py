# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Distillation workflow routes, mounted at ``/apis/distillation``."""

from __future__ import annotations

import uuid
from typing import ClassVar

from fastapi import APIRouter, Depends, HTTPException
from nemo_helix_plugin.api.filters import make_filter_obj_dep
from nemo_helix_plugin.api.pagination import make_page_dep
from nemo_helix_plugin.authz import CallerKind, path_rule
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.dependencies import get_nemo_client
from nemo_helix_plugin.entity_client import (
    NemoEntitiesClient,
    NemoEntityConflictError,
    NemoEntityNotFoundError,
    get_entity_client,
)
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest, FilesetPurpose
from nemo_helix_plugin.schema import PaginationData
from nemo_helix_plugin.service import NemoService, RouterSpec
from pydantic import BaseModel, ConfigDict, Field

from nemo_distillation_plugin._perms import WorkflowPerms
from nemo_distillation_plugin.authz import scope
from nemo_distillation_plugin.models import DistillInputs, GraphError, GraphNode, Workflow
from nemo_distillation_plugin.projection import descendants, project
from nemo_distillation_plugin.template import assemble, render


class CreateWorkflowRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    inputs: DistillInputs


class WorkflowFilter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    template: str | None = None
    source_agent: str | None = None
    status: str | None = None


class RenderCounts(BaseModel):
    nodes: int
    candidates: int
    passing_trials: int
    estimated_tokens: int
    gpu_hours: int


class RenderResponse(BaseModel):
    graph: dict[str, GraphNode]
    counts: RenderCounts


class WorkflowPage(BaseModel):
    data: list[Workflow] = Field(default_factory=list)
    pagination: PaginationData


async def get_files(client: AsyncNemoClient = Depends(get_nemo_client)) -> AsyncFilesClient:
    return AsyncFilesClient.from_client(client)


class DistillationService(NemoService):
    """Workflow CRUD, cancel, retry, and the distillation template render."""

    name: ClassVar[str] = "distillation"
    dependencies: ClassVar[list[str]] = ["entities", "files", "auth"]

    def get_routers(self) -> list[RouterSpec]:
        return [
            RouterSpec(
                _router(),
                tag="Distillation Workflows",
                description="Distillation workflow entity, template render, cancel, and retry.",
                prefix="/v2/workspaces/{workspace}",
            )
        ]


def _router() -> APIRouter:
    router = APIRouter()
    page_dep = make_page_dep()
    filter_dep = make_filter_obj_dep(WorkflowFilter)

    @router.post("/workflows", response_model=Workflow, status_code=201)
    @scope.write
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[WorkflowPerms.CREATE])
    async def create_workflow(
        workspace: str,
        body: CreateWorkflowRequest,
        entity_client: NemoEntitiesClient = Depends(get_entity_client),
        files: AsyncFilesClient = Depends(get_files),
    ) -> Workflow:
        name = body.name or f"distill-{uuid.uuid4().hex[:8]}"
        try:
            workflow = assemble(name, workspace, body.inputs)
        except GraphError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        await files.create_fileset(
            workspace=workspace,
            body=CreateFilesetRequest(
                name=workflow.fileset,
                purpose=FilesetPurpose.DATASET,
                description=f"Files for distillation workflow {workflow.name}",
            ),
            exist_ok=True,
        )
        try:
            saved = await entity_client.create(workflow)
        except NemoEntityConflictError as exc:
            raise HTTPException(status_code=409, detail=f"Workflow '{name}' already exists.") from exc
        project(saved)
        return saved

    @router.get("/workflows", response_model=WorkflowPage)
    @scope.read
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[WorkflowPerms.LIST])
    async def list_workflows(
        workspace: str,
        entity_client: NemoEntitiesClient = Depends(get_entity_client),
        page=Depends(page_dep),
        filter_obj: dict | None = Depends(filter_dep),
    ) -> WorkflowPage:
        result = await entity_client.list(
            Workflow,
            workspace=workspace,
            filter_obj=filter_obj or None,
            page=page.page,
            page_size=page.page_size,
            sort="created_at",
        )
        for workflow in result.data:
            project(workflow)
        info = result.pagination
        return WorkflowPage(
            data=result.data,
            pagination=PaginationData(
                page=info.page,
                page_size=info.page_size,
                current_page_size=info.current_page_size,
                total_pages=info.total_pages,
                total_results=info.total_results,
            ),
        )

    @router.get("/workflows/{name}", response_model=Workflow)
    @scope.read
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[WorkflowPerms.READ])
    async def get_workflow(
        workspace: str,
        name: str,
        entity_client: NemoEntitiesClient = Depends(get_entity_client),
    ) -> Workflow:
        workflow = await _load(entity_client, workspace, name)
        project(workflow)
        return workflow

    @router.post("/workflows/{name}/cancel", response_model=Workflow, status_code=202)
    @scope.write
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[WorkflowPerms.CANCEL])
    async def cancel_workflow(
        workspace: str,
        name: str,
        entity_client: NemoEntitiesClient = Depends(get_entity_client),
    ) -> Workflow:
        workflow = await _load(entity_client, workspace, name)
        if workflow.status in {"completed", "cancelled"}:
            raise HTTPException(status_code=409, detail=f"Workflow '{name}' is {workflow.status}.")
        workflow.status = "cancelling"
        saved = await _save(entity_client, workflow)
        project(saved)
        return saved

    @router.post("/workflows/{name}/nodes/{node}/retry", response_model=Workflow, status_code=202)
    @scope.write
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[WorkflowPerms.RETRY])
    async def retry_node(
        workspace: str,
        name: str,
        node: str,
        entity_client: NemoEntitiesClient = Depends(get_entity_client),
    ) -> Workflow:
        workflow = await _load(entity_client, workspace, name)
        runtime = workflow.nodes.get(node)
        if runtime is None:
            raise HTTPException(status_code=404, detail=f"Node '{node}' was not found on workflow '{name}'.")
        if runtime.status != "failed":
            raise HTTPException(
                status_code=409, detail=f"Node '{node}' is {runtime.status}; only a failed node can be retried."
            )
        _reset(runtime, increment=True)
        for child in descendants(workflow.graph, node):
            downstream = workflow.nodes[child]
            if downstream.status in {"skipped", "failed", "cancelled"}:
                _reset(downstream, increment=False)
        if workflow.status in {"failed", "cancelled"}:
            workflow.status = "running"
        project(workflow)
        saved = await _save(entity_client, workflow)
        return saved

    @router.post("/workflow-templates/distillation:render", response_model=RenderResponse)
    @scope.read
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[WorkflowPerms.RENDER])
    async def render_template(workspace: str, body: DistillInputs) -> RenderResponse:
        del workspace
        try:
            rendered = render(body)
        except GraphError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return RenderResponse(graph=rendered["graph"], counts=RenderCounts(**rendered["counts"]))

    return router


def _reset(runtime, *, increment: bool) -> None:
    runtime.status = "pending"
    runtime.error = None
    runtime.outputs = {}
    runtime.confirm_since = None
    if increment:
        runtime.attempt += 1


async def _load(entity_client: NemoEntitiesClient, workspace: str, name: str) -> Workflow:
    try:
        return await entity_client.get(Workflow, name=name, workspace=workspace)
    except NemoEntityNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Workflow '{name}' was not found.") from exc


async def _save(entity_client: NemoEntitiesClient, workflow: Workflow) -> Workflow:
    try:
        return await entity_client.update(workflow)
    except NemoEntityConflictError as exc:
        raise HTTPException(status_code=409, detail=f"Workflow '{workflow.name}' was updated concurrently.") from exc
