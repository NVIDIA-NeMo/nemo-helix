# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock
from urllib.parse import quote

from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_distillation_plugin.models import DistillInputs, Workflow
from nemo_distillation_plugin.service import DistillationService, get_files
from nemo_distillation_plugin.template import assemble
from nemo_helix_plugin.entity_client import (
    NemoEntityConflictError,
    NemoPaginationInfo,
    get_entity_client,
)
from samples import INPUTS


def _page(items: list[Workflow]):
    page = AsyncMock()
    page.data = items
    page.pagination = NemoPaginationInfo(
        page=1, page_size=20, current_page_size=len(items), total_pages=1, total_results=len(items)
    )
    return page


def _app(entities: AsyncMock, files: AsyncMock) -> FastAPI:
    app = FastAPI()
    for spec in DistillationService().get_routers():
        app.include_router(spec.router, prefix=spec.prefix)
    app.dependency_overrides[get_entity_client] = lambda: entities
    app.dependency_overrides[get_files] = lambda: files
    return app


def _persist(workflow: Workflow) -> Workflow:
    workflow._id = "id-1"
    return workflow


def test_create_renders_the_graph_and_creates_the_fileset() -> None:
    entities = AsyncMock()
    files = AsyncMock()

    async def create(workflow: Workflow) -> Workflow:
        return _persist(workflow)

    entities.create.side_effect = create
    client = TestClient(_app(entities, files))

    response = client.post("/v2/workspaces/default/workflows", json={"name": "distill", "inputs": INPUTS})

    assert response.status_code == 201
    body = response.json()
    assert body["fileset"] == "distill-data"
    assert body["status"] == "pending"
    assert "train[lora]" in body["graph"]
    assert body["experiment"]
    assert body["stages"][0]["id"] == "prepare"
    assert body["stages"][0]["status"] == "pending"
    assert body["results"] is None
    files.create_fileset.assert_awaited()
    assert files.create_fileset.await_args.kwargs["body"].purpose.value == "dataset"


def test_create_rejects_a_name_that_cannot_hold_a_suffix() -> None:
    client = TestClient(_app(AsyncMock(), AsyncMock()))
    response = client.post("/v2/workspaces/default/workflows", json={"name": "Nope", "inputs": INPUTS})
    assert response.status_code == 422


def test_create_conflict() -> None:
    entities = AsyncMock()
    entities.create.side_effect = NemoEntityConflictError("exists")
    client = TestClient(_app(entities, AsyncMock()))
    response = client.post("/v2/workspaces/default/workflows", json={"name": "distill", "inputs": INPUTS})
    assert response.status_code == 409


def test_render_writes_nothing() -> None:
    entities = AsyncMock()
    client = TestClient(_app(entities, AsyncMock()))
    response = client.post("/v2/workspaces/default/workflow-templates/distillation:render", json=INPUTS)
    assert response.status_code == 200
    body = response.json()
    assert body["counts"]["candidates"] == 1
    assert body["counts"]["nodes"] == 21
    assert body["counts"]["gpu_hours"] == 4
    assert body["counts"]["parallel_jobs"] == 1
    assert "train[lora]" in body["graph"]
    entities.create.assert_not_called()


def test_get_lists_and_cancel() -> None:
    workflow = _persist(assemble("distill", "default", DistillInputs(**INPUTS)))
    entities = AsyncMock()
    entities.get.return_value = workflow
    entities.list.return_value = _page([workflow])
    entities.update.return_value = workflow
    client = TestClient(_app(entities, AsyncMock()))

    got = client.get("/v2/workspaces/default/workflows/distill")
    assert got.status_code == 200
    body = got.json()
    assert body["graph"]["experiment"]["depends_on"] == []
    assert body["graph"]["prepare"]["depends_on"] == ["eval_original[tasks]", "eval_original[dataset]"]

    listed = client.get("/v2/workspaces/default/workflows", params={"filter[source_agent]": "default/teacher"})
    assert listed.status_code == 200
    assert listed.json()["data"][0]["name"] == "distill"

    cancelled = client.post("/v2/workspaces/default/workflows/distill/cancel")
    assert cancelled.status_code == 202
    assert workflow.status == "cancelling"


def test_retry_reuses_external_names_and_reopens_skipped_dependents() -> None:
    workflow = _persist(assemble("distill", "default", DistillInputs(**INPUTS)))
    workflow.status = "failed"
    workflow.nodes["train[lora]"].status = "failed"
    workflow.nodes["train[lora]"].error = "boom"
    workflow.nodes["deploy[lora]"].status = "skipped"
    external = dict(workflow.nodes["train[lora]"].external)
    entities = AsyncMock()
    entities.get.return_value = workflow
    entities.update.return_value = workflow
    client = TestClient(_app(entities, AsyncMock()))

    response = client.post(f"/v2/workspaces/default/workflows/distill/nodes/{quote('train[lora]', safe='')}/retry")

    assert response.status_code == 202
    assert workflow.nodes["train[lora]"].status == "pending"
    assert workflow.nodes["train[lora]"].attempt == 2
    assert workflow.nodes["train[lora]"].external == external
    assert workflow.nodes["deploy[lora]"].status == "pending"
    assert workflow.status == "running"


def test_retry_of_a_node_that_did_not_fail_conflicts() -> None:
    workflow = _persist(assemble("distill", "default", DistillInputs(**INPUTS)))
    entities = AsyncMock()
    entities.get.return_value = workflow
    client = TestClient(_app(entities, AsyncMock()))
    response = client.post("/v2/workspaces/default/workflows/distill/nodes/prepare/retry")
    assert response.status_code == 409
