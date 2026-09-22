# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
from fastapi import status
from nemo_helix import NeMoHelix
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.client.errors import NemoHTTPError as APIStatusError
from nemo_helix_plugin.workspaces.client import WorkspacesClient
from nemo_helix_plugin.workspaces.types import CreateWorkspaceRequest
from nhx.core.entities.utils.identifiers import generate_entity_id

base_url = os.getenv("BASE_URL", "http://localhost:8080")
sdk = NeMoHelix(base_url=base_url, max_retries=0)


@pytest.fixture(scope="module")
def workspace():
    workspace = (
        client_from_platform(sdk.v2, WorkspacesClient)
        .create_workspace(body=CreateWorkspaceRequest(name=generate_entity_id("workspace")))
        .data()
    )
    yield workspace
    client_from_platform(sdk.v2, WorkspacesClient).delete_workspace(name=workspace.name).data()


def test_crud_entity(workspace):
    entity = sdk.v2.entities.create(
        name="test-entity",
        data={"key": "value"},
        entity_type="test-type",
        workspace=workspace.name,
    )
    entity_id = entity.id

    entity = sdk.v2.entities.retrieve(entity_id=entity_id)
    assert entity.entity_type == "test-type"
    assert entity.id.startswith("test-type-")
    assert entity.name == "test-entity"
    assert entity.data["key"] == "value"
    assert entity.workspace == workspace.name

    entity = sdk.v2.entities.update(entity_id=entity_id, data={"key": "new-value"})
    assert entity.data["key"] == "new-value"

    sdk.v2.entities.delete(entity_id=entity_id)
    with pytest.raises(APIStatusError) as e:
        sdk.v2.entities.retrieve(entity_id=entity_id)
    assert isinstance(e.value, APIStatusError)
    assert e.value.status_code == status.HTTP_404_NOT_FOUND


def test_list_entities(workspace):
    for i in range(10):
        sdk.v2.entities.create(
            name=f"test-entity-{i}",
            data={"key": f"value-{i}"},
            entity_type="test-type",
            workspace=workspace.name,
        )

    response = sdk.entities.list(workspace=workspace.name, entity_type="test-type")
    assert len(response.data) == 10
    for entity in response.data:
        assert entity.name.startswith("test-entity-")
        assert entity.data["key"] == f"value-{entity.name.split('-')[-1]}"
        assert entity.entity_type == "test-type"
        assert entity.workspace == workspace.name

    response = sdk.entities.list(workspace=workspace.name, entity_type="test-type", sort="created_at")
    assert len(response.data) == 10
    assert response.data[0].name == "test-entity-9"
    assert response.data[9].name == "test-entity-0"
