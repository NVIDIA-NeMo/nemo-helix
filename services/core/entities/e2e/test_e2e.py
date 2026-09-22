# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
from fastapi import status
from nemo_platform import NeMoPlatform
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.client.errors import NemoHTTPError as APIStatusError
from nemo_platform_plugin.entities.client import EntitiesClient
from nemo_platform_plugin.entities.types import EntityCreateInput, EntityUpdate
from nemo_platform_plugin.workspaces.client import WorkspacesClient
from nemo_platform_plugin.workspaces.types import CreateWorkspaceRequest
from nmp.core.entities.utils.identifiers import generate_entity_id

base_url = os.getenv("BASE_URL", "http://localhost:8080")
sdk = NeMoPlatform(base_url=base_url, max_retries=0)


@pytest.fixture(scope="module")
def workspace():
    workspace = (
        client_from_platform(sdk, WorkspacesClient)
        .create_workspace(body=CreateWorkspaceRequest(name=generate_entity_id("workspace")))
        .data()
    )
    yield workspace
    client_from_platform(sdk, WorkspacesClient).delete_workspace(name=workspace.name).data()


def test_crud_entity(workspace):
    entities_client = client_from_platform(sdk, EntitiesClient)

    entity = entities_client.create_entity(
        entity_type="test-type",
        workspace=workspace.name,
        body=EntityCreateInput(name="test-entity", data={"key": "value"}),
    ).data()
    entity_id = entity.id

    entity = entities_client.get_entity_by_id(entity_id=entity_id).data()
    assert entity.entity_type == "test-type"
    assert entity.id.startswith("test-type-")
    assert entity.name == "test-entity"
    assert entity.data["key"] == "value"
    assert entity.workspace == workspace.name

    entity = entities_client.update_entity_by_name(
        entity_type="test-type",
        workspace=workspace.name,
        name="test-entity",
        body=EntityUpdate(data={"key": "new-value"}),
    ).data()
    assert entity.data["key"] == "new-value"

    entities_client.delete_entity_by_name(
        entity_type="test-type",
        workspace=workspace.name,
        name="test-entity",
    ).data()
    with pytest.raises(APIStatusError) as e:
        entities_client.get_entity_by_id(entity_id=entity_id).data()
    assert isinstance(e.value, APIStatusError)
    assert e.value.status_code == status.HTTP_404_NOT_FOUND


def test_list_entities(workspace):
    entities_client = client_from_platform(sdk, EntitiesClient)
    for i in range(10):
        entities_client.create_entity(
            entity_type="test-type",
            workspace=workspace.name,
            body=EntityCreateInput(name=f"test-entity-{i}", data={"key": f"value-{i}"}),
        ).data()

    response = entities_client.list_entities(workspace=workspace.name, entity_type="test-type")
    entities = response.page().items
    assert len(entities) == 10
    for entity in entities:
        assert entity.name.startswith("test-entity-")
        assert entity.data["key"] == f"value-{entity.name.split('-')[-1]}"
        assert entity.entity_type == "test-type"
        assert entity.workspace == workspace.name

    response = entities_client.list_entities(
        workspace=workspace.name,
        entity_type="test-type",
        query_params={"sort": "-created_at"},
    )
    entities = response.page().items
    assert len(entities) == 10
    assert entities[0].name == "test-entity-9"
    assert entities[9].name == "test-entity-0"
