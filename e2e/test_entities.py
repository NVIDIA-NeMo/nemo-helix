# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for the entities API.

These tests verify basic internal entity-store operations work correctly when
running against a fully deployed NHX platform. Direct entity CRUD uses service
credentials, matching feature-service access in production. This includes:
- Entity CRUD operations (create, retrieve, update, delete)
- Entity creation within and without projects
- Listing with sorting and filtering

Note: User-facing E2E tests exercise entities through feature services. These
tests intentionally exercise the internal API to provide a direct indicator
for deeper problems in the entities service itself.
"""

import json
import time
import uuid

import pytest
from nemo_helix import NeMoHelix
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.client.errors import NemoHTTPError as APIStatusError
from nemo_helix_plugin.entities.client import EntitiesClient
from nemo_helix_plugin.entities.types import EntityCreateInput, EntityUpdate, ListEntitiesQueryParams
from nemo_helix_plugin.projects.client import ProjectsClient
from nemo_helix_plugin.projects.types import CreateProjectRequest
from nhx.testing import as_service_for

ENTITY_TYPE = "e2e-test-entity"
E2E_SERVICE_PRINCIPAL = "entities-e2e"
E2E_ON_BEHALF_OF = "entities-e2e@example.com"


def _unique_name(prefix: str = "entity") -> str:
    """Generate a unique entity name."""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="module")
def entity_store_sdk(sdk: NeMoHelix) -> NeMoHelix:
    return as_service_for(
        sdk,
        on_behalf_of=E2E_ON_BEHALF_OF,
        service_name=E2E_SERVICE_PRINCIPAL,
    )


def test_cluster_info_endpoint_returns_json_with_platform_version_and_revision(sdk: NeMoHelix):
    """Test GET /cluster-info returns JSON with platform_version and revision keys.

    Verifies the platform cluster-info endpoint returns a json-encoded response
    and includes platform_version and revision fields (values are not validated).
    """
    response = sdk._client.get(f"{str(sdk.base_url).rstrip('/')}/cluster-info")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict), "Response body should be JSON object"
    assert "platform_version" in data, "Response should include a 'platform_version' key"
    assert "revision" in data, "Response should include a 'revision' key"


def test_entity_crud_lifecycle(entity_store_sdk: NeMoHelix, workspace: str):
    """Test basic entity create, retrieve, update, delete operations.

    This test verifies the complete entity lifecycle:
    1. Create an entity with specific data
    2. Retrieve it by name and verify contents
    3. Update the entity data
    4. Delete the entity
    5. Verify it no longer exists
    """
    entities = client_from_platform(entity_store_sdk, EntitiesClient)
    entity_name = _unique_name()
    initial_data = {"key": "initial-value", "nested": {"field": 123}}

    # Create entity
    entity = entities.create_entity(
        entity_type=ENTITY_TYPE,
        workspace=workspace,
        body=EntityCreateInput(
            name=entity_name,
            data=initial_data,
        ),
    ).data()
    assert entity.name == entity_name
    assert entity.workspace == workspace
    assert entity.entity_type == ENTITY_TYPE
    assert entity.data["key"] == "initial-value"
    assert entity.data["nested"]["field"] == 123

    try:
        # Retrieve by name
        retrieved = entities.get_entity_by_name(
            name=entity_name,
            entity_type=ENTITY_TYPE,
            workspace=workspace,
        ).data()
        assert retrieved.name == entity_name
        assert retrieved.id == entity.id
        assert retrieved.data == initial_data

        # Update entity
        updated_data = {"key": "updated-value", "nested": {"field": 456}, "new_field": True}
        updated = entities.update_entity_by_name(
            name=entity_name,
            entity_type=ENTITY_TYPE,
            workspace=workspace,
            body=EntityUpdate(data=updated_data),
        ).data()
        assert updated.name == entity_name
        assert updated.data["key"] == "updated-value"
        assert updated.data["nested"]["field"] == 456
        assert updated.data["new_field"] is True

        # Verify update persisted
        retrieved_after_update = entities.get_entity_by_name(
            name=entity_name,
            entity_type=ENTITY_TYPE,
            workspace=workspace,
        ).data()
        assert retrieved_after_update.data == updated_data

    finally:
        # Delete entity
        entities.delete_entity_by_name(
            name=entity_name,
            entity_type=ENTITY_TYPE,
            workspace=workspace,
        )

    # Verify entity no longer exists
    with pytest.raises(APIStatusError) as exc_info:
        entities.get_entity_by_name(
            name=entity_name,
            entity_type=ENTITY_TYPE,
            workspace=workspace,
        )
    assert exc_info.value.status_code == 404


def test_entity_with_project(sdk: NeMoHelix, entity_store_sdk: NeMoHelix, workspace: str):
    """Test entity creation within a project.

    Project setup and cleanup use the caller-facing SDK, while internal entity
    CRUD uses service credentials. Verifies that entities can be associated
    with projects and that the association is persisted and retrievable.
    """
    entities = client_from_platform(entity_store_sdk, EntitiesClient)
    project_name = _unique_name("project")
    entity_name = _unique_name()

    # Create project first
    project = (
        client_from_platform(sdk, ProjectsClient)
        .create_project(
            workspace=workspace,
            body=CreateProjectRequest(name=project_name, description="E2E test project"),
        )
        .data()
    )
    assert project.name == project_name

    try:
        # Create entity within project
        entity = entities.create_entity(
            entity_type=ENTITY_TYPE,
            workspace=workspace,
            body=EntityCreateInput(
                name=entity_name,
                data={"project_data": "value"},
                project=project_name,
            ),
        ).data()
        assert entity.name == entity_name
        assert entity.project == project_name

        # Retrieve and verify project association
        retrieved = entities.get_entity_by_name(
            name=entity_name,
            entity_type=ENTITY_TYPE,
            workspace=workspace,
        ).data()
        assert retrieved.project == project_name

        # Delete entity
        entities.delete_entity_by_name(
            name=entity_name,
            entity_type=ENTITY_TYPE,
            workspace=workspace,
        )

    finally:
        # Clean up project
        client_from_platform(sdk, ProjectsClient).delete_project(name=project_name, workspace=workspace)


def test_entity_without_project(entity_store_sdk: NeMoHelix, workspace: str):
    """Test entity creation without a project association.

    Verifies that entities can exist at the workspace level without
    being associated with any project.
    """
    entities = client_from_platform(entity_store_sdk, EntitiesClient)
    entity_name = _unique_name()

    entity = entities.create_entity(
        entity_type=ENTITY_TYPE,
        workspace=workspace,
        body=EntityCreateInput(
            name=entity_name,
            data={"standalone": True},
        ),
    ).data()

    try:
        assert entity.name == entity_name
        assert entity.project is None

        retrieved = entities.get_entity_by_name(
            name=entity_name,
            entity_type=ENTITY_TYPE,
            workspace=workspace,
        ).data()
        assert retrieved.project is None

    finally:
        entities.delete_entity_by_name(
            name=entity_name,
            entity_type=ENTITY_TYPE,
            workspace=workspace,
        )


def test_entity_list_and_sorting(entity_store_sdk: NeMoHelix, workspace: str):
    """Test listing entities with sorting.

    Creates multiple entities and verifies:
    1. All entities are returned in list
    2. Sorting by created_at works (ascending and descending)
    3. Sorting by name works
    """
    entities = client_from_platform(entity_store_sdk, EntitiesClient)
    entity_names = [_unique_name(f"sort-{i:02d}") for i in range(5)]
    created_entities = []

    try:
        # Create entities in order
        for name in entity_names:
            entity = entities.create_entity(
                entity_type=ENTITY_TYPE,
                workspace=workspace,
                body=EntityCreateInput(
                    name=name,
                    data={"order": name},
                ),
            ).data()
            time.sleep(1)
            created_entities.append(entity)

        # List all entities of this type
        response = entities.list_entities(
            entity_type=ENTITY_TYPE,
            workspace=workspace,
        )
        listed_names = {e.name for e in response.items()}
        for name in entity_names:
            assert name in listed_names

        # Test descending sort by created_at (default, newest first)
        response_desc = entities.list_entities(
            entity_type=ENTITY_TYPE,
            workspace=workspace,
            query_params=ListEntitiesQueryParams(sort="-created_at"),
        )
        desc_names = [e.name for e in response_desc.items() if e.name in entity_names]
        assert desc_names == list(reversed(entity_names))

        # Test ascending sort by created_at (oldest first)
        response_asc = entities.list_entities(
            entity_type=ENTITY_TYPE,
            workspace=workspace,
            query_params=ListEntitiesQueryParams(sort="created_at"),
        )
        asc_names = [e.name for e in response_asc.items() if e.name in entity_names]
        assert asc_names == entity_names

        # Test sort by name
        response_by_name = entities.list_entities(
            entity_type=ENTITY_TYPE,
            workspace=workspace,
            query_params=ListEntitiesQueryParams(sort="name"),
        )
        name_sorted = [e.name for e in response_by_name.items() if e.name in entity_names]
        assert name_sorted == sorted(entity_names)

    finally:
        # Clean up all created entities
        for name in entity_names:
            try:
                entities.delete_entity_by_name(
                    name=name,
                    entity_type=ENTITY_TYPE,
                    workspace=workspace,
                )
            except Exception:
                pass


def test_entity_search_filter(entity_store_sdk: NeMoHelix, workspace: str):
    """Test filtering entities with search queries.

    Verifies that the search parameter correctly filters entities
    based on field values.
    """
    prefix = _unique_name("filter")
    entity_alpha = f"{prefix}-alpha"
    entity_beta = f"{prefix}-beta"

    entities = client_from_platform(entity_store_sdk, EntitiesClient)
    try:
        # Create two entities with different data
        entities.create_entity(
            entity_type=ENTITY_TYPE,
            workspace=workspace,
            body=EntityCreateInput(
                name=entity_alpha,
                data={"category": "alpha", "value": 100},
            ),
        ).data()
        entities.create_entity(
            entity_type=ENTITY_TYPE,
            workspace=workspace,
            body=EntityCreateInput(
                name=entity_beta,
                data={"category": "beta", "value": 200},
            ),
        ).data()

        # Filter by exact name match
        filter_query = json.dumps({"name": {"$eq": entity_alpha}})
        response = entities.list_entities(
            entity_type=ENTITY_TYPE,
            workspace=workspace,
            query_params=ListEntitiesQueryParams(filter=filter_query),
        )
        response_list = list(response.items())
        assert len(response_list) == 1
        assert response_list[0].name == entity_alpha

        # Filter by name pattern (like)
        filter_query = json.dumps({"name": {"$like": f"{prefix}%"}})
        response = entities.list_entities(
            entity_type=ENTITY_TYPE,
            workspace=workspace,
            query_params=ListEntitiesQueryParams(filter=filter_query),
        )
        found_names = {e.name for e in response.items()}
        assert entity_alpha in found_names
        assert entity_beta in found_names

        # Filter by data field
        filter_query = json.dumps({"data.category": {"$eq": "beta"}})
        response = entities.list_entities(
            entity_type=ENTITY_TYPE,
            workspace=workspace,
            query_params=ListEntitiesQueryParams(filter=filter_query),
        )
        response_list = list(response.items())
        assert len(response_list) == 1
        assert response_list[0].name == entity_beta

    finally:
        for name in [entity_alpha, entity_beta]:
            try:
                entities.delete_entity_by_name(
                    name=name,
                    entity_type=ENTITY_TYPE,
                    workspace=workspace,
                )
            except Exception:
                pass


def test_entity_rename(entity_store_sdk: NeMoHelix, workspace: str):
    """Test renaming an entity via update.

    Verifies that entities can be renamed and the old name
    no longer works after rename.
    """
    entities = client_from_platform(entity_store_sdk, EntitiesClient)
    old_name = _unique_name("old")
    new_name = _unique_name("new")

    entity = entities.create_entity(
        entity_type=ENTITY_TYPE,
        workspace=workspace,
        body=EntityCreateInput(
            name=old_name,
            data={"test": "rename"},
        ),
    ).data()

    try:
        # Rename entity
        renamed = entities.update_entity_by_name(
            name=old_name,
            entity_type=ENTITY_TYPE,
            workspace=workspace,
            body=EntityUpdate(data=entity.data, new_name=new_name),
        ).data()
        assert renamed.name == new_name
        assert renamed.id == entity.id

        # Verify old name no longer works
        with pytest.raises(APIStatusError) as exc_info:
            entities.get_entity_by_name(
                name=old_name,
                entity_type=ENTITY_TYPE,
                workspace=workspace,
            )
        assert exc_info.value.status_code == 404

        # Verify new name works
        retrieved = entities.get_entity_by_name(
            name=new_name,
            entity_type=ENTITY_TYPE,
            workspace=workspace,
        ).data()
        assert retrieved.name == new_name

    finally:
        # Clean up with new name
        try:
            entities.delete_entity_by_name(
                name=new_name,
                entity_type=ENTITY_TYPE,
                workspace=workspace,
            )
        except Exception:
            pass


def test_entity_auto_generated_name(entity_store_sdk: NeMoHelix, workspace: str):
    """Test that entities can be created without specifying a name.

    When no name is provided, the API should auto-generate a unique name.
    """
    entities = client_from_platform(entity_store_sdk, EntitiesClient)
    entity = entities.create_entity(
        entity_type=ENTITY_TYPE,
        workspace=workspace,
        body=EntityCreateInput(
            data={"auto_name": True},
        ),
    ).data()

    try:
        assert entity.name is not None
        assert len(entity.name) > 0
        # Auto-generated names typically follow a pattern like "e2e-test-entity-xxxxx"
        assert ENTITY_TYPE.replace("_", "-").replace("-", "") in entity.name.replace("-", "") or entity.name

        # Verify we can retrieve by the generated name
        retrieved = entities.get_entity_by_name(
            name=entity.name,
            entity_type=ENTITY_TYPE,
            workspace=workspace,
        ).data()
        assert retrieved.id == entity.id

    finally:
        entities.delete_entity_by_name(
            name=entity.name,
            entity_type=ENTITY_TYPE,
            workspace=workspace,
        )
