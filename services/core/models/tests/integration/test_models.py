# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for the models service API.

These tests verify basic CRUD functionality for:
- Model entities
- Model providers
- Model deployment configs
- Model deployments

Uses the create_test_client pattern for fast in-memory testing.
"""

import uuid
from unittest.mock import AsyncMock, patch

from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.client.errors import ConflictError
from nemo_platform_plugin.workspaces.client import WorkspacesClient
from nemo_platform_plugin.workspaces.types import CreateWorkspaceRequest
from nmp.core.models.config import ControllerConfig, ModelsConfig
from nmp.core.models.controllers.backends.registry import BackendRegistry
from nmp.testing import ClientContext

# Default workspace for tests
DEFAULT_WORKSPACE = "default"


def ensure_workspace_exists(test_clients: ClientContext, workspace_id: str) -> None:
    """Ensure a workspace exists, creating it if necessary.

    Args:
        test_clients: ClientContext with SDK client
        workspace_id: The workspace ID to ensure exists
    """
    try:
        client_from_platform(test_clients.sdk, WorkspacesClient).create_workspace(
            body=CreateWorkspaceRequest(name=workspace_id, description=f"Test workspace: {workspace_id}")
        ).data()
    except ConflictError:
        # Workspace already exists
        pass


# =============================================================================
# Model Entity Tests
# =============================================================================


def test_model_crud_lifecycle(test_clients: ClientContext):
    """Test full CRUD lifecycle for model entities."""
    test_name = f"test-model-{uuid.uuid4().hex[:8]}"

    # CREATE
    model_data = {
        "name": test_name,
        "description": "A large language model for testing",
        "model_providers": ["provider-a", "provider-b"],
        "spec": {
            "context_size": 4096,
            "num_virtual_tokens": 0,
            "is_chat": True,
            "checkpoint_model_name": "meta-llama/Llama-3.2-1b-instruct",
            "family": "llama",
            "num_layers": 32,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_kv_heads": 32,
            "ffn_hidden_size": 16384,
            "vocab_size": 32000,
            "tied_embeddings": True,
            "gated_mlp": True,
            "base_num_parameters": 7000000000,
            "precision": "fp16",
        },
        "custom_fields": {"domain": "general", "version": "1.0"},
    }
    response = test_clients.test_client.post(f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/models", json=model_data)
    assert response.status_code == 201, f"Create failed: {response.text}"
    created = response.json()
    assert created["name"] == test_name
    assert created["description"] == "A large language model for testing"
    assert created["model_providers"] == ["provider-a", "provider-b"]
    assert created["spec"]["base_num_parameters"] == 7000000000
    assert created["spec"]["context_size"] == 4096
    assert created["custom_fields"]["domain"] == "general"

    # READ
    response = test_clients.test_client.get(f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/models/{test_name}")
    assert response.status_code == 200, f"Get failed: {response.text}"
    fetched = response.json()
    assert fetched["name"] == created["name"]
    assert fetched["spec"]["is_chat"] is True

    # LIST
    response = test_clients.test_client.get("/apis/models/v2/workspaces/default/models")
    assert response.status_code == 200, f"List failed: {response.text}"
    models = response.json()
    assert any(m["name"] == created["name"] for m in models["data"])

    # UPDATE
    update_data = {
        "description": "Updated description for the model",
        "model_providers": ["provider-a", "provider-b", "provider-c"],
    }
    response = test_clients.test_client.patch(
        f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/models/{test_name}", json=update_data
    )
    assert response.status_code == 200, f"Update failed: {response.text}"
    updated = response.json()
    assert updated["description"] == "Updated description for the model"
    assert "provider-c" in updated["model_providers"]

    # DELETE
    response = test_clients.test_client.delete(f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/models/{test_name}")
    assert response.status_code == 204, f"Delete failed: {response.text}"

    # Verify deleted
    response = test_clients.test_client.get(f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/models/{test_name}")
    assert response.status_code == 404


def test_model_duplicate_returns_409(test_clients: ClientContext):
    """Test that creating a duplicate model entity returns 409 Conflict."""
    test_name = f"test-model-dup-{uuid.uuid4().hex[:8]}"

    # Create first model
    model_data = {
        "name": test_name,
        "description": "First model instance",
        "model_providers": ["provider-1"],
    }
    response = test_clients.test_client.post(f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/models", json=model_data)
    assert response.status_code == 201

    try:
        # Attempt to create duplicate (same name in same workspace)
        response = test_clients.test_client.post(
            f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/models", json=model_data
        )
        assert response.status_code == 409, f"Expected 409 Conflict, got {response.status_code}: {response.text}"
    finally:
        test_clients.test_client.delete(f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/models/{test_name}")


def test_model_not_found_returns_404(test_clients: ClientContext):
    """Test that retrieving a non-existent model entity returns 404."""
    response = test_clients.test_client.get(f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/models/non-existent-model")
    assert response.status_code == 404


def test_model_update_not_found_returns_404(test_clients: ClientContext):
    """Test that updating a non-existent model entity returns 404."""
    update_data = {"description": "This update should fail"}
    response = test_clients.test_client.patch(
        f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/models/non-existent-model",
        json=update_data,
    )
    assert response.status_code == 404


def test_model_delete_not_found_returns_404(test_clients: ClientContext):
    """Test that deleting a non-existent model entity returns 404."""
    response = test_clients.test_client.delete(
        f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/models/non-existent-model"
    )
    assert response.status_code == 404


def test_model_invalid_input_returns_400(test_clients: ClientContext):
    """Test that creating a model with invalid input returns 400 Bad Request."""
    # Missing required 'name' field
    model_data = {
        "description": "Model without a name",
    }
    response = test_clients.test_client.post(f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/models", json=model_data)
    assert response.status_code == 422, f"Expected 422 for missing required field, got {response.status_code}"

    # Invalid spec field type (string instead of object)
    model_data = {
        "name": f"invalid-model-{uuid.uuid4().hex[:8]}",
        "spec": "this should be an object not a string",
    }
    response = test_clients.test_client.post(f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/models", json=model_data)
    assert response.status_code == 422, f"Expected 422 for invalid spec type, got {response.status_code}"


def test_model_list_pagination(test_clients: ClientContext):
    """Test pagination for model entity list endpoint."""
    test_uuid = uuid.uuid4().hex[:8]
    created_models = []

    try:
        # Create 5 models
        for i in range(5):
            model_data = {
                "name": f"pagination-model-{test_uuid}-{i:02d}",
                "description": f"Pagination test model {i}",
                "model_providers": ["test-provider"],
            }
            response = test_clients.test_client.post(
                f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/models", json=model_data
            )
            assert response.status_code == 201, f"Failed to create model {i}: {response.text}"
            created_models.append(response.json())

        # Test page_size=2, page=1 (first page)
        response = test_clients.test_client.get(
            f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/models?page=1&page_size=2"
        )
        assert response.status_code == 200
        page1 = response.json()
        assert len(page1["data"]) == 2
        assert page1["pagination"]["page"] == 1
        assert page1["pagination"]["page_size"] == 2
        assert page1["pagination"]["total_results"] >= 5  # At least our 5 models

        # Test page_size=2, page=2 (second page)
        response = test_clients.test_client.get(
            f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/models?page=2&page_size=2"
        )
        assert response.status_code == 200
        page2 = response.json()
        assert len(page2["data"]) == 2
        assert page2["pagination"]["page"] == 2

        # Verify page 1 and page 2 have different models
        page1_names = {m["name"] for m in page1["data"]}
        page2_names = {m["name"] for m in page2["data"]}
        assert page1_names.isdisjoint(page2_names), "Pages should contain different models"

        # Test page_size=2, page=3 (third page - should have at least 1)
        response = test_clients.test_client.get(
            f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/models?page=3&page_size=2"
        )
        assert response.status_code == 200
        page3 = response.json()
        assert len(page3["data"]) >= 1

        # Test large page_size (should return all)
        response = test_clients.test_client.get(
            f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/models?page=1&page_size=100"
        )
        assert response.status_code == 200
        all_models = response.json()
        assert len(all_models["data"]) >= 5
        assert all_models["pagination"]["page"] == 1

    finally:
        # Cleanup
        for model in created_models:
            test_clients.test_client.delete(f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/models/{model['name']}")


def test_model_workspace_isolation(test_clients: ClientContext):
    """Test that models are correctly isolated by workspace.

    Verifies:
    - CRUD operations work in non-default workspaces
    - Workspace-scoped list only returns models from that workspace
    - Cross-workspace list returns models from all workspaces
    """
    test_uuid = uuid.uuid4().hex[:8]
    alt_workspace = f"test-workspace-{test_uuid}"
    model_in_default = f"model-default-{test_uuid}"
    model_in_alt = f"model-alt-{test_uuid}"

    # Create alternate workspace
    ensure_workspace_exists(test_clients, alt_workspace)

    try:
        # Create model in default workspace
        default_model_data = {
            "name": model_in_default,
            "description": "Model in default workspace",
            "model_providers": ["provider-default"],
        }
        response = test_clients.test_client.post(
            f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/models", json=default_model_data
        )
        assert response.status_code == 201, f"Create in default failed: {response.text}"
        default_model = response.json()

        # Create model in alternate workspace
        alt_model_data = {
            "name": model_in_alt,
            "description": "Model in alternate workspace",
            "model_providers": ["provider-alt"],
        }
        response = test_clients.test_client.post(
            f"/apis/models/v2/workspaces/{alt_workspace}/models", json=alt_model_data
        )
        assert response.status_code == 201, f"Create in alt workspace failed: {response.text}"
        alt_model = response.json()
        assert alt_model["workspace"] == alt_workspace

        # READ from alternate workspace
        response = test_clients.test_client.get(f"/apis/models/v2/workspaces/{alt_workspace}/models/{model_in_alt}")
        assert response.status_code == 200, f"Get from alt workspace failed: {response.text}"
        fetched = response.json()
        assert fetched["name"] == alt_model["name"]
        assert fetched["workspace"] == alt_workspace

        # Workspace-scoped LIST should only return models from that workspace
        response = test_clients.test_client.get(f"/apis/models/v2/workspaces/{alt_workspace}/models")
        assert response.status_code == 200
        alt_models = response.json()
        alt_model_names = {m["name"] for m in alt_models["data"]}
        assert alt_model["name"] in alt_model_names, "Alt model should be in alt workspace list"
        assert default_model["name"] not in alt_model_names, "Default model should NOT be in alt workspace list"

        response = test_clients.test_client.get(f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/models")
        assert response.status_code == 200
        default_models = response.json()
        default_model_names = {m["name"] for m in default_models["data"]}
        assert default_model["name"] in default_model_names, "Default model should be in default workspace list"
        assert alt_model["name"] not in default_model_names, "Alt model should NOT be in default workspace list"

        # Cross-workspace LIST (workspace="-") should return models from ALL workspaces
        response = test_clients.test_client.get("/apis/models/v2/workspaces/-/models")
        assert response.status_code == 200
        all_models = response.json()
        all_model_names = {m["name"] for m in all_models["data"]}
        assert default_model["name"] in all_model_names, "Default model should be in cross-workspace list"
        assert alt_model["name"] in all_model_names, "Alt model should be in cross-workspace list"

        # UPDATE in alternate workspace
        update_data = {"description": "Updated model in alternate workspace"}
        response = test_clients.test_client.patch(
            f"/apis/models/v2/workspaces/{alt_workspace}/models/{model_in_alt}", json=update_data
        )
        assert response.status_code == 200, f"Update in alt workspace failed: {response.text}"
        updated = response.json()
        assert updated["description"] == "Updated model in alternate workspace"

        # DELETE from alternate workspace
        response = test_clients.test_client.delete(f"/apis/models/v2/workspaces/{alt_workspace}/models/{model_in_alt}")
        assert response.status_code == 204, f"Delete from alt workspace failed: {response.text}"

        # Verify deleted
        response = test_clients.test_client.get(f"/apis/models/v2/workspaces/{alt_workspace}/models/{model_in_alt}")
        assert response.status_code == 404

    finally:
        # Cleanup
        test_clients.test_client.delete(f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/models/{model_in_default}")
        test_clients.test_client.delete(f"/apis/models/v2/workspaces/{alt_workspace}/models/{model_in_alt}")


# =============================================================================
# Model Provider Tests
# =============================================================================


def test_provider_crud_lifecycle(test_clients: ClientContext):
    """Test full CRUD lifecycle for model providers."""
    test_name = f"test-provider-{uuid.uuid4().hex[:8]}"

    # CREATE with realistic data
    provider_data = {
        "name": test_name,
        "description": "A model provider for inference",
        "host_url": "http://localhost:8080/v1",
        "enabled_models": ["llama-3-8b", "mistral-7b"],
    }
    response = test_clients.test_client.post(
        f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/providers", json=provider_data
    )
    assert response.status_code == 201, f"Create failed: {response.text}"
    created = response.json()
    assert created["name"] == test_name
    assert created["description"] == "A model provider for inference"
    assert created["host_url"] == "http://localhost:8080/v1"
    assert "llama-3-8b" in created["enabled_models"]
    assert "mistral-7b" in created["enabled_models"]
    assert "id" in created

    # READ
    response = test_clients.test_client.get(f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/providers/{test_name}")
    assert response.status_code == 200, f"Get failed: {response.text}"
    fetched = response.json()
    assert fetched["id"] == created["id"]
    assert fetched["host_url"] == "http://localhost:8080/v1"

    # LIST
    response = test_clients.test_client.get("/apis/models/v2/workspaces/default/providers")
    assert response.status_code == 200, f"List failed: {response.text}"
    providers = response.json()
    assert any(p["id"] == created["id"] for p in providers["data"])

    # UPDATE (PUT is upsert for providers)
    update_data = {
        "description": "Updated provider description",
        "host_url": "http://updated-host:9090/v1",
        "enabled_models": ["llama-3-8b", "mistral-7b", "gemma-2b"],
    }
    response = test_clients.test_client.put(
        f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/providers/{test_name}",
        json=update_data,
    )
    assert response.status_code == 200, f"Update failed: {response.text}"
    updated = response.json()
    assert updated["description"] == "Updated provider description"
    assert updated["host_url"] == "http://updated-host:9090/v1"
    assert "gemma-2b" in updated["enabled_models"]

    # DELETE
    response = test_clients.test_client.delete(f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/providers/{test_name}")
    assert response.status_code == 204, f"Delete failed: {response.text}"

    # Verify deleted
    response = test_clients.test_client.get(f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/providers/{test_name}")
    assert response.status_code == 404


def test_provider_duplicate_returns_409(test_clients: ClientContext):
    """Test that creating a duplicate provider returns 409 Conflict."""
    test_name = f"test-provider-dup-{uuid.uuid4().hex[:8]}"

    # Create first provider
    provider_data = {
        "name": test_name,
        "description": "First provider instance",
        "host_url": "http://localhost:8080",
        "enabled_models": ["model-1"],
    }
    response = test_clients.test_client.post(
        f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/providers", json=provider_data
    )
    assert response.status_code == 201

    try:
        # Attempt to create duplicate (same name in same workspace)
        response = test_clients.test_client.post(
            f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/providers", json=provider_data
        )
        assert response.status_code == 409, f"Expected 409 Conflict, got {response.status_code}: {response.text}"
    finally:
        test_clients.test_client.delete(f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/providers/{test_name}")


def test_provider_not_found_returns_404(test_clients: ClientContext):
    """Test that retrieving a non-existent provider returns 404."""
    response = test_clients.test_client.get(
        f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/providers/non-existent-provider"
    )
    assert response.status_code == 404


def test_provider_delete_not_found_returns_404(test_clients: ClientContext):
    """Test that deleting a non-existent provider returns 404."""
    response = test_clients.test_client.delete(
        f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/providers/non-existent-provider"
    )
    assert response.status_code == 404


def test_provider_invalid_input_returns_422(test_clients: ClientContext):
    """Test that creating a provider with invalid input returns 422."""
    # Missing required 'name' field
    provider_data = {
        "description": "Provider without a name",
        "host_url": "http://localhost:8080",
    }
    response = test_clients.test_client.post(
        f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/providers", json=provider_data
    )
    assert response.status_code == 422, f"Expected 422 for missing required field, got {response.status_code}"

    # Invalid field type (enabled_models should be a list, not a string)
    provider_data = {
        "name": f"invalid-provider-{uuid.uuid4().hex[:8]}",
        "host_url": "http://localhost:8080",
        "enabled_models": "should-be-a-list",
    }
    response = test_clients.test_client.post(
        f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/providers", json=provider_data
    )
    assert response.status_code == 422, f"Expected 422 for invalid enabled_models type, got {response.status_code}"


def test_provider_upsert_creates_if_not_exists(test_clients: ClientContext):
    """Test that PUT (upsert) creates a provider if it doesn't exist."""
    test_name = f"test-provider-upsert-{uuid.uuid4().hex[:8]}"

    # PUT to a non-existent provider should create it
    provider_data = {
        "description": "Created via upsert",
        "host_url": "http://upsert-host:8080",
        "enabled_models": ["upsert-model"],
    }
    response = test_clients.test_client.put(
        f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/providers/{test_name}",
        json=provider_data,
    )
    assert response.status_code == 200, f"Upsert create failed: {response.text}"
    created = response.json()
    assert created["name"] == test_name
    assert created["description"] == "Created via upsert"
    assert created["host_url"] == "http://upsert-host:8080"

    try:
        # Verify it was created by reading it back
        response = test_clients.test_client.get(f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/providers/{test_name}")
        assert response.status_code == 200
        fetched = response.json()
        assert fetched["name"] == test_name

        # PUT again should update it
        update_data = {
            "description": "Updated via upsert",
            "host_url": "http://updated-upsert-host:9090",
            "enabled_models": ["upsert-model", "new-model"],
        }
        response = test_clients.test_client.put(
            f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/providers/{test_name}",
            json=update_data,
        )
        assert response.status_code == 200, f"Upsert update failed: {response.text}"
        updated = response.json()
        assert updated["description"] == "Updated via upsert"
        assert updated["host_url"] == "http://updated-upsert-host:9090"
    finally:
        test_clients.test_client.delete(f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/providers/{test_name}")


def test_provider_list_pagination(test_clients: ClientContext):
    """Test pagination for model provider list endpoint."""
    test_uuid = uuid.uuid4().hex[:8]
    created_providers = []

    try:
        # Create 5 providers
        for i in range(5):
            provider_data = {
                "name": f"pagination-provider-{test_uuid}-{i:02d}",
                "description": f"Pagination test provider {i}",
                "host_url": f"http://provider-{i}:8080",
                "enabled_models": [f"model-{i}"],
            }
            response = test_clients.test_client.post(
                f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/providers",
                json=provider_data,
            )
            assert response.status_code == 201, f"Failed to create provider {i}: {response.text}"
            created_providers.append(response.json())

        # Test page_size=2, page=1 (first page)
        response = test_clients.test_client.get(
            f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/providers?page=1&page_size=2"
        )
        assert response.status_code == 200
        page1 = response.json()
        assert len(page1["data"]) == 2
        assert page1["pagination"]["page"] == 1
        assert page1["pagination"]["page_size"] == 2
        assert page1["pagination"]["total_results"] >= 5  # At least our 5 providers

        # Test page_size=2, page=2 (second page)
        response = test_clients.test_client.get(
            f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/providers?page=2&page_size=2"
        )
        assert response.status_code == 200
        page2 = response.json()
        assert len(page2["data"]) == 2
        assert page2["pagination"]["page"] == 2

        # Verify page 1 and page 2 have different providers
        page1_ids = {p["id"] for p in page1["data"]}
        page2_ids = {p["id"] for p in page2["data"]}
        assert page1_ids.isdisjoint(page2_ids), "Pages should contain different providers"

        # Test page_size=2, page=3 (third page - should have at least 1)
        response = test_clients.test_client.get(
            f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/providers?page=3&page_size=2"
        )
        assert response.status_code == 200
        page3 = response.json()
        assert len(page3["data"]) >= 1

        # Test large page_size (should return all)
        response = test_clients.test_client.get(
            f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/providers?page=1&page_size=100"
        )
        assert response.status_code == 200
        all_providers = response.json()
        assert len(all_providers["data"]) >= 5
        assert all_providers["pagination"]["page"] == 1

    finally:
        # Cleanup
        for provider in created_providers:
            test_clients.test_client.delete(
                f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/providers/{provider['name']}"
            )


def test_provider_workspace_isolation(test_clients: ClientContext):
    """Test that providers are correctly isolated by workspace.

    Verifies:
    - CRUD operations work in non-default workspaces
    - Workspace-scoped list only returns providers from that workspace
    - Cross-workspace list returns providers from all workspaces
    """
    test_uuid = uuid.uuid4().hex[:8]
    alt_workspace = f"test-workspace-{test_uuid}"
    provider_in_default = f"provider-default-{test_uuid}"
    provider_in_alt = f"provider-alt-{test_uuid}"

    # Create alternate workspace
    ensure_workspace_exists(test_clients, alt_workspace)

    try:
        # Create provider in default workspace
        default_provider_data = {
            "name": provider_in_default,
            "description": "Provider in default workspace",
            "host_url": "http://default-host:8080",
            "enabled_models": ["default-model"],
        }
        response = test_clients.test_client.post(
            f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/providers",
            json=default_provider_data,
        )
        assert response.status_code == 201, f"Create in default failed: {response.text}"
        default_provider = response.json()

        # Create provider in alternate workspace
        alt_provider_data = {
            "name": provider_in_alt,
            "description": "Provider in alternate workspace",
            "host_url": "http://alt-host:8080",
            "enabled_models": ["alt-model"],
        }
        response = test_clients.test_client.post(
            f"/apis/models/v2/workspaces/{alt_workspace}/providers", json=alt_provider_data
        )
        assert response.status_code == 201, f"Create in alt workspace failed: {response.text}"
        alt_provider = response.json()
        assert alt_provider["workspace"] == alt_workspace

        # READ from alternate workspace
        response = test_clients.test_client.get(
            f"/apis/models/v2/workspaces/{alt_workspace}/providers/{provider_in_alt}"
        )
        assert response.status_code == 200, f"Get from alt workspace failed: {response.text}"
        fetched = response.json()
        assert fetched["id"] == alt_provider["id"]
        assert fetched["workspace"] == alt_workspace

        # Workspace-scoped LIST should only return providers from that workspace
        response = test_clients.test_client.get(f"/apis/models/v2/workspaces/{alt_workspace}/providers")
        assert response.status_code == 200
        alt_providers = response.json()
        alt_provider_ids = {p["id"] for p in alt_providers["data"]}
        assert alt_provider["id"] in alt_provider_ids, "Alt provider should be in alt workspace list"
        assert default_provider["id"] not in alt_provider_ids, "Default provider should NOT be in alt workspace list"

        response = test_clients.test_client.get(f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/providers")
        assert response.status_code == 200
        default_providers = response.json()
        default_provider_ids = {p["id"] for p in default_providers["data"]}
        assert default_provider["id"] in default_provider_ids, "Default provider should be in default workspace list"
        assert alt_provider["id"] not in default_provider_ids, "Alt provider should NOT be in default workspace list"

        # Cross-workspace LIST (workspace="-") should return providers from ALL workspaces
        response = test_clients.test_client.get("/apis/models/v2/workspaces/-/providers")
        assert response.status_code == 200
        all_providers = response.json()
        all_provider_ids = {p["id"] for p in all_providers["data"]}
        assert default_provider["id"] in all_provider_ids, "Default provider should be in cross-workspace list"
        assert alt_provider["id"] in all_provider_ids, "Alt provider should be in cross-workspace list"

        # UPDATE in alternate workspace
        update_data = {
            "description": "Updated provider in alternate workspace",
            "host_url": "http://updated-alt-host:9090",
            "enabled_models": ["alt-model", "new-alt-model"],
        }
        response = test_clients.test_client.put(
            f"/apis/models/v2/workspaces/{alt_workspace}/providers/{provider_in_alt}",
            json=update_data,
        )
        assert response.status_code == 200, f"Update in alt workspace failed: {response.text}"
        updated = response.json()
        assert updated["description"] == "Updated provider in alternate workspace"

        # DELETE from alternate workspace
        response = test_clients.test_client.delete(
            f"/apis/models/v2/workspaces/{alt_workspace}/providers/{provider_in_alt}"
        )
        assert response.status_code == 204, f"Delete from alt workspace failed: {response.text}"

        # Verify deleted
        response = test_clients.test_client.get(
            f"/apis/models/v2/workspaces/{alt_workspace}/providers/{provider_in_alt}"
        )
        assert response.status_code == 404

    finally:
        # Cleanup
        test_clients.test_client.delete(
            f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/providers/{provider_in_default}"
        )
        test_clients.test_client.delete(f"/apis/models/v2/workspaces/{alt_workspace}/providers/{provider_in_alt}")


# =============================================================================
# Model Deployment Config Tests
# =============================================================================


def test_deployment_config_crud_lifecycle(test_clients: ClientContext):
    """Test full CRUD lifecycle for deployment configs including versioning."""
    test_name = f"test-config-{uuid.uuid4().hex[:8]}"

    # CREATE with realistic data
    config_data = {
        "name": test_name,
        "description": "A deployment configuration for LLM inference",
        "engine": "nim",
        "model_spec": {"lora_enabled": True},
        "executor_config": {
            "gpu": 2,
            "image_name": "nvcr.io/nvidia/nim",
            "image_tag": "latest",
        },
    }
    response = test_clients.test_client.post(
        f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/deployment-configs", json=config_data
    )
    assert response.status_code == 201, f"Create failed: {response.text}"
    created = response.json()
    assert created["name"] == test_name
    assert created["description"] == "A deployment configuration for LLM inference"
    assert created["executor_config"]["gpu"] == 2
    assert created["model_spec"]["lora_enabled"] is True
    assert created["entity_version"] == 1

    # READ (returns latest version)
    response = test_clients.test_client.get(
        f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/deployment-configs/{test_name}"
    )
    assert response.status_code == 200, f"Get failed: {response.text}"
    fetched = response.json()
    assert fetched["id"] == created["id"]
    assert fetched["entity_version"] == 1

    # LIST
    response = test_clients.test_client.get("/apis/models/v2/workspaces/default/deployment-configs")
    assert response.status_code == 200, f"List failed: {response.text}"
    configs = response.json()
    assert any(c["id"] == created["id"] for c in configs["data"])

    # UPDATE (creates new version via POST to /{name})
    update_data = {
        "description": "Updated deployment configuration",
        "engine": "nim",
        "model_spec": {"lora_enabled": True},
        "executor_config": {
            "gpu": 4,
            "image_name": "nvcr.io/nvidia/nim",
            "image_tag": "v2",
        },
    }
    response = test_clients.test_client.post(
        f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/deployment-configs/{test_name}",
        json=update_data,
    )
    assert response.status_code == 201, f"Update failed: {response.text}"
    updated = response.json()
    assert updated["description"] == "Updated deployment configuration"
    assert updated["executor_config"]["gpu"] == 4
    assert updated["entity_version"] == 2

    # READ should now return version 2
    response = test_clients.test_client.get(
        f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/deployment-configs/{test_name}"
    )
    assert response.status_code == 200
    latest = response.json()
    assert latest["entity_version"] == 2

    # DELETE
    response = test_clients.test_client.delete(
        f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/deployment-configs/{test_name}"
    )
    assert response.status_code == 204, f"Delete failed: {response.text}"

    # Verify deleted
    response = test_clients.test_client.get(
        f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/deployment-configs/{test_name}"
    )
    assert response.status_code == 404


def test_deployment_config_duplicate_returns_409(test_clients: ClientContext):
    """Test that creating a duplicate deployment config returns 409 Conflict."""
    test_name = f"test-config-dup-{uuid.uuid4().hex[:8]}"

    # Create first config
    config_data = {
        "name": test_name,
        "description": "First config instance",
        "engine": "nim",
        "model_spec": {},
        "executor_config": {"gpu": 1},
    }
    response = test_clients.test_client.post(
        f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/deployment-configs", json=config_data
    )
    assert response.status_code == 201

    try:
        # Attempt to create duplicate (same name in same workspace)
        response = test_clients.test_client.post(
            f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/deployment-configs", json=config_data
        )
        assert response.status_code == 409, f"Expected 409 Conflict, got {response.status_code}: {response.text}"
    finally:
        test_clients.test_client.delete(
            f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/deployment-configs/{test_name}"
        )


def test_deployment_config_not_found_returns_404(test_clients: ClientContext):
    """Test that retrieving a non-existent deployment config returns 404."""
    response = test_clients.test_client.get(
        f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/deployment-configs/non-existent-config"
    )
    assert response.status_code == 404


def test_deployment_config_delete_not_found_returns_404(test_clients: ClientContext):
    """Test that deleting a non-existent deployment config returns 404."""
    response = test_clients.test_client.delete(
        f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/deployment-configs/non-existent-config"
    )
    assert response.status_code == 404


def test_deployment_config_invalid_input_returns_422(test_clients: ClientContext):
    """Test that creating a deployment config with invalid input returns 422."""
    # Missing required 'name' field
    config_data = {
        "description": "Config without a name",
        "engine": "nim",
        "model_spec": {},
        "executor_config": {"gpu": 1},
    }
    response = test_clients.test_client.post(
        f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/deployment-configs", json=config_data
    )
    assert response.status_code == 422, f"Expected 422 for missing required field, got {response.status_code}"

    # Invalid executor_config type (string instead of object)
    config_data = {
        "name": f"invalid-config-{uuid.uuid4().hex[:8]}",
        "engine": "nim",
        "model_spec": {},
        "executor_config": "should-be-an-object",
    }
    response = test_clients.test_client.post(
        f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/deployment-configs", json=config_data
    )
    assert response.status_code == 422, f"Expected 422 for invalid executor_config type, got {response.status_code}"


def test_deployment_config_versioning(test_clients: ClientContext):
    """Test deployment config versioning - list versions and get specific version."""
    test_name = f"test-config-ver-{uuid.uuid4().hex[:8]}"

    # Create version 1
    config_data = {
        "name": test_name,
        "description": "Version 1",
        "engine": "nim",
        "model_spec": {},
        "executor_config": {"gpu": 1},
    }
    response = test_clients.test_client.post(
        f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/deployment-configs", json=config_data
    )
    assert response.status_code == 201
    v1 = response.json()
    assert v1["entity_version"] == 1

    try:
        # Create version 2
        update_data = {
            "description": "Version 2",
            "engine": "nim",
            "model_spec": {},
            "executor_config": {"gpu": 2},
        }
        response = test_clients.test_client.post(
            f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/deployment-configs/{test_name}",
            json=update_data,
        )
        assert response.status_code == 201
        v2 = response.json()
        assert v2["entity_version"] == 2

        # Create version 3
        update_data = {
            "description": "Version 3",
            "engine": "nim",
            "model_spec": {},
            "executor_config": {"gpu": 4},
        }
        response = test_clients.test_client.post(
            f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/deployment-configs/{test_name}",
            json=update_data,
        )
        assert response.status_code == 201
        v3 = response.json()
        assert v3["entity_version"] == 3

        # List all versions
        response = test_clients.test_client.get(
            f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/deployment-configs/{test_name}/versions"
        )
        assert response.status_code == 200
        versions = response.json()
        assert len(versions) == 3
        version_numbers = {v["entity_version"] for v in versions}
        assert version_numbers == {1, 2, 3}

        # Get specific version (version 1)
        response = test_clients.test_client.get(
            f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/deployment-configs/{test_name}/versions/1"
        )
        assert response.status_code == 200
        fetched_v1 = response.json()
        assert fetched_v1["entity_version"] == 1
        assert fetched_v1["description"] == "Version 1"
        assert fetched_v1["executor_config"]["gpu"] == 1

        # Get specific version (version 2)
        response = test_clients.test_client.get(
            f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/deployment-configs/{test_name}/versions/2"
        )
        assert response.status_code == 200
        fetched_v2 = response.json()
        assert fetched_v2["entity_version"] == 2
        assert fetched_v2["description"] == "Version 2"

        # Get latest (should be version 3)
        response = test_clients.test_client.get(
            f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/deployment-configs/{test_name}"
        )
        assert response.status_code == 200
        latest = response.json()
        assert latest["entity_version"] == 3
        assert latest["description"] == "Version 3"

    finally:
        test_clients.test_client.delete(
            f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/deployment-configs/{test_name}"
        )


def test_deployment_config_list_pagination(test_clients: ClientContext):
    """Test pagination for deployment config list endpoint."""
    test_uuid = uuid.uuid4().hex[:8]
    created_configs = []

    try:
        # Create 5 configs
        for i in range(5):
            config_data = {
                "name": f"pagination-config-{test_uuid}-{i:02d}",
                "description": f"Pagination test config {i}",
                "engine": "nim",
                "model_spec": {},
                "executor_config": {"gpu": i + 1},
            }
            response = test_clients.test_client.post(
                f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/deployment-configs",
                json=config_data,
            )
            assert response.status_code == 201, f"Failed to create config {i}: {response.text}"
            created_configs.append(response.json())

        # Test page_size=2, page=1 (first page)
        response = test_clients.test_client.get(
            f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/deployment-configs?page=1&page_size=2"
        )
        assert response.status_code == 200
        page1 = response.json()
        assert len(page1["data"]) == 2
        assert page1["pagination"]["page"] == 1
        assert page1["pagination"]["page_size"] == 2
        assert page1["pagination"]["total_results"] >= 5

        # Test page_size=2, page=2 (second page)
        response = test_clients.test_client.get(
            f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/deployment-configs?page=2&page_size=2"
        )
        assert response.status_code == 200
        page2 = response.json()
        assert len(page2["data"]) == 2
        assert page2["pagination"]["page"] == 2

        # Verify page 1 and page 2 have different configs
        page1_ids = {c["id"] for c in page1["data"]}
        page2_ids = {c["id"] for c in page2["data"]}
        assert page1_ids.isdisjoint(page2_ids), "Pages should contain different configs"

    finally:
        # Cleanup
        for config in created_configs:
            test_clients.test_client.delete(
                f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/deployment-configs/{config['name']}"
            )


def test_deployment_config_workspace_isolation(test_clients: ClientContext):
    """Test that deployment configs are correctly isolated by workspace.

    Verifies:
    - CRUD operations work in non-default workspaces
    - Workspace-scoped list only returns configs from that workspace
    - Cross-workspace list returns configs from all workspaces
    """
    test_uuid = uuid.uuid4().hex[:8]
    alt_workspace = f"test-workspace-{test_uuid}"
    config_in_default = f"config-default-{test_uuid}"
    config_in_alt = f"config-alt-{test_uuid}"

    # Create alternate workspace
    ensure_workspace_exists(test_clients, alt_workspace)

    try:
        # Create config in default workspace
        default_config_data = {
            "name": config_in_default,
            "description": "Config in default workspace",
            "engine": "nim",
            "model_spec": {},
            "executor_config": {"gpu": 1},
        }
        response = test_clients.test_client.post(
            f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/deployment-configs",
            json=default_config_data,
        )
        assert response.status_code == 201, f"Create in default failed: {response.text}"
        default_config = response.json()

        # Create config in alternate workspace
        alt_config_data = {
            "name": config_in_alt,
            "description": "Config in alternate workspace",
            "engine": "nim",
            "model_spec": {},
            "executor_config": {"gpu": 2},
        }
        response = test_clients.test_client.post(
            f"/apis/models/v2/workspaces/{alt_workspace}/deployment-configs",
            json=alt_config_data,
        )
        assert response.status_code == 201, f"Create in alt workspace failed: {response.text}"
        alt_config = response.json()
        assert alt_config["workspace"] == alt_workspace

        # READ from alternate workspace
        response = test_clients.test_client.get(
            f"/apis/models/v2/workspaces/{alt_workspace}/deployment-configs/{config_in_alt}"
        )
        assert response.status_code == 200, f"Get from alt workspace failed: {response.text}"
        fetched = response.json()
        assert fetched["id"] == alt_config["id"]
        assert fetched["workspace"] == alt_workspace

        # Workspace-scoped LIST should only return configs from that workspace
        response = test_clients.test_client.get(f"/apis/models/v2/workspaces/{alt_workspace}/deployment-configs")
        assert response.status_code == 200
        alt_configs = response.json()
        alt_config_ids = {c["id"] for c in alt_configs["data"]}
        assert alt_config["id"] in alt_config_ids, "Alt config should be in alt workspace list"
        assert default_config["id"] not in alt_config_ids, "Default config should NOT be in alt workspace list"

        response = test_clients.test_client.get(f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/deployment-configs")
        assert response.status_code == 200
        default_configs = response.json()
        default_config_ids = {c["id"] for c in default_configs["data"]}
        assert default_config["id"] in default_config_ids, "Default config should be in default workspace list"
        assert alt_config["id"] not in default_config_ids, "Alt config should NOT be in default workspace list"

        # Cross-workspace LIST (workspace="-") should return configs from ALL workspaces
        response = test_clients.test_client.get("/apis/models/v2/workspaces/-/deployment-configs")
        assert response.status_code == 200
        all_configs = response.json()
        all_config_ids = {c["id"] for c in all_configs["data"]}
        assert default_config["id"] in all_config_ids, "Default config should be in cross-workspace list"
        assert alt_config["id"] in all_config_ids, "Alt config should be in cross-workspace list"

        # UPDATE in alternate workspace (creates new version)
        update_data = {
            "description": "Updated config in alternate workspace",
            "engine": "nim",
            "model_spec": {},
            "executor_config": {"gpu": 4},
        }
        response = test_clients.test_client.post(
            f"/apis/models/v2/workspaces/{alt_workspace}/deployment-configs/{config_in_alt}",
            json=update_data,
        )
        assert response.status_code == 201, f"Update in alt workspace failed: {response.text}"
        updated = response.json()
        assert updated["description"] == "Updated config in alternate workspace"
        assert updated["entity_version"] == 2

        # DELETE from alternate workspace
        response = test_clients.test_client.delete(
            f"/apis/models/v2/workspaces/{alt_workspace}/deployment-configs/{config_in_alt}"
        )
        assert response.status_code == 204, f"Delete from alt workspace failed: {response.text}"

        # Verify deleted
        response = test_clients.test_client.get(
            f"/apis/models/v2/workspaces/{alt_workspace}/deployment-configs/{config_in_alt}"
        )
        assert response.status_code == 404

    finally:
        # Cleanup
        test_clients.test_client.delete(
            f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/deployment-configs/{config_in_default}"
        )
        test_clients.test_client.delete(
            f"/apis/models/v2/workspaces/{alt_workspace}/deployment-configs/{config_in_alt}"
        )


# =============================================================================
# Model Deployment Tests
# =============================================================================


def test_deployment_crud_lifecycle(test_clients: ClientContext):
    """Test full CRUD lifecycle for deployments."""
    test_uuid = uuid.uuid4().hex[:8]
    config_name = f"test-config-{test_uuid}"
    deployment_name = f"test-deployment-{test_uuid}"

    # First create a deployment config (required dependency)
    config_data = {
        "name": config_name,
        "workspace": DEFAULT_WORKSPACE,
        "engine": "nim",
        "model_spec": {},
        "executor_config": {"gpu": 1},
    }
    response = test_clients.test_client.post(
        f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/deployment-configs", json=config_data
    )
    assert response.status_code == 201, f"Config create failed: {response.text}"

    try:
        # CREATE deployment
        deployment_data = {
            "name": deployment_name,
            "workspace": DEFAULT_WORKSPACE,
            "config": config_name,
        }
        response = test_clients.test_client.post(
            f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/deployments", json=deployment_data
        )
        assert response.status_code == 201, f"Create failed: {response.text}"
        created = response.json()
        assert created["name"] == deployment_name
        assert "id" in created

        # READ
        response = test_clients.test_client.get(
            f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/deployments/{deployment_name}"
        )
        assert response.status_code == 200, f"Get failed: {response.text}"
        fetched = response.json()
        assert fetched["id"] == created["id"]

        # LIST
        response = test_clients.test_client.get("/apis/models/v2/workspaces/default/deployments")
        assert response.status_code == 200, f"List failed: {response.text}"
        deployments = response.json()
        assert any(d["id"] == created["id"] for d in deployments["data"])

        # DELETE deployment (marks as DELETING - full deletion requires controller)
        response = test_clients.test_client.delete(
            f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/deployments/{deployment_name}"
        )
        assert response.status_code in (202, 204), f"Delete failed: {response.text}"

        # Verify deployment is marked for deletion (status = DELETING)
        response = test_clients.test_client.get(
            f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/deployments/{deployment_name}"
        )
        if response.status_code == 200:
            assert response.json()["status"] == "DELETING"
    finally:
        # Cleanup: force delete by setting status to DELETED first, then delete
        # This is needed because the controller isn't running in tests
        test_clients.test_client.post(
            f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/deployments/{deployment_name}/status",
            json={"status": "DELETED"},
        )
        test_clients.test_client.delete(f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/deployments/{deployment_name}")
        test_clients.test_client.delete(
            f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/deployment-configs/{config_name}"
        )


def test_deployment_duplicate_returns_409(test_clients: ClientContext):
    """Test that creating a duplicate deployment returns 409 Conflict."""
    test_uuid = uuid.uuid4().hex[:8]
    config_name = f"test-config-dup-{test_uuid}"
    deployment_name = f"test-deployment-dup-{test_uuid}"

    # Create config first
    config_data = {
        "name": config_name,
        "workspace": DEFAULT_WORKSPACE,
        "engine": "nim",
        "model_spec": {},
        "executor_config": {"gpu": 1},
    }
    response = test_clients.test_client.post(
        f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/deployment-configs", json=config_data
    )
    assert response.status_code == 201

    try:
        # Create first deployment
        deployment_data = {
            "name": deployment_name,
            "workspace": DEFAULT_WORKSPACE,
            "config": config_name,
        }
        response = test_clients.test_client.post(
            f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/deployments", json=deployment_data
        )
        assert response.status_code == 201

        # Attempt to create duplicate
        response = test_clients.test_client.post(
            f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/deployments", json=deployment_data
        )
        assert response.status_code == 409, f"Expected 409 Conflict, got {response.status_code}: {response.text}"
    finally:
        # Cleanup
        test_clients.test_client.post(
            f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/deployments/{deployment_name}/status",
            json={"status": "DELETED"},
        )
        test_clients.test_client.delete(f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/deployments/{deployment_name}")
        test_clients.test_client.delete(
            f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/deployment-configs/{config_name}"
        )


def test_deployment_not_found_returns_404(test_clients: ClientContext):
    """Test that retrieving a non-existent deployment returns 404."""
    response = test_clients.test_client.get(
        f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/deployments/non-existent-deployment"
    )
    assert response.status_code == 404


def test_deployment_delete_not_found_returns_404(test_clients: ClientContext):
    """Test that deleting a non-existent deployment returns 404."""
    response = test_clients.test_client.delete(
        f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/deployments/non-existent-deployment"
    )
    assert response.status_code == 404


def test_deployment_invalid_input_returns_422(test_clients: ClientContext):
    """Test that invalid input returns 422 Unprocessable Entity."""
    # Missing required 'config' field
    invalid_data = {
        "name": "test-deployment-invalid",
        "workspace": DEFAULT_WORKSPACE,
        # Missing 'config' - required field
    }
    response = test_clients.test_client.post(
        f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/deployments", json=invalid_data
    )
    assert response.status_code == 422, f"Expected 422, got {response.status_code}: {response.text}"


def test_deployment_status_lifecycle(test_clients: ClientContext):
    """Test deployment status transitions: CREATED -> PENDING -> READY -> DELETING."""
    test_uuid = uuid.uuid4().hex[:8]
    config_name = f"test-config-status-{test_uuid}"
    deployment_name = f"test-deployment-status-{test_uuid}"

    # Create deployment config first
    config_data = {
        "name": config_name,
        "workspace": DEFAULT_WORKSPACE,
        "engine": "nim",
        "model_spec": {},
        "executor_config": {"gpu": 1},
    }
    response = test_clients.test_client.post(
        f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/deployment-configs", json=config_data
    )
    assert response.status_code == 201

    try:
        # Create deployment - starts in CREATED status
        deployment_data = {
            "name": deployment_name,
            "workspace": DEFAULT_WORKSPACE,
            "config": config_name,
        }
        response = test_clients.test_client.post(
            f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/deployments", json=deployment_data
        )
        assert response.status_code == 201
        created = response.json()
        assert created["status"] == "CREATED"

        # Update status to PENDING
        response = test_clients.test_client.post(
            f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/deployments/{deployment_name}/status",
            json={"status": "PENDING"},
        )
        assert response.status_code == 200

        response = test_clients.test_client.get(
            f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/deployments/{deployment_name}"
        )
        assert response.json()["status"] == "PENDING"

        # Update status to READY
        response = test_clients.test_client.post(
            f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/deployments/{deployment_name}/status",
            json={"status": "READY", "status_message": "Deployment is ready"},
        )
        assert response.status_code == 200

        response = test_clients.test_client.get(
            f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/deployments/{deployment_name}"
        )
        fetched = response.json()
        assert fetched["status"] == "READY"
        assert fetched["status_message"] == "Deployment is ready"

        # Delete triggers DELETING status
        response = test_clients.test_client.delete(
            f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/deployments/{deployment_name}"
        )
        assert response.status_code in (202, 204)

        response = test_clients.test_client.get(
            f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/deployments/{deployment_name}"
        )
        if response.status_code == 200:
            assert response.json()["status"] == "DELETING"
    finally:
        # Cleanup
        test_clients.test_client.post(
            f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/deployments/{deployment_name}/status",
            json={"status": "DELETED"},
        )
        test_clients.test_client.delete(f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/deployments/{deployment_name}")
        test_clients.test_client.delete(
            f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/deployment-configs/{config_name}"
        )


def test_deployment_list_pagination(test_clients: ClientContext):
    """Test that deployment listing supports pagination."""
    test_uuid = uuid.uuid4().hex[:8]
    config_name = f"test-config-page-{test_uuid}"
    deployment_names = [f"test-deployment-page-{test_uuid}-{i}" for i in range(5)]

    # Create deployment config first
    config_data = {
        "name": config_name,
        "workspace": DEFAULT_WORKSPACE,
        "engine": "nim",
        "model_spec": {},
        "executor_config": {"gpu": 1},
    }
    response = test_clients.test_client.post(
        f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/deployment-configs", json=config_data
    )
    assert response.status_code == 201

    try:
        # Create 5 deployments
        for name in deployment_names:
            deployment_data = {
                "name": name,
                "workspace": DEFAULT_WORKSPACE,
                "config": config_name,
            }
            response = test_clients.test_client.post(
                f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/deployments", json=deployment_data
            )
            assert response.status_code == 201, f"Failed to create {name}: {response.text}"

        # Test pagination with page_size=2
        response = test_clients.test_client.get(
            f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/deployments", params={"page_size": 2}
        )
        assert response.status_code == 200
        page1 = response.json()
        assert len(page1["data"]) == 2
        assert "pagination" in page1

        # Get second page
        response = test_clients.test_client.get(
            f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/deployments", params={"page_size": 2, "page": 2}
        )
        assert response.status_code == 200
        page2 = response.json()
        assert len(page2["data"]) == 2

        # Ensure pages have different items
        page1_ids = {d["id"] for d in page1["data"]}
        page2_ids = {d["id"] for d in page2["data"]}
        assert page1_ids.isdisjoint(page2_ids), "Pages should have different deployments"
    finally:
        # Cleanup all deployments
        for name in deployment_names:
            test_clients.test_client.post(
                f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/deployments/{name}/status",
                json={"status": "DELETED"},
            )
            test_clients.test_client.delete(f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/deployments/{name}")
        test_clients.test_client.delete(
            f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/deployment-configs/{config_name}"
        )


def test_deployment_workspace_isolation(test_clients: ClientContext):
    """Test that deployments are properly isolated by workspace."""
    test_uuid = uuid.uuid4().hex[:8]
    workspace1 = "workspace1"
    workspace2 = "workspace2"
    config_name = f"test-config-ws-{test_uuid}"
    deployment_name = f"test-deployment-ws-{test_uuid}"

    # Ensure workspaces exist
    ensure_workspace_exists(test_clients, workspace1)
    ensure_workspace_exists(test_clients, workspace2)

    # Create config in workspace1
    config_data = {
        "name": config_name,
        "workspace": workspace1,
        "engine": "nim",
        "model_spec": {},
        "executor_config": {"gpu": 1},
    }
    response = test_clients.test_client.post(
        f"/apis/models/v2/workspaces/{workspace1}/deployment-configs", json=config_data
    )
    assert response.status_code == 201

    try:
        # Create deployment in workspace1
        deployment_data = {
            "name": deployment_name,
            "workspace": workspace1,
            "config": config_name,
        }
        response = test_clients.test_client.post(
            f"/apis/models/v2/workspaces/{workspace1}/deployments", json=deployment_data
        )
        assert response.status_code == 201
        created = response.json()

        # List deployments in workspace1 - should find it
        response = test_clients.test_client.get(f"/apis/models/v2/workspaces/{workspace1}/deployments")
        assert response.status_code == 200
        ws1_deployments = response.json()
        assert any(d["id"] == created["id"] for d in ws1_deployments["data"])

        # List deployments in workspace2 - should NOT find it
        response = test_clients.test_client.get(f"/apis/models/v2/workspaces/{workspace2}/deployments")
        assert response.status_code == 200
        ws2_deployments = response.json()
        assert not any(d["id"] == created["id"] for d in ws2_deployments["data"])

        # Cross-workspace LIST (workspace="-") should return deployments from ALL workspaces
        response = test_clients.test_client.get("/apis/models/v2/workspaces/-/deployments")
        assert response.status_code == 200
        all_deployments = response.json()
        assert any(d["id"] == created["id"] for d in all_deployments["data"]), (
            "Deployment should be in cross-workspace list"
        )

        # GET from wrong workspace should return 404
        response = test_clients.test_client.get(
            f"/apis/models/v2/workspaces/{workspace2}/deployments/{deployment_name}"
        )
        assert response.status_code == 404
    finally:
        # Cleanup
        test_clients.test_client.post(
            f"/apis/models/v2/workspaces/{workspace1}/deployments/{deployment_name}/status",
            json={"status": "DELETED"},
        )
        test_clients.test_client.delete(f"/apis/models/v2/workspaces/{workspace1}/deployments/{deployment_name}")
        test_clients.test_client.delete(f"/apis/models/v2/workspaces/{workspace1}/deployment-configs/{config_name}")


# =============================================================================
# Backend Config Key Validation Tests
# =============================================================================
# These tests verify that backend config keys in ModelsConfig match the
# BackendRegistry keys. This catches naming mismatches like "k8s" vs
# "nim_operator" that would cause service startup failures.


def test_backend_config_key_deployments_plugin_works_end_to_end():
    """Verify deployments_plugin key in ModelsConfig works with BackendRegistry."""
    from nmp.core.models.controllers.backends.deployments_plugin.config import DeploymentsPluginBackendConfigModel

    config = ModelsConfig(
        controller=ControllerConfig(
            backends={"deployments_plugin": DeploymentsPluginBackendConfigModel(enabled=True)},
        )
    )
    backend_configs: dict[str, DeploymentsPluginBackendConfigModel] = {
        "deployments_plugin": config.controller.backends["deployments_plugin"],
    }

    with patch(
        "nmp.core.models.controllers.backends.deployments_plugin.backend.NemoEntitiesClient",
    ):
        registry = BackendRegistry.from_config(
            nmp_sdk=AsyncMock(),
            backend_configs=backend_configs,
            huggingface_model_puller=config.huggingface_model_puller,
        )

    assert registry.list_backends() == ["deployments_plugin"]
