# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for Models Controller.

Backend-agnostic tests use a mock backend (no Docker required). Full-stack
docker coverage lives in test_deployments_plugin_lifecycle.py.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock

import pytest
from nemo_platform_plugin.client.errors import NotFoundError
from nemo_platform_plugin.models.types import ModelDeploymentStatus
from nmp.core.models.controllers.backends.backends import DeploymentStatusUpdate

from .conftest import (
    create_deployment,
    create_deployment_config,
    delete_deployment,
    get_deployment,
    get_provider,
)

# =============================================================================
# Backend-Agnostic Tests (Mock Backend)
# =============================================================================


def test_controller_initializes_correctly(controller_with_mock_backend):
    """Test that controller initializes with expected state."""
    controller, _, _ = controller_with_mock_backend

    assert controller is not None
    assert controller._backend_registry is not None
    assert not controller.is_healthy  # Not healthy until first step completes


def test_controller_step_marks_healthy(controller_with_mock_backend):
    """Test that controller step marks itself healthy on success."""
    controller, _, _ = controller_with_mock_backend

    # Run one controller step
    controller.step()

    assert controller.is_healthy


def test_controller_reconciles_created_deployment(controller_with_mock_backend):
    """Test that controller calls backend.create_model_deployment for CREATED deployments."""
    controller, mock_backend, models_client = controller_with_mock_backend
    test_uuid = uuid.uuid4().hex[:8]
    config_name = f"test-config-{test_uuid}"
    deployment_name = f"test-deployment-{test_uuid}"

    # Configure mock backend to keep this specific deployment in PENDING state
    # (otherwise the reconciler processes PENDING->READY in the same step)
    mock_backend.status_responses[deployment_name] = DeploymentStatusUpdate(
        status=ModelDeploymentStatus.PENDING,
        status_message="Still starting",
        host_url="http://localhost:8500",
    )

    # Create deployment config first
    create_deployment_config(
        models_client,
        name=config_name,
        workspace="default",
        engine="nim",
        model_spec={},
        executor_config={"gpu": 0},  # No GPU for mock
    )

    # Create deployment - starts in CREATED status
    create_deployment(
        models_client,
        name=deployment_name,
        workspace="default",
        config=config_name,
    )

    # Verify deployment is in CREATED status
    deployment = get_deployment(models_client, name=deployment_name, workspace="default")
    assert deployment.status == "CREATED"

    # Run controller step - should call backend.create_model_deployment
    controller.step()

    # Verify backend was called
    assert len(mock_backend.create_calls) >= 1
    # Find our deployment in the calls
    our_calls = [(d, c, e) for d, c, e in mock_backend.create_calls if d.name == deployment_name]
    assert len(our_calls) == 1
    called_deployment, called_config, _ = our_calls[0]
    assert called_deployment.name == deployment_name
    assert called_config.name == config_name

    # Verify deployment status was updated to PENDING
    deployment = get_deployment(models_client, name=deployment_name, workspace="default")
    assert deployment.status == "PENDING"


def test_controller_polls_pending_deployment(controller_with_mock_backend):
    """Test that controller calls backend.get_model_deployment_status for PENDING deployments."""
    controller, mock_backend, models_client = controller_with_mock_backend
    test_uuid = uuid.uuid4().hex[:8]
    config_name = f"test-config-poll-{test_uuid}"
    deployment_name = f"test-deployment-poll-{test_uuid}"

    # Create config and deployment
    create_deployment_config(
        models_client, name=config_name, workspace="default", engine="nim", model_spec={}, executor_config={"gpu": 0}
    )
    create_deployment(models_client, name=deployment_name, workspace="default", config=config_name)

    # Configure status to PENDING so first step doesn't immediately go to READY
    mock_backend.status_responses[deployment_name] = DeploymentStatusUpdate(
        status=ModelDeploymentStatus.PENDING,
        status_message="Still starting",
        host_url="http://localhost:8500",
    )

    # Run first step to move from CREATED -> PENDING
    controller.step()

    # Clear call history
    mock_backend.create_calls.clear()
    mock_backend.status_calls.clear()

    # Configure status response to return READY for this specific deployment
    mock_backend.status_responses[deployment_name] = DeploymentStatusUpdate(
        status=ModelDeploymentStatus.READY,
        status_message="Container ready",
        host_url="http://localhost:8500",
    )

    # Run second step - should call get_model_deployment_status
    controller.step()

    # Verify status was checked for our deployment
    deployment_status_calls = [d for d in mock_backend.status_calls if d.name == deployment_name]
    assert len(deployment_status_calls) == 1

    # Verify deployment was updated to READY
    deployment = get_deployment(models_client, name=deployment_name, workspace="default")
    assert deployment.status == "READY"


def test_controller_handles_backend_error(controller_with_mock_backend):
    """Test that controller handles backend errors gracefully."""
    controller, mock_backend, models_client = controller_with_mock_backend
    test_uuid = uuid.uuid4().hex[:8]
    config_name = f"test-config-err-{test_uuid}"
    deployment_name = f"test-deployment-err-{test_uuid}"

    # Create config and deployment
    create_deployment_config(
        models_client, name=config_name, workspace="default", engine="nim", model_spec={}, executor_config={"gpu": 0}
    )
    create_deployment(models_client, name=deployment_name, workspace="default", config=config_name)

    # Configure backend to return ERROR
    mock_backend.create_response = DeploymentStatusUpdate(
        status=ModelDeploymentStatus.ERROR,
        status_message="Failed to create container",
        error_details={"error": "Image not found"},
    )

    # Run controller step
    controller.step()

    # Verify deployment status was updated to ERROR
    deployment = get_deployment(models_client, name=deployment_name, workspace="default")
    assert deployment.status == "ERROR"
    assert "Failed to create container" in (deployment.status_message or "")


def test_controller_deletes_when_deleting(controller_with_mock_backend):
    """Test that controller calls backend.delete_model_deployment for DELETING deployments."""
    controller, mock_backend, models_client = controller_with_mock_backend
    test_uuid = uuid.uuid4().hex[:8]
    config_name = f"test-config-del-{test_uuid}"
    deployment_name = f"test-deployment-del-{test_uuid}"

    # Create config and deployment
    create_deployment_config(
        models_client, name=config_name, workspace="default", engine="nim", model_spec={}, executor_config={"gpu": 0}
    )
    create_deployment(models_client, name=deployment_name, workspace="default", config=config_name)

    # Move to READY state
    mock_backend.create_response = DeploymentStatusUpdate(
        status=ModelDeploymentStatus.PENDING, status_message="Starting"
    )
    controller.step()
    mock_backend.status_responses[deployment_name] = DeploymentStatusUpdate(
        status=ModelDeploymentStatus.READY, status_message="Ready"
    )
    controller.step()

    # Delete the deployment (moves to DELETING)
    delete_deployment(models_client, name=deployment_name, workspace="default")

    # Clear call history
    mock_backend.delete_calls.clear()

    # Configure delete response
    mock_backend.delete_response = DeploymentStatusUpdate(
        status=ModelDeploymentStatus.DELETED,
        status_message="Container deleted",
    )

    # Run controller step - should call delete
    controller.step()

    # Verify delete was called for our deployment
    our_delete_calls = [c for c in mock_backend.delete_calls if c[0] == "default" and c[1] == deployment_name]
    assert len(our_delete_calls) == 1


def test_controller_garbage_collects_deleted_deployment(controller_with_mock_backend):
    """Test that controller hard-deletes DELETED deployments after grace period expires."""
    controller, mock_backend, models_client = controller_with_mock_backend
    test_uuid = uuid.uuid4().hex[:8]
    config_name = f"test-config-gc-{test_uuid}"
    deployment_name = f"test-deployment-gc-{test_uuid}"

    # Create config and deployment
    create_deployment_config(
        models_client, name=config_name, workspace="default", engine="nim", model_spec={}, executor_config={"gpu": 0}
    )
    create_deployment(models_client, name=deployment_name, workspace="default", config=config_name)

    # Progress through lifecycle: CREATED → PENDING → READY → DELETING → DELETED
    mock_backend.create_response = DeploymentStatusUpdate(
        status=ModelDeploymentStatus.PENDING, status_message="Starting"
    )
    controller.step()

    mock_backend.status_responses[deployment_name] = DeploymentStatusUpdate(
        status=ModelDeploymentStatus.READY, status_message="Ready", host_url="http://localhost:8080"
    )
    controller.step()

    # Delete deployment (moves to DELETING)
    delete_deployment(models_client, name=deployment_name, workspace="default")

    mock_backend.delete_response = DeploymentStatusUpdate(
        status=ModelDeploymentStatus.DELETED, status_message="Deleted"
    )
    controller.step()  # DELETING → DELETED

    # Verify deployment is in DELETED state (soft-deleted, still exists)
    deployment = get_deployment(models_client, name=deployment_name, workspace="default")
    assert deployment.status == "DELETED"

    # Patch the controller's reconciler to have 0 second grace period
    controller._deployment_reconciler._controller_config.model_deployment_garbage_collection_ttl_seconds = 0

    # Run controller step - should hard-delete since grace period expired
    controller.step()

    # Verify deployment is gone (hard-deleted)
    with pytest.raises(NotFoundError):
        get_deployment(models_client, name=deployment_name, workspace="default")


def test_controller_orphan_cleanup_after_deleted(controller_with_mock_backend):
    """Test that orphan cleanup deletes backend resources when deployment is DELETED and gone from API.

    Flow: create deployment → PENDING → READY → delete via API → DELETED → hard-delete (grace=0)
    → then simulate backend still reporting the deployment (orphan) → next step runs reconcile_orphans
    and calls delete_model_deployment(workspace, name) for the orphan.
    """
    controller, mock_backend, models_client = controller_with_mock_backend
    test_uuid = uuid.uuid4().hex[:8]
    config_name = f"test-config-orphan-{test_uuid}"
    deployment_name = f"test-deployment-orphan-{test_uuid}"

    # Create config and deployment
    create_deployment_config(
        models_client, name=config_name, workspace="default", engine="nim", model_spec={}, executor_config={"gpu": 0}
    )
    create_deployment(models_client, name=deployment_name, workspace="default", config=config_name)

    # CREATED → PENDING → READY
    mock_backend.create_response = DeploymentStatusUpdate(
        status=ModelDeploymentStatus.PENDING, status_message="Starting"
    )
    controller.step()

    mock_backend.status_responses[deployment_name] = DeploymentStatusUpdate(
        status=ModelDeploymentStatus.READY, status_message="Ready", host_url="http://localhost:8080"
    )
    controller.step()

    # Delete via API (moves to DELETING)
    delete_deployment(models_client, name=deployment_name, workspace="default")

    mock_backend.delete_response = DeploymentStatusUpdate(
        status=ModelDeploymentStatus.DELETED, status_message="Deleted"
    )
    controller.step()  # DELETING → DELETED

    deployment = get_deployment(models_client, name=deployment_name, workspace="default")
    assert deployment.status == "DELETED"

    # Hard-delete after grace period so deployment is no longer in API
    controller._deployment_reconciler._controller_config.model_deployment_garbage_collection_ttl_seconds = 0
    controller.step()

    with pytest.raises(NotFoundError):
        get_deployment(models_client, name=deployment_name, workspace="default")

    # Simulate backend still reporting this deployment (orphan)
    deployment_id = f"default/{deployment_name}"
    mock_backend.list_managed_deployment_names = AsyncMock(return_value=[deployment_id])
    mock_backend.delete_calls.clear()

    # Next step: reconcile_orphans sees backend has deployment_id but it's not in known set → delete orphan
    controller.step()

    # Orphan cleanup should have called delete_model_deployment("default", deployment_name)
    our_delete_calls = [c for c in mock_backend.delete_calls if c[0] == "default" and c[1] == deployment_name]
    assert len(our_delete_calls) == 1, (
        f"Expected one delete call for orphan {deployment_id}, got delete_calls={mock_backend.delete_calls}"
    )


def test_controller_creates_model_provider_when_ready(controller_with_mock_backend):
    """Test that controller creates ModelProvider when deployment becomes READY."""
    controller, mock_backend, models_client = controller_with_mock_backend
    test_uuid = uuid.uuid4().hex[:8]
    config_name = f"test-config-prov-{test_uuid}"
    deployment_name = f"test-deployment-prov-{test_uuid}"

    # Create config and deployment
    create_deployment_config(
        models_client, name=config_name, workspace="default", engine="nim", model_spec={}, executor_config={"gpu": 0}
    )
    create_deployment(models_client, name=deployment_name, workspace="default", config=config_name)

    # Move to READY state with host_url - this should trigger provider creation
    mock_backend.create_response = DeploymentStatusUpdate(
        status=ModelDeploymentStatus.READY, status_message="Ready", host_url="http://localhost:9000"
    )
    controller.step()

    # Verify deployment has model_provider_id set
    deployment = get_deployment(models_client, name=deployment_name, workspace="default")
    assert deployment.status == "READY"
    assert deployment.model_provider_id is not None

    # Verify provider was created with correct host_url and status
    provider_id = deployment.model_provider_id
    provider_workspace, provider_name = provider_id.split("/")
    provider = get_provider(models_client, name=provider_name, workspace=provider_workspace)
    assert provider.host_url == "http://localhost:9000"
    assert provider.status == "READY", "Provider should be READY when deployment is READY"


def test_controller_deletes_model_provider_on_delete(controller_with_mock_backend):
    """Test that controller deletes ModelProvider when deployment is deleted."""
    controller, mock_backend, models_client = controller_with_mock_backend
    test_uuid = uuid.uuid4().hex[:8]
    config_name = f"test-config-delprov-{test_uuid}"
    deployment_name = f"test-deployment-delprov-{test_uuid}"

    # Create config and deployment
    create_deployment_config(
        models_client, name=config_name, workspace="default", engine="nim", model_spec={}, executor_config={"gpu": 0}
    )
    create_deployment(models_client, name=deployment_name, workspace="default", config=config_name)

    # Move to READY state (creates provider)
    mock_backend.create_response = DeploymentStatusUpdate(
        status=ModelDeploymentStatus.READY, status_message="Ready", host_url="http://localhost:9001"
    )
    controller.step()

    # Get provider info before deletion
    deployment = get_deployment(models_client, name=deployment_name, workspace="default")
    provider_id = deployment.model_provider_id
    assert provider_id is not None
    provider_workspace, provider_name = provider_id.split("/")

    # Verify provider exists
    get_provider(models_client, name=provider_name, workspace=provider_workspace)

    # Delete deployment (moves to DELETING)
    delete_deployment(models_client, name=deployment_name, workspace="default")

    # Configure delete response and run controller
    mock_backend.delete_response = DeploymentStatusUpdate(
        status=ModelDeploymentStatus.DELETED, status_message="Deleted"
    )
    controller.step()

    # Verify provider was deleted
    with pytest.raises(NotFoundError):
        get_provider(models_client, name=provider_name, workspace=provider_workspace)
