# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ModelDeploymentReconciler."""

import logging
from datetime import datetime, timedelta, timezone
from enum import Enum
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nemo_platform import AsyncNeMoPlatform
from nmp.core.models.config import ControllerConfig
from nmp.core.models.controllers.backends.backends import DeploymentStatusUpdate
from nmp.core.models.controllers.backends.registry import BackendRegistry
from nmp.core.models.controllers.context import ModelContext
from nmp.core.models.controllers.deployment_reconciler import ModelDeploymentReconciler
from nmp.core.models.controllers.entity_cache import ModelEntityCache
from nmp.core.models.schemas import ModelDeployment

from .conftest import (
    _ModelResponse,
    _status_error,
    make_async_models_client,
    make_entity,
    seed_entity_cache,
)


def _entity(workspace, name, model_providers):
    """Model Entity stand-in addressable by the cache."""
    return make_entity(workspace, name, model_providers=model_providers)


def _enum_value(value: Enum | str) -> str:
    return value.value if isinstance(value, Enum) else value


def _deployment_status_call(update_status: AsyncMock, index: int = -1) -> dict[str, str | None]:
    call = update_status.call_args_list[index]
    body = call.kwargs["body"]
    query_params = call.kwargs["query_params"]
    return {
        "name": call.kwargs["name"],
        "workspace": call.kwargs["workspace"],
        "version": query_params.get("version"),
        "status": _enum_value(body.status),
        "status_message": body.status_message,
        "model_provider_id": body.model_provider_id,
    }


def _request_body_call(method: AsyncMock, index: int = -1) -> dict[str, object]:
    call = method.call_args_list[index]
    values: dict[str, object] = {}
    if "workspace" in call.kwargs:
        values["workspace"] = call.kwargs["workspace"]
    if "name" in call.kwargs:
        values["name"] = call.kwargs["name"]
    values.update(call.kwargs["body"].model_dump(exclude_unset=True, mode="json"))
    return values


@pytest.fixture
def mock_models_sdk():
    """Create a mock AsyncNeMoPlatform SDK."""
    sdk = MagicMock(spec=AsyncNeMoPlatform)
    sdk.models_client = make_async_models_client()
    return sdk


@pytest.fixture(autouse=True)
def _patch_entity_cache_client_from_platform(mock_models_sdk):
    """Route ``client_from_platform(sdk, AsyncModelsClient)`` in the entity cache
    back to the mock typed client on ``mock_models_sdk.models_client``."""
    with (
        patch(
            "nmp.core.models.controllers.entity_cache.client_from_platform",
            side_effect=lambda sdk, cls: sdk.models_client,
        ),
        patch(
            "nmp.core.models.controllers.deployment_reconciler.client_from_platform",
            side_effect=lambda sdk, cls: sdk.models_client,
        ),
    ):
        yield


@pytest.fixture
def mock_backend_registry():
    """Create a mock BackendRegistry."""
    return MagicMock(spec=BackendRegistry)


@pytest.fixture
def controller_config():
    """Create a ControllerConfig instance."""
    return ControllerConfig()


@pytest.fixture
def entity_cache(mock_models_sdk):
    """Model Entity cache backed by the mock SDK."""
    return ModelEntityCache(models_sdk=mock_models_sdk, emit_heartbeat=lambda: None)


@pytest.fixture
def heartbeat_calls():
    """Collects heartbeat emissions so tests can assert progress was reported."""
    return []


@pytest.fixture
def reconciler(mock_models_sdk, mock_backend_registry, controller_config, entity_cache, heartbeat_calls):
    """Create a ModelDeploymentReconciler instance."""
    return ModelDeploymentReconciler(
        models_sdk=mock_models_sdk,
        backend_registry=mock_backend_registry,
        controller_config=controller_config,
        entity_cache=entity_cache,
        emit_heartbeat=lambda: heartbeat_calls.append(1),
    )


@pytest.fixture
def make_deployment():
    """Factory fixture for creating mock deployments with common defaults."""

    def _make(
        *,
        workspace="default",
        name="test-deployment",
        entity_version="v1",
        status="CREATED",
        spec=None,
        **kwargs,
    ):
        deployment = MagicMock(spec=spec) if spec else MagicMock()
        deployment.workspace = workspace
        deployment.name = name
        deployment.entity_version = entity_version
        deployment.status = status
        for key, value in kwargs.items():
            setattr(deployment, key, value)
        return deployment

    return _make


# ============================================================================
# Individual Deployment Reconciliation Tests
# ============================================================================


@pytest.mark.asyncio
async def test_handle_created_deployment_success(reconciler, mock_backend_registry, make_deployment):
    """Test handling a CREATED deployment successfully."""
    deployment = make_deployment(status="CREATED")

    # Mock backend to return PENDING status
    mock_backend = MagicMock()
    mock_status_update = MagicMock()
    mock_status_update.status = "PENDING"
    mock_status_update.status_message = "Deployment created"
    mock_backend.create_model_deployment = AsyncMock(return_value=mock_status_update)
    mock_backend_registry.get_backend.return_value = mock_backend

    # Mock SDK update_status method
    reconciler._models_client.update_deployment_status = AsyncMock(return_value=_ModelResponse())

    # Call the handler with the backend function
    await reconciler._reconcile_individual_deployment(deployment, mock_backend.create_model_deployment, "create")

    # Verify backend was called
    mock_backend.create_model_deployment.assert_called_once_with(deployment)

    assert _deployment_status_call(reconciler._models_client.update_deployment_status) == {
        "name": "test-deployment",
        "workspace": "default",
        "status": "PENDING",
        "version": "v1",
        "status_message": "Deployment created",
        "model_provider_id": None,
    }


@pytest.mark.asyncio
async def test_handle_created_deployment_backend_failure(reconciler, mock_backend_registry, make_deployment):
    """Test handling a CREATED deployment when backend fails."""
    deployment = make_deployment(status="CREATED")

    # Mock backend to raise exception
    mock_backend = MagicMock()
    mock_backend.create_model_deployment = AsyncMock(side_effect=Exception("Backend error"))
    mock_backend_registry.get_backend.return_value = mock_backend

    # Mock SDK update_status method
    reconciler._models_client.update_deployment_status = AsyncMock(return_value=_ModelResponse())

    # Call the handler - should not raise exception
    await reconciler._reconcile_individual_deployment(deployment, mock_backend.create_model_deployment, "create")

    # Verify backend was called
    mock_backend.create_model_deployment.assert_called_once_with(deployment)

    # Verify SDK update was called with ERROR status
    reconciler._models_client.update_deployment_status.assert_called_once()
    call_kwargs = _deployment_status_call(reconciler._models_client.update_deployment_status)
    assert call_kwargs["name"] == "test-deployment"
    assert call_kwargs["workspace"] == "default"
    assert call_kwargs["version"] == "v1"
    assert call_kwargs["status"] == "ERROR"
    assert "Failed to create deployment default/test-deployment" in call_kwargs["status_message"]


@pytest.mark.asyncio
async def test_reconcile_individual_deployment_monitor_ready_no_message_logs_debug(reconciler, make_deployment, caplog):
    """Routine monitor + READY with no status message should log at DEBUG, not INFO."""
    deployment = make_deployment(status="READY")
    status_update = DeploymentStatusUpdate(status="READY", status_message="", host_url=None)
    reconciler._models_client.update_deployment_status = AsyncMock(return_value=_ModelResponse())

    with caplog.at_level(logging.DEBUG, logger="nmp.core.models.controllers.deployment_reconciler"):
        await reconciler._reconcile_individual_deployment(
            deployment,
            AsyncMock(),
            "monitor",
            status_update=status_update,
        )

    monitor_records = [r for r in caplog.records if "Backend monitor status update" in r.message]
    assert len(monitor_records) == 1
    assert monitor_records[0].levelno == logging.DEBUG


@pytest.mark.asyncio
async def test_reconcile_individual_deployment_monitor_ready_with_message_logs_info(
    reconciler, make_deployment, caplog
):
    """Monitor + READY with a status message stays at INFO."""
    deployment = make_deployment(status="READY")
    status_update = DeploymentStatusUpdate(status="READY", status_message="NIM loading", host_url=None)
    reconciler._models_client.update_deployment_status = AsyncMock(return_value=_ModelResponse())

    with caplog.at_level(logging.INFO, logger="nmp.core.models.controllers.deployment_reconciler"):
        await reconciler._reconcile_individual_deployment(
            deployment,
            AsyncMock(),
            "monitor",
            status_update=status_update,
        )

    monitor_records = [r for r in caplog.records if "Backend monitor status update" in r.message]
    assert len(monitor_records) == 1
    assert monitor_records[0].levelno == logging.INFO


@pytest.mark.asyncio
async def test_reconcile_individual_deployment_conflict_is_noop(reconciler, mock_backend_registry, make_deployment):
    """Test ConflictError during status update is treated as no-op."""
    deployment = make_deployment(status="CREATED")

    # Mock backend to return PENDING status
    mock_backend = MagicMock()
    status_update = DeploymentStatusUpdate(
        status="PENDING",
        status_message="Deployment created",
        host_url=None,
    )
    mock_backend.create_model_deployment = AsyncMock(return_value=status_update)
    mock_backend_registry.get_backend.return_value = mock_backend

    # Main status update conflicts (deployment was marked DELETING server-side)
    reconciler._models_client.update_deployment_status = AsyncMock(side_effect=_status_error(409, "Conflict"))

    # Should not raise and should not attempt ERROR update
    await reconciler._reconcile_individual_deployment(deployment, mock_backend.create_model_deployment, "create")

    reconciler._models_client.update_deployment_status.assert_called_once()
    call_kwargs = _deployment_status_call(reconciler._models_client.update_deployment_status)
    assert call_kwargs["status"] == "PENDING"


@pytest.mark.asyncio
async def test_reconcile_individual_deployment_error_fallback_conflict_is_noop(
    reconciler, mock_backend_registry, make_deployment
):
    """Test ConflictError during fallback ERROR status update is treated as no-op."""
    deployment = make_deployment(status="CREATED")

    # Backend fails so reconciler enters fallback path to set ERROR
    mock_backend = MagicMock()
    mock_backend.create_model_deployment = AsyncMock(side_effect=Exception("Backend error"))
    mock_backend_registry.get_backend.return_value = mock_backend

    reconciler._models_client.update_deployment_status = AsyncMock(side_effect=_status_error(409, "Conflict"))

    # Should not raise if fallback ERROR update hits 409 conflict
    await reconciler._reconcile_individual_deployment(deployment, mock_backend.create_model_deployment, "create")

    reconciler._models_client.update_deployment_status.assert_called_once()
    call_kwargs = _deployment_status_call(reconciler._models_client.update_deployment_status)
    assert call_kwargs["status"] == "ERROR"


@pytest.mark.asyncio
async def test_reconcile_created_backend_error_persisted(reconciler, mock_backend_registry, make_deployment):
    """CREATED + backend ERROR is persisted verbatim to the deployment status."""
    deployment = make_deployment(status="CREATED")
    mock_backend = MagicMock()
    mock_backend.create_model_deployment = AsyncMock(
        return_value=DeploymentStatusUpdate(
            status="ERROR",
            status_message="Backend create failed for some reason",
        )
    )
    mock_backend_registry.get_backend.return_value = mock_backend
    reconciler._models_client.update_deployment_status = AsyncMock(return_value=_ModelResponse())

    await reconciler._reconcile_individual_deployment(deployment, mock_backend.create_model_deployment, "create")

    call_kwargs = _deployment_status_call(reconciler._models_client.update_deployment_status)
    assert call_kwargs["status"] == "ERROR"
    assert call_kwargs["status_message"] == "Backend create failed for some reason"


@pytest.mark.asyncio
async def test_reconcile_deployments_with_created_status(reconciler, mock_backend_registry, make_deployment):
    """Test processing deployments calls handler for CREATED deployments."""
    created_deployment = make_deployment(
        name="created-deployment", status="CREATED", config="test-config", config_version="v1"
    )
    pending_deployment = make_deployment(name="pending-deployment", status="PENDING")

    # Mock deployment config
    mock_deployment_config = MagicMock()

    # Create context objects (simulating what get_non_terminal_deployments returns)
    created_context = ModelContext(
        model_deployment=created_deployment,
        model_deployment_config=mock_deployment_config,
        model_provider=None,
        model_entity=None,
    )
    pending_context = ModelContext(
        model_deployment=pending_deployment,
        model_deployment_config=None,
        model_provider=None,
        model_entity=None,
    )

    # Mock backend
    mock_backend = MagicMock()
    mock_status_update = MagicMock()
    mock_status_update.status = "PENDING"
    mock_status_update.status_message = "Created"
    mock_backend.create_model_deployment = AsyncMock(return_value=mock_status_update)
    mock_backend.get_model_deployment_status = AsyncMock(return_value=mock_status_update)
    mock_backend_registry.get_backend.return_value = mock_backend

    # Mock SDK
    reconciler._models_client.update_deployment_status = AsyncMock(return_value=_ModelResponse())

    # Process deployments (now passing contexts with pre-fetched data)
    await reconciler.reconcile_deployments([created_context, pending_context])

    # Verify backend.create was called for CREATED deployment with the full context
    # (the context bundles deployment + config + entity; already pre-fetched, so no
    # retrieve happens during reconciliation).
    mock_backend.create_model_deployment.assert_called_once_with(created_context)

    # Verify backend.get_status was called for PENDING deployment with the context.
    mock_backend.get_model_deployment_status.assert_called_once_with(pending_context)

    # Verify SDK update was called twice (once for each deployment)
    assert reconciler._models_client.update_deployment_status.call_count == 2


# ============================================================================
# ModelProvider Lifecycle Management Tests
# ============================================================================


@pytest.mark.asyncio
async def test_ensure_model_provider_creates_when_not_exists(reconciler, make_deployment):
    """Test that ensure_model_provider creates provider when it doesn't exist."""
    # Mock provider doesn't exist (retrieve raises NotFoundError)
    reconciler._models_client.get_provider = AsyncMock(side_effect=_status_error(404, "Not found"))
    reconciler._models_client.create_provider = AsyncMock()

    deployment = make_deployment(workspace="test-ns", project="test-project")

    host_url = "http://test-ns/test-deployment"
    model_provider_id = await reconciler._ensure_model_provider(deployment, host_url)

    # Verify correct model_provider_id is returned
    assert model_provider_id == "test-ns/test-deployment"

    # Verify retrieve was called to check existence
    reconciler._models_client.get_provider.assert_called_once_with(
        name="test-deployment",
        workspace="test-ns",
    )

    assert _request_body_call(reconciler._models_client.create_provider) == {
        "workspace": "test-ns",
        "name": "test-deployment",
        "host_url": "http://test-ns/test-deployment",
        "description": "Auto-created provider for deployment test-deployment",
        "project": "test-project",
        "model_deployment_id": "test-ns/test-deployment",
        "status": "READY",
    }


@pytest.mark.asyncio
@patch("nmp.core.models.controllers.deployment_reconciler.uuid.uuid4")
async def test_ensure_model_provider_handles_name_collision(mock_uuid, reconciler, make_deployment):
    """Test that ensure_model_provider creates provider with UUID suffix when name collision occurs."""
    # Mock UUID to return predictable value
    mock_uuid_obj = MagicMock()
    mock_uuid_obj.hex = "abcdef1234567890"
    mock_uuid.return_value = mock_uuid_obj

    # Mock provider exists (retrieve succeeds on first call - collision)
    mock_provider = MagicMock()
    reconciler._models_client.get_provider = AsyncMock(return_value=_ModelResponse(mock_provider))
    reconciler._models_client.create_provider = AsyncMock()

    deployment = make_deployment(workspace="test-ns", project="test-project")

    host_url = "http://test-ns/test-deployment"
    model_provider_id = await reconciler._ensure_model_provider(deployment, host_url)

    # Verify UUID suffix was applied
    assert model_provider_id == "test-ns/test-deployment_abcdef12"

    # Verify retrieve was called to check existence
    reconciler._models_client.get_provider.assert_called_once()

    assert _request_body_call(reconciler._models_client.create_provider) == {
        "workspace": "test-ns",
        "name": "test-deployment_abcdef12",
        "host_url": "http://test-ns/test-deployment",
        "description": "Auto-created provider for deployment test-deployment",
        "project": "test-project",
        "model_deployment_id": "test-ns/test-deployment",
        "status": "READY",
    }


@pytest.mark.asyncio
async def test_ensure_model_provider_reuses_existing_when_already_set(reconciler, make_deployment):
    """Test that ensure_model_provider reuses existing provider when deployment.model_provider_id is set and host_url matches."""
    # Mock provider exists (retrieve succeeds) with matching host_url
    mock_provider = MagicMock()
    mock_provider.host_url = "http://test-ns/test-deployment"
    mock_provider.description = "Existing provider"
    mock_provider.enabled_models = None
    reconciler._models_client.get_provider = AsyncMock(return_value=_ModelResponse(mock_provider))
    reconciler._models_client.create_provider = AsyncMock()
    reconciler._models_client.upsert_provider = AsyncMock()

    deployment = make_deployment(
        workspace="test-ns", project="test-project", model_provider_id="test-ns/existing-provider"
    )

    host_url = "http://test-ns/test-deployment"
    model_provider_id = await reconciler._ensure_model_provider(deployment, host_url)

    # Verify the existing provider ID is returned
    assert model_provider_id == "test-ns/existing-provider"

    # Verify retrieve was called to check the existing provider exists
    reconciler._models_client.get_provider.assert_called_once_with(
        name="existing-provider",
        workspace="test-ns",
    )

    # Verify create was NOT called since we're reusing existing provider
    reconciler._models_client.create_provider.assert_not_called()

    # Verify update was NOT called since host_url matches
    reconciler._models_client.upsert_provider.assert_not_called()


@pytest.mark.asyncio
async def test_ensure_model_provider_updates_when_host_url_changes(reconciler, make_deployment):
    """Test that ensure_model_provider updates existing provider when host_url changes."""
    # Mock provider exists (retrieve succeeds) with different host_url
    mock_provider = MagicMock()
    mock_provider.host_url = "http://old-host/test-deployment"
    mock_provider.description = "Existing provider"
    mock_provider.enabled_models = ["model1", "model2"]
    reconciler._models_client.get_provider = AsyncMock(return_value=_ModelResponse(mock_provider))
    reconciler._models_client.create_provider = AsyncMock()
    reconciler._models_client.upsert_provider = AsyncMock()

    deployment = make_deployment(
        workspace="test-ns", project="test-project", model_provider_id="test-ns/existing-provider"
    )

    new_host_url = "http://new-host/test-deployment"
    model_provider_id = await reconciler._ensure_model_provider(deployment, new_host_url)

    # Verify the existing provider ID is returned
    assert model_provider_id == "test-ns/existing-provider"

    # Verify retrieve was called to check the existing provider
    reconciler._models_client.get_provider.assert_called_once_with(
        name="existing-provider",
        workspace="test-ns",
    )

    assert _request_body_call(reconciler._models_client.upsert_provider) == {
        "name": "existing-provider",
        "workspace": "test-ns",
        "host_url": new_host_url,
        "description": "Existing provider",
        "enabled_models": ["model1", "model2"],
        "status": "READY",
    }

    # Verify create was NOT called since we're updating existing provider
    reconciler._models_client.create_provider.assert_not_called()


@pytest.mark.asyncio
async def test_ensure_model_provider_creates_new_when_existing_not_found(reconciler, make_deployment):
    """Test that ensure_model_provider creates new provider when existing provider_id points to non-existent provider."""
    # First retrieve (checking existing provider) fails, second retrieve (checking name collision) fails too
    reconciler._models_client.get_provider = AsyncMock(side_effect=_status_error(404, "Not found"))
    reconciler._models_client.create_provider = AsyncMock()

    deployment = make_deployment(
        workspace="test-ns", project="test-project", model_provider_id="test-ns/missing-provider"
    )

    host_url = "http://test-ns/test-deployment"
    model_provider_id = await reconciler._ensure_model_provider(deployment, host_url)

    # Verify a new provider was created with the deployment name
    assert model_provider_id == "test-ns/test-deployment"

    # Verify retrieve was called twice (once for existing, once for name collision check)
    assert reconciler._models_client.get_provider.call_count == 2

    assert _request_body_call(reconciler._models_client.create_provider) == {
        "workspace": "test-ns",
        "name": "test-deployment",
        "host_url": "http://test-ns/test-deployment",
        "description": "Auto-created provider for deployment test-deployment",
        "project": "test-project",
        "model_deployment_id": "test-ns/test-deployment",
        "status": "READY",
    }


@pytest.mark.asyncio
async def test_delete_model_provider_deletes_when_exists(reconciler, make_deployment):
    """Test that delete_model_provider deletes provider when it exists."""
    # Mock provider exists and delete succeeds
    reconciler._models_client.delete_provider = AsyncMock()

    # Mock the cleanup method to track if it's called
    reconciler._cleanup_model_entities_for_provider = AsyncMock()

    deployment = make_deployment(workspace="test-ns", model_provider_id="test-ns/test-deployment")

    await reconciler._delete_model_provider(deployment)

    # Verify cleanup was called before deletion
    reconciler._cleanup_model_entities_for_provider.assert_called_once_with(
        "test-ns", "test-deployment", "test-ns/test-deployment"
    )

    # Verify delete was called with correct parameters
    reconciler._models_client.delete_provider.assert_called_once_with(
        name="test-deployment",
        workspace="test-ns",
    )


@pytest.mark.asyncio
async def test_delete_model_provider_handles_not_found(reconciler, make_deployment):
    """Test that delete_model_provider handles NotFoundError gracefully."""
    # Mock provider doesn't exist (delete raises NotFoundError)
    reconciler._models_client.delete_provider = AsyncMock(side_effect=_status_error(404, "Not found"))

    # Mock the cleanup method to track if it's called
    reconciler._cleanup_model_entities_for_provider = AsyncMock()

    deployment = make_deployment(workspace="test-ns", model_provider_id="test-ns/test-deployment")

    # Should not raise exception
    await reconciler._delete_model_provider(deployment)

    # Verify cleanup was called even though delete failed
    reconciler._cleanup_model_entities_for_provider.assert_called_once_with(
        "test-ns", "test-deployment", "test-ns/test-deployment"
    )

    # Verify delete was called
    reconciler._models_client.delete_provider.assert_called_once()


@pytest.mark.asyncio
async def test_delete_model_provider_skips_when_no_provider_id(reconciler, make_deployment):
    """Test that delete_model_provider skips deletion when model_provider_id is not set."""
    reconciler._models_client.delete_provider = AsyncMock()

    # Mock the cleanup method to track if it's called
    reconciler._cleanup_model_entities_for_provider = AsyncMock()

    deployment = make_deployment(workspace="test-ns", model_provider_id=None)

    await reconciler._delete_model_provider(deployment)

    # Verify cleanup was NOT called (early return)
    reconciler._cleanup_model_entities_for_provider.assert_not_called()

    # Verify delete was NOT called
    reconciler._models_client.delete_provider.assert_not_called()


@pytest.mark.asyncio
async def test_delete_model_provider_handles_uuid_suffix(reconciler, make_deployment):
    """Test that delete_model_provider correctly parses provider ID with UUID suffix."""
    reconciler._models_client.delete_provider = AsyncMock()

    # Mock the cleanup method to track if it's called
    reconciler._cleanup_model_entities_for_provider = AsyncMock()

    deployment = make_deployment(workspace="test-ns", model_provider_id="test-ns/test-deployment_abcdef12")

    await reconciler._delete_model_provider(deployment)

    # Verify cleanup was called with UUID-suffixed provider ID
    reconciler._cleanup_model_entities_for_provider.assert_called_once_with(
        "test-ns", "test-deployment_abcdef12", "test-ns/test-deployment_abcdef12"
    )

    # Verify delete was called with UUID-suffixed name
    reconciler._models_client.delete_provider.assert_called_once_with(
        name="test-deployment_abcdef12",
        workspace="test-ns",
    )


@pytest.mark.asyncio
async def test_reconcile_model_provider_creates_for_ready_status(reconciler, make_deployment):
    """Test that reconcile_model_provider creates provider when status is READY."""
    # Mock provider doesn't exist
    reconciler._models_client.get_provider = AsyncMock(side_effect=_status_error(404, "Not found"))
    reconciler._models_client.create_provider = AsyncMock()

    deployment = make_deployment(workspace="test-ns", project="test-project")

    status_update = DeploymentStatusUpdate(
        status="READY", status_message="Deployment is ready", host_url="http://test-ns/test-deployment"
    )
    model_provider_id = await reconciler._reconcile_model_provider(deployment, status_update)

    # Verify model_provider_id is returned
    assert model_provider_id == "test-ns/test-deployment"

    # Verify create was called
    reconciler._models_client.create_provider.assert_called_once()


@pytest.mark.asyncio
async def test_reconcile_model_provider_deletes_for_deleted_status(reconciler, make_deployment):
    """Test that reconcile_model_provider deletes provider when status is DELETED or DELETING."""
    # Mock provider exists
    reconciler._models_client.delete_provider = AsyncMock()

    deployment = make_deployment(workspace="test-ns", model_provider_id="test-ns/test-deployment")

    status_update = DeploymentStatusUpdate(status="DELETED", status_message="Deployment deleted", host_url=None)
    model_provider_id = await reconciler._reconcile_model_provider(deployment, status_update)

    # Verify None is returned for DELETED status
    assert model_provider_id is None

    # Verify delete was called
    reconciler._models_client.delete_provider.assert_called_once()


@pytest.mark.asyncio
async def test_reconcile_model_provider_deletes_for_deleting_status(reconciler, make_deployment):
    """Test that reconcile_model_provider deletes provider when status is DELETING."""
    # Mock provider exists
    reconciler._models_client.delete_provider = AsyncMock()

    deployment = make_deployment(workspace="test-ns", model_provider_id="test-ns/test-deployment")

    status_update = DeploymentStatusUpdate(status="DELETING", status_message="Deployment deleting", host_url=None)
    model_provider_id = await reconciler._reconcile_model_provider(deployment, status_update)

    # Verify None is returned for DELETING status
    assert model_provider_id is None

    # Verify delete was called
    reconciler._models_client.delete_provider.assert_called_once()


@pytest.mark.asyncio
async def test_reconcile_model_provider_does_nothing_for_other_statuses(reconciler, make_deployment):
    """Test that reconcile_model_provider does nothing for statuses other than READY/DELETED/DELETING."""
    reconciler._models_client.get_provider = AsyncMock()
    reconciler._models_client.create_provider = AsyncMock()
    reconciler._models_client.delete_provider = AsyncMock()

    deployment = make_deployment(workspace="test-ns")

    # Test PENDING status
    status_update = DeploymentStatusUpdate(status="PENDING", status_message="Deployment pending")
    result = await reconciler._reconcile_model_provider(deployment, status_update)
    assert result is None

    # Test CREATED status
    status_update = DeploymentStatusUpdate(status="CREATED", status_message="Deployment created")
    result = await reconciler._reconcile_model_provider(deployment, status_update)
    assert result is None

    # Test ERROR status
    status_update = DeploymentStatusUpdate(status="ERROR", status_message="Deployment error")
    result = await reconciler._reconcile_model_provider(deployment, status_update)
    assert result is None

    # Verify no provider operations were called
    reconciler._models_client.get_provider.assert_not_called()
    reconciler._models_client.create_provider.assert_not_called()
    reconciler._models_client.delete_provider.assert_not_called()


@pytest.mark.asyncio
async def test_reconcile_model_provider_handles_errors_gracefully(reconciler, make_deployment):
    """Test that reconcile_model_provider handles errors without failing deployment update."""
    # Mock provider creation fails with unexpected error
    reconciler._models_client.get_provider = AsyncMock(side_effect=_status_error(404, "Not found"))
    reconciler._models_client.create_provider = AsyncMock(side_effect=Exception("API Error"))

    deployment = make_deployment(workspace="test-ns", project="test-project")

    # Should not raise exception - error is logged as warning
    status_update = DeploymentStatusUpdate(
        status="READY", status_message="Deployment is ready", host_url="http://test-ns/test-deployment"
    )
    result = await reconciler._reconcile_model_provider(deployment, status_update)

    # Verify None is returned when error occurs
    assert result is None

    # Verify create was attempted
    reconciler._models_client.create_provider.assert_called_once()


@pytest.mark.asyncio
async def test_full_deployment_lifecycle_with_provider_management(reconciler, mock_backend_registry, make_deployment):
    """Test full deployment lifecycle: CREATED -> READY -> DELETED with provider management."""
    # Setup mocks
    mock_backend = MagicMock()
    mock_backend_registry.get_backend.return_value = mock_backend

    # Mock backend responses for different states
    created_status = DeploymentStatusUpdate(
        status="PENDING",
        status_message="Deployment created",
        host_url="http://test-ns/test-deployment",
    )

    ready_status = DeploymentStatusUpdate(
        status="READY",
        status_message="Deployment ready",
        host_url="http://test-ns/test-deployment",
    )

    deleted_status = DeploymentStatusUpdate(
        status="DELETED",
        status_message="Deployment deleted",
        host_url=None,
    )

    mock_backend.create_model_deployment = AsyncMock(return_value=created_status)
    mock_backend.get_model_deployment_status = AsyncMock(return_value=ready_status)
    mock_backend.delete_model_deployment = AsyncMock(return_value=deleted_status)

    # Mock provider operations
    reconciler._models_client.update_deployment_status = AsyncMock(return_value=_ModelResponse())
    reconciler._models_client.get_provider = AsyncMock(side_effect=_status_error(404, "Not found"))
    reconciler._models_client.create_provider = AsyncMock()
    reconciler._models_client.delete_provider = AsyncMock()

    deployment = make_deployment(workspace="test-ns", status="CREATED", project="test-project")

    # Step 1: Create deployment (CREATED -> PENDING)
    await reconciler._reconcile_individual_deployment(deployment, mock_backend.create_model_deployment, "create")

    # Provider should NOT be created for PENDING status
    reconciler._models_client.create_provider.assert_not_called()

    # Step 2: Deployment becomes READY
    deployment.status = "PENDING"
    await reconciler._reconcile_individual_deployment(
        deployment, mock_backend.get_model_deployment_status, "check status"
    )

    assert _request_body_call(reconciler._models_client.create_provider) == {
        "workspace": "test-ns",
        "name": "test-deployment",
        "host_url": "http://test-ns/test-deployment",
        "description": "Auto-created provider for deployment test-deployment",
        "project": "test-project",
        "model_deployment_id": "test-ns/test-deployment",
        "status": "READY",
    }

    # Verify the status update for READY included model_provider_id
    ready_call = _deployment_status_call(reconciler._models_client.update_deployment_status, 1)
    assert ready_call["model_provider_id"] == "test-ns/test-deployment"

    # Step 3: Delete deployment (READY -> DELETED)
    # The deployment should now have the model_provider_id set from when it was READY
    deployment.status = "DELETING"
    deployment.model_provider_id = "test-ns/test-deployment"
    await reconciler._reconcile_individual_deployment(
        deployment, lambda d: mock_backend.delete_model_deployment(d.workspace, d.name), "delete"
    )

    # Provider SHOULD be deleted for DELETED status
    reconciler._models_client.delete_provider.assert_called_once_with(
        name="test-deployment",
        workspace="test-ns",
    )

    # Verify all deployment status updates were called
    assert reconciler._models_client.update_deployment_status.call_count == 3


# ============================================================================
# DELETED Deployment Cleanup Tests
# ============================================================================


@pytest.mark.asyncio
async def test_handle_deleted_deployment_cleanup_after_grace_period(reconciler, make_deployment):
    """Test that DELETED deployments are hard-deleted after grace period."""
    past_time = datetime.now(timezone.utc) - timedelta(seconds=31)
    deployment = make_deployment(
        spec=ModelDeployment,
        workspace="test-workspace",
        status="DELETED",
        entity_version=1,
        updated_at=past_time,
    )

    # Mock the SDK versions.delete method
    reconciler._models_client.delete_deployment_version = AsyncMock()

    # Call handle_deleted_deployment
    await reconciler._handle_deleted_deployment(deployment)

    # Verify hard delete was called for the specific version
    reconciler._models_client.delete_deployment_version.assert_called_once_with(
        name="1",
        workspace="test-workspace",
        deployment="test-deployment",
    )


@pytest.mark.asyncio
async def test_handle_deleted_deployment_no_cleanup_within_grace_period(reconciler, make_deployment):
    """Test that DELETED deployments are NOT hard-deleted within grace period."""
    past_time = datetime.now(timezone.utc) - timedelta(seconds=15)
    deployment = make_deployment(
        spec=ModelDeployment,
        workspace="test-workspace",
        status="DELETED",
        entity_version=1,
        updated_at=past_time,
    )

    # Mock the SDK versions.delete method
    reconciler._models_client.delete_deployment_version = AsyncMock()

    # Call handle_deleted_deployment
    await reconciler._handle_deleted_deployment(deployment)

    # Verify hard delete was NOT called
    reconciler._models_client.delete_deployment_version.assert_not_called()


@pytest.mark.asyncio
async def test_handle_deleted_deployment_with_naive_datetime(reconciler, make_deployment):
    """Test that handle_deleted_deployment handles timezone-naive datetimes correctly."""
    # Use a naive datetime that represents UTC time minus 31 seconds.
    # We use datetime.now(timezone.utc).replace(tzinfo=None) to get a naive datetime
    # that represents actual UTC time, ensuring the test works regardless of local timezone.
    past_time_naive = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(seconds=31)
    deployment = make_deployment(
        spec=ModelDeployment,
        workspace="test-workspace",
        status="DELETED",
        entity_version=1,
        updated_at=past_time_naive,
    )

    # Mock the SDK versions.delete method
    reconciler._models_client.delete_deployment_version = AsyncMock()

    # Call handle_deleted_deployment - should NOT raise TypeError
    await reconciler._handle_deleted_deployment(deployment)

    # Verify hard delete was called for the specific version (deployment is past grace period)
    reconciler._models_client.delete_deployment_version.assert_called_once_with(
        name="1",
        workspace="test-workspace",
        deployment="test-deployment",
    )


@pytest.mark.asyncio
async def test_reconcile_deployments_calls_handle_deleted(reconciler, make_deployment):
    """Test that reconcile_deployments calls handle_deleted_deployment for DELETED status."""
    past_time = datetime.now(timezone.utc) - timedelta(seconds=31)
    deleted_deployment = make_deployment(
        spec=ModelDeployment,
        name="deleted-deployment",
        workspace="test-workspace",
        status="DELETED",
        entity_version=1,
        updated_at=past_time,
    )

    # Create context object
    deleted_context = ModelContext(
        model_deployment=deleted_deployment,
        model_deployment_config=None,
        model_provider=None,
        model_entity=None,
    )

    # Mock the SDK versions.delete method
    reconciler._models_client.delete_deployment_version = AsyncMock()

    # Call reconcile_deployments with a list containing the DELETED deployment context
    await reconciler.reconcile_deployments([deleted_context])

    # Verify hard delete was called for the specific version (since it's past grace period)
    reconciler._models_client.delete_deployment_version.assert_called_once_with(
        name="1",
        workspace="test-workspace",
        deployment="deleted-deployment",
    )


# ============================================================================
# Tests for _cleanup_model_entities_for_provider
# ============================================================================


@pytest.mark.asyncio
async def test_cleanup_model_entities_removes_provider_from_entities(reconciler):
    """Test that cleanup removes provider from Model Entity model_providers list."""
    # Mock provider with served_models
    mock_provider = MagicMock()
    mock_provider.served_models = [
        MagicMock(model_entity_id="test-ns/model-1"),
        MagicMock(model_entity_id="test-ns/model-2"),
    ]

    reconciler._models_client.get_provider = AsyncMock(return_value=_ModelResponse(mock_provider))
    await seed_entity_cache(
        reconciler._models_sdk,
        reconciler._entity_cache,
        [
            _entity("test-ns", "model-1", ["test-ns/provider-1", "other-ns/other-provider"]),
            _entity("test-ns", "model-2", ["test-ns/provider-1"]),
        ],
    )

    # Call cleanup
    await reconciler._cleanup_model_entities_for_provider("test-ns", "provider-1", "test-ns/provider-1")
    await reconciler._entity_cache.flush()

    # Verify provider was retrieved
    reconciler._models_client.get_provider.assert_called_once_with(
        name="provider-1",
        workspace="test-ns",
    )

    # Verify model entities were updated with provider removed
    update = reconciler._models_sdk.models_client.update_model
    assert update.await_count == 2
    calls = {call.kwargs["name"]: call.kwargs["body"].model_providers for call in update.await_args_list}
    assert calls["model-1"] == ["other-ns/other-provider"]
    assert calls["model-2"] == []


@pytest.mark.asyncio
async def test_cleanup_model_entities_no_served_models(reconciler):
    """Test that cleanup returns early when provider has no served_models."""
    # Mock provider with empty served_models
    mock_provider = MagicMock()
    mock_provider.served_models = []

    reconciler._models_client.get_provider = AsyncMock(return_value=_ModelResponse(mock_provider))

    # Call cleanup
    await reconciler._cleanup_model_entities_for_provider("test-ns", "provider-1", "test-ns/provider-1")
    await reconciler._entity_cache.flush()

    # Verify provider was retrieved
    reconciler._models_client.get_provider.assert_called_once()

    # Verify no model entity operations were performed
    reconciler._models_sdk.models_client.update_model.assert_not_awaited()


@pytest.mark.asyncio
async def test_cleanup_model_entities_provider_not_found(reconciler):
    """Test that cleanup handles NotFoundError gracefully when provider doesn't exist."""
    reconciler._models_client.get_provider = AsyncMock(side_effect=_status_error(404, "Provider not found"))

    # Call cleanup - should not raise
    await reconciler._cleanup_model_entities_for_provider("test-ns", "provider-1", "test-ns/provider-1")
    await reconciler._entity_cache.flush()

    # Verify no model entity operations were performed
    reconciler._models_sdk.models_client.update_model.assert_not_awaited()


@pytest.mark.asyncio
async def test_cleanup_model_entities_provider_not_in_list(reconciler):
    """Test that cleanup skips update when provider not in Model Entity's model_providers list."""
    # Mock provider with served_models
    mock_provider = MagicMock()
    mock_provider.served_models = [
        MagicMock(model_entity_id="test-ns/model-1"),
    ]

    reconciler._models_client.get_provider = AsyncMock(return_value=_ModelResponse(mock_provider))
    await seed_entity_cache(
        reconciler._models_sdk,
        reconciler._entity_cache,
        [_entity("test-ns", "model-1", ["other-ns/other-provider"])],
    )

    # Call cleanup
    await reconciler._cleanup_model_entities_for_provider("test-ns", "provider-1", "test-ns/provider-1")
    await reconciler._entity_cache.flush()

    # Verify model entity was NOT updated (provider wasn't in the list)
    reconciler._models_sdk.models_client.update_model.assert_not_awaited()


@pytest.mark.asyncio
async def test_cleanup_model_entities_skips_missing_entity_and_continues(reconciler):
    """A served model with no Model Entity is skipped without affecting the others."""
    # Mock provider with multiple served_models
    mock_provider = MagicMock()
    mock_provider.served_models = [
        MagicMock(model_entity_id="test-ns/model-1"),
        MagicMock(model_entity_id="test-ns/model-2"),
    ]

    reconciler._models_client.get_provider = AsyncMock(return_value=_ModelResponse(mock_provider))
    # Only model-2 exists.
    await seed_entity_cache(
        reconciler._models_sdk,
        reconciler._entity_cache,
        [_entity("test-ns", "model-2", ["test-ns/provider-1"])],
    )

    # Call cleanup - should not raise
    await reconciler._cleanup_model_entities_for_provider("test-ns", "provider-1", "test-ns/provider-1")
    await reconciler._entity_cache.flush()

    # Verify only the existing model was updated
    update = reconciler._models_sdk.models_client.update_model
    update.assert_awaited_once()
    call = update.await_args
    assert call is not None
    assert call.kwargs["workspace"] == "test-ns"
    assert call.kwargs["name"] == "model-2"
    assert call.kwargs["body"].model_providers == []


@pytest.mark.asyncio
async def test_cleanup_model_entities_handles_model_update_failure(reconciler):
    """Test that cleanup continues processing other models when one update fails."""
    # Mock provider with multiple served_models
    mock_provider = MagicMock()
    mock_provider.served_models = [
        MagicMock(model_entity_id="test-ns/model-1"),
        MagicMock(model_entity_id="test-ns/model-2"),
    ]

    reconciler._models_client.get_provider = AsyncMock(return_value=_ModelResponse(mock_provider))
    await seed_entity_cache(
        reconciler._models_sdk,
        reconciler._entity_cache,
        [
            _entity("test-ns", "model-1", ["test-ns/provider-1"]),
            _entity("test-ns", "model-2", ["test-ns/provider-1"]),
        ],
    )
    # First update fails, second succeeds
    reconciler._models_sdk.models_client.update_model = AsyncMock(
        side_effect=[Exception("Update failed"), _ModelResponse()]
    )

    # Call cleanup - should not raise
    await reconciler._cleanup_model_entities_for_provider("test-ns", "provider-1", "test-ns/provider-1")
    await reconciler._entity_cache.flush()

    # Verify both models were attempted to be updated
    assert reconciler._models_sdk.models_client.update_model.await_count == 2


@pytest.mark.asyncio
async def test_cleanup_model_entities_with_null_model_providers(reconciler):
    """Test that cleanup handles Model Entity with null/None model_providers list."""
    # Mock provider with served_models
    mock_provider = MagicMock()
    mock_provider.served_models = [
        MagicMock(model_entity_id="test-ns/model-1"),
    ]

    reconciler._models_client.get_provider = AsyncMock(return_value=_ModelResponse(mock_provider))
    await seed_entity_cache(
        reconciler._models_sdk,
        reconciler._entity_cache,
        [_entity("test-ns", "model-1", None)],
    )

    # Call cleanup - should not raise
    await reconciler._cleanup_model_entities_for_provider("test-ns", "provider-1", "test-ns/provider-1")
    await reconciler._entity_cache.flush()

    # Verify model entity was NOT updated (provider wasn't in the empty/null list)
    reconciler._models_sdk.models_client.update_model.assert_not_awaited()


@pytest.mark.asyncio
async def test_lost_status_triggers_drift_recovery(reconciler, mock_backend_registry, make_deployment):
    """Test that LOST status from backend triggers drift recovery."""
    deployment = make_deployment(status="PENDING")

    # Mock deployment config
    mock_deployment_config = MagicMock()

    # Create context
    ctx = ModelContext(
        model_deployment=deployment,
        model_deployment_config=mock_deployment_config,
        model_provider=None,
        model_entity=None,
    )

    # Mock backend to return LOST on status check, then PENDING on create
    mock_backend = MagicMock()
    lost_status = DeploymentStatusUpdate(
        status="LOST",
        status_message="Container not found",
        host_url=None,
    )
    pending_status = DeploymentStatusUpdate(
        status="PENDING",
        status_message="Container created",
        host_url="http://localhost:8500",
    )
    mock_backend.get_model_deployment_status = AsyncMock(return_value=lost_status)
    mock_backend.create_model_deployment = AsyncMock(return_value=pending_status)
    mock_backend_registry.get_backend.return_value = mock_backend

    # Mock SDK
    reconciler._models_client.update_deployment_status = AsyncMock(return_value=_ModelResponse())

    # Process deployment
    await reconciler.reconcile_deployments([ctx])

    # Verify backend.create_model_deployment was called for recovery with the context
    mock_backend.create_model_deployment.assert_called_once_with(ctx)

    # Verify status was updated to PENDING with recovery message
    reconciler._models_client.update_deployment_status.assert_called_once()
    call_kwargs = _deployment_status_call(reconciler._models_client.update_deployment_status)
    assert call_kwargs["status"] == "PENDING"
    assert "Recovering deployment" in call_kwargs["status_message"]
    assert "attempt 1/" in call_kwargs["status_message"]


@pytest.mark.asyncio
async def test_successful_status_clears_drift_state(reconciler, mock_backend_registry, make_deployment):
    """Test that successful status check clears drift recovery state."""
    deployment = make_deployment(status="READY")

    ctx = ModelContext(
        model_deployment=deployment,
        model_deployment_config=None,
        model_provider=None,
        model_entity=None,
    )

    # Pre-populate drift recovery state via cache internals (acceptable for tests)
    from nmp.core.models.controllers.deployment_reconciler import DriftRecoveryState

    reconciler._drift_recovery_cache._states["default/test-deployment"] = DriftRecoveryState(attempts=2)

    # Mock backend to return READY status (healthy)
    mock_backend = MagicMock()
    ready_status = DeploymentStatusUpdate(
        status="READY",
        status_message="Container is ready",
        host_url="http://localhost:8500",
    )
    mock_backend.get_model_deployment_status = AsyncMock(return_value=ready_status)
    mock_backend_registry.get_backend.return_value = mock_backend

    # Mock SDK
    reconciler._models_client.update_deployment_status = AsyncMock(return_value=_ModelResponse())
    reconciler._models_client.get_provider = AsyncMock(side_effect=_status_error(404, "Not found"))
    reconciler._models_client.create_provider = AsyncMock()

    # Process deployment
    await reconciler.reconcile_deployments([ctx])

    # Verify drift recovery state was cleared
    assert "default/test-deployment" not in reconciler._drift_recovery_cache._states


@pytest.mark.asyncio
async def test_pending_status_preserves_drift_state(reconciler, mock_backend_registry, make_deployment):
    """Test that PENDING status does NOT clear drift recovery state.

    This is critical to prevent infinite LOST -> PENDING -> LOST loops.
    The retry counter should only be cleared when deployment reaches READY.
    """
    deployment = make_deployment(status="PENDING", model_provider_id=None)

    ctx = ModelContext(
        model_deployment=deployment,
        model_deployment_config=None,
        model_provider=None,
        model_entity=None,
    )

    # Pre-populate drift recovery state (simulating a previous recovery attempt)
    from nmp.core.models.controllers.deployment_reconciler import DriftRecoveryState

    reconciler._drift_recovery_cache._states["default/test-deployment"] = DriftRecoveryState(attempts=2)

    # Mock backend to return PENDING status (still starting up after recreation)
    mock_backend = MagicMock()
    pending_status = DeploymentStatusUpdate(
        status="PENDING",
        status_message="Container is starting",
        host_url=None,
    )
    mock_backend.get_model_deployment_status = AsyncMock(return_value=pending_status)
    mock_backend_registry.get_backend.return_value = mock_backend

    # Mock SDK
    reconciler._models_client.update_deployment_status = AsyncMock(return_value=_ModelResponse())

    # Process deployment
    await reconciler.reconcile_deployments([ctx])

    # Verify drift recovery state was NOT cleared - this is the key assertion
    assert "default/test-deployment" in reconciler._drift_recovery_cache._states
    assert reconciler._drift_recovery_cache.get_attempts("default/test-deployment") == 2


@pytest.mark.asyncio
async def test_drift_recovery_max_retries_exceeded(reconciler, mock_backend_registry, make_deployment):
    """Test that exceeding max retries sets deployment to ERROR."""
    deployment = make_deployment(status="PENDING")

    mock_deployment_config = MagicMock()

    ctx = ModelContext(
        model_deployment=deployment,
        model_deployment_config=mock_deployment_config,
        model_provider=None,
        model_entity=None,
    )

    # Set max attempts to 3 for testing (need to recreate cache with new config)
    reconciler._controller_config.drift_recovery_max_attempts = 3
    reconciler._drift_recovery_cache._max_attempts = 3

    # Pre-populate drift recovery state at max attempts
    from nmp.core.models.controllers.deployment_reconciler import DriftRecoveryState

    reconciler._drift_recovery_cache._states["default/test-deployment"] = DriftRecoveryState(attempts=3)

    # Mock backend to return LOST
    mock_backend = MagicMock()
    lost_status = DeploymentStatusUpdate(
        status="LOST",
        status_message="Container not found",
        host_url=None,
    )
    mock_backend.get_model_deployment_status = AsyncMock(return_value=lost_status)
    mock_backend.create_model_deployment = AsyncMock()  # Should not be called
    mock_backend_registry.get_backend.return_value = mock_backend

    # Mock SDK
    reconciler._models_client.update_deployment_status = AsyncMock(return_value=_ModelResponse())

    # Process deployment
    await reconciler.reconcile_deployments([ctx])

    # Verify create was NOT called (max retries exceeded)
    mock_backend.create_model_deployment.assert_not_called()

    # Verify status was updated to ERROR
    reconciler._models_client.update_deployment_status.assert_called_once()
    call_kwargs = _deployment_status_call(reconciler._models_client.update_deployment_status)
    assert call_kwargs["status"] == "ERROR"
    assert "failed after 3 attempts" in call_kwargs["status_message"]


@pytest.mark.asyncio
async def test_drift_recovery_respects_backoff(reconciler, mock_backend_registry, make_deployment):
    """Test that drift recovery respects exponential backoff."""
    deployment = make_deployment(status="PENDING")

    mock_deployment_config = MagicMock()

    ctx = ModelContext(
        model_deployment=deployment,
        model_deployment_config=mock_deployment_config,
        model_provider=None,
        model_entity=None,
    )

    # Set short backoff for testing (update cache config)
    reconciler._controller_config.drift_recovery_base_delay_seconds = 60
    reconciler._controller_config.drift_recovery_max_delay_seconds = 300
    reconciler._drift_recovery_cache._base_delay_seconds = 60
    reconciler._drift_recovery_cache._max_delay_seconds = 300

    # Pre-populate drift recovery state with recent attempt
    from nmp.core.models.controllers.deployment_reconciler import DriftRecoveryState

    reconciler._drift_recovery_cache._states["default/test-deployment"] = DriftRecoveryState(
        attempts=1,
        last_attempt_at=datetime.now(timezone.utc),  # Just now
    )

    # Mock backend to return LOST
    mock_backend = MagicMock()
    lost_status = DeploymentStatusUpdate(
        status="LOST",
        status_message="Container not found",
        host_url=None,
    )
    mock_backend.get_model_deployment_status = AsyncMock(return_value=lost_status)
    mock_backend.create_model_deployment = AsyncMock()
    mock_backend_registry.get_backend.return_value = mock_backend

    # Mock SDK
    reconciler._models_client.update_deployment_status = AsyncMock(return_value=_ModelResponse())

    # Process deployment
    await reconciler.reconcile_deployments([ctx])

    # Verify create was NOT called (in backoff period)
    mock_backend.create_model_deployment.assert_not_called()

    # Verify status was NOT updated (skipped this cycle)
    reconciler._models_client.update_deployment_status.assert_not_called()


@pytest.mark.asyncio
async def test_drift_recovery_proceeds_after_backoff(reconciler, mock_backend_registry, make_deployment):
    """Test that drift recovery proceeds after backoff period."""
    deployment = make_deployment(status="PENDING")

    mock_deployment_config = MagicMock()

    ctx = ModelContext(
        model_deployment=deployment,
        model_deployment_config=mock_deployment_config,
        model_provider=None,
        model_entity=None,
    )

    # Set short backoff for testing (update cache config)
    reconciler._controller_config.drift_recovery_base_delay_seconds = 30
    reconciler._controller_config.drift_recovery_max_delay_seconds = 300
    reconciler._drift_recovery_cache._base_delay_seconds = 30
    reconciler._drift_recovery_cache._max_delay_seconds = 300

    # Pre-populate drift recovery state with old attempt (backoff expired)
    from nmp.core.models.controllers.deployment_reconciler import DriftRecoveryState

    reconciler._drift_recovery_cache._states["default/test-deployment"] = DriftRecoveryState(
        attempts=1,
        last_attempt_at=datetime.now(timezone.utc) - timedelta(seconds=120),  # 2 minutes ago
    )

    # Mock backend
    mock_backend = MagicMock()
    lost_status = DeploymentStatusUpdate(
        status="LOST",
        status_message="Container not found",
        host_url=None,
    )
    pending_status = DeploymentStatusUpdate(
        status="PENDING",
        status_message="Container created",
        host_url="http://localhost:8500",
    )
    mock_backend.get_model_deployment_status = AsyncMock(return_value=lost_status)
    mock_backend.create_model_deployment = AsyncMock(return_value=pending_status)
    mock_backend_registry.get_backend.return_value = mock_backend

    # Mock SDK
    reconciler._models_client.update_deployment_status = AsyncMock(return_value=_ModelResponse())

    # Process deployment
    await reconciler.reconcile_deployments([ctx])

    # Verify create WAS called (backoff expired)
    mock_backend.create_model_deployment.assert_called_once()

    # Verify attempts was incremented
    assert reconciler._drift_recovery_cache.get_attempts("default/test-deployment") == 2


@pytest.mark.asyncio
async def test_drift_recovery_ready_deployment(reconciler, mock_backend_registry, make_deployment):
    """Test that LOST status triggers recovery for READY deployments too."""
    deployment = make_deployment(status="READY")  # Was READY, now lost

    mock_deployment_config = MagicMock()

    ctx = ModelContext(
        model_deployment=deployment,
        model_deployment_config=mock_deployment_config,
        model_provider=None,
        model_entity=None,
    )

    # Mock backend
    mock_backend = MagicMock()
    lost_status = DeploymentStatusUpdate(
        status="LOST",
        status_message="Container not found",
        host_url=None,
    )
    pending_status = DeploymentStatusUpdate(
        status="PENDING",
        status_message="Container recreated",
        host_url="http://localhost:8500",
    )
    mock_backend.get_model_deployment_status = AsyncMock(return_value=lost_status)
    mock_backend.create_model_deployment = AsyncMock(return_value=pending_status)
    mock_backend_registry.get_backend.return_value = mock_backend

    # Mock SDK
    reconciler._models_client.update_deployment_status = AsyncMock(return_value=_ModelResponse())

    # Process deployment
    await reconciler.reconcile_deployments([ctx])

    # Verify recovery was triggered
    mock_backend.create_model_deployment.assert_called_once()

    # Verify status message indicates recovery
    call_kwargs = _deployment_status_call(reconciler._models_client.update_deployment_status)
    assert "Recovering deployment" in call_kwargs["status_message"]


@pytest.mark.asyncio
async def test_unknown_status_triggers_handler_and_updates_status(reconciler, mock_backend_registry, make_deployment):
    """Test that UNKNOWN status from backend triggers unknown handler and updates deployment."""
    deployment = make_deployment(status="PENDING")

    ctx = ModelContext(
        model_deployment=deployment,
        model_deployment_config=None,
        model_provider=None,
        model_entity=None,
    )

    # Mock backend to return UNKNOWN (simulating Docker API failure)
    mock_backend = MagicMock()
    unknown_status = DeploymentStatusUpdate(
        status="UNKNOWN",
        status_message="Failed to communicate with backend: connection refused",
        host_url=None,
    )
    mock_backend.get_model_deployment_status = AsyncMock(return_value=unknown_status)
    mock_backend_registry.get_backend.return_value = mock_backend

    # Mock SDK
    reconciler._models_client.update_deployment_status = AsyncMock(return_value=_ModelResponse())

    # Process deployment
    await reconciler.reconcile_deployments([ctx])

    # Verify status was updated to UNKNOWN with attempt info
    reconciler._models_client.update_deployment_status.assert_called_once()
    call_kwargs = _deployment_status_call(reconciler._models_client.update_deployment_status)
    assert call_kwargs["status"] == "UNKNOWN"
    assert "attempt 1/" in call_kwargs["status_message"]
    assert "Unable to determine deployment status" in call_kwargs["status_message"]

    # Verify attempt was tracked
    assert reconciler._drift_recovery_cache.get_attempts("default/test-deployment") == 1


@pytest.mark.asyncio
async def test_unknown_status_max_retries_sets_error(reconciler, mock_backend_registry, make_deployment):
    """Test that repeated UNKNOWN status eventually transitions to ERROR."""
    deployment = make_deployment(status="UNKNOWN")

    ctx = ModelContext(
        model_deployment=deployment,
        model_deployment_config=None,
        model_provider=None,
        model_entity=None,
    )

    # Set max attempts to 3 for testing
    reconciler._controller_config.drift_recovery_max_attempts = 3
    reconciler._drift_recovery_cache._max_attempts = 3

    # Pre-populate state at max attempts
    from nmp.core.models.controllers.deployment_reconciler import DriftRecoveryState

    reconciler._drift_recovery_cache._states["default/test-deployment"] = DriftRecoveryState(attempts=3)

    # Mock backend to return UNKNOWN
    mock_backend = MagicMock()
    unknown_status = DeploymentStatusUpdate(
        status="UNKNOWN",
        status_message="Backend unavailable",
        host_url=None,
    )
    mock_backend.get_model_deployment_status = AsyncMock(return_value=unknown_status)
    mock_backend_registry.get_backend.return_value = mock_backend

    # Mock SDK
    reconciler._models_client.update_deployment_status = AsyncMock(return_value=_ModelResponse())

    # Process deployment
    await reconciler.reconcile_deployments([ctx])

    # Verify status was set to ERROR
    reconciler._models_client.update_deployment_status.assert_called_once()
    call_kwargs = _deployment_status_call(reconciler._models_client.update_deployment_status)
    assert call_kwargs["status"] == "ERROR"
    assert "Unable to communicate with backend after 3 attempts" in call_kwargs["status_message"]


@pytest.mark.asyncio
async def test_unknown_status_respects_backoff(reconciler, mock_backend_registry, make_deployment):
    """Test that UNKNOWN handling respects exponential backoff."""
    deployment = make_deployment(status="UNKNOWN")

    ctx = ModelContext(
        model_deployment=deployment,
        model_deployment_config=None,
        model_provider=None,
        model_entity=None,
    )

    # Set backoff config
    reconciler._controller_config.drift_recovery_base_delay_seconds = 60
    reconciler._drift_recovery_cache._base_delay_seconds = 60

    # Pre-populate state with recent attempt
    from nmp.core.models.controllers.deployment_reconciler import DriftRecoveryState

    reconciler._drift_recovery_cache._states["default/test-deployment"] = DriftRecoveryState(
        attempts=1,
        last_attempt_at=datetime.now(timezone.utc),  # Just now
    )

    # Mock backend to return UNKNOWN
    mock_backend = MagicMock()
    unknown_status = DeploymentStatusUpdate(
        status="UNKNOWN",
        status_message="Backend unavailable",
        host_url=None,
    )
    mock_backend.get_model_deployment_status = AsyncMock(return_value=unknown_status)
    mock_backend_registry.get_backend.return_value = mock_backend

    # Mock SDK
    reconciler._models_client.update_deployment_status = AsyncMock(return_value=_ModelResponse())

    # Process deployment
    await reconciler.reconcile_deployments([ctx])

    # Verify status was NOT updated (in backoff period)
    reconciler._models_client.update_deployment_status.assert_not_called()

    # Verify attempts was NOT incremented
    assert reconciler._drift_recovery_cache.get_attempts("default/test-deployment") == 1


@pytest.mark.asyncio
async def test_unknown_status_clears_on_recovery(reconciler, mock_backend_registry, make_deployment):
    """Test that recovery state is cleared when backend returns READY after UNKNOWN."""
    deployment = make_deployment(status="UNKNOWN")  # Was UNKNOWN, now checking again

    ctx = ModelContext(
        model_deployment=deployment,
        model_deployment_config=None,
        model_provider=None,
        model_entity=None,
    )

    # Pre-populate recovery state (from previous UNKNOWN attempts)
    from nmp.core.models.controllers.deployment_reconciler import DriftRecoveryState

    reconciler._drift_recovery_cache._states["default/test-deployment"] = DriftRecoveryState(attempts=2)

    # Mock backend to return READY (backend recovered)
    mock_backend = MagicMock()
    ready_status = DeploymentStatusUpdate(
        status="READY",
        status_message="Container is healthy",
        host_url="http://localhost:8500",
    )
    mock_backend.get_model_deployment_status = AsyncMock(return_value=ready_status)
    mock_backend_registry.get_backend.return_value = mock_backend

    # Mock SDK
    reconciler._models_client.update_deployment_status = AsyncMock(return_value=_ModelResponse())
    reconciler._models_client.get_provider = AsyncMock(side_effect=_status_error(404, "Not found"))
    reconciler._models_client.create_provider = AsyncMock()

    # Process deployment
    await reconciler.reconcile_deployments([ctx])

    # Verify recovery state was cleared
    assert "default/test-deployment" not in reconciler._drift_recovery_cache._states

    # Verify status was updated to READY
    call_kwargs = _deployment_status_call(reconciler._models_client.update_deployment_status)
    assert call_kwargs["status"] == "READY"


# ============================================================================
# Orphan reconciliation tests
# ============================================================================


@pytest.mark.asyncio
async def test_reconcile_orphans_empty_backend_list(reconciler, mock_backend_registry):
    """reconcile_orphans with backend returning empty list does not call delete."""
    mock_backend_registry.list_backends.return_value = ["docker"]
    mock_backend = MagicMock()
    mock_backend.list_managed_deployment_names = AsyncMock(return_value=[])
    mock_backend_registry.get_backend.return_value = mock_backend

    await reconciler.reconcile_orphans({"ws/a"})

    mock_backend.list_managed_deployment_names.assert_called_once()
    mock_backend.delete_model_deployment.assert_not_called()


@pytest.mark.asyncio
async def test_reconcile_orphans_no_orphans_when_all_known(reconciler, mock_backend_registry):
    """reconcile_orphans when backend list equals known set does not call delete."""
    mock_backend_registry.list_backends.return_value = ["docker"]
    mock_backend = MagicMock()
    mock_backend.list_managed_deployment_names = AsyncMock(return_value=["ws/a", "ws/b"])
    mock_backend_registry.get_backend.return_value = mock_backend

    await reconciler.reconcile_orphans({"ws/a", "ws/b"})

    mock_backend.delete_model_deployment.assert_not_called()


@pytest.mark.asyncio
async def test_reconcile_orphans_deletes_orphans(reconciler, mock_backend_registry):
    """reconcile_orphans calls delete_model_deployment(workspace, name) for backend names not in known set."""
    mock_backend_registry.list_backends.return_value = ["docker"]
    mock_backend = MagicMock()
    mock_backend.list_managed_deployment_names = AsyncMock(return_value=["ws/a", "ws/b", "ws/c"])
    mock_backend.delete_model_deployment = AsyncMock(
        return_value=DeploymentStatusUpdate(status="DELETED", status_message="")
    )
    mock_backend_registry.get_backend.return_value = mock_backend

    await reconciler.reconcile_orphans({"ws/a"})

    assert mock_backend.delete_model_deployment.call_count == 2
    calls = [c.args for c in mock_backend.delete_model_deployment.call_args_list]
    assert ("ws", "b") in calls
    assert ("ws", "c") in calls


@pytest.mark.asyncio
async def test_reconcile_orphans_continues_on_delete_failure(reconciler, mock_backend_registry):
    """reconcile_orphans continues to next orphan when delete_model_deployment raises."""
    mock_backend_registry.list_backends.return_value = ["docker"]
    mock_backend = MagicMock()
    mock_backend.list_managed_deployment_names = AsyncMock(return_value=["ws/orphan1", "ws/orphan2"])
    mock_backend_registry.get_backend.return_value = mock_backend
    mock_backend.delete_model_deployment = AsyncMock(
        side_effect=[Exception("delete failed"), DeploymentStatusUpdate(status="DELETED", status_message="")]
    )

    await reconciler.reconcile_orphans(set())

    assert mock_backend.delete_model_deployment.call_count == 2


@pytest.mark.asyncio
async def test_reconcile_orphans_skips_invalid_deployment_id(reconciler, mock_backend_registry):
    """reconcile_orphans skips backend names that do not contain '/' and does not call delete with invalid id."""
    mock_backend_registry.list_backends.return_value = ["docker"]
    mock_backend = MagicMock()
    mock_backend.list_managed_deployment_names = AsyncMock(return_value=["no-slash"])
    mock_backend_registry.get_backend.return_value = mock_backend

    await reconciler.reconcile_orphans(set())

    mock_backend.delete_model_deployment.assert_not_called()


# ============================================================================
# ERROR Deployment Garbage Collection Tests
# ============================================================================

ERROR_GC_TTL = 10800  # 3 hours — default from ControllerConfig


@pytest.fixture
def gc_reconciler(mock_models_sdk, mock_backend_registry):
    """Create a reconciler with default ERROR GC TTL for GC tests."""
    config = ControllerConfig()
    reconciler = ModelDeploymentReconciler(
        models_sdk=mock_models_sdk,
        backend_registry=mock_backend_registry,
        controller_config=config,
        entity_cache=ModelEntityCache(models_sdk=mock_models_sdk, emit_heartbeat=lambda: None),
        emit_heartbeat=lambda: None,
    )
    mock_backend = MagicMock()
    mock_backend.delete_model_deployment = AsyncMock(
        return_value=DeploymentStatusUpdate(status="DELETED", status_message="")
    )
    mock_backend_registry.get_backend.return_value = mock_backend
    reconciler._models_client.update_deployment_status = AsyncMock(return_value=_ModelResponse())
    reconciler._delete_model_provider = AsyncMock()
    return reconciler


_UNSET = object()


def _make_error_deployment(
    *,
    workspace="default",
    name="err-deploy",
    entity_version=1,
    updated_at=_UNSET,
    status_message="NIM health check failed",
    model_provider_id=None,
):
    """Helper to create a mock ERROR deployment with sensible defaults."""
    dep = MagicMock(spec=ModelDeployment)
    dep.workspace = workspace
    dep.name = name
    dep.entity_version = entity_version
    dep.status = "ERROR"
    dep.status_message = status_message
    dep.model_provider_id = model_provider_id
    dep.updated_at = (
        updated_at if updated_at is not _UNSET else (datetime.now(timezone.utc) - timedelta(seconds=ERROR_GC_TTL + 60))
    )
    return dep


@pytest.mark.asyncio
async def test_gc_triggers_after_ttl(gc_reconciler, mock_backend_registry):
    """ERROR deployment past TTL has backend resources deleted and status set to DELETING."""
    dep = _make_error_deployment(
        updated_at=datetime.now(timezone.utc) - timedelta(seconds=ERROR_GC_TTL + 100),
    )

    await gc_reconciler.gc_error_deployments([dep])

    mock_backend = mock_backend_registry.get_backend()
    mock_backend.delete_model_deployment.assert_called_once_with("default", "err-deploy")

    gc_reconciler._models_client.update_deployment_status.assert_called_once()
    call_kw = _deployment_status_call(gc_reconciler._models_client.update_deployment_status)
    assert call_kw["status"] == "DELETING"
    assert call_kw["name"] == "err-deploy"
    assert call_kw["workspace"] == "default"
    assert "garbage collected" in call_kw["status_message"]


@pytest.mark.asyncio
async def test_gc_handles_naive_datetime(gc_reconciler, mock_backend_registry):
    """Timezone-naive updated_at is treated as UTC."""
    naive_past = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(seconds=ERROR_GC_TTL + 60)
    dep = _make_error_deployment(updated_at=naive_past)

    await gc_reconciler.gc_error_deployments([dep])

    mock_backend = mock_backend_registry.get_backend()
    mock_backend.delete_model_deployment.assert_called_once()


@pytest.mark.asyncio
async def test_gc_skips_deployment_with_no_updated_at(gc_reconciler, mock_backend_registry):
    """Deployment with updated_at=None is skipped gracefully."""
    dep = _make_error_deployment(updated_at=None)

    await gc_reconciler.gc_error_deployments([dep])

    mock_backend = mock_backend_registry.get_backend()
    mock_backend.delete_model_deployment.assert_not_called()
    gc_reconciler._models_client.update_deployment_status.assert_not_called()


@pytest.mark.asyncio
async def test_gc_backend_delete_failure_still_transitions(gc_reconciler, mock_backend_registry):
    """If backend.delete raises, GC still transitions deployment to DELETING."""
    mock_backend = mock_backend_registry.get_backend()
    mock_backend.delete_model_deployment = AsyncMock(side_effect=Exception("container gone"))

    dep = _make_error_deployment()

    await gc_reconciler.gc_error_deployments([dep])

    gc_reconciler._models_client.update_deployment_status.assert_called_once()
    call_kw = _deployment_status_call(gc_reconciler._models_client.update_deployment_status)
    assert call_kw["status"] == "DELETING"


@pytest.mark.asyncio
async def test_gc_status_update_failure_does_not_block_others(gc_reconciler, mock_backend_registry):
    """If status update fails for one deployment, subsequent deployments are still processed."""
    dep1 = _make_error_deployment(name="dep-1")
    dep2 = _make_error_deployment(name="dep-2")

    gc_reconciler._models_client.update_deployment_status = AsyncMock(side_effect=[Exception("version conflict"), None])

    await gc_reconciler.gc_error_deployments([dep1, dep2])

    mock_backend = mock_backend_registry.get_backend()
    assert mock_backend.delete_model_deployment.call_count == 2


@pytest.mark.asyncio
async def test_gc_mixed_ttl_only_expired_cleaned(gc_reconciler, mock_backend_registry):
    """Only deployments past TTL are GC'd; recent ones are left alone."""
    expired = _make_error_deployment(
        name="old-deploy",
        updated_at=datetime.now(timezone.utc) - timedelta(seconds=ERROR_GC_TTL + 3600),
    )
    recent = _make_error_deployment(
        name="new-deploy",
        updated_at=datetime.now(timezone.utc) - timedelta(seconds=600),
    )

    await gc_reconciler.gc_error_deployments([expired, recent])

    mock_backend = mock_backend_registry.get_backend()
    mock_backend.delete_model_deployment.assert_called_once_with("default", "old-deploy")
    gc_reconciler._models_client.update_deployment_status.assert_called_once()


@pytest.mark.asyncio
async def test_gc_calls_delete_model_provider(gc_reconciler, mock_backend_registry):
    """GC cleans up the associated ModelProvider."""
    dep = _make_error_deployment(model_provider_id="default/my-provider")

    await gc_reconciler.gc_error_deployments([dep])

    gc_reconciler._delete_model_provider.assert_called_once_with(dep)


@pytest.mark.asyncio
async def test_gc_provider_cleanup_failure_is_non_fatal(gc_reconciler, mock_backend_registry):
    """Provider deletion failure does not prevent status transition to DELETING."""
    dep = _make_error_deployment(model_provider_id="default/my-provider")
    gc_reconciler._delete_model_provider = AsyncMock(side_effect=Exception("provider gone"))

    await gc_reconciler.gc_error_deployments([dep])

    mock_backend = mock_backend_registry.get_backend()
    mock_backend.delete_model_deployment.assert_called_once()

    gc_reconciler._models_client.update_deployment_status.assert_called_once()
    call_kw = _deployment_status_call(gc_reconciler._models_client.update_deployment_status)
    assert call_kw["status"] == "DELETING"
    assert "Provider cleanup failed" in call_kw["status_message"]


@pytest.mark.asyncio
async def test_gc_empty_list_no_ops(gc_reconciler, mock_backend_registry):
    """Empty deployment list results in no backend or SDK calls."""
    await gc_reconciler.gc_error_deployments([])

    mock_backend = mock_backend_registry.get_backend()
    mock_backend.delete_model_deployment.assert_not_called()
    gc_reconciler._models_client.update_deployment_status.assert_not_called()


@pytest.mark.asyncio
async def test_gc_custom_ttl_respected(mock_models_sdk, mock_backend_registry):
    """Non-default TTL value from config is respected."""
    custom_ttl = 3600  # 1 hour
    config = ControllerConfig(error_deployment_ttl_seconds=custom_ttl)
    reconciler = ModelDeploymentReconciler(
        models_sdk=mock_models_sdk,
        backend_registry=mock_backend_registry,
        controller_config=config,
        entity_cache=ModelEntityCache(models_sdk=mock_models_sdk, emit_heartbeat=lambda: None),
        emit_heartbeat=lambda: None,
    )

    mock_backend = MagicMock()
    mock_backend.delete_model_deployment = AsyncMock(
        return_value=DeploymentStatusUpdate(status="DELETED", status_message="")
    )
    mock_backend_registry.get_backend.return_value = mock_backend
    reconciler._models_client.update_deployment_status = AsyncMock(return_value=_ModelResponse())
    reconciler._delete_model_provider = AsyncMock()

    within_default_but_past_custom = _make_error_deployment(
        updated_at=datetime.now(timezone.utc) - timedelta(seconds=7200),
    )

    await reconciler.gc_error_deployments([within_default_but_past_custom])

    mock_backend.delete_model_deployment.assert_called_once()


@pytest.mark.asyncio
async def test_gc_status_message_includes_original_error(gc_reconciler, mock_backend_registry):
    """DELETING status message preserves the original error context."""
    dep = _make_error_deployment(status_message="NIM health check timed out after 7200s")

    await gc_reconciler.gc_error_deployments([dep])

    call_kw = _deployment_status_call(gc_reconciler._models_client.update_deployment_status)
    assert "garbage collected" in call_kw["status_message"]
    assert "NIM health check timed out after 7200s" in call_kw["status_message"]
    assert "Original error:" in call_kw["status_message"]


@pytest.mark.asyncio
async def test_gc_status_message_without_original_error(gc_reconciler, mock_backend_registry):
    """DELETING status message works when original status_message is empty."""
    dep = _make_error_deployment(status_message="")

    await gc_reconciler.gc_error_deployments([dep])

    call_kw = _deployment_status_call(gc_reconciler._models_client.update_deployment_status)
    assert "garbage collected" in call_kw["status_message"]
    assert "Original error:" not in call_kw["status_message"]


@pytest.mark.asyncio
async def test_gc_not_found_on_status_update_handled(gc_reconciler, mock_backend_registry):
    """NotFoundError on status update (deployment deleted between query and GC) is handled."""
    dep = _make_error_deployment()
    gc_reconciler._models_client.update_deployment_status = AsyncMock(side_effect=_status_error(404, "Not found"))

    # Should not raise
    await gc_reconciler.gc_error_deployments([dep])

    mock_backend = mock_backend_registry.get_backend()
    mock_backend.delete_model_deployment.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ttl,age_seconds,should_gc",
    [
        (10800, 10801, True),
        (10800, 10800, True),
        (10800, 10799, False),
        (3600, 3601, True),
        (3600, 3599, False),
        (0, 0, True),
    ],
    ids=[
        "just-past-default-ttl",
        "exactly-at-default-ttl",
        "just-before-default-ttl",
        "just-past-1h-ttl",
        "just-before-1h-ttl",
        "zero-ttl-immediate-gc",
    ],
)
async def test_gc_ttl_boundary_parametrized(mock_models_sdk, mock_backend_registry, ttl, age_seconds, should_gc):
    """Parametrized boundary tests for various TTL values and ages."""
    now = datetime.now(timezone.utc)
    config = ControllerConfig(error_deployment_ttl_seconds=ttl)
    reconciler = ModelDeploymentReconciler(
        models_sdk=mock_models_sdk,
        backend_registry=mock_backend_registry,
        controller_config=config,
        entity_cache=ModelEntityCache(models_sdk=mock_models_sdk, emit_heartbeat=lambda: None),
        emit_heartbeat=lambda: None,
    )

    mock_backend = MagicMock()
    mock_backend.delete_model_deployment = AsyncMock(
        return_value=DeploymentStatusUpdate(status="DELETED", status_message="")
    )
    mock_backend_registry.get_backend.return_value = mock_backend
    reconciler._models_client.update_deployment_status = AsyncMock(return_value=_ModelResponse())
    reconciler._delete_model_provider = AsyncMock()

    dep = _make_error_deployment(
        updated_at=now - timedelta(seconds=age_seconds),
    )

    with patch("nmp.core.models.controllers.deployment_reconciler.datetime") as mock_datetime:
        mock_datetime.now.return_value = now
        await reconciler.gc_error_deployments([dep])

    if should_gc:
        mock_backend.delete_model_deployment.assert_called_once()
        reconciler._models_client.update_deployment_status.assert_called_once()
    else:
        mock_backend.delete_model_deployment.assert_not_called()
        reconciler._models_client.update_deployment_status.assert_not_called()
