# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import TypeAlias
from unittest.mock import AsyncMock, Mock, patch

import pytest
from nemo_deployments_plugin.entities import Deployment, DeploymentConfig, Volume
from nemo_deployments_plugin.reconciler.volume_reconciler import VolumeReconciler
from nemo_deployments_plugin.types import Endpoint
from nemo_platform_plugin.auth import AuthContext as DeploymentAuthContext
from nemo_platform_plugin.entity_client import NemoEntityConflictError, NemoEntityNotFoundError
from nmp.common.config import Runtime
from nmp.core.models.app import ModelWeightsType
from nmp.core.models.controllers.backends.backends import DeploymentStatusUpdate
from nmp.core.models.controllers.backends.common import DeploymentConfigView
from nmp.core.models.controllers.backends.deployments_plugin.backend import DeploymentsPluginServiceBackend
from nmp.core.models.controllers.backends.deployments_plugin.config import DeploymentsPluginConfig
from nmp.core.models.controllers.backends.deployments_plugin.resolve import ResolvedPluginDeployment

CreatedEntity: TypeAlias = Deployment | DeploymentConfig


def _ctx() -> SimpleNamespace:
    return SimpleNamespace(
        model_deployment=SimpleNamespace(name="my-dep", workspace="default", status="CREATED"),
        model_deployment_config=SimpleNamespace(engine="vllm"),
        model_entity=None,
    )


def _resolved() -> ResolvedPluginDeployment:
    return ResolvedPluginDeployment(
        deployment=SimpleNamespace(name="my-dep", workspace="default", entity_version=1, auth_context=None),
        config=SimpleNamespace(engine="vllm"),
        model_entity=None,
        view=DeploymentConfigView(model_namespace="org", model_name="model"),
        weights_type=ModelWeightsType.FILES_SERVICE,
        model_namespace="org",
        model_name="model",
        model_revision=None,
        files_hf_url="http://files/hf",
        huggingface_model_puller="puller:latest",
        runtime=Runtime.KUBERNETES,
    )


@pytest.mark.asyncio
async def test_get_status_projects_ready_endpoint() -> None:
    backend = DeploymentsPluginServiceBackend(AsyncMock(), {}, "puller:latest")
    backend.init()
    backend._entities = AsyncMock()
    backend._entities.get = AsyncMock(
        side_effect=[
            Deployment(
                name="my-dep-server",
                workspace="default",
                deployment_config="my-dep-server",
                status="READY",
                endpoints=[Endpoint(name="http", url="http://server", protocol="http")],
            ),
            NemoEntityNotFoundError("missing"),
            NemoEntityNotFoundError("missing"),
        ]
    )
    result = await backend.get_model_deployment_status(
        SimpleNamespace(
            model_deployment=SimpleNamespace(
                name="my-dep",
                workspace="default",
                status="PENDING",
                created_at=datetime.now(timezone.utc),
            )
        )
    )
    assert result.status == "READY"
    assert result.host_url == "http://server"


@pytest.mark.asyncio
async def test_missing_ready_server_is_lost() -> None:
    backend = DeploymentsPluginServiceBackend(AsyncMock(), {}, "puller:latest")
    backend.init()
    backend._entities = AsyncMock()
    backend._entities.get = AsyncMock(side_effect=NemoEntityNotFoundError("missing"))
    result = await backend.get_model_deployment_status(
        SimpleNamespace(
            model_deployment=SimpleNamespace(
                name="my-dep",
                workspace="default",
                status="READY",
                created_at=datetime.now(timezone.utc),
            )
        )
    )
    assert result.status == "LOST"


@pytest.mark.asyncio
async def test_create_order_volume_puller_server_with_prerequisite() -> None:
    backend = DeploymentsPluginServiceBackend(AsyncMock(), {}, "puller:latest")
    backend.init()
    backend._entities = AsyncMock()
    created: list[CreatedEntity] = []

    async def _create(entity: CreatedEntity) -> CreatedEntity:
        created.append(entity)
        return entity

    backend._entities.create = AsyncMock(side_effect=_create)
    backend._entities.get = AsyncMock(side_effect=NemoEntityNotFoundError("missing"))
    backend._entities.delete = AsyncMock(side_effect=NemoEntityNotFoundError("missing"))
    with (
        patch(
            "nmp.core.models.controllers.backends.deployments_plugin.backend.resolve_plugin_deployment",
            return_value=_resolved(),
        ),
        patch(
            "nmp.core.models.controllers.backends.deployments_plugin.backend.executor_for_runtime",
            return_value="local-k8s",
        ),
    ):
        result = await backend.create_model_deployment(_ctx())

    assert result.status == "PENDING"
    assert [type(item) for item in created] == [Volume, DeploymentConfig, Deployment, DeploymentConfig, Deployment]
    puller_dep = created[2]
    server_dep = created[4]
    assert isinstance(puller_dep, Deployment) and puller_dep.name == "my-dep-puller"
    assert isinstance(server_dep, Deployment) and server_dep.name == "my-dep-server"
    assert server_dep.prerequisites[0].deployment_name == "my-dep-puller"
    assert server_dep.prerequisites[0].condition == "succeeded"


def _resolved_docker_lora() -> ResolvedPluginDeployment:
    return ResolvedPluginDeployment(
        deployment=SimpleNamespace(name="my-dep", workspace="default", entity_version=1, auth_context=None),
        config=SimpleNamespace(engine="vllm"),
        model_entity=None,
        view=DeploymentConfigView(model_namespace="org", model_name="model", lora_enabled=True, gpu=1),
        weights_type=ModelWeightsType.FILES_SERVICE,
        model_namespace="org",
        model_name="model",
        model_revision=None,
        files_hf_url="http://files/hf",
        huggingface_model_puller="puller:latest",
        runtime=Runtime.DOCKER,
    )


@pytest.mark.asyncio
async def test_docker_lora_creates_substrate() -> None:
    """Docker + LoRA is now supported: it creates substrate like any other deploy.

    The docker backend runs the LoRA shape as a multi-container group (server +
    adapters sidecar) so there is no longer a fast-fail guardrail.
    """
    backend = DeploymentsPluginServiceBackend(AsyncMock(), {}, "puller:latest")
    backend.init()
    backend._entities = AsyncMock()
    created: list[CreatedEntity] = []

    async def _create(entity: CreatedEntity) -> CreatedEntity:
        created.append(entity)
        return entity

    backend._entities.create = AsyncMock(side_effect=_create)
    backend._entities.get = AsyncMock(side_effect=NemoEntityNotFoundError("missing"))
    backend._entities.delete = AsyncMock(side_effect=NemoEntityNotFoundError("missing"))
    with (
        patch(
            "nmp.core.models.controllers.backends.deployments_plugin.backend.resolve_plugin_deployment",
            return_value=_resolved_docker_lora(),
        ),
        patch(
            "nmp.core.models.controllers.backends.deployments_plugin.backend.executor_for_runtime",
            return_value="local-docker",
        ),
    ):
        result = await backend.create_model_deployment(_ctx())

    assert result.status == "PENDING"
    # Substrate is created (volume(s) + puller + server configs/deployments),
    # not rejected. The server config carries the multi-container LoRA shape.
    assert any(isinstance(item, Deployment) and item.name == "my-dep-server" for item in created)

    # Assert the full multi-container LoRA contract on the server DeploymentConfig,
    # so a regression that drops the adapters sidecar / init / its port/GPU
    # settings fails here rather than silently passing.
    server_config = next(
        item for item in created if isinstance(item, DeploymentConfig) and item.name.endswith("-server")
    )
    container_names = [c.name for c in server_config.containers]
    assert container_names == ["server", "lora-adapters"]

    server_container = server_config.containers[0]
    sidecar_container = server_config.containers[1]

    # The adapters sidecar publishes no ports and requests no GPU (it shares the
    # server's network namespace and GPU); only the server owns those.
    assert not sidecar_container.ports
    assert not sidecar_container.resources.limits.get("nvidia.com/gpu")
    assert server_container.ports  # server exposes the inference port
    assert server_container.resources.limits.get("nvidia.com/gpu") == "1"

    # The lora-cache-init init container prepares the shared scratch volume.
    init_names = [c.name for c in server_config.init_containers]
    assert "lora-cache-init" in init_names


@pytest.mark.asyncio
async def test_create_retries_after_prior_teardown_completes() -> None:
    backend = DeploymentsPluginServiceBackend(AsyncMock(), {}, "puller:latest")
    backend.init()
    backend._entities = AsyncMock()
    backend._entities.create = AsyncMock(side_effect=lambda entity: entity)
    with (
        patch(
            "nmp.core.models.controllers.backends.deployments_plugin.backend.resolve_plugin_deployment",
            return_value=_resolved(),
        ),
        patch.object(
            backend,
            "delete_model_deployment",
            AsyncMock(
                side_effect=[
                    DeploymentStatusUpdate(status="DELETING", status_message="waiting"),
                    DeploymentStatusUpdate(status="DELETED", status_message="deleted"),
                ]
            ),
        ),
        patch(
            "nmp.core.models.controllers.backends.deployments_plugin.backend.executor_for_runtime",
            return_value="local-k8s",
        ),
    ):
        waiting = await backend.create_model_deployment(_ctx())
        backend._entities.create.assert_not_called()
        created = await backend.create_model_deployment(_ctx())

    assert waiting.status == "CREATED"
    assert "teardown" in waiting.status_message.lower()
    assert created.status == "PENDING"
    assert backend._entities.create.await_count == 5


@pytest.mark.asyncio
async def test_missing_executor_fails_fast_before_touching_substrate() -> None:
    backend = DeploymentsPluginServiceBackend(AsyncMock(), {}, "puller:latest")
    backend.init()
    backend._entities = AsyncMock()
    backend._entities.get = AsyncMock(side_effect=NemoEntityNotFoundError("missing"))
    backend._entities.delete = AsyncMock(side_effect=NemoEntityNotFoundError("missing"))
    with (
        patch(
            "nmp.core.models.controllers.backends.deployments_plugin.backend.resolve_plugin_deployment",
            return_value=_resolved(),
        ),
        patch(
            "nmp.core.models.controllers.backends.deployments_plugin.backend.executor_for_runtime",
            return_value=None,
        ),
    ):
        result = await backend.create_model_deployment(_ctx())
    assert result.status == "ERROR"
    assert "executor" in result.status_message.lower()
    assert result.error_details is not None
    assert result.error_details["reason"] == "executor_not_configured"
    backend._entities.create.assert_not_called()


@pytest.mark.asyncio
async def test_pending_timeout_escalates_stuck_deployment() -> None:
    backend = DeploymentsPluginServiceBackend(AsyncMock(), {}, "puller:latest")
    backend.init()
    backend._backend_config = DeploymentsPluginConfig(pending_timeout_seconds=60)
    backend._entities = AsyncMock()
    server = Deployment(
        name="my-dep-server",
        workspace="default",
        deployment_config="my-dep-server",
        status="STARTING",
    )
    backend._entities.get = AsyncMock(
        side_effect=[
            server,
            NemoEntityNotFoundError("missing"),
            NemoEntityNotFoundError("missing"),
        ]
    )
    created_at = datetime.now(timezone.utc) - timedelta(seconds=120)
    result = await backend.get_model_deployment_status(
        SimpleNamespace(
            model_deployment=SimpleNamespace(
                name="my-dep",
                workspace="default",
                status="PENDING",
                created_at=created_at,
            )
        )
    )
    assert result.status == "ERROR"
    assert result.error_details is not None
    assert result.error_details["reason"] == "pending_timeout"
    assert result.error_details["timeout_seconds"] == 60


@pytest.mark.asyncio
async def test_delete_returns_deleting_when_server_still_exists() -> None:
    backend = DeploymentsPluginServiceBackend(AsyncMock(), {}, "puller:latest")
    backend.init()
    backend._entities = AsyncMock()
    server = Deployment(
        name="my-dep-server",
        workspace="default",
        deployment_config="my-dep-server",
        status="READY",
    )

    async def _get(entity_type: type, name: str, workspace: str | None = None) -> Deployment:
        del entity_type, workspace
        if name == "my-dep-server":
            return server
        raise NemoEntityNotFoundError("missing")

    backend._entities.get = AsyncMock(side_effect=_get)
    backend._entities.update = AsyncMock(side_effect=lambda entity: entity)
    backend._entities.delete = AsyncMock()

    result = await backend.delete_model_deployment("default", "my-dep")
    assert result.status == "DELETING"
    backend._entities.delete.assert_not_called()
    backend._entities.update.assert_awaited_once()
    assert server.status == "DELETING"
    assert server.desired_state == "STOPPED"


@pytest.mark.asyncio
async def test_delete_returns_deleting_without_blocking_poll() -> None:
    backend = DeploymentsPluginServiceBackend(AsyncMock(), {}, "puller:latest")
    backend.init()
    backend._entities = AsyncMock()
    server = Deployment(
        name="my-dep-server",
        workspace="default",
        deployment_config="my-dep-server",
        status="READY",
    )
    backend._entities.get = AsyncMock(return_value=server)
    backend._entities.update = AsyncMock(side_effect=lambda entity: entity)
    backend._entities.delete = AsyncMock()

    with patch("asyncio.sleep", AsyncMock()) as sleep_mock:
        result = await backend.delete_model_deployment("default", "my-dep")

    assert result.status == "DELETING"
    sleep_mock.assert_not_called()
    backend._entities.delete.assert_not_called()


@pytest.mark.asyncio
async def test_delete_retries_on_next_call_when_deployment_still_exists() -> None:
    backend = DeploymentsPluginServiceBackend(AsyncMock(), {}, "puller:latest")
    backend.init()
    backend._entities = AsyncMock()
    server = Deployment(
        name="my-dep-server",
        workspace="default",
        deployment_config="my-dep-server",
        status="DELETING",
        desired_state="STOPPED",
    )
    seen_server = False

    async def _get(entity_type: type, name: str, workspace: str | None = None) -> Deployment:
        del entity_type, workspace
        nonlocal seen_server
        if name == "my-dep-server":
            if not seen_server:
                seen_server = True
                return server
            raise NemoEntityNotFoundError("missing")
        raise NemoEntityNotFoundError("missing")

    backend._entities.get = AsyncMock(side_effect=_get)
    backend._entities.update = AsyncMock(side_effect=lambda entity: entity)
    backend._entities.delete = AsyncMock()

    first = await backend.delete_model_deployment("default", "my-dep")
    assert first.status == "DELETING"

    second = await backend.delete_model_deployment("default", "my-dep")
    assert second.status == "DELETED"


@pytest.mark.asyncio
async def test_delete_marks_weights_and_scratch_volumes_deleting_and_waits() -> None:
    backend = DeploymentsPluginServiceBackend(AsyncMock(), {}, "puller:latest")
    backend.init()
    backend._entities = AsyncMock()
    volumes = {
        "my-dep-weights": Volume(name="my-dep-weights", workspace="default", size="50Gi", status="BOUND"),
        "my-dep-scratch": Volume(name="my-dep-scratch", workspace="default", size="10Gi", status="BOUND"),
    }

    async def _get(entity_type: type, name: str, workspace: str | None = None) -> Volume:
        del workspace
        if entity_type is Volume and name in volumes:
            return volumes[name]
        raise NemoEntityNotFoundError("missing")

    backend._entities.get = AsyncMock(side_effect=_get)
    backend._entities.update = AsyncMock(side_effect=lambda entity: entity)
    backend._entities.delete = AsyncMock()
    with patch(
        "nmp.core.models.controllers.backends.deployments_plugin.backend.deployment_config_names_referencing_volume",
        AsyncMock(return_value=[]),
    ):
        result = await backend.delete_model_deployment("default", "my-dep")

    assert result.status == "DELETING"
    assert {volume.status for volume in volumes.values()} == {"DELETING"}
    assert {call.args[0].name for call in backend._entities.update.await_args_list} == {
        "my-dep-scratch",
        "my-dep-weights",
    }
    assert all(call.args[0] is not Volume for call in backend._entities.delete.await_args_list)

    substrate_backend = AsyncMock()
    registry = AsyncMock()
    registry.resolve = Mock(return_value=substrate_backend)
    volume_reconciler = VolumeReconciler(backend._entities, registry)
    weights = volumes["my-dep-weights"]

    await volume_reconciler.reconcile_one(weights)

    substrate_backend.delete_volume.assert_awaited_once_with(
        "default",
        "my-dep-weights",
        backend_config=weights.backend_config.model_dump(by_alias=True, exclude_none=True),
    )
    backend._entities.delete.assert_any_await(
        Volume,
        name="my-dep-weights",
        workspace="default",
        expected_db_version=weights.db_version,
    )


@pytest.mark.asyncio
async def test_delete_preserves_volume_referenced_by_another_config() -> None:
    backend = DeploymentsPluginServiceBackend(AsyncMock(), {}, "puller:latest")
    backend.init()
    backend._entities = AsyncMock()
    weights = Volume(name="my-dep-weights", workspace="default", size="50Gi", status="BOUND")

    async def _get(entity_type: type, name: str, workspace: str | None = None) -> Volume:
        del workspace
        if entity_type is Volume and name == weights.name:
            return weights
        raise NemoEntityNotFoundError("missing")

    backend._entities.get = AsyncMock(side_effect=_get)
    backend._entities.delete = AsyncMock()
    with patch(
        "nmp.core.models.controllers.backends.deployments_plugin.backend.deployment_config_names_referencing_volume",
        AsyncMock(return_value=["shared-config"]),
    ):
        result = await backend.delete_model_deployment("default", "my-dep")

    assert result.status == "DELETED"
    assert weights.status == "BOUND"
    backend._entities.update.assert_not_awaited()
    assert all(call.args[0] is not Volume for call in backend._entities.delete.await_args_list)


@pytest.mark.asyncio
async def test_delete_already_deleting_volume_skips_reference_scan() -> None:
    backend = DeploymentsPluginServiceBackend(AsyncMock(), {}, "puller:latest")
    backend.init()
    backend._entities = AsyncMock()
    volume = Volume(name="my-dep-weights", workspace="default", size="50Gi", status="DELETING")
    backend._entities.get = AsyncMock(return_value=volume)

    with patch(
        "nmp.core.models.controllers.backends.deployments_plugin.backend.deployment_config_names_referencing_volume",
        AsyncMock(),
    ) as references:
        removed = await backend._complete_volume_delete("default", volume.name)

    assert not removed
    references.assert_not_awaited()
    backend._entities.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_continues_with_weights_when_scratch_teardown_fails() -> None:
    backend = DeploymentsPluginServiceBackend(AsyncMock(), {}, "puller:latest")
    backend.init()
    backend._entities = AsyncMock()

    async def _complete_volume_delete(workspace: str, volume_name: str) -> bool:
        del workspace
        if volume_name == "my-dep-scratch":
            raise RuntimeError("scratch teardown failed")
        return True

    with (
        patch.object(backend, "_complete_deployment_delete", AsyncMock(return_value=True)),
        patch.object(
            backend, "_complete_volume_delete", AsyncMock(side_effect=_complete_volume_delete)
        ) as delete_volume,
    ):
        result = await backend.delete_model_deployment("default", "my-dep")

    assert result.status == "DELETING"
    assert [call.args[1] for call in delete_volume.await_args_list] == ["my-dep-scratch", "my-dep-weights"]


@pytest.mark.asyncio
async def test_volume_delete_update_not_found_is_removed() -> None:
    backend = DeploymentsPluginServiceBackend(AsyncMock(), {}, "puller:latest")
    backend.init()
    backend._entities = AsyncMock()
    volume = Volume(name="my-dep-weights", workspace="default", size="50Gi", status="BOUND")
    backend._entities.get = AsyncMock(return_value=volume)
    backend._entities.update = AsyncMock(side_effect=NemoEntityNotFoundError("removed"))

    with patch(
        "nmp.core.models.controllers.backends.deployments_plugin.backend.deployment_config_names_referencing_volume",
        AsyncMock(return_value=[]),
    ):
        removed = await backend._complete_volume_delete("default", volume.name)

    assert removed


@pytest.mark.asyncio
async def test_volume_delete_update_conflict_remains_deleting() -> None:
    backend = DeploymentsPluginServiceBackend(AsyncMock(), {}, "puller:latest")
    backend.init()
    backend._entities = AsyncMock()
    volume = Volume(name="my-dep-weights", workspace="default", size="50Gi", status="BOUND")
    backend._entities.get = AsyncMock(return_value=volume)
    backend._entities.update = AsyncMock(side_effect=NemoEntityConflictError("changed"))

    with patch(
        "nmp.core.models.controllers.backends.deployments_plugin.backend.deployment_config_names_referencing_volume",
        AsyncMock(return_value=[]),
    ):
        removed = await backend._complete_volume_delete("default", volume.name)

    assert not removed
    assert volume.status == "DELETING"


@pytest.mark.asyncio
async def test_delete_completes_when_plugin_deployment_failed() -> None:
    backend = DeploymentsPluginServiceBackend(AsyncMock(), {}, "puller:latest")
    backend.init()
    backend._entities = AsyncMock()
    server = Deployment(
        name="my-dep-server",
        workspace="default",
        deployment_config="my-dep-server",
        status="FAILED",
        desired_state="STOPPED",
    )

    async def _get(entity_type: type, name: str, workspace: str | None = None) -> Deployment:
        del workspace
        if entity_type is Deployment and name == server.name:
            return server
        raise NemoEntityNotFoundError("missing")

    backend._entities.get = AsyncMock(side_effect=_get)
    backend._entities.delete = AsyncMock()

    result = await backend.delete_model_deployment("default", "my-dep")

    assert result.status == "DELETED"
    backend._entities.delete.assert_any_await(
        Deployment,
        name="my-dep-server",
        workspace="default",
        expected_db_version=server.db_version,
    )


@pytest.mark.asyncio
async def test_delete_escalates_to_error_after_deleting_timeout() -> None:
    backend = DeploymentsPluginServiceBackend(AsyncMock(), {}, "puller:latest")
    backend.init()
    backend._backend_config = DeploymentsPluginConfig(deleting_timeout_seconds=60)
    backend._entities = AsyncMock()
    server = Deployment(
        name="my-dep-server",
        workspace="default",
        deployment_config="my-dep-server",
        status="DELETING",
    )
    backend._entities.get = AsyncMock(return_value=server)
    backend._entities.update = AsyncMock(side_effect=lambda entity: entity)

    result = await backend.delete_model_deployment("default", "my-dep", deleting_elapsed_seconds=120)
    assert result.status == "ERROR"
    assert result.error_details is not None
    assert result.error_details["reason"] == "deleting_timeout"
    assert result.error_details["timeout_seconds"] == 60


@pytest.mark.asyncio
async def test_delete_escalates_to_error_after_volume_teardown_timeout() -> None:
    backend = DeploymentsPluginServiceBackend(AsyncMock(), {}, "puller:latest")
    backend.init()
    backend._backend_config = DeploymentsPluginConfig(deleting_timeout_seconds=60)
    backend._entities = AsyncMock()

    with (
        patch.object(backend, "_complete_deployment_delete", AsyncMock(return_value=True)),
        patch.object(backend, "_complete_volume_delete", AsyncMock(return_value=False)),
    ):
        result = await backend.delete_model_deployment("default", "my-dep", deleting_elapsed_seconds=120)

    assert result.status == "ERROR"
    assert result.error_details is not None
    assert result.error_details["reason"] == "deleting_timeout"
    assert result.error_details["timeout_seconds"] == 60


@pytest.mark.asyncio
async def test_create_propagates_deployment_auth_context_to_plugin_deployments() -> None:
    auth_context = DeploymentAuthContext(principal_id="user:alice", principal_groups=["research"])
    resolved = _resolved()
    resolved.deployment.auth_context = auth_context
    backend = DeploymentsPluginServiceBackend(AsyncMock(), {}, "puller:latest")
    backend.init()
    backend._entities = AsyncMock()
    created: list[CreatedEntity] = []

    async def _create(entity: CreatedEntity) -> CreatedEntity:
        created.append(entity)
        return entity

    backend._entities.create = AsyncMock(side_effect=_create)
    backend._entities.get = AsyncMock(side_effect=NemoEntityNotFoundError("missing"))
    backend._entities.delete = AsyncMock(side_effect=NemoEntityNotFoundError("missing"))
    with (
        patch(
            "nmp.core.models.controllers.backends.deployments_plugin.backend.resolve_plugin_deployment",
            return_value=resolved,
        ),
        patch(
            "nmp.core.models.controllers.backends.deployments_plugin.backend.executor_for_runtime",
            return_value="local-k8s",
        ),
    ):
        result = await backend.create_model_deployment(_ctx())

    assert result.status == "PENDING"
    plugin_deployments = [entity for entity in created if isinstance(entity, Deployment)]
    assert len(plugin_deployments) == 2
    assert all(deployment.auth_context == auth_context for deployment in plugin_deployments)
