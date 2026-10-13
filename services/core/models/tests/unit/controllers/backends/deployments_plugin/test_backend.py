# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from nemo_helix_plugin.auth import AuthContext as DeploymentAuthContext
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.deployments.types import (
    CreateDeploymentConfigRequest,
    CreateDeploymentRequest,
    CreateVolumeRequest,
    Endpoint,
)
from nemo_helix_plugin.deployments.types import (
    Deployment as DeploymentDTO,
)
from nemo_helix_plugin.deployments.types import (
    DeploymentConfig as DeploymentConfigDTO,
)
from nemo_helix_plugin.deployments.types import (
    Volume as VolumeDTO,
)
from nhx.common.config import Runtime
from nhx.core.models.app import ModelWeightsType
from nhx.core.models.controllers.backends.backends import DeploymentStatusUpdate
from nhx.core.models.controllers.backends.common import DeploymentConfigView
from nhx.core.models.controllers.backends.deployments_plugin.backend import DeploymentsPluginServiceBackend
from nhx.core.models.controllers.backends.deployments_plugin.config import DeploymentsPluginConfig
from nhx.core.models.controllers.backends.deployments_plugin.resolve import ResolvedPluginDeployment


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


# ---------------------------------------------------------------------------
# Deployments-client test harness
#
# The models backend is a pure HTTP consumer of the deployments plugin, so tests
# mock the AsyncDeploymentsClient. Single reads return an envelope whose .data()
# yields the DTO (or the get raises NotFoundError); lists return an envelope whose
# .items() is an async iterator. with_headers() returns the same client so the
# on-behalf-of create chain works.
# ---------------------------------------------------------------------------


def _envelope(data: object) -> Mock:
    env = Mock()
    env.data = Mock(return_value=data)
    return env


def _list_envelope(items: list[object]) -> Mock:
    async def _aiter() -> object:
        for item in items:
            yield item

    env = Mock()
    env.items = Mock(return_value=_aiter())
    return env


def _client_mock() -> AsyncMock:
    client = AsyncMock()
    client.with_headers = Mock(return_value=client)
    return client


def _install_empty_teardown(client: AsyncMock) -> None:
    """Make the pre-create teardown a no-op: nothing exists, so delete reports DELETED."""
    get_dep, get_vol = _get_router({}, {})
    client.get_deployment = AsyncMock(side_effect=get_dep)
    client.get_volume = AsyncMock(side_effect=get_vol)
    client.list_deployment_configs = AsyncMock(return_value=_list_envelope([]))


def _backend_with_client(
    client: AsyncMock, *, config: DeploymentsPluginConfig | None = None
) -> DeploymentsPluginServiceBackend:
    backend = DeploymentsPluginServiceBackend(AsyncMock(), {}, "puller:latest")
    backend.init()
    if config is not None:
        backend._backend_config = config
    backend._deployments = client
    return backend


def _get_router(deployments: dict[str, object], volumes: dict[str, object]):
    """A get_deployment/get_volume side_effect over name->DTO maps (404 when absent)."""

    async def _get_deployment(*, name: str, workspace: str | None = None) -> Mock:
        del workspace
        if name in deployments:
            return _envelope(deployments[name])
        raise NotFoundError.__new__(NotFoundError)

    async def _get_volume(*, name: str, workspace: str | None = None) -> Mock:
        del workspace
        if name in volumes:
            return _envelope(volumes[name])
        raise NotFoundError.__new__(NotFoundError)

    return _get_deployment, _get_volume


@pytest.mark.asyncio
async def test_get_status_projects_ready_endpoint() -> None:
    client = _client_mock()
    get_dep, get_vol = _get_router(
        {
            "my-dep-server": DeploymentDTO(
                name="my-dep-server",
                workspace="default",
                deployment_config="my-dep-server",
                status="READY",
                endpoints=[Endpoint(name="http", url="http://server", protocol="http")],
            )
        },
        {},
    )
    client.get_deployment = AsyncMock(side_effect=get_dep)
    client.get_volume = AsyncMock(side_effect=get_vol)
    backend = _backend_with_client(client)

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
    client = _client_mock()
    get_dep, get_vol = _get_router({}, {})
    client.get_deployment = AsyncMock(side_effect=get_dep)
    client.get_volume = AsyncMock(side_effect=get_vol)
    backend = _backend_with_client(client)

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
    client = _client_mock()
    order: list[tuple[str, CreateDeploymentRequest | CreateDeploymentConfigRequest | CreateVolumeRequest]] = []
    client.create_volume = AsyncMock(side_effect=lambda **kw: order.append(("volume", kw["body"])))
    client.create_deployment_config = AsyncMock(side_effect=lambda **kw: order.append(("config", kw["body"])))
    client.create_deployment = AsyncMock(side_effect=lambda **kw: order.append(("deployment", kw["body"])))
    _install_empty_teardown(client)
    backend = _backend_with_client(client)

    with (
        patch(
            "nhx.core.models.controllers.backends.deployments_plugin.backend.resolve_plugin_deployment",
            return_value=_resolved(),
        ),
        patch(
            "nhx.core.models.controllers.backends.deployments_plugin.backend.executor_for_runtime",
            return_value="local-k8s",
        ),
    ):
        result = await backend.create_model_deployment(_ctx())

    assert result.status == "PENDING"
    assert [kind for kind, _ in order] == ["volume", "config", "deployment", "config", "deployment"]
    puller_dep = order[2][1]
    server_dep = order[4][1]
    assert isinstance(puller_dep, CreateDeploymentRequest)
    assert isinstance(server_dep, CreateDeploymentRequest)
    assert puller_dep.name == "my-dep-puller"
    assert server_dep.name == "my-dep-server"
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
    """Docker + LoRA is supported: it creates substrate like any other deploy.

    The docker backend runs the LoRA shape as a multi-container group (server +
    adapters sidecar) so there is no fast-fail guardrail.
    """
    client = _client_mock()
    configs: list[CreateDeploymentConfigRequest] = []
    client.create_volume = AsyncMock()
    client.create_deployment_config = AsyncMock(side_effect=lambda **kw: configs.append(kw["body"]))
    client.create_deployment = AsyncMock()
    _install_empty_teardown(client)
    backend = _backend_with_client(client)

    with (
        patch(
            "nhx.core.models.controllers.backends.deployments_plugin.backend.resolve_plugin_deployment",
            return_value=_resolved_docker_lora(),
        ),
        patch(
            "nhx.core.models.controllers.backends.deployments_plugin.backend.executor_for_runtime",
            return_value="local-docker",
        ),
    ):
        result = await backend.create_model_deployment(_ctx())

    assert result.status == "PENDING"
    # Assert the full multi-container LoRA contract on the server DeploymentConfig,
    # so a regression that drops the adapters sidecar / init / its port/GPU
    # settings fails here rather than silently passing.
    server_config = next(cfg for cfg in configs if cfg.name.endswith("-server"))
    assert [c.name for c in server_config.containers] == ["server", "lora-adapters"]

    server_container, sidecar_container = server_config.containers
    # The adapters sidecar publishes no ports and requests no GPU (it shares the
    # server's network namespace and GPU); only the server owns those.
    assert not sidecar_container.ports
    assert not sidecar_container.resources.limits.get("nvidia.com/gpu")
    assert server_container.ports  # server exposes the inference port
    assert server_container.resources.limits.get("nvidia.com/gpu") == "1"
    assert "lora-cache-init" in [c.name for c in server_config.init_containers]


@pytest.mark.asyncio
async def test_create_retries_after_prior_teardown_completes() -> None:
    client = _client_mock()
    client.create_volume = AsyncMock()
    client.create_deployment_config = AsyncMock()
    client.create_deployment = AsyncMock()
    backend = _backend_with_client(client)

    with (
        patch(
            "nhx.core.models.controllers.backends.deployments_plugin.backend.resolve_plugin_deployment",
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
            "nhx.core.models.controllers.backends.deployments_plugin.backend.executor_for_runtime",
            return_value="local-k8s",
        ),
    ):
        waiting = await backend.create_model_deployment(_ctx())
        client.create_deployment.assert_not_called()
        created = await backend.create_model_deployment(_ctx())

    assert waiting.status == "CREATED"
    assert "teardown" in waiting.status_message.lower()
    assert created.status == "PENDING"
    # 1 weights volume + puller config + puller deployment + server config + server deployment.
    assert client.create_volume.await_count == 1
    assert client.create_deployment_config.await_count == 2
    assert client.create_deployment.await_count == 2


@pytest.mark.asyncio
async def test_missing_executor_fails_fast_before_touching_substrate() -> None:
    client = _client_mock()
    client.create_volume = AsyncMock()
    client.create_deployment_config = AsyncMock()
    client.create_deployment = AsyncMock()
    _install_empty_teardown(client)
    backend = _backend_with_client(client)

    with (
        patch(
            "nhx.core.models.controllers.backends.deployments_plugin.backend.resolve_plugin_deployment",
            return_value=_resolved(),
        ),
        patch(
            "nhx.core.models.controllers.backends.deployments_plugin.backend.executor_for_runtime",
            return_value=None,
        ),
    ):
        result = await backend.create_model_deployment(_ctx())
    assert result.status == "ERROR"
    assert "executor" in result.status_message.lower()
    assert result.error_details is not None
    assert result.error_details["reason"] == "executor_not_configured"
    client.create_volume.assert_not_called()
    client.create_deployment.assert_not_called()


@pytest.mark.asyncio
async def test_pending_timeout_escalates_stuck_deployment() -> None:
    client = _client_mock()
    get_dep, get_vol = _get_router(
        {
            "my-dep-server": DeploymentDTO(
                name="my-dep-server",
                workspace="default",
                deployment_config="my-dep-server",
                status="STARTING",
            )
        },
        {},
    )
    client.get_deployment = AsyncMock(side_effect=get_dep)
    client.get_volume = AsyncMock(side_effect=get_vol)
    backend = _backend_with_client(client, config=DeploymentsPluginConfig(pending_timeout_seconds=60))

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
    client = _client_mock()
    server = DeploymentDTO(name="my-dep-server", workspace="default", deployment_config="my-dep-server", status="READY")
    get_dep, get_vol = _get_router({"my-dep-server": server}, {})
    client.get_deployment = AsyncMock(side_effect=get_dep)
    client.get_volume = AsyncMock(side_effect=get_vol)
    backend = _backend_with_client(client)

    result = await backend.delete_model_deployment("default", "my-dep")

    assert result.status == "DELETING"
    # The server deployment is still present, so we issue a (soft) delete against it
    # and do NOT remove its config or reach the volumes yet.
    client.delete_deployment.assert_any_await(name="my-dep-server", workspace="default")
    client.delete_deployment_config.assert_not_called()
    client.delete_volume.assert_not_called()


@pytest.mark.asyncio
async def test_delete_returns_deleting_without_blocking_poll() -> None:
    client = _client_mock()
    server = DeploymentDTO(name="my-dep-server", workspace="default", deployment_config="my-dep-server", status="READY")
    get_dep, get_vol = _get_router({"my-dep-server": server}, {})
    client.get_deployment = AsyncMock(side_effect=get_dep)
    client.get_volume = AsyncMock(side_effect=get_vol)
    backend = _backend_with_client(client)

    with patch("asyncio.sleep", AsyncMock()) as sleep_mock:
        result = await backend.delete_model_deployment("default", "my-dep")

    assert result.status == "DELETING"
    sleep_mock.assert_not_called()


@pytest.mark.asyncio
async def test_delete_config_already_absent_still_completes() -> None:
    # No deployments/volumes remain, but the DeploymentConfig delete 404s because
    # it was already removed (repeat pass, or a puller-less deploy). The 404 must
    # be swallowed so delete still reaches DELETED rather than crashing.
    client = _client_mock()
    get_dep, get_vol = _get_router({}, {})
    client.get_deployment = AsyncMock(side_effect=get_dep)
    client.get_volume = AsyncMock(side_effect=get_vol)
    client.delete_deployment_config = AsyncMock(side_effect=NotFoundError.__new__(NotFoundError))
    backend = _backend_with_client(client)

    result = await backend.delete_model_deployment("default", "my-dep")

    assert result.status == "DELETED"
    client.delete_deployment_config.assert_awaited()


@pytest.mark.asyncio
async def test_delete_deployment_already_absent_is_idempotent() -> None:
    # The deployment GET returns it (present), but the DELETE races and 404s
    # because it vanished in between. The 404 must be swallowed; the pass still
    # reports DELETING (the entity was present when observed).
    client = _client_mock()
    server = DeploymentDTO(
        name="my-dep-server", workspace="default", deployment_config="my-dep-server", status="DELETING"
    )
    get_dep, get_vol = _get_router({"my-dep-server": server}, {})
    client.get_deployment = AsyncMock(side_effect=get_dep)
    client.get_volume = AsyncMock(side_effect=get_vol)
    client.delete_deployment = AsyncMock(side_effect=NotFoundError.__new__(NotFoundError))
    backend = _backend_with_client(client)

    result = await backend.delete_model_deployment("default", "my-dep")

    assert result.status == "DELETING"
    client.delete_deployment.assert_awaited()


@pytest.mark.asyncio
async def test_delete_retries_on_next_call_when_deployment_still_exists() -> None:
    client = _client_mock()
    server = DeploymentDTO(
        name="my-dep-server",
        workspace="default",
        deployment_config="my-dep-server",
        status="DELETING",
        desired_state="STOPPED",
    )
    present = {"my-dep-server": server}

    async def _get_deployment(*, name: str, workspace: str | None = None) -> Mock:
        del workspace
        if name in present:
            return _envelope(present[name])
        raise NotFoundError.__new__(NotFoundError)

    async def _get_volume(*, name: str, workspace: str | None = None) -> Mock:
        del name, workspace
        raise NotFoundError.__new__(NotFoundError)

    client.get_deployment = AsyncMock(side_effect=_get_deployment)
    client.get_volume = AsyncMock(side_effect=_get_volume)
    backend = _backend_with_client(client)

    first = await backend.delete_model_deployment("default", "my-dep")
    assert first.status == "DELETING"

    # Simulate the plugin finishing teardown before the next reconcile pass.
    present.clear()
    second = await backend.delete_model_deployment("default", "my-dep")
    assert second.status == "DELETED"


@pytest.mark.asyncio
async def test_delete_marks_weights_and_scratch_volumes_deleting_and_waits() -> None:
    client = _client_mock()
    volumes = {
        "my-dep-weights": VolumeDTO(name="my-dep-weights", workspace="default", size="50Gi", status="BOUND"),
        "my-dep-scratch": VolumeDTO(name="my-dep-scratch", workspace="default", size="10Gi", status="BOUND"),
    }
    get_dep, get_vol = _get_router({}, volumes)
    client.get_deployment = AsyncMock(side_effect=get_dep)
    client.get_volume = AsyncMock(side_effect=get_vol)
    client.list_deployment_configs = AsyncMock(return_value=_list_envelope([]))
    backend = _backend_with_client(client)

    result = await backend.delete_model_deployment("default", "my-dep")

    # No deployments left, so both volumes are unreferenced and get a delete issued;
    # delete stays DELETING until the volume entities are actually gone.
    assert result.status == "DELETING"
    deleted_volumes = {call.kwargs["name"] for call in client.delete_volume.await_args_list}
    assert deleted_volumes == {"my-dep-weights", "my-dep-scratch"}


@pytest.mark.asyncio
async def test_delete_preserves_volume_referenced_by_another_config() -> None:
    client = _client_mock()
    weights = VolumeDTO(name="my-dep-weights", workspace="default", size="50Gi", status="BOUND")
    get_dep, get_vol = _get_router({}, {"my-dep-weights": weights})
    client.get_deployment = AsyncMock(side_effect=get_dep)
    client.get_volume = AsyncMock(side_effect=get_vol)
    # A shared config still mounts the weights volume.
    shared = DeploymentConfigDTO(
        name="shared-config",
        workspace="default",
        volume_mounts=[{"name": "my-dep-weights", "mountPath": "/w"}],
    )
    client.list_deployment_configs = AsyncMock(return_value=_list_envelope([shared]))
    backend = _backend_with_client(client)

    result = await backend.delete_model_deployment("default", "my-dep")

    assert result.status == "DELETED"
    # The referenced weights volume is preserved (never deleted); only the
    # unreferenced scratch (absent here) would be.
    deleted_volumes = {call.kwargs["name"] for call in client.delete_volume.await_args_list}
    assert "my-dep-weights" not in deleted_volumes


@pytest.mark.asyncio
async def test_delete_already_deleting_volume_skips_reference_scan() -> None:
    client = _client_mock()
    volume = VolumeDTO(name="my-dep-weights", workspace="default", size="50Gi", status="DELETING")
    _, get_vol = _get_router({}, {"my-dep-weights": volume})
    client.get_volume = AsyncMock(side_effect=get_vol)
    client.list_deployment_configs = AsyncMock()
    backend = _backend_with_client(client)

    removed = await backend._complete_volume_delete("default", volume.name)

    assert not removed
    # An already-DELETING volume needs neither a reference scan nor another delete.
    client.list_deployment_configs.assert_not_called()
    client.delete_volume.assert_not_called()


@pytest.mark.asyncio
async def test_delete_continues_with_weights_when_scratch_teardown_fails() -> None:
    client = _client_mock()
    backend = _backend_with_client(client)

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
async def test_volume_delete_issues_delete_and_waits() -> None:
    client = _client_mock()
    volume = VolumeDTO(name="my-dep-weights", workspace="default", size="50Gi", status="BOUND")
    _, get_vol = _get_router({}, {"my-dep-weights": volume})
    client.get_volume = AsyncMock(side_effect=get_vol)
    client.list_deployment_configs = AsyncMock(return_value=_list_envelope([]))
    backend = _backend_with_client(client)

    removed = await backend._complete_volume_delete("default", volume.name)

    # Unreferenced volume: a delete is issued, but the entity is still present so
    # teardown is not complete yet.
    assert not removed
    client.delete_volume.assert_awaited_once_with(name="my-dep-weights", workspace="default")


@pytest.mark.asyncio
async def test_volume_delete_completes_once_volume_absent() -> None:
    client = _client_mock()
    _, get_vol = _get_router({}, {})
    client.get_volume = AsyncMock(side_effect=get_vol)
    backend = _backend_with_client(client)

    removed = await backend._complete_volume_delete("default", "my-dep-weights")

    assert removed
    client.delete_volume.assert_not_called()


@pytest.mark.asyncio
async def test_delete_stays_deleting_when_plugin_deployment_failed() -> None:
    # A deployment the plugin left in FAILED (e.g. deleting-timeout) stays present.
    # The models controller must NOT force-remove it (that would orphan backend
    # substrate); it keeps issuing delete and reports DELETING until a human
    # resolves the plugin-side failure and the entity actually disappears.
    client = _client_mock()
    server = DeploymentDTO(
        name="my-dep-server",
        workspace="default",
        deployment_config="my-dep-server",
        status="FAILED",
        desired_state="STOPPED",
    )
    get_dep, get_vol = _get_router({"my-dep-server": server}, {})
    client.get_deployment = AsyncMock(side_effect=get_dep)
    client.get_volume = AsyncMock(side_effect=get_vol)
    backend = _backend_with_client(client)

    result = await backend.delete_model_deployment("default", "my-dep")

    assert result.status == "DELETING"
    client.delete_deployment.assert_any_await(name="my-dep-server", workspace="default")
    # It never removes the config while the deployment is still present.
    client.delete_deployment_config.assert_not_called()


@pytest.mark.asyncio
async def test_delete_stays_deleting_while_deployment_teardown_pending() -> None:
    # Models mirrors the plugin's teardown state rather than inventing a terminal
    # ERROR: while the plugin deployment is still present (even long-running or
    # stuck), the model deployment stays DELETING. The models status API forbids
    # DELETING->ERROR anyway; the plugin owns the terminal outcome.
    client = _client_mock()
    server = DeploymentDTO(
        name="my-dep-server", workspace="default", deployment_config="my-dep-server", status="DELETING"
    )
    get_dep, get_vol = _get_router({"my-dep-server": server}, {})
    client.get_deployment = AsyncMock(side_effect=get_dep)
    client.get_volume = AsyncMock(side_effect=get_vol)
    backend = _backend_with_client(client, config=DeploymentsPluginConfig(deleting_timeout_seconds=60))

    result = await backend.delete_model_deployment("default", "my-dep", deleting_elapsed_seconds=120)
    assert result.status == "DELETING"
    assert result.error_details is None


@pytest.mark.asyncio
async def test_delete_stays_deleting_while_volume_teardown_pending() -> None:
    client = _client_mock()
    backend = _backend_with_client(client, config=DeploymentsPluginConfig(deleting_timeout_seconds=60))

    with (
        patch.object(backend, "_complete_deployment_delete", AsyncMock(return_value=True)),
        patch.object(backend, "_complete_volume_delete", AsyncMock(return_value=False)),
    ):
        result = await backend.delete_model_deployment("default", "my-dep", deleting_elapsed_seconds=120)

    assert result.status == "DELETING"
    assert result.error_details is None


@pytest.mark.asyncio
async def test_create_acts_on_behalf_of_requester() -> None:
    auth_context = DeploymentAuthContext(principal_id="user:alice", principal_groups=["research"])
    resolved = _resolved()
    resolved.deployment.auth_context = auth_context
    client = _client_mock()
    client.create_volume = AsyncMock()
    client.create_deployment_config = AsyncMock()
    client.create_deployment = AsyncMock()
    _install_empty_teardown(client)
    backend = _backend_with_client(client)

    with (
        patch(
            "nhx.core.models.controllers.backends.deployments_plugin.backend.resolve_plugin_deployment",
            return_value=resolved,
        ),
        patch(
            "nhx.core.models.controllers.backends.deployments_plugin.backend.executor_for_runtime",
            return_value="local-k8s",
        ),
    ):
        result = await backend.create_model_deployment(_ctx())

    assert result.status == "PENDING"
    # Every create is issued on behalf of the requester (not the models service).
    client.with_headers.assert_called_once()
    headers = client.with_headers.call_args.args[0]
    assert headers["X-NHX-Principal-On-Behalf-Of"] == "user:alice"
    assert headers["X-NHX-Principal-On-Behalf-Of-Groups"] == "research"


def _managed_config(workspace: str, name: str, *, role: str = "server", managed: bool = True) -> DeploymentConfigDTO:
    """A deployments-plugin DeploymentConfig DTO with models-controller ownership labels."""
    labels = {
        "nhx.nvidia.com/deployment-workspace": workspace,
        "nhx.nvidia.com/deployment-name": name,
        "nhx.nvidia.com/models-role": role,
    }
    if managed:
        labels["nhx.nvidia.com/managed-by"] = "models-controller"
    return DeploymentConfigDTO(name=f"{name}-{role}", workspace=workspace, labels=labels)


@pytest.mark.asyncio
async def test_list_managed_deployment_names_filters_and_queries_all_workspaces() -> None:
    # Server-role managed configs across two workspaces, plus rows that must be
    # excluded: a puller-role config, and one from another controller.
    configs = [
        _managed_config("ws-b", "dep-2"),
        _managed_config("ws-a", "dep-1"),
        _managed_config("ws-a", "dep-1", role="puller"),
        _managed_config("ws-c", "other", managed=False),
    ]
    client = _client_mock()
    client.list_deployment_configs = AsyncMock(return_value=_list_envelope(configs))
    backend = _backend_with_client(client)

    names = await backend.list_managed_deployment_names()

    assert names == ["ws-a/dep-1", "ws-b/dep-2"]
    client.list_deployment_configs.assert_awaited_once_with(workspace="-")
