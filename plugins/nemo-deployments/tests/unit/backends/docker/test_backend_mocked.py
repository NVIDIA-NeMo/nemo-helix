# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mocked docker client tests for DockerDeploymentBackend."""

from __future__ import annotations

import io
import socket
import tarfile
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from backends.docker.docker_helpers import (
    config_files_config,
    container_attrs,
    lora_config,
    published_port_config,
    sample_config,
)
from docker.errors import APIError, DockerException, NotFound
from nemo_deployments_plugin.backends.base import BackendStatusUpdate, MissingBackendDependencyError
from nemo_deployments_plugin.backends.docker import ports as ports_mod
from nemo_deployments_plugin.backends.docker.backend import (
    _PORT_CONFLICT_ATTEMPTS,
    _PORT_CONFLICT_MARKER,
    DOCKER_WORKLOAD_IDENTITY_TOKEN_FILE_LABEL,
    DOCKER_WORKLOAD_IDENTITY_VOLUME_LABEL,
    DockerDeploymentBackend,
)
from nemo_deployments_plugin.backends.docker.config import DEFAULT_WORKLOAD_TOKEN_WRITER_IMAGE
from nemo_deployments_plugin.backends.labels import (
    BACKOFF_LIMIT_LABEL,
    CONFIG_NAME_LABEL,
    CONTAINER_ROLE_LABEL,
    CONTAINER_ROLE_SERVER,
    DEFAULT_RESOURCE_SCOPE,
    DEPLOYMENT_NAME_LABEL,
    DEPLOYMENT_WORKSPACE_LABEL,
    MANAGED_BY_KEY,
    RESOURCE_SCOPE_LABEL,
    RESTART_POLICY_LABEL,
    companion_container_name,
    container_name,
    deployment_key,
)
from nemo_deployments_plugin.constants import MANAGED_BY_LABEL
from nemo_deployments_plugin.entities import Deployment, DeploymentConfig, WorkloadIdentitySpec
from nemo_deployments_plugin.types import RestartPolicy
from nemo_helix_plugin.auth import AuthContext
from nemo_helix_plugin.auth.workload_delegations import (
    WorkloadDelegationConflictError,
    WorkloadDelegationEntity,
    WorkloadDelegationLookupScope,
    WorkloadDelegationScope,
    parse_opaque_docker_proof_token,
    verify_opaque_docker_proof_token_hash,
)
from nemo_helix_plugin.auth.workload_identity import (
    WORKLOAD_IDENTITY_TOKEN_FILE_PATH,
    WORKLOAD_IDENTITY_VOLUME_PATH,
    build_docker_opaque_workload_delegation,
)
from nemo_helix_plugin.client.constants import WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR
from nemo_helix_plugin.entity_client import NemoEntityNotFoundError
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import ReadTimeout


def _workload_auth_context() -> AuthContext:
    return AuthContext(
        principal_id="user:alice",
        principal_email="alice@example.com",
        principal_groups=["research"],
    )


def _workload_identity_config(*, token_expiration_seconds: int = 900) -> DeploymentConfig:
    return sample_config().model_copy(
        update={
            "workload_identity": WorkloadIdentitySpec(
                enabled=True,
                workloadKind="agent_deployment",
                workloadId="logical-srv",
                tokenExpirationSeconds=token_expiration_seconds,
            )
        }
    )


def _workload_store() -> MagicMock:
    store = MagicMock()
    store.register = AsyncMock()
    store.revoke = AsyncMock()
    store.revoke_by_workload = AsyncMock(return_value=[])
    store.list_by_workload = AsyncMock(return_value=[])
    store.update = AsyncMock()
    return store


def _docker_workload_delegation(*, expires_in_seconds: int) -> WorkloadDelegationEntity:
    delegation, _ = build_docker_opaque_workload_delegation(
        scope=WorkloadDelegationScope(
            workload_workspace="default",
            workload_kind="agent_deployment",
            workload_instance_id=deployment_key("default", "srv"),
            workload_claim_id="logical-srv",
        ),
        workload_audience="nemo-helix",
        workload_generation=CONTAINER_ROLE_SERVER,
        auth_context=_workload_auth_context(),
        ttl_seconds_active=900,
    )
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in_seconds)
    return delegation.model_copy(update={"expires_at": expires_at})


def test_init_raises_missing_dependency_when_docker_daemon_unavailable(mock_sdk: MagicMock) -> None:
    with (
        patch("nemo_deployments_plugin.backends.docker.backend.client_from_platform"),
        patch("nemo_deployments_plugin.backends.docker.backend.NemoEntitiesClient"),
        patch("nemo_deployments_plugin.backends.docker.backend.get_shared_gpu_pool", return_value=None),
        patch("docker.from_env", side_effect=DockerException("Error while fetching server API version")),
    ):
        with pytest.raises(MissingBackendDependencyError, match="Docker daemon is unavailable"):
            DockerDeploymentBackend(mock_sdk, {"docker_timeout": 5, "pull_images": False})


def test_init_raises_missing_dependency_when_docker_ping_fails(mock_sdk: MagicMock) -> None:
    client = MagicMock()
    client.ping.side_effect = RequestsConnectionError("Connection refused")
    with (
        patch("nemo_deployments_plugin.backends.docker.backend.client_from_platform"),
        patch("nemo_deployments_plugin.backends.docker.backend.NemoEntitiesClient"),
        patch("nemo_deployments_plugin.backends.docker.backend.get_shared_gpu_pool", return_value=None),
        patch("docker.from_env", return_value=client),
    ):
        with pytest.raises(MissingBackendDependencyError, match="Docker daemon is unavailable"):
            DockerDeploymentBackend(mock_sdk, {"docker_timeout": 5, "pull_images": False})


@pytest.mark.asyncio
async def test_create_deployment_starts_container(
    docker_backend: DockerDeploymentBackend,
    mock_entities: AsyncMock,
    mock_docker_client: MagicMock,
) -> None:
    mock_entities.get.return_value = sample_config()
    mock_docker_client.containers.get.side_effect = NotFound("missing")
    mock_docker_client.containers.create.return_value = MagicMock(id="abc123")

    update = await docker_backend.create_deployment(
        workspace="default",
        name="srv",
        config_name="cfg1",
        labels={"managed-by": MANAGED_BY_LABEL},
        backend_config={},
    )

    assert update.status == "STARTING"
    mock_docker_client.containers.create.assert_called_once()
    mock_entities.get.assert_awaited()


@pytest.mark.asyncio
async def test_create_deployment_mounts_executor_additional_volumes(
    mock_sdk: MagicMock,
    mock_entities: AsyncMock,
    mock_docker_client: MagicMock,
) -> None:
    with (
        patch("nemo_deployments_plugin.backends.docker.backend.client_from_platform"),
        patch("nemo_deployments_plugin.backends.docker.backend.NemoEntitiesClient", return_value=mock_entities),
        patch("nemo_deployments_plugin.backends.docker.backend.get_shared_gpu_pool", return_value=None),
        patch("docker.from_env", return_value=mock_docker_client),
    ):
        backend = DockerDeploymentBackend(
            mock_sdk,
            {
                "docker_timeout": 60,
                "pull_images": False,
                "additional_volume_mounts": [
                    {
                        "volume_name": "gateway-tls",
                        "mount_path": "/etc/nhx/gateway-tls",
                        "read_only": True,
                    }
                ],
            },
        )

    mock_entities.get.return_value = sample_config()
    mock_docker_client.containers.get.side_effect = NotFound("missing")
    mock_docker_client.containers.create.return_value = MagicMock(id="abc123")

    update = await backend.create_deployment(
        workspace="default",
        name="srv",
        config_name="cfg1",
        labels={"managed-by": MANAGED_BY_LABEL},
        backend_config={},
    )

    assert update.status == "STARTING"
    _, create_kwargs = mock_docker_client.containers.create.call_args
    assert create_kwargs["volumes"]["gateway-tls"] == {"bind": "/etc/nhx/gateway-tls", "mode": "ro"}


@pytest.mark.asyncio
async def test_create_deployment_injects_workload_identity_volume_and_delegation(
    docker_backend: DockerDeploymentBackend,
    mock_entities: AsyncMock,
    mock_docker_client: MagicMock,
) -> None:
    config = _workload_identity_config()
    workload_store = _workload_store()
    docker_backend._workload_delegations = workload_store
    writer = AsyncMock()
    mock_entities.get.return_value = config
    mock_docker_client.containers.get.side_effect = NotFound("missing")
    mock_docker_client.containers.create.return_value = MagicMock(id="abc123")
    mock_docker_client.volumes.get.side_effect = NotFound("missing")

    with (
        patch(
            "nemo_deployments_plugin.backends.workload_identity.is_workload_identity_token_exchange_enabled",
            return_value=True,
        ),
        patch.object(docker_backend, "_write_workload_identity_subject_token", writer),
    ):
        update = await docker_backend.create_deployment(
            workspace="default",
            name="srv",
            config_name="cfg1",
            labels={"managed-by": MANAGED_BY_LABEL},
            backend_config={},
            auth_context=_workload_auth_context(),
        )

    assert update.status == "STARTING"
    volume_name = docker_backend._workload_identity_volume_name(
        workspace="default",
        name="srv",
        role=CONTAINER_ROLE_SERVER,
    )
    mock_docker_client.volumes.create.assert_called_once()
    assert mock_docker_client.volumes.create.call_args.kwargs["name"] == volume_name
    volume_labels = mock_docker_client.volumes.create.call_args.kwargs["labels"]
    assert volume_labels[DOCKER_WORKLOAD_IDENTITY_TOKEN_FILE_LABEL] == WORKLOAD_IDENTITY_TOKEN_FILE_PATH
    assert volume_labels[DOCKER_WORKLOAD_IDENTITY_VOLUME_LABEL] == volume_name
    _, create_kwargs = mock_docker_client.containers.create.call_args
    assert create_kwargs["environment"][WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR] == WORKLOAD_IDENTITY_TOKEN_FILE_PATH
    assert create_kwargs["volumes"][volume_name] == {"bind": WORKLOAD_IDENTITY_VOLUME_PATH, "mode": "ro"}
    assert create_kwargs["labels"][CONTAINER_ROLE_LABEL] == CONTAINER_ROLE_SERVER
    writer.assert_awaited_once()
    workload_store.register.assert_awaited_once()
    delegation = workload_store.register.await_args.args[0]
    assert delegation.workload_workspace == "default"
    assert delegation.workload_kind == "agent_deployment"
    assert delegation.workload_id == deployment_key("default", "srv")
    assert delegation.workload_claim_id == "logical-srv"
    assert delegation.workload_generation == CONTAINER_ROLE_SERVER
    assert delegation.workload_subject == delegation.name
    assert delegation.opaque_subject_token_hash
    assert delegation.auth_context == _workload_auth_context()
    assert workload_store.register.await_args.kwargs["require_opaque_subject_token_hash"] is True


@pytest.mark.asyncio
async def test_create_deployment_uses_workload_identity_token_ttl(
    docker_backend: DockerDeploymentBackend,
    mock_docker_client: MagicMock,
) -> None:
    config = _workload_identity_config(token_expiration_seconds=1200)
    workload_store = _workload_store()
    docker_backend._workload_delegations = workload_store
    mock_docker_client.volumes.get.side_effect = NotFound("missing")

    with (
        patch(
            "nemo_deployments_plugin.backends.workload_identity.is_workload_identity_token_exchange_enabled",
            return_value=True,
        ),
        patch.object(docker_backend, "_write_workload_identity_subject_token", AsyncMock()),
    ):
        before = datetime.now(timezone.utc) + timedelta(seconds=1500)
        await docker_backend._prepare_workload_identity_for_container(
            workspace="default",
            deployment_name="srv",
            config=config,
            role=CONTAINER_ROLE_SERVER,
            base_labels={MANAGED_BY_KEY: MANAGED_BY_LABEL},
            auth_context=_workload_auth_context(),
        )
        after = datetime.now(timezone.utc) + timedelta(seconds=1500)

    delegation = workload_store.register.await_args.args[0]
    assert before <= delegation.expires_at <= after


@pytest.mark.asyncio
async def test_create_deployment_mounts_workload_identity_on_auth_proxy_sidecar(
    docker_backend: DockerDeploymentBackend,
    mock_entities: AsyncMock,
    mock_docker_client: MagicMock,
) -> None:
    config = _workload_identity_config().model_copy(
        update={
            "auth_proxy_sidecar": True,
            "auth_proxy_sidecar_identity": "agents",
            "auth_proxy_sidecar_on_behalf_of": "user:alice",
        }
    )
    workload_store = _workload_store()
    docker_backend._workload_delegations = workload_store
    mock_entities.get.return_value = config
    mock_docker_client.containers.get.side_effect = NotFound("missing")
    mock_docker_client.containers.create.side_effect = [MagicMock(id="server123"), MagicMock(id="proxy123")]
    mock_docker_client.volumes.get.side_effect = NotFound("missing")

    with (
        patch("nemo_deployments_plugin.auth_proxy.platform_auth_enabled", return_value=True),
        patch(
            "nemo_deployments_plugin.backends.workload_identity.is_workload_identity_token_exchange_enabled",
            return_value=True,
        ),
        patch.object(docker_backend, "_write_workload_identity_subject_token", AsyncMock()),
    ):
        update = await docker_backend.create_deployment(
            workspace="default",
            name="srv",
            config_name="cfg1",
            labels={"managed-by": MANAGED_BY_LABEL},
            backend_config={},
            auth_context=_workload_auth_context(),
        )

    assert update.status == "STARTING"
    assert workload_store.register.await_count == 2
    assert mock_docker_client.volumes.create.call_count == 2
    sidecar_kwargs = mock_docker_client.containers.create.call_args_list[1].kwargs
    assert sidecar_kwargs["labels"][CONTAINER_ROLE_LABEL] == "auth-proxy"
    assert sidecar_kwargs["labels"][DOCKER_WORKLOAD_IDENTITY_TOKEN_FILE_LABEL] == WORKLOAD_IDENTITY_TOKEN_FILE_PATH
    assert DOCKER_WORKLOAD_IDENTITY_VOLUME_LABEL in sidecar_kwargs["labels"]
    assert sidecar_kwargs["environment"][WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR] == WORKLOAD_IDENTITY_TOKEN_FILE_PATH
    assert any(binding["bind"] == WORKLOAD_IDENTITY_VOLUME_PATH for binding in sidecar_kwargs["volumes"].values())


@pytest.mark.asyncio
async def test_prepare_workload_identity_scopes_shared_workload_id_by_deployment(
    docker_backend: DockerDeploymentBackend,
    mock_docker_client: MagicMock,
) -> None:
    config = _workload_identity_config()
    workload_store = _workload_store()
    docker_backend._workload_delegations = workload_store
    writer = AsyncMock()
    mock_docker_client.volumes.get.side_effect = NotFound("missing")

    with (
        patch(
            "nemo_deployments_plugin.backends.workload_identity.is_workload_identity_token_exchange_enabled",
            return_value=True,
        ),
        patch.object(docker_backend, "_write_workload_identity_subject_token", writer),
    ):
        await docker_backend._prepare_workload_identity_for_container(
            workspace="default",
            deployment_name="srv-a",
            config=config,
            role=CONTAINER_ROLE_SERVER,
            base_labels={MANAGED_BY_KEY: MANAGED_BY_LABEL},
            auth_context=_workload_auth_context(),
        )
        await docker_backend._prepare_workload_identity_for_container(
            workspace="default",
            deployment_name="srv-b",
            config=config,
            role=CONTAINER_ROLE_SERVER,
            base_labels={MANAGED_BY_KEY: MANAGED_BY_LABEL},
            auth_context=_workload_auth_context(),
        )

    delegations = [call.args[0] for call in workload_store.register.await_args_list]
    assert [delegation.workload_id for delegation in delegations] == [
        deployment_key("default", "srv-a"),
        deployment_key("default", "srv-b"),
    ]
    assert [delegation.workload_claim_id for delegation in delegations] == ["logical-srv", "logical-srv"]
    assert delegations[0].name != delegations[1].name
    assert writer.await_count == 2


@pytest.mark.asyncio
async def test_create_one_shot_recreate_cleans_prior_workload_identity_before_register(
    docker_backend: DockerDeploymentBackend,
    mock_entities: AsyncMock,
    mock_docker_client: MagicMock,
) -> None:
    config = _workload_identity_config().model_copy(update={"restart_policy": "Never"})
    workload_store = _workload_store()
    docker_backend._workload_delegations = workload_store
    existing = MagicMock()
    existing.status = "exited"
    existing.labels = {
        MANAGED_BY_KEY: MANAGED_BY_LABEL,
        DEPLOYMENT_WORKSPACE_LABEL: "default",
        DEPLOYMENT_NAME_LABEL: "srv",
        CONFIG_NAME_LABEL: "cfg1",
        RESTART_POLICY_LABEL: "Never",
        RESOURCE_SCOPE_LABEL: DEFAULT_RESOURCE_SCOPE,
    }
    order: list[str] = []

    async def revoke_by_workload(
        scope: WorkloadDelegationLookupScope,
    ) -> list[WorkloadDelegationEntity]:
        order.append("revoke")
        return []

    def list_volumes(*, filters: dict[str, list[str]]) -> list[MagicMock]:
        order.append("cleanup-volumes")
        return []

    async def register_delegation(
        delegation: WorkloadDelegationEntity,
        *,
        require_opaque_subject_token_hash: bool,
    ) -> None:
        order.append("register")

    workload_store.revoke_by_workload.side_effect = revoke_by_workload
    workload_store.register.side_effect = register_delegation
    mock_entities.get.return_value = config
    mock_docker_client.containers.get.return_value = existing
    mock_docker_client.containers.create.return_value = _one_shot_server_container(
        restart_policy="Never",
        exit_code=0,
    )
    mock_docker_client.volumes.get.side_effect = NotFound("missing")
    mock_docker_client.volumes.list.side_effect = list_volumes

    with (
        patch(
            "nemo_deployments_plugin.backends.workload_identity.is_workload_identity_token_exchange_enabled",
            return_value=True,
        ),
        patch.object(docker_backend, "_write_workload_identity_subject_token", AsyncMock()),
    ):
        update = await docker_backend.create_deployment(
            workspace="default",
            name="srv",
            config_name="cfg1",
            labels={"managed-by": MANAGED_BY_LABEL},
            backend_config={},
            auth_context=_workload_auth_context(),
        )

    assert update.status == "SUCCEEDED"
    existing.remove.assert_called_once_with(force=True)
    assert workload_store.revoke_by_workload.await_args_list[0].args == (
        WorkloadDelegationLookupScope(
            workload_workspace="default",
            workload_kind=None,
            workload_instance_id=deployment_key("default", "srv"),
        ),
    )
    assert order[:3] == ["revoke", "cleanup-volumes", "register"]


@pytest.mark.asyncio
async def test_write_workload_identity_subject_token_labels_token_writer_container(
    docker_backend: DockerDeploymentBackend,
    mock_docker_client: MagicMock,
) -> None:
    token_writer = MagicMock()
    token_writer.wait.return_value = {"StatusCode": 0}
    mock_docker_client.containers.create.return_value = token_writer

    await docker_backend._write_workload_identity_subject_token(
        "wi-volume",
        "subject-token",
        workspace="default",
        deployment_name="srv",
    )

    create_kwargs = mock_docker_client.containers.create.call_args.kwargs
    assert create_kwargs["image"] == DEFAULT_WORKLOAD_TOKEN_WRITER_IMAGE
    assert create_kwargs["labels"][MANAGED_BY_KEY] == MANAGED_BY_LABEL
    assert create_kwargs["labels"][DEPLOYMENT_WORKSPACE_LABEL] == "default"
    assert create_kwargs["labels"][DEPLOYMENT_NAME_LABEL] == "srv"
    assert create_kwargs["labels"][RESOURCE_SCOPE_LABEL] == DEFAULT_RESOURCE_SCOPE
    token_writer.put_archive.assert_called_once()
    token_writer.start.assert_called_once()
    token_writer.wait.assert_called_once_with(timeout=60)
    token_writer.remove.assert_called_once_with(force=True)


@pytest.mark.asyncio
async def test_workload_identity_volume_label_mismatch_recreates_volume(
    docker_backend: DockerDeploymentBackend,
    mock_docker_client: MagicMock,
) -> None:
    existing = MagicMock()
    existing.attrs = {"Labels": {MANAGED_BY_KEY: MANAGED_BY_LABEL, DEPLOYMENT_NAME_LABEL: "other"}}
    mock_docker_client.volumes.get.return_value = existing
    labels = {
        MANAGED_BY_KEY: MANAGED_BY_LABEL,
        DEPLOYMENT_WORKSPACE_LABEL: "default",
        DEPLOYMENT_NAME_LABEL: "srv",
        RESOURCE_SCOPE_LABEL: DEFAULT_RESOURCE_SCOPE,
    }

    await docker_backend._ensure_workload_identity_volume(volume_name="wi-volume", labels=labels)

    existing.remove.assert_called_once_with(force=True)
    mock_docker_client.volumes.create.assert_called_once_with(name="wi-volume", labels=labels)


@pytest.mark.asyncio
async def test_delete_deployment_revokes_workload_identity_delegations_and_volumes(
    docker_backend: DockerDeploymentBackend,
    mock_entities: AsyncMock,
    mock_docker_client: MagicMock,
) -> None:
    config = _workload_identity_config()
    workload_store = _workload_store()
    docker_backend._workload_delegations = workload_store
    mock_entities.get.side_effect = [
        Deployment(name="srv", workspace="default", deployment_config="cfg1"),
        config,
    ]
    mock_docker_client.containers.list.return_value = []
    mock_docker_client.containers.get.side_effect = NotFound("missing")
    volume = MagicMock()
    volume.name = "wi-volume"
    volume.attrs = {
        "Labels": {
            MANAGED_BY_KEY: MANAGED_BY_LABEL,
            DEPLOYMENT_WORKSPACE_LABEL: "default",
            DEPLOYMENT_NAME_LABEL: "srv",
            RESOURCE_SCOPE_LABEL: DEFAULT_RESOURCE_SCOPE,
        }
    }
    mock_docker_client.volumes.list.return_value = [volume]

    update = await docker_backend.delete_deployment("default", "srv")

    assert update.status == "SUCCEEDED"
    workload_store.revoke_by_workload.assert_awaited_once_with(
        WorkloadDelegationLookupScope(
            workload_workspace="default",
            workload_kind=None,
            workload_instance_id=deployment_key("default", "srv"),
        )
    )
    volume.remove.assert_called_once_with(force=True)


@pytest.mark.asyncio
async def test_delete_deployment_revokes_default_workload_identity_when_config_missing(
    docker_backend: DockerDeploymentBackend,
    mock_entities: AsyncMock,
    mock_docker_client: MagicMock,
) -> None:
    workload_store = _workload_store()
    docker_backend._workload_delegations = workload_store
    mock_entities.get.side_effect = NemoEntityNotFoundError("missing")
    mock_docker_client.containers.list.return_value = []
    mock_docker_client.containers.get.side_effect = NotFound("missing")
    mock_docker_client.volumes.list.return_value = []

    update = await docker_backend.delete_deployment("default", "srv")

    assert update.status == "SUCCEEDED"
    workload_store.revoke_by_workload.assert_awaited_once_with(
        WorkloadDelegationLookupScope(
            workload_workspace="default",
            workload_kind=None,
            workload_instance_id=deployment_key("default", "srv"),
        )
    )


@pytest.mark.asyncio
async def test_revoke_workload_identity_delegations_uses_broad_scope(
    docker_backend: DockerDeploymentBackend,
) -> None:
    workload_store = _workload_store()
    docker_backend._workload_delegations = workload_store

    await docker_backend._revoke_workload_delegations_for_deployment(workspace="default", name="srv")

    workload_store.revoke_by_workload.assert_awaited_once_with(
        WorkloadDelegationLookupScope(
            workload_workspace="default",
            workload_kind=None,
            workload_instance_id=deployment_key("default", "srv"),
        )
    )


@pytest.mark.asyncio
async def test_cleanup_missing_workload_identity_uses_broad_scope(
    docker_backend: DockerDeploymentBackend,
) -> None:
    workload_store = _workload_store()
    docker_backend._workload_delegations = workload_store

    await docker_backend._cleanup_missing_workload_identity(workspace="default", name="srv")

    workload_store.revoke_by_workload.assert_awaited_once_with(
        WorkloadDelegationLookupScope(
            workload_workspace="default",
            workload_kind=None,
            workload_instance_id=deployment_key("default", "srv"),
        )
    )


@pytest.mark.asyncio
async def test_refresh_workload_identity_delegation_skips_distant_expiry(
    docker_backend: DockerDeploymentBackend,
) -> None:
    config = _workload_identity_config()
    workload_store = _workload_store()
    workload_store.list_by_workload.return_value = [_docker_workload_delegation(expires_in_seconds=1000)]
    docker_backend._workload_delegations = workload_store

    await docker_backend._refresh_workload_delegations_for_config(workspace="default", name="srv", config=config)

    workload_store.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_refresh_workload_identity_delegation_extends_near_expiry(
    docker_backend: DockerDeploymentBackend,
) -> None:
    config = _workload_identity_config()
    delegation = _docker_workload_delegation(expires_in_seconds=60)
    original_expires_at = delegation.expires_at
    original_hash = delegation.opaque_subject_token_hash
    workload_store = _workload_store()
    workload_store.list_by_workload.return_value = [delegation]
    docker_backend._workload_delegations = workload_store
    writer = AsyncMock()

    with patch.object(docker_backend, "_write_workload_identity_subject_token", writer):
        await docker_backend._refresh_workload_delegations_for_config(workspace="default", name="srv", config=config)

    workload_store.update.assert_awaited_once()
    updated = workload_store.update.await_args.args[0]
    assert workload_store.update.await_args.kwargs == {"expected_db_version": delegation.db_version}
    assert updated.expires_at > original_expires_at
    assert updated.opaque_subject_token_hash != original_hash
    writer.assert_awaited_once()
    await_args = writer.await_args
    assert await_args is not None
    written_token = await_args.args[1]
    parsed = parse_opaque_docker_proof_token(written_token)
    assert parsed.delegation_name == delegation.name
    assert verify_opaque_docker_proof_token_hash(parsed.secret, updated.opaque_subject_token_hash)


@pytest.mark.asyncio
async def test_refresh_workload_identity_delegation_handles_naive_expiry(
    docker_backend: DockerDeploymentBackend,
) -> None:
    config = _workload_identity_config()
    delegation = _docker_workload_delegation(expires_in_seconds=60).model_copy(
        update={"expires_at": datetime.now() + timedelta(seconds=60)}
    )
    workload_store = _workload_store()
    workload_store.list_by_workload.return_value = [delegation]
    docker_backend._workload_delegations = workload_store

    with patch.object(docker_backend, "_write_workload_identity_subject_token", AsyncMock()):
        await docker_backend._refresh_workload_delegations_for_config(workspace="default", name="srv", config=config)

    workload_store.update.assert_awaited_once()


@pytest.mark.asyncio
async def test_refresh_workload_identity_delegation_continues_after_row_failure(
    docker_backend: DockerDeploymentBackend,
) -> None:
    config = _workload_identity_config()
    first = _docker_workload_delegation(expires_in_seconds=60)
    second = _docker_workload_delegation(expires_in_seconds=60).model_copy(update={"workload_generation": "worker"})
    workload_store = _workload_store()
    workload_store.list_by_workload.return_value = [first, second]
    docker_backend._workload_delegations = workload_store
    writer = AsyncMock(side_effect=[RuntimeError("volume unavailable"), None])

    with patch.object(docker_backend, "_write_workload_identity_subject_token", writer):
        await docker_backend._refresh_workload_delegations_for_config(workspace="default", name="srv", config=config)

    assert writer.await_count == 2
    workload_store.update.assert_awaited_once()
    assert workload_store.update.await_args.args[0].workload_generation == "worker"


@pytest.mark.asyncio
async def test_refresh_workload_identity_delegation_skips_stale_conflict(
    docker_backend: DockerDeploymentBackend,
) -> None:
    config = _workload_identity_config()
    delegation = _docker_workload_delegation(expires_in_seconds=60)
    workload_store = _workload_store()
    workload_store.list_by_workload.return_value = [delegation]
    workload_store.update.side_effect = WorkloadDelegationConflictError("stale")
    docker_backend._workload_delegations = workload_store

    with patch.object(docker_backend, "_write_workload_identity_subject_token", AsyncMock()):
        await docker_backend._refresh_workload_delegations_for_config(workspace="default", name="srv", config=config)

    workload_store.update.assert_awaited_once()
    assert workload_store.update.await_args.kwargs == {"expected_db_version": delegation.db_version}


@pytest.mark.asyncio
async def test_create_deployment_uses_executor_default_network(
    mock_sdk: MagicMock,
    mock_entities: AsyncMock,
    mock_docker_client: MagicMock,
) -> None:
    with (
        patch("nemo_deployments_plugin.backends.docker.backend.client_from_platform"),
        patch("nemo_deployments_plugin.backends.docker.backend.NemoEntitiesClient", return_value=mock_entities),
        patch("nemo_deployments_plugin.backends.docker.backend.get_shared_gpu_pool", return_value=None),
        patch("docker.from_env", return_value=mock_docker_client),
    ):
        backend = DockerDeploymentBackend(
            mock_sdk,
            {"docker_timeout": 60, "pull_images": False, "network": "nhx-e2e-test-network"},
        )
        backend._client = mock_docker_client

    mock_entities.get.return_value = sample_config()
    mock_docker_client.containers.get.side_effect = NotFound("missing")
    mock_docker_client.containers.create.return_value = MagicMock(id="abc123")

    update = await backend.create_deployment(
        workspace="default",
        name="srv",
        config_name="cfg1",
        labels={"managed-by": MANAGED_BY_LABEL},
        backend_config={},
    )

    assert update.status == "STARTING"
    _, create_kwargs = mock_docker_client.containers.create.call_args
    assert create_kwargs["network"] == "nhx-e2e-test-network"


@pytest.mark.asyncio
async def test_create_deployment_backend_config_network_overrides_executor_default(
    mock_sdk: MagicMock,
    mock_entities: AsyncMock,
    mock_docker_client: MagicMock,
) -> None:
    with (
        patch("nemo_deployments_plugin.backends.docker.backend.client_from_platform"),
        patch("nemo_deployments_plugin.backends.docker.backend.NemoEntitiesClient", return_value=mock_entities),
        patch("nemo_deployments_plugin.backends.docker.backend.get_shared_gpu_pool", return_value=None),
        patch("docker.from_env", return_value=mock_docker_client),
    ):
        backend = DockerDeploymentBackend(
            mock_sdk,
            {"docker_timeout": 60, "pull_images": False, "network": "executor-network"},
        )
        backend._client = mock_docker_client

    mock_entities.get.return_value = sample_config()
    mock_docker_client.containers.get.side_effect = NotFound("missing")
    mock_docker_client.containers.create.return_value = MagicMock(id="abc123")

    update = await backend.create_deployment(
        workspace="default",
        name="srv",
        config_name="cfg1",
        labels={"managed-by": MANAGED_BY_LABEL},
        backend_config={"docker": {"network": "deployment-network"}},
    )

    assert update.status == "STARTING"
    _, create_kwargs = mock_docker_client.containers.create.call_args
    assert create_kwargs["network"] == "deployment-network"


@pytest.mark.asyncio
async def test_create_deployment_network_endpoint_mode_reports_container_url(
    mock_sdk: MagicMock,
    mock_entities: AsyncMock,
    mock_docker_client: MagicMock,
    free_host_ports: None,
) -> None:
    del free_host_ports
    with (
        patch("nemo_deployments_plugin.backends.docker.backend.client_from_platform"),
        patch("nemo_deployments_plugin.backends.docker.backend.NemoEntitiesClient", return_value=mock_entities),
        patch("nemo_deployments_plugin.backends.docker.backend.get_shared_gpu_pool", return_value=None),
        patch("docker.from_env", return_value=mock_docker_client),
    ):
        backend = DockerDeploymentBackend(
            mock_sdk,
            {
                "docker_timeout": 60,
                "pull_images": False,
                "network": "nhx-e2e-test-network",
                "endpoint_mode": "network",
            },
        )
        backend._client = mock_docker_client

    mock_entities.get.return_value = published_port_config()
    mock_docker_client.containers.get.side_effect = NotFound("missing")
    mock_docker_client.containers.create.return_value = MagicMock(id="abc123")

    update = await backend.create_deployment(
        workspace="default",
        name="srv",
        config_name="cfg1",
        labels={"managed-by": MANAGED_BY_LABEL},
        backend_config={},
    )

    assert update.status == "STARTING"
    assert len(update.endpoints) == 1
    assert update.endpoints[0].url == f"http://{container_name('default', 'srv')}:8000"


@pytest.mark.asyncio
async def test_create_deployment_maps_command_to_entrypoint(
    docker_backend: DockerDeploymentBackend,
    mock_entities: AsyncMock,
    mock_docker_client: MagicMock,
) -> None:
    """A spec's ``command`` overrides the image ENTRYPOINT; ``args`` is the CMD.

    This mirrors Kubernetes semantics and is required so a driven container
    (e.g. a packaged agent that bakes its own ``ENTRYPOINT``) runs the
    platform-supplied command instead of the image default.
    """
    mock_entities.get.return_value = sample_config()  # command=["echo"], args=["hello"]
    mock_docker_client.containers.get.side_effect = NotFound("missing")
    mock_docker_client.containers.create.return_value = MagicMock(id="abc123")

    await docker_backend.create_deployment(
        workspace="default",
        name="srv",
        config_name="cfg1",
        labels={"managed-by": MANAGED_BY_LABEL},
        backend_config={},
    )

    _, create_kwargs = mock_docker_client.containers.create.call_args
    assert create_kwargs["entrypoint"] == ["echo"]
    assert create_kwargs["command"] == ["hello"]


def _delivered_files(put_archive: MagicMock) -> dict[str, bytes]:
    """Return {path: content} for the regular files in a ``put_archive`` payload."""
    (dest, archive), _ = put_archive.call_args
    assert dest == "/", "paths in the tar are absolute-rooted, so the target must be /"
    delivered: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            payload = tar.extractfile(member)
            assert payload is not None, f"regular file {member.name} has no payload"
            delivered[f"/{member.name}"] = payload.read()
    return delivered


@pytest.mark.asyncio
async def test_create_deployment_delivers_config_files_before_start(
    docker_backend: DockerDeploymentBackend,
    mock_entities: AsyncMock,
    mock_docker_client: MagicMock,
) -> None:
    """``config_files`` land in the container's filesystem before it starts.

    The server command reads its config at startup, so delivery after ``start``
    would race the process. Asserting the ordering is the point: dropping the
    delivery entirely used to be silent (AIRCORE-999), and delivering it late
    fails the same way.
    """
    mock_entities.get.return_value = config_files_config()
    mock_docker_client.containers.get.side_effect = NotFound("missing")
    created = MagicMock(id="abc123")
    mock_docker_client.containers.create.return_value = created

    update = await docker_backend.create_deployment(
        workspace="default",
        name="srv",
        config_name="cfg1",
        labels={"managed-by": MANAGED_BY_LABEL},
        backend_config={},
    )

    assert update.status == "STARTING"
    assert _delivered_files(created.put_archive) == {"/tmp/nemo/config.yaml": b"workflow:\n  _type: react_agent\n"}
    method_order = [name for name, _, _ in created.mock_calls]
    assert method_order.index("put_archive") < method_order.index("start")


@pytest.mark.asyncio
async def test_config_files_tar_carries_parent_dirs_and_mode(
    docker_backend: DockerDeploymentBackend,
    mock_entities: AsyncMock,
    mock_docker_client: MagicMock,
) -> None:
    """Nested paths bring their parent directories; files keep their declared mode.

    ``put_archive`` does not create missing parents, so a tar carrying only the
    leaf fails for any path below the image's existing tree.
    """
    mock_entities.get.return_value = config_files_config(path="/tmp/nemo/sub/agent.yaml", mode=0o600)
    mock_docker_client.containers.get.side_effect = NotFound("missing")
    created = MagicMock(id="abc123")
    mock_docker_client.containers.create.return_value = created

    await docker_backend.create_deployment(
        workspace="default",
        name="srv",
        config_name="cfg1",
        labels={"managed-by": MANAGED_BY_LABEL},
        backend_config={},
    )

    (_, archive), _ = created.put_archive.call_args
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r") as tar:
        dirs = {m.name: m.mode for m in tar.getmembers() if m.isdir()}
        files = {m.name: m.mode for m in tar.getmembers() if m.isfile()}

    assert dirs == {"tmp": 0o1777, "tmp/nemo": 0o777, "tmp/nemo/sub": 0o755}
    assert files == {"tmp/nemo/sub/agent.yaml": 0o600}


@pytest.mark.asyncio
async def test_create_deployment_without_config_files_skips_put_archive(
    docker_backend: DockerDeploymentBackend,
    mock_entities: AsyncMock,
    mock_docker_client: MagicMock,
) -> None:
    mock_entities.get.return_value = sample_config()
    mock_docker_client.containers.get.side_effect = NotFound("missing")
    created = MagicMock(id="abc123")
    mock_docker_client.containers.create.return_value = created

    await docker_backend.create_deployment(
        workspace="default",
        name="srv",
        config_name="cfg1",
        labels={"managed-by": MANAGED_BY_LABEL},
        backend_config={},
    )

    created.put_archive.assert_not_called()
    created.start.assert_called_once()


@pytest.mark.asyncio
async def test_failed_config_delivery_removes_container_and_reports_error(
    docker_backend: DockerDeploymentBackend,
    mock_entities: AsyncMock,
    mock_docker_client: MagicMock,
) -> None:
    """A delivery failure must not leave a started container or a healthy status.

    Without this the container would run with no config and report STARTING,
    which is the silent-failure shape the original bug had.
    """
    mock_entities.get.return_value = config_files_config()
    mock_docker_client.containers.get.side_effect = NotFound("missing")
    created = MagicMock(id="abc123")
    created.put_archive.side_effect = APIError("no such container")
    mock_docker_client.containers.create.return_value = created

    update = await docker_backend.create_deployment(
        workspace="default",
        name="srv",
        config_name="cfg1",
        labels={"managed-by": MANAGED_BY_LABEL},
        backend_config={},
    )

    created.start.assert_not_called()
    created.remove.assert_called_once_with(force=True)
    assert update.status == "FAILED"


def _port_conflict_error(port: int) -> APIError:
    return APIError(
        f"driver failed programming external connectivity: Bind for 0.0.0.0:{port} failed: {_PORT_CONFLICT_MARKER}"
    )


@pytest.fixture
def free_host_ports(monkeypatch: pytest.MonkeyPatch, mock_docker_client: MagicMock) -> None:
    """Make host port allocation deterministic: nothing published, every probe free."""
    mock_docker_client.containers.list.return_value = []
    monkeypatch.setattr(ports_mod, "is_port_free", lambda port: True)


def _published_host_ports(run_mock: MagicMock) -> list[int]:
    return [kwargs["ports"]["8000/tcp"] for _, kwargs in run_mock.call_args_list]


@pytest.mark.asyncio
async def test_create_deployment_reallocates_port_after_docker_conflict(
    docker_backend: DockerDeploymentBackend,
    mock_entities: AsyncMock,
    mock_docker_client: MagicMock,
    free_host_ports: None,
) -> None:
    """Docker's port reservations are invisible to the probe, so a publish can still lose a race."""
    first_port = docker_backend._executor_config.port_range_start
    server_container = MagicMock(id="abc123")
    mock_entities.get.return_value = published_port_config()
    mock_docker_client.containers.get.side_effect = NotFound("missing")
    mock_docker_client.containers.create.return_value = server_container
    server_container.start.side_effect = [_port_conflict_error(first_port), None]

    update = await docker_backend.create_deployment(
        workspace="default",
        name="srv",
        config_name="cfg1",
        labels={"managed-by": MANAGED_BY_LABEL},
        backend_config={},
    )

    assert update.status == "STARTING"
    assert _published_host_ports(mock_docker_client.containers.create) == [first_port, first_port + 1]
    server_container.remove.assert_called_once_with(force=True)


@pytest.mark.asyncio
async def test_create_deployment_fails_after_repeated_port_conflicts(
    docker_backend: DockerDeploymentBackend,
    mock_entities: AsyncMock,
    mock_docker_client: MagicMock,
    free_host_ports: None,
) -> None:
    first_port = docker_backend._executor_config.port_range_start
    server_container = MagicMock(id="abc123")
    mock_entities.get.return_value = published_port_config()
    mock_docker_client.containers.get.side_effect = NotFound("missing")
    mock_docker_client.containers.create.return_value = server_container
    server_container.start.side_effect = [
        _port_conflict_error(first_port + offset) for offset in range(_PORT_CONFLICT_ATTEMPTS)
    ]

    update = await docker_backend.create_deployment(
        workspace="default",
        name="srv",
        config_name="cfg1",
        labels={"managed-by": MANAGED_BY_LABEL},
        backend_config={},
    )

    assert update.status == "FAILED"
    assert _PORT_CONFLICT_MARKER in (update.status_message or "")
    published = _published_host_ports(mock_docker_client.containers.create)
    assert published == [first_port + offset for offset in range(_PORT_CONFLICT_ATTEMPTS)]


@pytest.mark.asyncio
async def test_create_deployment_does_not_retry_unrelated_start_failure(
    docker_backend: DockerDeploymentBackend,
    mock_entities: AsyncMock,
    mock_docker_client: MagicMock,
    free_host_ports: None,
) -> None:
    server_container = MagicMock(id="abc123")
    mock_entities.get.return_value = published_port_config()
    mock_docker_client.containers.get.side_effect = NotFound("missing")
    mock_docker_client.containers.create.return_value = server_container
    server_container.start.side_effect = APIError("no such image")

    update = await docker_backend.create_deployment(
        workspace="default",
        name="srv",
        config_name="cfg1",
        labels={"managed-by": MANAGED_BY_LABEL},
        backend_config={},
    )

    assert update.status == "FAILED"
    mock_docker_client.containers.create.assert_called_once()


@pytest.mark.asyncio
async def test_create_lora_group_runs_init_server_and_sidecar(
    docker_backend: DockerDeploymentBackend,
    mock_entities: AsyncMock,
    mock_docker_client: MagicMock,
) -> None:
    """A LoRA-shaped config runs init-to-completion, then server, then sidecar.

    The sidecar shares the server's network namespace (network=container:<server>)
    and publishes no host ports; only the server maps ports.
    """
    mock_entities.get.return_value = lora_config()
    mock_docker_client.containers.get.side_effect = NotFound("missing")

    init_container = MagicMock()
    init_container.wait.return_value = {"StatusCode": 0}
    server_container = MagicMock(id="server123")
    sidecar_container = MagicMock(id="sidecar123")
    mock_docker_client.containers.create.side_effect = [init_container, server_container, sidecar_container]

    update = await docker_backend.create_deployment(
        workspace="default",
        name="srv",
        config_name="cfg1",
        labels={"managed-by": MANAGED_BY_LABEL},
        backend_config={},
    )

    assert update.status == "STARTING"
    calls = mock_docker_client.containers.create.call_args_list
    assert len(calls) == 3

    init_container.start.assert_called_once()
    init_container.wait.assert_called_once()
    init_container.remove.assert_called_once()

    server_kwargs = calls[1].kwargs
    assert server_kwargs["name"] == container_name("default", "srv")
    assert server_kwargs["labels"][CONTAINER_ROLE_LABEL] == "server"
    assert "ports" in server_kwargs
    assert server_kwargs.get("network", "") == ""

    sidecar_kwargs = calls[2].kwargs
    assert sidecar_kwargs["name"] == companion_container_name("default", "srv", "lora-adapters")
    assert sidecar_kwargs["labels"][CONTAINER_ROLE_LABEL] == "lora-adapters"
    assert sidecar_kwargs["network"] == f"container:{container_name('default', 'srv')}"
    assert "ports" not in sidecar_kwargs


@pytest.mark.asyncio
async def test_create_lora_group_fails_when_init_nonzero(
    docker_backend: DockerDeploymentBackend,
    mock_entities: AsyncMock,
    mock_docker_client: MagicMock,
) -> None:
    """A non-zero init container fails the deployment before the server starts."""
    mock_entities.get.return_value = lora_config()
    mock_docker_client.containers.get.side_effect = NotFound("missing")

    init_container = MagicMock()
    init_container.wait.return_value = {"StatusCode": 1}
    mock_docker_client.containers.create.return_value = init_container

    update = await docker_backend.create_deployment(
        workspace="default",
        name="srv",
        config_name="cfg1",
        labels={"managed-by": MANAGED_BY_LABEL},
        backend_config={},
    )

    assert update.status == "FAILED"
    assert "init" in update.status_message.lower()
    assert mock_docker_client.containers.create.call_count == 1


@pytest.mark.asyncio
async def test_create_falls_back_to_local_image_when_pull_fails(
    mock_sdk: MagicMock,
    mock_entities: AsyncMock,
    mock_docker_client: MagicMock,
) -> None:
    """A pull failure is tolerated when the image is already present locally.

    Local-only images (e.g. the LoRA adapters sidecar ``nhx-api:local``) are not
    pullable from a registry; the backend uses the local copy instead of failing.
    """
    from unittest.mock import patch

    with (
        patch("nemo_deployments_plugin.backends.docker.backend.client_from_platform"),
        patch("nemo_deployments_plugin.backends.docker.backend.NemoEntitiesClient", return_value=mock_entities),
        patch("nemo_deployments_plugin.backends.docker.backend.get_shared_gpu_pool", return_value=None),
        patch("docker.from_env", return_value=mock_docker_client),
    ):
        # pull_images enabled so the pull path runs
        backend = DockerDeploymentBackend(mock_sdk, {"docker_timeout": 60, "pull_images": True})
        backend._client = mock_docker_client

    mock_entities.get.return_value = sample_config()
    mock_docker_client.containers.get.side_effect = NotFound("missing")
    mock_docker_client.images.pull.side_effect = APIError("404 not found")
    mock_docker_client.images.get.return_value = MagicMock()  # present locally
    mock_docker_client.containers.create.return_value = MagicMock(id="abc123")

    update = await backend.create_deployment(
        workspace="default",
        name="srv",
        config_name="cfg1",
        labels={"managed-by": MANAGED_BY_LABEL},
        backend_config={},
    )

    assert update.status == "STARTING"
    mock_docker_client.images.get.assert_called_once()
    mock_docker_client.containers.create.assert_called_once()


@pytest.mark.asyncio
async def test_create_fails_when_pull_fails_and_no_local_image(
    mock_sdk: MagicMock,
    mock_entities: AsyncMock,
    mock_docker_client: MagicMock,
) -> None:
    """If the image cannot be pulled AND is not present locally, create fails."""
    from unittest.mock import patch

    with (
        patch("nemo_deployments_plugin.backends.docker.backend.client_from_platform"),
        patch("nemo_deployments_plugin.backends.docker.backend.NemoEntitiesClient", return_value=mock_entities),
        patch("nemo_deployments_plugin.backends.docker.backend.get_shared_gpu_pool", return_value=None),
        patch("docker.from_env", return_value=mock_docker_client),
    ):
        backend = DockerDeploymentBackend(mock_sdk, {"docker_timeout": 60, "pull_images": True})
        backend._client = mock_docker_client

    mock_entities.get.return_value = sample_config()
    mock_docker_client.containers.get.side_effect = NotFound("missing")
    mock_docker_client.images.pull.side_effect = APIError("404 not found")
    mock_docker_client.images.get.side_effect = NotFound("missing locally")

    update = await backend.create_deployment(
        workspace="default",
        name="srv",
        config_name="cfg1",
        labels={"managed-by": MANAGED_BY_LABEL},
        backend_config={},
    )

    assert update.status == "FAILED"
    assert "pull image" in update.status_message.lower()
    mock_docker_client.containers.create.assert_not_called()


@pytest.mark.asyncio
async def test_delete_removes_whole_group(
    docker_backend: DockerDeploymentBackend,
    mock_docker_client: MagicMock,
) -> None:
    """delete_deployment stops+removes every container in the group."""
    server = MagicMock()
    server.name = container_name("default", "srv")
    server.labels = {
        "managed-by": MANAGED_BY_LABEL,
        DEPLOYMENT_WORKSPACE_LABEL: "default",
        DEPLOYMENT_NAME_LABEL: "srv",
        RESOURCE_SCOPE_LABEL: DEFAULT_RESOURCE_SCOPE,
    }
    sidecar = MagicMock()
    sidecar.name = companion_container_name("default", "srv", "lora-adapters")
    sidecar.labels = {
        "managed-by": MANAGED_BY_LABEL,
        DEPLOYMENT_WORKSPACE_LABEL: "default",
        DEPLOYMENT_NAME_LABEL: "srv",
        RESOURCE_SCOPE_LABEL: DEFAULT_RESOURCE_SCOPE,
    }
    mock_docker_client.containers.list.return_value = [server, sidecar]

    update = await docker_backend.delete_deployment("default", "srv")

    assert update.status == "SUCCEEDED"
    server.remove.assert_called_once()
    sidecar.remove.assert_called_once()


@pytest.mark.asyncio
async def test_delete_scoped_deployment_does_not_remove_foreign_primary(
    mock_sdk: MagicMock,
    mock_entities: AsyncMock,
    mock_docker_client: MagicMock,
) -> None:
    with (
        patch("nemo_deployments_plugin.backends.docker.backend.client_from_platform"),
        patch("nemo_deployments_plugin.backends.docker.backend.NemoEntitiesClient", return_value=mock_entities),
        patch("nemo_deployments_plugin.backends.docker.backend.get_shared_gpu_pool", return_value=None),
        patch("docker.from_env", return_value=mock_docker_client),
    ):
        backend = DockerDeploymentBackend(
            mock_sdk,
            {"docker_timeout": 60, "pull_images": False, "resource_scope": "e2e-abc123"},
        )

    foreign = MagicMock()
    foreign.name = container_name("default", "srv")
    foreign.labels = {
        "managed-by": MANAGED_BY_LABEL,
        DEPLOYMENT_WORKSPACE_LABEL: "default",
        DEPLOYMENT_NAME_LABEL: "srv",
        RESOURCE_SCOPE_LABEL: "other-scope",
    }
    mock_docker_client.containers.list.return_value = []
    mock_docker_client.containers.get.return_value = foreign

    update = await backend.delete_deployment("default", "srv")

    assert update.status == "SUCCEEDED"
    foreign.stop.assert_not_called()
    foreign.remove.assert_not_called()


@pytest.mark.asyncio
async def test_delete_default_scoped_deployment_does_not_remove_foreign_primary(
    docker_backend: DockerDeploymentBackend,
    mock_docker_client: MagicMock,
) -> None:
    scoped = MagicMock()
    scoped.name = container_name("default", "srv")
    scoped.labels = {
        "managed-by": MANAGED_BY_LABEL,
        DEPLOYMENT_WORKSPACE_LABEL: "default",
        DEPLOYMENT_NAME_LABEL: "srv",
        RESOURCE_SCOPE_LABEL: "e2e-abc123",
    }
    mock_docker_client.containers.list.return_value = [scoped]
    mock_docker_client.containers.get.return_value = scoped

    update = await docker_backend.delete_deployment("default", "srv")

    assert update.status == "SUCCEEDED"
    scoped.stop.assert_not_called()
    scoped.remove.assert_not_called()


@pytest.mark.asyncio
async def test_create_lora_group_does_not_remove_foreign_stale_init_container(
    mock_sdk: MagicMock,
    mock_entities: AsyncMock,
    mock_docker_client: MagicMock,
) -> None:
    with (
        patch("nemo_deployments_plugin.backends.docker.backend.client_from_platform"),
        patch("nemo_deployments_plugin.backends.docker.backend.NemoEntitiesClient", return_value=mock_entities),
        patch("nemo_deployments_plugin.backends.docker.backend.get_shared_gpu_pool", return_value=None),
        patch("docker.from_env", return_value=mock_docker_client),
    ):
        backend = DockerDeploymentBackend(
            mock_sdk,
            {"docker_timeout": 60, "pull_images": False, "resource_scope": "e2e-abc123"},
        )

    foreign_stale = MagicMock()
    foreign_stale.labels = {
        "managed-by": MANAGED_BY_LABEL,
        DEPLOYMENT_WORKSPACE_LABEL: "default",
        DEPLOYMENT_NAME_LABEL: "srv",
        RESOURCE_SCOPE_LABEL: "other-scope",
    }
    init_container = MagicMock()
    init_container.wait.return_value = {"StatusCode": 0}
    server_container = MagicMock(id="server123")
    sidecar_container = MagicMock(id="sidecar123")
    mock_entities.get.return_value = lora_config()
    mock_docker_client.containers.get.side_effect = [NotFound("missing"), foreign_stale]
    mock_docker_client.containers.create.side_effect = [init_container, server_container, sidecar_container]

    update = await backend.create_deployment(
        workspace="default",
        name="srv",
        config_name="cfg1",
        labels={"managed-by": MANAGED_BY_LABEL},
        backend_config={},
    )

    assert update.status == "STARTING"
    foreign_stale.remove.assert_not_called()


@pytest.mark.asyncio
async def test_create_volume_runs_init_chmod_container(
    docker_backend: DockerDeploymentBackend,
    mock_docker_client: MagicMock,
) -> None:
    """initChmod/initImage in the docker volume backend_config trigger an init
    chmod container so a non-root workload (the HF weight-puller) can write the
    otherwise root-owned named volume. This is the docker analogue of k8s fsGroup.
    """
    mock_docker_client.volumes.get.side_effect = NotFound("missing")
    mock_docker_client.volumes.create.return_value = MagicMock(name="vol")

    update = await docker_backend.create_volume(
        workspace="default",
        name="weights",
        size="40Gi",
        access_modes=["ReadWriteOnce"],
        backend_config={"docker": {"initChmod": "0777", "initImage": "docker.io/library/busybox"}},
    )

    assert update.status == "BOUND"
    labels = mock_docker_client.volumes.create.call_args.kwargs["labels"]
    assert labels[RESOURCE_SCOPE_LABEL] == DEFAULT_RESOURCE_SCOPE
    mock_docker_client.containers.run.assert_called_once()
    args, run_kwargs = mock_docker_client.containers.run.call_args
    assert args[0] == "docker.io/library/busybox"
    # chmod is invoked directly (no `sh -c`) so the mode is never shell-interpolated
    assert run_kwargs["entrypoint"] == ["chmod"]
    assert run_kwargs["command"] == ["0777", "/vol"]
    assert run_kwargs["remove"] is True
    # the init container must mount the volume it is fixing up
    assert any(b.get("bind") == "/vol" for b in run_kwargs["volumes"].values())


@pytest.mark.asyncio
async def test_create_volume_without_init_chmod_skips_container(
    docker_backend: DockerDeploymentBackend,
    mock_docker_client: MagicMock,
) -> None:
    mock_docker_client.volumes.get.side_effect = NotFound("missing")
    mock_docker_client.volumes.create.return_value = MagicMock(name="vol")

    update = await docker_backend.create_volume(
        workspace="default",
        name="weights",
        size="40Gi",
        access_modes=["ReadWriteOnce"],
        backend_config={},
    )

    assert update.status == "BOUND"
    mock_docker_client.containers.run.assert_not_called()


@pytest.mark.asyncio
async def test_create_volume_skips_init_chmod_when_volume_exists(
    docker_backend: DockerDeploymentBackend,
    mock_docker_client: MagicMock,
) -> None:
    """The init chmod container runs only on a fresh create, not when reusing an
    already-initialized volume — otherwise every reconcile would spin one up.
    """
    mock_docker_client.volumes.get.return_value = MagicMock(name="vol")  # already exists

    update = await docker_backend.create_volume(
        workspace="default",
        name="weights",
        size="40Gi",
        access_modes=["ReadWriteOnce"],
        backend_config={"docker": {"initChmod": "0777", "initImage": "docker.io/library/busybox"}},
    )

    assert update.status == "BOUND"
    mock_docker_client.volumes.create.assert_not_called()
    mock_docker_client.containers.run.assert_not_called()


@pytest.mark.asyncio
async def test_read_status_ready_when_running_without_probe(
    docker_backend: DockerDeploymentBackend,
    mock_entities: AsyncMock,
    mock_docker_client: MagicMock,
) -> None:
    container = MagicMock()
    container.id = "abc123def456"
    container.status = "running"
    container.labels = {
        "managed-by": MANAGED_BY_LABEL,
        DEPLOYMENT_WORKSPACE_LABEL: "default",
        DEPLOYMENT_NAME_LABEL: "srv",
        RESTART_POLICY_LABEL: "Always",
        CONFIG_NAME_LABEL: "cfg1",
        RESOURCE_SCOPE_LABEL: DEFAULT_RESOURCE_SCOPE,
    }
    container.ports = {}
    container.attrs = container_attrs()
    mock_docker_client.containers.get.return_value = container
    mock_entities.get.return_value = sample_config()

    update = await docker_backend.read_status(workspace="default", name="srv")

    assert update.status == "READY"


def _running_container_with_published_port(host_port: int) -> MagicMock:
    container = MagicMock()
    container.id = "abc123def456"
    container.status = "running"
    container.labels = {
        "managed-by": MANAGED_BY_LABEL,
        DEPLOYMENT_WORKSPACE_LABEL: "default",
        DEPLOYMENT_NAME_LABEL: "srv",
        RESTART_POLICY_LABEL: "Always",
        CONFIG_NAME_LABEL: "cfg1",
        RESOURCE_SCOPE_LABEL: DEFAULT_RESOURCE_SCOPE,
    }
    container.ports = {"8000/tcp": [{"HostPort": str(host_port)}]}
    container.attrs = container_attrs()
    return container


@pytest.mark.asyncio
async def test_read_status_ready_when_running_port_bound(
    docker_backend: DockerDeploymentBackend,
    mock_entities: AsyncMock,
    mock_docker_client: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No declared probe, published port accepting connections -> READY.
    monkeypatch.setenv("NHX_LOOPBACK_ADDRESS", "127.0.0.1")
    mock_entities.get.return_value = sample_config()
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        host_port = server.getsockname()[1]
        mock_docker_client.containers.get.return_value = _running_container_with_published_port(host_port)

        update = await docker_backend.read_status(workspace="default", name="srv")

    assert update.status == "READY"


@pytest.mark.asyncio
async def test_read_status_starting_when_running_port_not_bound(
    docker_backend: DockerDeploymentBackend,
    mock_entities: AsyncMock,
    mock_docker_client: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No declared probe, nothing yet listening on the published port -> STARTING, so
    # READY does not race the workload's bind(). Hold the port bound-but-not-listening
    # for the whole probe so nothing else can bind and listen on it mid-test; a
    # connect() still gets ECONNREFUSED, the not-yet-bound state under test.
    monkeypatch.setenv("NHX_LOOPBACK_ADDRESS", "127.0.0.1")
    mock_entities.get.return_value = sample_config()
    with socket.socket() as probe_socket:
        probe_socket.bind(("127.0.0.1", 0))
        host_port = probe_socket.getsockname()[1]
        mock_docker_client.containers.get.return_value = _running_container_with_published_port(host_port)

        update = await docker_backend.read_status(workspace="default", name="srv")

    assert update.status == "STARTING"
    assert "not ready" in update.status_message


@pytest.mark.asyncio
async def test_read_status_network_endpoint_mode_probes_container_url(
    mock_sdk: MagicMock,
    mock_entities: AsyncMock,
    mock_docker_client: MagicMock,
) -> None:
    with (
        patch("nemo_deployments_plugin.backends.docker.backend.client_from_platform"),
        patch("nemo_deployments_plugin.backends.docker.backend.NemoEntitiesClient", return_value=mock_entities),
        patch("nemo_deployments_plugin.backends.docker.backend.get_shared_gpu_pool", return_value=None),
        patch("docker.from_env", return_value=mock_docker_client),
    ):
        backend = DockerDeploymentBackend(
            mock_sdk,
            {
                "docker_timeout": 60,
                "pull_images": False,
                "network": "nhx-e2e-test-network",
                "endpoint_mode": "network",
            },
        )
        backend._client = mock_docker_client

    mock_entities.get.return_value = published_port_config()
    mock_docker_client.containers.get.return_value = _running_container_with_published_port(49152)
    with patch(
        "nemo_deployments_plugin.backends.docker.backend.check_readiness_probe",
        new=AsyncMock(return_value=(True, "http probe 200")),
    ) as readiness:
        update = await backend.read_status(workspace="default", name="srv")

    target_url = f"http://{container_name('default', 'srv')}:8000"
    assert update.status == "READY"
    assert len(update.endpoints) == 1
    assert update.endpoints[0].url == target_url
    readiness.assert_awaited_once()
    await_args = readiness.await_args
    assert await_args is not None
    assert await_args.kwargs["host_url"] == target_url
    assert await_args.kwargs["host_ports"] == {8000: 8000}


@pytest.mark.asyncio
async def test_read_status_ready_when_running_udp_only_port(
    docker_backend: DockerDeploymentBackend,
    mock_entities: AsyncMock,
    mock_docker_client: MagicMock,
) -> None:
    # A UDP-only workload has no TCP listener, so the default TCP probe is skipped and
    # running implies ready rather than wedging STARTING until the progress deadline.
    mock_entities.get.return_value = sample_config()
    container = _running_container_with_published_port(0)
    container.ports = {"9000/udp": [{"HostPort": "34567"}]}
    mock_docker_client.containers.get.return_value = container

    update = await docker_backend.read_status(workspace="default", name="srv")

    assert update.status == "READY"


def _running_server_container() -> MagicMock:
    container = MagicMock()
    container.id = "abc123def456"
    container.status = "running"
    container.labels = {
        "managed-by": MANAGED_BY_LABEL,
        DEPLOYMENT_WORKSPACE_LABEL: "default",
        DEPLOYMENT_NAME_LABEL: "srv",
        RESTART_POLICY_LABEL: "Always",
        CONFIG_NAME_LABEL: "cfg1",
        RESOURCE_SCOPE_LABEL: DEFAULT_RESOURCE_SCOPE,
    }
    container.ports = {}
    container.attrs = container_attrs()
    return container


def _sidecar_container(role: str, status: str) -> MagicMock:
    sidecar = MagicMock()
    sidecar.status = status
    sidecar.labels = {
        "managed-by": MANAGED_BY_LABEL,
        DEPLOYMENT_WORKSPACE_LABEL: "default",
        DEPLOYMENT_NAME_LABEL: "srv",
        CONTAINER_ROLE_LABEL: role,
        RESOURCE_SCOPE_LABEL: DEFAULT_RESOURCE_SCOPE,
    }
    return sidecar


@pytest.mark.asyncio
async def test_read_status_ready_when_sidecar_running(
    docker_backend: DockerDeploymentBackend,
    mock_entities: AsyncMock,
    mock_docker_client: MagicMock,
) -> None:
    """Server ready + adapters sidecar running => READY."""
    mock_docker_client.containers.get.return_value = _running_server_container()
    mock_entities.get.return_value = lora_config()  # server + lora-adapters sidecar
    mock_docker_client.containers.list.return_value = [_sidecar_container("lora-adapters", "running")]

    update = await docker_backend.read_status(workspace="default", name="srv")

    assert update.status == "READY"


@pytest.mark.asyncio
async def test_read_status_starting_when_sidecar_stopped(
    docker_backend: DockerDeploymentBackend,
    mock_entities: AsyncMock,
    mock_docker_client: MagicMock,
) -> None:
    """Server ready but the adapters sidecar is stopped => drop back to STARTING.

    Locks in the readiness gating: a present-but-not-running sidecar keeps the
    deployment out of READY.
    """
    mock_docker_client.containers.get.return_value = _running_server_container()
    mock_entities.get.return_value = lora_config()
    mock_docker_client.containers.list.return_value = [_sidecar_container("lora-adapters", "exited")]

    update = await docker_backend.read_status(workspace="default", name="srv")

    assert update.status == "STARTING"
    assert "sidecar" in update.status_message.lower()


@pytest.mark.asyncio
async def test_read_status_starting_when_sidecar_removed(
    docker_backend: DockerDeploymentBackend,
    mock_entities: AsyncMock,
    mock_docker_client: MagicMock,
) -> None:
    """Server ready but the expected adapters sidecar is entirely gone => STARTING.

    A removed (not merely exited) sidecar is detected by comparing the expected
    sidecar roles from the config against the containers actually present.
    """
    mock_docker_client.containers.get.return_value = _running_server_container()
    mock_entities.get.return_value = lora_config()
    # The sidecar container has been removed: only unrelated containers remain.
    mock_docker_client.containers.list.return_value = []

    update = await docker_backend.read_status(workspace="default", name="srv")

    assert update.status == "STARTING"
    assert "sidecar" in update.status_message.lower()


@pytest.mark.asyncio
async def test_read_status_lost_when_missing_always(
    docker_backend: DockerDeploymentBackend,
    mock_entities: AsyncMock,
    mock_docker_client: MagicMock,
) -> None:
    mock_docker_client.containers.get.side_effect = NotFound("missing")

    deployment_entity = MagicMock()
    deployment_entity.deployment_config = "cfg1"

    async def get_side_effect(entity_type, name, workspace=None):
        if entity_type is Deployment:
            return deployment_entity
        return sample_config(restart_policy="Always")

    mock_entities.get.side_effect = get_side_effect
    gpu_pool = MagicMock()
    docker_backend._gpu_pool = gpu_pool

    update = await docker_backend.read_status(workspace="default", name="srv")

    assert update.status == "LOST"
    gpu_pool.release_gpu.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("restart_policy", ["Never", "OnFailure"])
async def test_read_status_failed_and_releases_gpu_when_missing_one_shot(
    docker_backend: DockerDeploymentBackend,
    mock_entities: AsyncMock,
    mock_docker_client: MagicMock,
    restart_policy: RestartPolicy,
) -> None:
    expected_container_name = container_name("default", "job")
    mock_docker_client.containers.get.side_effect = NotFound("missing")

    deployment_entity = MagicMock()
    deployment_entity.deployment_config = "cfg1"

    async def get_side_effect(entity_type, name, workspace=None):
        if entity_type is Deployment:
            return deployment_entity
        return sample_config(restart_policy=restart_policy)

    mock_entities.get.side_effect = get_side_effect
    gpu_pool = MagicMock()
    docker_backend._gpu_pool = gpu_pool

    update = await docker_backend.read_status(workspace="default", name="job")

    assert update.status == "FAILED"
    assert update.exit_code is None
    assert update.error_details == {
        "expected_container_name": expected_container_name,
    }
    gpu_pool.release_gpu.assert_called_once_with(deployment_key("default", "job"))


@pytest.mark.asyncio
async def test_read_status_observes_exited_never_container_without_removing(
    docker_backend: DockerDeploymentBackend,
    mock_docker_client: MagicMock,
) -> None:
    container = MagicMock()
    container.id = "abc123def456"
    container.status = "exited"
    container.labels = {
        "managed-by": MANAGED_BY_LABEL,
        DEPLOYMENT_WORKSPACE_LABEL: "default",
        DEPLOYMENT_NAME_LABEL: "job",
        RESTART_POLICY_LABEL: "Never",
        CONFIG_NAME_LABEL: "cfg1",
        RESOURCE_SCOPE_LABEL: DEFAULT_RESOURCE_SCOPE,
    }
    container.ports = {}
    container.attrs = container_attrs(exit_code=0)
    mock_docker_client.containers.get.return_value = container

    update = await docker_backend.read_status(workspace="default", name="job")

    assert update.status == "SUCCEEDED"
    assert update.exit_code == 0
    container.remove.assert_not_called()


@pytest.mark.asyncio
async def test_read_status_retries_exited_on_failure_container_under_backoff_limit(
    docker_backend: DockerDeploymentBackend,
    mock_docker_client: MagicMock,
) -> None:
    container = MagicMock()
    container.id = "abc123def456"
    container.status = "exited"
    container.labels = {
        "managed-by": MANAGED_BY_LABEL,
        DEPLOYMENT_WORKSPACE_LABEL: "default",
        DEPLOYMENT_NAME_LABEL: "job",
        RESTART_POLICY_LABEL: "OnFailure",
        CONFIG_NAME_LABEL: "cfg1",
        BACKOFF_LIMIT_LABEL: "3",
        RESOURCE_SCOPE_LABEL: DEFAULT_RESOURCE_SCOPE,
    }
    container.ports = {}
    container.attrs = container_attrs(exit_code=1, restart_count=2)
    mock_docker_client.containers.get.return_value = container
    gpu_pool = MagicMock()
    docker_backend._gpu_pool = gpu_pool

    update = await docker_backend.read_status(workspace="default", name="job")

    assert update.status == "STARTING"
    assert update.exit_code == 1
    assert "retry 2/3" in update.status_message
    gpu_pool.release_gpu.assert_not_called()


@pytest.mark.asyncio
async def test_read_status_fails_exited_on_failure_container_after_backoff_limit(
    docker_backend: DockerDeploymentBackend,
    mock_docker_client: MagicMock,
) -> None:
    container = MagicMock()
    container.id = "abc123def456"
    container.status = "exited"
    container.labels = {
        "managed-by": MANAGED_BY_LABEL,
        DEPLOYMENT_WORKSPACE_LABEL: "default",
        DEPLOYMENT_NAME_LABEL: "job",
        RESTART_POLICY_LABEL: "OnFailure",
        CONFIG_NAME_LABEL: "cfg1",
        BACKOFF_LIMIT_LABEL: "3",
        RESOURCE_SCOPE_LABEL: DEFAULT_RESOURCE_SCOPE,
    }
    container.ports = {}
    container.attrs = container_attrs(exit_code=1, restart_count=3)
    mock_docker_client.containers.get.return_value = container
    gpu_pool = MagicMock()
    docker_backend._gpu_pool = gpu_pool

    update = await docker_backend.read_status(workspace="default", name="job")

    assert update.status == "FAILED"
    assert update.exit_code == 1
    gpu_pool.release_gpu.assert_called_once_with(deployment_key("default", "job"))


@pytest.mark.asyncio
async def test_read_status_treats_zero_on_failure_backoff_as_unlimited(
    docker_backend: DockerDeploymentBackend,
    mock_docker_client: MagicMock,
) -> None:
    container = MagicMock()
    container.id = "abc123def456"
    container.status = "exited"
    container.labels = {
        "managed-by": MANAGED_BY_LABEL,
        DEPLOYMENT_WORKSPACE_LABEL: "default",
        DEPLOYMENT_NAME_LABEL: "job",
        RESTART_POLICY_LABEL: "OnFailure",
        CONFIG_NAME_LABEL: "cfg1",
        BACKOFF_LIMIT_LABEL: "0",
        RESOURCE_SCOPE_LABEL: DEFAULT_RESOURCE_SCOPE,
    }
    container.ports = {}
    container.attrs = container_attrs(exit_code=1, restart_count=7)
    mock_docker_client.containers.get.return_value = container
    gpu_pool = MagicMock()
    docker_backend._gpu_pool = gpu_pool

    update = await docker_backend.read_status(workspace="default", name="job")

    assert update.status == "STARTING"
    assert update.exit_code == 1
    assert "retry 7/unlimited" in update.status_message
    gpu_pool.release_gpu.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("restart_policy", "expected_status", "should_release_gpu"),
    [
        ("Always", "LOST", False),
        ("Never", "FAILED", True),
    ],
)
async def test_read_status_treats_foreign_container_as_missing(
    mock_sdk: MagicMock,
    mock_entities: AsyncMock,
    mock_docker_client: MagicMock,
    restart_policy: RestartPolicy,
    expected_status: str,
    should_release_gpu: bool,
) -> None:
    gpu_pool = MagicMock()
    with (
        patch("nemo_deployments_plugin.backends.docker.backend.client_from_platform"),
        patch("nemo_deployments_plugin.backends.docker.backend.NemoEntitiesClient", return_value=mock_entities),
        patch("nemo_deployments_plugin.backends.docker.backend.get_shared_gpu_pool", return_value=gpu_pool),
        patch("docker.from_env", return_value=mock_docker_client),
    ):
        backend = DockerDeploymentBackend(
            mock_sdk,
            {"docker_timeout": 60, "pull_images": False, "resource_scope": "e2e-abc123"},
        )

    foreign = MagicMock()
    foreign.labels = {
        "managed-by": MANAGED_BY_LABEL,
        DEPLOYMENT_WORKSPACE_LABEL: "default",
        DEPLOYMENT_NAME_LABEL: "srv",
        RESOURCE_SCOPE_LABEL: "other-scope",
    }
    mock_docker_client.containers.get.return_value = foreign

    deployment_entity = MagicMock()
    deployment_entity.deployment_config = "cfg1"

    async def get_side_effect(entity_type, name, workspace=None):
        if entity_type is Deployment:
            return deployment_entity
        return sample_config(restart_policy=restart_policy)

    mock_entities.get.side_effect = get_side_effect

    update = await backend.read_status(workspace="default", name="srv")

    assert update.status == expected_status
    foreign.reload.assert_not_called()
    if should_release_gpu:
        gpu_pool.release_gpu.assert_called_once_with(deployment_key("default", "srv"))
    else:
        gpu_pool.release_gpu.assert_not_called()


@pytest.mark.asyncio
async def test_read_status_unknown_on_transient_docker_error(
    docker_backend: DockerDeploymentBackend,
    mock_docker_client: MagicMock,
) -> None:
    mock_docker_client.containers.get.side_effect = APIError("connection reset")

    update = await docker_backend.read_status(workspace="default", name="srv")

    assert update.status == "UNKNOWN"
    assert "Docker API error" in update.status_message


@pytest.mark.asyncio
async def test_delete_deployment_idempotent(
    docker_backend: DockerDeploymentBackend,
    mock_docker_client: MagicMock,
) -> None:
    mock_docker_client.containers.get.side_effect = NotFound("missing")

    update = await docker_backend.delete_deployment("default", "srv")

    assert update.status == "SUCCEEDED"


@pytest.mark.asyncio
async def test_list_managed_deployment_names(
    docker_backend: DockerDeploymentBackend,
    mock_docker_client: MagicMock,
) -> None:
    container = MagicMock()
    container.labels = {
        "managed-by": MANAGED_BY_LABEL,
        DEPLOYMENT_WORKSPACE_LABEL: "default",
        DEPLOYMENT_NAME_LABEL: "srv",
        RESOURCE_SCOPE_LABEL: DEFAULT_RESOURCE_SCOPE,
    }
    mock_docker_client.containers.list.return_value = [container]

    names = await docker_backend.list_managed_deployment_names()

    assert names == ["default/srv"]


def _one_shot_server_container(
    *,
    restart_policy: str,
    status: str = "exited",
    exit_code: int = 0,
    restart_count: int = 0,
    backoff_limit: str = "6",
) -> MagicMock:
    container = MagicMock()
    container.name = container_name("default", "job")
    container.wait.return_value = {"StatusCode": exit_code}
    container.labels = {
        RESTART_POLICY_LABEL: restart_policy,
        BACKOFF_LIMIT_LABEL: backoff_limit,
    }
    container.status = status
    container.attrs = {
        **container_attrs(exit_code=exit_code),
        "RestartCount": restart_count,
    }
    return container


@pytest.mark.asyncio
async def test_create_never_job_returns_succeeded_when_container_exits_immediately(
    docker_backend: DockerDeploymentBackend,
    mock_entities: AsyncMock,
    mock_docker_client: MagicMock,
) -> None:
    mock_entities.get.return_value = sample_config(restart_policy="Never")
    mock_docker_client.containers.get.side_effect = NotFound("missing")
    mock_docker_client.containers.create.return_value = _one_shot_server_container(
        restart_policy="Never",
        exit_code=0,
    )

    update = await docker_backend.create_deployment(
        workspace="default",
        name="job",
        config_name="cfg1",
        labels={"managed-by": MANAGED_BY_LABEL},
        backend_config={},
    )

    assert update.status == "SUCCEEDED"
    assert update.exit_code == 0
    mock_docker_client.containers.create.return_value.wait.assert_called_once_with(timeout=5)


@pytest.mark.asyncio
async def test_create_never_job_uses_configured_oneshot_observe_timeout(
    mock_sdk: MagicMock,
    mock_entities: AsyncMock,
    mock_docker_client: MagicMock,
) -> None:
    with (
        patch("nemo_deployments_plugin.backends.docker.backend.client_from_platform"),
        patch("nemo_deployments_plugin.backends.docker.backend.NemoEntitiesClient", return_value=mock_entities),
        patch("nemo_deployments_plugin.backends.docker.backend.get_shared_gpu_pool", return_value=None),
        patch("docker.from_env", return_value=mock_docker_client),
    ):
        backend = DockerDeploymentBackend(
            mock_sdk,
            {"docker_timeout": 600, "oneshot_observe_timeout_seconds": 7, "pull_images": False},
        )
        backend._client = mock_docker_client

    mock_entities.get.return_value = sample_config(restart_policy="Never")
    mock_docker_client.containers.get.side_effect = NotFound("missing")
    mock_docker_client.containers.create.return_value = _one_shot_server_container(
        restart_policy="Never",
        exit_code=0,
    )

    update = await backend.create_deployment(
        workspace="default",
        name="job",
        config_name="cfg1",
        labels={"managed-by": MANAGED_BY_LABEL},
        backend_config={},
    )

    assert update.status == "SUCCEEDED"
    mock_docker_client.containers.create.return_value.wait.assert_called_once_with(timeout=7)


@pytest.mark.asyncio
async def test_create_never_job_returns_failed_on_non_zero_exit(
    docker_backend: DockerDeploymentBackend,
    mock_entities: AsyncMock,
    mock_docker_client: MagicMock,
) -> None:
    mock_entities.get.return_value = sample_config(restart_policy="Never")
    mock_docker_client.containers.get.side_effect = NotFound("missing")
    mock_docker_client.containers.create.return_value = _one_shot_server_container(
        restart_policy="Never",
        exit_code=42,
    )

    update = await docker_backend.create_deployment(
        workspace="default",
        name="job",
        config_name="cfg1",
        labels={"managed-by": MANAGED_BY_LABEL},
        backend_config={},
    )

    assert update.status == "FAILED"
    assert update.exit_code == 42


@pytest.mark.asyncio
async def test_create_on_failure_returns_succeeded_when_already_exited_zero(
    docker_backend: DockerDeploymentBackend,
    mock_entities: AsyncMock,
    mock_docker_client: MagicMock,
) -> None:
    mock_entities.get.return_value = sample_config(restart_policy="OnFailure")
    mock_docker_client.containers.get.side_effect = NotFound("missing")
    server = _one_shot_server_container(restart_policy="OnFailure", exit_code=0)
    mock_docker_client.containers.create.return_value = server

    update = await docker_backend.create_deployment(
        workspace="default",
        name="job",
        config_name="cfg1",
        labels={"managed-by": MANAGED_BY_LABEL},
        backend_config={},
    )

    assert update.status == "SUCCEEDED"
    assert update.exit_code == 0
    server.wait.assert_not_called()


@pytest.mark.asyncio
async def test_create_on_failure_returns_starting_when_failed_under_backoff(
    docker_backend: DockerDeploymentBackend,
    mock_entities: AsyncMock,
    mock_docker_client: MagicMock,
) -> None:
    mock_entities.get.return_value = sample_config(restart_policy="OnFailure")
    mock_docker_client.containers.get.side_effect = NotFound("missing")
    server = _one_shot_server_container(
        restart_policy="OnFailure",
        exit_code=1,
        restart_count=2,
        backoff_limit="6",
    )
    mock_docker_client.containers.create.return_value = server

    update = await docker_backend.create_deployment(
        workspace="default",
        name="job",
        config_name="cfg1",
        labels={"managed-by": MANAGED_BY_LABEL},
        backend_config={},
    )

    assert update.status == "STARTING"
    assert update.exit_code == 1
    assert "retry 2/6" in update.status_message


@pytest.mark.asyncio
async def test_create_on_failure_returns_starting_when_still_running(
    docker_backend: DockerDeploymentBackend,
    mock_entities: AsyncMock,
    mock_docker_client: MagicMock,
) -> None:
    mock_entities.get.return_value = sample_config(restart_policy="OnFailure")
    mock_docker_client.containers.get.side_effect = NotFound("missing")
    server = _one_shot_server_container(restart_policy="OnFailure", status="running", exit_code=0)
    mock_docker_client.containers.create.return_value = server

    update = await docker_backend.create_deployment(
        workspace="default",
        name="job",
        config_name="cfg1",
        labels={"managed-by": MANAGED_BY_LABEL},
        backend_config={},
    )

    assert update.status == "STARTING"
    assert "created" in update.status_message.lower()
    server.wait.assert_not_called()


@pytest.mark.asyncio
async def test_create_never_job_returns_starting_when_wait_times_out(
    docker_backend: DockerDeploymentBackend,
    mock_entities: AsyncMock,
    mock_docker_client: MagicMock,
) -> None:
    mock_entities.get.return_value = sample_config(restart_policy="Never")
    mock_docker_client.containers.get.side_effect = NotFound("missing")
    server = _one_shot_server_container(restart_policy="Never", exit_code=0)
    server.wait.side_effect = ReadTimeout("timed out")
    mock_docker_client.containers.create.return_value = server

    update = await docker_backend.create_deployment(
        workspace="default",
        name="job",
        config_name="cfg1",
        labels={"managed-by": MANAGED_BY_LABEL},
        backend_config={},
    )

    assert update.status == "STARTING"
    assert "after observe wait (5s)" in update.status_message
    server.wait.assert_called_once_with(timeout=5)


@pytest.mark.asyncio
async def test_create_never_job_returns_starting_when_wait_connection_error(
    docker_backend: DockerDeploymentBackend,
    mock_entities: AsyncMock,
    mock_docker_client: MagicMock,
) -> None:
    mock_entities.get.return_value = sample_config(restart_policy="Never")
    mock_docker_client.containers.get.side_effect = NotFound("missing")
    server = _one_shot_server_container(restart_policy="Never", exit_code=0)
    server.wait.side_effect = RequestsConnectionError("connection reset")
    mock_docker_client.containers.create.return_value = server

    update = await docker_backend.create_deployment(
        workspace="default",
        name="job",
        config_name="cfg1",
        labels={"managed-by": MANAGED_BY_LABEL},
        backend_config={},
    )

    assert update.status == "STARTING"
    assert "after observe wait (5s)" in update.status_message
    server.wait.assert_called_once_with(timeout=5)


@pytest.mark.asyncio
async def test_create_never_job_cleans_up_on_wait_error(
    docker_backend: DockerDeploymentBackend,
    mock_entities: AsyncMock,
    mock_docker_client: MagicMock,
) -> None:
    mock_entities.get.return_value = sample_config(restart_policy="Never")
    mock_docker_client.containers.get.side_effect = NotFound("missing")
    server = _one_shot_server_container(restart_policy="Never", exit_code=0)
    server.wait.side_effect = RuntimeError("boom")
    mock_docker_client.containers.create.return_value = server

    with patch.object(
        docker_backend,
        "delete_deployment",
        new_callable=AsyncMock,
        return_value=BackendStatusUpdate(status="SUCCEEDED", status_message="deleted"),
    ) as mock_delete:
        update = await docker_backend.create_deployment(
            workspace="default",
            name="job",
            config_name="cfg1",
            labels={"managed-by": MANAGED_BY_LABEL},
            backend_config={},
        )

    assert update.status == "FAILED"
    mock_delete.assert_awaited_once_with("default", "job")


@pytest.mark.asyncio
async def test_create_always_still_returns_starting(
    docker_backend: DockerDeploymentBackend,
    mock_entities: AsyncMock,
    mock_docker_client: MagicMock,
) -> None:
    mock_entities.get.return_value = sample_config(restart_policy="Always")
    mock_docker_client.containers.get.side_effect = NotFound("missing")
    mock_docker_client.containers.create.return_value = MagicMock(id="abc123")

    update = await docker_backend.create_deployment(
        workspace="default",
        name="srv",
        config_name="cfg1",
        labels={"managed-by": MANAGED_BY_LABEL},
        backend_config={},
    )

    assert update.status == "STARTING"
    mock_docker_client.containers.create.return_value.wait.assert_not_called()


@pytest.mark.asyncio
async def test_default_list_managed_deployment_names_ignores_foreign_scoped_resources(
    docker_backend: DockerDeploymentBackend,
    mock_docker_client: MagicMock,
) -> None:
    default_scoped = MagicMock()
    default_scoped.labels = {
        "managed-by": MANAGED_BY_LABEL,
        DEPLOYMENT_WORKSPACE_LABEL: "default",
        DEPLOYMENT_NAME_LABEL: "srv",
        RESOURCE_SCOPE_LABEL: DEFAULT_RESOURCE_SCOPE,
    }
    scoped = MagicMock()
    scoped.labels = {
        "managed-by": MANAGED_BY_LABEL,
        DEPLOYMENT_WORKSPACE_LABEL: "default",
        DEPLOYMENT_NAME_LABEL: "foreign",
        RESOURCE_SCOPE_LABEL: "e2e-abc123",
    }
    mock_docker_client.containers.list.return_value = [default_scoped, scoped]

    names = await docker_backend.list_managed_deployment_names()

    assert names == ["default/srv"]


@pytest.mark.asyncio
async def test_list_managed_deployment_names_scopes_docker_query(
    mock_sdk: MagicMock,
    mock_entities: AsyncMock,
    mock_docker_client: MagicMock,
) -> None:
    with (
        patch("nemo_deployments_plugin.backends.docker.backend.client_from_platform"),
        patch("nemo_deployments_plugin.backends.docker.backend.NemoEntitiesClient", return_value=mock_entities),
        patch("nemo_deployments_plugin.backends.docker.backend.get_shared_gpu_pool", return_value=None),
        patch("docker.from_env", return_value=mock_docker_client),
    ):
        backend = DockerDeploymentBackend(
            mock_sdk,
            {"docker_timeout": 60, "pull_images": False, "resource_scope": "e2e-abc123"},
        )

    container = MagicMock()
    container.labels = {
        "managed-by": MANAGED_BY_LABEL,
        DEPLOYMENT_WORKSPACE_LABEL: "default",
        DEPLOYMENT_NAME_LABEL: "srv",
        RESOURCE_SCOPE_LABEL: "e2e-abc123",
    }
    mock_docker_client.containers.list.return_value = [container]

    names = await backend.list_managed_deployment_names()

    assert names == ["default/srv"]
    mock_docker_client.containers.list.assert_called_once_with(
        all=True,
        filters={
            "label": [
                f"{MANAGED_BY_KEY}={MANAGED_BY_LABEL}",
                f"{RESOURCE_SCOPE_LABEL}=e2e-abc123",
            ]
        },
    )


@pytest.mark.asyncio
async def test_get_logs_treats_foreign_container_as_missing(
    mock_sdk: MagicMock,
    mock_entities: AsyncMock,
    mock_docker_client: MagicMock,
) -> None:
    with (
        patch("nemo_deployments_plugin.backends.docker.backend.client_from_platform"),
        patch("nemo_deployments_plugin.backends.docker.backend.NemoEntitiesClient", return_value=mock_entities),
        patch("nemo_deployments_plugin.backends.docker.backend.get_shared_gpu_pool", return_value=None),
        patch("docker.from_env", return_value=mock_docker_client),
    ):
        backend = DockerDeploymentBackend(
            mock_sdk,
            {"docker_timeout": 60, "pull_images": False, "resource_scope": "e2e-abc123"},
        )

    foreign = MagicMock()
    foreign.labels = {
        "managed-by": MANAGED_BY_LABEL,
        DEPLOYMENT_WORKSPACE_LABEL: "default",
        DEPLOYMENT_NAME_LABEL: "srv",
        RESOURCE_SCOPE_LABEL: "other-scope",
    }
    mock_docker_client.containers.get.return_value = foreign

    logs = await backend.get_logs(workspace="default", name="srv")

    assert logs.lines == [f"Container {container_name('default', 'srv')} not found"]
    foreign.logs.assert_not_called()


@pytest.mark.asyncio
async def test_create_volume_bound(
    docker_backend: DockerDeploymentBackend,
    mock_docker_client: MagicMock,
) -> None:
    volume = MagicMock()
    volume.name = "dep-vol-default-data"
    mock_docker_client.volumes.get.side_effect = NotFound("missing")
    mock_docker_client.volumes.create.return_value = volume

    update = await docker_backend.create_volume(
        workspace="default",
        name="data",
        size="1Gi",
        access_modes=["ReadWriteOnce"],
        backend_config={},
    )

    assert update.status == "BOUND"
