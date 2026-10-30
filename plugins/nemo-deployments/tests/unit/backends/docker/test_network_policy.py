# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Docker network selection tests; no Docker daemon or network access."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from backends.docker.docker_helpers import lora_config, sample_config
from docker.errors import NotFound
from nemo_deployments_plugin.backends.docker.backend import DockerDeploymentBackend
from nemo_deployments_plugin.backends.docker.config import DockerExecutorConfig
from nemo_deployments_plugin.entities import DeploymentBackendConfig, DockerDeploymentConfig
from nemo_deployments_plugin.schema import CreateDeploymentConfigRequest
from pydantic import ValidationError


@pytest.mark.asyncio
@pytest.mark.parametrize("network", ["host", "container:unrelated-primary", "unapproved-network", "", " host "])
@pytest.mark.parametrize("persisted", [True, False])
async def test_client_network_rejected_before_container_creation(
    docker_backend: DockerDeploymentBackend,
    mock_entities: AsyncMock,
    mock_docker_client: MagicMock,
    network: str,
    persisted: bool,
) -> None:
    # Include an init container and sidecar so neither can run before the guard.
    config = lora_config()
    if persisted:
        # Old entity rows remain loadable, but cannot bypass launch-time policy.
        config.backend_config = DeploymentBackendConfig.model_validate({"docker": {"network": network}})
    mock_entities.get.return_value = config
    mock_docker_client.containers.get.side_effect = NotFound("missing")

    update = await docker_backend.create_deployment(
        workspace="default",
        name="srv",
        config_name="cfg1",
        labels={},
        backend_config={} if persisted else {"docker": {"network": network}},
    )

    assert update.status == "FAILED"
    assert "network" in update.status_message.lower()
    mock_docker_client.containers.create.assert_not_called()
    mock_docker_client.images.pull.assert_not_called()
    mock_docker_client.volumes.create.assert_not_called()


@pytest.mark.parametrize("network", ["host", "HOST", "container:other", "", " bridge "])
@pytest.mark.parametrize("alias", ["backendConfig", "backend_config"])
def test_request_rejects_namespace_modes(network: str, alias: str) -> None:
    with pytest.raises(ValidationError, match="network override"):
        CreateDeploymentConfigRequest.model_validate({"name": "cfg", alias: {"docker": {"network": network}}})
    # Do not apply request-only restrictions to persisted entities.
    assert DockerDeploymentConfig.model_validate({"network": network}).network == network


@pytest.mark.parametrize("network", [None, "workloads", "bridge", "none"])
def test_request_accepts_names_for_later_executor_authorization(network: str | None) -> None:
    request = CreateDeploymentConfigRequest.model_validate(
        {"name": "cfg", "backendConfig": {"docker": {"network": network}}}
    )
    assert request.backend_config.docker is not None
    assert request.backend_config.docker.network == network


def test_executor_network_policy() -> None:
    executor = DockerExecutorConfig(network="workloads", allowed_networks=["other-workloads"])
    assert executor.resolve_network(None) == "workloads"
    assert executor.resolve_network("workloads") == "workloads"
    assert executor.resolve_network("other-workloads") == "other-workloads"
    with pytest.raises(ValueError, match="not allowed"):
        executor.resolve_network("unknown")
    with pytest.raises(ValueError, match="not allowed"):
        DockerExecutorConfig(network="separate").resolve_network("workloads")
    assert DockerExecutorConfig().resolve_network(None) is None
    # Docker built-ins are not implicit client permissions.
    for network in ("bridge", "none", "default"):
        with pytest.raises(ValueError, match="not allowed"):
            DockerExecutorConfig().resolve_network(network)
        assert DockerExecutorConfig(allowed_networks=[network]).resolve_network(network) == network


@pytest.mark.parametrize("network", ["host", "container:other", "", " bridge "])
def test_allowlist_cannot_enable_namespace_modes(network: str) -> None:
    with pytest.raises(ValidationError, match="network override"):
        DockerExecutorConfig(allowed_networks=[network])


@pytest.mark.parametrize("network", ["host", "container:other"])
def test_trusted_operator_default_does_not_authorize_client_namespace_modes(network: str) -> None:
    # An operator default is a separate trust decision; equality never bypasses
    # request validation. Internal sidecar namespace joins do not use this path.
    executor = DockerExecutorConfig(network=network)
    assert executor.resolve_network(None) == network
    with pytest.raises(ValueError, match="network override"):
        executor.resolve_network(network)


@pytest.mark.asyncio
@pytest.mark.parametrize("network", [None, "workloads", "approved"])
async def test_allowed_primary_network_reaches_docker(
    docker_backend: DockerDeploymentBackend,
    mock_entities: AsyncMock,
    mock_docker_client: MagicMock,
    network: str | None,
) -> None:
    docker_backend._executor_config = DockerExecutorConfig(network="workloads", allowed_networks=["approved"])
    config = sample_config()
    config.backend_config = DeploymentBackendConfig(docker=DockerDeploymentConfig(network=network))
    mock_entities.get.return_value = config
    mock_docker_client.containers.get.side_effect = NotFound("missing")
    mock_docker_client.containers.create.return_value = MagicMock(id="simulated")

    update = await docker_backend.create_deployment(
        workspace="default", name="srv", config_name="cfg1", labels={}, backend_config={}
    )

    assert update.status == "STARTING"
    assert mock_docker_client.containers.create.call_args.kwargs["network"] == (network or "workloads")
