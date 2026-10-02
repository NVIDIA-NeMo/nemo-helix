# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for deployment reconciliation.

Requires the DockerDeploymentBackend to be registered in BACKEND_CLASSES.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from docker.errors import NotFound
from docker_availability import skip_without_docker
from integration_helpers import force_remove_container
from nemo_deployments_plugin.backends.docker.backend import DockerDeploymentBackend
from nemo_deployments_plugin.backends.labels import container_name, docker_volume_name
from nemo_deployments_plugin.backends.registry import BACKEND_CLASSES, ExecutorRegistry
from nemo_deployments_plugin.config import ControllerConfig
from nemo_deployments_plugin.entities import (
    Container,
    Deployment,
    DeploymentConfig,
    Prerequisite,
    Volume,
    VolumeMount,
)
from nemo_deployments_plugin.reconciler.deployment_reconciler import DeploymentReconciler
from nemo_deployments_plugin.reconciler.volume_reconciler import VolumeReconciler

import docker

pytestmark = [
    pytest.mark.skipif(
        "docker" not in BACKEND_CLASSES,
        reason="Requires DockerDeploymentBackend",
    ),
    skip_without_docker,
]


@pytest.fixture
def docker_registry() -> ExecutorRegistry:
    mock_sdk = MagicMock()
    with (
        patch("nemo_deployments_plugin.backends.docker.backend.client_from_platform"),
        patch("nemo_deployments_plugin.backends.docker.backend.NemoEntitiesClient"),
        patch("nemo_deployments_plugin.backends.docker.backend.get_shared_gpu_pool", return_value=None),
    ):
        backend = DockerDeploymentBackend(mock_sdk, {"pull_images": True})
    return ExecutorRegistry({"docker": backend}, default_executor="docker")


def _backend(docker_registry: ExecutorRegistry) -> DockerDeploymentBackend:
    backend = docker_registry.resolve("docker")
    assert isinstance(backend, DockerDeploymentBackend)
    return backend


@pytest.mark.asyncio
async def test_puller_server_prerequisite_chain(docker_registry: ExecutorRegistry) -> None:
    """Volume → puller (OnFailure) → server (Always + prerequisite) end-to-end."""
    entities = AsyncMock()
    entities.update = AsyncMock(side_effect=lambda entity: entity)

    volume = Volume(name="weights", workspace="itest", size="1Gi", status="PENDING")
    puller_dep = Deployment(
        name="puller",
        workspace="itest",
        deployment_config="puller-cfg",
        status="PENDING",
    )
    server_dep = Deployment(
        name="server",
        workspace="itest",
        deployment_config="server-cfg",
        status="PENDING",
        prerequisites=[Prerequisite(deployment_name="puller", condition="succeeded")],
    )

    puller_cfg = DeploymentConfig(
        name="puller-cfg",
        workspace="itest",
        restart_policy="OnFailure",  # ty: ignore[unknown-argument]  # pyright: ignore[reportCallIssue]
        containers=[
            Container(
                name="puller",
                image="alpine:3.20",
                command=["sh", "-c"],
                args=["echo pulled > /data/ready && sleep 1"],
                volumeMounts=[VolumeMount(name="weights", mountPath="/data")],
            )
        ],
        volumeMounts=[VolumeMount(name="weights", mountPath="/data")],
    )
    server_cfg = DeploymentConfig(
        name="server-cfg",
        workspace="itest",
        restart_policy="Always",  # ty: ignore[unknown-argument]  # pyright: ignore[reportCallIssue]
        containers=[
            Container(
                name="server",
                image="alpine:3.20",
                command=["sleep"],
                args=["3600"],
                volumeMounts=[
                    VolumeMount(
                        name="weights",
                        mountPath="/data",
                        read_only=True,  # ty: ignore[unknown-argument]  # pyright: ignore[reportCallIssue]
                    )
                ],
            )
        ],
        volumeMounts=[VolumeMount(name="weights", mountPath="/data")],
    )

    config_cache = {
        ("itest", "puller-cfg"): puller_cfg,
        ("itest", "server-cfg"): server_cfg,
    }

    async def get_side_effect(
        entity_type: type,
        name: str,
        workspace: str | None = None,
    ) -> Volume | Deployment | DeploymentConfig:
        ws = workspace or "itest"
        if entity_type is Volume:
            return volume
        if entity_type is Deployment:
            if name == "puller":
                return puller_dep
            return server_dep
        if entity_type is DeploymentConfig:
            return config_cache[(ws, name)]
        raise KeyError(name)

    entities.get.side_effect = get_side_effect

    backend = _backend(docker_registry)
    backend._entities = entities

    volume_reconciler = VolumeReconciler(entities, docker_registry)
    deployment_reconciler = DeploymentReconciler(
        entities,
        docker_registry,
        ControllerConfig(interval_seconds=1),
    )
    deployment_reconciler.set_config_cache(config_cache)

    volumes_by_name = {("itest", "weights"): volume}
    by_name = {("itest", "puller"): puller_dep, ("itest", "server"): server_dep}

    try:
        await volume_reconciler.reconcile_one(volume)
        assert volume.status == "BOUND"
        volumes_by_name[("itest", "weights")] = volume

        for _ in range(40):
            await deployment_reconciler.reconcile_one(
                puller_dep,
                deployments_by_name=by_name,
                volumes_by_name=volumes_by_name,
            )
            if puller_dep.status == "SUCCEEDED":
                break
            await asyncio.sleep(0.5)
        assert puller_dep.status == "SUCCEEDED"

        for _ in range(40):
            await deployment_reconciler.reconcile_one(
                server_dep,
                deployments_by_name=by_name,
                volumes_by_name=volumes_by_name,
            )
            if server_dep.status in {"READY", "STARTING"}:
                break
            await asyncio.sleep(0.5)
        assert server_dep.status in {"READY", "STARTING"}
    finally:
        client = docker.from_env()
        await backend.delete_deployment("itest", "puller")
        await backend.delete_deployment("itest", "server")
        await backend.delete_volume("itest", "weights")
        for c_name in (container_name("itest", "puller"), container_name("itest", "server")):
            force_remove_container(client, c_name)
        try:
            client.volumes.get(docker_volume_name("itest", "weights")).remove(force=True)
        except NotFound:
            pass
