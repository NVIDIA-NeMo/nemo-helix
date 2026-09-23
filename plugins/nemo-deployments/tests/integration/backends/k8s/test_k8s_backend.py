# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for K8sDeploymentBackend against a real cluster (kind or kubeconfig).

Namespace defaults to ``default`` (present on any cluster, and where a kind admin
kubeconfig has full rights). Point at a different namespace you control via
``NHX_K8S_ITEST_NAMESPACE`` — e.g. your dev-blue namespace, which only has the RBAC
verbs the deploy chart's ``controller-role.yaml`` grants (see AIRCORE-757 Phase 6).
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from kubeconfig_availability import skip_without_kubeconfig
from kubernetes.client.rest import ApiException
from nemo_deployments_plugin.backends.k8s.backend import K8sDeploymentBackend
from nemo_deployments_plugin.backends.k8s.client import k8s_client_module
from nemo_deployments_plugin.backends.labels import k8s_deployment_resource_name
from nemo_deployments_plugin.backends.registry import BACKEND_CLASSES
from nemo_deployments_plugin.constants import MANAGED_BY_LABEL
from nemo_deployments_plugin.entities import ConfigFile, Container, ContainerPort, Deployment, DeploymentConfig

pytestmark = [
    pytest.mark.skipif("k8s" not in BACKEND_CLASSES, reason="K8sDeploymentBackend not registered"),
    skip_without_kubeconfig,
]

NAMESPACE = os.environ.get("NHX_K8S_ITEST_NAMESPACE", "default")
LABELS = {"managed-by": MANAGED_BY_LABEL}
POLL_ATTEMPTS = 60
POLL_INTERVAL_SECONDS = 1
ALPINE_IMAGE = "docker.io/library/alpine:3.20"
NGINX_IMAGE = "docker.io/library/nginx:alpine"


@pytest.fixture
def mock_entities() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def k8s_backend(mock_entities: AsyncMock) -> Iterator[K8sDeploymentBackend]:
    mock_sdk = MagicMock()
    with (
        patch("nemo_deployments_plugin.backends.k8s.backend.client_from_platform"),
        patch("nemo_deployments_plugin.backends.k8s.backend.NemoEntitiesClient", return_value=mock_entities),
    ):
        backend = K8sDeploymentBackend(mock_sdk, {"default_namespace": NAMESPACE, "request_timeout": 30})
    backend._entities = mock_entities
    try:
        yield backend
    finally:
        backend.shutdown()


def _never_config(
    *,
    name: str = "echo-cfg",
    args: list[str] | None = None,
    config_files: list[ConfigFile] | None = None,
) -> DeploymentConfig:
    return DeploymentConfig(
        name=name,
        workspace="itest",
        restart_policy="Never",  # ty: ignore[unknown-argument]  # pyright: ignore[reportCallIssue]
        containers=[
            Container(name="main", image=ALPINE_IMAGE, command=["sh", "-c"], args=args or ["echo hello-from-k8s"])
        ],
        config_files=config_files or [],  # ty: ignore[unknown-argument]  # pyright: ignore[reportCallIssue]
    )


def _always_http_config() -> DeploymentConfig:
    return DeploymentConfig(
        name="http-cfg",
        workspace="itest",
        restart_policy="Always",  # ty: ignore[unknown-argument]  # pyright: ignore[reportCallIssue]
        containers=[
            Container(
                name="main",
                image=NGINX_IMAGE,
                ports=[ContainerPort(containerPort=80, protocol="TCP", name="http")],
            )
        ],
    )


def _configure_deployment_lookup(
    mock_entities: AsyncMock,
    *,
    name: str,
    config_name: str,
    config: DeploymentConfig,
    workspace: str = "itest",
) -> None:
    """Wire the mocked entity client to resolve both the Deployment and its DeploymentConfig.

    Unlike the docker backend (which derives container identity from workspace/name alone),
    the k8s backend's read/delete paths load the Deployment entity first to discover
    ``deployment_config``, then load that DeploymentConfig — so a single blanket
    ``get.return_value`` isn't enough here.
    """
    deployment = Deployment(name=name, workspace=workspace, deployment_config=config_name)

    async def get_side_effect(
        entity_type: type,
        entity_name: str,
        workspace: str | None = None,
    ) -> Deployment | DeploymentConfig:
        if entity_type is Deployment:
            return deployment
        if entity_type is DeploymentConfig:
            return config
        raise KeyError(entity_name)

    mock_entities.get.side_effect = get_side_effect


async def _wait_for_status(
    k8s_backend: K8sDeploymentBackend,
    name: str,
    *,
    terminal_statuses: tuple[str, ...],
    attempts: int = POLL_ATTEMPTS,
) -> None:
    for _ in range(attempts):
        status = await k8s_backend.read_status(workspace="itest", name=name)
        if status.status in terminal_statuses:
            return
        await asyncio.sleep(POLL_INTERVAL_SECONDS)
    pytest.fail(f"{name!r} did not reach {terminal_statuses} within {attempts} attempts")


@pytest.mark.asyncio
async def test_pvc_lifecycle(k8s_backend: K8sDeploymentBackend) -> None:
    created = await k8s_backend.create_volume(
        workspace="itest-pvc",
        name="data",
        size="1Gi",
        access_modes=["ReadWriteOnce"],
        backend_config={},
    )
    try:
        # Unconsumed PVCs stay Pending under kind's default WaitForFirstConsumer
        # storage class, so both Pending and Bound are valid terminal states here.
        assert created.status in {"PENDING", "BOUND"}

        read = await k8s_backend.read_volume_status(workspace="itest-pvc", name="data")
        assert read.status in {"PENDING", "BOUND"}
    finally:
        deleted = await k8s_backend.delete_volume("itest-pvc", "data")

    assert deleted.status == "RELEASED"


@pytest.mark.asyncio
async def test_never_job_succeeds(k8s_backend: K8sDeploymentBackend, mock_entities: AsyncMock) -> None:
    config = _never_config()
    _configure_deployment_lookup(mock_entities, name="echo-job", config_name="echo-cfg", config=config)

    try:
        created = await k8s_backend.create_deployment(
            workspace="itest",
            name="echo-job",
            config_name="echo-cfg",
            labels=LABELS,
            backend_config={},
        )
        assert created.status in {"STARTING", "SUCCEEDED"}

        await _wait_for_status(k8s_backend, "echo-job", terminal_statuses=("SUCCEEDED", "FAILED"))
        status = await k8s_backend.read_status(workspace="itest", name="echo-job")
        assert status.status == "SUCCEEDED"
        assert status.exit_code == 0
    finally:
        await k8s_backend.delete_deployment("itest", "echo-job")


@pytest.mark.asyncio
async def test_configmap_mount_round_trip(k8s_backend: K8sDeploymentBackend, mock_entities: AsyncMock) -> None:
    """A ConfigFile mounted via ConfigMap is readable inside the container."""
    config = _never_config(
        name="cm-cfg",
        args=["cat /etc/nemo-config/hello.txt"],
        config_files=[ConfigFile(path="/etc/nemo-config/hello.txt", content="hello-from-configmap")],
    )
    _configure_deployment_lookup(mock_entities, name="cm-job", config_name="cm-cfg", config=config)

    try:
        await k8s_backend.create_deployment(
            workspace="itest",
            name="cm-job",
            config_name="cm-cfg",
            labels=LABELS,
            backend_config={},
        )
        await _wait_for_status(k8s_backend, "cm-job", terminal_statuses=("SUCCEEDED", "FAILED"))
        status = await k8s_backend.read_status(workspace="itest", name="cm-job")
        assert status.status == "SUCCEEDED"

        logs = await k8s_backend.get_logs(workspace="itest", name="cm-job")
        assert any("hello-from-configmap" in line for line in logs.lines)
    finally:
        await k8s_backend.delete_deployment("itest", "cm-job")


@pytest.mark.asyncio
async def test_always_deployment_becomes_ready(k8s_backend: K8sDeploymentBackend, mock_entities: AsyncMock) -> None:
    config = _always_http_config()
    _configure_deployment_lookup(mock_entities, name="http-svc", config_name="http-cfg", config=config)
    resource_name = k8s_deployment_resource_name("itest", "http-svc")

    try:
        await k8s_backend.create_deployment(
            workspace="itest",
            name="http-svc",
            config_name="http-cfg",
            labels=LABELS,
            backend_config={},
        )

        await _wait_for_status(k8s_backend, "http-svc", terminal_statuses=("READY",))
        status = await k8s_backend.read_status(workspace="itest", name="http-svc")
        assert status.status == "READY"
        assert status.endpoints

        core_v1 = k8s_backend.clients.core_v1
        service = await asyncio.to_thread(
            core_v1.read_namespaced_service,
            name=resource_name,
            namespace=NAMESPACE,
        )
        assert getattr(service, "metadata").name == resource_name
    finally:
        await k8s_backend.delete_deployment("itest", "http-svc")


@pytest.mark.asyncio
async def test_delete_rejects_foreign_resource(k8s_backend: K8sDeploymentBackend, mock_entities: AsyncMock) -> None:
    """A resource with the plugin's expected name but no identity labels is not touched."""
    config = _never_config(name="foreign-cfg")
    _configure_deployment_lookup(mock_entities, name="foreign-job", config_name="foreign-cfg", config=config)
    resource_name = k8s_deployment_resource_name("itest", "foreign-job")

    batch_v1 = k8s_backend.clients.batch_v1
    k8s = k8s_client_module()
    foreign_job = k8s.client.V1Job(
        api_version="batch/v1",
        kind="Job",
        metadata=k8s.client.V1ObjectMeta(name=resource_name, labels={"owner": "someone-else"}),
        spec=k8s.client.V1JobSpec(
            template=k8s.client.V1PodTemplateSpec(
                spec=k8s.client.V1PodSpec(
                    restart_policy="Never",
                    containers=[k8s.client.V1Container(name="main", image=ALPINE_IMAGE, command=["true"])],
                )
            )
        ),
    )

    try:
        await asyncio.to_thread(
            batch_v1.create_namespaced_job,
            namespace=NAMESPACE,
            body=foreign_job,
        )

        result = await k8s_backend.delete_deployment("itest", "foreign-job")
        assert result.status == "FAILED"
        assert "not managed by this plugin" in result.status_message

        # Foreign resource must still exist.
        existing = await asyncio.to_thread(
            batch_v1.read_namespaced_job,
            name=resource_name,
            namespace=NAMESPACE,
        )
        assert getattr(existing, "metadata").name == resource_name
    finally:
        try:
            await asyncio.to_thread(
                batch_v1.delete_namespaced_job,
                name=resource_name,
                namespace=NAMESPACE,
                propagation_policy="Background",
            )
        except ApiException as exc:
            if exc.status != 404:
                raise
