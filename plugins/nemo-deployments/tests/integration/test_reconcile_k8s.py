# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration test for deployment reconciliation against a real k8s backend.

Requires AIRCORE-757 K8sDeploymentBackend to be registered in BACKEND_CLASSES.

No volume/PVC mount here (unlike ``test_reconcile_docker.py``'s puller/server chain):
kind's default ``local-path`` StorageClass uses ``WaitForFirstConsumer`` binding, so an
unconsumed PVC never reaches BOUND, and ``DeploymentReconciler`` gates deployment create
on the mounted volume already being BOUND (see ``volume_mounts_ready``) — a chicken-and-egg
that only resolves on storage classes with ``Immediate`` binding (the common case outside
kind, e.g. most cloud block-storage classes). Prerequisite gating alone is exercised here.
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from kubeconfig_availability import skip_without_kubeconfig
from nemo_deployments_plugin.backends.k8s.backend import K8sDeploymentBackend
from nemo_deployments_plugin.backends.registry import BACKEND_CLASSES, ExecutorRegistry
from nemo_deployments_plugin.config import ControllerConfig
from nemo_deployments_plugin.entities import Container, ContainerPort, Deployment, DeploymentConfig, Prerequisite
from nemo_deployments_plugin.reconciler.deployment_reconciler import DeploymentReconciler

pytestmark = [
    pytest.mark.skipif("k8s" not in BACKEND_CLASSES, reason="Requires K8sDeploymentBackend (AIRCORE-757)"),
    skip_without_kubeconfig,
]

NAMESPACE = os.environ.get("NHX_K8S_ITEST_NAMESPACE", "default")
POLL_ATTEMPTS = 60
POLL_INTERVAL_SECONDS = 1
ALPINE_IMAGE = "docker.io/library/alpine:3.20"
NGINX_IMAGE = "docker.io/library/nginx:alpine"


@pytest.fixture
def k8s_registry() -> ExecutorRegistry:
    mock_sdk = MagicMock()
    with (
        patch("nemo_deployments_plugin.backends.k8s.backend.client_from_platform"),
        patch("nemo_deployments_plugin.backends.k8s.backend.NemoEntitiesClient"),
    ):
        backend = K8sDeploymentBackend(mock_sdk, {"default_namespace": NAMESPACE, "request_timeout": 30})
    return ExecutorRegistry({"k8s": backend}, default_executor="k8s")


def _backend(k8s_registry: ExecutorRegistry) -> K8sDeploymentBackend:
    backend = k8s_registry.resolve("k8s")
    assert isinstance(backend, K8sDeploymentBackend)
    return backend


@pytest.mark.asyncio
async def test_puller_server_prerequisite_chain(k8s_registry: ExecutorRegistry) -> None:
    """Puller (Never) -> server (Always + prerequisite) driven end-to-end by the reconciler."""
    entities = AsyncMock()
    entities.update = AsyncMock(side_effect=lambda entity: entity)

    puller_dep = Deployment(name="puller", workspace="itest", deployment_config="puller-cfg", status="PENDING")
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
        restart_policy="Never",  # ty: ignore[unknown-argument]  # pyright: ignore[reportCallIssue]
        containers=[Container(name="puller", image=ALPINE_IMAGE, command=["sh", "-c"], args=["echo pulled"])],
    )
    server_cfg = DeploymentConfig(
        name="server-cfg",
        workspace="itest",
        restart_policy="Always",  # ty: ignore[unknown-argument]  # pyright: ignore[reportCallIssue]
        containers=[Container(name="server", image=NGINX_IMAGE, ports=[ContainerPort(name="http", containerPort=80)])],
    )

    config_cache = {
        ("itest", "puller-cfg"): puller_cfg,
        ("itest", "server-cfg"): server_cfg,
    }

    async def get_side_effect(
        entity_type: type,
        name: str,
        workspace: str | None = None,
    ) -> Deployment | DeploymentConfig:
        if entity_type is Deployment:
            if name == "puller":
                return puller_dep
            if name == "server":
                return server_dep
            raise KeyError(name)
        if entity_type is DeploymentConfig:
            return config_cache[(workspace or "itest", name)]
        raise KeyError(name)

    entities.get.side_effect = get_side_effect

    backend = _backend(k8s_registry)
    backend._entities = entities

    deployment_reconciler = DeploymentReconciler(
        entities,
        k8s_registry,
        ControllerConfig(interval_seconds=1),
    )
    deployment_reconciler.set_config_cache(config_cache)

    by_name = {("itest", "puller"): puller_dep, ("itest", "server"): server_dep}

    try:
        # The "succeeded" prerequisite the server depends on requires both status and exit_code
        # (see prerequisite.py); reconcile_one stops polling the backend once status is terminal, so
        # wait for both together rather than breaking on status alone.
        for _ in range(POLL_ATTEMPTS):
            await deployment_reconciler.reconcile_one(puller_dep, deployments_by_name=by_name, volumes_by_name={})
            if puller_dep.status == "SUCCEEDED" and puller_dep.exit_code == 0:
                break
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
        assert puller_dep.status == "SUCCEEDED"
        assert puller_dep.exit_code == 0

        for _ in range(POLL_ATTEMPTS):
            await deployment_reconciler.reconcile_one(server_dep, deployments_by_name=by_name, volumes_by_name={})
            if server_dep.status in {"READY", "STARTING"}:
                break
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
        assert server_dep.status in {"READY", "STARTING"}
    finally:
        results = await asyncio.gather(
            backend.delete_deployment("itest", "puller"),
            backend.delete_deployment("itest", "server"),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, Exception):
                raise result
