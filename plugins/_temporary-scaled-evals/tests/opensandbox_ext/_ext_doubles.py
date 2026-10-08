# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared inputs for the OpenSandbox server extension tests: an example spec, a stock manifest, and pods.

Only depends on the kubernetes client models, so the spec and pod-builder tests run without
opensandbox-server installed.
"""

from __future__ import annotations

import json
from typing import Any

from kubernetes.client import (
    V1ContainerState,
    V1ContainerStateRunning,
    V1ContainerStateTerminated,
    V1ContainerStateWaiting,
    V1ContainerStatus,
    V1ObjectMeta,
    V1Pod,
    V1PodStatus,
)
from nemo_ext_pod import ROLES_ANNOTATION
from nemo_ext_spec import Role

NAMESPACE = "nemo"
SPEC_ROLES: dict[str, Role] = {"db": "sidecar", "migrate": "run_once", "proxy": "sidecar", "worker": "after_main"}


def spec_dict(**overrides: Any) -> dict[str, Any]:
    """A services spec with one service of each role, a shared volume, and sandbox options.

    ``proxy`` listens on 18080, the egress filter's default port, so the egress port has to move.
    """
    spec: dict[str, Any] = {
        "version": 1,
        "volumes": ["shared"],
        "services": [
            {
                "name": "db",
                "image": "docker.io/library/postgres:16",
                "env": {"POSTGRES_PASSWORD": "pw"},
                "ports": [5432],
                "readiness": {"exec": ["pg_isready", "-U", "postgres"]},
                "volume_mounts": [{"name": "shared", "mount_path": "/data"}],
            },
            {"name": "migrate", "image": "docker.io/library/busybox:1", "command": ["true"], "run_once": True},
            {
                "name": "proxy",
                "image": "docker.io/library/nginx:1",
                "ports": [18080],
                "readiness": {"exec": ["curl", "-sf", "http://127.0.0.1:18080/healthz"]},
                "shm_size": "64Mi",
            },
            {
                "name": "worker",
                "image": "docker.io/library/redis:7",
                "after_main": True,
                "readiness": {"exec": ["redis-cli", "ping"]},
            },
        ],
        "sandbox": {"volume_mounts": [{"name": "shared", "mount_path": "/shared"}], "add_capabilities": ["SYS_PTRACE"]},
    }
    spec.update(overrides)
    return spec


def stock_batchsandbox(*, egress: bool = True) -> dict[str, Any]:
    """The parts of the BatchSandbox manifest opensandbox-server 0.2.1 builds that the pod builder touches.

    Trimmed from the real output, which the server integration tests also cover. With
    ``egress=False`` it is the manifest of a sandbox without a network policy.
    """
    bin_mount = {"name": "opensandbox-bin", "mountPath": "/opt/opensandbox"}
    containers: list[dict[str, Any]] = [
        {
            "name": "sandbox",
            "image": "registry.example.com/task/main:1",
            "imagePullPolicy": "IfNotPresent",
            "volumeMounts": [bin_mount],
            "securityContext": {"capabilities": {"drop": ["NET_ADMIN"]}},
        }
    ]
    if egress:
        containers.append(
            {
                "name": "egress",
                "image": "registry.example.com/opensandbox/egress:v1",
                "env": [{"name": "OPENSANDBOX_EGRESS_MODE", "value": "dns"}],
                "securityContext": {"capabilities": {"add": ["NET_ADMIN"]}},
                "ports": [{"name": "egress-api", "containerPort": 18080}],
                "readinessProbe": {"httpGet": {"path": "/healthz", "port": 18080}, "periodSeconds": 1},
                "volumeMounts": [bin_mount],
            }
        )

    return {
        "apiVersion": "sandbox.opensandbox.io/v1alpha1",
        "kind": "BatchSandbox",
        "metadata": {"name": "sbx-1", "namespace": NAMESPACE},
        "spec": {
            "replicas": 1,
            "template": {
                "metadata": {"annotations": {}},
                "spec": {
                    "initContainers": [
                        {"name": "execd-installer", "image": "registry.example.com/opensandbox/execd:v1"}
                    ],
                    "containers": containers,
                    "volumes": [{"name": "opensandbox-bin", "emptyDir": {}}],
                },
            },
        },
    }


def by_name(containers: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index container manifests by name."""
    return {c["name"]: c for c in containers}


# ---- pods --------------------------------------------------------------------------------


def running(name: str, *, ready: bool = True, restarts: int = 0) -> V1ContainerStatus:
    """Status of a running container."""
    return V1ContainerStatus(
        name=name,
        image="i",
        image_id="",
        ready=ready,
        restart_count=restarts,
        state=V1ContainerState(running=V1ContainerStateRunning()),
    )


def terminated(name: str, exit_code: int, *, restarts: int = 0, finished_at: Any = None) -> V1ContainerStatus:
    """Status of a container that exited with ``exit_code``."""
    return V1ContainerStatus(
        name=name,
        image="i",
        image_id="",
        ready=False,
        restart_count=restarts,
        state=V1ContainerState(
            terminated=V1ContainerStateTerminated(exit_code=exit_code, reason="Error", finished_at=finished_at)
        ),
    )


def waiting(name: str, reason: str) -> V1ContainerStatus:
    """Status of a container that hasn't started, for ``reason`` (e.g. ``ImagePullBackOff``)."""
    return V1ContainerStatus(
        name=name,
        image="i",
        image_id="",
        ready=False,
        restart_count=0,
        state=V1ContainerState(waiting=V1ContainerStateWaiting(reason=reason, message="no such image")),
    )


def pod(init: list[V1ContainerStatus], containers: list[V1ContainerStatus], phase: str = "Pending") -> V1Pod:
    """The sandbox pod of ``spec_dict()`` with the given container statuses."""
    return V1Pod(
        metadata=V1ObjectMeta(
            name="sbx-1-pod",
            namespace=NAMESPACE,
            annotations={ROLES_ANNOTATION: json.dumps(SPEC_ROLES)},
        ),
        status=V1PodStatus(phase=phase, init_container_statuses=init, container_statuses=containers),
    )


def all_ready() -> V1Pod:
    """A running pod in which every service of ``spec_dict()`` is ready or done."""
    return pod(
        [running("egress"), running("db"), terminated("migrate", 0), running("proxy")],
        [running("sandbox"), running("worker")],
        phase="Running",
    )
