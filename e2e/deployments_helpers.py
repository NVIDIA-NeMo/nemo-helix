# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for nemo-deployments plugin e2e tests.

Both the Docker (``test_nemo_deployments_docker.py``) and Kubernetes
(``test_nemo_deployments_k8s.py``) modules drive the *deployments plugin's own*
public API end to end — DeploymentConfig / Deployment / Volume CRUD plus the
reconcile controller that turns those entities into real backend resources
(Docker containers+volumes, or Kubernetes Job/Deployment+Service+PVC).

The chain they prove::

    sdk._client POST /apis/deployments/v2/.../deployment-configs   (template)
    sdk._client POST /apis/deployments/v2/.../volumes              (optional PVC/volume)
    sdk._client POST /apis/deployments/v2/.../deployments          (desired state)
      -> deployments reconcile controller
      -> executor backend (docker | k8s) creates the real workload
      -> Deployment.status converges (READY for services, SUCCEEDED for jobs)

Unlike ``e2e/agents_deploy_helpers.py`` (which goes through the higher-level
``sdk.agents`` surface), the deployments plugin is not exposed on the typed SDK,
so this module talks to the REST API directly via ``sdk._client``. The
per-backend modules own only what genuinely differs: pytest markers, the backend
key passed to volume/deployment ``backend_config``, and any best-effort reaping
of leaked backend resources.
"""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from nemo_helix import NeMoHelix

# Small, widely-cached public images used by the deployment workloads. ``alpine``
# runs a one-shot job (restart_policy=Never -> SUCCEEDED); ``nginx`` runs a
# long-lived service (restart_policy=Always -> READY with an endpoint).
#
# These default to fully-qualified ``docker.io/library/...`` refs but are
# env-overridable, mirroring the ``POSTGRES_IMAGE`` / ``BUSYBOX_IMAGE`` knobs the
# kind e2e Helm install already exposes (see
# ``.github/actions/setup-kind-cluster/action.yaml`` and
# ``e2e/k8s/scripts/install_helm_e2e.sh``). The workspace has no transparent
# DockerHub pull-through cache today, so if one is ever introduced (it would
# require the mirror registry to be named explicitly in the ref), CI can point
# these at it without a code change — exactly as it can for postgres/busybox.
ALPINE_IMAGE = os.environ.get("NHX_E2E_DEPLOYMENTS_ALPINE_IMAGE", "docker.io/library/alpine:3.20")
NGINX_IMAGE = os.environ.get("NHX_E2E_DEPLOYMENTS_NGINX_IMAGE", "docker.io/library/nginx:alpine")

# Terminal deployment statuses (the reconciler will not transition out of these).
_TERMINAL_DEPLOYMENT_STATUSES = frozenset({"SUCCEEDED", "FAILED", "LOST"})


def unique_name(prefix: str) -> str:
    return f"e2e-{prefix}-{uuid.uuid4().hex[:8]}"


def _base(workspace: str) -> str:
    return f"/apis/deployments/v2/workspaces/{workspace}"


# ---- Raw API wrappers ------------------------------------------------------


def create_deployment_config(
    sdk: NeMoHelix,
    *,
    workspace: str,
    name: str,
    containers: list[dict[str, Any]],
    restart_policy: str = "Always",
    volume_mounts: list[dict[str, Any]] | None = None,
    config_files: list[dict[str, Any]] | None = None,
    backend_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "name": name,
        "containers": containers,
        "restart_policy": restart_policy,
    }
    if volume_mounts is not None:
        body["volume_mounts"] = volume_mounts
    if config_files is not None:
        body["config_files"] = config_files
    if backend_config is not None:
        body["backend_config"] = backend_config
    response = sdk._client.post(f"{_base(workspace)}/deployment-configs", json=body)
    response.raise_for_status()
    return response.json()


def create_volume(
    sdk: NeMoHelix,
    *,
    workspace: str,
    name: str,
    size: str = "1Gi",
    access_modes: list[str] | None = None,
    backend_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {"name": name, "size": size}
    if access_modes is not None:
        body["access_modes"] = access_modes
    if backend_config is not None:
        body["backend_config"] = backend_config
    response = sdk._client.post(f"{_base(workspace)}/volumes", json=body)
    response.raise_for_status()
    return response.json()


def create_deployment(
    sdk: NeMoHelix,
    *,
    workspace: str,
    name: str,
    deployment_config: str,
    desired_state: str = "READY",
    executor: str | None = None,
    prerequisites: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "name": name,
        "deployment_config": deployment_config,
        "desired_state": desired_state,
    }
    if executor is not None:
        body["executor"] = executor
    if prerequisites is not None:
        body["prerequisites"] = prerequisites
    response = sdk._client.post(f"{_base(workspace)}/deployments", json=body)
    response.raise_for_status()
    return response.json()


def get_deployment(sdk: NeMoHelix, *, workspace: str, name: str) -> dict[str, Any]:
    response = sdk._client.get(f"{_base(workspace)}/deployments/{name}")
    response.raise_for_status()
    return response.json()


def get_volume(sdk: NeMoHelix, *, workspace: str, name: str) -> dict[str, Any]:
    response = sdk._client.get(f"{_base(workspace)}/volumes/{name}")
    response.raise_for_status()
    return response.json()


def list_deployments(sdk: NeMoHelix, *, workspace: str) -> list[dict[str, Any]]:
    response = sdk._client.get(f"{_base(workspace)}/deployments", params={"page_size": 100})
    response.raise_for_status()
    data = response.json().get("data", [])
    assert isinstance(data, list)
    return data


def delete_deployment_if_exists(sdk: NeMoHelix, *, workspace: str, name: str) -> None:
    try:
        response = sdk._client.delete(f"{_base(workspace)}/deployments/{name}")
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code != 404:
            raise


def delete_volume_if_exists(sdk: NeMoHelix, *, workspace: str, name: str) -> None:
    try:
        response = sdk._client.delete(f"{_base(workspace)}/volumes/{name}")
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code != 404:
            raise


def delete_deployment_config_if_exists(sdk: NeMoHelix, *, workspace: str, name: str) -> None:
    try:
        response = sdk._client.delete(f"{_base(workspace)}/deployment-configs/{name}")
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code != 404:
            raise


# ---- Wait helpers ----------------------------------------------------------


def wait_for_deployment_status(
    sdk: NeMoHelix,
    *,
    workspace: str,
    name: str,
    target_statuses: tuple[str, ...],
    timeout_seconds: float = 240,
) -> dict[str, Any]:
    """Poll a deployment until it reaches one of ``target_statuses``.

    Fails fast if the deployment lands in a terminal status that was not one of
    the requested targets (e.g. FAILED while waiting for READY), surfacing the
    ``status_message``/``error_details`` to make debugging tractable.
    """
    deadline = time.monotonic() + timeout_seconds
    last: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        deployment = get_deployment(sdk, workspace=workspace, name=name)
        last = deployment
        status = deployment["status"]
        if status in target_statuses:
            return deployment
        if status in _TERMINAL_DEPLOYMENT_STATUSES and status not in target_statuses:
            pytest.fail(
                f"Deployment {name!r} reached unexpected terminal status {status!r} "
                f"while waiting for {target_statuses}: "
                f"message={deployment.get('status_message')!r} "
                f"error_details={deployment.get('error_details')!r}"
            )
        time.sleep(2)
    pytest.fail(f"Deployment {name!r} did not reach {target_statuses} within {timeout_seconds}s: {last}")


def wait_for_volume_status(
    sdk: NeMoHelix,
    *,
    workspace: str,
    name: str,
    target_statuses: tuple[str, ...],
    timeout_seconds: float = 120,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        volume = get_volume(sdk, workspace=workspace, name=name)
        last = volume
        if volume["status"] in target_statuses:
            return volume
        if volume["status"] == "FAILED" and "FAILED" not in target_statuses:
            pytest.fail(f"Volume {name!r} FAILED while waiting for {target_statuses}: {volume.get('status_message')!r}")
        time.sleep(2)
    pytest.fail(f"Volume {name!r} did not reach {target_statuses} within {timeout_seconds}s: {last}")


def wait_for_deployment_deleted(
    sdk: NeMoHelix,
    *,
    workspace: str,
    name: str,
    timeout_seconds: float = 120,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_status: str | None = None
    while time.monotonic() < deadline:
        try:
            deployment = get_deployment(sdk, workspace=workspace, name=name)
            last_status = deployment.get("status")
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return
            raise
        time.sleep(2)
    pytest.fail(f"Deployment {name!r} was not deleted within {timeout_seconds}s; last status={last_status!r}")


def wait_for_volume_deleted(
    sdk: NeMoHelix,
    *,
    workspace: str,
    name: str,
    timeout_seconds: float = 120,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_status: str | None = None
    while time.monotonic() < deadline:
        try:
            volume = get_volume(sdk, workspace=workspace, name=name)
            last_status = volume.get("status")
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return
            raise
        time.sleep(2)
    pytest.fail(f"Volume {name!r} was not deleted within {timeout_seconds}s; last status={last_status!r}")


# ---- Backend-agnostic scenario cores ---------------------------------------


def run_service_deployment_lifecycle(
    sdk: NeMoHelix,
    *,
    workspace: str,
    backend_key: str,
    deployment_backend_config: dict[str, Any] | None = None,
    running_timeout_seconds: float = 240,
    reap_backend_resources: Callable[[str], None] | None = None,
) -> None:
    """Deploy a long-lived nginx service and assert it reconciles to READY.

    Proves the full create-template -> create-deployment -> reconcile -> READY
    chain for a ``restart_policy=Always`` service, including that the backend
    surfaces a routable endpoint. Cleans up the deployment and config (best
    effort, isolated steps) and, on backends that support it, reaps any leaked
    workload via ``reap_backend_resources``.

    ``backend_key`` (``"docker"`` / ``"k8s"``) only affects the optional
    ``deployment_backend_config`` the caller passes through; the reconcile path
    is otherwise identical across backends.
    """
    config_name = unique_name("svc-cfg")
    deployment_name = unique_name("svc")

    # Everything that creates a resource lives inside the try so the finally
    # cleanup runs even if config creation fails partway.
    try:
        create_deployment_config(
            sdk,
            workspace=workspace,
            name=config_name,
            restart_policy="Always",
            containers=[
                {
                    "name": "main",
                    "image": NGINX_IMAGE,
                    "ports": [{"containerPort": 80, "protocol": "TCP", "name": "http"}],
                }
            ],
            backend_config=deployment_backend_config,
        )

        created = create_deployment(
            sdk,
            workspace=workspace,
            name=deployment_name,
            deployment_config=config_name,
        )
        assert created["name"] == deployment_name
        assert created["deployment_config"] == config_name
        assert created["status"] == "PENDING"

        deployment = wait_for_deployment_status(
            sdk,
            workspace=workspace,
            name=deployment_name,
            target_statuses=("READY",),
            timeout_seconds=running_timeout_seconds,
        )
        # A long-lived service must expose at least one routable endpoint.
        endpoints = deployment.get("endpoints") or []
        assert endpoints and endpoints[0]["url"], deployment

        # It must also show up in the workspace listing while active.
        listed = {d["name"] for d in list_deployments(sdk, workspace=workspace)}
        assert deployment_name in listed
    finally:
        _safe(delete_deployment_if_exists, sdk, workspace=workspace, name=deployment_name)
        _safe(wait_for_deployment_deleted, sdk, workspace=workspace, name=deployment_name)
        if reap_backend_resources is not None:
            _safe(reap_backend_resources, deployment_name)
        _safe(delete_deployment_config_if_exists, sdk, workspace=workspace, name=config_name)


def run_job_deployment_lifecycle(
    sdk: NeMoHelix,
    *,
    workspace: str,
    backend_key: str,
    running_timeout_seconds: float = 240,
    reap_backend_resources: Callable[[str], None] | None = None,
) -> None:
    """Deploy a one-shot alpine job and assert it reconciles to SUCCEEDED.

    Proves the ``restart_policy=Never`` path: the workload runs to completion and
    the reconciler records the terminal SUCCEEDED status with exit code 0.
    """
    config_name = unique_name("job-cfg")
    deployment_name = unique_name("job")

    # Everything that creates a resource lives inside the try so the finally
    # cleanup runs even if config creation fails partway.
    try:
        create_deployment_config(
            sdk,
            workspace=workspace,
            name=config_name,
            restart_policy="Never",
            containers=[
                {
                    "name": "main",
                    "image": ALPINE_IMAGE,
                    "command": ["sh", "-c"],
                    "args": ["echo hello-from-deployments-e2e"],
                }
            ],
        )

        create_deployment(
            sdk,
            workspace=workspace,
            name=deployment_name,
            deployment_config=config_name,
        )
        deployment = wait_for_deployment_status(
            sdk,
            workspace=workspace,
            name=deployment_name,
            target_statuses=("SUCCEEDED",),
            timeout_seconds=running_timeout_seconds,
        )
        assert deployment.get("exit_code") == 0, deployment
    finally:
        _safe(delete_deployment_if_exists, sdk, workspace=workspace, name=deployment_name)
        _safe(wait_for_deployment_deleted, sdk, workspace=workspace, name=deployment_name)
        if reap_backend_resources is not None:
            _safe(reap_backend_resources, deployment_name)
        _safe(delete_deployment_config_if_exists, sdk, workspace=workspace, name=config_name)


def run_volume_deployment_round_trip(
    sdk: NeMoHelix,
    *,
    workspace: str,
    backend_key: str,
    volume_backend_config: dict[str, Any] | None = None,
    mount_path: str = "/data",
    running_timeout_seconds: float = 240,
    reap_backend_resources: Callable[[str], None] | None = None,
) -> None:
    """Prove a volume is provisioned, mounted, written to, and read back.

    1. Create a Volume and wait for it to reconcile to BOUND.
    2. Deploy a one-shot job whose DeploymentConfig mounts the volume, writes a
       sentinel file, then reads it back and asserts the content. If the mount
       did not work the ``grep`` fails and the container exits non-zero, so the
       deployment lands FAILED and the wait-for-SUCCEEDED fails the test.
    3. Delete the deployment, then the volume (deletion is blocked by referencing
       configs, so the config is dropped first in teardown).

    Backend portability: this requires the volume to reach BOUND *before* the
    mounting deployment starts, because ``DeploymentReconciler`` gates deployment
    create on all mounted volumes being BOUND (see ``volume_mounts_ready``). That
    holds on eagerly-binding storage (the docker backend binds immediately; k8s
    ``Immediate``-binding StorageClasses likewise). It does **not** hold on
    ``WaitForFirstConsumer`` storage (e.g. kind's default ``local-path``), where
    the PVC only binds once a consumer pod is scheduled — a chicken-and-egg with
    the reconciler's gate. This helper is therefore used by the docker module
    only; see ``test_nemo_deployments_k8s.py`` for why the k8s module omits it.
    """
    config_name = unique_name("vol-cfg")
    volume_name = unique_name("vol")
    deployment_name = unique_name("vol-job")
    sentinel = f"volume-payload-{uuid.uuid4().hex[:8]}"
    sentinel_file = f"{mount_path.rstrip('/')}/sentinel.txt"

    # Everything that creates a resource lives inside the try so the finally
    # cleanup runs even if volume creation, polling, or config creation fails
    # partway (otherwise a created volume/config would leak).
    try:
        create_volume(
            sdk,
            workspace=workspace,
            name=volume_name,
            size="1Gi",
            access_modes=["ReadWriteOnce"],
            backend_config=volume_backend_config,
        )

        # The reconciler gates the mounting deployment on the volume being BOUND,
        # and this helper only runs on eagerly-binding backends (docker), so
        # require BOUND up front rather than tolerating a lingering PENDING.
        wait_for_volume_status(
            sdk,
            workspace=workspace,
            name=volume_name,
            target_statuses=("BOUND",),
            timeout_seconds=120,
        )

        create_deployment_config(
            sdk,
            workspace=workspace,
            name=config_name,
            restart_policy="Never",
            volume_mounts=[{"name": volume_name, "mountPath": mount_path}],
            containers=[
                {
                    "name": "main",
                    "image": ALPINE_IMAGE,
                    "command": ["sh", "-c"],
                    "args": [
                        # Write a sentinel to the mounted volume then read it back
                        # and assert its content, exiting non-zero (=> FAILED) on
                        # mismatch.
                        f"set -e; echo {sentinel} > {sentinel_file}; grep -q {sentinel} {sentinel_file}; "
                        f"echo mount-verified",
                    ],
                    "volumeMounts": [{"name": volume_name, "mountPath": mount_path}],
                }
            ],
        )

        create_deployment(
            sdk,
            workspace=workspace,
            name=deployment_name,
            deployment_config=config_name,
        )
        deployment = wait_for_deployment_status(
            sdk,
            workspace=workspace,
            name=deployment_name,
            target_statuses=("SUCCEEDED",),
            timeout_seconds=running_timeout_seconds,
        )
        assert deployment.get("exit_code") == 0, deployment
    finally:
        _safe(delete_deployment_if_exists, sdk, workspace=workspace, name=deployment_name)
        _safe(wait_for_deployment_deleted, sdk, workspace=workspace, name=deployment_name)
        if reap_backend_resources is not None:
            _safe(reap_backend_resources, deployment_name)
        # The volume delete is blocked while a config still mounts it, so the
        # config must be dropped before the volume.
        _safe(delete_deployment_config_if_exists, sdk, workspace=workspace, name=config_name)
        _safe(delete_volume_if_exists, sdk, workspace=workspace, name=volume_name)
        _safe(wait_for_volume_deleted, sdk, workspace=workspace, name=volume_name)


def _safe(fn: Any, *args: Any, **kwargs: Any) -> None:
    try:
        fn(*args, **kwargs)
    except Exception:
        pass
