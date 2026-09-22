# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for the nemo-deployments plugin against a real Docker daemon.

Where ``test_nemo_agents_docker.py`` exercises the *agents* surface (which uses
the deployments plugin under the hood for a single opinionated agent container),
this module exercises the deployments plugin's **own** public API directly:
DeploymentConfig / Deployment / Volume CRUD plus the reconcile controller that
turns those entities into real Docker containers and volumes. The
backend-agnostic scenario cores are shared with the Kubernetes variant
(``test_nemo_deployments_k8s.py``) via ``e2e.deployments_helpers``; this module
owns only the docker-specific wiring.

What it proves — the deployments reconcile chain end to end, on Docker::

    sdk._client POST /apis/deployments/v2/...   (config / volume / deployment)
      -> deployments reconcile controller
      -> docker executor creates the container / named volume
      -> Deployment.status converges (READY for the nginx service,
         SUCCEEDED for the alpine job and the volume round-trip job)

How it runs, and where:

- The platform runs as a normal local process (subprocess harness) wired with a
  nemo-deployments Docker executor (see ``e2e/configs/local-docker-deployments.yaml``).
  The harness runs both the deployments service and its reconcile controller.
- The workloads use small public images (``alpine`` / ``nginx``); the executor
  pulls them on demand (``pull_images: true`` in the config), so no prebuilt
  ``nhx-api`` image is needed here — hence no ``needs_nhx_api_image`` marker.
  The image refs are env-overridable (see ``e2e.deployments_helpers``) to match
  the ``POSTGRES_IMAGE`` / ``BUSYBOX_IMAGE`` knobs the k8s e2e install exposes,
  should a DockerHub mirror ever be introduced.
- ``subprocess_only``: this module drives its own subprocess-harness platform
  configured with a docker deployments executor. It must NOT run against an
  external cluster (``NHX_BASE_URL`` set), where its ``e2e_config`` / harness are
  ignored and no docker executor exists.
- Docker-only workloads run on the host daemon directly. Unlike the agents docker
  test, nothing here needs the docker-bridge base-url rewrite (no in-container
  callback to the platform), so there is no ``container_base_url_host`` harness
  option and no Linux-only skip is strictly required — but a working Docker
  daemon is. The module skips cleanly if the daemon is unreachable.
"""

from __future__ import annotations

import pytest
from nemo_helix import NeMoHelix

from e2e.deployments_helpers import (
    run_job_deployment_lifecycle,
    run_service_deployment_lifecycle,
    run_volume_deployment_round_trip,
)

pytestmark = [
    pytest.mark.subprocess_only,
    pytest.mark.e2e_config(
        "e2e/configs/local-docker-deployments.yaml",
        harness={"backend": "subprocess"},
    ),
]


def _remove_deployment_container_if_present(deployment_name: str) -> None:
    """Best-effort removal of a leaked deployment container after teardown."""
    try:
        from docker.errors import NotFound

        import docker
    except Exception:
        return
    try:
        client = docker.from_env()
    except Exception:
        return
    # The docker backend names containers after the deployment (hashed identity);
    # match loosely so a naming-scheme change does not silently leak containers.
    for container in client.containers.list(all=True):
        if deployment_name in container.name:
            try:
                container.remove(force=True)
            except NotFound:
                pass
            except Exception:
                pass


def _skip_without_docker() -> None:
    try:
        import docker
    except Exception:
        pytest.skip("docker SDK not importable")
        return
    try:
        client = docker.from_env()
        client.ping()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Docker daemon not reachable: {exc}")


def test_docker_service_deployment_reaches_ready(sdk: NeMoHelix, workspace: str) -> None:
    """A restart_policy=Always nginx service reconciles to READY with an endpoint."""
    _skip_without_docker()
    run_service_deployment_lifecycle(
        sdk,
        workspace=workspace,
        backend_key="docker",
        reap_backend_resources=_remove_deployment_container_if_present,
    )


def test_docker_job_deployment_reaches_succeeded(sdk: NeMoHelix, workspace: str) -> None:
    """A restart_policy=Never alpine job runs to completion (SUCCEEDED, exit 0)."""
    _skip_without_docker()
    run_job_deployment_lifecycle(
        sdk,
        workspace=workspace,
        backend_key="docker",
        reap_backend_resources=_remove_deployment_container_if_present,
    )


def test_docker_volume_is_provisioned_mounted_and_readable(sdk: NeMoHelix, workspace: str) -> None:
    """A named volume is provisioned, mounted into a job, written to, and read back."""
    _skip_without_docker()
    run_volume_deployment_round_trip(
        sdk,
        workspace=workspace,
        backend_key="docker",
        reap_backend_resources=_remove_deployment_container_if_present,
    )
