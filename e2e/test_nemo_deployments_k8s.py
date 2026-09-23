# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for the nemo-deployments plugin on Kubernetes.

The Kubernetes counterpart to ``test_nemo_deployments_docker.py``: it drives the
deployments plugin's own public API (DeploymentConfig / Deployment) and asserts
the reconcile controller turns those entities into real Kubernetes workloads — a
Deployment+Service for the long-lived nginx service and a Job for the one-shot
alpine workload. The backend-agnostic scenario cores are shared with the docker
variant via ``e2e.deployments_helpers``; this module owns only the k8s-specific
wiring.

What it proves — the deployments reconcile chain end to end, on Kubernetes::

    sdk._client POST /apis/deployments/v2/...   (config / deployment)
      -> deployments reconcile controller
      -> k8s executor creates the Deployment+Service / Job
      -> Deployment.status converges (READY for the service, SUCCEEDED for the job)

No volume round-trip here (unlike the docker module). The reconciler gates a
deployment's create on every mounted Volume already being ``BOUND`` (see
``volume_mounts_ready``), but kind's default ``local-path`` StorageClass — which
the Kind CPU e2e job uses — binds ``WaitForFirstConsumer``: the PVC only binds
once a consuming pod is scheduled, and that pod is never created while the
deployment is gated. That chicken-and-egg only resolves on ``Immediate``-binding
storage (common outside kind, e.g. most cloud block-storage classes), so a
mounted-volume e2e is not portable to the kind CI environment. This mirrors the
existing k8s reconcile integration test, which deliberately omits a PVC mount for
the same reason (see ``plugins/nemo-deployments/tests/integration/test_reconcile_k8s.py``).
The docker module covers the full provision -> mount -> write -> read-back
round-trip, where volumes bind eagerly.

How it runs, and where:

- ``container_only``: this test only runs against an **external cluster**
  (``NHX_BASE_URL`` set) — the Kind CPU e2e CI job, whose Helm-deployed platform
  is configured with a nemo-deployments ``k8s`` executor (see
  ``e2e/k8s/values/kind.yaml``). It is skipped for the subprocess harness (local
  / plain e2e job), which has no k8s executor. This is the inverse of the docker
  module's ``subprocess_only``.
- The workloads use small public images (``alpine`` / ``nginx``) pulled by the
  kind nodes on demand, so — unlike the agents k8s test — this does not depend on
  the node-pre-pulled agent E2E image and carries no ``needs_agents_e2e_image``
  marker. Pulling public ``docker.io/library/...`` images at cluster runtime is
  the same pattern the kind e2e job already relies on for postgres / busybox /
  cloud-provider-kind (there is no pull-through cache configured). The refs are
  env-overridable (see ``e2e.deployments_helpers``) for parity with the
  ``POSTGRES_IMAGE`` / ``BUSYBOX_IMAGE`` install knobs.
- The workloads land in the executor's namespace (the Helm release namespace,
  beside the platform), reachable in-cluster.
- Pod scheduling and (internet) image pulls can take longer than the local
  docker path, so the scenario cores are given a wider timeout.
"""

from __future__ import annotations

import pytest
from nemo_helix import NeMoHelix

from e2e.deployments_helpers import (
    run_job_deployment_lifecycle,
    run_service_deployment_lifecycle,
)

# Pod scheduling + image pulls in a fresh cluster take longer than a local docker
# container start.
_K8S_TIMEOUT_SECONDS = 420

pytestmark = [pytest.mark.container_only]


def test_k8s_service_deployment_reaches_ready(sdk: NeMoHelix, workspace: str) -> None:
    """A restart_policy=Always nginx service reconciles to a k8s Deployment+Service (READY)."""
    run_service_deployment_lifecycle(
        sdk,
        workspace=workspace,
        backend_key="k8s",
        running_timeout_seconds=_K8S_TIMEOUT_SECONDS,
    )


def test_k8s_job_deployment_reaches_succeeded(sdk: NeMoHelix, workspace: str) -> None:
    """A restart_policy=Never alpine job reconciles to a k8s Job that completes (SUCCEEDED)."""
    run_job_deployment_lifecycle(
        sdk,
        workspace=workspace,
        backend_key="k8s",
        running_timeout_seconds=_K8S_TIMEOUT_SECONDS,
    )
