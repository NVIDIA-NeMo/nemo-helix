<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NeMo Deployments Plugin

Backend-agnostic deployment lifecycle for the NeMo Helix: entity schemas,
CRUD APIs, a `DeploymentBackend` ABC, an executor registry, and a background
reconcile controller (`DeploymentsController`).

## Controller

Register `DeploymentsController` via the `nemo.controllers` entry point. The controller
paginates non-terminal deployment/volume lists, reconciles volumes before deployments,
gates deployment create on mounted volumes reaching `BOUND`, and writes status via the
entity client (including endpoints and status history). Orphan backend resource cleanup
runs on a configurable interval and is skipped when the deployment list is unhealthy.

The controller exposes `is_healthy`, which is `False` when either the deployment-list or
volume-list query fails. Internally these are tracked separately so operators can tell
which list query failed without losing that signal behind a single boolean.

Per-config drift backoff overrides live on `DeploymentConfig.driftRecovery`; unset fields
fall back to `DeploymentsConfig.controller`.

Prerequisites are declared on each `Deployment` and reference other deployment
names in the same workspace (bare name or `workspace/name`). The controller loads
terminal prerequisite deployments from the entity store when they are not in the
active non-terminal list.

## Docker executors

Host port allocation for published container ports is configured per named docker
executor (not on `DeploymentConfig.backend_config.docker`). Set the inclusive
`port_range_start` / `port_range_end` bounds on the executor `config` block in
platform YAML. The allocator scans every host port from `port_range_start`
through `port_range_end`, including both endpoints (for example, 49152–49251
allows 100 ports):

```yaml
deployments:
  executors:
    - name: local-docker
      backend: docker
      config:
        port_range_start: 49152
        port_range_end: 49251  # inclusive
  default_executor: local-docker
```

A port is considered taken when any container on the daemon publishes it — not just
platform-managed ones — or when the host socket is already bound. Docker's own
reservations are not observable until a publish is attempted, so a port claimed
concurrently is retried against a different port rather than failing the deployment.
Pick a range nothing else on the host publishes on.

Entity-level `backend_config.docker` accepts only deployment-specific overrides such
as `network`.

## Docker resource naming

Docker container and volume names use a readable prefix plus a deterministic
8-character hash suffix. The hash is computed from ``{workspace}/{name}``, not
from the hyphen-joined string, so pairs like ``foo``/``bar-baz`` and
``foo-bar``/``baz`` cannot collide. Naming logic is shared via
``nemo_helix_plugin.k8s_naming`` (same module used by the models service).
Orphan cleanup matches identity labels, not names alone; existing containers
keep their old names after upgrade.

## Kubernetes executors

The k8s backend emits native `apps/v1.Deployment` + `v1.Service` for
`restart_policy: Always` workloads, `batch/v1.Job` for finite (`Never`/
`OnFailure`) workloads, and `v1.PersistentVolumeClaim` for volumes — no
`k8s-nim-operator` dependency. Configure a named executor in platform YAML:

```yaml
deployments:
  executors:
    - name: local-k8s
      backend: k8s
      config:
        kubeconfig_path: /path/to/kubeconfig  # unset: in-cluster config, then default kubeconfig
        default_namespace: default  # namespace the controller's ServiceAccount has RBAC in
        request_timeout: 60
  default_executor: local-k8s
```

Entity-level `backend_config.k8s.namespace` overrides `default_namespace` per
deployment/volume; it must be a namespace the controller's ServiceAccount has
RBAC in (see below).

### RBAC

The `DeploymentsController` runs inside the `nhx-core` controller pod
(registered via the `nemo.controllers` entry point), so it reuses that pod's
existing ServiceAccount and Role rather than a dedicated one. The deploy
chart's `k8s/helm/templates/core/controller-role.yaml` grants that Role the
verbs the k8s backend needs in `.Release.Namespace`: `get`/`list`/`watch` on
pods, `get`/`list` on pods/log, `create`/`get`/`list`/`watch`/`update`/`patch`/`delete` on
`batch/v1.Job`, `get`/`list`/`create`/`delete` on PVCs, ConfigMaps, and
Services, and `get`/`list`/`watch`/`create`/`delete` on `apps/v1.Deployment`.

This Role is namespace-scoped to the release namespace. Pointing
`backend_config.k8s.namespace` at a namespace outside the release namespace
requires additional RBAC that the chart does not provision today;
namespace-per-workspace provisioning is a documented future enhancement.

### Native sidecars

The LoRA-adapter-style native sidecar pattern (an `init_containers` entry with
`restart_policy: "Always"`) requires Kubernetes >= 1.29. On older clusters,
omit the per-container `restart_policy` on init containers and run that
container as a regular main container instead — the compiler does not
fall back to legacy sidecar emulation (emptyDir readiness files, etc.)
automatically.

## Cluster validation

`tests/integration/backends/k8s/` and `tests/integration/test_reconcile_k8s.py`
exercise the k8s backend and reconciler against a real cluster. Like the Docker
integration tests above, they skip automatically (CI stays green) when no
cluster is reachable.

**Prerequisites:**

- `kind` and `kubectl` on your `PATH` (for the local run) or an existing
  cluster context (for the dev-blue run).
- `uv` for running the test suite.
- A kubeconfig pointing at a reachable cluster — either the one `kind create
  cluster` sets automatically, or your own context via `kubectl config
  use-context`.

**Local run against kind** (no dev-blue access needed):

```bash
kind create cluster  # or reuse an existing one; sets your kubeconfig context
uv run pytest plugins/nemo-deployments/tests/integration/backends/k8s -v
uv run pytest plugins/nemo-deployments/tests/integration/test_reconcile_k8s.py -v
```

**Run against dev-blue** (uses your namespace and your own RBAC there):

```bash
kubectl config use-context <dev-blue>
export NHX_K8S_ITEST_NAMESPACE=<your-namespace>  # defaults to "default"
uv run pytest plugins/nemo-deployments/tests/integration/backends/k8s -v
```

Tests authenticate with whatever kubeconfig identity is currently active in
your shell, not the restricted `nhx-core` controller ServiceAccount, so they
validate backend behavior against a real API server but do **not** exercise
the RBAC scope granted by the deploy chart's `controller-role.yaml`. That
still needs a manual smoke test of the deployed platform pod.

`test_reconcile_k8s.py` covers prerequisite gating (one deployment blocking on
another) but not PVC-mount gating: kind's default `local-path` StorageClass uses
`WaitForFirstConsumer` binding, so an unconsumed PVC never reaches `BOUND`, and
`DeploymentReconciler` requires a mounted volume to already be `BOUND` before it
creates the pod that would consume it — a chicken-and-egg specific to
`WaitForFirstConsumer` storage classes that doesn't arise on `Immediate`-binding
classes (the common case outside kind). The standalone PVC lifecycle scenario in
`test_k8s_backend.py` accepts `PENDING` as a valid terminal state for this reason.
