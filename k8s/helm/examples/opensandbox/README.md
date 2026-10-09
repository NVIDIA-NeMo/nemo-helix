<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# OpenSandbox example overlays

The NeMo Helix Helm chart does **not** install OpenSandbox. These files are
values and BatchSandbox templates for the upstream OpenSandbox charts. Point
jobs at **one** already-installed server.

Use OpenSandbox for **sandboxed GRPO / NeMo Gym**: untrusted custom environment
FileSets run in isolated pods, not in the training container. It does **not**
sandbox the rest of the platform (API, DPO, SFT, inference). The default path
is **shared-kernel** (cluster default OCI runtime: often runc on containerd, or
crun on CRI-O including OKE/OpenShift). **A kernel-isolated runtime is not
required.** Use Kata (or another isolated runtime) only when those Gym/GRPO
sandbox pods must be isolated from the host kernel. The documented example is
Kata QEMU because it runs each sandbox in a VM with its own guest kernel; other
isolated runtimes may work but have not been tested.

Full procedure: [OpenSandbox](https://docs.nvidia.com/nemo-helix/latest/documentation/kubernetes-deployment/setup/helm/open-sandbox)
and [OpenSandbox with Kata](https://docs.nvidia.com/nemo-helix/latest/documentation/kubernetes-deployment/setup/helm/opensandbox-kata)
in the NeMo Helix documentation. `helm show readme` of this chart points at
those pages.

## Prerequisites

- A local [OpenSandbox](https://github.com/opensandbox-group/OpenSandbox) checkout with Helm charts under `kubernetes/charts/`, or a published chart tarball
- `kubectl` access to the cluster
- The NeMo Helix Helm release namespace (jobs run here; it is also `[kubernetes] namespace` in the server values)

## Namespace rule

`[kubernetes] namespace` in the server TOML **must be the Helm release
namespace** (the namespace platform jobs run in). Control plane can stay in
`opensandbox-system`. PVC remounts and image-pull secrets do not work across
namespaces.

Replace `REPLACE_WITH_RELEASE_NAMESPACE` in the server values before install.

## Files

| File | Role |
|------|------|
| `opensandbox-controller.yaml` | Shared controller (snapshots unused on CRI-O) |
| `opensandbox-server.yaml` | Shared-kernel server (no `[secure_runtime]`) |
| `opensandbox-server-kata-qemu.yaml` | Kata QEMU server (`[secure_runtime] type=kata`) |
| `batchsandbox-template.yaml` | ConfigMap — exclude control-plane; pull Secret name `nvcrimagepullsecret`; [Harbor directories](#harbor-directories) |
| `batchsandbox-template-kata-qemu.yaml` | ConfigMap — example Kata node selectors; same pull Secret name; [Harbor directories](#harbor-directories) |
| `opensandbox-ext-rbac.yaml` | Role — `pods/exec` and `pods/log` in the release namespace for the [Compose services](#compose-services) extension |

`[secure_runtime]` is server-global. Install **one** server for production
(shared-kernel **or** Kata). Dual releases are only for proving both paths.

## Harbor directories

Both BatchSandbox templates mount writable `emptyDir` volumes at `/logs`
(2Gi), `/solution` (1Gi) and `/installed-agent` (2Gi) in the sandbox
container. Scaled-evals' `harbor_opensandbox` runtime needs them: Harbor
creates its log, solution and agent directories through `execd`, which runs as
the task image's `USER`, so an image with a non-root `USER` can't create them
on its own. Kubernetes creates an `emptyDir` world-writable, so the sandbox
doesn't have to run as root.

- Keep these mounts if you run Harbor tasks; removing them breaks every task
  image with a non-root `USER` at sandbox start (`mkdir /logs: permission
  denied`).
- `/tests` is not mounted on purpose: verifier images bake their tests into
  `/tests`, and an empty mount would hide them.
- The sizes count against node ephemeral storage. A sandbox that writes more
  than a volume's `sizeLimit` is evicted; raise the limits for long agent runs.
- The OpenSandbox server reads the template only at startup. After changing
  it, restart the server: `kubectl rollout restart deploy/opensandbox-server
  -n opensandbox-system`.

## Install (shared-kernel)

```bash
export NHX_NAMESPACE=nemo-helix          # must match the platform job namespace
export OPENSANDBOX_DIR=/path/to/OpenSandbox
export EXAMPLES=k8s/helm/examples/opensandbox

# Replace the placeholder in the server values
sed -i.bak "s/REPLACE_WITH_RELEASE_NAMESPACE/${NHX_NAMESPACE}/g" \
  "${EXAMPLES}/opensandbox-server.yaml"

kubectl create namespace opensandbox-system --dry-run=client -o yaml | kubectl apply -f -
kubectl create namespace "${NHX_NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -

kubectl apply -f "${EXAMPLES}/batchsandbox-template.yaml"

# API key in the control-plane namespace (server reads it)
kubectl create secret generic opensandbox-server-api-key \
  -n opensandbox-system --from-literal=api-key="$(openssl rand -hex 32)"

# Copy the same key into the job namespace (jobs use secretKeyRef)
kubectl get secret opensandbox-server-api-key -n opensandbox-system -o json \
  | jq 'del(.metadata.uid,.metadata.resourceVersion,.metadata.creationTimestamp,.metadata.namespace)' \
  | kubectl apply -n "${NHX_NAMESPACE}" -f -

# Image pull: templates hard-code imagePullSecrets.name: nvcrimagepullsecret.
# That Secret must exist in the job namespace. If yours has a different name,
# edit imagePullSecrets in batchsandbox-template.yaml (and the Kata template)
# before applying.
kubectl get secret nvcrimagepullsecret -n "${NHX_NAMESPACE}"

helm upgrade --install opensandbox-controller \
  "${OPENSANDBOX_DIR}/kubernetes/charts/opensandbox-controller" \
  --namespace opensandbox-system \
  -f "${EXAMPLES}/opensandbox-controller.yaml"

helm upgrade --install opensandbox-server \
  "${OPENSANDBOX_DIR}/kubernetes/charts/opensandbox-server" \
  --namespace opensandbox-system \
  -f "${EXAMPLES}/opensandbox-server.yaml"
```

Then set on the platform chart:

```yaml
sandboxClusterCapable: true
opensandbox:
  domain: opensandbox-server.opensandbox-system.svc.cluster.local
  protocol: http
  apiKeySecret: opensandbox-server-api-key
  apiKeySecretKey: api-key
```

Client env names are `OPEN_SANDBOX_DOMAIN` and `OPEN_SANDBOX_API_KEY` (not
Gym's `OPENSANDBOX_*`).

## Install (Kata)

Follow the Kata page for `kata-deploy` and CRI-O retrofit, then the same
controller install plus `-f opensandbox-server-kata-qemu.yaml`. Override
`opensandbox.domain` to `opensandbox-server-kata.opensandbox-system.svc.cluster.local`.

## Compose services

Benchmark tasks that ship a Docker Compose file (a database or a mock API next
to the agent's container) need those services in the sandbox. The optional
NeMo Compose services extension adds them to the sandbox pod as Kubernetes
native sidecars when a create request carries
`extensions["opensandbox.extensions.nemo-compose-services"]`. It runs inside
the stock server image: the server values above put `/opt/nemo-opensandbox-ext`
on `PYTHONPATH` and mount the `nemo-opensandbox-ext` ConfigMap there. Without
the ConfigMap the server runs unchanged. With it, the server refuses to start
if the extension can't load, rather than silently ignoring Compose requests.

The extension adds three routes behind the server's API key:
`GET /v1/nemo-ext/health`, `GET /v1/nemo-ext/sandboxes/{id}/services` (which
services are ready, and the first one that failed, with its last log lines),
and `POST /v1/nemo-ext/sandboxes/{id}/services/{service}/exec`, which runs a
command in one service container (and only those).

```bash
EXT=plugins/_temporary-scaled-evals/opensandbox_ext
kubectl create configmap nemo-opensandbox-ext -n opensandbox-system \
  --from-file="${EXT}/sitecustomize.py" \
  --from-file="${EXT}/nemo_ext_server.py" \
  --from-file="${EXT}/nemo_ext_pod.py" \
  --from-file=plugins/_temporary-scaled-evals/src/scaled_evals/harbor_opensandbox_services.py \
  --dry-run=client -o yaml | kubectl apply -f -
sed -i.bak "s/REPLACE_WITH_RELEASE_NAMESPACE/${NHX_NAMESPACE}/g" \
  "${EXAMPLES}/opensandbox-ext-rbac.yaml"
kubectl apply -f "${EXAMPLES}/opensandbox-ext-rbac.yaml"
kubectl rollout restart deploy/opensandbox-server -n opensandbox-system
kubectl logs -n opensandbox-system deploy/opensandbox-server | grep nemo-ext
```

The log shows the registered provider and routes. Repeat the ConfigMap apply
and restart after changing those sources. The extension is tested against the
server version in the values files (`v0.2.1`); upgrade the server and the
extension together.

## Verify

```bash
export OPEN_SANDBOX_WORKLOAD_NS="${NHX_NAMESPACE}"
./k8s/helm/examples/opensandbox/verify/shared-kernel.sh
# or, after installing the Kata server:
./k8s/helm/examples/opensandbox/verify/kata-qemu.sh
# with the Compose services extension installed:
./k8s/helm/examples/opensandbox/verify/services.sh
```
