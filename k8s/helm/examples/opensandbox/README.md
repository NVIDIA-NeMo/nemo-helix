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
| `batchsandbox-template.yaml` | ConfigMap — exclude control-plane; pull Secret name `nvcrimagepullsecret` |
| `batchsandbox-template-kata-qemu.yaml` | ConfigMap — example Kata node selectors; same pull Secret name |

`[secure_runtime]` is server-global. Install **one** server for production
(shared-kernel **or** Kata). Dual releases are only for proving both paths.

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

## Verify

```bash
export OPEN_SANDBOX_WORKLOAD_NS="${NHX_NAMESPACE}"
./k8s/helm/examples/opensandbox/verify/shared-kernel.sh
# or, after installing the Kata server:
./k8s/helm/examples/opensandbox/verify/kata-qemu.sh
```
