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

### Agent bundles

The same extension delivers scaled-evals agent bundles. A bundle is a
prebuilt, pinned image holding an agent (for example Claude Code) and
an installer. Harbor's installed agents need root to install themselves
(`apt-get`), which fails on task images with a non-root `USER`, and they
download the agent on every trial. With a bundle, a create request carries
`extensions["opensandbox.extensions.nemo-agent-bundle"]`, and the extension
adds an `agent-bundle` init container that runs the bundle's installer into
`/installed-agent`:

- It runs after `egress`, so the copy is under the sandbox's egress policy,
  and before any Compose service.
- It reuses the template's `/installed-agent` volume (see
  [Harbor directories](#harbor-directories)), or adds an `emptyDir` if the
  template has none, and mounts it read-only into the sandbox container.
- If it can't pull the image or exits non-zero, the services route reports
  `agent bundle ...` with its last log lines, and the create fails.

A bundle only takes effect when the agent's Harbor install finds the agent
already on `PATH` and skips itself (Harbor 0.20 or later); most of Harbor's
installed agents always install. scaled-evals therefore accepts bundles only
for agents verified end to end, currently `claude-code`, and only when the
evaluation runs that agent.

Only the agent's sandbox gets the bundle, never a separate verifier's. The
client checks that `GET /v1/nemo-ext/health` lists `agent-bundle` under
`features` before it creates a sandbox, so an older extension fails the trial
instead of starting a sandbox without the agent. The bundle image must be
pullable with the template's `imagePullSecrets`.

Like the task image, the bundle is pulled by its digest, or by its signed tag
when scaled-evals runs with `SANDBOX_K8S_TASK_IMAGE_REFERENCE_MODE=tag` (which
hosted mode requires). A tag is checked to still resolve to the registered
digest at launch.

### Enable it with the chart

The NeMo Helix chart ships the extension, so you don't need a checkout of the
repository. Set these values on the platform release and run `helm upgrade`:

```yaml
sandboxClusterCapable: true
opensandbox:
  namespace: opensandbox-system      # where the server runs; the ConfigMap goes here
  composeServices:
    enabled: true
    sandboxNamespace: ""             # [kubernetes] namespace in the server values; empty is the release namespace
    serverServiceAccount: opensandbox-server
    kataServerServiceAccount: ""     # opensandbox-server-kata, if you run the Kata server
```

The chart renders the `nemo-opensandbox-ext` ConfigMap into `opensandbox.namespace`,
and a Role and RoleBinding for `pods/exec`, `pods/log` and `pods` into the sandbox
namespace. It doesn't install or restart OpenSandbox, so install the server with
the values in this directory first (they mount the ConfigMap as optional), then:

```bash
helm upgrade nemo-helix <chart> -n "${NHX_NAMESPACE}" --reuse-values -f compose-services.yaml
kubectl rollout restart deploy/opensandbox-server -n opensandbox-system
kubectl rollout status deploy/opensandbox-server -n opensandbox-system
kubectl logs -n opensandbox-system deploy/opensandbox-server | grep nemo-ext
```

The log line is `nemo-ext: compose services extension loaded`. **The server
reads the ConfigMap only when it starts**, so a new or changed ConfigMap has no
effect until you restart it. If you deploy with GitOps, trigger a rollout of the
server (for example, change a pod annotation) after the chart syncs.

If you applied the ConfigMap and Role by hand before, read
[Move a manual install to the chart](#move-a-manual-install-to-the-chart) first.

### Enable it without Helm

Apply the generated ConfigMap and the example Role yourself:

```bash
kubectl apply -n opensandbox-system -f k8s/helm/files/opensandbox-ext/configmap.yaml
sed -i.bak "s/REPLACE_WITH_RELEASE_NAMESPACE/${NHX_NAMESPACE}/g" \
  "${EXAMPLES}/opensandbox-ext-rbac.yaml"
kubectl apply -f "${EXAMPLES}/opensandbox-ext-rbac.yaml"
kubectl rollout restart deploy/opensandbox-server -n opensandbox-system
```

`configmap.yaml` is generated from the extension's source files by
`script/generate_opensandbox_ext_configmap.py`, and CI fails if the committed
copy is out of date. Don't edit it by hand.

### Move a manual install to the chart

`helm upgrade` refuses to take over objects it didn't create, and stops with
`exists and cannot be imported into the current release`. Before enabling
`composeServices`, find the objects from the manual steps: the ConfigMap
`nemo-opensandbox-ext` in the server's namespace, and the Role and RoleBinding
of the same name in the sandbox namespace. Either delete them (the server keeps
running the loaded extension until you restart it, and the upgrade recreates
them), or hand them to the release:

```bash
# For each of the three objects: the ConfigMap in the server's namespace, the Role and RoleBinding in the sandbox namespace
kubectl label <kind>/nemo-opensandbox-ext -n <namespace> app.kubernetes.io/managed-by=Helm
kubectl annotate <kind>/nemo-opensandbox-ext -n <namespace> \
  meta.helm.sh/release-name=<release> meta.helm.sh/release-namespace=<release namespace>
```

Then run the `helm upgrade` above and restart the server.

### Upgrade, remove, troubleshoot

The extension is tested against OpenSandbox server `v0.2.1` only. Upgrade the
chart and the server together, then restart the server.

To remove it, set `opensandbox.composeServices.enabled=false` (or delete the
ConfigMap) and restart the server. It then runs stock, and scaled-evals fails a
compose task at its first check, before any sandbox starts.

| Symptom | Cause |
| --- | --- |
| The server crash-loops with `nemo-ext: not starting without the compose services extension` | The ConfigMap is present but the extension can't load: a changed file, or a server version other than `v0.2.1`. The log above the message has the reason. |
| No `nemo-ext` line in the server log | The ConfigMap is missing, in the wrong namespace, or the server wasn't restarted since it was created. |
| `GET /v1/nemo-ext/health` doesn't answer | The server runs stock. Same causes as the line above. A working server answers `{"status":"ok","extension":"nemo-compose-services","features":["compose-services","agent-bundle"]}`. |
| A trial with an agent bundle fails with `doesn't support agent bundles` | The extension predates agent bundles: its health answer has no `features`. Upgrade the chart and restart the server. |
| A trial fails with `agent bundle ErrImagePull` or `ImagePullBackOff` | The bundle's registry isn't covered by the template's `imagePullSecrets`. |
| A compose task fails reading service status or running a command in a service, with a permission error | The Role is missing in the sandbox namespace, or bound to the wrong ServiceAccount. Check `sandboxNamespace` and `serverServiceAccount`. |

## Verify

```bash
export OPEN_SANDBOX_WORKLOAD_NS="${NHX_NAMESPACE}"
./k8s/helm/examples/opensandbox/verify/shared-kernel.sh
# or, after installing the Kata server:
./k8s/helm/examples/opensandbox/verify/kata-qemu.sh
# with the Compose services extension installed:
./k8s/helm/examples/opensandbox/verify/services.sh
```
