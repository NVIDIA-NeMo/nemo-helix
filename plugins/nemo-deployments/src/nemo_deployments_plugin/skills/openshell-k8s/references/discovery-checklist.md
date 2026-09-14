<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Discovery checklist

Run every block. Each command is read-only. Record the result next to the decision it drives, then present the summary table from the skill before changing anything. A command that fails is itself a finding (for example, `openshell` not installed), not a reason to stop.

## Cluster

```bash
kubectl config current-context
kubectl version
kubectl get nodes -o custom-columns='NAME:.metadata.name,ARCH:.status.nodeInfo.architecture,OS:.status.nodeInfo.osImage,K8S:.status.nodeInfo.kubeletVersion,RUNTIME:.status.nodeInfo.containerRuntimeVersion'
```

Decisions: server version must be 1.29 or newer for OpenShell. Node architecture selects `--platform` for the agent image. More than one node means images must be pushed to a registry, not imported.

## Cluster type

```bash
kubectl config current-context
kubectl get nodes -o name
```

Map the context name to an import method:

| Context or node name pattern | Type | Image path into the cluster |
|---|---|---|
| `k3d-<name>` | k3d | `k3d image import <tag> -c <name>` |
| `kind-<name>` | kind | `kind load docker-image <tag> --name <name>` |
| `minikube` | minikube | `minikube image load <tag>` |
| `docker-desktop` | Docker Desktop | shared image store, no import [UNVERIFIED] |
| `orbstack` | OrbStack | shared image store, no import [UNVERIFIED] |
| anything else (`arn:aws:eks:...`, `gke_...`, `<cluster>-admin@...`) | remote | push to a registry the nodes can pull from |

## Storage

```bash
kubectl get storageclass
kubectl get storageclass -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.metadata.annotations.storageclass\.kubernetes\.io/is-default-class}{"\t"}{.provisioner}{"\n"}{end}'
```

Decisions: no default StorageClass means OpenShell sandbox workspace volumes stay `Pending`; set `server.workspaceStorageClass`. Local provisioners (`rancher.io/local-path`, `k8s.io/minikube-hostpath`, kind's `rancher.io/local-path`) do not support `ReadWriteMany`; the platform values must use `ReadWriteOnce`. Provisioners such as `efs.csi.aws.com`, `file.csi.azure.com`, `filestore.csi.storage.gke.io`, or a NFS class support RWX and can keep the chart default.

## Agent Sandbox controller

```bash
kubectl get crd sandboxes.agents.x-k8s.io
kubectl -n agent-sandbox-system get pods
```

Decision: CRD present and controller `Running` means reuse. Either missing means install in Step 4a.

## OpenShell

```bash
kubectl get namespace openshell
helm list -n openshell
kubectl -n openshell get statefulset,deployment,svc,pods
helm get values openshell -n openshell
kubectl -n openshell get sandboxes.agents.x-k8s.io
```

Decisions: a release at chart version `0.0.116` with `server.disableTls: true` and `server.auth.allowUnauthenticatedUsers: true` is reusable as is. Any other posture: tell the user; do not weaken it. Existing sandboxes belong to someone; never delete them.

## NeMo Platform

```bash
helm list -A | grep -i nemo
kubectl get namespace nemo-platform
kubectl -n nemo-platform get deployment,svc,pods
kubectl -n nemo-platform get configmap -o name | grep -- '-config$'
```

If a release exists, read its rendered platform config and look for the `deployments.executors` list and the `agents.deployments` block:

```bash
kubectl -n <namespace> get configmap <release>-config -o jsonpath='{.data.config\.yaml}' | grep -n -A 30 '^deployments:'
kubectl -n <namespace> get configmap <release>-config -o jsonpath='{.data.config\.yaml}' | grep -n -A 8 '^agents:'
```

Decisions: no release means install (Step 5). A release without an `openshell` executor means `helm upgrade` with merged values. A release with one means reuse; note its `gateway_endpoint` and `platform_egress` for the egress proof.

## Operator tools

```bash
command -v kubectl && kubectl version --client
command -v helm && helm version --short
command -v docker && docker version --format '{{.Server.Version}} {{.Server.Arch}}'
command -v uv && uv --version
command -v nemo && nemo --version
command -v openshell && openshell --version
```

Decisions: any missing tool goes to Step 3 with a gate, after a second look through the user's login shell (`bash -lc 'command -v kubectl helm docker uv'`): tool managers such as mise, asdf, and brew often put binaries on PATH only for interactive shells, and a tool that exists there should be used, not reinstalled. `docker` must reach a daemon, otherwise the agent image cannot be built here. `openshell` must report `0.0.116` to match the chart; a different version is a finding to raise, not silently replace.

## Registry access

```bash
helm show chart oci://ghcr.io/nvidia/openshell/helm-chart --version 0.0.116 | head -3
docker manifest inspect ghcr.io/nvidia/openshell/gateway:0.0.116 >/dev/null && echo OPENSHELL_IMAGE_OK
```

The platform chart version is supplied by the user (nightly `0.5.0-nightly-YYYYMMDDHHMMSS` from GHCR, or a stable version from NGC). Check it anonymously once known:

```bash
helm show chart oci://ghcr.io/nvidia-nemo/nemo-platform/nemo-platform --version <version> | head -3
```

Decisions: an auth error means the user needs a `GITHUB_TOKEN` with `read:packages` (nightly) or an `NGC_API_KEY` (stable), and the cluster needs a matching pull secret. [UNVERIFIED] whether nightly GHCR artifacts are anonymous.

For remote clusters also ask which registry the nodes can pull from and whether `docker login` to it already works on this machine.

## Credentials present

Print names only. Never print values.

```bash
for v in NVIDIA_API_KEY OPENAI_API_KEY ANTHROPIC_API_KEY GEMINI_API_KEY NEMO_DEFAULT_INFERENCE_KEY NGC_API_KEY GITHUB_TOKEN; do
  if [ -n "${!v:-}" ]; then echo "$v=set"; else echo "$v=unset"; fi
done
```

Decisions: at least one model provider key must be set for `nemo setup --auto`. On macOS `bash` 3 lacks `${!v}`; run the loop with `zsh` or check each variable with `test -n "$NVIDIA_API_KEY"`.

## Local ports

```bash
lsof -iTCP:8080 -sTCP:LISTEN
lsof -iTCP:18080 -sTCP:LISTEN
```

Decisions: both should print nothing. If either port has a listener, it belongs to something else: never stop it, whatever it looks like. Choose a free port for `NMP_PORT` or `OPENSHELL_PORT` in the steps that follow and carry it through to `nemo config set --base-url` and `openshell gateway add`.
