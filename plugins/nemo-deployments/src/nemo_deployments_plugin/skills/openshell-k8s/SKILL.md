---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: openshell-k8s
description: >-
  Takes a Kubernetes cluster from nothing to a NeMo Platform Fabric agent running
  inside an OpenShell sandbox, answering questions, with proof that the sandbox has
  no network egress beyond the platform. Interactive and discovery-first: inspects
  the cluster and the operator machine, reuses what already exists, installs only
  what is missing (Agent Sandbox controller, OpenShell gateway, NeMo Platform), and
  stops at each gate for confirmation. Use when the user asks to deploy agents on
  OpenShell on Kubernetes, sandbox agents in a cluster, install OpenShell next to
  NeMo Platform, or run a NeMo agent under a sandbox policy on k8s. Trigger
  keywords - openshell kubernetes, openshell k8s, sandbox agents on k8s, install
  openshell helm, nemo platform openshell executor, sandboxed agent deployment,
  zero egress agent, agent sandbox controller, nemo agents deploy mode k8s.
triggers:
  - deploy agents on openshell on kubernetes
  - set up openshell on my cluster
  - install openshell next to nemo platform
  - sandbox my agent on k8s
  - run the agent in an openshell sandbox on kubernetes
  - openshell executor on kubernetes
  - prove the sandbox has no egress
not-for:
  - deploy-sandbox (use for the docker-driver, local-platform sandbox flow)
  - nemo-build-agent (use to design and build a real agent; this skill deploys a small demo agent)
  - nemo-try-agent (use to query an already-deployed agent)
  - nemo-status (use for a read-only health dashboard)
  - production hardening of OpenShell (this skill installs an evaluation posture)
compatibility: >-
  Operator machine on Linux or macOS with kubectl, helm 3, and Docker (to build the
  agent image). Any Kubernetes 1.29+ cluster with RBAC: local (k3d, kind, minikube,
  Docker Desktop, OrbStack) or remote (EKS, GKE, AKS, on-prem). OpenShell pinned to
  release 0.0.116 (Helm chart and CLI). NeMo Platform installed in the same cluster
  from its Helm chart; the nemo agents k8s deploy path onto an openshell executor
  and the in-cluster invoke proxy require a platform build that includes both
  fixes described under "If the platform predates the fixes". Node architecture is
  discovered, not assumed; amd64 is typical.
maturity: experimental
license: Apache-2.0
user-invocable: true
allowed-tools: [Bash, Read, Write, Edit]
---

# Deploy a NeMo agent into an OpenShell sandbox on Kubernetes

Stand up OpenShell and NeMo Platform in one Kubernetes cluster, then deploy a small Fabric (LangChain Deep Agents) agent through `nemo agents deploy --mode k8s` so it runs inside an OpenShell sandbox pod. Model calls leave the sandbox through exactly one allowed route, the platform API Service. Everything else is denied at the sandbox boundary.

"Done" looks like this: `nemo agents invoke --agent-deployment <name>` returns a real answer, `kubectl -n openshell get sandboxes.agents.x-k8s.io` shows the platform-created sandbox, and `openshell sandbox exec` inside that sandbox shows `curl https://example.com` failing while `curl` to the platform health endpoint succeeds. Do not claim success before all three are true.

This skill installs OpenShell in an evaluation posture: plaintext gateway (`server.disableTls=true`) and unauthenticated users (`server.auth.allowUnauthenticatedUsers=true`). Say this plainly to the user at the OpenShell gate. The reason: on the Kubernetes driver the gateway does not treat a client certificate as a user identity and expects an OIDC bearer token for user-level calls, which the NeMo Platform executor cannot present yet. Production changes two things: TLS with a real PKI (cert-manager, `certManager.enabled=true`) and OIDC for user auth (`server.oidc.*`), with `allowUnauthenticatedUsers` left `false`.

## How to behave

- Discover before you touch anything. Run the checklist in Step 1, then summarise the findings back to the user as a table of "found / will reuse" and "missing / will install" before running a single mutating command.
- Stop at every gate listed in Step 2 and wait for an explicit yes. Show the exact command you are about to run.
- After every state change, run the verification command shown and compare to the expected output. If it differs, go to the recovery table. Do not proceed on a failed verification.
- Never ask the user to paste an API key or token into the chat. Ask them to export it in the shell you run in, or to run the credential-bearing command themselves in a terminal.
- Never delete or modify anything you discovered as pre-existing. Record what this skill installed so cleanup only removes that.
- One state change at a time. No compound `&&` chains that hide which step failed.

Set these names once and reuse them. Adjust only if discovery shows a conflict.

```bash
export OPENSHELL_VERSION=0.0.116
export OPENSHELL_NS=openshell
export NMP_NS=nemo-platform
export NMP_RELEASE=nemo-platform
export NMP_CHART_REF=oci://ghcr.io/nvidia-nemo/nemo-platform/nemo-platform
export AGENT_NAME=sandbox-hello
export IMAGE_TAG=sandbox-hello:0.1
export WORKDIR="$PWD/openshell-k8s"
mkdir -p "$WORKDIR"
```

The platform API Service is `${NMP_RELEASE}-api.${NMP_NS}.svc.cluster.local:8080` when the release is named `nemo-platform`. A release name that does not contain `nemo-platform` becomes `<release>-nemo-platform-api`; keep the default unless the user insists, and always confirm with `kubectl get svc` in Step 6.

## Step 1: Discover

Run every command in [references/discovery-checklist.md](references/discovery-checklist.md). Each finding maps to a decision later. Then present this summary and wait for the user to confirm it is accurate:

| Area | Finding | Decision |
|---|---|---|
| Cluster | context, server version, node count, node arch | reuse cluster / stop if unreachable |
| Cluster type | k3d, kind, minikube, Docker Desktop, OrbStack, or remote | image import method (Step 7) |
| Storage | default StorageClass, RWX support | platform storage values (Step 4) |
| Agent Sandbox | `agent-sandbox-system` namespace and CRD present and healthy | reuse / install (Step 3) |
| OpenShell | `openshell` namespace, Helm release, gateway ready, version | reuse / install (Step 4) |
| NeMo Platform | release in `nemo-platform` (or elsewhere), API ready, executor config | reuse / install / upgrade values (Step 5) |
| Operator tools | kubectl, helm, docker, nemo, openshell and their versions | install missing (Step 2) |
| Registry access | can pull the platform chart and image anonymously; where agent images can be pushed | pull secrets (Step 5), push vs import (Step 7) |
| Credentials | which of NVIDIA_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, NGC_API_KEY, GITHUB_TOKEN are set (names only, never values) | provider choice (Step 6), pull secrets (Step 5) |
| Ports | 8080 and 18080 free on the operator machine | port-forward targets |

If an existing OpenShell gateway is found, check its TLS and auth posture with `helm get values openshell -n openshell`. A gateway with TLS enabled and no `allowUnauthenticatedUsers` cannot be driven by the platform executor today; tell the user and offer either a second evaluation gateway in another namespace or stopping here. Do not weaken an existing shared gateway.

If an existing NeMo Platform release is found, read its rendered config: `kubectl -n "$NMP_NS" get configmap "${NMP_RELEASE}-config" -o yaml`. If it lacks an `openshell` executor, Step 5 becomes a `helm upgrade` with the values file instead of an install. Never reinstall a platform the user already has.

## Step 2: Gates

Ask before each of these. Wait for the answer.

1. Installing or upgrading operator tools (`nemo`, `openshell`, `helm`, `kubectl`).
2. Creating namespaces and installing charts: Agent Sandbox controller, OpenShell, NeMo Platform. One gate each.
3. Choosing the model provider and the credential env var to use.
4. Which chart version to install for the platform (nightly GHCR or stable NGC) and whether a GitHub token or NGC key is available for pulls.
5. Pushing an image to a registry (remote clusters).
6. Every delete, including cleanup.

## Step 3: Operator tools

Install only what discovery marked missing. For each, show the command, get the gate, run it, verify.

kubectl and helm: use the user's package manager (`brew install kubectl helm` on macOS, distro packages on Linux). Verify with `kubectl version --client` and `helm version --short` (needs 3.x).

nemo CLI (needs `uv`):

```bash
uv tool install --python 3.13 "nemo-platform[all]"
nemo --version
nemo agents --help >/dev/null && echo NEMO_AGENTS_OK
```

`[all]` includes the agents and deployments plugins, which `nemo agents package --sandbox-runtime openshell` needs. Fabric packaging pins the installed `nemo-platform` version inside the image, so the CLI must be a released version from PyPI, not a source checkout. [UNVERIFIED]

openshell CLI, pinned. The PyPI `openshell` package at this release is the Python SDK only and does not install the CLI, and the `install.sh` installer installs a system package with sudo and on macOS also starts a local Docker gateway you do not need. Use the release tarball:

```bash
case "$(uname -s)-$(uname -m)" in
  Linux-x86_64)  OS_ASSET=openshell-x86_64-unknown-linux-musl.tar.gz ;;
  Linux-aarch64) OS_ASSET=openshell-aarch64-unknown-linux-musl.tar.gz ;;
  Darwin-arm64)  OS_ASSET=openshell-aarch64-apple-darwin.tar.gz ;;
  *) echo "no prebuilt openshell CLI for this host"; false ;;
esac
curl -fsSL -o "$WORKDIR/$OS_ASSET" \
  "https://github.com/NVIDIA/OpenShell/releases/download/v${OPENSHELL_VERSION}/${OS_ASSET}"
tar -xzf "$WORKDIR/$OS_ASSET" -C "$WORKDIR"
mkdir -p ~/.local/bin
install -m 0755 "$WORKDIR/openshell" ~/.local/bin/openshell
openshell --version
```

Expected: `openshell 0.0.116`. If the tarball nests the binary in a directory, adjust the `install` source path. [UNVERIFIED] Intel macOS has no prebuilt CLI at this release; tell the user and use `OPENSHELL_VERSION=v0.0.116 sh install.sh` only if they accept its side effects.

## Step 4: Agent Sandbox controller and OpenShell gateway

### 4a. Agent Sandbox controller

Skip if discovery found the CRD `sandboxes.agents.x-k8s.io` and a running controller. Otherwise, after the gate:

```bash
kubectl apply -f https://github.com/kubernetes-sigs/agent-sandbox/releases/latest/download/manifest.yaml
kubectl -n agent-sandbox-system rollout status deployment --timeout=120s
kubectl get crd sandboxes.agents.x-k8s.io
```

Expected: the rollout completes and the CRD is listed. Record "installed agent-sandbox" for cleanup.

### 4b. OpenShell gateway

Skip if discovery found a usable gateway (plaintext, unauthenticated users allowed, version 0.0.116). Otherwise, after the gate, write the values file from [references/openshell-values.yaml](references/openshell-values.yaml) into `$WORKDIR/openshell-values.yaml`, then:

```bash
kubectl create namespace "$OPENSHELL_NS" --dry-run=client -o yaml | kubectl apply -f -
helm upgrade --install openshell oci://ghcr.io/nvidia/openshell/helm-chart \
  --version "$OPENSHELL_VERSION" \
  --namespace "$OPENSHELL_NS" \
  --values "$WORKDIR/openshell-values.yaml" \
  --wait --timeout 5m
kubectl -n "$OPENSHELL_NS" rollout status statefulset/openshell --timeout=180s
kubectl -n "$OPENSHELL_NS" get pods
```

Expected: `openshell-0` is `Running` and `1/1`. The chart's preflight fails fast when the Agent Sandbox CRD is missing, which points back to 4a.

Connect the CLI through a port-forward on 18080 (8080 is reserved for the platform):

```bash
kubectl -n "$OPENSHELL_NS" port-forward svc/openshell 18080:8080 >"$WORKDIR/pf-openshell.log" 2>&1 &
echo $! > "$WORKDIR/pf-openshell.pid"
sleep 2
openshell gateway add http://127.0.0.1:18080 --local --name nemo-k8s
openshell gateway select nemo-k8s
openshell status
openshell sandbox list
```

Expected: `openshell status` reports the gateway healthy and `sandbox list` prints an empty list or `No sandboxes found`. An `UNAUTHENTICATED` here means the values file was not applied; see the recovery table. [UNVERIFIED]

Note for the user: sandbox pods are created in the `openshell` namespace, and sandboxes with a workspace volume need a default StorageClass. If discovery found no default StorageClass, set `server.workspaceStorageClass` in the values file before installing.

## Step 5: NeMo Platform in the cluster

Skip to the "existing platform" branch at the end of this step if discovery found a release.

Pick the chart source at the gate. The nightly GHCR chart carries this week's fixes for the k8s sandbox deploy path; stable NGC `0.5.0` predates them and needs the fallback in [references/raw-deployments-api.md](references/raw-deployments-api.md). Ask the user for the nightly version string (it looks like `0.5.0-nightly-YYYYMMDDHHMMSS`), then check access:

```bash
export NMP_CHART_VERSION=<nightly version>
helm show chart "$NMP_CHART_REF" --version "$NMP_CHART_VERSION" | head -5
```

If this fails with an auth error, the packages are not anonymous. Ask the user to export `GITHUB_TOKEN` (read:packages) in this shell, then:

```bash
echo "$GITHUB_TOKEN" | helm registry login ghcr.io --username "$GITHUB_USER" --password-stdin
helm show chart "$NMP_CHART_REF" --version "$NMP_CHART_VERSION" | head -5
```

[UNVERIFIED] whether the nightly chart and image are anonymously pullable.

Create the namespace and the secrets the chart expects. The chart needs a secret named `ngc-api`; with GHCR images and no NGC downloads a placeholder value is acceptable for this evaluation, and a real `NGC_API_KEY` is used when present:

```bash
kubectl create namespace "$NMP_NS" --dry-run=client -o yaml | kubectl apply -f -
kubectl -n "$NMP_NS" create secret generic ngc-api \
  --from-literal=NGC_API_KEY="${NGC_API_KEY:-unused-evaluation}"
```

If the chart source needed a GitHub token, the cluster needs the same for image pulls:

```bash
kubectl -n "$NMP_NS" create secret docker-registry ghcr-pull \
  --docker-server=ghcr.io \
  --docker-username="$GITHUB_USER" \
  --docker-password="$GITHUB_TOKEN"
```

For the stable NGC chart, create `nvcrimagepullsecret` with `--docker-server=nvcr.io --docker-username='$oauthtoken' --docker-password="$NGC_API_KEY"` instead.

Write the values file from [references/nemo-platform-values.yaml](references/nemo-platform-values.yaml) into `$WORKDIR/nemo-platform-values.yaml`. Edit three things based on discovery: the `imagePullSecrets` entry (name it after the secret you created, or remove the list if pulls are anonymous), the `core.storage` block (keep `ReadWriteOnce` only on single-node local clusters; on clusters with an RWX StorageClass delete the block or set `storageClass`), and the two hostnames if you changed `NMP_RELEASE` or `NMP_NS`.

The values restate the `local-k8s` executor next to the `openshell` one on purpose: `platformConfig` lists replace the chart's base list rather than merging, so omitting `local-k8s` would delete the default executor and the platform would fail to start.

Install after the gate:

```bash
helm upgrade --install "$NMP_RELEASE" "$NMP_CHART_REF" \
  --version "$NMP_CHART_VERSION" \
  --namespace "$NMP_NS" \
  --values "$WORKDIR/nemo-platform-values.yaml" \
  --wait --timeout 20m
```

Image pulls and database init take several minutes. If helm exits with a post-install hook timeout but the pods below come up, that is the seed job racing readiness; continue and re-run the same `helm upgrade` later to let the seed complete. [UNVERIFIED]

Verify:

```bash
kubectl -n "$NMP_NS" rollout status deployment/"${NMP_RELEASE}-api" --timeout=10m
kubectl -n "$NMP_NS" get pods
kubectl -n "$NMP_NS" get svc "${NMP_RELEASE}-api"
kubectl -n "$NMP_NS" get configmap "${NMP_RELEASE}-config" -o yaml | grep -n -A 12 'name: openshell'
kubectl -n "$NMP_NS" logs deployment/"${NMP_RELEASE}-core-controller" --tail=300 | grep -i 'skipping executor' || echo EXECUTORS_REGISTERED
```

Expected: the api rollout completes, the Service exists with that exact name, the ConfigMap shows the `openshell` executor block, and the last line prints `EXECUTORS_REGISTERED`. A `Skipping executor 'openshell'` log line means the image lacks the OpenShell SDK extra; see the recovery table.

Port-forward the API and confirm readiness:

```bash
kubectl -n "$NMP_NS" port-forward svc/"${NMP_RELEASE}-api" 8080:8080 >"$WORKDIR/pf-nemo.log" 2>&1 &
echo $! > "$WORKDIR/pf-nemo.pid"
sleep 2
curl -sf http://localhost:8080/health/ready
```

Expected: `{"status":"ready"}`.

Existing platform branch: if a release already exists without the `openshell` executor, take its current values with `helm get values "$NMP_RELEASE" -n "$NMP_NS" -o yaml > "$WORKDIR/current-values.yaml"`, merge the `platformConfig` block from the reference values into it by hand (keeping every executor the user already has), show the diff to the user, and run `helm upgrade` with the same chart version they run. Then run the same verification.

## Step 6: Connect the CLI and register a model provider

Ask which provider the user wants (NVIDIA build.nvidia.com, OpenAI, Anthropic, Gemini) and confirm the matching env var is exported in this shell (check with `test -n "$NVIDIA_API_KEY" && echo SET`; never echo the value). Then:

```bash
export NMP_BASE_URL=http://localhost:8080
nemo config set --base-url "$NMP_BASE_URL"
nemo setup --auto --no-start-services --no-install-skills --no-deploy-agent
nemo models list --all-pages
```

Expected: setup registers the provider, discovers models, and picks a default and a fast model; `nemo models list` prints them. `--auto` reads the provider key from the environment and works without a TTY. If the user prefers the interactive wizard, they run `nemo setup --no-start-services --no-install-skills --no-deploy-agent` themselves in a terminal and tell you when it finishes.

## Step 7: Author, package, and register the agent

Create a minimal agent directory holding only `agent.yaml` (the register step uploads the directory, and the package step uses it as the Docker build context):

```bash
mkdir -p "$WORKDIR/agents/$AGENT_NAME"
```

Write [references/agent.yaml](references/agent.yaml) to `$WORKDIR/agents/$AGENT_NAME/agent.yaml`, replacing `sandbox-hello` with `$AGENT_NAME` if different. It uses `${NEMO_DEFAULT_MODEL}`, which `nemo agents create` resolves to the default model chosen in Step 6, and `provider: nvidia`; use `provider: openai` when the platform provider is OpenAI. The deploy step rewrites the model `base_url` to the in-cluster Inference Gateway, so the agent needs no key of its own.

Build the image with the OpenShell runtime profile. Pass `--platform` when the node architecture from discovery differs from the operator machine:

```bash
nemo agents package \
  --agent "$WORKDIR/agents/$AGENT_NAME/agent.yaml" \
  --sandbox-runtime openshell \
  --tag "$IMAGE_TAG"
```

Add `--platform linux/amd64` (or `linux/arm64`) to match the nodes. Cross-architecture builds need Docker buildx emulation and are slow. [UNVERIFIED]

Verify the image has what the sandbox needs:

```bash
docker image inspect "$IMAGE_TAG" --format '{{.Architecture}}'
docker run --rm --entrypoint sh "$IMAGE_TAG" -c 'id sandbox; ls -l /workspace/.venv/bin/python /workspace/.venv/bin/python3.13; command -v curl'
```

Expected: the architecture matches the nodes, a `sandbox` user exists, both interpreter paths are listed, and `/usr/bin/curl` is present. These three paths are the `platform_egress.binaries` in the platform values; if the python paths differ, update the values and `helm upgrade` before deploying. [UNVERIFIED]

Get the image into the cluster. Pick the branch discovery chose:

```bash
# k3d
k3d image import "$IMAGE_TAG" -c <cluster-name>
# kind
kind load docker-image "$IMAGE_TAG" --name <cluster-name>
# minikube
minikube image load "$IMAGE_TAG"
```

Docker Desktop Kubernetes and OrbStack share the local Docker image store with the node, so no import is needed. [UNVERIFIED] Keep a versioned tag (not `latest`) so the sandbox pod uses `IfNotPresent` and finds the imported image.

Remote cluster: after the push gate, rebuild with `--publish --registry <registry>` and `--push-tag <registry>/<repo>:0.1`, set `IMAGE_TAG` to that reference, and if the registry is private create a pull secret in the `openshell` namespace and reference it via `server.sandboxImagePullSecrets` in the OpenShell values, then `helm upgrade` OpenShell.

Register the agent:

```bash
nemo agents create --name "$AGENT_NAME" --agent-config "$WORKDIR/agents/$AGENT_NAME/agent.yaml"
nemo agents get "$AGENT_NAME" | head -20
```

Expected: the agent is returned with `config_format: nemo-agents-spec-v1` and a concrete model name where `${NEMO_DEFAULT_MODEL}` was.

## Step 8: Deploy into the sandbox and invoke

```bash
nemo agents deploy \
  --agent "$AGENT_NAME" \
  --name "$AGENT_NAME-sandbox" \
  --mode k8s \
  --image "$IMAGE_TAG" \
  --timeout 600
```

Expected: the command blocks and exits 0 with status `running`. While it waits, watch the sandbox appear:

```bash
kubectl -n "$OPENSHELL_NS" get sandboxes.agents.x-k8s.io
kubectl -n "$OPENSHELL_NS" get pods -l openshell.ai/managed-by=openshell
openshell sandbox list
```

Expected: one Sandbox named `default--nmp-<hash>`, its pod `Running`, and `openshell sandbox list` showing `nmp-<hash>` as ready. Save the name:

```bash
export SBX=$(openshell sandbox list --names | grep '^nmp-' | head -1)
echo "$SBX"
```

[UNVERIFIED] `--names` output shape; fall back to reading the name from `nemo agents deployments get "$AGENT_NAME-sandbox"` (`endpoints[0].url` is `http://default--<sandbox>--http.openshell.localhost:8080/`).

Invoke:

```bash
nemo agents deployments get "$AGENT_NAME-sandbox"
nemo agents invoke --agent-deployment "$AGENT_NAME-sandbox" \
  --input "In one sentence, where are you running?"
```

Expected: a non-empty answer from the model. The platform proxies the request to the sandbox through the OpenShell gateway. Exit code 0 with an empty body is not success.

If the deploy fails or `invoke` cannot reach the endpoint, read `nemo agents deployments get` for the status message, then the serve log inside the sandbox:

```bash
openshell sandbox exec --name "$SBX" -- cat /tmp/nemo-serve.log
```

## Step 9: Prove zero egress

Run both checks from inside the platform-created sandbox. The first must fail, the second must succeed:

```bash
openshell sandbox exec --name "$SBX" -- curl -sS -m 5 https://example.com
echo "exit=$?  (non-zero is the PASS)"
openshell sandbox exec --name "$SBX" -- curl -sS -m 5 \
  "http://${NMP_RELEASE}-api.${NMP_NS}.svc.cluster.local:8080/health/ready"
echo "exit=$?  (zero is the PASS)"
```

Expected: the first `curl` is refused by the sandbox proxy and exits non-zero; the second prints `{"status":"ready"}`. Then invoke the agent once more to show it still answers while general egress is blocked. Show the user all three outputs together. [UNVERIFIED] on the Kubernetes driver; the same proof is verified on the Docker driver.

## Step 10: Cleanup

Offer two paths and wait for the choice. In both, stop the port-forwards you started:

```bash
kill "$(cat "$WORKDIR/pf-nemo.pid")" "$(cat "$WORKDIR/pf-openshell.pid")" 2>/dev/null
```

Keep everything: leave the cluster as is. Tell the user how to reconnect later (the two port-forward commands, `nemo config set --base-url`, `openshell gateway select nemo-k8s`).

Remove what this skill installed, in this order, skipping anything discovery marked as pre-existing, with a gate before each delete:

```bash
nemo agents undeploy "$AGENT_NAME-sandbox" --yes
nemo agents delete "$AGENT_NAME" --yes
helm uninstall "$NMP_RELEASE" -n "$NMP_NS"
kubectl delete namespace "$NMP_NS"
helm uninstall openshell -n "$OPENSHELL_NS"
kubectl delete namespace "$OPENSHELL_NS"
kubectl delete -f https://github.com/kubernetes-sigs/agent-sandbox/releases/latest/download/manifest.yaml
openshell gateway remove nemo-k8s
docker image rm "$IMAGE_TAG"
```

`helm uninstall` leaves PersistentVolumeClaims behind by design; deleting the namespace removes them. Never run the namespace or controller deletes against namespaces the user had before this session.

## If verification fails

| Symptom | Cause | Recovery |
|---|---|---|
| OpenShell `helm install` fails before creating resources, message about Sandbox API | Agent Sandbox CRD missing | Run Step 4a, confirm `kubectl get crd sandboxes.agents.x-k8s.io`, retry |
| Platform PVC `Pending`, api pod `Pending` | Default StorageClass cannot bind `ReadWriteMany` (k3d, kind, minikube local-path) | Set `core.storage.accessModes: [ReadWriteOnce]` and a small size in the values, `helm upgrade`; or create the hostPath RWX PV from the platform install docs |
| api pod `CreateContainerConfigError`, secret `ngc-api` not found | Chart expects an existing `ngc-api` secret | Create it (Step 5), or set `existingSecret: ""` and `ngcAPIKey: <value>` in values |
| Pods `ImagePullBackOff` in the platform namespace | Chart or image registry needs auth | Create the `ghcr-pull` (or `nvcrimagepullsecret`) secret and list it under `imagePullSecrets`; confirm the tag exists with `docker manifest inspect` |
| Deploy fails with unknown or unregistered executor `openshell` | `platformConfig.deployments.executors` was overwritten or missing the entry | Re-check the values file restates both executors; `kubectl get configmap ${NMP_RELEASE}-config -o yaml` must show `name: openshell`; `helm upgrade` |
| Controller log `Skipping executor 'openshell': backend 'openshell' is unavailable` | Platform image without the OpenShell SDK extra | Use a platform build whose `nmp-api` image ships `nemo-deployments-plugin[openshell]`; nightlies from this month do |
| Deploy fails `UNAUTHENTICATED: missing authorization header` | Gateway installed with `allowUnauthenticatedUsers=false` | Confirm the OpenShell values were applied (`helm get values openshell -n openshell`), `helm upgrade` with the reference values |
| `nemo agents deploy` returns 400 `deployment_mode 'k8s' resolved to executor 'openshell', which runs on 'openshell'` | Platform predates fix 1 (mode check rejects openshell executors) | Use the raw deployments API in [references/raw-deployments-api.md](references/raw-deployments-api.md) |
| Deployment `running` but `nemo agents invoke` cannot reach the endpoint or times out | Platform predates fix 2 (proxy to sandbox through the gateway with a Host header) | Reach the endpoint through the gateway port-forward as shown in the fallback reference |
| Sandbox pod `ImagePullBackOff` | Image not imported into the local cluster, not pushed to a registry the node can pull, or `:latest` tag forcing a pull | Import or push (Step 7), use a versioned tag, add `server.sandboxImagePullSecrets` for private registries |
| Sandbox pod `CrashLoopBackOff` with `exec format error` | Image architecture differs from the node | Rebuild with `--platform linux/<node-arch>`, re-import |
| Sandbox stuck `Pending`, workspace PVC pending | Cluster has no default StorageClass | Set `server.workspaceStorageClass` in OpenShell values, `helm upgrade` |
| Deployment `FAILED`: serve process exited, no log | `serve_workdir` not writable by the sandbox uid on the k8s driver | Keep `serve_workdir: /tmp` in the executor config (the docker default `/home/sandbox` is denied on k8s) |
| Deployment `FAILED`: awaiting readiness never passes, 502 on invoke | Fabric server not listening on 8000 yet, or crashed | `openshell sandbox exec --name $SBX -- cat /tmp/nemo-serve.log`; a model name the gateway does not serve shows here |
| Agent answers fail with connection refused or name resolution errors to `<release>-api` | Agent dials the namespace-less Service name the chart exports | Set `agents.deployments.k8s_internal_base_url` to the FQDN as in the reference values, `helm upgrade`, redeploy |
| Model call blocked, serve log shows the proxy denying the platform host | Egress rule binary does not match the interpreter that opens the socket | Compare `platform_egress.binaries` to the paths printed in Step 7; add the missing path, `helm upgrade`, redeploy |
| `example.com` reachable in Step 9 | Policy not applied, or the sandbox is not the platform-created one | Confirm `SBX` is the `nmp-<hash>` sandbox and the deployment ran on executor `openshell` (`nemo agents deployments get`) |
| `nemo setup` exits: non-interactive shell | No TTY | Add `--auto` with the provider key exported, or have the user run it in a terminal |
| `nemo agents package` error: installed nemo-platform version is a developmental release | CLI installed from a source checkout | Install the released CLI (`uv tool install "nemo-platform[all]"`) or set `NEMO_AGENTS_WHEEL` per the packaging reference in nemo-build-agent |
| Helm reports a post-install hook timeout while pods are healthy | Seed job raced API readiness | Re-run the same `helm upgrade`; verify with `kubectl get pods` |
| Platform pods fine, sandbox cannot reach the platform, cluster has NetworkPolicies | `networkPolicies.enabled=true` on the platform chart blocks ingress from the `openshell` namespace | Add an `extraIngress` rule for the `openshell` namespace, or keep policies disabled for the evaluation |

## If the platform predates the fixes

Two platform changes land together and this skill assumes both: `nemo agents deploy --mode k8s` accepts an executor whose backend is `openshell`, and the platform proxies agent traffic to the sandbox through the OpenShell gateway with the minted hostname as the `Host` header so `nemo agents invoke` works in-cluster. On an older platform, replace Step 8 with [references/raw-deployments-api.md](references/raw-deployments-api.md): create a `DeploymentConfig` and a `Deployment` with `executor: openshell` directly, then reach the served agent through the gateway port-forward with a `Host` header.

## Related skills

- `deploy-sandbox`: the same idea on the Docker driver with a local platform, for laptops without Kubernetes.
- `nemo-build-agent`: design and build a real agent from an approved Ethos; deploy it with Step 8 of this skill.
- `nemo-try-agent`: query the deployed agent after this skill finishes.
- `nemo-status`: read-only platform health.
- `inference`: register additional providers or virtual models the agent can use.
