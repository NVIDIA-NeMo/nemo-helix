<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Shared OpenSandbox for NeMo Helix

> [!NOTE]
> **Draft.** Being verified end to end. The OpenShift-only parts (the SCCs in step 2) haven't been tested yet.

One-time setup of one shared OpenSandbox server for the cluster. NeMo Helix installs in tenant
namespaces use it for two kinds of evaluation, both of which keep untrusted code out of the Helix
job pod:

- **Harbor taskset runs** (for example Terminal-Bench): each trial's task runs in its own sandbox
  pod, and a separate verifier sandbox when the task has one.
- **Sandboxed NeMo Gym** (Evaluator `agent-evaluate` jobs): each job runs NeMo Gym in one sandbox,
  the *Gym host* (`nhx-gym-host`), which mounts the tenant's job storage volume. The job pod calls it
  over HTTP through the server.

```text
tenant namespace (one per Helix install)            opensandbox-system (once per cluster)
┌─────────────────────────────────────────┐        ┌──────────────────────────────────┐
│ Helix API, controller, job pods ────────┼─ :80 ─▶│ opensandbox-server (API + proxy) │
│                                         │        │ opensandbox-controller           │
│ sandbox pods  ◀─────────────────────────┼────────┤   creates BatchSandboxes in the  │
│  task container + execd + egress sidecar│        │   caller's tenant namespace      │
└─────────────────────────────────────────┘        └──────────────────────────────────┘
```

Each tenant gets its own API key. The server maps the key to the tenant's namespace, so a tenant
only sees and creates sandboxes there.

| Component | Version |
| --- | --- |
| OpenSandbox charts (base, controller, server) | `release-1.1.1`, upstream commit `531700ca18a7fa5fab6a2f25d6dc68a76eddab12` |
| Images (`docker.io/opensandbox/…`) | `controller`, `server`, `execd`, `egress`, `image-committer`, all `release-1.1.1` |
| NeMo Helix | 0.10.0 or later, for an OpenSandbox extension that supports tenants. Gym also needs the fix in [Known issue](#known-issue-sandboxed-gym-on-a-tenant-server) |

## Who does what

| Step | What | Who |
| --- | --- | --- |
| 1 | CRDs and controller | Platform (cluster admin) |
| 2 | Prepare each tenant namespace: ServiceAccount, exec Role, SCC, NetworkPolicy | Platform |
| 3 | Tenant keys | Platform |
| 4 | Sandbox pod template | Platform |
| 5 | Compose services: the NeMo Helix extension | Platform, with the file from the Helix team |
| 6 | Server | Platform |
| 7 | Check the server | Platform |
| 8 | Point each Helix release at the server | Helix team |
| 9 | Run an evaluation end to end | Helix team |

## Before you start

- **Tools:** `oc` (or `kubectl`), `helm` 3, `git`, `curl`, `openssl` and `python3`, logged in to
  the cluster as a cluster admin.
- **This folder** (README, `values/`, `manifests/`, `helix/`): `k8s/helm/examples/opensandbox-shared`
  in the [NeMo Helix repository](https://github.com/NVIDIA-NeMo/nemo-helix), or a copy from the Helix
  team. Run every command from it, in one shell: later steps use variables earlier ones set.
- **The OpenSandbox charts.** OpenSandbox doesn't publish its 1.1.1 charts to a Helm repository; they
  ship in its source tree at the release tag. Fetch that tag and check it's the commit this doc was
  tested with:

  ```bash
  git clone --depth 1 --branch release-1.1.1 https://github.com/opensandbox-group/OpenSandbox.git
  git -C OpenSandbox rev-parse HEAD   # 531700ca18a7fa5fab6a2f25d6dc68a76eddab12
  export OPENSANDBOX_SRC=$PWD/OpenSandbox
  ```

  Without GitHub access, ask the Helix team for the three charts packaged with `helm package`, and
  use those `.tgz` files wherever the commands say `"$OPENSANDBOX_SRC/manifests/charts/<chart>"`.
- **Nothing else from the NeMo Helix repository.** Step 5 downloads one file from it.
- For each Helix install, its tenant namespace (`TENANT_NS` below), provisioned like any other
  tenant: pull Secrets, quotas, the platform's own NetworkPolicy. Its build namespace,
  `$TENANT_NS-builds`, is covered under [Builder](#builder).
- For Gym, the tenant's job storage volume must be **ReadWriteMany** (for example an NFS-backed
  storage class): the Gym host mounts the same claim as the job pods, possibly from another node.
  The Helix chart's claim is `nemo-helix-core-storage`.
- Nodes can pull from docker.io, or every image above is mirrored and the values point at the mirror.

## 1. CRDs and controller

```bash
oc create namespace opensandbox-system

helm upgrade --install opensandbox-base "$OPENSANDBOX_SRC/manifests/charts/base" \
  -n opensandbox-system -f values/base.yaml
helm upgrade --install opensandbox-controller "$OPENSANDBOX_SRC/manifests/charts/controller" \
  -n opensandbox-system -f values/controller.yaml --wait

oc get crd batchsandboxes.sandbox.opensandbox.io pools.sandbox.opensandbox.io sandboxsnapshots.sandbox.opensandbox.io
oc -n opensandbox-system rollout status deploy -l app.kubernetes.io/instance=opensandbox-controller
```

The controller must be able to run in `opensandbox-system` under its SCC (it runs as a non-root
user). The base release keeps the CRDs if it's uninstalled.

## 2. Prepare each tenant namespace

For each tenant namespace:

```bash
export TENANT_NS=<tenant namespace>

# The sandboxes' ServiceAccount, and the extension's pods/exec permission
sed "s/<TENANT_NS>/$TENANT_NS/g" manifests/tenant-namespace.yaml | oc apply -f -

# OpenShift: let sandboxes run as root with NET_ADMIN, and Builder's kaniko as root
sed -e "s/<TENANT_NS>/$TENANT_NS/g" -e "s/<BUILDS_NS>/$TENANT_NS-builds/g" manifests/scc.yaml | oc apply -f -

# Where a NetworkPolicy already restricts the namespace
sed "s/<TENANT_NS>/$TENANT_NS/g" manifests/networkpolicy-additions.yaml | oc apply -f -
```

What these are for:

- **SCC** (`manifests/scc.yaml`). Benchmarks run their task images unmodified, and many run as
  root. Every sandbox also has an `egress` sidecar that enforces the sandbox's network allowlist
  with nftables, which needs `NET_ADMIN`. The SCC is granted only to the `opensandbox-sandbox`
  ServiceAccount, which only sandbox pods use, and none of them gets a token. When you onboard
  another tenant, add its users to the two SCCs instead of applying them again.
- **NetworkPolicy** (`manifests/networkpolicy-additions.yaml`). Helix job pods must reach the
  server on port 80, and the server must reach sandbox pods on any port. If the platform manages
  tenant policies through GitOps, add the same rules there.

Check the SCC admits a sandbox-shaped pod:

```bash
cat <<EOF | oc -n "$TENANT_NS" policy scc-subject-review -f -
apiVersion: v1
kind: Pod
metadata: {name: scc-check}
spec:
  serviceAccountName: opensandbox-sandbox
  containers:
    - {name: task, image: docker.io/library/busybox:1.36, securityContext: {runAsUser: 0}}
    - {name: egress, image: docker.io/opensandbox/egress:release-1.1.1, securityContext: {capabilities: {add: [NET_ADMIN]}}}
EOF
```

The `ALLOWED BY` column should name `opensandbox-sandbox`.

## 3. Tenant keys

One key per tenant namespace. The server reads every key from one Secret in `opensandbox-system`;
each tenant namespace gets its own key in a Secret its Helix release reads.

```bash
# Generate a key for each tenant (repeat per tenant, with its own variable)
KEY=$(openssl rand -hex 32)

cat > tenants.toml <<EOF
[[tenants]]
name = "$TENANT_NS"
namespace = "$TENANT_NS"
api_keys = ["$KEY"]
EOF
# One [[tenants]] block per tenant namespace. Each must exist before the server starts.

oc -n opensandbox-system create secret generic opensandbox-tenants --from-file=tenants.toml
oc -n "$TENANT_NS" create secret generic opensandbox-server-api-key --from-literal=api-key="$KEY"
rm tenants.toml
```

To add a tenant later, add its block to the Secret's `tenants.toml`
(`oc -n opensandbox-system edit secret opensandbox-tenants`, or recreate it from the file) and
create its namespace-local Secret. The server reloads the file without a restart.

## 4. Sandbox pod template

Edit `manifests/batchsandbox-template.yaml`: replace `<PULL_SECRET>` with the pull Secret every
tenant namespace has (for example the namespace's NGC pull Secret), and add one per registry that
sandbox images come from: Harbor task images, Builder output, agent bundles, and the Gym host
(`nhx-gym-host`, from the same registry as the Helix release's images). Then:

```bash
oc apply -n opensandbox-system -f manifests/batchsandbox-template.yaml
```

The template gives each sandbox the `opensandbox-sandbox` ServiceAccount without a token, the
`/logs`, `/solution` and `/installed-agent` directories Harbor writes to, and keeps sandboxes off
control-plane nodes.

## 5. Compose services: the NeMo Helix extension

This step enables **Compose services**: benchmark tasks that ship a `docker-compose.yaml` (a
database or a mock API next to the agent's container) get those services as sidecars in their
sandbox pod. Many benchmarks need them, so this step isn't optional. The same extension also adds
agent bundles. It's a ConfigMap of Python files, `nemo-opensandbox-ext`, that the server
loads at startup. It ships in the NeMo Helix repository
([github.com/NVIDIA-NeMo/nemo-helix](https://github.com/NVIDIA-NeMo/nemo-helix), public) at
`k8s/helm/files/opensandbox-ext/configmap.yaml`. Download it for the Helix version the tenants run;
no clone is needed.

**1. Find the Helix version.** It's the tag of the tenant's `nhx-api` image:

```bash
oc -n "$TENANT_NS" get deploy -o jsonpath='{range .items[*]}{.spec.template.spec.containers[0].image}{"\n"}{end}' | grep nhx-api
# nvcr.io/nvidia/nemo-helix/nhx-api:0.10.0                          -> HELIX_REF=0.10.0
# ghcr.io/nvidia-nemo/nemo-helix/nhx-api:<40-character commit SHA>  -> HELIX_REF=<that SHA>
```

The extension needs NeMo Helix **0.10.0 or later** (earlier versions don't know about tenants). If
tenants run different versions, use the newest one: one server serves them all.

**2. Download and apply it:**

```bash
HELIX_REF=0.10.0   # from step 1
curl -fsSL -o nemo-opensandbox-ext.yaml \
  "https://raw.githubusercontent.com/NVIDIA-NeMo/nemo-helix/$HELIX_REF/k8s/helm/files/opensandbox-ext/configmap.yaml"
head -4 nemo-opensandbox-ext.yaml   # "# GENERATED by script/generate_opensandbox_ext_configmap.py."

oc apply -n opensandbox-system -f nemo-opensandbox-ext.yaml
oc -n opensandbox-system get configmap nemo-opensandbox-ext   # DATA 4
```

If the cluster can't reach GitHub, the Helix team sends the file.

## 6. Server

```bash
helm upgrade --install opensandbox-server "$OPENSANDBOX_SRC/manifests/charts/server" \
  -n opensandbox-system -f values/server.yaml --wait

oc -n opensandbox-system logs deploy/opensandbox-server | grep -E "nemo-ext|tenant"
```

Expect `nemo-ext: compose services extension loaded`. The server refuses to start if a tenant
namespace in `tenants.toml` is missing, or if the extension can't load.

> [!IMPORTANT]
> **When a tenant's NeMo Helix is upgraded**, update the extension to match, or Compose tasks and
> agent bundles can fail:
>
> 1. Repeat step 5 with the new version.
> 2. Bump `opensandbox.nvidia.com/extension-revision` in `values/server.yaml` (for example `"1"` to `"2"`).
> 3. Run this step's `helm upgrade` again. It restarts the server, which reads the extension only
>    when it starts.

## 7. Check the server

From a tenant namespace, with that tenant's key:

```bash
KEY=$(oc -n "$TENANT_NS" get secret opensandbox-server-api-key -o jsonpath='{.data.api-key}' | base64 -d)
oc -n opensandbox-system port-forward svc/opensandbox-server 18080:80 >/dev/null &
H="OPEN-SANDBOX-API-KEY: $KEY"

curl -s localhost:18080/health                    # {"status":"healthy"}
curl -s -H "$H" localhost:18080/v1/nemo-ext/health
# {"status":"ok","extension":"nemo-compose-services","features":["compose-services","agent-bundle"]}

# A sandbox, in the tenant's namespace
ID=$(curl -s -H "$H" -H 'Content-Type: application/json' localhost:18080/v1/sandboxes -d '{
  "image": {"uri": "docker.io/library/busybox:1.36"}, "entrypoint": ["sleep", "infinity"],
  "timeout": 300, "resourceLimits": {"cpu": "250m", "memory": "256Mi"},
  "networkPolicy": {"defaultAction": "deny", "egress": [{"action": "allow", "target": "pypi.org"}]}}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')
oc -n "$TENANT_NS" get pods -l opensandbox.io/id="$ID"     # Running, with the egress container
curl -s -X DELETE -H "$H" localhost:18080/v1/sandboxes/"$ID"

# Compose services: a sandbox with a Redis service beside it
SERVICES='{"services": [{"name": "cache", "image": "docker.io/library/redis:7-alpine", "ports": [6379],
  "readiness": {"exec": ["redis-cli", "ping"]}}]}'
ID=$(curl -s -H "$H" -H 'Content-Type: application/json' localhost:18080/v1/sandboxes -d "{
  \"image\": {\"uri\": \"docker.io/library/busybox:1.36\"}, \"entrypoint\": [\"sleep\", \"infinity\"],
  \"timeout\": 300, \"resourceLimits\": {\"cpu\": \"250m\", \"memory\": \"256Mi\"},
  \"extensions\": {\"opensandbox.extensions.nemo-compose-services\": $(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$SERVICES")}}" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')
oc -n "$TENANT_NS" wait pod -l opensandbox.io/id="$ID" --for=condition=Ready --timeout=300s
curl -s -H "$H" localhost:18080/v1/nemo-ext/sandboxes/"$ID"/services; echo     # "failure": null, "not_ready": []
curl -s -H "$H" -H 'Content-Type: application/json' localhost:18080/v1/nemo-ext/sandboxes/"$ID"/services/cache/exec \
  -d '{"command": ["redis-cli", "ping"]}'; echo                                  # "stdout": "PONG\n"
curl -s -X DELETE -H "$H" localhost:18080/v1/sandboxes/"$ID"
kill %1
```

The last check runs a command in the Compose service through the extension, which needs both the
extension (step 5) and its exec Role in the tenant namespace (step 2).

## 8. Point each Helix release at the server

**Who:** the Helix team, once per Helix release (one per tenant namespace). Until now the release
doesn't know the server exists. This step tells it where the server is and which key to use, turns
on Builder, which Harbor taskset runs need to build task images, and turns on sandboxed Gym.

**Before you start:**

- Steps 1–7 are done, and `$TENANT_NS` has the Secret `opensandbox-server-api-key` from step 3.
- The account running `helm upgrade` can create NetworkPolicies in `$TENANT_NS` (the release adds
  one, below) and the namespace `$TENANT_NS-builds` (see [Builder](#builder)).
- You know the release's name, its chart version, and the values files it was installed with:
  `helm list -n "$TENANT_NS"` and `helm get values <release> -n "$TENANT_NS"`.
- A registry Builder can push task images to, with a username and password.

**8a. Fill in the values file.** Copy `helix/helix-values.yaml` and set its placeholders:

| Placeholder | What to put | Example |
| --- | --- | --- |
| `<REGISTRY_HOST>` | The registry Builder pushes task images to, host only | `nvcr.io` |
| `<PREFIX>` | The project or path in that registry that builds go under | `"my-org/helix-builds"` |
| `<IMAGE_REGISTRY>`, `<IMAGE_TAG>` | Where the release's own images come from, and their tag: the Gym host image is `<IMAGE_REGISTRY>/nhx-gym-host:<IMAGE_TAG>`. Same as `platformConfig.platform.image_registry` and `image_tag` | `nvcr.io/nvidia/nemo-helix`, `0.10.0` |

The rest needs no editing. The release's own API address is filled in by a chart template. The file:

- points the release at `opensandbox-server.opensandbox-system.svc.cluster.local`, with the key in
  `opensandbox-server-api-key`;
- sets the **Harbor sandbox policy** (below);
- turns Builder and sandboxed Gym on.

**Compose services stay on** even though the file says `composeServices.enabled: false`. That
setting only decides whether the release creates the extension's ConfigMap and Role itself. Here
the platform created them in steps 2 and 5, because a tenant release can't write to
`opensandbox-system`, and several releases would all try to own the same ConfigMap.

If the release's job storage claim isn't `nemo-helix-core-storage`, change
`sandbox_job_storage_pvc_claim` to match.

**The Harbor sandbox policy.** A Harbor taskset run applies a *sandbox profile* to every sandbox it
creates. The file sets the default profile, `harbor-standard`, under
`platformConfig.evals.runs.harbor.sandbox_profiles`:

| Key | Set to | What it controls |
| --- | --- | --- |
| `egress_allow` | The release's API, plus PyPI, the Debian, Ubuntu and Alpine mirrors, npm, yarn, GitHub and astral.sh | What a `public` task's sandboxes may reach for the whole trial, agent run and grading included. Terminal-Bench tasks are all `public`, and some install packages during the run or grading, so the API alone isn't enough. Narrow it for benchmarks that need less. |
| `agent_channel` | (chart default: the release's API) | The model endpoint for `no-network` and `allowlist` tasks, which otherwise reach nothing |
| `agent_setup_egress` | (chart default: package registries and agent installers) | Extra hosts open only while the agent installs itself, then closed |
| `max_cpus`, `max_memory_mb` | 16, 32768 | The largest sandbox a task may ask for; larger requests are refused at submit |
| `max_gpus` | 0 | GPU tasks are refused. Raise it, and set `gpu_types`, on clusters with GPU nodes |
| `max_sandbox_lifetime_s` | 30600 (8.5 h) | How long OpenSandbox keeps a sandbox: the longest trial plus a margin |

Everything else is denied: each sandbox's egress sidecar enforces the profile. Runs choose another
profile with `--sandbox-profile <name>`; add profiles beside `harbor-standard` in the same map. The
NeMo Helix guide *Configure Harbor Taskset Runs* lists every key.

**8b. Upgrade the release** with its existing values plus this file (later files win):

```bash
helm upgrade <release> nemo-helix/nemo-helix --version <chart version> -n "$TENANT_NS" \
  -f <the release's values files> -f helix-values.yaml --wait --timeout 15m
```

The upgrade changes four things in `$TENANT_NS`:

- **Job pods get the server's address and the tenant key**, as `OPEN_SANDBOX_DOMAIN` and
  `OPEN_SANDBOX_API_KEY` (from `opensandbox-server-api-key`).
- **A NetworkPolicy** (`<release>-opensandbox-sandbox-ingress`) lets only `opensandbox-system` reach
  sandbox pods.
- **Builder is on**, with its build namespace `$TENANT_NS-builds`, created by the release.
- **Gym runs sandboxed:** `agent-evaluate` jobs start a Gym host through the server instead of running
  Gym in the job pod. The Gym host may reach only the model URLs in `sandbox_policy_base_urls`.

**8c. Give Builder its registry credentials.** Builder pushes and signs images with three secrets
in each Helix workspace that runs taskset evaluations:

```bash
openssl ecparam -name prime256v1 -genkey -noout | openssl pkcs8 -topk8 -nocrypt -out signing.key
nemo secrets create builder-registry-username --workspace <workspace> --value <registry username>
nemo secrets create builder-registry-password --workspace <workspace> --from-file <file holding the password>
nemo secrets create builder-signing-key --workspace <workspace> --from-file signing.key
rm signing.key
```

**8d. Check it:**

```bash
oc -n "$TENANT_NS" get configmap -o name | grep -- '-config$'     # the release's config, e.g. nemo-helix-config
oc -n "$TENANT_NS" get configmap <release>-config -o jsonpath='{.data.config\.yaml}' \
  | grep -E 'sandbox_server_domain|sandbox_cluster_capable|sandboxed_gym_default|sandbox_runtime_image|sandbox_job_storage_pvc_claim'
# sandbox_cluster_capable: true          (twice: platform and evals)
# sandbox_server_domain: opensandbox-server.opensandbox-system.svc.cluster.local
# sandboxed_gym_default: true
# sandbox_runtime_image: <IMAGE_REGISTRY>/nhx-gym-host:<IMAGE_TAG>
# sandbox_job_storage_pvc_claim: nemo-helix-core-storage
oc -n "$TENANT_NS" get pvc nemo-helix-core-storage   # ACCESS MODES RWX
# The Harbor sandbox policy: harbor-standard's egress_allow starts with the release's API
oc -n "$TENANT_NS" get configmap <release>-config -o jsonpath='{.data.config\.yaml}' | grep -A3 'egress_allow:'
oc -n "$TENANT_NS" get networkpolicy | grep opensandbox-sandbox-ingress
oc get namespace "$TENANT_NS-builds"
```

## 9. Run an evaluation end to end

**Harbor:**

```bash
nemo evals runs submit --harbor-dir <folder of task dirs> --workspace <workspace> --agent oracle
nemo evals runs results <run id>
```

An oracle run of a simple task (hello-world) should score 1.0. A task with a `docker-compose.yaml`
exercises the extension.

**Gym:** submit an `agent-evaluate` job with a Gym target, as in the NeMo Helix guide *Evaluate a
NeMo Gym Environment*:

```bash
nemo evals agent-evaluate --spec-file gym_agent_eval.json
```

While it runs, a Gym host sandbox appears in the tenant namespace with the job storage claim mounted:

```bash
oc -n "$TENANT_NS" get pods -l opensandbox.io/id \
  -o jsonpath='{range .items[*]}{.metadata.name} {.spec.containers[0].image} {.spec.volumes[*].persistentVolumeClaim.claimName}{"\n"}{end}'
# <id>-0  <IMAGE_REGISTRY>/nhx-gym-host:<IMAGE_TAG>  nemo-helix-core-storage
```

The job should complete with rewards, and the Gym host sandbox should be gone afterwards.

## Known issue: sandboxed Gym on a tenant server

> [!WARNING]
> **Sandboxed Gym doesn't work with this server yet.** Harbor does.
>
> In tenant mode the server requires the API key on proxied requests too, not only on its own API.
> Evaluator calls the Gym host through the server's proxy (`/sandboxes/<id>/proxy/<port>/…`), and
> NeMo Helix sends only the headers the server returns for that address, which don't include the
> key. Every call to the Gym host gets `401 MISSING_API_KEY`, so jobs fail waiting for the host to
> become ready. Harbor's requests go through the OpenSandbox SDK, which sends the key, so Harbor is
> unaffected.
>
> The fix belongs in NeMo Helix: send the API key on Gym host requests when they go through the
> server's proxy. Use a Helix version that includes it for tenants that run Gym. Nothing in this
> install changes.

## Builder

Harbor taskset runs build every task's images with NeMo Helix Builder, which runs kaniko in
`<release namespace>-builds`:

- The Helix chart creates that namespace, its ServiceAccounts, RBAC and work volume. If the
  platform must create namespaces itself, pre-create it with Helm ownership metadata
  (`app.kubernetes.io/managed-by=Helm`, and annotations `meta.helm.sh/release-name` and
  `meta.helm.sh/release-namespace`) so the release can adopt it.
- It needs the same SCC as the tenant namespace for the build steps (UID 1000), plus
  `nemo-helix-builder-kaniko` for kaniko (UID 0), from `manifests/scc.yaml`.
- kaniko resolves names with `8.8.8.8` and `1.1.1.1` unless `platformConfig.builder.sandbox.dns_nameservers`
  is set. Set it where public resolvers are blocked.
- It needs a registry to push to, with credentials stored as Helix workspace secrets, and the
  sandbox template's pull Secrets must cover that registry.

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| Server crash-loops: `nemo-ext: not starting without the compose services extension` | The extension ConfigMap doesn't match this server. Apply the file from NeMo Helix 0.10.0 or later, which supports tenants. |
| Server won't start, naming a namespace | A tenant in `tenants.toml` names a namespace that doesn't exist or the server can't read. |
| Every request returns 401 | The client's key isn't in `tenants.toml`, or `OPENSANDBOX_SERVER_API_KEY` is set as well. |
| Sandbox pod `CreateContainerError` / forbidden by SCC | The `opensandbox-sandbox` SCC doesn't list this namespace's ServiceAccount, or the template's `serviceAccountName` was changed. |
| Sandbox pods never Ready: `Readiness probe failed` on `egress` | A NetworkPolicy blocks the kubelet's probe to port 18080 (some CNIs police node traffic). Admit the node range: Helix value `opensandbox.sandboxNetworkPolicy.probeSourceCIDRs`. |
| Trials fail with `Environment start timed out`, pods `ImagePullBackOff` | The template's `imagePullSecrets` don't cover the task image's registry, or the Secret is missing in the tenant namespace. |
| Compose task fails with a permission error from the services route | The `nemo-opensandbox-ext` Role/RoleBinding is missing in the tenant namespace. |
| Gym job fails waiting for the Gym host; the server logs `401` on `/sandboxes/<id>/proxy/…` | The Helix version predates the fix in [Known issue](#known-issue-sandboxed-gym-on-a-tenant-server). |
| Gym host pod `Pending` with a volume error | The job storage claim isn't ReadWriteMany, or `sandbox_job_storage_pvc_claim` names a claim that doesn't exist in the tenant namespace. |
| Gym host can't reach its model | The model URL isn't in `sandbox_policy_base_urls`. The Gym host's egress allows only those. |

## Uninstall

```bash
helm -n opensandbox-system uninstall opensandbox-server opensandbox-controller opensandbox-base
```

The CRDs stay (with any sandboxes still in them); delete them deliberately, after checking
`oc get batchsandboxes -A` is empty.
