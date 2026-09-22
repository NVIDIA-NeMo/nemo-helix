<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# nemo-builder — in-cluster container image builds

A proof of concept for [RFC 001](../../../journal/RFCs/001-in-cluster-container-image-build-system.md),
aimed at Milestone 2: a real Dockerfile builds in a `baseline`-enforcing namespace and pushes
signed. Built on the branch `builder-poc/albcui`, one commit per piece.

```
deploy/                  the cluster substrate -- namespace, SAs, RBAC, NetworkPolicy, PVC
  README.md              what was measured on log2-gke-lab, including one surprise
src/nemo_builder_plugin/
  identity.py            reference parsing, the digest model, the system tag
  schema.py              BuildSet / BuildSpec / BuildOutput / FileSetSource
  entities.py            ContainerImage -- the only entity in the design
  steps.py               the three step config contracts
  config.py              operator settings, including the kill switch
  compile.py             BuildSet -> three-step PlatformJobSpec, as a pure function
  submit.py              rows first, then the job
  service.py             POST /builds, and reads for the rows it creates
  registry.py            an OCI Distribution client (not a GAR one)
  controller.py          the reconciler -- the only writer of observed state
  run/                   the three step binaries: fetch, supervise, push
docker/Dockerfile        the nmp-build image the trusted steps run as
```

## What the design is, in one paragraph

A build compiles to **one job with three steps**, and the untrusted work happens in a pod none of
those steps runs in. `fetch` holds a Files client. `build` holds pod-create RBAC and nothing else.
`push` holds the registry credential and the signing key. The **sandbox** — a bare pod created by
`build`, not a job step — is the only thing that executes a caller's `RUN`, and it is the only
thing holding nothing at all: no Files client, no registry credential, no service account token.
Identity is a digest the *reconciler* reads from the registry, never one the build reported.

## Status — what is proven, and what is not

The spine is proven in **two halves that do not yet touch.**

### Proven on the cluster, by hand (`deploy/README.md`)

| | |
|---|---|
| `nmp-builds` refuses the BuildKit posture | ✅ `violates PodSecurity "baseline:latest"` |
| kaniko's posture clears `baseline` | ✅ zero relaxations, `drop: [ALL]` + five capabilities |
| a `RUN pip install` reaches the internet through the policy | ✅ |
| metadata server, Service CIDR, Pod CIDR, API server reachable | ✅ **no** — all four closed |
| kaniko writes a usable OCI layout to the shared PVC | ✅ |
| `crane push` accepts an OCI **layout directory** | ✅ no skopeo fallback needed |
| registry digest == kaniko's offline digest | ✅ byte for byte |
| cosign `.sig` tag present, `cosign verify` passes | ✅ |

### Proven through the platform

| | |
|---|---|
| `POST /builds` → rows `pending` → three-step job | ✅ |
| three steps carry three distinct profiles → three ServiceAccounts | ✅ |
| job schedules into `nmp-builds`, pods run under the right SAs | ✅ |
| the `nmp-build` image runs on a cluster node, dispatcher works | ✅ |
| step config delivered via ConfigMap and parsed | ✅ |
| reconciler loop runs, retries registry errors, fails unsigned rows | ✅ |
| 106 unit tests, `ruff` and `ty` clean | ✅ |

### Not proven — one blocker, and it is not in this code

**Build pods cannot call back to a control plane running on a laptop.** `fetch` started
correctly and died at `NemoTransportError: [Errno -2] Name or service not known`, resolving
`host.docker.internal:8080`. The Jobs Kubernetes backend happily *creates* pods from a laptop via
`load_kube_config()` — but the steps then call *back* to the platform (Files, Secrets, status),
and a laptop behind NAT has no address a GKE pod can reach.

That was a planning error rather than a code defect: running the control plane locally was chosen
for the fast iteration loop, on the assumption that traffic only flowed outward.

So these remain untested end-to-end:

- `fetch` actually downloading a fileset as the submitting principal
- `supervise` creating a sandbox from a real job
- `push` publishing and signing from a real job
- the reconciler flipping a real build's row to `ready`

Every one of them is exercised by unit tests against fakes, and every underlying mechanism is
measured by hand in `deploy/README.md`. What is missing is the join.

**To close it:** deploy the platform into the cluster (a platform image plus chart config), or
expose the local one through a tunnel. The tunnel is faster and publishes an auth-disabled
control plane at a public URL, which is why it was not done unilaterally.

## Four defects the first real submit found

Recorded because none was visible to a test that only looked at objects in memory.

1. **The executor discriminator vanished on the wire.** `CPUExecutionProvider.provider` is
   `Literal["cpu"] = "cpu"`; the Jobs client serializes with `exclude_unset`; a defaulted field
   is not "set". The server answered 422 *"Unable to extract tag using discriminator 'provider'"*.
2. **A registry host is not a repository path.** `default_registry` held
   `<host>/<project>/<repo>`, which reads fine in a log and builds `https://host/project/repo/v2/…`
   the moment anything resolves it.
3. **`push` received the credential and never handed it to anything.** crane and cosign both read
   a Docker config; the value has to reach a file.
4. **cosign's `k8s://` key needs RBAC.** It resolves through the Kubernetes API as the pod's own
   ServiceAccount, so `nmp-build-push` needs `get` on that one Secret — by `resourceName`, not
   `list`.

And one from `deploy/README.md` worth repeating, because it generalises past this cluster: **the
link-local denial that closes the metadata server also closes NodeLocal DNSCache.** Allowing
kube-dns back reopened the Pod CIDR, measured. The sandbox now uses public resolvers via
`dnsPolicy: None` — it needs to resolve `pypi.org`, not `kubernetes.default.svc` — which is
strictly more closed than the version with a DNS hole in it.

## Deliberately out of scope

Each of these is a real gap, not an oversight:

- **No pull-through mirror** (`M2-1`). Requirement 9 is unmet: base images resolve upstream and
  nothing at the network bounds what a Dockerfile may pull `FROM`. `registry_mirror` is wired but
  unset, and `--skip-default-registry-fallback` is emitted only when it is set.
- **No registry allowlist.** Three of its semantics are open decisions (`M1-9`). Nothing bounds
  where this system pushes; acceptable only because the PoC is single-tenant.
- **No `POST /container-images`** (`M1-10`) — M1's headline, but it proves nothing about builds.
- **No supersession or revision precondition.** Resubmitting is create-or-get only. A consequence
  found during the run: adopting an existing row is right for a concurrent submit and wrong when
  that row is already terminal-`failed`, so a re-run needs a new revision.
- **`source_digest` is never populated.** The context hash cannot reach the image as a label,
  because the only step that could apply it is `supervise`, which deliberately mounts nothing.
  Three ways out are written up at the bottom of `run/fetch.py`. The field is advisory and never
  identity, so this costs provenance detail rather than correctness.
- **No KMS** (`M1-7`), **no builder-image fork** (`M2-3`), **no `ephemeral-storage`** (`M1-3`),
  no admission bound or fairness, no health alerting.
- **`ReadWriteOnce` work volume with node pinning.** This cluster has no RWX StorageClass. Every
  build pod is pinned to one labelled node — a throughput ceiling, and the first thing to remove.
- **`runtime_class` unset.** gVisor exists here and kaniko was measured to run under it, but its
  nodes are tainted and in separate pools, which fights the RWO pinning.

## Running it

```bash
# 1. substrate, and the two assertions that make a green build mean anything
kubectl apply -f plugins/nemo-builder/deploy/
kubectl label node <one-node> nmp.nvidia.com/build-node=true
plugins/nemo-builder/deploy/negative-control.sh
plugins/nemo-builder/deploy/sandbox-egress-probe.sh

# 2. the fast loop -- no cluster, no registry, no database
.venv/bin/pytest plugins/nemo-builder/tests -q
.venv/bin/ruff check plugins/nemo-builder && .venv/bin/ty check plugins/nemo-builder

# 3. the platform. See deploy/platform-config.example.yaml; every CHANGE_ME must go.
#    NOTE the blocker above: with the control plane off-cluster, steps cannot call back.
export NMP_CONFIG_FILE_PATH=$PWD/plugins/nemo-builder/deploy/platform-config.example.yaml
export POD_NAMESPACE=nmp-builds
export NMP_DATA_DIR="$HOME/.local/share/nemo"
.venv/bin/nemo services run --services auth,entities,jobs,files,secrets,builder \
                            --controllers entities,jobs,builder --port 8080
```

Two environment notes that cost time:

- `uv run` needs uv ≥ 0.10.10 (bumped 2026-09-16). Even then `uv run --frozen` fails here
  building `litellm` against rustc 1.93.1 — use `uv run --no-sync`, or the `.venv/bin/` tools
  directly as above.
- The full `--service-group all` pulls in `nemo-data-designer`, whose installed `data_designer`
  is stale in this venv (`SeedReaderConfigError` import fails). Unrelated to the builder.
