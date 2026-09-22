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

## Status — the spine closes

**A `POST /builds` now produces signed images whose digests the control plane observed from a
registry.** End to end, on minikube with Calico, against an in-cluster registry.

```
submit ──▶ 2 rows `pending`
       ──▶ fetch      (SA nmp-build-fetch,   has work volume)   Completed
       ──▶ supervise  (SA nmp-build-control, NO work volume)    Completed
             └─ 2 sandbox pods, baseline PSS, no SA token, kaniko
       ──▶ push       (SA nmp-build-push,    has work volume)   Completed
       ──▶ reconciler ──▶ 2 rows `ready`
```

Verified independently of the control plane, from inside the cluster:

| | |
|---|---|
| row digest == what the registry serves | ✅ both images, exactly |
| signature artifact at the `.sig` tag | ✅ both |
| `cosign verify` against the public key | ✅ |
| three steps, three ServiceAccounts | ✅ observed on the real pods |
| `supervise` has **no** work-volume mount | ✅ observed — absence is the control, and it holds |
| namespace still refuses the BuildKit posture | ✅ re-run after the green build |
| sandbox reaches the internet and nothing private | ✅ 7/7, against cluster-derived addresses |
| 111 unit tests, `ruff` and `ty` clean | ✅ |

### The two environments, and why both exist

Neither alone is sufficient, and it is worth being precise about which answers what.

**minikube (`deploy/local/`) — the regression environment.** It closes the spine, because the
control plane runs *in* the cluster and build pods can call back to it. That was the single thing
that blocked the first run. Everything about the builder's own logic is provable here.

**GKE (`deploy/README.md`) — the fidelity environment.** It holds the cloud-shaped facts, and a
green local run would not have caught any of them:

- the **metadata server** does not exist locally, so threat-model path A cannot be exercised —
  `sandbox-egress-probe.sh` says so rather than claiming a pass it did not earn
- **NodeLocal DNSCache** is a GKE addon, so the link-local DNS collision — the most surprising
  finding in this work — cannot reproduce locally
- the registry is **anonymous and plain HTTP**, so `push`'s credential handling and the
  reconciler's Bearer-challenge/token-endpoint path never execute
- single-node **hides the ReadWriteOnce constraint** entirely; node pinning is a no-op here
- SQLite and one replica, so nothing about Postgres or the reconciler's single-writer property
  under more than one replica is tested. `replicas: 1` is load-bearing.

## Eight defects the real runs found

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

And four more that only the in-cluster run could surface:

5. **The sandbox log arrives as the repr of a bytes object.** When the Kubernetes client
   deserializes a non-UTF-8 log body into its declared `str` return type, newlines come back as
   the two characters backslash-n. `splitlines()` then yields one line, every result marker
   disappears, and the symptom is *every image reporting "no result recorded" while every build
   succeeded*. Nothing raises. kaniko's output is ANSI-coloured, so this is the normal case.
   Fixed by reading the raw body with `_preload_content=False` and decoding it directly.
6. **`/var/run` is a symlink to `/run`.** `validate_layout` resolved its walk root but compared
   the results against the unresolved path, so every file looked "unexpected" and a correct
   layout was refused for being correct.
7. **`storageClassName: standard-rwo` is a GKE name.** On any other cluster the claim stays
   `Pending` and every build pod is unschedulable with "pod has unbound immediate
   PersistentVolumeClaims" — which reads as a scheduling problem, not a portability bug. Now the
   cluster default.
8. **The jobs controller also reconciles its own namespace.** Its default execution profile
   targets `POD_NAMESPACE`, so without a grant there it logs a 403 on every poll forever, while
   builds themselves work fine.

And one from `deploy/README.md` worth repeating, because it generalises past any one cluster:
**the link-local denial that closes the metadata server also closes NodeLocal DNSCache.** Allowing
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

Two environments. The local one proves the builder; the cloud one proves the cloud-shaped facts.

### Local (minikube) — closes the spine

```bash
# Calico, NOT the default CNI. kind's kindnet and minikube's default do not enforce
# NetworkPolicy at all, which makes the sandbox policy decorative -- silently.
minikube start --driver=docker --cni=calico --cpus=3 --memory=5500
kubectl label node minikube nmp.nvidia.com/build-node=true

kubectl apply -f plugins/nemo-builder/deploy/          # substrate (manifests only)
plugins/nemo-builder/deploy/negative-control.sh        # MUST report refused
plugins/nemo-builder/deploy/sandbox-egress-probe.sh    # MUST report 7/7

# images. `nmp-api` needs the Studio UI context stubbed out -- it is a Node build this does
# not use, and it is the slowest and flakiest part of the chain.
mkdir -p /tmp/empty-studio/artifacts && touch /tmp/empty-studio/artifacts/.keep
docker buildx bake -f docker-bake.hcl nmp-api-docker --load \
  --allow=fs.read=/tmp/empty-studio --set '*.platform=linux/arm64' \
  --set nmp-api-docker.contexts.nmp-studio-ui=/tmp/empty-studio
docker buildx build --platform linux/arm64 -f plugins/nemo-builder/docker/Dockerfile.platform \
  --build-arg NMP_API_IMAGE=my-registry/nmp-api:local -t nmp-api-builder:local --load .
docker buildx build --platform linux/arm64 -f plugins/nemo-builder/docker/Dockerfile \
  -t nmp-build:v3 --load .
for i in nmp-api-builder:local nmp-build:v3 nmp-jobs-launcher:local; do minikube image load $i; done

kubectl apply -f plugins/nemo-builder/deploy/local/
kubectl create configmap nemo-platform-config -n nmp-platform \
  --from-file=config.yaml=plugins/nemo-builder/config/platform-config.minikube.yaml
kubectl -n nmp-platform port-forward svc/nemo-platform 8080:8080
```

**Use a new image tag for every rebuild.** `minikube image load` over an existing tag does not
reliably replace what the kubelet already cached, and the symptom is a fix that appears not to
work — which costs a full debugging cycle chasing the wrong thing.

### The fast loop — no cluster at all

```bash
.venv/bin/pytest plugins/nemo-builder/tests -q
.venv/bin/ruff check plugins/nemo-builder && .venv/bin/ty check plugins/nemo-builder
```

### Cloud (GKE) — see `config/platform-config.example.yaml`

Two environment notes that cost time:

- `uv run` needs uv ≥ 0.10.10 (bumped 2026-09-16). Even then `uv run --frozen` fails here
  building `litellm` against rustc 1.93.1 — use `uv run --no-sync`, or the `.venv/bin/` tools
  directly as above.
- The full `--service-group all` pulls in `nemo-data-designer`, whose installed `data_designer`
  is stale in this venv (`SeedReaderConfigError` import fails). Unrelated to the builder.
