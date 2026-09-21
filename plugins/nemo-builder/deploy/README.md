<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Build namespace substrate

The cluster side of [RFC 001](../../../../journal/RFCs/001-in-cluster-container-image-build-system.md)'s
execution mode: the namespace a build runs in, and everything bounding the pod inside it. No
platform code is involved — this is `kubectl apply` and two assertion scripts.

```
00-namespace.yaml         nmp-builds, enforcing the `baseline` Pod Security Standard
10-serviceaccounts.yaml   three SAs -- the trust split IS these three strings
20-rbac.yaml              pod create/delete for `nmp-build-control` and nothing else
30-networkpolicy.yaml     the sandbox egress policy    <- the high-severity one
40-work-volume.yaml       the shared work PVC

negative-control.sh       asserts the namespace still REFUSES the BuildKit posture
sandbox-egress-probe.sh   asserts what the NetworkPolicy actually closes, with packets
```

## Apply

```bash
kubectl apply -f plugins/nemo-builder/deploy/
kubectl label node <one-node> nmp.nvidia.com/build-node=true    # see 40-work-volume.yaml
./negative-control.sh && ./sandbox-egress-probe.sh
```

Both scripts must pass before any build in this namespace means anything. The first proves the
namespace is enforcing at all; the second proves the egress policy is not decorative. Re-run
`sandbox-egress-probe.sh` after every edit to `30-networkpolicy.yaml` — that file is not
reviewable by eye.

## Measured on `log2-gke-lab`, 2026-09-21

Spiked by hand before any platform code existed, which is the order RFC 001's *Early validation*
section argues for. Every assumption held except one, and that one would have been expensive to
find later.

| Question | Answer |
|---|---|
| Does `nmp-builds` refuse the BuildKit posture? | **Yes** — `violates PodSecurity "baseline:latest"`, naming AppArmor and seccomp |
| Does kaniko's posture clear `baseline`? | **Yes**, zero relaxations, with `drop: [ALL]` + the five capabilities |
| Does a `RUN pip install` reach the internet through the policy? | **Yes** |
| Is the metadata server reachable? | **No** — path A closed |
| Are the Service CIDR, Pod CIDR, API server reachable? | **No** |
| Does kaniko write a usable OCI layout to the shared PVC? | **Yes** — valid `index.json`, `oci-layout`, blobs |
| Does `crane push` accept an OCI **layout directory**? | **Yes** — no skopeo fallback needed |
| Does the registry's digest match kaniko's offline digest? | **Yes**, byte for byte |
| Is the cosign `.sig` tag present for the reconciler to find? | **Yes** |
| Does `cosign verify` pass against the public key? | **Yes** |

### The one that did not hold: cluster DNS

The first draft of `30-networkpolicy.yaml` allowed egress to kube-dns, which is the standard
companion to a default-deny policy. It did not work, and the reason generalises:

**This cluster runs GKE's NodeLocal DNSCache addon**, so pods resolve against `169.254.20.10` — a
**link-local** address. The `169.254.0.0/16` denial that closes the metadata server closes
node-local DNS with it, and the build died at `lookup index.docker.io: i/o timeout`. The kube-dns
allow rule was never consulted.

Adding the kube-dns rule back "fixed" DNS and **reopened the Pod CIDR** — measured, `nc` to a
kube-dns pod IP on `:53` succeeded — because that rule is by construction an exception to the
`except` list.

The resolution is to stop giving the sandbox cluster DNS at all: `dnsPolicy: None` with public
resolvers. It needs to resolve `pypi.org`, not `kubernetes.default.svc`. That removes the DNS rule,
closes the Pod CIDR completely, and leaves the sandbox unable to resolve an internal name at all
(`NXDOMAIN`) — strictly more closed than the version with a DNS hole in it.

**Carry this forward:** when the pull-through mirror lands (`M2-1`) it is in-cluster, and a sandbox
with public-only DNS cannot resolve it. Reach it by ClusterIP via `--registry-mirror` plus a
`hostAliases` entry, and add that one address back as its own narrow egress rule — a deliberate
single-address hole, not a restored DNS rule.

## PoC deviations

- **`ReadWriteOnce` work volume + node pinning.** This cluster has no RWX StorageClass
  (`kubectl get csidrivers` → only `pd.csi.storage.gke.io`; Filestore CSI not installed). See the
  header of `40-work-volume.yaml`. Production needs RWX; pinning is a throughput ceiling.
- **`runtimeClassName` unset.** gVisor exists here, but its nodes are tainted and in separate
  pools, which fights the RWO pinning. The lab measured kaniko runs fine under gVisor, so this is
  a scheduling decision, not a capability one.
- **Upstream `gcr.io/kaniko-project/executor`**, not the fork RFC 001 asks for (`M2-3`).
- **Static cosign key in a Secret**, not KMS (`M1-7`). Reuses the lab keypair so
  `scratch/builderlab/cosign.pub` still verifies what this produces.
- **GAR credential is a ~60-minute access token.** Re-mint if a push 401s; production uses
  Workload Identity.
