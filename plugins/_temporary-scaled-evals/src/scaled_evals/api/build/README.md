<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# `scaled_evals.api.build` — task image build

Builds a task revision's sandbox image at `POST /tasks/{id}/finalize`.
This note records **what exists today and its known limitations**, so the
context behind the current choices isn't lost. Architecture rationale lives in
[docs/internals/ARCHITECTURE.md § Container Build](../../../../docs/internals/ARCHITECTURE.md#container-build).

## Layout

- **`queue_worker.py`** — shared build execution. Platform Jobs calls it for one
  persisted revision; the legacy worker can still claim rows when Platform Jobs
  is disabled.
- **`worker.py`** — legacy synchronous primitives retained for compatibility.
- **`buildkit.py`** — local fallback builds via `buildctl`/gRPC.
- **`image_builder_service.py`** — image-builder-service build/sign integration.
- **`cloud_build.py`** — Google Cloud Build + GAR integration for GKE installs
  where task packs already live in GCS.

## Current status (MVP)

- **Multiple build approaches behind one queue.** The same finalize queue can
  run image-builder-service builds, Google Cloud Build builds from GCS to GAR,
  prebuilt-image registry verification, or local BuildKit fallback.
- **Scheduling is durable.** Finalize stores backend parameters and credential
  IDs on `task_revisions`; the Platform Jobs controller claims rows with
  `FOR UPDATE SKIP LOCKED` and binds each attempt to a deterministic Job.
  Decrypted credentials never enter the queue. Stale leases are reclaimed and
  failures retry up to the configured attempt limit.

## Current deployment shape

Kubernetes installs use the Helm chart (`charts/scaled-evals`); the raw
`k8s/workloads` manifests are retired. See
[`docs/KUBERNETES_DEPLOYMENT.md`](../../../../docs/KUBERNETES_DEPLOYMENT.md).

- **Hosted (signed-images-only cluster):** `buildkit.enabled=false` — the sandboxed cluster exposes no
  BuildKit-compatible runtime/SCC. Finalize uses the image-builder-service
  build/sign path or prebuilt `--image-ref`/`--image-digest` registry
  verification against the signed-image policy.
- **GKE:** Google Cloud Build builds the uploaded GCS task pack and pushes
  to GAR; otherwise the build worker only resolves and records prebuilt task
  images. BuildKit source builds stay disabled.
- **Local compose:** the in-network BuildKit + registry pair remains the local
  fallback build path.

## Future work (recorded; not in scope for this MVP)

- **Queue policy hardening:** per-tenant concurrency, explicit cancellation,
  retry classification, and queue observability metrics.
- **Caching (BuildKit fallback path), in bang-for-buck order:**
  1. PVC for the buildkit state dir → cache survives pod restarts.
  2. `--export-cache`/`--import-cache type=registry` → cache survives *and* is
     shared across daemons; prerequisite for >1 buildkit replica.
  3. Content-hash **skip-if-exists** (tag by build-context hash, check the
     registry, reuse) → avoid rebuilds entirely; also makes re-finalize
     idempotent. (Promised as "content-hash cache on repeat runs" in API.md.)
- **Hardening for untrusted multi-tenant input** (engine-independent): build
  **timeout**, image **size cap**, per-tenant build **quota/isolation**, allowed
  **base-image** policy. The single shared buildkitd is also a cross-tenant cache
  surface to consider here.
