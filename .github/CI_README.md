<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# GitHub Automation Docs

This directory contains the repository's GitHub automation: workflow files,
reusable actions, and supporting docs.

## What Lives Here

- `workflows/`
  GitHub Actions workflow definitions for source validation, Studio CI,
  documentation, security scanning, semantic PR checks, DCO merge-queue
  handling, and releases.

- `actions/`
  Local composite actions shared across workflows: change detection, policy
  WASM builds, disk cleanup, and self-hosted runner metadata.

## Workflow Overview

### CI Check Workflows

- `ci.yaml`
  Main source validation workflow. It runs linting, OPA policy WASM build,
  Python unit tests, Python integration tests, OPA policy tests, Studio web
  checks for relevant web changes, and PR coverage comments. It runs on pushes
  to `main`, pull requests to `main` or `release/*`, merge queue checks, and
  manual dispatch.
  On successful `main` pushes, it also sends a completion event to an external
  CI consumer.

  The final `ci-status` job is the merge gate for this workflow. Repository
  branch protection or rulesets should require `CI status`, not the individual
  test jobs. The job checks every job listed in its `needs` and passes only
  when each one is `success` or `skipped`, which lets path-filtered jobs remain
  optional. When adding a new CI job that should block merges, add it to
  `ci-status.needs`.

- `security.yaml`
  Security workflow. It runs TruffleHog secrets scanning and CodeQL analysis on
  pushes to `main`, pull requests to `main` or `release/*`, and manual dispatch.
  It also runs in merge queues so required checks can resolve, but the
  TruffleHog job intentionally skips its scan for `merge_group` events.

- `docs.yaml`
  Documentation workflow. It builds docs for relevant docs changes, deploys
  GitHub Pages on `main` pushes, tag pushes, and manual dispatch, deploys PR
  previews for same-repository PRs, and cleans up PR previews when those PRs
  close.

- `semantic-pull-requests.yaml`
  Pull request title validation.
- `dco-war.yaml`
  Merge queue compatibility shim for the DCO check. Normal DCO validation comes
  from the installed DCO app.

### Deployment and Release Workflows

- `release.yaml`
  Single workflow for scheduled and manual timestamped RC builds and versioned
  releases. It selects artifacts from the requested source, stages Helm,
  dispatches wheel/container work, polls final destinations, and signals
  deployment. See [RELEASING.md](../RELEASING.md) for inputs and destinations.

- `release-smoke.yaml`
  Verifies published charts, containers and wheels. Timestamped RC charts map
  to `.dev` wheels; numbered prereleases such as `1.0.0-rc1` remain distinct.

## Docker Layer Caches

The CPU, garak, and Safe Synthesizer image jobs use
`actions/configure-docker-cache` to configure Bake's registry-backed layer
cache. All events import the trusted `main` cache. Only `push` runs on
`refs/heads/main` export it; PRs (including forks), merge groups, and manual
runs never export. Fork PRs attempt anonymous imports, so private or missing
caches may be unavailable. BuildKit continues with a cold build when an import
is missing. Cache exports are best effort (`CACHE_IGNORE_ERROR=true`), so a
cache upload failure does not fail an otherwise successful image build.

Caches live in separate GHCR packages under
`ghcr.io/<owner>/<repo>/ci-cache/<scope>/<image>:cache-main-linux-<arch>`.
The scope is `cpu`, `garak`, or `safe-synthesizer`. This keeps parallel image
jobs from overwriting each other's base-layer caches. Main pushes are serialized
by the existing workflow concurrency group. The cache packages are outside the
NGC metadata package list used by `cleanup-ghcr-ci-images.sh`, so image cleanup
does not delete them. GHCR packages must allow this repository's Actions token
to read them and, for main pushes, write them. Other events only import caches
even when they have permission to publish CI images.

Caching reuses layers while building the current checkout, including the merge
group commit. Docker invalidates layers when their inputs change. For a forced
reset, bump Bake's `CACHE_VERSION` (default `cache`); remove obsolete cache
packages/versions separately when retiring a scope or version. Cache manifests
are replaced at stable tags; monitor GHCR storage for untagged historical
versions, which are not covered by CI image cleanup.

### Verification and Timing

1. Inspect the `Print Docker bake graph` step. Cached image targets should have
   `cache-from` entries under their scope. Main pushes should also have
   `cache-to` with `mode=max` and the matching architecture. Consumers should
   have no `cache-to`. The job summary records the import/export policy.
2. After landing on main, verify the first main build logs successful cache
   export. In a later PR or merge queue build, verify successful cache manifest
   imports and reused (`CACHED`) layers. An import warning on the first run is
   expected until main has populated the cache. Investigate persistent export
   warnings even if the image build passes.
3. Compare warm runs with similar changes against the November 2–3, 2026 sample:
   38 completed merge queue runs had median CPU image build time 15.8 minutes
   (13.6–28.8 minute range), median Gym E2E job time 9.3 minutes, and median
   elapsed time to `CI status` about 26 minutes. Record image-build duration,
   cache hits/misses, and elapsed time from workflow creation to the required
   gate. Keep cold builds and reruns separate; do not infer speedup from graph
   validation alone.

Local validation needs no registry access or Docker daemon: run
`make check-github-scripts`, then inspect `docker buildx bake --print` with
`BUILD_ARCH=linux/amd64`, the scoped `CACHE_REGISTRY`,
`BAKE_CACHE_SOURCE_BRANCH=main`, and `BAKE_CACHE_TARGET_BRANCH=main` (writer)
or an empty target branch (consumer). Workflow syntax is checked with
`flox activate --dir tools/actionlint -- actionlint`.

## GitHub Apps

- DCO app
  Developer Certificate of Origin (DCO) checks are handled by the installed DCO
  app, not by a repository workflow. The `dco-war.yaml` workflow exists only
  because the DCO app does not currently understand merge queue checks. The
  ruleset allows any source named `DCO` to satisfy the required check so merge
  queue entries can pass. When the DCO app supports merge queues, change the
  ruleset to require the app-owned DCO check instead and remove the workaround.
