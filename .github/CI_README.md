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

  Python integration tests run in two duration-balanced, whole-package groups
  with four xdist workers each. `services` covers services except core secrets;
  `platform` covers secrets, plugins, packages, and root tests. Both retain normal
  discovery and `loadgroup` scheduling. Reports and crash dumps are separate per
  group; the coverage comment combines raw data from both groups.

  Local `make test-integration` still runs the entire suite. To select a group,
  use `make test-integration PYTEST_EXTRA=--integration-group=services` (or
  `platform`). Allocation lives in `tests/integration_groups.py`. Grouping does
  not change whether integration tests block merges or introduce retries.

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

## Post-merge image and end-to-end checks

Container image builds and platform e2e checks retain their `main` push and
`workflow_dispatch` behavior, including path filters and publishing inputs.
Merge-group validation does not run these checks.

For PRs to `main` or `release/*`, maintainers can opt in by adding **`e2e`** to a
**non-draft, same-repository PR**. Forks and missing/invalid PR context cannot opt
in. Current labels are checked on synchronize, reopen, ready-for-review, and
label events; removing `e2e` or converting to draft cancels/replaces the prior
PR run and skips opt-in jobs. No `pull_request_target` or new permissions are used.

The label enables CPU image publication, all Kind suites, Authentik/ZITADEL
runtime tests and their static prerequisites, and both Python e2e jobs even
without matching path changes. Existing policy-WASM and successful published
image prerequisites remain enforced. The auth-IDP static/runtime checks and
both Python e2e results participate in the `CI status` aggregate.

Separate garak and Safe Synthesizer image builds are not enabled by the label.
Studio UI e2e and Docker source-image smoke remain manual-only. Manual
`publish: false` still skips tests needing published images. These opt-in checks
are expensive and execute trusted PR code; apply the label deliberately.

Lint, unit tests, integration tests, wheel build/install checks, Helm validation,
and Docker bake-graph validation stay pre-merge. Printing the bake graph does
not build an image. Integration required-check policy is unchanged.

On automatic `main` runs, `Post-merge image and e2e status` waits for the image/e2e
jobs and their prerequisites, writes a result summary, and sends one failure
alert per run through the existing `SLACK_ALERTS_WEBHOOK`. The alert identifies
failed/cancelled jobs, the commit, and the run containing logs and artifacts.
Path-filtered skips are normal; prerequisite failures still alert when their
dependent tests cannot run. Missing webhook configuration or failed delivery
fails the notification job rather than silently dropping an alert. Manual runs
do not send Slack alerts. Main-run concurrency behavior is unchanged.

## GitHub Apps

- DCO app
  Developer Certificate of Origin (DCO) checks are handled by the installed DCO
  app, not by a repository workflow. The `dco-war.yaml` workflow exists only
  because the DCO app does not currently understand merge queue checks. The
  ruleset allows any source named `DCO` to satisfy the required check so merge
  queue entries can pass. When the DCO app supports merge queues, change the
  ruleset to require the app-owned DCO check instead and remove the workaround.
