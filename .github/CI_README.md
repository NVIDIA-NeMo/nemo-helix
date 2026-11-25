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

- `request-nvskills-ci.yml`
  Restores explicit `/nvskills-ci` requests for PRs changing
  `plugins/nemo-customizer/src/nemo_customizer/skills/nemo-customizer/`.
  A repository maintainer or admin must post the exact command. The workflow
  reads the current open, same-repository PR into `main`, paginates changed files
  (including rename sources), and dispatches the central NVIDIA NVSkills CI
  service with metadata only. It never executes PR code. Other skills are not
  opted into local requests yet. The central service independently determines
  which content to evaluate; this request filter does not restrict its scan scope.

  This pilot is manually requested and is not part of `ci-status.needs`.
  Do not restore the previous signing-author merge gate: it made skill changes
  unmergeable when the external pipeline could not sign.

  **MR gate policy:** keep the live `NVSkills CI` status and this request workflow
  out of required branch checks and out of `ci-status.needs`. Required offline
  Customizer tests already run in the unconditional plugin unit-test matrix:
  `test_skills.py` checks installed references/frontmatter,
  `test_skill_job_examples.py` validates documented job JSON against the backend
  schemas, and `test_skill_eval_dataset.py` checks authored task identity,
  coverage, routing labels and judge inputs. They need no model calls or GPUs.
  These checks validate authored artifacts, not an agent's success rate.

  Live evaluations are advisory for MRs and provide evidence for a separate
  catalog-publication review. A failed or incomplete live run remains failed
  or incomplete in its report; do not use `continue-on-error`, weaken safety
  assertions, or rerun until green to manufacture publication evidence.
  Record all scored attempts and inspect trajectories when judging disagrees.
  Pin the evaluator, agent runtime, agent model and judge model for comparisons.
  Three attempts are reliability measurements, not retries that erase earlier
  failures: SkillEvaluator 1.2.0 fails an agent's quality verdict if any scored
  with-skill attempt fails a required goal/security check, even when pass@3 is
  100%. An all-attempts-pass live gate therefore becomes more failure-prone as
  attempts increase. Promoting live results to an MR gate requires a separately
  reviewed statistical regression policy and evidence that the pinned runtime
  is reliable; a single run's PASS/FAIL label is insufficient.

  Setup and publication checklist:
  1. Keep this workflow and its helper under the existing `.github/` CODEOWNERS rule.
     Merge the workflow to the default branch before testing comment triggers.
  2. Configure `NVSKILLS_CI_DISPATCH_TOKEN` with permission to dispatch
     `nvskills-ci.yml` in `NVIDIA/nvskills-ci`. Confirm the central service's
     GitHub App can read and sign this repository. `NVIDIA-NeMo/nemo-platform`
     is listed in its onboarding configuration; confirm the repository's current
     canonical name and allowed base refs with the service owners.
  3. Comment `/nvskills-ci` on a same-repository Customizer skill PR into `main`.
     Follow the central run and its returned `NVSkills CI` status. A successful
     dispatch alone is not a successful evaluation. Request another run after
     changing evaluated content. This workflow does not automatically dispatch
     on PR updates or signing-bot pushes; verify the central run's signature
     verification result, or request a follow-up run explicitly if needed.
  4. Verify all three evaluation tiers, with positive baseline uplift for both
     Claude Code and Codex. Keep the generated `BENCHMARK.md`, `skill-card.md`
     and `skill.oms.sig` beside the canonical plugin `SKILL.md`. The initial
     benchmark file explicitly records that evaluation is pending.
  5. After verification and signing, submit this entry as
     `components.d/nemo-helix.yml` in `NVIDIA/skills`, using the verified source
     repository name (the local remote currently uses `nemo-platform`):

     ```yaml
     name: NeMo Helix
     repo: NVIDIA-NeMo/nemo-platform
     description: Fine-tune open models through NeMo Helix with platform jobs, evaluation and deployment.
     skills:
       - path: plugins/nemo-customizer/src/nemo_customizer/skills/nemo-customizer/
         catalog_dir: nemo-customizer
     ```

     Do not copy the skill into top-level `skills/`; catalog entries can point
     directly at its canonical plugin directory. The current shared
     `NVIDIA/skills` team-request workflow only watches `skills/` and
     `team-skills/`, so this pilot uses a local adapter for the plugin path.
     After catalog sync, verify installation with
     `npx skills add nvidia/skills --skill nemo-customizer`.

  Contracts: [central onboarding](https://github.com/NVIDIA/nvskills-ci/blob/main/docs/team-onboarding.md),
  [central request inputs](https://github.com/NVIDIA/nvskills-ci/blob/main/.github/workflows/nvskills-ci.yml),
  [catalog entry schema](https://github.com/NVIDIA/skills/blob/main/components.d/README.md),
  [evaluation requirements](https://github.com/NVIDIA/skills/blob/main/docs/evaluating-agent-skills.mdx).

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

- `e2e.yaml`
  Independent Kind, Python, auth-IDP runtime, and Studio E2E suites with their
  CPU image build and policy prerequisites. Runs on `main` pushes, labeled
  same-repository PRs, or manual dispatch; not part of the required CI gate.

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

`e2e.yaml` runs the E2E suite on pushes to `main`, explicit PR opt-in, and
`workflow_dispatch`. It does not run for merge groups or use path filters.
Manual runs retain the `image_tag` and `publish` inputs.

For PRs to `main` or `release/*`, maintainers can opt in by adding **`e2e`** to a
**non-draft, same-repository PR**. Forks and missing/invalid PR context cannot opt
in. PR runs are triggered only by label additions, checking current labels and
draft status. New commits, reopening, and readiness changes do not rerun E2E;
remove and reapply `e2e` to test a newer commit or a previously draft PR.
Unlabeling or converting to draft does not cancel an existing run. No
`pull_request_target` or new permissions are used.

The label enables CPU image publication, all Kind suites (including builder
with auth off and on), Authentik/ZITADEL runtime tests and their static
prerequisites, both Python e2e jobs, and Studio
UI E2E tests. Policy-WASM is built within the E2E workflow, and successful
published image prerequisites remain enforced. E2E jobs no longer participate
in the required `CI status` aggregate; auth-IDP static tests remain in CI too.

CI also builds CPU smoke images without E2E opt-in when CPU-smoke, auth-IDP,
customization-auth, or builder-e2e inputs change (including on PRs and merge
groups), or on manual dispatch. This build is included in required `CI status` and the
main-push image failure notification. The CI and E2E build steps are duplicated
so each workflow runs independently; a static test keeps those steps aligned.
Fork PRs build locally without publishing; same-repository PRs, merge groups,
and `main` pushes retain publication. Manual CI publication requires
`publish: true`, and `image_tag` can override the default commit-SHA tag.

Separate garak and Safe Synthesizer image builds are not enabled by the label.
Docker source-image smoke remains manual-only. Manual
`publish: false` still skips tests needing published images. These opt-in checks
are expensive and execute trusted PR code; apply the label deliberately.

Lint, unit tests, integration tests, wheel build/install checks, Helm validation,
and Docker bake-graph validation stay pre-merge. Printing the bake graph does
not build an image. Integration required-check policy is unchanged.

On automatic `main` runs, `Post-merge image status` reports CI's CPU smoke,
garak, and Safe Synthesizer image builds and prerequisites through the existing
`SLACK_ALERTS_WEBHOOK`. It no longer observes E2E results across workflows.
`Post-merge E2E status` independently waits for every E2E job and prerequisite,
then uses the same helper and webhook to send one failure alert per E2E run.
Both notifications include failed/cancelled jobs, the commit, and a diagnostics
link; success/skipped results do not alert. Missing webhooks or failed delivery
fail the notification job rather than silently dropping alerts. PR and manual
E2E runs do not send Slack alerts.

## GitHub Apps

- DCO app
  Developer Certificate of Origin (DCO) checks are handled by the installed DCO
  app, not by a repository workflow. The `dco-war.yaml` workflow exists only
  because the DCO app does not currently understand merge queue checks. The
  ruleset allows any source named `DCO` to satisfy the required check so merge
  queue entries can pass. When the DCO app supports merge queues, change the
  ruleset to require the app-owned DCO check instead and remove the workaround.
