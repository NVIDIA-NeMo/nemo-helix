<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Releasing NeMo Platform

[`release.yaml`](.github/workflows/release.yaml) is the single release workflow
for NeMo Platform. It handles scheduled nightlies and manually dispatched
nightly or stable releases. The release catalog is deliberately defined in that
workflow so contributors can see and validate every releasable artifact in one
place.

Anyone with permission to run repository workflows can start a release. A
stable release requires a specific source commit and version; a nightly can use
the default branch head.

## Before starting a stable release

Choose the exact 40-character commit SHA and the `MAJOR.MINOR.PATCH` version
to release. The source must contain the desired generated SDKs. If the API
surface changed since the last SDK update, update the SDKs before releasing:

```bash
make update-sdk
```

This regenerates the OpenAPI specifications and synchronizes the SDKs. The
specifications intentionally retain `info.version: 0.0.0`; do not copy the
release version into them.

## Release catalog

The catalog in [`release.yaml`](.github/workflows/release.yaml) is the source
of truth for selection. Do not add a separate release manifest or configuration
file. When changing the catalog, also update the matching `workflow_dispatch`
input description in that workflow.

| Type | IDs |
| --- | --- |
| Wheels | `nemo-platform`, `nemo-platform-plugin` |
| Containers | `nmp-api`, `nmp-cpu-tasks`, `nmp-gym-tasks`, `nmp-gym-host`, `nmp-customizer-tasks`, `nmp-automodel-training`, `nmp-unsloth-training`, `nmp-rl-training`, `auditor-tasks`, `safe-synthesizer-tasks` |
| Helm chart | `nemo-platform` |

For every selected wheel, the workflow checks that its package configuration
declares the expected project name. For every selected container, it checks the
Docker Bake target and the corresponding
`.github/assets/ngc/containers/<id>.md` overview file. These checks happen
before any external release work is dispatched.

## Starting a release

Open the [Release workflow](https://github.com/NVIDIA-NeMo/nemo-platform/actions/workflows/release.yaml)
and select **Run workflow**. The form shows the allowed custom artifact IDs.

| Input | Use |
| --- | --- |
| `release-type` | `nightly` by default. Select `stable` for a versioned release. Every successful non-dry-run stable release creates or confirms its Git tag and GitHub Release. |
| `source-sha` | Required for stable releases. Optional for nightlies; a normal nightly with no SHA uses the current default-branch head. A dry-run nightly with no SHA uses the workflow commit so a branch can be validated. |
| `version` | Required for stable releases. Enter the `MAJOR.MINOR.PATCH` release version. |
| `release-scope` | `all` by default. Select `wheels`, `containers`, `helm`, or `custom` for a subset. Scope selects artifacts only; it does not control the stable tag or GitHub Release. |
| `wheel-ids`, `container-ids` | Comma-separated IDs used only with `release-scope: custom`. Each ID must be in the catalog above; duplicates and empty entries fail validation. |
| `include-helm` | Includes the Helm chart in a custom release. |
| `helm-version` | Optional exact SemVer Helm chart version for stable Helm-only releases. The stable release label still comes from `version`. |
| `update-ngc-metadata` | Also runs the reusable NGC metadata workflow for `nemo-platform` and `nemo-platform-dev`. It checks out the workflow ref, normally `main`. |
| `send-notifications` | Sends Slack start and final-status notifications. Defaults to `true`. |
| `dry-run` | Validates the selected source and packages the selected Helm chart, but does not publish, dispatch external work, poll, create a GitHub release, or signal deployment. The start notification intentionally still runs when notifications are enabled. |

Examples:

| Goal | Inputs |
| --- | --- |
| Scheduled-style nightly | Leave `release-type` as `nightly` and use the default `all` scope. |
| Stable full release | `release-type: stable`, `source-sha: <40-character SHA>`, `version: <MAJOR.MINOR.PATCH>`, `release-scope: all`. |
| One container | `release-scope: custom`, `container-ids: nmp-customizer-tasks`. |
| Stable one-container release or retry | `release-type: stable`, `source-sha: <40-character SHA>`, `version: <MAJOR.MINOR.PATCH>`, `release-scope: custom`, `container-ids: nmp-customizer-tasks`. |
| Helm-only validation | `release-scope: helm`, `dry-run: true`. |
| Stable Helm-only chart override | `release-type: stable`, `source-sha: <40-character SHA>`, `version: 0.1.0`, `release-scope: helm`, `helm-version: 0.1.0+helmfix1`. |

Nightlies also run automatically Monday through Friday at 8:00 PM
America/Los_Angeles.

## What the workflow does

1. Resolves the source, release label, selected artifacts, wheel version, and
   Helm chart version. Stable versions use the supplied release version.
   Nightly labels use `nightly-<UTC timestamp>` and the wheel version is
   resolved by `.github/scripts/stamp_sdk_version.py`.
2. Checks out the selected source and validates the selected wheel paths,
   Docker Bake targets, and NGC overview files.
3. Optionally synchronizes NGC metadata, when requested on a non-dry-run.
4. Dispatches wheel, container, and stable-release registration work to the
   configured internal release repository. The selected source SHA, release
   type, version, and selected IDs are passed with the dispatch.
5. Packages the Helm chart with the planned chart version. A nightly chart uses
   the latest release or RC Git tag core reachable from the selected source with
   `-nightly-<UTC timestamp>` appended, falling back to the `Chart.yaml`
   version if no such tag exists. A stable chart uses the stable release version
   unless `helm-version` is set for a stable Helm-only release.
6. Waits for every selected final artifact to become available at its configured
   publication destination before continuing. The polling job times out after
   four hours and sends a Slack alert after two hours if it is still waiting.
7. For every non-dry-run stable release, creates or confirms the Git tag and
   GitHub release at the selected SHA once the selected artifacts are
   available. This is independent of `release-scope`, so subset and custom runs
   can finish or retry a release. A rerun is a no-op when the tag and release
   already match, creates the missing release when only the matching tag
   exists, and fails when the version tag points to a different SHA or a draft
   release exists. GitHub generates the release notes from the previous numeric
   SemVer tag.
8. After polling succeeds, releases that include the Helm chart dispatch a
   deployment signal to the configured internal release repository. The
   downstream workflow creates a pending GitHub Deployment, and the deployment
   controller completes it independently. Releases without Helm skip this step.

## Publication destinations

| Artifact | Nightly | Stable |
| --- | --- | --- |
| Wheels | [pypi.nvidia.com](https://pypi.nvidia.com) | [PyPI](https://pypi.org) |
| Containers | `ghcr.io/nvidia-nemo/nemo-platform/<id>:nightly-...` | `nvcr.io/nvidia/nemo-platform/<id>:<version>` and the public NGC catalog |
| Helm chart | OCI chart at `oci://ghcr.io/nvidia-nemo/nemo-platform` | Initially staged at `0921617854601259/nemo-platform`, then promoted to the public [NGC Helm repository](https://helm.ngc.nvidia.com/nvidia/nemo-platform) |

The stable Helm promotion is external to this workflow. The workflow polls the
public NGC Helm repository, not the internal staging endpoint, before it marks
the release complete.

## Notifications

With `send-notifications: true`:

- A start message is sent to `SLACK_ALERTS_WEBHOOK`, including the selected
  artifacts and source commit. This is also sent for dry-runs so the webhook can
  be tested.
- If final artifact polling exceeds two hours, a delay alert is sent to
  `SLACK_ALERTS_WEBHOOK`.
- A successful non-dry-run release sends its completion message to
  `SLACK_RELEASE_WEBHOOK`. A failed or cancelled release sends its final status
  to `SLACK_ALERTS_WEBHOOK`.

Dry-runs do not poll or send the delayed or final notification.

## Required secrets

| Secret | Used for |
| --- | --- |
| `CI_DISPATCH_REPO` | `owner/repo` of the internal release repository that receives release dispatches. |
| `CI_DISPATCH_TOKEN` | Authenticating those cross-repository dispatches. |
| `AIRE_NVCR_GITHUB` | Staging stable Helm charts in NGC. |
| `AIRE_NGC_GITHUB_PLATFORM_RW` | Optional NGC metadata synchronization. |
| `SLACK_ALERTS_WEBHOOK` | Release starts, delay alerts, and failed final statuses. |
| `SLACK_RELEASE_WEBHOOK` | Successful final release status. |

## Verifying a completed release

The workflow summary records the selected wheels and containers. After a live
release completes, check the selected artifacts at their destination above.
For a stable release, also verify that the GitHub tag and generated GitHub
release point to the requested source SHA.

For a stable wheel release, a quick client check is:

```bash
uv tool upgrade nemo-platform
nemo --version
```
