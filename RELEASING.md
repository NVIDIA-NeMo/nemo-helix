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
the selected release branch commit.

## Maintenance releases

Run this workflow from the matching `release/0.5` or `release/0.6` branch.
Set the repository variable `CI_LEGACY_DISPATCH_REF` to the maintenance tooling
branch before live runs; keep the existing dispatch destination and token secrets.
The downstream tooling must support `release-branch-payload` before enabling
these branch-push requests.

Pushes request container builds tagged with the source SHA and readiness for the
next patch version. Before a stable release, check that
`release/container-readiness/<version>` is successful for the selected SHA.
The release workflow does not enforce that status automatically. It publishes
the existing containers from `nemo-platform-dev` under the patch version,
while still building wheels and packaging Helm. Nightlies build new artifacts.
Maintenance releases do not request deployment to shared environments.

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
| Containers | `nmp-api`, `nmp-cpu-tasks`, `nmp-gym-tasks`, `nmp-customizer-tasks`, `nmp-automodel-training`, `nmp-unsloth-training`, `nmp-rl-training`, `auditor-tasks`, `safe-synthesizer-tasks` |
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
| `release-type` | `nightly` by default. Select `stable` for a full release. |
| `source-sha` | Required for stable releases. Optional for nightlies; a nightly launched from a release branch with no SHA uses that workflow commit. A dry-run nightly with no SHA uses the workflow commit so a branch can be validated. |
| `version` | Required for stable releases. Enter the `MAJOR.MINOR.PATCH` release version. |
| `release-scope` | `all` by default. Select `wheels`, `containers`, `helm`, or `custom` for a subset. |
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
4. Dispatches wheel builds, nightly container builds, and stable-release registration to the
   configured internal release repository. The selected source SHA, release
   type, version, and selected IDs are passed with the dispatch.
5. Packages the Helm chart with the planned chart version. A nightly chart uses
   the latest release or RC Git tag core reachable from the selected source with
   `-nightly-<UTC timestamp>` appended, falling back to the `Chart.yaml`
   version if no such tag exists. A stable chart uses the stable release version
   unless `helm-version` is set for a stable Helm-only release.
6. Waits for every selected final artifact that is published publicly to become
   available before continuing. Nightly wheels are staged but not published, so
   they are not polled. The polling job times out after four hours and sends
   a Slack alert after two hours if it is still waiting.
7. For a non-dry-run stable release with `release-scope: all`, creates the
   GitHub release and tag at the selected SHA. GitHub generates the release
   notes from the previous numeric SemVer tag. Subset releases do not create a
   GitHub release or tag.
8. Deployment signaling is disabled for maintenance releases.

## Publication destinations

| Artifact | Nightly | Stable |
| --- | --- | --- |
| Wheels | Staged only | [PyPI](https://pypi.org) |
| Containers | `ghcr.io/nvidia-nemo/nemo-platform/<id>:nightly-...` | `nvcr.io/nvidia/nemo-platform/<id>:<version>` and the public NGC catalog |
| Helm chart | OCI chart at `oci://ghcr.io/nvidia-nemo/nemo-platform` | Initially staged at `0921617854601259/nemo-platform-dev`, then promoted to the public [NGC Helm repository](https://helm.ngc.nvidia.com/nvidia/nemo-platform) |

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
For a full stable release, also verify that the GitHub tag and generated GitHub
release point to the requested source SHA.

For a stable wheel release, a quick client check is:

```bash
uv tool upgrade nemo-platform
nemo --version
```
