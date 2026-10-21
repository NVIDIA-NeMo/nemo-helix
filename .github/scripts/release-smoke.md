<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Post-release smoke checks

`release-smoke.yaml` tests exact published artifacts on macOS (Intel/Colima) and
Kubernetes (Kind). Each job checks startup, workspace CRUD, rendered Studio,
real inference, and the sample agent bundled in that release. These are CPU
checks; they do not validate GPU workloads or replace the QA suite.

## Configuration

- Allow this repository to access organization secrets `NVIDIA_INFERENCE_URL`
  and `NVIDIA_INFERENCE_KEY`. The endpoint must be reachable from GitHub-hosted
  runners and return the configured model from its OpenAI-compatible models API.
- Set repository variable `SMOKE_INFERENCE_MODEL` to a tool-calling chat model ID
  served by that endpoint. Initial selection: `aws-anthropic-bedrock-claude-haiku-5-5`.
- Existing `NGC_REGISTRY_READ_TOKEN` and the workflow's `GITHUB_TOKEN` authenticate
  published image/chart pulls. `SLACK_RELEASE_WEBHOOK` is used only when requested.

## Manual validation and rollout

Run **Release smoke checks** with the full chart version and **Send notifications
to Slack** unchecked. The version resolves to the corresponding wheel and image
tag; nightly versions must include the 14-digit timestamp.

```sh
gh workflow run release-smoke.yaml -f release-version=0.8.0 -f send-notifications=false
gh workflow run release-smoke.yaml -f release-version=0.8.1-nightly-YYYYMMDDHHMMSS -f send-notifications=false
```

Replace the nightly placeholder with a full published release. Verify both jobs
pass, both diagnostic artifacts are uploaded, and resources are cleaned up.
On a validation branch, temporarily use an unavailable model to check that the
result is red, the failing stage and logs are retained, and cleanup still runs.
Notification payloads and suppression are covered by mocked unit tests; do not
send test failures to the release channel.

Only after those runs pass, set repository variable `RELEASE_SMOKE_ENABLED=true`.
This enables the call after `notify-end` for full, successfully published stable
and nightly releases. A skipped or failed Slack announcement does not block it.
While disabled, releases do not announce pending smoke checks. Partial releases,
dry runs and `act` runs do not schedule them.

With notifications enabled, a scheduled release announcement says **Smoke checks
pending**, and one final message reports both jobs as passed, failed, or
cancelled/incomplete. Manual runs default to no Slack messages. Failed or
incomplete tests still fail CI. As with other Actions jobs, force cancellation or
runner loss can prevent `always()` cleanup/reporting from finishing.

The Mac check first installs `[all]` and tests the platform. It then stops it,
reinstalls the **same version** with `[all,nemo-agents-plugin-deepagents]`, and tests
the sample in subprocess mode. This explicitly records the extra dependency
needed by the sample; it does not change packaging. Kubernetes explicitly uses
the released `nhx-tasks` image and `k8s` mode. Removal/replacement of the bundled
sample requires updating the check, not silently skipping it.

Helpers execute outside the source environment. Only redacted logs, a screenshot
and `result.json` are uploaded; installation state, credentials and databases are
not artifacts. Each job has a 45-minute limit; inference/agent requests are bounded
to two minutes. Readiness polling does not retry failed inference or agent requests.

## Local helper checks

```sh
make check-github-scripts
uv run --isolated --no-project --with pytest==8.4.2 pytest -c /dev/null --confcutdir=.github/scripts/tests -p no:cacheprovider .github/scripts/tests/test_release_smoke.py
flox activate --dir tools/actionlint -- actionlint
```
