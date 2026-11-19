<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Missing demo logs

**TL;DR:** The local Docker tasks ran without jobs-launcher because its configured
binary was absent. Building the Linux ARM64 launcher restored forwarding for new
tasks. Logs from prepare and worker nodes were verified through the Jobs API.

Investigated 2026-11-19. Local repair verified; no product source change or push.

## Original problem

The user saw workflow nodes execute but no logs in watch. Expected behavior was
live combined output from the demo's echo/sleep tasks.

## Investigation performed

Confirmed the API process on port 8080 used the DAG worktree. Queried both demo
Jobs through the typed SDK: each was active, and each returned zero log records.
Checked the configured local launcher path and traced Docker scheduling's
launcher lookup and entrypoint selection. Built the binary for the ARM64 task
image, committed the layered demo's input once, and queried logs again.

## Results

The binary was missing at the configured relative path. Docker scheduling permits
a missing launcher for tasks without Fileset transfers, warning and using the
original command instead. This bypasses launcher log forwarding.

After the build, newly scheduled containers had entrypoint
`/jobs-launcher run --`. The API returned four prepare messages on the first
successful poll. A later sample contained 18 records, including six worker
records. This verifies task output reached the API used by watch. The user's
interactive terminal display was not inspected.

## Resolution

Built the existing launcher from this checkout for Linux ARM64 and triggered one
new pass of `continuous-demo-layers`. The running service discovered the binary
without restart. Older task output was not recovered. Keep the launcher build
step in the demo startup instructions.

## Evidence appendix

- Worktree: `/Users/rsadler/wt/dag-workflows-rsadler`, branch `dag-workflows/rsadler`.
- Base HEAD: `09eaee10f3`, with local worktree changes. Remote refs were not refreshed.
- Launcher: `services/core/jobs/jobs-launcher/jobs-launcher`, built using
  `bash build-manual.sh linux arm64`.
- Source: `DockerJobBackend.get_jobs_launcher_binary` and
  `_run_container_in_thread` in the Jobs controller's `backends/docker.py`.
- API checks: typed `JobsClient.get_job_status` and `list_job_logs`, workspace
  `default`, Jobs `continuous-demo` and `continuous-demo-layers`.
- Verification commit marker: `b71459bf-0ca7-4de7-91f9-247db007f4d6`.
- No Docker test suite was run. Verification used the already-running demo.
