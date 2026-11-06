<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Job stress

This tool submits jobs to a NeMo Helix platform that is already running and writes a report from the records those jobs leave behind. It does not start a cluster, install packages, or build an image. Run the commands from the repository root.

The shipped sweep is `sweep.yaml`. Each step is one measurement. `count` is the number of jobs started in each workspace. A step with more than one workspace starts that count in every workspace, round-robin, so the job total is `count` times the workspace count. The report keeps a separate section for exit, idle, barrier, and each hammer service, because those sections do not share a metric. A section with no completed steps says so. Raw operation logs stay in the run records.

A result from the docker backend describes that backend. It is not a Kubernetes capacity number, and the report is not a pass/fail gate.

## What each test measures

### Exit

Exit runs `/bin/bash -c` with `exit 0`. The job does not download the worker. This measures how quickly the platform accepts a job and reports it finished when the task itself does no work.

The sweep grows the count through 1, 10, 25, 50, 100, and 500, all in one workspace. Throughput is completed jobs divided by the time from the first submit until the controller last saw a job finished, in jobs per second. That window includes submission, queueing, the task, and the controller's polling. Scheduling counts creates the platform refused, plus accepted jobs that reached error without becoming active. These jobs exit immediately, so they do not hold executor slots. The number is an admission and lifecycle rate.

### Barrier

Barrier jobs download the worker, write a ready file under `barrier/<step>/ready/<job>`, and wait. The controller writes `barrier/<step>/release` into each workspace fileset only after every job in the cohort has written ready. A job that reads the release writes `barrier` and exits 0. A job that reaches `timeout_seconds` first writes `timeout` and exits 1. If any job is still missing at the timeout, the controller does not write the release. Jobs that already reached ready then exit on their own timeout.

This measures whether a cohort can be active together, and how long a synchronized exit takes after the release.

- Ready is how many jobs wrote a ready file before the timeout.
- Peak running is the largest number of these jobs the controller saw in the active status while it waited for ready files.
- Submit is the seconds from the first create until the last create returned. The controller issues creates one at a time.
- Ready time runs from the start of submit until every job is ready, or until the timeout. It includes submit.
- Release to finalized is the seconds from the release write until every job was terminal and deleted. The cell says `not released` when the cohort missed the ready deadline.
- Barrier, timeout, and unknown count how the jobs left: the release was seen, the timeout elapsed, or the job ended with neither reason recorded.

Steps with 10 jobs per workspace use a 180 second timeout. Steps with 100 jobs per workspace use 600 seconds. Steps whose id ends in `-w2` use two workspaces, so `barrier-10-w2` is 20 jobs and `barrier-100-w2` is 200. When a step uses more than one workspace, the report repeats accepted, ready, peak, barrier, timeout, and unknown per workspace. Similar counts across those workspaces are what fair sharing would look like.

### Service hammers

Files, secrets, and auth hammers download the worker, then call that one service for `duration_seconds` at `rate_per_second`. The shipped steps use 60 seconds and 5 calls per second. Files and auth grow through 1, 10, and 25 jobs. Secrets also has 100 and 300. Each job keeps one call in flight and sleeps only when the call returns sooner than `1 / rate_per_second`. The worker round-robins that service's operations.

This measures how that service responds as the number of concurrent jobs grows.

Throughput in the report is the mean rate each worker reported, in operations per second. It is the rate inside one job. Multiply by the count for a cluster total only when those jobs actually overlap. The operation table splits the same mean by operation, with one row per step, so a slow operation shows up next to the others. Zip fetch is the mean seconds spent downloading the worker before the calls, and it is separate from the service rate. Missing counts accepted jobs whose `service-stress` result was not copied. Partial is `yes` when any job is missing that result or its operation log.

Auth steps run only when discovery says authentication is enabled. When it is off, `run` prints their ids, skips them, and does not write completion markers. `run --step` on one of those steps fails before any job is submitted.

### Idle

Idle runs `/bin/bash -c` with `sleep` for the step duration and does not download the worker. The shipped sweep has no idle steps. A report that included one would use the exit columns plus duration. It does not sample peak active.

## How to run

The platform has to be listening, and `nemo` needs a saved context for it. `run` requires `--context`. With no arguments the tool prints help and does not use the CLI's current context. `-h` is the same as `--help` on every command.

`--job-profile` defaults to `default`. Pass a profile name, or `provider/profile` as printed by a profile error, such as `cpu/gpu`. Startup uses that profile when its backend is `kubernetes_job` or `docker` and its task image is unset or contains `nhx-tasks`. A profile that is also registered as `subprocess` is rejected, because a CPU job with that name runs on the host. `nemo services run` registers `default` that way, so a local Docker test passes `cpu/gpu`. The error lists the profiles that can be used.

`image` in the sweep is sent with every job. `--image` overrides it for one run. Omit both and the job uses the profile's task image, which is unset on a local `cpu/gpu` profile, and the run stops before submitting. Hammer and barrier jobs need the SDK the worker imports. Exit and idle need `/bin/bash` in that image.

```bash
uv run python tools/stress-test/main.py list
uv run python tools/stress-test/main.py run --context local --job-profile cpu/gpu --step exit-01
uv run python tools/stress-test/main.py run --context local --job-profile cpu/gpu
uv run python tools/stress-test/main.py report --context local --job-profile cpu/gpu
```

`exit-01` is the first live check: one job that exits immediately. A bare `run` then continues through the sweep.

`list` prints every sweep file in this directory and the steps in each file. It does not contact the platform. `list --config path/to/sweep.yaml` prints one file.

`run` without `--step` skips a step whose completion marker still matches the loaded count, duration, rate, timeout, and workspace count, then continues in file order. `--step` must be one of those ids. It deletes that step's records and submits it again. `--force` reruns every step in the sweep, including finished ones. `--workspaces` replaces every step's workspace count for that invocation. One workspace is named `stress-<run_key>`. More than one are named `stress-<run_key>-1`, `stress-<run_key>-2`, and so on.

`run` writes a short progress log to stderr. It names the steps it will run, then each step logs `preparing step N/M <id> ...`, `submitting <count> job(s)`, `started <job>`, `waiting for <count> job(s)`, `stopped <job>` or `error <job>: <reason>`, and a one-line step summary. `report: <path>` is logged when the report is written. `--debug` adds each HTTP request, each status change, and the log tail. Those lines stay in the job record either way.

The marker is written after the jobs are terminal and deleted and a follow-up list of those names is empty. Hammer and barrier jobs also copy a `service-stress` result when the platform has one. A missing result stays in the report. If jobs are still listed when the wait ends, `run` stops and later steps are not submitted. A terminal job error does not by itself stop the sweep.

`run` writes the report when the step or sweep stops, including when a step stops the sweep, and copies it to the run fileset. `report` rebuilds that same report from the local records. It reads the local worker bundle and does not rebuild it from the cluster. After the local files exist it copies them to the run fileset.

Reports are written to `tools/stress-test/output/<run_key>/`. Raw records stay in `tools/stress-test/runs/<run_key>/`. Both directories are gitignored. Pod sizes live in `sweep.yaml`. The report copies the inputs that stay the same across a section, including CPU and memory, into that section's parameter table. Columns that change stay in the comparison table.

`publish` uploads the worker zip to the run fileset. `run` uploads it too when a kept step is a hammer or a barrier, and it skips the upload when the fileset already has the same digest. Publish a new zip when the worker changes. Do not rebuild the task image for a worker-only change.

`clean-jobs` cancels and deletes every job for the run key. Losing an unpublished artifact during that reset is expected. Omit `--workspaces` to clean every workspace the sweep uses. `clean-results` deletes this run's fileset objects, the local bundle, and the report directory. It leaves the fileset in place.
