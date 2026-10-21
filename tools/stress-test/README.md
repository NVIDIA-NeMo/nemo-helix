<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Kubernetes job stress

Point this tool at a NeMo Helix context that already has authentication enabled and a Kubernetes jobs profile. It does not start a cluster, install packages, or build an image. The worker uses the SDK already present in the profile's `nhx-tasks` image. Publish a new zip when the worker changes; do not rebuild the image.

Run the commands from the repository root:

```bash
uv run python tools/stress-test/main.py publish --context my-context --job-profile default
uv run python tools/stress-test/main.py run --context my-context --job-profile default
uv run python tools/stress-test/main.py report --context my-context
```

`sweep.yaml` is the whole measurement list. Counts, durations, and rates are explicit, so a non-linear sweep is a config edit. `run` without `--step` skips steps whose completion marker still matches the loaded measurement and continues with the rest in file order. `--step` must be one of those ids. It replaces that step's records and runs it again.

A step is finished only after its jobs are terminal, their `service-stress` results have been copied, the jobs have been deleted, and a follow-up list of those names is empty. If that does not happen, `run` stops. Later steps are not submitted. A terminal job error or a missing artifact stays in the report and does not by itself stop the sweep.

Reports are written to `tools/stress-test/output/<run_key>/`. Raw records stay in `tools/stress-test/runs/<run_key>/`. Both directories are gitignored. `report` reads the local bundle and does not rebuild it from the cluster. After the local files exist it copies them to the run fileset.

`clean-jobs` is a development reset. It cancels and deletes every job for the run key. Losing an unpublished artifact during that reset is expected. `clean-results` deletes this run's fileset objects, the local bundle, and the report directory. It leaves the fileset in place.

The first live check is one immediate-exit job: `uv run python tools/stress-test/main.py run --context my-context --job-profile default --step exit-01`. The report should show that job. A capacity number is not an acceptance criterion.
