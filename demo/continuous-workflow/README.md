<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Layered continuous workflow

Use the [shared workflow CLI](../README.md) to create, trigger, watch, download
and cancel named runs:

```bash
uv run python -m demo.workflows create layers my-dag --watch
uv run python -m demo.workflows trigger my-dag
uv run python -m demo.workflows trigger my-dag --step worker-03
```

The saved `job.json` template defines 19 nodes over five layers:
`prepare → 10 workers → 5 reducers → 2 validators → summarize`.
Tasks echo simulated progress every few seconds. Fileset staging, publication,
commit markers and controller scheduling use the normal platform paths.

A commit to either the run's input or configuration Fileset triggers `prepare`
(OR semantics). Workers require successful preparation and committed
configuration (AND semantics). Preparation publishes a manifest before its
success releases workers. The final Step publishes `report/report.txt` in the
output Fileset. Output commits do not trigger another pass.

Workers use typed Step-success readiness, which accepts a prior success.
Reducers, validators and the final join keep string dependencies so they wait
for dirty parents in the current pass. Each reducer joins its own pair of
workers; there is no extra global barrier between layers. Manual invalidation
reruns only the affected chain, reusing other nodes' successful results.

The shared CLI binds the template's symbolic Fileset IDs and sets the Job's
requested name, image and execution profile. Existing Attempts retain their
captured definitions, so create a fresh named Job after editing the template.

For unified TUI logs, press Tab, then `a`, then `f`. The earlier missing-launcher
incident is recorded in [the local investigation](logging-investigation.md).
