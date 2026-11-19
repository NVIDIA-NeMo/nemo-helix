<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

Run the stored agent-smoke Taskset against the local NeMo Helix service in
workspace `default`. Start with `/workspace/agent-evaluate-spec.json`, which
selects the Taskset and its live loopback agent. Apply any installed
agent-evaluation run-naming convention to `labels.run_alias`. Submit the spec
with `nemo evals agent-evaluate --workspace default --spec-file` and wait for
the job to complete.

Download the job's `agent-eval-results` and use its `summary.json` to write
`/workspace/report.json` with `job_name`, `run_alias`, `status`, `metric`, and
numeric `mean`. In your final answer, state the job name, run alias, and metric
mean.
