<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

Run a dataset-driven offline evaluation against the local NeMo Helix service
at `http://127.0.0.1:8080` in workspace `default`. Submit the two-row exact-match
evaluation described by `/workspace/exact_match_metric.json`. Use its rows,
metric and templates as supplied. Wait for the actual platform job to complete.

Save the job's aggregate scores from the service to
`/workspace/aggregate-scores.json`. Write `/workspace/report.json` as a JSON
object with the fields `job_name`, `status`, `metric` (the aggregate's score
name), and `mean` (a number). In your final answer, summarize the outcome with
the job name and metric mean.
