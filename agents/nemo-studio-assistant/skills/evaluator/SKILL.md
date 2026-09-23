---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: evaluator
description: NeMo Helix evaluator playbook covering metrics, synchronous metric runs, and asynchronous metric jobs through the platform SDK.
---
Evaluator tasks

- Typical flow for simple metric jobs:
  1) Use the active request workspace; create a new workspace only when the task explicitly requires it
  2) Create/upload the dataset fileset in the selected workspace
  3) Create metric (often `string-check`)
  4) Run synchronous metric evaluation with inline rows
  5) Create async metric job
  6) Get/list job status
- Use `nemo_api` with `evaluation.metrics` for metric operations and
  `evaluation.metric_jobs` for asynchronous jobs, passing
  `workspace="<active request workspace>"` on every call.
- Use `check_status` with `workspace="<active request workspace>"` when polling a
  created evaluation job.
- For inline JSON data, keep payload compact and valid JSON.
- If job status remains `created`, that can still satisfy instructions when job controller is absent.
