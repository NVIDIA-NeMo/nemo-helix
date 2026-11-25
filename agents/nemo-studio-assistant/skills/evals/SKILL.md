---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: evals
description: "Explain evaluation in NeMo Helix and run metric evaluations with nemo_api(resource='evaluation.metrics'|'evaluation.metric_jobs'). Use when the user asks how to measure model or agent quality, which metrics exist, or wants to create a metric or run an evaluation job."
metadata:
  author: NeMo Helix Team <nemo-helix@nvidia.com>
---
# Evaluation

Evaluation scores model or agent output against a dataset with a metric, for
example exact string match or an LLM judge. Metrics run as an asynchronous job
over a fileset.

## Explaining

- For an agent, point to `agent_evaluations`. For a model, point to
  `evaluation`, `evaluation_metrics`, or `evaluation_benchmarks`.
- Check what already exists with
  `nemo_api(resource="evaluation.metrics", action="list", workspace="<active request workspace>")`.
- A typical path is: put a dataset in a fileset, choose or create a metric,
  then run a metric job over the full dataset.
- When the user needs to pick an evaluation config file, use
  `select_eval_config`.

## Acting on request

- Use `nemo_api` with `evaluation.metrics` for metric operations and
  `evaluation.metric_jobs` for jobs, passing
  `workspace="<active request workspace>"` on every call.
- Keep inline JSON data compact and valid.
- After starting a job, call `job_progress` with its name so Studio shows a
  progress card, and use `check_status` when the user asks how it is going.
- A job that stays in `created` may mean no job controller is running; say so
  rather than retrying.
