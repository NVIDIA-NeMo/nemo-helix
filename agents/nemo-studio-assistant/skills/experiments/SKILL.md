---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: experiments
description: NeMo Helix experiments playbook for creating, listing, updating, and deleting experiments and the evaluations inside them through the platform SDK. Use when the user mentions experiments, comparing evaluation runs, or the Experiments page in Studio.
metadata:
  author: NeMo Helix Team <nemo-helix@nvidia.com>
---
# Experiments

An **experiment** groups **evaluations** (runs) so they can be compared side by side in Studio. Experiments
are not the same as the evaluator's metrics and metric jobs: use `experiments` and `experiments.evaluations`
here, and `evaluation.metrics` / `evaluation.metric_jobs` only for metric runs.

- Use `nemo_api` with `workspace="<active request workspace>"` on every call.
- Resources and actions:
  - `experiments`: `list`, `retrieve`, `create`, `update`, `delete`.
  - `experiments.evaluations`: `list`, `retrieve`, `create`, `update`. There is no delete.
- Reads (`list`, `retrieve`) run immediately. `create`, `update`, and `delete` ask the user for approval, so
  pass the Studio session id from the user context on those calls and never invent one.

## Create an experiment, then an evaluation in it

1. `experiments` / `create` with `name` (required) and optional `description`, `default_sort`,
   `show_evaluations_over_time`, `is_favorite`, `metadata`, and `summary`. To reuse an experiment that may
   already exist, pass `exist_ok: true`; it returns the existing one instead of failing on a name clash.
2. Keep the `id` from the response. An evaluation belongs to experiments by **id**, not by name.
3. `experiments.evaluations` / `create` with `name`, `dataset_name` (both required) and
   `experiment_ids: ["<experiment id>"]`. Optional: `dataset_version`, `description`, `source_link`,
   `metadata`, `parent_evaluation_id`, `status`.
4. Read the result back with `retrieve` (pass the evaluation `name`) to verify it before reporting success.

Only the documented fields are accepted; an unknown field is rejected, so do not add extras. If the user
has not said which dataset name an evaluation should record, ask them instead of guessing.

## Find and change things

- `experiments` / `list` to find an experiment; `retrieve` takes the experiment `name`.
- `experiments` / `update` replaces the experiment, so send the full set of fields you want to keep, not just
  the one that changed. Retrieve it first and carry its current values forward.
- Confirm the exact experiment name with the user before any `delete`.

## Link the user to the result

Call `studio_link` after creating or finding something:

- An experiment: `destination="experiment_group"` with `name` set to the experiment name.
- An evaluation inside it: `destination="experiment_detail"` with `name` set to the experiment name and
  `experiment_name` set to the evaluation name.
- The Experiments list: `destination="experiment"`.

## Limits

This covers the experiment and evaluation records only. Uploading traces or scores into an experiment is a
separate ingest workflow that this assistant cannot do; say so plainly instead of attempting it, and never
claim results were uploaded.
