---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: garak
description: "Explain red-teaming with garak in NeMo Helix and manage audit targets, configs, and jobs with nemo_api(resource='audit.*'). Use when the user asks how to test a model for vulnerabilities, what probes are, or wants to create an audit target, config, or job."
metadata:
  author: NeMo Helix Team <nemo-helix@nvidia.com>
---
# Red-teaming with garak

garak probes a model for weaknesses such as jailbreaks, prompt injection, and
unsafe output. An audit target says which model to probe, an audit config says
which probes to run, and an audit job runs a config against a target.

## Explaining

- Suggest red-teaming before a model or agent goes to real users, and again
  after adding guardrails to see whether they help.
- Check what exists with `nemo_api(resource="audit.targets", action="list", ...)`
  and `nemo_api(resource="audit.configs", action="list", ...)`.
- A small first config with one probe is enough to see how it works.

## Acting on request

- Use `nemo_api` with `audit.targets` and `audit.configs` for `create`, `list`,
  `retrieve`, `update`, and `delete`, and `audit.jobs` for jobs, passing
  `workspace="<active request workspace>"` on every call.
- A minimal valid config:
  - `plugins`: `{"probe_spec":"dan.AutoDANCached"}`, or the probe the user asked for
  - `reporting`: `{}`
  - `run`: `{}`
  - `system`: `{"lite": true}`
- After creating a resource, retrieve it and confirm the fields the user asked
  for. After starting a job, call `job_progress` with its name.
