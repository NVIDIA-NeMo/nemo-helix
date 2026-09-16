---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: auditor
description: NeMo Helix auditor playbook for audit target and config CRUD through the platform SDK. Use when the task involves audit targets, audit configs, or probes.
---
Auditor tasks

- Use `nemo_api` with `audit.targets` for target CRUD and `audit.configs`
  for config CRUD.
- Use the standard SDK actions: `create`, `list`, `retrieve`, `update`, and
  `delete`.
- Pass the target or config fields as compact JSON in `params`.
- For config create, prefer minimal valid JSON:
  - `plugins`: `{"probe_spec":"dan.AutoDANCached"}` (or requested probe)
  - `reporting`: `{}`
  - `run`: `{}`
  - `system`: `{"lite": true}`
- Follow full lifecycle: create temp resource, verify/list/update/delete, then create final verification resource.
- Retrieve or list the final audit resource and compare every required target or
  config field before reporting success.
