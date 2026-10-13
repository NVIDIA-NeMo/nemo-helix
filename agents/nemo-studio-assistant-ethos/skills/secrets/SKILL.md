---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: secrets
description: NeMo Helix secret CRUD lifecycle through the platform SDK.
---
# Secret tasks

- Use `nemo_api` with resource `secrets`, passing
  `workspace="<active request workspace>"` on every call.
- Use `create`, `retrieve`, `list`, `update`, and `delete` actions as needed,
  passing only secret references (`<workspace>/<name>`), names, and metadata in
  JSON `params`, never resolved secret values. Because approval inputs and tool
  results can appear in traces, do not print or echo secret values during any
  lifecycle or verification step.
- If create or update requires a real credential value, have the user provision
  it through an approved secure secret-entry path and continue with its
  reference. Use only explicitly non-sensitive synthetic values for an automated
  CRUD lifecycle.
- Always finish the whole lifecycle: create temporary secret, verify/list/update/delete it, then create the final verification secret.
