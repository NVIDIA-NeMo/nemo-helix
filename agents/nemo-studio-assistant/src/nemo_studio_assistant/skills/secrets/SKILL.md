---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: secrets
description: NeMo Helix secret CRUD lifecycle through the platform SDK.
---
Secret tasks

- Use `nemo_api` with resource `secrets`.
- Use `create`, `retrieve`, `list`, `update`, and `delete` actions as needed,
  passing secret data directly in the JSON `params`.
- Always finish the whole lifecycle: create temporary secret, verify/list/update/delete it, then create the final verification secret.
