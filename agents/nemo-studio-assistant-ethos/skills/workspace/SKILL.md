---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: workspace
description: NeMo Helix workspace CRUD playbook through `nemo_api(resource='workspaces')`. Use for workspace creation, listing, retrieval, or deletion.
---
# Workspace tasks

- If instructions mention MCP workspace tools (`create_workspace`, `list_workspaces`),
  complete the same intent with the corresponding `nemo_api` workspace operation below.
- SDK/API operations:
  - `nemo_api(resource="workspaces", action="create", params='{"name":"...", "description":"..."}', studio_session_id="<active Studio session UUID>", workspace="<active request workspace>")`
  - `nemo_api(resource="workspaces", action="list", workspace="<active request workspace>")`
  - `nemo_api(resource="workspaces", action="retrieve", params='{"name":"..."}', workspace="<active request workspace>")`
  - `nemo_api(resource="workspaces", action="delete", params='{"name":"..."}', studio_session_id="<active Studio session UUID>", workspace="<active request workspace>")`
- Complete all numbered requirements before stopping.
