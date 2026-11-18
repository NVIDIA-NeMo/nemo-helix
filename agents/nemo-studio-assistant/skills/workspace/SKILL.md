---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: workspace
description: "Explain NeMo Helix workspaces and manage them with nemo_api(resource='workspaces'). Use when the user asks what a workspace is, which workspaces exist, or wants to create, inspect, or delete one."
metadata:
  author: NeMo Helix Team <nemo-helix@nvidia.com>
---
# Workspaces

A workspace is the unit of isolation in NeMo Helix. Models, providers,
filesets, secrets, agents, and jobs all belong to one workspace, and members
are granted access per workspace. The user's current workspace is in the
Studio context.

## Explaining

- When someone asks why they cannot see a resource, check whether it lives in
  a different workspace before anything else.
- Suggest a new workspace for a separate team or project, not for every
  experiment.
- Link to workspace settings with `studio_link` destination `settings`, and to
  access management with `members`.

## Acting on request

- `nemo_api(resource="workspaces", action="list", workspace="<active request workspace>")`
- `nemo_api(resource="workspaces", action="retrieve", params='{"name":"..."}', workspace="<active request workspace>")`
- `nemo_api(resource="workspaces", action="create", params='{"name":"...", "description":"..."}', studio_session_id="<active Studio session UUID>", workspace="<active request workspace>")`
- `nemo_api(resource="workspaces", action="delete", params='{"name":"..."}', studio_session_id="<active Studio session UUID>", workspace="<active request workspace>")`

Deleting a workspace removes everything in it. Confirm the name with the user
before you request approval for a delete.
