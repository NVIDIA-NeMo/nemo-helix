---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: auditor
description: >
  NeMo auditor CLI reference for audit configs, targets, and jobs.
  Use when the task involves audit configurations, audit targets, audit jobs,
  vulnerability scanning, probes, or `nemo auditor` CLI commands.
allowed-tools: Bash
metadata:
  author: NeMo Helix Team <nemo-helix@nvidia.com>
---

# NeMo Auditor CLI Reference

## Environment

- **API server**: `http://localhost:8080` (default)
- **Default workspace/namespace**: `default`

## Audit Config Commands

```bash
# List workspace configs
nemo auditor configs list

# List workspace configs
nemo auditor configs list --workspace <ws>

# Create a config with a JSON payload
# The -d body includes description plus plugins, reporting, run, and system sections.
nemo auditor configs create <name> \
  -d '{"description": "<description>", "plugins": {"probe_spec": "dan.AutoDANCached"}, "reporting": {}, "run": {}, "system": {"lite": true}}'

# Get a config
nemo auditor configs get <name>

# Update a config
nemo auditor configs update <name> \
  -d '{"description": "<new description>", "plugins": {"probe_spec": "dan.AutoDANCached"}, "reporting": {}, "run": {}, "system": {"lite": true}}'

# Delete a config
nemo auditor configs delete <name>
```

### Config JSON Structure

Minimal example for each required field:
- **plugins**: `{"probe_spec": "dan.AutoDANCached"}` — specifies which probes to run
- **reporting**: `{}`
- **run**: `{}` or `{"generations": 5}`
- **system**: `{"lite": true}`

Common probe specs: `dan.AutoDANCached`, `dan.DanInTheWild`, `dan.goodside`

## Audit Target Commands

```bash
# Create a target
nemo auditor targets create <name> \
  -d '{"model": "<model-name>", "type": "<type>", "description": "<description>"}'

# Create a target with a provider (for real inference endpoints)
nemo auditor targets create <name> \
  -d '{"model": "<model-name>", "type": "<type>", "options": {"provider": "<provider-name>"}}'

# List targets
nemo auditor targets list

# Get a target
nemo auditor targets get <name>

# Update a target
nemo auditor targets update <name> \
  -d '{"model": "<model-name>", "type": "<type>", "description": "<new description>"}'

# Delete a target
nemo auditor targets delete <name>
```

Target types: `nim`, `openai`

## Audit Job Commands

```bash
# Run an audit locally (spec references config and target as namespace/name)
nemo auditor audit run \
  --spec '{"config": "default/<config-name>", "target": "default/<target-name>"}'

# Submit an audit to a configured cluster
nemo auditor audit submit \
  --spec '{"config": "default/<config-name>", "target": "default/<target-name>"}'
```

Jobs may take a long time or remain in pending/created status. That is expected.

## Typical Workflows

### Config CRUD

1. `nemo auditor configs list` — inspect configs in the active workspace
2. `nemo auditor configs create my-config -d '{...}'` — create
3. `nemo auditor configs get my-config` — verify
4. `nemo auditor configs update my-config -d '{...}'` — update
5. `nemo auditor configs delete my-config` — delete

### Run an Audit Job

1. Create a target pointing to the model endpoint
2. Create a config with probe selection
3. Create a job referencing `default/<config>` and `default/<target>`
4. Run locally with `nemo auditor audit run --spec '{...}'`, or submit with `nemo auditor audit submit --spec '{...}'`
