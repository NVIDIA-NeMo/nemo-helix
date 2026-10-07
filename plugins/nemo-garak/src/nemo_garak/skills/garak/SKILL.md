---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: garak
description: >
  NeMo garak CLI reference for audit configs, targets, and jobs.
  Use when the task involves audit configurations, audit targets, audit jobs,
  vulnerability scanning, probes, or `nemo garak` CLI commands.
allowed-tools: Bash
metadata:
  author: NeMo Helix Team <nemo-helix@nvidia.com>
---

# NeMo Garak CLI Reference

## Environment

- **API server**: `http://localhost:8080` (default)
- **Default workspace/namespace**: `default`

## Audit Config Commands

```bash
# List workspace configs
nemo garak configs list

# List workspace configs
nemo garak configs list --workspace <ws>

# Create a config with a JSON payload
# The -d body includes description plus plugins, reporting, run, and system sections.
nemo garak configs create <name> \
  -d '{"description": "<description>", "plugins": {"probe_spec": "dan.AutoDANCached"}, "reporting": {}, "run": {}, "system": {"lite": true}}'

# Get a config
nemo garak configs get <name>

# Update a config
nemo garak configs update <name> \
  -d '{"description": "<new description>", "plugins": {"probe_spec": "dan.AutoDANCached"}, "reporting": {}, "run": {}, "system": {"lite": true}}'

# Delete a config
nemo garak configs delete <name>
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
nemo garak targets create <name> \
  -d '{"model": "<model-name>", "type": "<type>", "description": "<description>"}'

# Create a target with a provider (for real inference endpoints)
nemo garak targets create <name> \
  -d '{"model": "<model-name>", "type": "<type>", "options": {"provider": "<provider-name>"}}'

# List targets
nemo garak targets list

# Get a target
nemo garak targets get <name>

# Update a target
nemo garak targets update <name> \
  -d '{"model": "<model-name>", "type": "<type>", "description": "<new description>"}'

# Delete a target
nemo garak targets delete <name>
```

Target types: `nim`, `openai`

## Audit Job Commands

```bash
# Submit an audit job (spec references config and target as namespace/name)
nemo garak audit \
  --spec '{"config": "default/<config-name>", "target": "default/<target-name>"}'

# Print the audit job input/output schemas
nemo garak audit explain
```

Jobs may take a long time or remain in pending/created status. That is expected.

## Typical Workflows

### Config CRUD

1. `nemo garak configs list` — inspect configs in the active workspace
2. `nemo garak configs create my-config -d '{...}'` — create
3. `nemo garak configs get my-config` — verify
4. `nemo garak configs update my-config -d '{...}'` — update
5. `nemo garak configs delete my-config` — delete

### Run an Audit Job

1. Create a target pointing to the model endpoint
2. Create a config with probe selection
3. Create a job referencing `default/<config>` and `default/<target>`
4. Submit it with `nemo garak audit --spec '{...}'`
