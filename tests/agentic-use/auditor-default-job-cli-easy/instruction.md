<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Run Default Audit (CLI)

You have access to the `nemo` CLI for NeMo Helix operations. Note: MCP tools are not available in this environment - you must use the CLI.

The `nemo` CLI is available at `/app/.venv/bin/nemo`. The CLI connects to the local NeMo Helix API server at http://localhost:8080 by default. CLI auth is pre-configured.

## Context

An inference provider named `nvidia-inference` has been pre-configured in this environment with model access via NVIDIA's inference API. You can use this provider when creating audit targets.

## Task

Using the `nemo` CLI, create an audit target, create an audit config, and run an audit:

1. Create an audit target named `audit-target` pointing to model `aws/anthropic/bedrock-claude-sonnet-4-5-v1` through the `nvidia-inference` provider
2. Create an audit config named `default`
3. Run an audit using the `default` config and the `audit-target` target

Note: The audit may take a long time to complete. If the local Garak runtime is unavailable, capture the CLI error after invoking the audit command.

## Available CLI Commands

### Audit Targets

```bash
# Create an audit target (provider is passed via options in the JSON body)
nemo auditor targets create <name> -d '{"model": "<model-name>", "type": "<type>", "options": {"provider": "<provider-name>"}}'
```

### Audit Configs

```bash
# Create an audit config
nemo auditor configs create <name> -d '{"system": {"lite": true}, "run": {"generations": 5}, "plugins": {"probe_spec": "dan.AutoDANCached,goodside"}, "reporting": {}}'
```

### Audit Run

```bash
# Run an audit locally (spec references config and target as namespace/name)
nemo auditor audit run --spec '{"config": "<namespace>/<config-name>", "target": "<namespace>/<target-name>"}'
```

## Hints

- The provider type for `nvidia-inference` is `openai` (it's an OpenAI-compatible API)
- For the audit config, include minimal valid JSON for `system`, `run`, `plugins`, and `reporting`
- Configs and targets you create are in the `default` namespace, so reference them as `default/default` and `default/audit-target`
- Use `--spec` with a JSON string for the audit run, not separate flags

## Success Criteria

The task is complete when:
- An audit target named `audit-target` exists referencing the model through the provider
- An audit config named `default` exists
- The audit run command has been invoked with the default config and audit target
