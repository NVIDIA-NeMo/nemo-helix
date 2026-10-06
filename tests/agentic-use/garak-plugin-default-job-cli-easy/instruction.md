<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Run Default Scan (CLI)

You have access to the `nemo` CLI for NeMo Helix operations. Note: MCP tools are not available in this environment - you must use the CLI.

The `nemo` CLI is available at `/app/.venv/bin/nemo`. The CLI connects to the local NeMo Helix API server at http://localhost:8080 by default. CLI auth is pre-configured.

## Context

An inference provider named `nvidia-inference` has been pre-configured in this environment with model access via NVIDIA's inference API. You can use this provider when creating scan targets.

## Task

Using the `nemo` CLI, create a scan target, create a scan config, and run a scan:

1. Create a scan target named `scan-target` pointing to model `aws/anthropic/bedrock-claude-sonnet-4-5-v1` through the `nvidia-inference` provider
2. Create a scan config named `default`
3. Run a scan using the `default` config and the `scan-target` target

Note: The scan may take a long time to complete. If the local Garak runtime is unavailable, capture the CLI error after invoking the scan command.

## Available CLI Commands

### Scan Targets

```bash
# Create a scan target (provider is passed via options in the JSON body)
nemo garak-plugin targets create <name> -d '{"model": "<model-name>", "type": "<type>", "options": {"provider": "<provider-name>"}}'
```

### Scan Configs

```bash
# Create a scan config
nemo garak-plugin configs create <name> -d '{"system": {"lite": true}, "run": {"generations": 5}, "plugins": {"probe_spec": "dan.AutoDANCached,goodside"}, "reporting": {}}'
```

### Scan Run

```bash
# Submit a scan job (spec references config and target as namespace/name)
nemo garak-plugin scan --spec '{"config": "<namespace>/<config-name>", "target": "<namespace>/<target-name>"}'
```

## Hints

- The provider type for `nvidia-inference` is `openai` (it's an OpenAI-compatible API)
- For the scan config, include minimal valid JSON for `system`, `run`, `plugins`, and `reporting`
- Configs and targets you create are in the `default` namespace, so reference them as `default/default` and `default/scan-target`
- Use `--spec` with a JSON string for the scan job, not separate flags

## Success Criteria

The task is complete when:
- A scan target named `scan-target` exists referencing the model through the provider
- A scan config named `default` exists
- The scan command has been invoked with the default config and scan target
