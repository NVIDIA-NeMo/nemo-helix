<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Run Default Scan (CLI)

You have access to the `nhx` CLI for NeMo Helix operations. Note: MCP tools are not available in this environment - you must use the CLI.

The `nhx` CLI is available at `/app/.venv/bin/nhx`. The CLI connects to the local NeMo Helix API server at http://localhost:8080 by default. CLI auth is pre-configured.

## Context

An inference provider named `nvidia-inference` has been pre-configured in this environment with model access via NVIDIA's inference API. You can use this provider when creating scan targets.

## Task

Using the `nhx` CLI, create a scan target, create a scan config, and run a scan:

1. **Create a scan target** named `scan-target` that references model `aws/anthropic/bedrock-claude-sonnet-4-5-v1` through the `nvidia-inference` provider
2. **Create a scan config** named `default` with appropriate settings
3. **Run a scan** using the `default` config and the `scan-target` target

Note: The scan may take a long time to complete in this environment. The important thing is that the target and config are created correctly and the scan command is invoked with both of them.

## Available CLI Commands

```bash
# Create the scan target.
nhx garak-plugin targets create scan-target -d '{"model": "aws/anthropic/bedrock-claude-sonnet-4-5-v1", "type": "openai", "options": {"provider": "nvidia-inference"}}'

# Create the default scan config.
nhx garak-plugin configs create default -d '{"system": {"lite": true}, "run": {"generations": 5}, "plugins": {"probe_spec": "dan.AutoDANCached,goodside"}, "reporting": {}}'

# Submit the scan job.
nhx garak-plugin scan --spec '{"config": "default/default", "target": "default/scan-target"}'
```

## Success Criteria

The task is complete when:
- A scan target named `scan-target` exists referencing the model through the provider
- A scan config named `default` exists
- The scan command has been invoked with the default config and scan target
