<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Custom Scan with Selected Probes (CLI)

You have access to the `nhx` CLI for NeMo Helix operations. Note: MCP tools are not available in this environment - you must use the CLI.

The `nhx` CLI is available at `/app/.venv/bin/nhx`. The CLI connects to the local NeMo Helix API server at http://localhost:8080 by default. CLI auth is pre-configured.

## Context

An inference provider named `nvidia-inference` has been pre-configured in this environment with model access via NVIDIA's inference API. You can use this provider when creating scan targets.

## Task

Using the `nhx` CLI, create a custom scan configuration with specific selected probes, create a scan target, and run a scan with that custom config:

1. **Explore available global scan configs** to understand the config structure and available options
2. **Create a scan target** named `custom-scan-target` that references model `aws/anthropic/bedrock-claude-sonnet-4-5-v1` through the `nvidia-inference` provider
3. **Create a custom scan config** named `custom-probes-config` with description `Custom config with selected probes` that uses exactly these three probes: `dan.DanInTheWild`, `dan.AutoDANCached`, and `dan.Ablation_Dan_11_0`
4. **Verify the config** by retrieving it and confirming the probe selection
5. **Run a scan** using the `custom-probes-config` config and the `custom-scan-target` target
6. **Review the CLI output** for result artifact paths or errors

Note: The scan may take a long time to complete in this environment. If the local Garak runtime is unavailable, capture the CLI error after invoking the scan command. The important thing is that the target and custom config are created correctly and that the scan command is invoked with both of them.

## Available CLI Commands

```bash
# Create a scan target
nhx garak-plugin targets create <name> -d '{"model": "<model-name>", "type": "<type>", "options": {"provider": "<provider-name>"}}'

# Create a scan config
nhx garak-plugin configs create <name> -d '{"description": "<description>", "system": {"lite": true}, "run": {"generations": 5}, "plugins": {"probe_spec": "dan.DanInTheWild,dan.AutoDANCached,dan.Ablation_Dan_11_0"}, "reporting": {}}'

# Verify configs and targets
nhx garak-plugin configs get <name>
nhx garak-plugin targets get <name>

# Submit a scan job
nhx garak-plugin scan --spec '{"config": "default/<config-name>", "target": "default/<target-name>"}'
```

## Success Criteria

The task is complete when:
- A scan target named `custom-scan-target` exists referencing the model through the provider
- A scan config named `custom-probes-config` exists with the three specified probes
- The scan command has been invoked with the custom config and target
- The CLI output has been reviewed for result artifact paths or errors
