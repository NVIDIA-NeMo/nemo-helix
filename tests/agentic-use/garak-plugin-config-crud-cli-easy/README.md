<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Garak Plugin Config CRUD Operations (CLI)

Tests the agent's ability to perform CRUD operations on garak-plugin configurations using the NeMo Helix CLI.

## What This Tests

- Listing global scan configs to discover structure
- Creating a scan config with specific probe selection
- Retrieving a scan config by name
- Updating a scan config's description
- Deleting a scan config
- Creating a final config for verification

## Expected Agent Behavior

1. List global configs to understand the expected JSON structure
2. Create `harbor-test-config` with the `dan.AutoDANCached` probe
3. Verify, list, update, then delete `harbor-test-config`
4. Create `harbor-final-config` with the `dan.DanInTheWild` probe

## Verification

The verifier checks:
- `harbor-test-config` was deleted (no longer exists)
- `harbor-final-config` exists with correct description and probe spec
- Agent trajectory shows all CRUD CLI commands were executed
