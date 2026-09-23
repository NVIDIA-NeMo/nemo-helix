<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Auditor Config CRUD Operations (CLI)

Tests the agent's ability to perform CRUD operations on auditor configurations using the NeMo Helix CLI.

## What This Tests

- Listing global audit configs to discover structure
- Creating an audit config with specific probe selection
- Retrieving an audit config by name
- Updating an audit config's description
- Deleting an audit config
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
