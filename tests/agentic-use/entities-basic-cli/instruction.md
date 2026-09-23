<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Entity CRUD Operations (CLI)

You have access to the `nhx` CLI for NeMo Helix operations. Note: MCP tools are not available in this environment - you must use the CLI.

The `nhx` CLI is available at `/app/.venv/bin/nhx`. The CLI connects to the local NeMo Helix API server at http://localhost:8080 by default.

## Task

Complete the following entity CRUD operations using the `nhx` CLI:

1. Create an entity of type `model` named `harbor-test-model` with data `{"framework": "pytorch", "version": "1.0", "description": "Test model for harbor eval"}`
2. Verify the entity was created by retrieving it
3. List all entities of type `model` and confirm `harbor-test-model` appears
4. Update the entity's data to `{"framework": "pytorch", "version": "2.0", "description": "Updated test model"}`
5. Verify the update was applied by retrieving the entity again
6. Delete the entity `harbor-test-model`
7. Create a new entity of type `dataset` named `harbor-final-dataset` with data `{"format": "jsonl", "size": 1000, "description": "Final dataset for verification"}`

## Success Criteria

The task is complete when:
- An entity of type `model` named `harbor-test-model` was created, updated, and then deleted
- An entity of type `dataset` named `harbor-final-dataset` exists with data containing `"format": "jsonl"` and `"size": 1000`
- The entity `harbor-test-model` no longer exists (was deleted)
