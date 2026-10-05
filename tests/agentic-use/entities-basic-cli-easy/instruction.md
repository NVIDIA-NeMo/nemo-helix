<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Entity CRUD Operations (CLI)

You have access to the `nemo` CLI for NeMo Helix operations. Note: MCP tools are not available in this environment - you must use the CLI.

## Task

Complete the following entity CRUD operations using the `nemo` CLI:

1. Create an entity of type `model` named `harbor-test-model` with data `{"framework": "pytorch", "version": "1.0", "description": "Test model for harbor eval"}`
2. Verify the entity was created by retrieving it with `nemo entities get-entity-by-name harbor-test-model --entity-type model`
3. List all entities of type `model` and confirm `harbor-test-model` appears
4. Update the entity's data to `{"framework": "pytorch", "version": "2.0", "description": "Updated test model"}` using the update command
5. Verify the update was applied by retrieving the entity again
6. Delete the entity with `nemo entities delete-entity-by-name harbor-test-model --entity-type model`
7. Create a new entity of type `dataset` named `harbor-final-dataset` with data `{"format": "jsonl", "size": 1000, "description": "Final dataset for verification"}`

## Available CLI Commands

The `nemo` CLI is available at `/app/.venv/bin/nemo`. You can use these commands:

- `nemo entities create <entity_type> <name> --data '<json>'` - Create an entity
- `nemo entities get-entity-by-name <name> --entity-type <type>` - Retrieve an entity by name
- `nemo entities list <entity_type>` - List entities of a given type
- `nemo entities update-entity-by-name <name> --entity-type <type> --data '<json>'` - Update an entity
- `nemo entities delete-entity-by-name <name> --entity-type <type>` - Delete an entity

Note: The CLI connects to the local NeMo Helix API server at http://localhost:8080 by default.

## Success Criteria

The task is complete when:
- An entity of type `model` named `harbor-test-model` was created, updated, and then deleted
- An entity of type `dataset` named `harbor-final-dataset` exists with data containing `"format": "jsonl"` and `"size": 1000`
- The entity `harbor-test-model` no longer exists (was deleted)
