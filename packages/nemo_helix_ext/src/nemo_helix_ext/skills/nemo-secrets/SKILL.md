---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: nemo-secrets
description: >
  NeMo secrets CLI reference for creating, listing, and managing secrets.
  Use when the task involves creating API key secrets, managing credentials,
  or `nemo secrets` CLI commands.
preconditions:
  - nemo_setup_complete
  - workspace_exists
user-invocable: true
allowed-tools: Bash, Read, Grep
---

# NeMo Secrets CLI Reference

## Environment

- **CLI path**: `/app/.venv/bin/nemo`
- **API server**: `http://localhost:8080` (default)
- **Default workspace**: `default`

## Commands

```bash
# Create a secret (direct value)
nemo secrets create <name> \
  --value "<value>" \
  --description "<description>"

# Create a secret (pipe value, useful for special characters)
echo "<value>" | nemo secrets create <name> --from-file - --description "<description>"

# Create a secret in a specific workspace
nemo secrets create <name> --value "<value>" --description "<desc>" --workspace <workspace>

# List secrets
nemo secrets list

# Get a secret by name
nemo secrets get <name>

# Update a secret
echo "<new-value>" | nemo secrets update <name> --from-file - --description "<new description>"

# Delete a secret
nemo secrets delete <name>
```

**Note**: The command is `nemo secrets` (plural), not `nemo secret`.

## Common Usage

### Store an API key for inference providers

```bash
nemo secrets create nvidia-api-key \
  --value "$ANTHROPIC_API_KEY" \
  --description "NVIDIA inference API key"
```

**Note**: In Harbor eval environments, `$ANTHROPIC_API_KEY` is the canonical env var for the inference API key, regardless of the actual provider.

The secret name is then referenced by other commands, e.g. `--api-key-secret-name nvidia-api-key` when creating inference providers.
