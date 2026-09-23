<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# CLI Runbook

## Configuration Management

### First-Time Setup

```bash
# Set the base URL and authenticate
nemo config set --base-url https://nhx.staging.example.com
nemo auth login

# Or configure a named context
nemo config set --context production --base-url https://nhx.example.com --activate
nemo auth login
```

### View Config

```bash
nemo config view                    # Current context and its references
nemo config view -f json            # Current context in JSON format
nemo config view --all-contexts     # All contexts, clusters, and users
```

### Quick Configuration

Set values on the current (or specified) context:

```bash
# Set base URL on current context's cluster
nemo config set --base-url https://api.example.com

# Set access token on current context's user
nemo config set --access-token YOUR_ACCESS_TOKEN
nemo config set --access-token -            # Prompt for token securely

# Set default workspace
nemo config set --workspace production

# Set output preferences
nemo config set --output-format json --timestamp-format relative

# Set values on a specific context
nemo config set --context staging --workspace staging

# Activate a context while setting values
nemo config set --context production --activate --workspace production
```

### Context Management

```bash
nemo config current-context         # Show the effective current context name
nemo config use-context staging     # Switch to a different context

# Create another context without switching from the current one
nemo config set --context development --base-url https://nhx.dev.example.com

# Create a context and make it current
nemo config set --context production --base-url https://nhx.example.com --activate

# Update a context without changing the current context
nemo config set --context development --workspace development

# Authenticate a specific context
nemo auth login --context development
```

## Common Operations

### List Resources

```bash
nemo workspaces list

# With pagination
nemo workspaces list --page 1 --page-size 20

# Fetch all pages
nemo workspaces list --all-pages

# Filter columns
nemo workspaces list --output-columns id,description,created_at
```

### Get Resource

```bash
nemo workspaces get my-workspace
```

### Create Resource

```bash
# From inline data
nemo workspaces create --input-data '{"id": "dev", "description": "Development"}'

# From file
nemo workspaces create --input-file config.json

# From stdin
cat config.json | nemo workspaces create --input-file -

# With field overrides
nemo workspaces create --input-file base.json --id "production"
```

### Update Resource

```bash
nemo workspaces update my-workspace --input-file updates.json
nemo workspaces update my-workspace --input-data '{"description": "Updated"}'
```

### Delete Resource

```bash
nemo workspaces delete my-workspace
```

## Output Formats

```bash
nemo workspaces list -f table       # Default: rich table
nemo workspaces list -f json        # Pretty JSON
nemo workspaces list -f yaml        # YAML
nemo workspaces list -f markdown    # Markdown table
nemo workspaces list -f csv         # CSV
nemo workspaces list -f raw         # Compact JSON
nemo workspaces list -f code        # Python SDK code
```

### Export to File

```bash
nemo workspaces list --all-pages --no-truncate -f markdown > workspaces.md
nemo workspaces list -f csv > workspaces.csv
```

## Multiple Contexts

```bash
# Create contexts with isolated connection settings
nemo config set --context production --base-url https://nhx.example.com --activate
nemo auth login --context production
nemo config set --context staging --base-url https://nhx.staging.example.com
nemo auth login --context staging

# Switch between contexts
nemo config use-context staging
nemo config current-context         # Shows: staging

# Use context override for single command
nemo --context production workspaces list
```

## Environment Variables

Override any setting via environment variables:

```bash
# Override config file path
NHX_CONFIG_FILE=/etc/nhx/config.yaml nemo config view

# Override base URL for single command
NHX_BASE_URL=http://localhost:8080 nemo workspaces list

# Set defaults for session
export NHX_OUTPUT_FORMAT=json
export NHX_WORKSPACE=staging
```

## Troubleshooting

### Base URL Not Set

```
Error: Base URL not specified
```

Fix: Set via config, env var, or CLI flag:
```bash
nemo config set --base-url https://nhx.example.com
# or
export NHX_BASE_URL=https://nhx.example.com
# or
nemo --base-url https://nhx.example.com workspaces list
```

### No Config File Found

```
Error: No config file found
```

Fix: Set the base URL:
```bash
nemo config set --base-url https://nhx.example.com
```

### Debug Mode

Enable verbose logging:
```bash
nemo -v workspaces list
```

## Testing Commands

### Unit Tests

```bash
cd packages/nemo_helix_ext
uv run pytest
```
