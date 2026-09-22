<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NeMo Platform SDK extensions

This package provides extensions to the generated `nemo-platform` SDK. It includes things that are built on top
of the SDK and is added to the SDK package (this way we only need to maintain a single package).

These extensions include:
- The CLI module (`nemo_platform_ext/src/nemo_platform_ext/cli`) - provides a `nmp` command that provides a way to interact with the platform.
- The high-level SDK: config, quickstart, files, models (these are still in a separate package).
  - These modules enhance interacting with the platform by writing/running Python code.

## Installation

This package is a member of the NeMo Platform workspace, so `uv sync` from the root of the workspace makes it available as `nmp` (or `uv run nmp` without venv activated).

## Build troubleshooting

The wheel packages the repo docs source tree for `nemo docs`. If a local build fails with a long `docs/_build/html/_build/html/...` path, remove stale legacy docs output and rebuild:

```bash
rm -rf docs/_build docs/_generated/nemo-nb/_build
```

## CLI

### Getting started

```bash
# Quick setup
nemo config set --base-url https://nmp.example.com && nemo auth login

# Or use the recommended local setup flow
nemo setup

# List workspaces
nemo workspaces list

# Interactive chat with a model
nemo chat nvidia-build --model nvidia/llama-3.3-nemotron-super-49b-v1.5
```

## Configuration

The CLI uses a kubeconfig-style configuration system with clusters, users, and contexts.

### Config File

The CLI reads configuration from `~/.config/nmp/config.yaml`:

```yaml
current_context: production

clusters:
  - name: prod-cluster
    base_url: https://nmp.example.com
  - name: local
    base_url: http://localhost:8080

users:
  - name: prod-admin
    type: api-key
    api_key: your-api-key-here
  - name: local-user
    type: no-auth

contexts:
  - name: production
    cluster: prod-cluster
    user: prod-admin
    workspace: default
    preferences:
      output_format: table
      timestamp_format: relative
  - name: local
    cluster: local
    user: local-user
```

**Key concepts:**
- **Clusters** define API endpoints
- **Users** define authentication (API key, OAuth, or no auth)
- **Contexts** bind a cluster + user + preferences together

### Quick Setup

Configure a context with `nemo config set`:

```bash
nemo config set --context production --base-url https://nmp.example.com --activate
```

### Priority

Configuration values are resolved in priority order (highest to lowest):

1. **CLI flags** (`--base-url`, `--output-format`, etc.)
2. **Environment variables** (`NMP_BASE_URL`)
3. **Config file** (`~/.config/nmp/config.yaml`)
4. **Defaults**

### Environment Variables

Override any setting via environment variables:

```bash
NMP_CONFIG_FILE=/path/to/config.yaml   # Override config file path
NMP_CURRENT_CONTEXT=local              # Which context to use
NMP_WORKSPACE=my-workspace             # Default workspace
NMP_BASE_URL=https://api.example.com   # API endpoint
NMP_API_KEY=your-key                   # Authentication
NMP_OUTPUT_FORMAT=json                 # Output format
NMP_TIMESTAMP_FORMAT=relative          # Timestamp display
NMP_PAGE_SIZE=50                       # Results per page
```

### Context Management

Switch between environments using contexts:

```bash
# View current context name
nemo config current-context

# View full configuration
nemo config view

# View all contexts
nemo config view --all-contexts

# Switch context
nemo config use-context production

# Override context for single command
nemo --context staging workspaces list
```

### Config Commands

Manage configuration programmatically:

```bash
# Quick configuration
nemo auth login --base-url https://api.example.com
nemo config set --base-url https://api.example.com
nemo config set --api-key YOUR_API_KEY
nemo config set --workspace my-workspace --output-format json

# Configure and activate a named context in one step
nemo config set --context production --base-url https://nmp.example.com --activate

# Switch to an existing context
nemo config use-context production
```

## Setup (Local Development)

Run NeMo Platform locally with the recommended setup flow:

```bash
# Recommended setup flow
nemo setup

# Start local services directly when needed
nemo services run
```

## Usage

### API Commands

```bash
# List resources
nemo workspaces list
nemo workspaces list --output-format table

# Get a resource
nemo workspaces get my-workspace

# Create a resource
nemo workspaces create --input-data '{"id": "my-workspace", "description": "My workspace"}'
nemo workspaces create --input-file config.json

# Create with file + field overrides
nemo workspaces create --input-file base.json --description "Override description"

# Stdin support
echo '{"id": "prod"}' | nemo workspaces create --input-file -

# Update a resource
nemo workspaces update my-workspace --input-file updates.json

# Delete a resource
nemo workspaces delete my-workspace

# Generate Python SDK code
nemo workspaces get my-workspace --output-format code
```

## Output Formats

Supports multiple output formats via `--output-format` / `--output` / `-f`:

- **table** (default) - Rich-formatted table with borders and colors
- **json** - Syntax-highlighted JSON
- **yaml** - Syntax-highlighted YAML
- **markdown** - Markdown table (for docs/issues)
- **csv** - Standard CSV (for Excel/spreadsheets)
- **code** - Python SDK code equivalent
- **raw** - Compact JSON without formatting

For list commands, combine `--stream` with `--output json` or `--output raw` to emit one newline-delimited JSON record per item.

## Output Columns

Customize columns in list commands with `--output-columns`:

```bash
nemo workspaces list --output-columns default                          # Default columns
nemo workspaces list --output-columns all                              # All columns
nemo workspaces list --output-columns id,description,created_at        # Specific columns
nemo workspaces list --output-columns resource_id                      # Column alias
```

Quote values with special characters: `--output-columns 'field.$name'`

## Development

Built with [Typer](https://typer.tiangolo.com/), [Rich](https://rich.readthedocs.io/), and the typed clients in `nemo_platform_plugin`.

See [docs/overview.md](docs/overview.md) for architecture details.

### Command Groups

`nemo <group> *` commands are hand-written on the typed clients in `nemo_platform_plugin`
(`src/nemo_platform_ext/cli/commands/`); the CLI has no dependency on the generated
`nemo_platform` SDK. Functional groups (`guardrail`, `intake`, `experiments`, ...) are
`nemo.cli` entry points shipped by the owning package.

To vendor the CLI and regenerate its reference docs (from repo root):
```shell
make update-cli
```
