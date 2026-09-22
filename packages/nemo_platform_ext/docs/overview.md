<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# CLI Architecture Overview

## Project Structure

```
src/nemo_platform_ext/cli/
├── app.py                    # Main Typer app, global options callback
├── core/                     # Core logic for handling the CLI commands
│   ├── context.py            # CLIContext - state management
│   └── types.py              # Type definitions (OutputFormat, TimestampFormat)
└── commands/                 # This is where all CLI commands are defined
    ├── api/                  # AUTO-GENERATED - do not edit manually
    │   ├── workspaces.py
    │   └── ...
    ├── config.py             # Config management commands
    └── quickstart/           # Quickstart deployment commands
```

## Configuration System

The CLI uses a layered configuration system built on the SDK's config module. Configuration can come from:

1. **Config file** (lowest priority) - YAML file at `~/.config/nmp/config.yaml`
2. **Environment variables** - Variables with `NMP_` prefix
3. **CLI arguments** (highest priority) - Flags like `--base-url`, `--context`

### Configuration File Structure

The config file follows a kubeconfig-style pattern with clusters, users, and contexts:

```yaml
current_context: prod

clusters:
  - name: prod
    base_url: https://api.prod.example.com

users:
  - name: prod
    api_key: your-api-key-here

contexts:
  - name: prod
    cluster: prod
    user: prod
    workspace: prod-workspace
    preferences:
      output_format: table
      timestamp_format: iso8601
      page_size: 20
      color_output: true
```

**Key concepts:**
- **Clusters** define API endpoints
- **Users** define authentication (OAuth token or no auth)
- **Contexts** bind a cluster + user + preferences together
- **Preferences** control output formatting

### Environment Variables

All settings can be overridden via environment variables:

```bash
NMP_CONFIG_FILE=/path/to/config.yaml   # Override config file path
NMP_CURRENT_CONTEXT=local              # Which context to use
NMP_WORKSPACE=my-workspace             # Default workspace
NMP_BASE_URL=https://api.example.com   # API endpoint
NMP_ACCESS_TOKEN=your-token            # Authentication
NMP_OUTPUT_FORMAT=json                 # Output format
NMP_TIMESTAMP_FORMAT=relative          # Timestamp display
NMP_PAGE_SIZE=50                       # Results per page
```

### State Management

Application state is managed via `CLIContext`, stored in `typer.Context.obj`:

```python
from nemo_platform_ext.cli.core.context import CLIContext

@dataclass
class CLIContext:
    # CLI overrides passed to SDK Config.load()
    overrides: ConfigParams = field(default_factory=dict)

    # Verbosity (CLI-specific, not in ConfigParams)
    verbosity: int = 0

    # Lazy-loaded SDK context
    _sdk_context: Context | None = None

    # Lazy-created client
    _client: NeMoPlatform | None = None

    # Additional settings
    quickstart_settings: QuickstartSettings | None = None
```

The main app callback (`app.py`) initializes `CLIContext` and collects CLI overrides:

```python
@app.callback()
def main(ctx: typer.Context, base_url: str | None = None, ...):
    cli_context = CLIContext()

    # Build ConfigParams from CLI arguments
    overrides: ConfigParams = {}
    if context_name:
        overrides["current_context"] = context_name
    if base_url:
        overrides["base_url"] = base_url
    # ... etc

    cli_context.overrides = overrides
    ctx.obj = cli_context
```

Subcommands access the context via `ctx.obj`:

```python
def list_models(ctx: typer.Context):
    cli_context: CLIContext = ctx.obj

    # Get resolved SDK context (lazy loads on first access)
    sdk_context = cli_context.get_sdk_context()

    # Get API client
    client = cli_context.get_client()
```

### Lazy Config Loading

Configuration is loaded lazily via `CLIContext.get_sdk_context()` to avoid file reads for `--help`:

```python
def get_sdk_context(self) -> Context:
    if self._sdk_context is None:
        self._sdk_context = get_context(overrides=self.overrides)
    return self._sdk_context
```

The SDK's `get_context()` function:
1. Loads the config file (if it exists)
2. Applies environment variable overrides
3. Applies CLI overrides (highest priority)
4. Resolves cluster/user references to actual objects
5. Returns a `Context` with resolved configuration

## Command Groups

API command groups are hand-written on the typed clients in `nemo_platform_plugin`
(see `commands/secrets.py` for the reference shape). Core groups are registered in
`commands/manifest_registry.py`; functional groups are `nemo.cli` entry points shipped
by the owning plugin or service package. `--output-format code` renders the equivalent
typed-client Python via `core/code_generator.py`.

### Regenerating Reference Docs

```bash
make update-cli
```

## Input Handling

Create/update commands support three input methods (see [architecture/create-update-operations.md](architecture/create-update-operations.md)):

1. `--input-file PATH` - JSON from file or stdin (`-`)
2. `--input-data DATA` - Inline JSON/YAML string
3. Field flags - Override specific fields (`--id`, `--description`)

Precedence: CLI flags > input-file/input-data
