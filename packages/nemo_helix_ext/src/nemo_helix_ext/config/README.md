<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

Examples:

Configuration File Structure:
```yaml
# Current active context
current_context: default

# Clusters define connection endpoints
clusters:
- name: production
  base_url: https://api.prod.example.com
- name: local
  base_url: http://localhost:8123

# Users define authentication credentials
users:
- name: prod-admin
  type: oauth
  token: eyJhbGc...
  refresh_token: eyJhbGc...
- name: prod-oauth
  type: oauth
  token: eyJhbGc...
  refresh_token: eyJhbGc...
- name: local-user  # No auth fields for no authentication
  type: no-auth

# Contexts bind together cluster + user + preferences
contexts:
- name: default
  cluster: production
  user: prod-admin
  preferences:
    output_format: table
    timestamp_format: relative
- name: experiment-x
  cluster: production
  user: prod-admin
  workspace: experiment-x  # Optional workspace
- name: local
  cluster: local
  user: local-user
  preferences:
    output_format: json
```

Basic Usage:
```python
from nemo_helix.config import get_context

# Get the resolved configuration context
context = get_context()

print(f"Using context: {context.context_name}")
print(f"Cluster URL: {context.cluster.base_url}")
print(f"Workspace: {context.workspace}")
print(f"Output format: {context.preferences.output_format}")
```

With Environment Variables:
```shell
# Override config file path
export NHX_CONFIG_FILE=/path/to/custom/config.yaml

# Override which context to use
export NHX_CURRENT_CONTEXT=local

# Override workspace
export NHX_WORKSPACE=my-workspace

# Override preferences
export NHX_OUTPUT_FORMAT=json

# For configurations without a config file
export NHX_BASE_URL=https://api.example.com
export NHX_ACCESS_TOKEN=your-access-token
```

```python
# These env vars are automatically picked up
context = get_context()
print(context.context_name)  # "local" (from env var)
print(context.workspace)     # "my-workspace" (from env var)
```

With CLI (Typer) - using dependency injection:
```python
# cli/main.py
from pathlib import Path
from typing import Optional
import typer
from nemo_helix.config import get_context, ConfigParams, OutputFormat, Context

app = typer.Typer()


@app.callback()
def main(
    ctx: typer.Context,
    config_file: Optional[Path] = typer.Option(None, "--config", "-c"),
    context: Optional[str] = typer.Option(None, "--context"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-n"),
    output: Optional[OutputFormat] = typer.Option(None, "--output", "-o"),
):
    """nemo_helix CLI."""
    # Build config parameters from CLI options
    params: ConfigParams = {}
    if context:
        params["current_context"] = context
    if workspace:
        params["workspace"] = workspace
    if output:
        params["output_format"] = output
    
    # Get resolved configuration context (stateless helper)
    app_context = get_context(config_path=config_file, overrides=params)
    
    # Store in context for subcommands (dependency injection)
    ctx.obj = app_context


@app.command()
def status(ctx: typer.Context):
    """Show current configuration."""
    app_context: Context = ctx.obj
    
    typer.echo(f"Context: {app_context.context_name}")
    typer.echo(f"Cluster: {app_context.cluster.name}")
    typer.echo(f"URL: {app_context.cluster.base_url}")
    if app_context.workspace:
        typer.echo(f"Workspace: {app_context.workspace}")
    typer.echo(f"\nPreferences:")
    typer.echo(f"  Output: {app_context.preferences.output_format.value}")
```

With SDK Parameters:
```python
from nemo_helix.config import get_context, ConfigParams

# Set parameters programmatically
params: ConfigParams = {
    "current_context": "local",
    "workspace": "test-workspace",
    "output_format": "json"
}

context = get_context(overrides=params)

# Now use context.cluster.base_url, context.workspace, etc.
print(f"Using: {context.context_name} / {context.workspace}")
```

Advanced Usage with Config class:
```python
from nemo_helix.config import Config

# Load config and use it explicitly (for more control)
config = Config.load(config_path="/path/to/config.yaml")

# Resolve to get context
context = config.resolve()
print(f"Context: {context.context_name}")

# You can modify runtime overrides and resolve again
config.workspace = "different-workspace"
context = config.resolve()
print(f"Workspace: {context.workspace}")
```
