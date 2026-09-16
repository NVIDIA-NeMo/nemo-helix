---
name: nhx-cli
description: Use when developing the NeMo Helix CLI - adding commands, modifying the generator, templates, or configuration. Covers both manual commands (use-cases, config) and auto-generated API commands. Also helps troubleshoot CLI generation failures (e.g., Typer errors after updates).
---

# NeMo Helix CLI Development

The CLI provides a command-line interface for NeMo Helix, implemented using Typer in `packages/nemo_helix_ext/src/nemo_helix_ext/cli/`.

## Command Categories

| Category | Location | Description | Editable? |
|----------|----------|-------------|-----------|
| **API** | `commands/api/` | Auto-generated from SDK | ❌ Never edit |
| **Configuration** | `commands/config.py`, `commands/configure.py` | Config management | ✅ Yes |
| **Use-case** | `commands/use_cases/` | High-level commands (e.g., `chat`) | ✅ Yes |
| **Quickstart** | `commands/quickstart/` | Local deployment | ✅ Yes |

## Directory Structure

```
packages/nemo_helix_ext/src/nemo_helix_ext/cli/
├── app.py                    # Entry point, command registration, global options
├── core/                     # Shared utilities
└── commands/
    ├── config.py             # kubectl-style config management
    ├── configure.py          # Interactive configuration
    ├── quickstart/           # Local deployment commands
    ├── use_cases/            # High-level commands (chat, wait)
    └── api/                  # ⚠️ AUTO-GENERATED - DO NOT EDIT
```

## Building the CLI

### Development Shortcut

During local CLI development, you can run `_nemo` to execute the CLI directly from `packages/nemo_helix_ext`.
This lets you test changes immediately without running vendoring first.

```bash
uv run _nemo --help
```

Use vendoring (`make vendor-nemo-helix-ext`) when you need to validate behavior in `sdk/python/nemo-helix`.

Build everything (recommended):
```bash
make update-cli
```

Individual steps:
```bash
# Step 1: Generate CLI commands from SDK
make generate-cli-commands

# Step 2: Vendor into SDK package
make vendor-nemo-helix-ext

# Step 3: Generate documentation
make generate-cli-reference-docs
```

## Adding a Manual Command

Manual commands go in `commands/use_cases/` or directly in `commands/`.

### Pattern

```python
from __future__ import annotations

from typing import Annotated

import typer

from nemo_helix_ext.cli.core.context import CLIContext
from nemo_helix_ext.cli.core.errors import handle_errors


@handle_errors
def my_command(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="The name argument")],
    option: Annotated[str | None,
        typer.Option(
            "--option",
            "-o",
            help="An optional flag",
            rich_help_panel="Options",
        ),
    ] = None,
) -> None:
    """
    Short description of what the command does.

    Examples:
      nemo my-command foo
      nemo my-command bar --option value
    """
    state: CLIContext = ctx.obj
    client = state.get_client()

    # Use client to make API calls
    result = client.some_resource.some_method(name=name)

    # Output result (use state.output() for formatted output)
    state.output(result)
```

### Register in app.py

```python
from nemo_helix_ext.cli.commands.my_module import my_command

# Single command
app.command(name="my-command", rich_help_panel="Use cases")(my_command)

# Command group (sub-app)
app.add_typer(my_sub_app, name="my-group", rich_help_panel="Category")
```

## CLI Generator

The generator creates commands in `commands/api/` from SDK introspection.

### Key Files

```
tools/nemo-helix-sdk-tools/src/nemo_helix_sdk_tools/sdk/cli_generator/
├── cli_config.yaml           # Generation configuration
├── generator.py              # Main generation logic
├── sdk_introspector.py       # SDK introspection
├── operation_classifier.py   # Classifies methods (list, get, create, etc.)
├── templates/                # Jinja2 templates
├── context_collectors/       # Collect template context per operation type
└── overrides/                # Custom command implementations
```

### Configuration (cli_config.yaml)

Configure table columns for `list` commands:
```yaml
- resource: [customization, jobs]
  methods:
    list:
      columns:
      - name
      - description
      - status
      - created_at
```

Skip a method:
```yaml
- resource: [filesets]
  methods:
    upload_file:
      skip: true
```

Use custom implementation:
```yaml
- resource: [secrets]
  methods:
    create:
      override: secrets/create.py
```

### Command Overrides

For methods requiring custom handling (file uploads, streaming), create overrides:

1. Add config in `cli_config.yaml`:
   ```yaml
   - resource: [files]
     additional_methods:
       upload:
         override: files/upload.py
   ```

2. Create override file in `overrides/files/upload.py`:
   ```python
   from typing import Annotated, Any, cast

   import typer

   from nemo_helix_ext.cli.core.context import CLIContext
   from nemo_helix_ext.cli.core.errors import handle_errors

   app = cast(Any, None)  # override-skip: provided by generated file


   @app.command("upload")
   @handle_errors
   def upload_files(
       ctx: typer.Context,
       path: Annotated[str, typer.Argument(help="Local path to upload")],
   ) -> None:
       """Upload files to a fileset."""
       state: CLIContext = ctx.obj
       client = state.get_client()
       # Custom implementation...
   ```

The `# override-skip` comment strips that line from output (useful for linter satisfaction).

### Templates

Templates use Jinja2. Key variables available:
- `command_name` - CLI command name (e.g., "list")
- `method_name` - SDK method name (e.g., "list")
- `resource_path` - Resource path (e.g., ["customization", "jobs"])
- `params` - List of parameter info
- `docstring` - Method docstring

## Core Utilities

### CLIContext

Access via `ctx.obj`:
```python
state: CLIContext = ctx.obj
client = state.get_client()  # Get SDK client
state.output(data)           # Format and print output
state.verbosity              # 0 or 1
```

### Error Handling

Always use `@handle_errors` decorator:
```python
from nemo_helix_ext.cli.core.errors import handle_errors

@handle_errors
def my_command(ctx: typer.Context) -> None:
    ...
```

### Building API Request Bodies

```python
from nemo_helix_ext.cli.core.api import build_kwargs

body = build_kwargs(
    model="my-model",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,
    temperature=0.7,
)
```

## Testing

Run CLI tests:
```bash
uv run pytest packages/nemo_helix_ext/tests/cli/ -v
```

Run generator tests:
```bash
uv run --package nemo-helix-sdk-tools pytest tools/nemo-helix-sdk-tools/tests/sdk/cli_generator/ -v
```

## Common Tasks

### Add table columns for a new resource

Edit `cli_config.yaml`:
```yaml
- resource: [my_service, my_resource]
  methods:
    list:
      columns:
      - name
      - status
      - created_at
```

Then rebuild: `make update-cli`

### Add a new use-case command

1. Create file in `commands/use_cases/my_command.py`
2. Implement using pattern above
3. Register in `app.py`
4. Vendor: `make vendor-nemo-helix-ext`

### Modify how a command is generated

1. Check which template is used (see `operation_classifier.py`)
2. Edit template in `templates/`
3. Rebuild: `make update-cli`

### Override an auto-generated command

1. Add override config in `cli_config.yaml`
2. Create override file in `overrides/`
3. Rebuild: `make update-cli`

## Troubleshooting

### Typer errors after CLI update

If tests fail with Typer errors after running `make update-cli`, the issue is usually one of:

**1. Unsupported type in Typer**

Typer doesn't support all Python types (e.g., complex unions, nested TypedDicts). Fix by updating the generator code to handle the type:

- Check `type_formatter.py` for type formatting
- Check `typed_dict_utils.py` for TypedDict handling
- Add type conversion or simplification in the relevant context collector

**2. Method miscategorized**

The operation classifier may assign the wrong template (e.g., "default" instead of "create"). Fix via `cli_config.yaml`:

```yaml
- resource: [my_service]
  methods:
    my_method:
      operation_type: create  # Override the auto-detected type
```

Available operation types: `list`, `get`, `create`, `update`, `delete`, `download`, `default`

**3. Skip the method temporarily**

If the fix is non-trivial, skip the method to unblock your work:

```yaml
- resource: [my_service]
  methods:
    problematic_method:
      skip: true  # TODO: Fix...
```

Create a GitLab issue to track the fix, then rebuild: `make update-cli`
