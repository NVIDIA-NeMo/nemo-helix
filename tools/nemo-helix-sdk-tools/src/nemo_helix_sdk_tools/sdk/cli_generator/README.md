<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# CLI Code Generation

Generates CLI commands from SDK introspection for the NeMo Helix.

## Overview

The CLI generation system uses a **command-by-command** approach:
1. **Reads** Stainless config to discover resources and methods
2. **Introspects** SDK at generation time to extract parameter details and types
3. **Generates** CLI commands using templates
4. **Combines** commands into resource files

## Usage

Generate CLI commands:

```bash
uv run --frozen nemo-helix-sdk-tools generate-cli
```

## Architecture

<!-- TODO: Update module list once architecture is stable -->

### Design Principles

Following principles from CLAUDE.local.md:

1. **Simplicity over defensive programming**
   - Fail fast on bad state vs guarding edge cases
   - Use specific exceptions (`TypeError`, `NameError`) not bare `except Exception:`
   - Skip unnecessary None checks when types guarantee validity

2. **Work with actual Type objects**
   - Use `Type` objects from `get_type_hints()`, not string representations
   - Never string match like `"str" in type_str.lower()`
   - Never parse strings like `"Union[str, int]"`
   - Use `evaluated_type` attribute with real Type objects

3. **Simple code generation**
   - OK to include unused imports (linters remove them)
   - Focus on correctness over minimal output

## Type Handling

### Always Use Type Objects

```python
# ✅ GOOD - Use actual Type object
from typing import get_origin, get_args, Union

origin = get_origin(tp)
if origin is Union:
    for arg in get_args(tp):
        if arg is str:  # Direct type comparison
            ...

# ❌ BAD - String matching
type_str = str(tp)
if "str" in type_str.lower():  # Fragile!
    ...
```

### TypedDictField.evaluated_type

`TypedDictField` has two type representations:
- `type_annotation` (Any) - Legacy string/raw annotation **(avoid)**
- `evaluated_type` (Type) - Actual resolved Type object **(use this!)**

```python
field = TypedDictField(...)

# ✅ GOOD - Use evaluated_type
if field.evaluated_type is not None:
    origin = get_origin(field.evaluated_type)
    ...

# ❌ BAD - String matching
if "str" in str(field.type_annotation).lower():
    ...
```

## Common Patterns

### Format Types

```python
from nemo_helix_sdk_tools.sdk.cli_generator.type_formatter import (
    format_type,           # For CLI signatures
    format_type_for_help,  # For help text (simplified)
    get_type_schema,       # For complex types
)

type_str = format_type(str | None)  # "str | None"
help_str = format_type_for_help(dict[str, int])  # "dict[str, int]"
schema = get_type_schema(MyTypedDict)  # "{field1: str, field2: int}"
```

### Parse Docstrings

```python
from nemo_helix_sdk_tools.sdk.cli_generator.docstring_parser import ParsedDocstring

parsed = ParsedDocstring.parse(docstring)
print(parsed.description)
print(parsed.param_descriptions["limit"])
```

### Transform Help Text

```python
from nemo_helix_sdk_tools.sdk.cli_generator.docstring_parser import transform_query_to_cli

description = "Use `?search[name]=foo` to search"
cli_help = transform_query_to_cli(description, "search")
# Result: "Use `--search.name foo` to search"
```

### Work with TypedDicts

```python
from nemo_helix_sdk_tools.sdk.cli_generator.typed_dict_utils import (
    is_explodable_typed_dict,
    introspect_typed_dict,
)

if is_explodable_typed_dict(param.type_annotation):
    fields = introspect_typed_dict(typed_dict_class)
    for field in fields:
        if field.is_simple_cli_type:  # Uses actual type checking!
            print(f"Simple field: {field.name}")
```

## Command Overrides

For commands that require custom implementation (e.g., file uploads, streaming), you can override the generated code with a hand-written implementation.

### Configuration

In `sdk/cli_config.yaml`, specify an override path for the method:

```yaml
- resource: [filesets]
  methods:
    upload_file:
      override: filesets/upload_file.py
```

### Override File Structure

Create the override file in `tools/nemo-helix-sdk-tools/src/nemo_helix_sdk_tools/sdk/cli_generator/overrides/`:

```
overrides/
└── filesets/
    └── upload_file.py
```

The override file should contain:
1. **Imports** at the top (will be merged with other imports in the generated file)
2. **Command function** with `@app.command()` and `@handle_errors` decorators

Example override file:

```python
from pathlib import Path
from typing import Annotated, Any, cast

import typer

from nemo_helix.filesets import FilesetFileSystem
from nemo_helix_ext.cli.core.context import CLIContext
from nemo_helix_ext.cli.core.errors import handle_errors

app = cast(Any, None)  # override-skip: provided by generated file


@app.command("upload-file")
@handle_errors
def upload_file_filesets(
    ctx: typer.Context,
    path: Annotated[Path, typer.Argument(help="Path to file to upload")],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    name: Annotated[str, typer.Option("--name")] = ...,
) -> None:
    """Upload file content to a fileset."""
    state: CLIContext = ctx.obj
    client = state.get_client()
    # Custom implementation...
```

### The `# override-skip` Comment

Lines containing `# override-skip` are stripped from the override file before inclusion. This allows you to add placeholder definitions that satisfy linters and enable IDE features, without polluting the generated output.

Common use case - defining `app` for linting to pass:
```python
app = cast(Any, None)  # override-skip: provided by generated file
```

### How It Works

1. During generation, `CLIConfig.get_method_override()` checks if an override exists
2. If found, the override file is read instead of rendering a template
3. Imports are extracted and merged with other imports in the generated file
4. The command code is included alongside other commands for that resource

### Notes

- The `app` variable is defined in the generated file, not the override
- Multiline imports (with parentheses) are supported
- Override files are not standalone runnable - they're templates included in generated output

## Testing

Run tests:
```bash
uv run pytest tests/cli_generator/ -v
```

All 15 tests should pass.

## Anti-Patterns

### ❌ String-Based Type Detection

```python
# BAD
if "str" in type_str.lower():
    return "str"

# GOOD
if tp is str:
    return "str"
```

### ❌ Bare Exception Handlers

```python
# BAD
try:
    hints = get_type_hints(cls)
except Exception:
    pass

# GOOD
try:
    hints = get_type_hints(cls)
except (NameError, TypeError):
    hints = {}
```

### ❌ String Parsing

```python
# BAD
if "Union[" in type_str:
    parse_union(type_str)

# GOOD
if get_origin(tp) is Union:
    args = get_args(tp)
```
