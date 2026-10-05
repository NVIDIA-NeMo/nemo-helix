---
name: nhx-cli
description: Use when developing the NeMo Helix CLI - adding or changing command groups, wiring a service's typed client into commands, output formatting, pagination, error mapping, or the `--output-format code` snippet generator. Covers core groups in nemo_helix_ext and plugin-hosted `nemo.cli` groups.
---

# NeMo Helix CLI Development

The `nemo` CLI host (entry point, global options, config, auth, lazy group loading) lives in
`packages/nemo_helix_ext/src/nemo_helix_ext/cli/`. The helpers commands are built from live in
`packages/nemo_helix_plugin` so core and plugin commands share them:

| Module (`nemo_helix_plugin.`) | Provides |
|---|---|
| `cli` | `NemoCLI` (plugin base class), `create_typer_app`, `HELP_OPTION_NAMES` |
| `cli_state` | `CLIState` protocol, `cli_state(ctx)`, `resolve_output_format`, `resolve_cli_workspace` |
| `cli_options` | `WorkspaceOption`, `ListOutputFormatOption`, `EntityOutputFormatOption`, `NoTruncateOption`, `OutputColumnsOption`, `StreamOutputOption`, `AllPagesOption`, format Literals |
| `cli_output` | `format_output` (unwraps `NemoResponse`), `Column`, `check_output_columns_with_format`, `is_tty` |
| `cli_pagination` | `collect_offset_pages`, `collect_cursor_pages`, `warn_if_more_pages`, `PaginationType` |
| `cli_warnings` | `collect_warnings`, `add_warning` |
| `cli_error_handling` | `handle_errors`: typed-client errors -> messages and exit codes |
| `cli_input` | `build_request_body`, `read_data_input_with_flags`, `read_payload`, `validate_required_fields` |
| `cli_kwargs` | `build_kwargs`, `merge_filter_dict` |
| `cli_codegen` | `handle_code_generation` for `--output-format code` |

Every command talks to the platform through the typed clients in `nemo_helix_plugin`
(`nemo_helix_plugin.<area>.client`, or a plugin's own `client.py`). The CLI has **no dependency on the
generated `nemo_helix` (Stainless) SDK**; `packages/nemo_helix_ext/tests/cli/test_stainless_boundary.py`
enforces this. Never add a `from nemo_helix` import under `cli/` or in plugin CLI code.

Plugin and service code must **not import `nemo_helix_ext`** — plugins only depend on
`nemo_helix_plugin`.

## Command Categories

| Category | Location | Description |
|----------|----------|-------------|
| **Core resource groups** | `commands/<group>.py` (`files`, `inference/`, `jobs`, `models`, `secrets`, `workspaces`, hidden `adapters`, `iam`, `projects`) | Hand-written on typed clients, registered in `commands/manifest_registry.py` |
| **Plugin-hosted groups** | owning package, `nemo.cli` entry point (`guardrail` → `plugins/nemo-guardrails`, `intake`/`experiments` → `services/intake`, `insights`, `agents`, `auditor`, ...) | Appear only when the package is installed |
| **Generated job/function verbs** | `nemo_helix_plugin/commands.py` | `submit`/`explain`/`run` for every `NemoJob`/`NemoFunction` a plugin registers |
| **Setup / use cases** | `commands/setup.py`, `commands/use_cases/`, `commands/auth.py`, `commands/config.py` | Wizards and workflows (`chat`, `wait`, ...) |
| **Services** | `commands/services/`, `commands/quickstart/` | Run the platform locally (imports server packages by design) |

## Directory Structure

```
packages/nemo_helix_ext/src/nemo_helix_ext/cli/
├── app.py                    # Entry point, global options (--base-url, --context, --output-format), lazy registration
├── manifest.py               # TopLevelEntry, panel order
├── core/
│   ├── context.py            # CLIContext: the concrete CLIState (config, auth, typed_client(XClient))
│   ├── help_formatter.py     # NhxGroup/NhxCommand: NeMo help rendering, applied to every mounted command
│   ├── lazy_load.py          # Loads core and plugin groups on demand
│   └── waiters.py            # --wait / --watch helpers (deployments, jobs, gateway readiness)
└── commands/
    ├── manifest_registry.py  # TOP_LEVEL_ENTRIES: every built-in group/command
    ├── secrets.py            # REFERENCE port for core groups
    └── ...

plugins/example-plugin/src/nemo_example_plugin/cli.py   # REFERENCE for plugin groups
```

## Running the CLI During Development

```bash
uv run _nemo --help                 # runs from packages/nemo_helix_ext
make update-cli                     # regenerate the CLI reference docs
```

## Adding or Changing a Command Group

Templates: `commands/secrets.py` + `tests/cli/commands/test_secrets.py` for core groups,
`plugins/example-plugin/src/nemo_example_plugin/cli.py` + its `tests/test_cli.py` for plugin groups.

### Pattern

```python
from __future__ import annotations

from typing import Annotated

import typer
from nemo_helix_plugin.cli import create_typer_app
from nemo_helix_plugin.cli_codegen import handle_code_generation
from nemo_helix_plugin.cli_error_handling import handle_errors
from nemo_helix_plugin.cli_options import (
    AllPagesOption,
    EntityOutputFormatOption,
    ListOutputFormatOption,
    WorkspaceOption,
)
from nemo_helix_plugin.cli_output import Column, format_output
from nemo_helix_plugin.cli_pagination import PaginationType, collect_offset_pages, warn_if_more_pages
from nemo_helix_plugin.cli_state import cli_state, resolve_cli_workspace, resolve_output_format
from nemo_helix_plugin.cli_warnings import collect_warnings
from nemo_helix_plugin.secrets.client import SecretsClient
from nemo_helix_plugin.secrets.types import ListSecretsQueryParams

app = create_typer_app(name="secrets", help="Manage secrets.")


@app.command("get")
@collect_warnings
@handle_errors
def retrieve_secrets(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: WorkspaceOption = None,
    output_format: EntityOutputFormatOption = None,
) -> None:
    """Retrieve a secret by its name."""
    state = cli_state(ctx)
    resolved_output_format = resolve_output_format(ctx, output_format)

    kwargs = {"name": name, "workspace": resolve_cli_workspace(ctx, workspace)}
    if handle_code_generation(SecretsClient, "get_secret", kwargs, resolved_output_format, state):
        return

    format_output(state.typed_client(SecretsClient).get_secret(**kwargs), output_format=resolved_output_format)


@app.command("list")
@collect_warnings
@handle_errors
def list_secrets(
    ctx: typer.Context,
    workspace: WorkspaceOption = None,
    all_pages: AllPagesOption = False,
    output_format: ListOutputFormatOption = None,
) -> None:
    state = cli_state(ctx)
    resolved_output_format = resolve_output_format(ctx, output_format)
    query_params: ListSecretsQueryParams = {...}   # omit keys the user did not set
    kwargs = {"workspace": resolve_cli_workspace(ctx, workspace), "query_params": query_params}
    if handle_code_generation(
        SecretsClient, "list_secrets", kwargs, resolved_output_format, state,
        result="all-pages" if all_pages else "list",
    ):
        return
    items = collect_offset_pages(state.typed_client(SecretsClient).list_secrets(**kwargs), all_pages=all_pages)
    format_output(items, is_list=True, output_format=resolved_output_format, output_columns=[Column("name"), ...])
    if not all_pages:
        warn_if_more_pages(items, PaginationType.PAGE_NUMBER)
```

Existing core commands still import some of these from their old `nemo_helix_ext.cli.core.*` paths,
which re-export them. New code imports from `nemo_helix_plugin`.

Conventions that keep the surface consistent:
- **Client**: `cli_state(ctx).typed_client(XClient)` derives the service client from the CLI's shared
  `NemoClient` (base URL, auth, token refresh). Never construct clients from config or env inside a
  command, and never add a per-command `--base-url` or `--cluster`: the platform comes from the global
  `nemo --base-url` / `nemo --context`.
- **Groups**: build every group with `create_typer_app(...)`, not `typer.Typer(...)`. Under `nemo` the host
  also renders it with `NhxGroup`.
- **Options**: use the shared aliases (`WorkspaceOption`, `ListOutputFormatOption`, ...). Output format is
  always `--output-format` / `--output` / `-f`; do not add other spellings or reuse `-f` for another
  option. Resolve with `resolve_output_format(ctx, output_format)` and name the result
  `resolved_output_format` (ty otherwise narrows the option Literal).
- **Every `list`/`get` command** takes `--output-format` and prints with `format_output`.
- **Request bodies** are the typed request models from `nemo_helix_plugin.<area>.types`. Build them from
  user input with `build_request_body(Model, data)` (rejects unknown fields; only provided fields are sent),
  or from only the fields the user set. Query params are the `TypedDict`s in `types.py`.
- **Output**: single-entity responses go straight into `format_output` (it unwraps `NemoResponse`).
  Entities parsed by typed clients keep their `id`/`created_at`/... metadata. List commands pass default
  `output_columns` so tables are readable.
- **`-f code`**: `handle_code_generation(XClient, "<method>", kwargs, ..., result="entity"|"list"|"all-pages"|"none"|"binary")`;
  pass the same kwargs you call the client with. Reject options the snippet cannot express (for example
  `--wait` with `-f code`) with a usage error rather than printing a misleading snippet.
- **Errors**: let `nemo_helix_plugin.client.errors` propagate; `@handle_errors` maps them (API errors,
  including 404 and connection failures → exit 3; usage and client-side validation → exit 2). Raise
  `click.BadParameter` for bad input and `click.ClickException` for expected failures.
- **Warnings**: decorate commands with `@collect_warnings` so `add_warning` (pagination, truncation) and
  agent-mode hints are printed.

### Register the group

Core group: add a `TopLevelEntry` to `commands/manifest_registry.py` (`help` must equal the Typer app's
help string; `tests/cli/test_app.py::test_manifest_help_matches_loaded_manual_entry` checks this).

Plugin-hosted group: subclass `nemo_helix_plugin.cli.NemoCLI` in the owning package and register it
under `[project.entry-points."nemo.cli"]` in that package's `pyproject.toml` (see
`plugins/example-plugin`). Run `uv sync --frozen --all-packages` so the entry point is installed. A plugin
group with the same name as a built-in core group replaces it.

### Missing typed endpoint

If the typed client lacks a route, add it to `nemo_helix_plugin/<area>/{endpoints,types,client}.py`
(or the plugin's own `types/endpoints.py` + `client.py`) mirroring the server's FastAPI route exactly,
with tests. Keep plugin clients in a module that does not import the generated SDK, so the CLI can use
them.

## Testing

Required for a new or changed group:

1. **Wire-level unit tests**: drive the real Typer app against a typed client over
   `httpx.MockTransport` and assert method, path, query string, exact JSON body, stdout, `--all-pages`,
   error mapping, and that `-f code` sends nothing and emits typed-client code.
   - Core groups (`tests/cli/commands/test_<group>.py`): pass
     `CLIContext(_client=NemoClient(... http_client=httpx.Client(transport=httpx.MockTransport(recorder))))`.
   - Plugin groups: pass a stand-in `CLIState` whose `typed_client` builds the client over the mock
     transport, or mount the app under an `NhxGroup` root with the real `CLIContext` to test what users see.
2. **Integration tests** (core, `tests/cli/integration/test_<group>.py`): the `runner` fixture injects a
   `NemoClient` backed by the in-process ASGI app from `nhx.testing`; add the service class to
   `create_test_client(...)` in `tests/cli/integration/conftest.py` if it is not hosted yet.
3. **Boundary**: add a core group's `--help` and a `-f code` invocation to `RUNTIME_COMMANDS` in
   `tests/cli/test_stainless_boundary.py`.

```bash
uv run --frozen pytest packages/nemo_helix_ext/tests/cli -q
uv run --frozen pytest packages/nemo_helix_plugin/tests -q
```

## Troubleshooting

- **Group help mismatch**: the manifest `help` and the Typer app `help` must be identical.
- **"Missing workspace" exit 2 when a workspace is configured**: the command passed `workspace=None`
  to an endpoint whose path has no `{workspace}` default; resolve it with `resolve_cli_workspace(ctx, workspace)`.
- **`-f code` snippet does not compile**: the kwargs contain a value `_render_value` cannot render;
  extend `nemo_helix_plugin/cli_codegen.py` (it handles pydantic models, `RootModel`, enums, `SecretStr`
  masking, datetimes, dicts, lists) and add a `compile()` test in `packages/nemo_helix_plugin/tests/test_cli_codegen.py`.
- **`RuntimeError: No NeMo Helix CLI state on this context`**: a test invoked the command without
  `obj=`; pass a CLI state.
- **Command needs a Stainless helper**: it does not; find the equivalent on the typed client or in
  `packages/filesets/src/filesets/transfer.py` (fileset upload/download/list/delete).
