---
name: nmp-cli
description: Use when developing the NeMo Platform CLI - adding or changing command groups, wiring a service's typed client into commands, output formatting, pagination, error mapping, or the `--output-format code` snippet generator. Covers core groups in nemo_platform_ext and plugin-hosted `nemo.cli` groups.
---

# NeMo Platform CLI Development

The CLI is implemented with Typer in `packages/nemo_platform_ext/src/nemo_platform_ext/cli/`. Every
command talks to the platform through the typed clients in `packages/nemo_platform_plugin`
(`nemo_platform_plugin.<area>.client`). The CLI has **no dependency on the generated `nemo_platform`
(Stainless) SDK**; `packages/nemo_platform_ext/tests/cli/test_stainless_boundary.py` enforces this
statically and by running the CLI with `nemo_platform` un-importable. Never add a `from nemo_platform`
import under `cli/`.

## Command Categories

| Category | Location | Description |
|----------|----------|-------------|
| **Core resource groups** | `commands/<group>.py` (`files`, `inference/`, `jobs`, `models`, `secrets`, `workspaces`, hidden `adapters`, `iam`, `projects`) | Hand-written on typed clients, registered in `commands/manifest_registry.py` |
| **Plugin-hosted groups** | owning package, `nemo.cli` entry point (`guardrail` → `plugins/nemo-guardrails`, `intake`/`experiments` → `services/intake`) | Appear only when the package is installed |
| **Setup / use cases** | `commands/setup.py`, `commands/use_cases/`, `commands/auth.py`, `commands/config.py` | Wizards and workflows (`chat`, `wait`, ...) |
| **Services** | `commands/services/`, `commands/quickstart/` | Run the platform locally (imports server packages by design) |

## Directory Structure

```
packages/nemo_platform_ext/src/nemo_platform_ext/cli/
├── app.py                    # Entry point, global options, lazy manifest registration
├── manifest.py               # TopLevelEntry, panel order
├── core/
│   ├── context.py            # CLIContext: get_client() -> NemoClient, typed_client(XClient)
│   ├── pagination.py         # collect_offset_pages / collect_cursor_pages / warn_if_more_pages
│   ├── formatters.py         # format_output (unwraps NemoResponse), Column, table/json/yaml/csv
│   ├── errors.py             # handle_errors: typed-client error hierarchy -> exit codes
│   ├── code_generator.py     # --output-format code -> typed-client Python snippet
│   └── waiters.py            # --wait / --watch helpers (deployments, jobs, gateway readiness)
└── commands/
    ├── manifest_registry.py  # TOP_LEVEL_ENTRIES: every built-in group/command
    ├── secrets.py            # REFERENCE port: copy its shape for new groups
    └── ...
```

## Running the CLI During Development

```bash
uv run _nemo --help                 # runs from packages/nemo_platform_ext, no vendoring needed
make update-cli                     # vendor into sdk/python/nemo-platform + regenerate reference docs
```

## Adding or Changing a Command Group

Use `commands/secrets.py` and `tests/cli/commands/test_secrets.py` as the template.

### Pattern

```python
from __future__ import annotations

from typing import Annotated

import typer
from nemo_platform_plugin.secrets.client import SecretsClient
from nemo_platform_plugin.secrets.types import ListSecretsQueryParams, PlatformSecretCreateRequest

from nemo_platform_ext.cli.core.api import build_kwargs
from nemo_platform_ext.cli.core.code_generator import handle_code_generation
from nemo_platform_ext.cli.core.context import CLIContext
from nemo_platform_ext.cli.core.errors import handle_errors
from nemo_platform_ext.cli.core.formatters import Column, format_output
from nemo_platform_ext.cli.core.help_formatter import collect_warnings, create_typer_app
from nemo_platform_ext.cli.core.pagination import PaginationType, collect_offset_pages, warn_if_more_pages
from nemo_platform_ext.cli.core.types import EntityOutputFormatOption, ListOutputFormatOption

app = create_typer_app(name="secrets", help="Manage secrets.")


@app.command("get")
@collect_warnings
@handle_errors
def retrieve_secrets(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    output_format: EntityOutputFormatOption = None,
) -> None:
    """Retrieve a secret by its name."""
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)

    kwargs = build_kwargs(name=name, workspace=workspace)
    if handle_code_generation(SecretsClient, "get_secret", kwargs, resolved_output_format, state):
        return

    result = state.typed_client(SecretsClient).get_secret(name=name, workspace=workspace)
    format_output(result, is_list=False, output_format=resolved_output_format, ...)


@app.command("list")
@collect_warnings
@handle_errors
def list_secrets(ctx: typer.Context, ..., all_pages: bool = False) -> None:
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)
    query_params: ListSecretsQueryParams = {...}   # omit keys the user did not set
    if handle_code_generation(SecretsClient, "list_secrets", kwargs, resolved_output_format, state, result="list"):
        return
    response = state.typed_client(SecretsClient).list_secrets(workspace=workspace, query_params=query_params)
    items = collect_offset_pages(response, all_pages=all_pages)
    format_output(items, is_list=True, output_columns=[Column("name", None), ...], ...)
    if not all_pages:
        warn_if_more_pages(items, PaginationType.PAGE_NUMBER)
```

Conventions that keep the surface consistent:
- `state.typed_client(XClient)` derives the service client from the CLI's `NemoClient` (shared auth
  and transport). Never construct clients from config inside a command.
- Name the resolved output format `resolved_output_format` (ty otherwise narrows the option Literal).
- Request bodies are the typed request models from `nemo_platform_plugin.<area>.types`, built only from
  fields the user provided (`model.model_copy(update={...})` for optionals) so the wire payload has no
  spurious nulls. Query params are the `TypedDict`s in `types.py`.
- Single-entity responses go straight into `format_output` (it unwraps `NemoResponse`).
- `handle_code_generation(XClient, "<method>", kwargs, ..., result="entity"|"list"|"none"|"binary")`
  renders the `-f code` snippet; pass the same kwargs you call the client with.
- Errors: let `nemo_platform_plugin.client.errors` propagate; `@handle_errors` maps them (404 → exit 3,
  missing workspace → exit 2, client-side pydantic validation → exit 2).

### Register the group

Core group: add a `TopLevelEntry` to `commands/manifest_registry.py` (`help` must equal the Typer app's
help string; `tests/cli/test_app.py::test_manifest_help_matches_loaded_manual_entry` checks this).

Plugin-hosted group: subclass `nemo_platform_plugin.cli.NemoCLI` in the owning package and register it
under `[project.entry-points."nemo.cli"]` in that package's `pyproject.toml` (see
`plugins/nemo-guardrails/src/nemo_guardrails_plugin/cli.py`). Run `uv sync --frozen --all-packages` so
the entry point is installed. A plugin group with the same name as a built-in core group replaces it.

### Missing typed endpoint

If the typed client lacks a route, add it to `nemo_platform_plugin/<area>/{endpoints,types,client}.py`
mirroring the server's FastAPI route exactly, with tests under `packages/nemo_platform_plugin/tests/<area>/`.

## Testing

Three layers, all required for a new group:

1. **Wire-level unit tests** (`tests/cli/commands/test_<group>.py`): drive the real Typer app with a
   `CLIContext(_client=NemoClient(... http_client=httpx.Client(transport=httpx.MockTransport(recorder))))`
   and assert method, path, query string, exact JSON body, stdout, `--all-pages`, error mapping, and that
   `-f code` sends nothing and emits typed-client code.
2. **Integration tests** (`tests/cli/integration/test_<group>.py`): the `runner` fixture injects a
   `NemoClient` backed by the in-process ASGI app from `nmp.testing`; add the service class to
   `create_test_client(...)` in `tests/cli/integration/conftest.py` if it is not hosted yet.
3. **Boundary**: add the group's `--help` and a `-f code` invocation to `RUNTIME_COMMANDS` in
   `tests/cli/test_stainless_boundary.py`.

```bash
uv run --frozen pytest packages/nemo_platform_ext/tests/cli -q
uv run --frozen pytest packages/nemo_platform_plugin/tests/<area> -q
```

## Troubleshooting

- **Group help mismatch**: the manifest `help` and the Typer app `help` must be identical.
- **"Missing workspace" exit 2 when a workspace is configured**: the command passed `workspace=None`
  to an endpoint whose path has no `{workspace}` default; pass the option through unchanged and let the
  client fill its default.
- **`-f code` snippet does not compile**: the kwargs contain a value `_render_value` cannot render;
  extend `core/code_generator.py` (it handles pydantic models, `RootModel`, enums, `SecretStr`
  masking, datetimes, dicts, lists) and add a `compile()` test in `tests/cli/core/test_code_generator.py`.
- **Command needs a Stainless helper**: it does not; find the equivalent on the typed client or in
  `packages/filesets/src/filesets/transfer.py` (fileset upload/download/list/delete).
