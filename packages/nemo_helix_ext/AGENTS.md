<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# CLI Development

The CLI is developed in the `src/nemo_helix_ext/cli` module.

## Guidelines

The CLI is implemented using `Typer` CLI framework.
All the operations that the CLI provides fall in one of these categories:
- **API** - auto-generated commands that call an API implemented by the platform. These commands use the SDK under the hood to make calls and they expose API inputs as CLI options (aka flags).
- **Configuration** - commands for managing the configuration that is used by the API and other commands.
- **Use-case** - commands that encapsulate a common use case, e.g. chat with an LLM using platform's inference gateway.
- **Quickstart** - commands for managing the quickstart deployment of the platform. Quickstart runs the platform on user's machine for quick evaluation, prototyping and POC.

## Structure

- `app.py` - Entry point, command registration, global options (`--context`, `--base-url`, `--output-format`)
- `core/` - CLI host internals: `CLIContext` (the concrete CLI state), help rendering (`NhxGroup`), lazy loading, waiters.
  The helpers commands are built from (output formatting, options, pagination, error handling, input parsing,
  `-f code`) live in `nemo_helix_plugin` (`cli`, `cli_state`, `cli_options`, `cli_output`, ...) so plugin
  commands share them; the old `core/` module paths re-export them for existing core commands.
- `commands/` - Command implementations:
  - `config.py` - kubectl-style config management
  - `quickstart/` - local deployment commands
  - `use_cases/` - high-level commands like `chat`
  - `api/` - auto-generated API commands (do not edit)

## Local Development Shortcut

For rapid CLI iteration, run `_nhx` to execute the CLI directly from `packages/nemo_helix_ext`.

```shell
uv run _nhx --help
```

The `nemo-helix` wheel bundles this package from source and publishes the public `nemo` and `nhx` scripts (see `packages/nemo_helix/BUNDLING.md`); `_nhx` runs the same code without going through the wheel.

## CLI Command Groups

Every `nemo <group> *` command is hand-written Python on the typed clients in
`nemo_helix_plugin` (for example `commands/secrets.py` uses `SecretsClient`).
There is no code generator, and no module in this package depends on the generated
`nemo_helix` (Stainless) SDK; `tests/cli/test_stainless_boundary.py` enforces this and
runs the CLI with `nemo_helix` un-importable.

- Core resource groups (`files`, `inference`, `jobs`, `models`, `secrets`, `workspaces`, and the hidden
  `adapters`, `iam`, `projects`) live in `src/nemo_helix_ext/cli/commands/` and are registered in
  `commands/manifest_registry.py`.
- Functional groups ship with the package that owns the service as `nemo.cli` entry points
  (`guardrail` in `plugins/nemo-guardrails`, `intake` and `experiments` in `services/intake`), so they
  appear only when that package is installed.
- Commands obtain a service client with `cli_state(ctx).typed_client(<Client>)`; `--output-format code`
  renders the typed-client call via `nemo_helix_plugin.cli_codegen`. See the `nhx-cli` skill.

Use `commands/secrets.py` and `tests/cli/commands/test_secrets.py` as the reference when adding a group:
mirror the structure, add wire-level tests (real Typer app over a recorded `httpx.MockTransport`) and,
when the service can be hosted by `nhx.testing`, in-process integration tests.

### Build

To regenerate the CLI reference docs:
```shell
make update-cli
```

The `nemo-helix` wheel picks up CLI changes automatically at build time; there is no vendoring step. If you change this package's dependencies or `nemo.*` entry points, run `make vendor` to refresh the wheel metadata in `packages/nemo_helix/pyproject.toml`.

---

See [README.md](README.md) for usage and configuration.
