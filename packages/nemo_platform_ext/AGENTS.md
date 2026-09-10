<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# CLI Development

The CLI is developed in the `src/nemo_platform_ext/cli` module. 

## Guidelines

The CLI is implemented using `Typer` CLI framework.
All the operations that the CLI provides fall in one of these categories:
- **API** - auto-generated commands that call an API implemented by the platform. These commands use the SDK under the hood to make calls and they expose API inputs as CLI options (aka flags).
- **Configuration** - commands for managing the configuration that is used by the API and other commands.
- **Use-case** - commands that encapsulate a common use case, e.g. chat with an LLM using platform's inference gateway.
- **Quickstart** - commands for managing the quickstart deployment of the platform. Quickstart runs the platform on user's machine for quick evaluation, prototyping and POC.

## Structure

- `app.py` - Entry point, command registration, global options (`--context`, `--base-url`, `--output-format`)
- `core/` - Shared utilities: error handling, output formatting, input parsing, pagination, CLIContext
- `commands/` - Command implementations:
  - `config.py` - kubectl-style config management
  - `quickstart/` - local deployment commands
  - `use_cases/` - high-level commands like `chat`
  - `api/` - auto-generated API commands (do not edit)

## Local Development Shortcut

For rapid CLI iteration, run `_nmp` to execute the CLI directly from `packages/nemo_platform_ext` without vendoring.

```shell
uv run _nmp --help
```

This is useful for testing new CLI changes before running `make vendor-nemo-platform-ext`.

## CLI Command Groups

Every `nemo <group> *` command is hand-written Python on the typed clients in
`nemo_platform_plugin` (for example `commands/secrets.py` uses `SecretsClient`).
There is no code generator and the CLI has no dependency on the generated
`nemo_platform` (Stainless) SDK; `tests/cli/test_stainless_boundary.py` enforces
this and runs the CLI with `nemo_platform` un-importable.

- Core resource groups (`files`, `inference`, `jobs`, `models`, `secrets`, `workspaces`, and the hidden
  `adapters`, `iam`, `projects`) live in `src/nemo_platform_ext/cli/commands/` and are registered in
  `commands/manifest_registry.py`.
- Functional groups ship with the package that owns the service as `nemo.cli` entry points
  (`guardrail` in `plugins/nemo-guardrails`, `intake` and `experiments` in `services/intake`), so they
  appear only when that package is installed.
- Commands obtain a service client with `state.typed_client(<Client>)`; `--output-format code`
  renders the typed-client call via `cli/core/code_generator.py`.

Use `commands/secrets.py` and `tests/cli/commands/test_secrets.py` as the reference when adding a group:
mirror the structure, add wire-level tests (real Typer app over a recorded `httpx.MockTransport`) and,
when the service can be hosted by `nmp.testing`, in-process integration tests.

### Build

To vendor the CLI into the distribution package and regenerate its reference docs:
```shell
make update-cli
```

It includes 2 steps.

#### Step 1.
Once the CLI is generated, we vendored it into the `sdk/python/nemo-platform` package. This way we bundle the SDK and the CLI together and the user needs to only install a single package.

In a nutshell, the vendoring process copies the code and updates all the imports.

This step can be run with:
```shell
make vendor-nemo-platform-ext
```

Note: this vendors all the extensions, not just the CLI.

#### Step 2.
We generate CLI reference for our documentation.

This step can be run with:
```shell
make generate-cli-reference-docs
```

---

See [README.md](README.md) for usage and configuration.
