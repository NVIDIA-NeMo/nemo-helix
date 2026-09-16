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
- `core/` - Shared utilities: error handling, output formatting, input parsing, pagination, CLIContext
- `commands/` - Command implementations:
  - `config.py` - kubectl-style config management
  - `quickstart/` - local deployment commands
  - `use_cases/` - high-level commands like `chat`
  - `api/` - auto-generated API commands (do not edit)

## Local Development Shortcut

For rapid CLI iteration, run `_nhx` to execute the CLI directly from `packages/nemo_helix_ext` without vendoring.

```shell
uv run _nhx --help
```

This is useful for testing new CLI changes before running `make vendor-nemo-helix-ext`.

## Auto-Generated Code

**IMPORTANT:** Files in `src/nemo_helix_ext/cli/commands/api/` are auto-generated.
- Do NOT manually edit these files
- Do NOT include in code reviews
- Generated from templates in `<ROOT>/tools/nemo-helix-sdk-tools/src/nemo_helix_sdk_tools/sdk/cli_generator/templates/`

### Build

To build the CLI run this command:
```shell
make update-cli
```

It includes all 3 steps.

#### Step 1.
The CLI is built with the `nemo-helix-sdk-tools generate-cli` command. The build process uses these inputs:
- The SDK (`sdk/python/nemo-helix`) - it introspects the SDK and creates a command for each SDK operation.
  - The CLI structure follows the SDK structure (based on `sdk/stainless.yaml`). `sdk.customization.jobs.list` becomes `nhx customization jobs list`.
- The CLI config (`tools/nemo-helix-sdk-tools/src/nemo_helix_sdk_tools/sdk/cli_generator/cli_config.yaml`) - configures some aspects of the CLI generation (default columns for list operations, methods/resources to skip, etc.)
- Templates + code inside of the CLI generator (`tools/nemo-helix-sdk-tools/src/nemo_helix_sdk_tools/sdk/cli_generator/`)

After the generation is done, we run ruff for formatting and to ensure correctness (e.g. there are no missing imports).

This step can be run with:
```shell
make generate-cli-commands
```

#### Step 2.
Once the CLI is generated, we vendored it into the `sdk/python/nemo-helix` package. This way we bundle the SDK and the CLI together and the user needs to only install a single package.

In a nutshell, the vendoring process copies the code and updates all the imports.

This step can be run with:
```shell
make vendor-nemo-helix-ext
```

Note: this vendors all the extensions, not just the CLI.

#### Step 3.
We generate CLI reference for our documentation.

This step can be run with:
```shell
make generate-cli-reference-docs
```

---

See [README.md](README.md) for usage and configuration.
