<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Generated SDK: Do Not Edit or Extend

`sdk/python/nemo-helix/` holds the legacy generated `nemo_helix` package. It is scheduled for deletion and is not regenerated. Do not edit it, add to it, or add new imports of `nemo_helix`.

Use the typed clients in `packages/nemo_helix_plugin` (`NemoClient` / `AsyncNemoClient` and the per-service clients such as `FilesClient`, `ModelsClient`, `JobsClient`) instead, and migrate any remaining callers to them.

## How the wheel is built

The published `nemo-helix` wheel is defined by `packages/nemo_helix/pyproject.toml`. It bundles the `nemo` CLI (`packages/nemo_helix_ext`), the typed clients, runtime packages, plugins, and services from source at build time; see `packages/nemo_helix/BUNDLING.md`. Nothing in the wheel definition reads this directory's `pyproject.toml`. The `nemo_helix` module is bundled through a bare `[tool.bundle-package.nemo-helix-sdk]` entry only while runtime packages still import it.

## Commands

| Command | What It Does |
|---|---|
| `make vendor` | Refresh the `nemo-helix` wheel metadata generated from `[tool.bundle-package]` |
| `make update-cli` | Regenerate the CLI reference docs |
| `make refresh-openapi` | Regenerate `openapi/openapi.yaml` from API definitions |
| `make update-sdk` | All of the above plus the TypeScript web SDK |

For CLI development, run the CLI straight from source:

```bash
uv run _nhx --help
```
