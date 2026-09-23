<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# SDK Code — Do Not Edit Directly

Code in `sdk/python/nemo-helix/` is **generated and vendored**. Do not edit it directly — your changes will be overwritten.

## How the SDK Is Built

The SDK is assembled from two sources:

1. **Stainless** — Generates low-level SDK code (API clients, types, resources) from the OpenAPI spec.
2. **Vendored client-side extensions** — Code from `packages/` is copied into the SDK with import rewriting (`nemo_helix_ext.X` → `nemo_helix.X`, etc.).

Only 6 client-side extension packages are file-vendored into the SDK. Runtime/server packages (services, `nhx_common`, `nemo_helix_plugin`, etc.) are **not** vendored into the SDK — they are bundled into the `nemo-helix` wrapper wheel via force-include from source.

## Vendored Client-Side Extensions

| Package | Source Location |
|---|---|
| `nemo_helix_ext` | `packages/nemo_helix_ext/` |
| `data_designer_sdk` | `packages/data_designer_sdk/` |
| `models` | `packages/models/` |
| `filesets` | `packages/filesets/` |
| `nemo_evaluator_sdk` | `packages/nemo_evaluator_sdk/` |

## Build Commands

| Command | What It Does |
|---|---|
| `make update-sdk` | Full SDK update (regenerate OpenAPI spec + Stainless + vendor) |
| `make vendor` | Vendor client extensions into SDK + generate wrapper metadata |
| `make vendor-nemo-helix-ext` | Vendor just the `nemo_helix_ext` package |
| `make refresh-openapi` | Regenerate `openapi/openapi.yaml` from API definitions |
| `make stainless` | Push spec to Stainless and pull generated code |

## Workflow

To change SDK behavior that comes from vendored packages:

1. Edit the source in `packages/<package_name>/`
2. Run `make vendor` (or the specific vendor command)
3. Verify the vendored output in `sdk/python/nemo-helix/`

For CLI development specifically, you can run `_nhx` directly from source to test changes without vendoring first:

```bash
uv run _nhx --help
```

`_nhx` uses `packages/nemo_helix_ext` directly, so use vendoring when you need to validate the SDK-vendored copy.

See `sdk/stainless.sh` for the Stainless generation flow.
