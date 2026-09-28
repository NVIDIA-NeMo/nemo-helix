<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# SDK Maintenance Tools

This package contains repo-local commands for keeping the `nemo-helix` wheel metadata, vendored packages, and license metadata in sync.

## Wrapper bundle metadata

Refresh the generated extras, scripts, and entry points of the `nemo-helix` wrapper from `[tool.bundle-package]`:

```sh
uv run --no-sync nemo-helix-sdk-tools vendor bundle-metadata
```

## SDK Vendoring

Vendor configured platform packages into the checked-in Python SDK tree and refresh the wrapper metadata:

```sh
uv run --no-sync nemo-helix-sdk-tools vendor all-from-configs \
  nemo_helix_ext models filesets nemo_evaluator_sdk
```

Run post-generation updates:

```sh
uv run --no-sync nemo-helix-sdk-tools post-generation update-license-headers
```

Prefer the Makefile targets (`make vendor` and `make update-sdk`) for normal repo workflows.
