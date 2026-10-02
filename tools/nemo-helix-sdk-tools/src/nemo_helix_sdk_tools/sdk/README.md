<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# SDK Maintenance Tools

This package contains repo-local commands for keeping the `nemo-helix` wheel metadata and license metadata in sync.

## Wrapper bundle metadata

Refresh the generated extras, scripts, and entry points of the `nemo-helix` wrapper from `[tool.bundle-package]`:

```sh
uv run --no-sync nemo-helix-sdk-tools vendor bundle-metadata
```

## License headers

Refresh SPDX license headers:

```sh
uv run --no-sync nemo-helix-sdk-tools post-generation update-license-headers
```

Prefer the Makefile targets (`make vendor` and `make update-sdk`) for normal repo workflows.
