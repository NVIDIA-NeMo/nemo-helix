<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 🧰 Helper tools for NeMo Helix packages

This directory contains helper tools for building and maintaining the NeMo Helix packages.

## 📦 Package Vendoring Tool

The `nemo-helix-sdk-tools vendor from-config` script copies package modules into the `nemo-helix` SDK.

### Usage

```bash
uv run --frozen nemo-helix-sdk-tools vendor from-config data_designer
```

Requires a `[tool.vendor-package]` section in the package's `pyproject.toml`. For example:

```toml
[tool.vendor-package]
package = "data_designer"
target_sdk_module = "data_designer"
included_paths = ["config/*.py"]
```

### Configuration

**Required:**
- `package`: Package name (valid Python identifier)
- `target_sdk_module`: Target module name (valid Python identifier)

**Optional:**
- `package_root`: Custom package root path
- `with_src`: Include `src/` directory (default: `true`)
- `sdk_optional_dependencies_name`: Name of the optional dependencies in the SDK's `pyproject.toml`. If None, no optional dependencies will be added to the SDK's config. Dependencies are **merged** with any entries already present under that extra. Vendoring finalizes the `services` extra after all package metadata has been merged.
- `included_paths`: List of specific files or directories to include (relative to `package_root`). It supports glob patterns.
- `exluded_dependencies`: List of dependencies in the package that should not be made dependencies in the SDK.
- `vendor_tests`: Whether to vendor the unit tests
- `tests_path`: The path for vendored tests
- `tests_included_paths`: List of specific test files or directories to include
- `included_transitive_dependencies`: List of objects including the dependency name and what files/directories to include from it

### Target Locations

- `sdk/python/nemo-helix/src/nemo_helix/{target_module}/`

### Exclusions

Automatically excludes:
- Hidden files/directories (`.` prefix)
- `__pycache__` directories

**Forbidden target modules:**
The following module names are forbidden because they are already used by the Stainless-generated SDK:
- `types`
- `resources`
- `lib`
