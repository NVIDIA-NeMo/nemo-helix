<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Garak API

This package provides a small subset of the [garak](https://github.com/NVIDIA/garak) library which does not require a full garak install.

It is useful for listing and validating plugins and probespecs from a pre-built plugin-cache.

# installation

```bash
uv sync --package garak_api
```

# upgrading

The package contains several files copied from the corresponding version of the garak repo, including the plugin cache json
and several `.py` files that access the cache. The `__init__.py` script bridges the gap between the downloaded python
files and the NeMo Helix side of the API.

To upgrade the API and plugin cache run `./scripts/update_plugin_cache.sh`. Run tests in `./tests/` after upgrade to confirm
that the newly downloaded `*.py` files play nicely with the rest of the package. There are only a few interactions,
so the tests should be comprehensive. If any tests fail, `__init__.py` probably needs to be updated to work with the
new `*.py` files.

Generally, we should only upgrade garak to a release version, rather than `*pre`.

# tests

Under the `./tests` directory -

```bash
$ pytest
```
