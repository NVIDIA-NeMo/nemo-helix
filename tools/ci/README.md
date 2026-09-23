<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# CI Test Selection Audit

`test_selection_audit.py` is a report-only helper for checking whether CI jobs gated by `.github/actions/changes/action.yaml` include the paths that can change the tests they run.

## Prerequisites

- `uv`
- A checkout of this repository
- The repository root as the working directory

Run it from the repository root:

```bash
uv run --frozen python tools/ci/test_selection_audit.py
```

For machine-readable output:

```bash
uv run --frozen python tools/ci/test_selection_audit.py --format json
```

The script intentionally exits `0` even when it finds gaps. It is meant for periodic review before becoming a lint or CI gate. The first pass is conservative: it parses workflow jobs, change-filter aliases, `uv` package/group options, pytest paths, direct Python scripts/modules, simple Make targets, and Python imports under collected test paths.

## Next Steps

Review the related [changes action](../../.github/actions/changes/action.yaml) when updating path filters.
