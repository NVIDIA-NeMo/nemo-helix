---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

description: Generate config reference markdown from service config classes
---
# Generate Config Reference Docs

Generate the NeMo Helix configuration reference documentation from service config classes and PlatformConfig.

**Run from repository root:**

```bash
uv run generate-config-docs
```

This writes:
- `docs/set-up/config-reference.md` — Markdown with YAML sections and inline comments (description, default, possible values)

Optionally, pass `--output-to-file` to also write a standalone example YAML file (default path: `packages/nhx_platform/config/example-config.yaml`, or supply a path).

**Options:**

| Option | Description |
|--------|-------------|
| `--output-to-file [PATH]` | Also write standalone example YAML; PATH defaults to `packages/nhx_platform/config/example-config.yaml` |
| `--output-dir PATH` | Write markdown under `PATH` (keeps default filename) |
| `--markdown PATH` | Custom path for the markdown file |
| `--help` | Show all options |

**Examples:**

```bash
uv run generate-config-docs --output-to-file
uv run generate-config-docs --output-to-file /tmp/example-config.yaml
uv run generate-config-docs --markdown docs/set-up/config-reference.md
uv run generate-config-docs --output-dir /tmp/nhx-config-docs
```

**Alternative (run script file with uv):**

```bash
uv run python script/generate_config_docs.py
```

The script discovers configs under `services/core/*/src/.../config.py` and `services/*/src/.../config.py`, and includes `PlatformConfig` from `packages/nhx_common`. See `packages/nhx_platform/config/local.yaml` for a minimal working example.
