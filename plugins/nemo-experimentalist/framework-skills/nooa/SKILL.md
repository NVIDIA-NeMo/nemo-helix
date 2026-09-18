---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: nooa
description: Build, modify, debug, or optimize agents using NVIDIA-labs OO Agents (NOOA). Use for nooa.Agent subclasses, ellipsis generation methods, CodeAct or Predict strategies, ShellTools, Skill/TextSkill, context and persistence, MCP, tracing, middleware, channels, or trace analysis.
compatibility: Python >= 3.12,<3.14; uv; nooa at the version floor declared by this plugin
allowed-tools: Read
metadata:
  author: NeMo Helix Team <nemo-helix@nvidia.com>
  upstream: https://github.com/NVIDIA-NeMo/labs-OO-Agents
  version: ">=0.0.9"
---
<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NVIDIA-labs OO Agents (NOOA)

Use the `nooa` package and namespace. It comes from PyPI, and the plugin's own
`pyproject.toml` declares the version floor.

For detailed framework guidance, consult the skills at the upstream tag that
matches the floor:
https://github.com/NVIDIA-NeMo/labs-OO-Agents/tree/v0.0.9/skills

Before changing framework behavior, inspect the installed implementation or the
matching upstream skill instead of guessing. In Experimentalist:

- Call `super().__init__()` before attaching tools or skills.
- Use `GuardedShellTools` where workspace access must remain restricted.
- Hide internal skill registries with `spec(self, "skills", hidden=True)`.
- Keep the explicit `orjson==3.11.9` workaround required by nooa's LiteLLM
  tool-call path.

## Verification

Use the repository root `.venv` and the README's standard `uv` workflow. Run:

```bash
uv run --frozen pytest -q <focused migration tests>
uv run --frozen ruff check .
uv run --frozen ruff format --check .
uv run --frozen pyright
```

Then run the changed agent end to end with its configured model.
