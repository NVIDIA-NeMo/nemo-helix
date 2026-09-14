<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Prompt Master upstream

The `SKILL.md` and `references/` files in this directory are vendored without
modification from:

- Repository: https://github.com/nidhinjs/prompt-master
- Revision: `2bd92518e26bf659e21e3d9ab90573fcf3ddeccb`
- Version: `1.8.0`
- License: MIT (see `LICENSE`)

NeMo-specific one-shot execution instructions are supplied by
`prompt_master_plugin.runner`; keeping them outside the vendored files
makes the upstream boundary explicit.
