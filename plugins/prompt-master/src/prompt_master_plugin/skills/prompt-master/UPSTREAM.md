<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Prompt Master upstream

The `SKILL.md` and `references/` files in this directory are vendored from:

- Repository: https://github.com/nidhinjs/prompt-master
- Revision: `2bd92518e26bf659e21e3d9ab90573fcf3ddeccb`
- Version: `1.8.0`
- License: MIT (see `LICENSE`)

The `references/` files and the body of `SKILL.md` are byte-identical to
upstream. The only local change is in the `SKILL.md` frontmatter, which
carries the two fields this repository requires of every skill
(`allowed-tools` and `metadata.author`; see `docs/contributing/skills-spec.mdx`).
Re-apply them when refreshing the snapshot.

NeMo-specific one-shot execution instructions are supplied by
`prompt_master_plugin.runner`; keeping them outside the vendored files
makes the upstream boundary explicit.
