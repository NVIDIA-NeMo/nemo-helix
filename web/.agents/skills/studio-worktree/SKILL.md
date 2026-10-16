---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: studio-worktree
description: Prepare the current git worktree for Studio web/ development — copies missing .env.*.local files and runs pnpm install. Use when the user is inside a newly-created linked worktree and wants to bring it to a runnable state. Does NOT create/remove worktrees — use your tool's built-in worktree features for that (Claude Code --worktree, Cursor, git worktree add, etc.).
metadata:
  author: NeMo Helix Team <nemo-helix@nvidia.com>
---

# Studio Worktree Setup

Brings the **current linked worktree** to a runnable state for Studio web development. Worktree creation is handled by external tooling — this skill only performs the follow-up steps that tools don't know about.

## What it does

1. Verifies the cwd is inside a **linked** worktree (not the main checkout). Bails otherwise.
2. Copies any missing `.env.*.local` files from the main worktree's `web/packages/*/env/` directories into the matching locations in the current worktree. Only packages that exist in the current worktree get touched (stale dirs in main are ignored).
3. Runs `pnpm install --frozen-lockfile` from `web/` so every workspace package (studio, common, etc.) has its `node_modules`.
4. Reports that the worktree is ready and tells the user to `pnpm dev`.

## Steps

1. Run the script from anywhere inside the worktree — resolve its path via `git rev-parse --show-toplevel` so the cwd doesn't matter:

   ```bash
   bash "$(git rev-parse --show-toplevel)/web/.agents/skills/studio-worktree/setup-worktree.sh"
   ```

2. If the script reports the user is in the main checkout, stop and tell them to create a linked worktree first (via their tool — e.g., Claude Code's `--worktree`, Cursor, or `git worktree add <path>`).

3. When the script finishes cleanly, surface the copy-pasteable command from its final line — the `cd <worktree>/web && pnpm dev` one-liner — so the user can start Studio without having to assemble the path themselves.

## Notes

- **Never read `.env.*.local` contents** — they contain credentials. The script copies them via `cp` without reading.
- Existing `.env.*.local` files in the current worktree are not overwritten.
