<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NeMo Helix rename HOW-TO

Use these scripts from the repository root to preview, apply, and verify the NeMo Helix to NeMo Helix rename.

## Prerequisites

- Run from a git checkout of `nemo-helix`.
- Install the standard repository tools, including `git`, `grep`, `sed`, and Python 3.7 or newer.
- Start with a clean worktree before the actual rename. The script refuses to run when `git status --short` is non-empty unless `--continue` is used.
- Review the dry-run output before applying changes.

## 1. Preview the rename

```bash
tools/rename/rename-to-nemo-helix.sh --dry-run
```

Expected outcome: the command prints the legacy content categories, tracked paths, and first-party published image names that would be renamed. It does not edit files.

## 2. Apply the rename

```bash
tools/rename/rename-to-nemo-helix.sh
```

Expected outcome: the script updates file contents, renames tracked and newly created non-ignored files, normalizes first-party image names, and prints the verification command to run next. The implementation uses Git's file set rather than walking the whole checkout, so ignored environments such as `.venv/` are not scanned.

If a previous rename attempt stopped after making changes, inspect the worktree and then resume the remaining passes with:

```bash
tools/rename/rename-to-nemo-helix.sh --continue
```

## 3. Verify the result

```bash
tools/rename/verify-nemo-helix-rename.sh
```

Expected outcome: the verifier prints `No legacy product, package, acronym, path, or first-party image names remain.` and exits with status 0. If it finds a remaining legacy name or unprefixed first-party image, it prints each match and exits non-zero.

## Next Steps

- Review `git diff` carefully, especially generated or vendored files.
- Run the relevant repository validation for the changed areas.
- Commit the reviewed rename with `git commit -s`.
