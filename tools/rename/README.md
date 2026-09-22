<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NeMo Helix rename HOW-TO

Use these scripts from the repository root to preview, apply, and verify the NeMo Platform to NeMo Helix rename.

## Prerequisites

- Run from a git checkout of this repository, or pass `--repo-dir /path/to/checkout` to target another checkout explicitly.
- Optionally pass repeated `--include-glob` and `--exclude-glob` filters to segment a rename by repo-relative path. Use patterns such as `docs/**`, `packages/**`, `web/**`, `*.md`, and `docker-bake.hcl`.
- Install the standard repository tools, including `git`, `grep`, `sed`, and Python 3.7 or newer.
- Start with a clean worktree before the actual rename. The script refuses to run when `git status --short` is non-empty unless `--continue` is used.
- Review the dry-run output before applying changes.

## 1. Preview the rename

```bash
tools/rename/rename-to-nemo-helix.sh --dry-run
# or, from any directory:
tools/rename/rename-to-nemo-helix.sh --repo-dir /path/to/checkout --dry-run
# or, for a segmented rename:
tools/rename/rename-to-nemo-helix.sh --dry-run --include-glob 'docs/**' --include-glob '*.md'
```

Expected outcome: the command prints the legacy content categories, tracked paths, and first-party published image names that would be renamed. It does not edit files.

## 2. Apply the rename

```bash
tools/rename/rename-to-nemo-helix.sh
# or, from any directory:
tools/rename/rename-to-nemo-helix.sh --repo-dir /path/to/checkout
# or, for a segmented rename:
tools/rename/rename-to-nemo-helix.sh --include-glob 'docs/**' --include-glob '*.md'
```

Expected outcome: the script updates UTF-8 text file contents, renames tracked and newly created non-ignored files, normalizes first-party image names, and prints the verification command to run next. The implementation uses Git's file set rather than walking the whole checkout, so ignored environments such as `.venv/` are not scanned. Acronym-only replacements are intentionally explicit: separated tokens such as `nmp_common`, `nmp-common`, and `NMP_*`; known all-caps/PascalCase code prefixes such as `NMP_BASE_URL`, `NMPJobContext`, `NMPOIDCConfig`, `NmpCliRunner`, and `NmpContext`; and allowlisted lowercase names such as `nmpclient`, `nmpcontext`, `nmpBaseURLEnv`, `nmp-intake`, and `nmp2` are renamed, while `snmp` and larger opaque alphanumeric values are left unchanged.

If a previous rename attempt stopped after making changes, inspect the worktree and then resume the remaining passes with:

```bash
tools/rename/rename-to-nemo-helix.sh --continue
# or, from any directory:
tools/rename/rename-to-nemo-helix.sh --repo-dir /path/to/checkout --continue
```

## 3. Verify the result

```bash
tools/rename/verify-nemo-helix-rename.sh
# or, from any directory:
tools/rename/verify-nemo-helix-rename.sh --repo-dir /path/to/checkout
# or, for the same segmented scope:
tools/rename/verify-nemo-helix-rename.sh --include-glob 'docs/**' --include-glob '*.md'
```

Expected outcome: the verifier prints `No legacy product, package, acronym, path, or first-party image names remain.` and exits with status 0. If it finds a remaining legacy name or unprefixed first-party image, it prints each match and exits non-zero.

## Next Steps

- Review `git diff` carefully, especially generated or vendored files.
- Run the relevant [repository validation](../../CONTRIBUTING.md) and [tests](../../TESTING.md) for the changed areas.
- Commit the reviewed rename with `git commit -s`.
