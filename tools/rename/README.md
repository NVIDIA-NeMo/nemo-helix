<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NeMo Helix rename HOW-TO

## Reusable package and plugin renames

`rename_packages.py` accepts a JSON profile with a required `package` section and
an optional `library` section. Use `library` for a companion SDK; omit it for a
plugin with no separate library. The existing Platform-to-Helix scripts below
remain a separate workflow. The new tool needs Git and Python 3.10 or newer and
uses only the Python standard library.

The Evals profile implements these mappings:

| Surface | Before | After |
| --- | --- | --- |
| Product | NeMo Evaluator | NeMo Helix Evals |
| Plugin distribution | nemo-evaluator-plugin | nemo-evals-plugin |
| Plugin directory | plugins/nemo-evaluator | plugins/nemo-evals |
| Implementation module | nemo_evaluator | nemo_evals |
| CLI and service | evaluator | evals |
| API prefix | /apis/evaluator/ | /apis/evals/ |
| Typed client module | nemo_helix_plugin.evaluator | nemo_helix_plugin.evals |
| Typed client property | client.evaluator | client.evals |
| Library distribution | nemo-evaluator-sdk | nhx-evals-sdk |
| Library module and directory | nemo_evaluator_sdk | nhx_evals_sdk |

Permission namespaces (`evaluator.*`), authorization scope, entity type names,
stored task-kind discriminators and the three job source tags stay unchanged.
The profile explicitly pins row-evaluation's source to `nemo-evaluator`, since
otherwise changing the implementation module would change the derived source.

Preview, apply, and verify from the repository root:

```bash
tools/rename/rename-to-nemo-evals.sh --dry-run
tools/rename/rename-to-nemo-evals.sh
tools/rename/rename-to-nemo-evals.sh --verify
```

All invocations accept `--repo-dir /path/to/checkout`. Apply requires a clean
worktree unless `--allow-dirty` is explicitly supplied. Inspect existing changes
before using that option. Re-running apply resumes a partial rename; it does not
stage files or commit. The wrapper resolves its profile relative to itself, so
it can target a different checkout without copying the scripts there.

For another plugin, create a profile and invoke:

```bash
uv run --frozen --no-sync python tools/rename/rename_packages.py \
  --profile /path/to/profile.json --dry-run
```

Minimal profile, without a companion library:

```json
{
  "name": "Example plugin rename",
  "exclude": ["tools/rename/**", "tests/tools/rename/**"],
  "package": {
    "replacements": {
      "old_plugin": "new_plugin",
      "plugins/old-plugin": "plugins/new-plugin"
    },
    "paths": {
      "plugins/old-plugin/src/old_plugin": "plugins/new-plugin/src/new_plugin",
      "plugins/old-plugin": "plugins/new-plugin"
    }
  }
}
```

Each section supports:

- `replacements`: literal content mappings, applied together with longest matches
  first. Library and package mappings share this pass, preventing a shorter
  plugin module name from swallowing its SDK name.
- `paths`: repository-relative file or directory prefix mappings. The longest
  matching prefix wins. Include nested module directories explicitly; content
  mappings and path mappings are independent.
- `rules`: ordered regex content rules with `pattern`, `replacement` (Python
  regex replacement syntax), and optional `include` / `exclude` glob lists.
  These run after literal mappings. Rules match the file's path before it moves.

Top-level `exclude` globs protect fixtures, profiles and other intentional old
names. `notes` prints follow-up requirements in preview, apply and verification.
`--include-glob` and `--exclude-glob` further restrict files using the same
semantics as the existing rename tools. After paths move, use globs covering the
new locations when verifying segmented work.

The tool scans tracked and non-ignored untracked files, preflights destination
collisions before editing, skips symlinks, and preserves binary contents while
moving their paths. It removes only empty source directories. It does not follow
symlink targets or rewrite serialized binary artifacts. Profiles should be
idempotent: `--verify` fails when another application would change any selected
file or path, and succeeds once the configured transformations are exhausted.
It does not prove runtime compatibility or detect names absent from the profile.

After applying Evals, review the diff, regenerate lockfiles with `uv`, run
`make update-sdk`, and validate library imports, packaging, plugin discovery,
CLI, API routes and authorization. Generated files receive mechanical edits;
they still need regeneration from their authoritative sources. Review service
configuration, UI consumers and external integration references as well.

Include this limitation in the eventual MR description: existing cloudpickle
metric bundles and compiled job specifications may reference the removed Python
module names. This rename supplies no compatibility aliases or data migrations
for those artifacts.

Run the new tool's isolated integration tests without bootstrapping the platform:

```bash
uv run --frozen --no-sync python -m unittest discover \
  -s tests/tools/rename -p test_package_rename.py -v
```

## Platform-to-Helix workflow

Use these scripts from the repository root to preview, apply, and verify the NeMo Helix to NeMo Helix rename.

## Prerequisites

- Run from a git checkout of this repository, or pass `--repo-dir /path/to/checkout` to target another checkout explicitly.
- Optionally pass repeated `--include-glob` and `--exclude-glob` filters to segment a rename by repo-relative path. Use patterns such as `docs/**`, `packages/**`, `web/**`, `*.md`, and `docker-bake.hcl`.
- Install the standard repository tools, including `git`, `grep`, `sed`, and Python 3.7 or newer.
- Start with a clean worktree before the actual rename when possible. The script refuses to run when `git status --short` is non-empty unless `--continue` or `--allow-dirty` is used.
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

Expected outcome: the script updates UTF-8 text file contents, renames tracked and newly created non-ignored files, normalizes first-party image names, and prints the verification command to run next. The implementation uses Git's file set rather than walking the whole checkout, so ignored environments such as `.venv/` are not scanned. Acronym-only replacements are intentionally explicit: separated tokens such as `nhx_common`, `nhx-common`, and `NHX_*`; known all-caps/PascalCase code prefixes such as `NHX_BASE_URL`, `NHXJobContext`, `NHXOIDCConfig`, `NhxCliRunner`, and `NhxContext`; and allowlisted lowercase names such as `nhxclient`, `nhxcontext`, `nhxBaseURLEnv`, `nhx-intake`, and `nhx2` are renamed, while `snmp` and larger opaque alphanumeric values are left unchanged. Bare `Platform` is renamed only inside CamelCase/code identifiers such as `HelixJobStep`, `CreateHelixJobRequest`, and `AsyncCustomizationHelixClients`; standalone prose `Platform` is left for human review.

If a previous rename attempt stopped after making changes, inspect the worktree and then resume the remaining passes with:

```bash
tools/rename/rename-to-nemo-helix.sh --continue
# or, from any directory:
tools/rename/rename-to-nemo-helix.sh --repo-dir /path/to/checkout --continue
# or, when intentionally applying on top of existing changes:
tools/rename/rename-to-nemo-helix.sh --allow-dirty
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
