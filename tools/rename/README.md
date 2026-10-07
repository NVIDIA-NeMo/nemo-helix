<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Plugin rename tools

## Reusable plugin renames

`rename_plugins.py` accepts a JSON profile with a required `plugin` section and
an optional `library` section. Use `library` for a companion SDK; omit it for a
plugin with no separate library. The tool needs Git and Python 3.10 or newer and
uses only the Python standard library. The existing Platform-to-Helix scripts
below remain a separate workflow.

Preview, apply, and verify a plugin rename from the repository root:

```bash
uv run --frozen --no-sync python tools/rename/rename_plugins.py \
  --profile /path/to/profile.json --dry-run
uv run --frozen --no-sync python tools/rename/rename_plugins.py \
  --profile /path/to/profile.json
uv run --frozen --no-sync python tools/rename/rename_plugins.py \
  --profile /path/to/profile.json --verify
```

All invocations accept `--repo-dir /path/to/checkout`. Apply requires a clean
worktree unless `--allow-dirty` is explicitly supplied. Inspect existing changes
before using that option. Re-running apply resumes a partial rename; it does not
stage files or commit. A plugin-specific shell wrapper can resolve its profile
relative to itself and pass the remaining arguments to `rename_plugins.py`.

Minimal profile, without a companion library:

```json
{
  "name": "Example plugin rename",
  "exclude": ["tools/rename/**", "tests/tools/rename/**"],
  "plugin": {
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

To rename a companion library, add a `library` section using the same structure:

```json
"library": {
  "replacements": {
    "old_plugin_sdk": "new_sdk",
    "old-plugin-sdk": "new-sdk"
  },
  "paths": {
    "packages/old_plugin_sdk/src/old_plugin_sdk": "packages/new_sdk/src/new_sdk",
    "packages/old_plugin_sdk": "packages/new_sdk"
  }
}
```

Each section supports:

- `replacements`: literal content mappings, applied together with longest matches
  first. Library and plugin mappings share this pass, preventing a shorter
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
collisions before editing, remaps symlink paths and repo-local targets, and preserves binary contents while
moving their paths. It removes only empty source directories. It does not follow
symlinks when reading file contents or rewrite serialized binary artifacts. Profiles should be
idempotent: `--verify` fails when another application would change any selected
file or path, and succeeds once the configured transformations are exhausted.
It does not prove runtime compatibility or detect names absent from the profile.

Before applying a rename, decide which identifiers should change: product names,
plugin distributions, modules, CLI groups, API prefixes, configuration names and
typed clients. Treat permission namespaces, entity type names, persisted job
sources and task-kind discriminators as separate compatibility decisions. A job
source derived from a module name may need an explicit override if that module
is renamed while existing jobs must retain their source.

The Evals profile renames job sources to `nemo-evals`,
`nemo-evals.agent-evaluate` and `nemo-evals.retrieve-eval`, together with job
registrations and source filters in consumers such as Studio. Row evaluation
derives its new source from the renamed `nemo_evals` module. Permission names
and the authorization scope remain `evaluator` / `evaluator.*`; entity type
names and task-kind discriminators also remain unchanged. Existing stored jobs
are not migrated, so jobs with the old sources no longer appear in lists
filtered by the new sources. This is an intentional breaking change.

The profile also moves `docs/evaluator` to `docs/evals` and updates Fern source
paths. Troubleshooting moves to the explicit `evals` navigation slug, with redirects
from the previous `evaluator` URL; other published navigation slugs are preserved.
It renames the web SDK service
configuration, generation scripts, OpenAPI tags, and corresponding consumer
imports together; regenerate the ignored SDK output before building Studio.
`Evaluator` classes, local `evaluator` variables, telemetry fields, and existing
permission namespaces are still valid names, so a raw count of `evaluator`
matches is not a completeness check. Ignored files such as Python bytecode caches
are left untouched and can keep old source directories on disk after all tracked
files have moved. Use `rg` to search source without ignored caches.

After applying a rename, review the diff, regenerate lockfiles with `uv`, run
`make update-sdk` when API or SDK surfaces change, and validate library imports,
packaging, plugin discovery, CLI, API routes and authorization. Generated files
receive mechanical edits; regenerate them from their authoritative sources.
Review service configuration, UI consumers and external integrations as well.

Document compatibility limitations in the rename's MR: serialized metric bundles
and compiled job specifications may reference removed Python modules. The generic
tool does not supply compatibility aliases or data migrations for those artifacts.

Run the tool's isolated integration tests without bootstrapping the platform:

```bash
uv run --frozen --no-sync python -m unittest discover \
  -s tests/tools/rename -p test_plugin_rename.py -v
```

Evals tests read fixed pre-rename snapshots from `tests/tools/rename/fixtures/evals`
instead of live product files. The profile excludes this directory, so the tests
retain their original inputs and remain enabled after the real rename. A
regression applies the rename in a disposable checkout and reruns both snapshot
tests with the old plugin directory removed.

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
