<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# nemo-platform wheel bundling

> Internal developer docs for the `nemo-platform` wrapper distribution. For an end-user introduction to NeMo Platform, see the [repo root README](../../README.md) or the [PyPI page](https://pypi.org/project/nemo-platform/).

Wrapper distribution for NeMo Platform. When users run `pip install nemo-platform[all]`, this is the wheel they get.

The wheel bundles the SDK, shared runtime packages, default first-party plugins, and services directly from source via hatch force-include. As sub-packages are published independently to PyPI, they'll be removed from the bundle and added as normal dependencies instead.

## How bundling works

All source bundling is configured in `pyproject.toml` via `[tool.bundle-package]`. Each entry declares a package source tree to include in the wheel:

```toml
[tool.bundle-package]
nemo-platform-plugin = { source = "../../packages/nemo_platform_plugin/src/nemo_platform_plugin", module = "nemo_platform_plugin" }
nemo-platform-sdk = { source = "../../sdk/python/nemo-platform/src/nemo_platform", module = "nemo_platform", inherit = { "optional-dependencies" = true, scripts = true } }
nemo-auditor-plugin = { source = "../../plugins/nemo-auditor/src/nemo_auditor", module = "nemo_auditor", inherit = { "entry-points" = ["nemo.*"] } }
nmp-auth = { source = "../../services/core/auth/src/nmp/core/auth", module = "nmp/core/auth", deps_group = "auth-service" }
```

Each entry has:
- **key** — the distribution/package name used to find its `pyproject.toml`, read its metadata, and match wheel `Requires-Dist` entries
- **source** — relative path to the source directory to include in the wheel
- **module** — target module path inside the wheel
- **deps_group** (optional) — name of the `[project.optional-dependencies]` group where the package's transitive deps are written. Defaults to the bundle key.
- **inherit** (optional) — structured metadata to re-export from the bundled package. Supported keys are `scripts`, `entry-points`, and `optional-dependencies`; each value is either `true` or a list of wildcard patterns. An inherited optional dependency includes the bundle's generated base extra unless that package is already a wrapper base dependency.
- **optional-dependencies-prefix** (optional) — prefix applied to inherited optional-dependency names, allowing plugin extras such as `codex` to be exposed as `nemo-agents-plugin-codex` without colliding with other bundled packages.
- **scripts** (optional) — explicit CLI entrypoints to register on the wrapper. Prefer `inherit.scripts` when copying scripts from the bundled package.
- **force_include** (optional) — extra source files, directories, or globs to bundle with this package, keyed relative to the entry's `source` path and mapped to their target wheel path. When using a glob, make the target end in `/` to copy each match into that package directory.

By default, bundled package metadata is not re-exported. `nemo-platform` opts into SDK scripts and SDK optional dependencies, and it opts into only `nemo.*` entry-point groups from the default first-party plugins. Other plugin entry-point groups, such as `data_designer.plugins`, are intentionally not inherited.

Two tools read this config:

### `hatch_build.py` (build hook)

At wheel build time, `hatch_build.py` reads `[tool.bundle-package]` and generates hatch force-include mappings dynamically, including each entry's per-bundle `force_include` mappings. Component-owned runtime assets such as Alembic migrations and Studio UI static files should live on the corresponding bundle entry. The hook still merges any static `[tool.hatch.build.targets.wheel.force-include]` entries when present, but per-bundle `force_include` is preferred for assets owned by a bundled component.

During editable installs (`uv sync`), the build hook does nothing. Workspace packages resolve via their normal editable/workspace installation, so there is no copied bundle and the source dependency graph stays intact.

After the wheel is built, the build hook opens the wheel, rewrites `METADATA`, regenerates `RECORD`, and repacks the wheel. Any `Requires-Dist` whose distribution name matches a `[tool.bundle-package]` key is rewritten to a self-referencing extra using that entry's `deps_group`, or the bundle key when `deps_group` is omitted. Existing requirement extras and environment markers are preserved.

For example, this source dependency:

```text
Requires-Dist: nmp-common
```

becomes this in the final wheel metadata:

```text
Requires-Dist: nemo-platform[nmp-common]
```

A dependency with extras and a marker, such as `nemo-platform-sdk[aiohttp] ; python_version >= "3.11"`, becomes `nemo-platform[nemo-platform-sdk,aiohttp] ; python_version >= "3.11"`.

### `make vendor` (vendor tool)

The vendor tool refreshes auto-generated metadata in workspace pyprojects without disturbing hand-written content. In `[project.optional-dependencies]`, each vendor-owned extra is preceded by a `# Generated from [tool.bundle-package]; do not edit by hand.` marker comment — that marker is the load-bearing signal that distinguishes vendor-owned extras (refreshed or removed by `make vendor`) from hand-written extras (preserved untouched). Generated `[project.scripts]` and `[project.entry-points.*]` tables get a `# Generated from [tool.bundle-package]; do not edit this table by hand.` header marker on the table itself, since those tables are wholly owned by the vendor flow.

The `_process_bundle_packages()` phase in `vendor_package.py` reads `[tool.bundle-package]` from every workspace package that has one. For each entry it:

1. Finds the bundled package's `pyproject.toml`, using the configured `source` path when needed
2. Reads its `[project.dependencies]`
3. Filters out workspace packages that are not bundled by the parent, because they are not installable from PyPI
4. Keeps bundled workspace dependency names readable in source metadata, so the wheel build hook can rewrite final `Requires-Dist` metadata to self-extras
5. Writes the resulting deps into the generated extra named by `deps_group`, or by the bundle key when `deps_group` is omitted
6. Copies only the metadata explicitly selected by `inherit`
7. Writes any explicit `scripts = [...]` declared directly on the bundle entry

For the wrapper specifically, `make vendor` also creates aggregate extras from `[tool.bundle-package]`:

- `core-service` references all core service `*-service` extras.
- `plugins` references all bundled first-party plugin extras whose source lives under `plugins/*/src`.
- `services` references `core-service`, all non-core service `*-service` extras, and `plugins`.

Hand-written extras (e.g. the wrapper's `all` alias, or a plugin's `test` group) live directly in the pyproject's `[project.optional-dependencies]` table without the generator marker. The vendor tool will preserve them on every run; only extras with the `# Generated from [tool.bundle-package]; do not edit by hand.` marker are touched by `make vendor`.

## Dependency groups

The wrapper's `[project.dependencies]` is hand-written with the true workspace dependencies for the base install:

```toml
dependencies = [
  "nemo-platform-sdk",
  "nmp-common",
  "nemo-platform-plugin",
]
```

Those direct workspace dependencies are what editable installs and repo-local tooling see. Wheel builds rewrite them to self-referencing extras so published wheels do not require unpublished workspace packages:

```toml
dependencies = [
  "nemo-platform[nemo-platform-sdk]",
  "nemo-platform[nmp-common]",
  "nemo-platform[nemo-platform-plugin]",
]
```

Service and plugin dependencies are behind optional extras. They are composed via:
- `auth-service`, `entities-service`, etc. — individual service deps
- `core-service` — aggregates all core service `-service` extras
- `plugins` — aggregates default first-party plugin extras
- `services` — aggregates `core-service`, all non-core service `-service` extras, and `plugins`
- `all` — hand-written alias for the full packaged install; expands to `services`. The recommended user-facing extra (`pip install nemo-platform[all]`).

The `services` extra includes `plugins` because Python entry points are distribution-level metadata and are not conditional on extras. If the wrapper publishes plugin `nemo.services` entry points, installing service discovery dependencies must also install the plugin dependencies needed by those entry points.

Vendor-owned extras (those generated from `[tool.bundle-package]`) are marked with a `# Generated from [tool.bundle-package]; do not edit by hand.` comment immediately above the key, and `make vendor` will overwrite them on every run. Extras without the marker are hand-written — add new ones (like `all`) directly in the pyproject and they will be left alone. The wheel rewrite step assumes the generated `deps_group` extras already exist before the build starts.

The wrapper's generated `[project.scripts]` currently re-exports only the SDK CLI entry points:

```toml
[project.scripts]
nemo = "nemo_platform.cli.app:cli"
nmp = "nemo_platform.cli.app:cli"
```

Service-specific server scripts are not exposed by the umbrella `nemo-platform` wheel. Individual service packages may still expose their own scripts, and the wrapper uses `nemo services run` through the platform runner instead.

## Extracting a package to PyPI

To publish a bundled package independently:

1. Remove its entry from `[tool.bundle-package]`
2. Add it as a normal dependency in `[project.dependencies]` (or in the appropriate optional group)
3. Run `make vendor` to regenerate the dependency groups

The wheel gets thinner, the dependency metadata stays correct, and `pip install nemo-platform[all]` (and `[services]`) continues to work.

## Other vendoring (`make vendor`)

The `make vendor` command also handles SDK client extensions (`nemo_platform_ext`, `data_designer_sdk`, `models`, `filesets`, `nemo_evaluator_sdk`). These are **not** bundled via `[tool.bundle-package]` — they use the older `[tool.vendor-package]` mechanism which copies source files into the SDK tree with import rewriting. This is separate from the bundling described above and is only relevant to SDK client-side extensions.
