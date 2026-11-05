<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# nemo-helix wheel bundling

> Internal developer docs for the `nemo-helix` wrapper distribution. For an end-user introduction to NeMo Helix, see the [repo root README](../../README.md) or the [PyPI page](https://pypi.org/project/nemo-helix/).

Wrapper distribution for NeMo Helix. When users run `pip install nemo-helix[all]`, this is the wheel they get.

The wheel bundles the CLI (`nemo_helix_ext`), the typed clients (`nemo_helix_plugin`), shared runtime packages, default first-party plugins, and services directly from source via hatch force-include. As sub-packages are published independently to PyPI, they are removed from the bundle, listed under `[tool.bundle-package-published]`, and added as normal dependencies instead.

## How bundling works

All source bundling is configured in `pyproject.toml` via `[tool.bundle-package]`. Each entry declares a package source tree to include in the wheel:

```toml
[tool.bundle-package]
nemo-helix-plugin = { source = "../../packages/nemo_helix_plugin/src/nemo_helix_plugin", module = "nemo_helix_plugin" }
nemo-helix-ext = { source = "../../packages/nemo_helix_ext/src/nemo_helix_ext", module = "nemo_helix_ext", inherit = { "entry-points" = ["nemo.*"] }, scripts = [{ name = "nemo", value = "nemo_helix_ext.cli.app:cli" }] }
nemo-garak-plugin = { source = "../../plugins/nemo-garak/src/nemo_garak", module = "nemo_garak", inherit = { "entry-points" = ["nemo.*"] } }
nhx-auth = { source = "../../services/core/auth/src/nhx/core/auth", module = "nhx/core/auth", deps_group = "auth-service" }
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

By default, bundled package metadata is not re-exported. `nemo-helix` declares the public `nemo` and `nhx` scripts on the `nemo-helix-ext` bundle entry (the source package registers them under private names so the workspace venv only exposes one `nemo`), and it opts into only `nemo.*` entry-point groups from the CLI and the default first-party plugins. Other plugin entry-point groups, such as `data_designer.plugins`, are intentionally not inherited.

Two tools read this config:

### `hatch_build.py` (build hook)

At wheel build time, `hatch_build.py` reads `[tool.bundle-package]` and generates hatch force-include mappings dynamically, including each entry's per-bundle `force_include` mappings. Component-owned runtime assets such as Alembic migrations and Studio UI static files should live on the corresponding bundle entry. The hook still merges any static `[tool.hatch.build.targets.wheel.force-include]` entries when present, but per-bundle `force_include` is preferred for assets owned by a bundled component.

During editable installs (`uv sync`), the build hook does nothing. Workspace packages resolve via their normal editable/workspace installation, so there is no copied bundle and the source dependency graph stays intact.

After the wheel is built, the build hook opens the wheel, rewrites `METADATA`, regenerates `RECORD`, and repacks the wheel. Any `Requires-Dist` whose distribution name matches a `[tool.bundle-package]` key is rewritten to a self-referencing extra using that entry's `deps_group`, or the bundle key when `deps_group` is omitted. Existing requirement extras and environment markers are preserved.

For example, this source dependency:

```text
Requires-Dist: nhx-common
```

becomes this in the final wheel metadata:

```text
Requires-Dist: nemo-helix[nhx-common]
```

With extras, `nhx-evals-sdk[harbor]` becomes `nemo-helix[nhx-evals-sdk,harbor]`.

### `make vendor` (vendor tool)

The vendor tool refreshes auto-generated metadata in workspace pyprojects without disturbing hand-written content. In `[project.optional-dependencies]`, each vendor-owned extra is preceded by a `# Generated from [tool.bundle-package]; do not edit by hand.` marker comment — that marker is the load-bearing signal that distinguishes vendor-owned extras (refreshed or removed by `make vendor`) from hand-written extras (preserved untouched). Generated `[project.scripts]` and `[project.entry-points.*]` tables get a `# Generated from [tool.bundle-package]; do not edit this table by hand.` header marker on the table itself, since those tables are wholly owned by the vendor flow.

The `_process_bundle_packages()` phase in `vendor_package.py` reads `[tool.bundle-package]` from every workspace package that has one. For each entry it:

1. Finds the bundled package's `pyproject.toml`, using the configured `source` path when needed
2. Reads its `[project.dependencies]`
3. Filters out workspace packages that are not bundled by the parent, because they are not installable from PyPI. Members listed under `[tool.bundle-package-published]` release to PyPI on their own version line and stay regular requirements.
4. Keeps bundled workspace dependency names readable in source metadata, so the wheel build hook can rewrite final `Requires-Dist` metadata to self-extras
5. Writes the resulting deps into the generated extra named by `deps_group`, or by the bundle key when `deps_group` is omitted
6. Copies only the metadata explicitly selected by `inherit`
7. Writes any explicit `scripts = [...]` declared directly on the bundle entry

For the wrapper specifically, `make vendor` also creates aggregate extras from `[tool.bundle-package]`:

- `core-service` references all core service `*-service` extras.
- `plugins` references all bundled first-party plugin extras whose source lives under `plugins/*/src`.
- `services` references `core-service`, all non-core service `*-service` extras, and `plugins`.

Hand-written extras (e.g. the wrapper's `all` alias, or a plugin's `test` group) live directly in the pyproject's `[project.optional-dependencies]` table without the generator marker. The vendor tool will preserve them on every run; only extras with the `# Generated from [tool.bundle-package]; do not edit by hand.` marker are touched by `make vendor`.

## Native OPA provisioning

The embedded PDP runs native OPA as a child managed by the auth service. The
universal `py3-none-any` wheel contains the Python resolver and Rego sources,
**not a platform-specific executable**. No Go/Python binding or custom native
wheel build pipeline is needed. Installing the wheel does not download OPA;
resolution occurs when the embedded PDP starts.

`nhx.core.auth.app.embedded_pdp.opa_binary.resolve_opa_binary()` resolves:

1. `auth.opa_binary_path` (`NHX_AUTH_OPA_BINARY_PATH`), when configured. An
   invalid explicit path fails startup rather than falling back to a download.
2. A compatible executable named `opa` on `PATH` (operator-provisioned).
   Stale/incompatible PATH binaries are skipped in favor of pinned provisioning.
3. A checksum-validated cache entry beneath
   `${XDG_CACHE_HOME:-~/.cache}/nemo-helix/opa/1.22.0/<asset>/`.
4. The official **v1.22.0** GitHub release asset and its published `.sha256`.

Automatic provisioning supports Linux amd64/arm64 static assets and macOS
amd64/arm64 assets. Other hosts require an operator-provisioned executable.
Explicit/PATH executables are trusted operator inputs: the resolver checks
executability and probes `opa version` with a five-second timeout, requiring
**>=1.22.0,<2** for the validated managed runtime flags. It does not verify their release
provenance. Provision the pinned version and validate it in your environment.
An incompatible explicit path fails startup rather than silently falling back.

Downloads use HTTPS, a 10-second socket timeout, a 60-second streaming budget,
a 256 MiB binary limit, and a 1 KiB checksum limit. A per-cache exclusive lock
has a separate 60-second acquisition budget. Files are staged privately, verified,
and atomically replaced under that lock; cache directories and executables are
owner-only. Cache hits rehash the binary against the saved official checksum,
so they work offline. The cache filesystem must permit execution.

**Supply-chain limitation:** the release version is pinned, but the SHA256 is
fetched over TLS from the same official release, not pinned in source or verified
with a signature. The saved checksum detects cache corruption, not a malicious
same-user rewrite of both files. Never substitute locally built binary hashes
for official release hashes. During implementation, GitHub release downloads
returned HTTP 503 and the release API returned 403; official hashes and live
first-run provisioning could not be verified. Verify all supported official
assets/checksums before release; consider committing their trusted hashes.

The dependency lock and license artifacts have been regenerated without a
Rego Python binding. Because the canonical DataDesignerPlugins index was
unreachable, uv resolved through a temporary mirror built by upstream's own
index generator from its official manifest and authentic release artifacts.
Only the two mirror registry/index URL fields were restored to the unchanged
canonical index; dependency versions, edges and hashes were generated by uv.
`uv lock --offline --check` passes. See
[NATIVE_OPA_LOCK_PROVENANCE.md](NATIVE_OPA_LOCK_PROVENANCE.md) for verified inputs
and reproduction. This removes the obsolete binding from frozen installation;
it does **not** establish that an entire Docker image or its official OPA asset
can be fetched/built in the current restricted network.

The core/API Dockerfiles copy `/opa` from the official
`openpolicyagent/opa:1.22.0-static` image into `/usr/local/bin/opa`, avoiding
runtime downloads. Both Dockerfiles pin its immutable multi-platform index digest:
`sha256:906bfc9904b2693733e7959d71773dd073e6c135cf06f26d955abe94603e82b3`.
`docker manifest inspect` and `docker buildx imagetools inspect` verified that
index and its Linux amd64/arm64 entries. The pinned image also ran successfully
with `--network none`; its extracted Linux amd64 `/opa` passed native runtime
tests. ARM64 execution was not tested (manifest verification only). This does
not establish a successful full NeMo core/API image build. Wheel release-asset
downloads remain a separate check. No custom Go rebuild is performed.
For offline wheel installs, provision OPA on PATH or configure an absolute
`auth.opa_binary_path` before starting auth. A network/cache failure fails startup
with provisioning guidance; there is no unverified fallback.

## Dependency groups

The wrapper's `[project.dependencies]` is hand-written with the true workspace dependencies for the base install:

```toml
dependencies = [
  "nemo-helix-ext",
  "nhx-common",
  "nemo-helix-plugin",
]
```

Those direct workspace dependencies are what editable installs and repo-local tooling see. Wheel builds rewrite them to self-referencing extras so published wheels do not require unpublished workspace packages:

```toml
dependencies = [
  "nemo-helix[nemo-helix-ext]",
  "nemo-helix[nhx-common]",
  "nemo-helix[nemo-helix-plugin]",
]
```

Service and plugin dependencies are behind optional extras. They are composed via:
- `auth-service`, `entities-service`, etc. — individual service deps
- `core-service` — aggregates all core service `-service` extras
- `plugins` — aggregates default first-party plugin extras
- `services` — aggregates `core-service`, all non-core service `-service` extras, and `plugins`
- `all` — hand-written alias for the full packaged install; expands to `services`. The recommended user-facing extra (`pip install nemo-helix[all]`).

The `services` extra includes `plugins` because Python entry points are distribution-level metadata and are not conditional on extras. If the wrapper publishes plugin `nemo.services` entry points, installing service discovery dependencies must also install the plugin dependencies needed by those entry points.

Vendor-owned extras (those generated from `[tool.bundle-package]`) are marked with a `# Generated from [tool.bundle-package]; do not edit by hand.` comment immediately above the key, and `make vendor` will overwrite them on every run. Extras without the marker are hand-written — add new ones (like `all`) directly in the pyproject and they will be left alone. The wheel rewrite step assumes the generated `deps_group` extras already exist before the build starts.

The wrapper's generated `[project.scripts]` currently exposes only the CLI entry points:

```toml
[project.scripts]
nemo = "nemo_helix_ext.cli.app:cli"
nhx = "nemo_helix_ext.cli.app:cli"
```

Service-specific server scripts are not exposed by the umbrella `nemo-helix` wheel. Individual service packages may still expose their own scripts, and the wrapper uses `nemo services run` through the platform runner instead.

## Extracting a package to PyPI

To publish a bundled package independently:

1. Remove its entry from `[tool.bundle-package]`
2. List it under `[tool.bundle-package-published]` so bundled packages can keep depending on it
3. Add it as a normal dependency in `[project.dependencies]` if the base install needs it
4. Run `make vendor` to regenerate the dependency groups

The wheel gets thinner, the dependency metadata stays correct, and `pip install nemo-helix[all]` (and `[services]`) continues to work.
