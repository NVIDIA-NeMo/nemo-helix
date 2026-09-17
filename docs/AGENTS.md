<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Documentation (Fern)

The docs site is built with **[Fern](https://buildwithfern.com)**, not MkDocs/Sphinx. Pages are `.mdx` under `docs/`; the Fern config lives in `docs/fern/`. Read `docs/fern/README.md` for the full guide — this file is the short version for anyone (human or agent) editing docs.

## Build / preview / check

Run these from the repo root (they wrap `cd docs/fern && npm run …`):

| Command | What it does |
|---|---|
| `make docs-deps` | Install docs tooling (first run on a machine) |
| `make docs` | Local dev server (live preview) |
| `make docs-watch` | Local dev server plus repo-level watcher for `docs/**` changes outside `docs/fern/` |
| `make docs-check` | `fern check` + MDX validation + NotebookViewer artifact validation + gated-link check (what CI runs) |
| `make docs-lint-snippets` | Syntax-check Python fenced snippets across all docs |
| `make docs-check-python-snippets DOCS_PATH=...` | Syntax-check and type-check Python fenced snippets in one doc |
| `make docs-run-notebook DOCS_PATH=...` | Execute the source notebook for one Fern `.mdx`/`.ipynb` doc using `nemo-nb` markers |
| `make docs-broken-links` | Report broken links |
| `make docs-fix-links` | Auto-delink references into gated pages |

Normal authoring and local preview read `docs/fern/versions/latest.yml`. During production publish, `.github/workflows/publish-fern-docs.yaml` also runs `npm run materialize:versions` to generate frozen release-version entries from stable SemVer tags that already contain Fern config. Tags from before the Fern migration, such as early `0.1.x` tags, cannot become Fern versions unless those docs are backported or migrated.

Use `make docs` when you are only editing `docs/fern/` config. Use `make docs-watch` when you are editing page content elsewhere under `docs/`, since it restarts the Fern dev server when repo-level docs files change outside `docs/fern/`.

## Rules that bite if you miss them

- **Navigation is the build.** Fern only builds pages listed in a version nav file. For ordinary docs work, that is `docs/fern/versions/latest.yml`. A `.mdx` not in the nav is **not built** (404, not indexed) — that is how unready features are gated. Do **not** use `hidden: true` for gating (it still builds/serves the page).
- **Publication state is nav-derived.** Do not maintain or rely on a hard-coded list of gated directories. Check `docs/fern/versions/latest.yml`: listed pages are published in Latest, and omitted pages are gated. `docs/fern/gated-nav.yml` contains reference blocks for some gated features. To publish one: move its block into `latest.yml`, re-add inbound links, run `make docs-check && make docs-broken-links`.
- **Don't link into gated pages.** A link from a published page into a gated page is a dead link. `make docs-check` fails on it; `make docs-fix-links` delinks it to plain text. (Replaces the old MkDocs `hide_unready_docs` auto-delinking.)
- **Internal links** use canonical nav URLs like `/documentation/get-started/core-concepts/workspaces`, not relative `.md`/source paths. `make docs-broken-links` is the check.
- **No `{{variable}}` substitutions.** Fern has no substitution step; product names are inlined as literal text. (Prompt-template tokens like `` `{{input}}` `` inside backticks are real content — leave them.)

## Generated pages — do not hand-edit

These are generated; edit the source and regenerate:

| Page | Regenerate with |
|---|---|
| `docs/cli/reference.mdx`, `docs/fern/snippets/_snippets/cli-summary.mdx` | `make generate-cli-reference-docs` |
| `docs/set-up/config-reference.mdx` | `make generate-config-reference-docs` |
| `docs/auth/authorization/permissions-reference.mdx` | `uv run python services/core/auth/scripts/auth-tools.py generate-docs` |

## API reference

The REST API reference is rendered natively by Fern from a docs-only filtered OpenAPI spec. `docs/fern/openapi/openapi.yaml` is a symlink to the generated repo-root `openapi/openapi.yaml`, and `npm run prepare:openapi` writes `docs/fern/openapi/openapi.public.yaml` for Fern. `docs/fern/generators.yml` must point at the public filtered spec, surfaced by the `- api: API Reference` nav node. No `<swagger-ui>` embed.
