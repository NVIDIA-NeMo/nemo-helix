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
| `make docs-lint-snippets` | Check Python syntax across all docs, type-check the curated executable-doc subset, validate Helm YAML and API examples, and run `docs-check-base-url` |
| `make docs-lint-helm-snippets` | Parse Helm docs YAML, reject duplicate keys, and check chart-value nesting against `k8s/helm/values.yaml` |
| `make docs-lint-api-examples` | Validate explicit OpenAPI examples against schemas, reject empty required objects that Fern drops, and check bodyless response samples |
| `make docs-check-base-url` | Flag snippets that bypass the CLI context: Python that reads `NHX_BASE_URL`, or shell that assigns it a literal URL |
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
- **Connect through the CLI context, not `NHX_BASE_URL`.** Build Python clients with `NemoClient.from_config()` and leave `NHX_BASE_URL` unset for `nemo` commands. Direct-URL tools such as `curl` may default it with `${NHX_BASE_URL:-http://localhost:8080}`. Mark a deliberate override with `{/* @nemo-docs: allow-nhx-base-url */}` before the fence. `make docs-check-base-url` enforces this; see `docs/cli/connect-to-deployments.mdx` for the user-facing guidance.
- **No `{{variable}}` substitutions.** Fern has no substitution step; product names are inlined as literal text. (Prompt-template tokens like `` `{{input}}` `` inside backticks are real content — leave them.)

## Snippet lint coverage

Python snippets are syntax-checked across all docs; the curated executable-doc subset is also checked with `ty`. These static checks do not verify notebook execution order, so an import in a later cell can hide an earlier runtime error. Prefer a narrow, explained `ty: ignore[unresolved-import]` for a tutorial-only dependency over skipping an entire block's type checks.

Helm YAML syntax alone does not catch flattened nesting that still parses. The Helm snippet linter also checks partial overrides against the chart defaults, flags implicit empty values, and allows free-form configuration maps. Use explicit `null` for intentional deletion of a Helm default. These checks do not validate SCC admission, credentials, filesystem permissions, or whether prose agrees with configuration; those still need review or runtime tests.

API examples live in service route/model definitions, not generated pages. Required object examples must have representative nonempty values: Fern can omit `{}` fields from its generated snippets even when the schema accepts them. The API example linter validates source examples and checks explicit bodyless-response code samples; it does not replace inspecting a Fern preview of generated language snippets.

## Release notes

Use one permanent file per release:
`docs/about/release-notes/release-<major>-<minor>-<patch>.mdx`. Label its Fern nav
entry with the version (for example `v0.8.0`) and give it the explicit slug
`release-0-8-0`. Keep the release index and nav in descending semantic-version
order, including when adding patches for an older release line. Do not overwrite
or archive another version's page when adding a release. This avoids unrelated
release lines sharing a rolling notes file; the index and nav still need normal
merge reconciliation. Link to a version's canonical page, or to the release
index when no specific version is intended.

## Generated pages — do not hand-edit

These are generated; edit the source and regenerate:

| Page | Regenerate with |
|---|---|
| `docs/cli/reference.mdx`, `docs/fern/snippets/_snippets/cli-summary.mdx` | `make generate-cli-reference-docs` |
| `docs/set-up/config-reference.mdx` | `make generate-config-reference-docs` |
| `docs/auth/authorization/permissions-reference.mdx` | `uv run python services/core/auth/scripts/auth-tools.py generate-docs` |

## API reference

The REST API reference is rendered natively by Fern from the generated OpenAPI spec. `docs/fern/openapi/openapi.yaml` is a symlink to the generated repo-root `openapi/openapi.yaml`. `docs/fern/generators.yml` must point at that spec, surfaced by the `- api: API Reference` nav node. No `<swagger-ui>` embed.
