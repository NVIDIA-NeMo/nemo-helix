<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NeMo Helix Fern Docs

This directory holds the Fern **configuration** for the NeMo Helix documentation site. The page content (`.mdx`) lives in the parent `docs/` tree (e.g. `docs/get-started/setup.mdx`); this `docs/fern/` directory holds the navigation, theme, components, snippets, and OpenAPI wiring. The nav references pages with relative paths (`../../<area>/<page>.mdx`).

## Quick Links

| What | Where |
| --- | --- |
| Fern dashboard | https://dashboard.buildwithfern.com (NVIDIA org) |
| Contributor/agent guide | [`../AGENTS.md`](../AGENTS.md) |
| Make targets | `make docs`, `make docs-watch`, `docs-check`, `docs-broken-links`, `docs-fix-links`, `docs-preview` (repo root) |
| CI workflows | [`../../.github/workflows/`](../../.github/workflows/) (`fern-docs-*.yaml`) |
| Publish workflow | [`../../.github/workflows/publish-fern-docs.yaml`](../../.github/workflows/publish-fern-docs.yaml) |

## Quickstart

From the repo root (these wrap `cd docs/fern && npm run …`):

```bash
make docs-deps     # one-time: install docs/fern tooling (needed for MDX validation)
make docs-login    # one-time per machine: Fern CLI auth for the nvidia org
make docs-check    # validate: fern check + MDX + NotebookViewer artifacts + gated links
make docs-lint-snippets  # check Python syntax/types, Helm YAML, and API examples
make docs          # start local preview (prints a localhost URL)
make docs-watch    # start local preview plus a repo-level watcher for docs/** changes
```

Use `make docs` when you are only editing `docs/fern/` config. Use `make docs-watch` when you are editing page content elsewhere under `docs/`, since it restarts the Fern dev server when repo-level docs files change outside `docs/fern/`.

`package.json` shells out to `npx -y fern-api@latest` for the Fern CLI itself, but `make docs-deps` (`npm ci`) is required once because the MDX validator (`@mdx-js/mdx`) is a local dependency.

## Layout

```text
docs/                          # page content (.mdx), one tree per product area
├── get-started/ , agents/ , evals/ , ...   # published pages
└── fern/                      # <- this directory: Fern site config
    ├── fern.config.json       # Fern organization + CLI version
    ├── package.json           # npm run check|dev|generate|preview|broken-links|*-gated-links
    ├── docs.yml               # Site config, theme, css, redirects, versions
    ├── generators.yml         # OpenAPI spec wiring for the API reference
    ├── openapi/openapi.yaml    # symlink -> repo-root openapi/openapi.yaml (generated)
    ├── styles/                # CSS (notebook-viewer.css, button.css)
    ├── assets/                # logos, shared images
    ├── components/            # custom TSX MDX components
    ├── snippets/              # reusable <Markdown src> fragments
    ├── scripts/               # validate-mdx.mjs, delink-gated.mjs, ipynb-to-fern-json.py
    ├── gated-nav.yml          # reference nav blocks for gated (unready) features
    ├── generated/             # ignored release snapshots created during publish
    └── versions/latest.yml    # Latest navigation tree (ordinary docs authoring)
```

Normal docs authoring targets `Latest`. `versions/latest.yml` defines the sidebar and maps each page file to its canonical route — and, because Fern only builds pages listed there, it is also what gates unready content (see below). Production publish additionally materializes one documentation version per release branch that contains Fern config.

## Authoring

Add pages under the relevant `docs/<area>/` directory and wire them into `versions/latest.yml`.

Use front matter for the rendered page title:

```yaml
---
title: "Page Title"
description: ""
---
```

Do not add a duplicate first `# Page Title` heading when it matches the front matter `title`; Fern renders that title automatically.

Use Fern-native MDX components such as `<Note>`, `<Tip>`, `<Warning>`, `<Tabs>`, `<Cards>`, and `<Card>`. Do not reintroduce MkDocs Material syntax like `!!! note`, `=== "Tab"`, `--8<--`, or `<div class="grid cards" markdown>`.

## Internal Links

Use Fern's nav-derived canonical URLs:

```mdx
[Workspaces](/documentation/get-started/core-concepts/workspaces)
```

Avoid source-path links such as `/get-started/concepts/workspaces`, `/latest/get-started/concepts/workspaces`, and relative `.md` links. If a public URL changes, add a redirect in `docs.yml`.

## API Reference

The REST API reference is generated natively by Fern from the OpenAPI spec — no `<swagger-ui>` embed. Two pieces wire it up:

- `generators.yml` declares the generated spec: `api.specs[].openapi: ./openapi/openapi.yaml`.
- `versions/latest.yml` surfaces it with an `- api: API Reference` navigation node (under the **Reference** section).

`openapi/openapi.yaml` is a symlink to the repo-root `openapi/openapi.yaml` (the generated source of truth). Regenerate the root spec with `make refresh-openapi` from the repo root, then run `npm run check`.

`openapi/overrides.yml` supplies a documentation-only local server URL so default cURL samples
include a host, without changing generated client connection defaults. Replace that URL with your
deployment URL when using the examples. Resource names, experiment IDs, model names, credentials,
and enabled services remain deployment prerequisites. Proxy examples show a chat request body;
PUT/PATCH/DELETE success still depends on the upstream provider supporting the selected method/path.

API reference code samples are limited to Python, JavaScript, and cURL. The
`settings.http-snippets` allowlist in `docs.yml` controls Fern's HTTP fallbacks;
`script/openapi_helper/code_samples.py` supplies Python/JavaScript overrides and
separate default-cURL SDK overrides. Authored route samples must follow the same
language policy; the fallback allowlist does not remove custom samples.

Fern groups endpoints by their OpenAPI tag in the sidebar (Customizer, Evaluator, Guardrails, …), which replaces the old per-service filter chips. Link to it from other pages with the nav URL `/documentation/reference/api-reference`.

## Release versioning

Fern does not infer docs versions from git tags. The version selector is built from `docs.yml` `versions:` entries, and each entry points at a version-specific nav file.

For public publishing, `.github/workflows/publish-fern-docs.yaml` runs:

```bash
npm run materialize:versions
```

That script refreshes `release/X.Y` and `release/X.Y.Z` branches from `origin`, skips branches without `docs/fern/versions/latest.yml`, and exports each branch tip's `docs/` tree into ignored `docs/fern/generated/release-versions/<version>/` snapshots. It rewrites the branch's navigation and injects one entry per branch into `docs.yml`, sorted newest first. For example, `release/0.9` has label `0.9` and slug `v0.9`. Patch tags do not create extra selectors.

Docs always come from the branch, even after tagging. Removed branches disappear on the next materialization. There are no tag snapshots or release-branch configuration files. Individual patch release-note pages remain in their release branch's navigation.

The script fetches branches and tags every run and unshallows when necessary, preventing stale content and incomplete ancestry checks. Network/auth failures stop materialization before generated files change. Production publishing still uses the default branch's publishing code and `Latest` docs.

To preview the release selector locally, run `npm run materialize:versions` from `docs/fern/`, then `npm run dev`. The generated snapshots and generated version YAMLs are gitignored; restore `docs/fern/docs.yml` afterward if you only needed a local preview.

### Preview status

A version is marked `availability: preview` until a stable SemVer tag from its own release line is reachable from the branch tip. Both lightweight and annotated tags on `origin` count. Tags are refreshed in an isolated Git namespace; local-only tags cannot remove the badge, and developers' own tags are not modified.

For example, `release/0.9` accepts a reachable `0.9.0`, `v0.9.1`, or another stable `0.9.*` tag. A patch branch such as `release/0.9.1` requires exactly that patch. Prerelease tags, tags inherited from older release lines, and tags on unmerged commits do not remove the badge.

Tags only establish whether a release line has shipped; they never choose content or add selector entries. Stable tag pushes still trigger publishing to refresh the preview badge.

The picker contains release lines, not individual patch releases. To preserve old links during migration, materialization generates redirects for reachable stable **0.5.x through 0.8.x** tags to the matching release-branch slug. For example, `/v0.8.1` redirects to `/nemo-helix/v0.8`, and `/v0.8.1/:path*` redirects to `/nemo-helix/v0.8/:path*` (with the configured instance basepath in destinations). Root and nested-page URLs are covered without adding picker entries or duplicate docs builds.

Redirect aliases are discovered from tags, not maintained in a JSON list. Authored redirects take precedence, generated aliases are refreshed idempotently, and a dedicated patch branch is never redirected away. No aliases are generated for versions before 0.5 or from 0.9 onward; newer release lines use branch-line URLs from the start.

Retired-page redirects follow the same rule per version. A release branch's `docs.yml` lists its retired pages under `/latest/...`, the only slug it publishes as. Materialization mirrors each of those under the branch's version slug, so `/v0.9/documentation/agents/plugins-and-skills` redirects to `/nemo-helix/v0.9/documentation/reference/plugin-architecture` just as `/latest/...` does. Only redirects from that branch's own `docs.yml` are mirrored, so a version never redirects to a page it does not contain.

Local materialization, CI previews, and production publishing use the same discovery and preview rules. Run `npm test` in this directory for isolated Git-fixture regression tests.

## Gated (unready) features

Some features are not shipped yet and must be **fully excluded from the build** — not just hidden from the sidebar. Fern's `hidden: true` still builds and serves the page (reachable by direct URL and indexable), so it is **not** used for this. Instead, the gated pages are simply **left out of `versions/latest.yml`**: Fern only builds pages referenced in the navigation, so an omitted page is never built (it 404s and is not indexed). This matches the old MkDocs `hide_unready_docs` hook, which dropped the same files from the build.

Gated `.mdx` files stay in the repo so they remain maintained. Do not keep a separate list of gated directories in contributor docs: publication state is derived from `versions/latest.yml`. A page listed there is published; an omitted page is gated.

Inbound links from visible pages into gated pages are **delinked to plain text** (not rewritten URLs), since the target is not built — otherwise they would be broken links.

### Automated delinking

The old MkDocs hook delinked those references automatically. Fern has no build hook, so `scripts/delink-gated.mjs` reproduces it. "Gated" is derived from the nav (any `.mdx` not in `versions/latest.yml`), so there is no second list to maintain.

- `npm run check:gated-links` — fails if any **published** page links into a gated page (runs as part of `npm run check`, so CI enforces it).
- `npm run fix:gated-links` — rewrites those links to plain text in place.

So if someone links into a gated section, CI flags it and one command delinks it — no hunting for dead links by hand. `npm run broken-links` (also run in CI) is the broader backstop for any other dead link.

One difference from the old MkDocs hook: that hook ran at build time and kept the link in the source, so re-publishing a page reactivated its inbound links automatically. This script delinks in the **source**, so the references become plain text. When you publish a feature, re-add the links you want from those plain-text references as part of the same step that moves the nav block (`check:gated-links` won't block you — it only flags live links that still point at gated pages). This keeps local `fern docs dev` honest (no build-time source rewriting) at the cost of re-linking on publish.

`gated-nav.yml` holds the ready-to-paste navigation blocks for the gated features (it is a reference only; Fern does not read it). **To publish a feature when it ships:** move its block from `gated-nav.yml` into `versions/latest.yml`, re-add any inbound links you want, then run `npm run check` and `npm run broken-links`.

## CI and Publishing

| Workflow | Trigger | Purpose |
| --- | --- | --- |
| `fern-docs-ci.yaml` | `pull_request` touching `docs/**`, `openapi/**`, `services/**/src/**/*.py`, `k8s/helm/values.yaml`, `pyproject.toml`, `uv.lock`, `Makefile`, `.nvmrc`, or `.github/workflows/fern-docs-ci.yaml` | `npm run check` (fern check + MDX + NotebookViewer artifacts + gated links), plus docs linter regression tests, Python/Helm/API snippet lint, and `npm run broken-links` |
| `fern-docs-preview-build.yaml` | `pull_request` touching `docs/**` | Upload PR `docs/` sources as an artifact (no secrets — fork-safe) |
| `fern-docs-preview-comment.yaml` | successful preview build (`workflow_run`) | Build a Fern preview with `DOCS_FERN_TOKEN` and post/update the PR comment |
| `publish-fern-docs.yaml` | push to `main` or `release/*` touching `docs/**`, stable SemVer tag, `docs/v*` republish tag, or manual dispatch | Materialize release versions and publish the Fern docs site |

Required secret: `DOCS_FERN_TOKEN` (org-level), from `fern token` for an account that can publish to the NVIDIA Fern organization.

PRs that touch `docs/**` get a shared preview URL posted as a comment after the two-part preview workflow finishes. Note: the `workflow_run`-triggered comment job only runs once these workflows are on the default branch (`main`), so the first preview appears after this PR merges.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `HTTP 403` or organization access error | Sign in at https://dashboard.buildwithfern.com, then run `npx -y fern-api@latest login` again |
| `fern check` YAML error | Use 2-space indentation; make sure `path:` values are relative to `versions/latest.yml` |
| Page 404 in preview | Check that `versions/latest.yml` lists the page (gated pages are intentionally omitted and *will* 404) |
| Broken internal link | Rewrite to the nav URL `/documentation/...`; if it targets a gated page, run `make docs-fix-links` to delink it. `make docs-broken-links` reports them all |
| JSX or MDX parse error | Escape raw `{}`, `<`, or `>` in prose, and use Fern components instead of raw MkDocs syntax |
| Release branch missing from version selector | Confirm `origin` has a `release/X.Y` or `release/X.Y.Z` branch containing `docs/fern/versions/latest.yml`; run `npm run materialize:versions` |
