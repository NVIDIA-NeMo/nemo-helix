<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NeMo Platform Fern Docs

This directory holds the Fern **configuration** for the NeMo Platform documentation site. The page content (`.mdx`) lives in the parent `docs/` tree (e.g. `docs/get-started/setup.mdx`); this `docs/fern/` directory holds the navigation, theme, components, snippets, and OpenAPI wiring. The nav references pages with relative paths (`../../<area>/<page>.mdx`).

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
make docs-lint-snippets  # syntax-check Python fenced snippets across all docs
make docs          # start local preview (prints a localhost URL)
make docs-watch    # start local preview plus a repo-level watcher for docs/** changes
```

Use `make docs` when you are only editing `docs/fern/` config. Use `make docs-watch` when you are editing page content elsewhere under `docs/`, since it restarts the Fern dev server when repo-level docs files change outside `docs/fern/`.

`package.json` shells out to `npx -y fern-api@latest` for the Fern CLI itself, but `make docs-deps` (`npm ci`) is required once because the MDX validator (`@mdx-js/mdx`) is a local dependency.

## Layout

```text
docs/                          # page content (.mdx), one tree per product area
├── get-started/ , agents/ , evaluator/ , ...   # published pages
└── fern/                      # <- this directory: Fern site config
    ├── fern.config.json       # Fern organization + CLI version
    ├── package.json           # npm run check|dev|generate|preview|broken-links|*-gated-links
    ├── docs.yml               # Site config, theme, css, redirects, versions
    ├── generators.yml         # OpenAPI spec wiring for the API reference
    ├── openapi/openapi.yaml    # symlink -> repo-root openapi/openapi.yaml (generated)
    ├── openapi/openapi.public.yaml # generated Fern-only filtered spec (gitignored)
    ├── styles/                # CSS (notebook-viewer.css, button.css)
    ├── assets/                # logos, shared images
    ├── components/            # custom TSX MDX components
    ├── snippets/              # reusable <Markdown src> fragments
    ├── scripts/               # validate-mdx.mjs, delink-gated.mjs, ipynb-to-fern-json.py
    ├── gated-nav.yml          # reference nav blocks for gated (unready) features
    ├── generated/             # ignored release snapshots created during publish
    └── versions/latest.yml    # Latest navigation tree (ordinary docs authoring)
```

Normal docs authoring targets `Latest`. `versions/latest.yml` defines the sidebar and maps each page file to its canonical route — and, because Fern only builds pages listed there, it is also what gates unready content (see below). Production publish additionally materializes frozen release versions from stable SemVer tags that already contain Fern config.

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

- `generators.yml` declares the docs-only public spec: `api.specs[].openapi: ./openapi/openapi.public.yaml`.
- `versions/latest.yml` surfaces it with an `- api: API Reference` navigation node (under the **Reference** section).

`openapi/openapi.yaml` is a symlink to the repo-root `openapi/openapi.yaml` (the generated source of truth). Fern reads `openapi/openapi.public.yaml`, which is generated by `scripts/filter-public-openapi.mjs` and excludes services that should stay out of public docs while remaining in the root spec for SDK generation and API validation. Regenerate the root spec with `make refresh-openapi` from the repo root, then run `npm run check`.

Fern groups endpoints by their OpenAPI tag in the sidebar (Customizer, Evaluator, Guardrails, …), which replaces the old per-service filter chips. Link to it from other pages with the nav URL `/documentation/reference/api-reference`.

## Release versioning

Fern does not infer docs versions from git tags. The version selector is built from `docs.yml` `versions:` entries, and each entry points at a version-specific nav file.

For public publishing, `.github/workflows/publish-fern-docs.yaml` runs:

```bash
npm run materialize:versions
```

That script discovers stable SemVer tags in the full checkout, skips tags that predate `docs/fern/versions/latest.yml`, exports each tag's `docs/` tree into ignored `docs/fern/generated/release-versions/<version>/` snapshots, rewrites the tag's `versions/latest.yml` to point at the snapshot, and injects generated entries into `docs.yml` before `npm run generate` publishes. Tag pushes are publish signals; the workflow checks out the default branch so the current Fern publishing code and `Latest` docs remain the source for the live site.

This means early `0.1.x` tags that were cut before the Fern migration cannot appear as true Fern versions unless someone deliberately backports/migrates those docs into Fern. A release tag such as `0.2.0`, which already contains `docs/fern`, can be materialized.

To preview the release selector locally, run `npm run materialize:versions` from `docs/fern/`, then `npm run dev`. The generated snapshots and generated version YAMLs are gitignored; restore `docs/fern/docs.yml` afterward if you only needed a local preview.

### Pre-release branches

`docs/fern/release-branches.json` is an optional allowlist of release branches to preview before their tags are cut. Standard branch names derive their eventual tag automatically: `release/0.6` becomes `0.6.0`, and `release/0.6.1` becomes `0.6.1`.

```json
[
  "release/0.6"
]
```

If the tag (`0.6.0`) doesn't exist yet but the branch (`release/0.6`) does and contains `docs/fern/versions/latest.yml`, `materialize-release-versions.mjs` builds that version's docs from the branch instead, under the same version path the real tag will use, and marks it `availability: preview` in `docs.yml` (Fern's version-selector schema has no literal "pre-release" value; `preview` is the closest fit and still renders a badge). Once the real tag is pushed, it takes over automatically — the branch entry is skipped as soon as the tag exists.

For patch or nonstandard cases, use object form to map a branch to the exact future tag:

```json
{
  "release/0.6": "0.6.1"
}
```

The allowlist is not read by Fern itself, so it isn't subject to `docs.yml`'s schema (which is `additionalProperties: false`). It's optional: an absent file or empty array/object is a no-op. A listed branch that isn't already fetched locally is pulled from `origin` on demand (anonymously — this repo is public, so no token is needed), then treated the same as any other local ref. This is the same code path everywhere it runs, so no extra CI step is needed: `npm run materialize:versions` behaves identically locally, in `publish-fern-docs.yaml`, and in `fern-docs-preview-build.yaml` (so a PR's Fern preview shows the same version selector, including selected pre-release branches, that a real publish would produce — not just `main`). A confirmed-absent branch is a soft skip; any other fetch failure (network, auth) fails the run instead of silently dropping the preview version.

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
| `fern-docs-ci.yaml` | `pull_request` touching `docs/**` | `npm run check` (fern check + MDX + NotebookViewer artifacts + gated links) and `npm run broken-links` |
| `fern-docs-preview-build.yaml` | `pull_request` touching `docs/**` | Upload PR `docs/` sources as an artifact (no secrets — fork-safe) |
| `fern-docs-preview-comment.yaml` | successful preview build (`workflow_run`) | Build a Fern preview with `DOCS_FERN_TOKEN` and post/update the PR comment |
| `publish-fern-docs.yaml` | push to `main` touching `docs/**`, stable SemVer tag, `docs/v*` republish tag, or manual dispatch | Materialize release versions and publish the Fern docs site |

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
| Release tag missing from version selector | Confirm the tag is stable SemVer and contains `docs/fern/versions/latest.yml`; pre-Fern tags are skipped by `npm run materialize:versions` |
