<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# <version> Release Test Scope

| Field | Value |
|---|---|
| Previous release | `<previous_ref>` (`<previous_sha>`) |
| Release/version snapshot | `<release_ref>` (`<release_sha>`) |
| Main verification snapshot | `<main_ref>` (`<main_sha>`) |
| Release/main merge base | `<merge_base_sha>` |
| Generated | `<YYYY-MM-DD>` |
| Review deadline | `<YYYY-MM-DD>` |
| GitHub enrichment | `Complete / Unavailable / Partial: <one-clause reason and what the artifact is based on>` |

## Release collection summary

- Candidate range: `<previous_ref>..<release_ref>`.
- Scope basis: `<what candidate discovery used: release-range commits, published docs in docs/fern/versions/latest.yml, and the draft release notes>`.
- Consolidated scope: `<short qualitative summary of the major review areas>`.
- Release-note coverage: `<where coverage is complete, partial, or missing without restating mechanical counts>`.
- Forwarding check: `<all release-derived work is present on main, or summarize only exceptions>`.
- Status: `Draft`; human review is required before the release scope is finalized.

## Release-to-main forwarding exceptions

Write `None.` plus one sentence naming the shared SHA when the release and main snapshots are identical. Otherwise include only rows where release-derived work is missing, materially different, or needs follow-up on main.

| Capability | Release evidence | Main evidence | Owner | Follow-up |
|---|---|---|---|---|
| `<item>` | `<PRs/docs>` | `<missing or differing evidence>` | `<owner>` | `<action>` |

## Highest Priorities

The short checklist testers read first. Group by product area using these headings, in this order, and omit an area with no items: **Platform**, **Agents**, **Evaluation**, **Fine-tuning**, **Other**. Rules:

- Each item is a one-line user journey or regression target drawn from a capability in **Proposed test scope**. Do not list anything that has no capability section; put it under **Manual additions** instead.
- Order items by release risk and user visibility, not by commit order.
- Append `(Studio and CLI)` or `(Studio)` or `(CLI)` when a surface applies. Nest at most one level of sub-journeys.
- Use `please regression test` for renames, rewrites, and component swaps where behavior should be unchanged.
- State a pass condition in the item when it is not obvious (for example, "each run should generate a report and artifacts").
- Keep it to roughly 3-6 items per area. Detail belongs in the capability sections below.

### <Area>

1. `<Journey> (<surfaces>)`
   1. `<Sub-journey> (<surfaces>)`
2. `<Journey>; please regression test`

## Proposed test scope

State that this is a consolidated feature scope, not an exhaustive PR manifest. Number capabilities in the order of the Highest Priorities areas, then by risk. Prefix the title with the surface only when it is Studio-only (for example `Studio: <capability>`).

### 1. <Capability name>

| Field | Value |
|---|---|
| Disposition | `Proposed / Needs clarification` |
| Confidence | `High / Medium / Low` |
| Area | `<component or plugin>` |
| Surface | `CLI / Studio / API-SDK / Deployment / Backend` |
| Forward-merge state | `<state>` |
| Owner | `<evidence-backed owner or TBD>` |
| Release sources | `<short SHA>` (#<PR>), `<short SHA>` (#<PR>) |
| Release documentation | `<repo doc paths>`; `<note on any doc that needs clarification>` |
| Release-note coverage | `Covered / Partial / Missing — <note path, section>` |
| Forward-merge follow-up | `<only when missing, different, or needing human action; otherwise omit this row>` |

**Description**

<User-visible change.>

**User impact**

<Who benefits or must adapt.>

#### Studio validation

Include only when the capability has a Studio surface or a Studio path needs clarification. Use bold for UI labels.

1. `<documented navigation or action>`
2. `<documented expected result>`

#### CLI validation

Include only when the capability has a CLI surface or a CLI command needs clarification.

```sh
<exact documented commands or clearly labeled placeholders>
```

Expected:

- `<observable outcome a tester can confirm, including Studio/CLI parity when both exist>`

#### Additional validation

Plain bullets, no label prefixes. Cover failure paths, upgrade/compatibility, and permissions/security only when applicable; write `Needs clarification` where evidence is missing.

- `<validation step>`

#### Draft release-note text

<Concise customer-facing statement, or omission reason for unresolved items.>

## Manual additions

| Capability | Rationale | Evidence | Owner | Disposition |
|---|---|---|---|---|
| `<item or None>` | `<why docs discovery missed it, or "No manual additions were provided for this run.">` | `<PR/docs/issue or N/A>` | `<owner or N/A>` | `<status or N/A>` |

## Excluded from this release

| Capability | Reason | Owner | Follow-up |
|---|---|---|---|
| `<item or None>` | `<reason>` | `<owner>` | `<issue or next release>` |

## Gaps and warnings

### Potentially undocumented changes

- `<PR and why it may require review>`

### Unverified commands or Studio paths

- `<capability and missing evidence>`

### Documentation without in-range implementation

- `<documentation change and related implementation evidence>`

### Release-note coverage gaps

- `<user-visible release or cherry-equivalent change absent or only partially represented in the release-note diff/content>`

### Release-to-main forwarding gaps

- `<release-derived capability and required follow-up, or None>`

### Boundary warnings

- `<tag ancestry, branch topology, a not-yet-cut release branch, a non-tag previous boundary, or other compatibility concern, or None>`

## Draft release notes

Include only proposed or approved capabilities with sufficient evidence and resolved release-note coverage. Omit excluded and unresolved entries. Group the blocks under established product-area headings (for example Agents and Fabric, Agent Evaluation, Studio, Platform, CLI, and Deployment). The text must match `docs/about/release-notes/current-release.mdx`; point to it rather than rewriting it when it already covers the capability.

### <Product area>

#### <User-visible capability>

**Description:** <What changed, who benefits, and any essential boundary.>

**Documentation:** [<Task or concept title>](<canonical published documentation URL, absolute with a real host>)

**Use it:**

- **CLI:** `<verified command>`
- **Studio:** Open **<verified navigation labels>**. Append "exact label needs final UI verification" when unverified.

Include only applicable interface bullets. When neither applies, write: `This behavior applies automatically; there is no separate CLI or Studio entry point.`

### Known limitations

- `<limitation>`
