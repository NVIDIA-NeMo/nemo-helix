<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# <version> Release Test Scope

> Status: Draft
>
| Field | Value |
|---|---|
| Previous release | `<previous_ref>` (`<previous_sha>`) |
| Release/version snapshot | `<release_ref>` (`<release_sha>`) |
| Main verification snapshot | `<main_ref>` (`<main_sha>`) |
| Release/main merge base | `<merge_base_sha>` |
| Generated | `<timestamp>` |
| Review deadline | `<timestamp>` |

## Release collection summary

- Candidate range: `<previous_ref>..<release_ref>`.
- Release-range commits: `<count>`.
- Associated PRs: `<count>`; unassociated commits: `<count>`.
- Documentation-touched PRs: `<count>`; substantive candidates after release-note exclusion: `<count>`.
- Consolidated capabilities: `<count>`.
- Forward-merge states: `<counts by state>`.
- Dispositions: `<counts by disposition>`.
- Status: `Draft`; human review is required before the release scope is finalized.

## Release-to-main forward-merge trace

| Capability | Forward-merge state | Release evidence | Main/forward-merge evidence | Follow-up |
|---|---|---|---|---|
| `<item>` | `<state>` | `<PRs/docs/commits>` | `<PRs/commits/docs>` | `<follow-up or None>` |

### Missing or different forward-merges

| Capability | Release evidence | Main evidence | Owner | Follow-up |
|---|---|---|---|---|
| `<item or None>` | `<PRs/docs>` | `<missing or differing evidence>` | `<owner>` | `<action>` |

## Proposed test scope

### <Capability name>

| Field | Value |
|---|---|
| Disposition | `Proposed / Needs clarification` |
| Confidence | `High / Medium / Low` |
| Area | `<component or plugin>` |
| Surface | `Studio / CLI / API-SDK / Deployment / Backend` |
| Forward-merge state | `<state>` |
| Owner | `<evidence-backed owner or TBD>` |
| Release PRs | `<links>` |
| Release documentation | `<paths and immutable links>` |
| Forward-merge evidence | `<main evidence>` |

**Description**

<User-visible change.>

**User impact**

<Who benefits or must adapt.>

#### Studio validation

1. `<documented navigation or action>`
2. `<documented expected result>`

Write `Not applicable` when no Studio surface exists. Write `Needs clarification` when evidence is incomplete.

#### CLI validation

```bash
<exact documented commands or clearly labeled placeholders>
```

Expected:

- `<documented observable outcome>`

Write `Not applicable` when no CLI surface exists. Write `Needs clarification` when evidence is incomplete.

#### Additional validation

- Failure or validation path: `<coverage or Needs clarification>`
- Upgrade or compatibility: `<coverage or Not applicable>`
- Permissions or security: `<coverage or Not applicable>`

#### Draft release-note text

<Concise customer-facing statement, or omission reason for unresolved items.>

## Manual additions

| Capability | Rationale | Evidence | Owner | Disposition |
|---|---|---|---|---|
| `<item or None>` | `<why docs discovery missed it>` | `<PR/docs/issue>` | `<owner>` | `<status>` |

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

### Release-to-main forwarding gaps

- `<release-derived capability and required follow-up, or None>`

### Boundary warnings

- `<tag ancestry, branch topology, or compatibility concern, or None>`

## Draft release notes

Include only proposed or approved capabilities with sufficient evidence. Omit excluded and unresolved entries.

### Highlights

- `<highlight>`

### Improvements

- `<improvement>`

### Fixes

- `<user-visible fix>`

### Known limitations

- `<limitation>`
