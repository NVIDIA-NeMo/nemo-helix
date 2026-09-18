---
name: release-test-scope
description: Build one pre-cut NeMo Platform release test-scope artifact with feature descriptions, QA validation, stakeholder review fields, and draft release notes from the previous-release-to-release-branch range while verifying release-derived work on main. Use for release planning, cross-functional feature review, QA scoping, and pre-release notes. Do not use it as a generic main-branch changelog.
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Release Test Scope

Generate one evidence-backed local artifact for QA, Engineering, PM, TPM, and developer review. It combines the feature list, source evidence, validation paths, risks, decisions, and draft release notes. Do not treat every PR as a separate feature.

## Easy invocation

Prefer this complete prompt:

```text
Use $release-test-scope for version <version>, previous tag <previous-tag>,
release ref <release-ref>, and main verification ref <main-ref>.
Generate the local shared release test scope. Do not create or update a GitHub issue.
```

Example:

```text
Use $release-test-scope for version 0.6.0, previous tag 0.5.0,
release ref origin/release/0.6, and main verification ref origin/main.
Generate the local shared release test scope. Do not create or update a GitHub issue.
```

The normal invocation needs only those four values. Ask for input only if a boundary is missing or ambiguous.

## Required inputs

- `version`: proposed release version.
- `previous_ref`: previous released SemVer tag.
- `release_ref`: release branch or proposed cut commit.
- `main_ref`: main snapshot used only to trace release-derived work.

Resolve all refs to immutable 40-character SHAs and record them in the artifact.

Optional inputs include manual additions, agreed exclusions, and a GitHub issue to refresh.

## Operating invariants

- The only automatic candidate-producing range is `previous_ref..release_ref`.
- Main is used only to trace or verify release-derived commits or PRs. Never enumerate unrelated main development as release candidates.
- Release reachability, not presence on main, determines whether work shipped.
- Substantive documentation in the release range that is published through the release snapshot's Fern navigation is the default automatic inclusion gate.
- Exclude `docs/about/release-notes/**` from automatic candidate qualification when it only summarizes the same release, but always inspect its content and history as release-note coverage evidence.
- Group related implementation, documentation, fix, and polish PRs into one user-visible capability.
- Do not invent behavior, Studio navigation, CLI commands, expected results, owners, or documentation links.
- Mark incomplete evidence `Needs clarification` and name the missing evidence.
- Keep manual additions and exclusions explicit; never silently discard them.
- Generate locally first. Creating or updating a GitHub issue requires explicit user authorization.
- Never trigger a release, modify a tag, publish release notes, or alter release branches.

## Required capabilities

- Read access to repository history.
- Local Git history is sufficient to generate a commit-backed draft.
- Network access and authenticated `gh` access enrich the artifact with authoritative PR metadata, owners, labels, and linked issues.
- GitHub write access only when the user explicitly authorizes issue creation or update.

If `gh auth status` or API access fails, continue from local Git unless the user explicitly required complete GitHub metadata. Mark GitHub enrichment unavailable, use commit SHAs instead of guessed PR metadata or owners, and identify the affected fields as incomplete. Do not call a locally generated artifact complete when GitHub-only evidence is missing.

## Workflow

### 1. Validate boundaries and topology

Run:

```bash
git rev-parse --verify "${previous_ref}^{commit}"
git rev-parse --verify "${release_ref}^{commit}"
git rev-parse --verify "${main_ref}^{commit}"
git merge-base --is-ancestor "${previous_ref}" "${release_ref}"
git merge-base --is-ancestor "${previous_ref}" "${main_ref}"
git rev-parse "${previous_ref}^{commit}"
git rev-parse "${release_ref}^{commit}"
git rev-parse "${main_ref}^{commit}"
git merge-base "${release_ref}" "${main_ref}"
```

Stop if a ref does not resolve. If the previous release is not an ancestor of either snapshot, report the topology and ask for corrected boundaries. Do not substitute a triple-dot range or merge base silently.

Run `gh auth status` before GitHub enrichment or issue operations. Authentication failure does not invalidate the local Git comparison; apply the fallback under **Required capabilities**.

### 2. Collect the release range

Use only:

```text
previous_ref..release_ref
```

1. Enumerate all commits in that range.
2. Preserve commit SHAs as the baseline source evidence. When GitHub is available, also associate each commit with PRs through GitHub commit-to-PR metadata. Do not rely only on commit-message PR numbers.
3. When GitHub is available, fetch PR number, title, URL, body, labels, author, base branch, merge commit, linked issues, and changed files. Otherwise retain commit-backed evidence and mark those fields unavailable.
4. Record release commits that have no associated PR.
5. Identify PRs with changed paths under `docs/**`.
6. Exclude release-note-only changes from automatic qualification.

Collect the authoritative documentation delta with:

```bash
git diff --name-status "${previous_ref}..${release_ref}" -- docs/
git diff --find-renames "${previous_ref}..${release_ref}" -- docs/

git diff --find-renames "${previous_ref}..${release_ref}" -- \
  docs/about/release-notes/
git log --format='%H%x09%cs%x09%s' \
  "${previous_ref}..${release_ref}" -- docs/about/release-notes/
git show "${release_ref}:docs/fern/versions/latest.yml"
```

The release-note diff is coverage evidence, not a candidate source. Compare every user-visible release capability with the actual note content. A change being reachable from the release branch or represented by a release-note commit does not prove that the capability itself is described. A documentation source file counts as published only when its page is reachable from the release snapshot's `docs/fern/versions/latest.yml`; record gated pages as missing publication evidence.

### 3. Verify release-derived work on main

For each release-range commit or PR, classify:

- `Exact commit on main`
- `Forwarded by PR`
- `Included by aggregate forward-merge`
- `Not found on main`
- `Different on main`

Use evidence in this order:

1. Commit ancestry at the immutable main SHA.
2. Commit-to-PR associations for release and merge commits.
3. Explicit release-PR references in a forward-merge PR.
4. Membership in an aggregate release-to-main merge.
5. Patch equivalence plus matching documentation paths and behavior.

Do not match similar titles alone. A forward-merge PR is supporting evidence, not a separate feature. If the release snapshot is an ancestor of main, classify its release-derived items as exact and do not enumerate main-only PRs.

Use a cherry-marked comparison to expose patch-equivalent changes that different commit IDs would otherwise hide:

```bash
git log --left-right --cherry-mark --no-merges \
  --format='%m%x09%H%x09%cs%x09%s' \
  "${release_ref}...${main_ref}"
```

`=` establishes patch equivalence, not release-note coverage. Keep user-visible equivalence groups in the evidence ledger until their documentation and release-note disposition are checked. For aggregate forward merges, inspect the commits introduced from the release parent and never treat the merge wrapper as a feature.

### 4. Build and consolidate evidence

For every candidate capability, capture:

- Plain-language feature description and user impact.
- Source commits, plus PRs and linked issues when GitHub enrichment is available.
- Release documentation paths and relevant headings.
- Component and surface: Studio, CLI, API/SDK, Deployment, or Backend.
- Exact documented Studio navigation.
- Exact documented CLI commands.
- Documented expected outcomes and failure paths.
- Upgrade, compatibility, permissions, and security considerations.
- Evidence-backed owner when available.
- Confidence, disposition, and forward-merge state.
- Release-note coverage: `Covered`, `Partial`, or `Missing`, with the exact note path and heading when present.
- Release evidence separately from main-forwarding evidence.

Keep command blocks verbatim except for clearly labeled placeholders. Do not infer Studio paths from component names or convert SDK examples into CLI commands.

Group PRs that implement, document, fix, or polish the same user-visible journey. Keep separate features when user journeys, owners, risk, or validation differ.

### 5. Report gaps without broadening scope

Explicitly identify:

- Release-range feature/fix PRs without documentation evidence.
- Documentation whose implementation is outside the release range.
- Release-derived work missing or materially different on main.
- Commands or Studio paths that cannot be sourced.
- Conflicting or incomplete descriptions.
- Critical security, migration, or compatibility work needing manual inclusion.
- User-visible release or cherry-equivalent changes missing from the release-note diff and resulting note text.

Do not add undocumented work automatically. Put it in a warning or manual-decision section.

Before finalizing draft release notes in an interactive run, present all otherwise-includable capabilities with missing substantive documentation or `Partial`/`Missing` release-note coverage as one grouped decision. Let the user include, omit, or defer each item. In headless or explicitly non-interactive runs, retain the items in gaps and warnings but omit them from draft release notes.

### 6. Generate the shared local artifact

Read [the shared release test-scope template](references/ticket-template.md).

Write:

```text
release-artifacts/<version>/qa-test-scope.md
```

The artifact must be readable by all release stakeholders. Use one section per consolidated feature and include the description, user impact, release-snapshot docs, Studio path, CLI command, expected result, risks, and draft release-note text. State clearly that it is a consolidated feature scope, not an exhaustive PR manifest.

Draft release notes must use one structured block per included user-visible capability: `Description`, `Documentation`, and `Use it`. Under `Use it`, include verified CLI commands and/or Studio navigation as applicable; state explicitly when neither surface applies. Keep internal-only maintenance compact and do not invent an interface to fill the template.

In the metadata table, set `Review deadline` to the report generation date. Do not add top-level QA owner or approver fields, an approval section, or generation notes. Track stakeholder decisions outside the generated artifact unless they are preserved human notes, manual additions, or exclusions.

If the artifact already exists, read it first and preserve human decisions, manual additions, exclusions, notes, and execution status.

For maintainers reviewing this skill, [the 0.6.0 example report](references/0.6.0-example-report.md) shows a complete generated artifact. Treat it as a dated review snapshot, not as input or an authoritative scope for later releases.

### 7. Verify before handoff

Verify:

- All refs still resolve to the recorded SHAs.
- Candidate discovery used only `previous_ref..release_ref`.
- Main was queried only for release-derived work.
- Every capability has one forward-merge state.
- Every user-visible cherry-equivalent group has a documented disposition.
- No unrelated main-only PR appears.
- Every automatic capability cites at least one release commit, a source PR when GitHub enrichment is available, and one published release-snapshot docs path.
- All cited release docs exist at the release SHA.
- Every capability has `Covered`, `Partial`, or `Missing` release-note coverage backed by the release-note diff and content.
- Commands and Studio paths are source-backed or explicitly need clarification.
- Draft release notes omit excluded and unresolved capabilities.
- Manual additions and exclusions have rationale and owner placeholders.
- Rerunning unchanged inputs produces the same candidate set and preserves review state.

State any verification that could not be completed.

### 8. Optional GitHub issue

Only after explicit authorization:

- Search for an existing open issue for the same version and purpose.
- Prefer updating it over creating a duplicate.
- Use the shared release test-scope artifact as the issue body.
- Apply only existing repository labels.
- Report whether the issue was created or updated and return its URL.

## Completion report

Report:

- Version and the three resolved SHAs.
- Release-range commit and PR counts.
- Docs-touched PR and consolidated feature counts.
- GitHub enrichment status and release-note coverage counts.
- Counts by forward-merge state and disposition.
- Path to the local shared artifact.
- Missing evidence and review decisions.
- State that the output is a draft for stakeholder review.
- GitHub issue URL only when creation or update was authorized.

Do not represent the generated artifact as an approval record; approval and scope finalization happen outside it.
