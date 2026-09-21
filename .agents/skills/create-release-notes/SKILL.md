---
name: create-release-notes
description: Draft or audit NeMo Platform release notes by comparing a release ref with a candidate ref, reconciling forward-merged release-line changes, checking published documentation coverage, and recording verified CLI or Studio entry points. Use when asked to create, update, compare, review, or check release notes for a NeMo Platform release. Not for publishing artifacts or running the release workflow.
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Create NeMo Platform release notes

Create evidence-backed product release notes from local Git history. Work at the level of user outcomes, not commit subjects or pull requests.

## Inputs and modes

Require a release ref. Resolve it locally; do not fetch or require GitHub access. The candidate ref defaults to `HEAD`.

- **Draft** is the default. Require a target version and release date, update `docs/about/release-notes/current-release.mdx`, and finish with a coverage report.
- **Audit** is read-only. Check an existing release note, defaulting to `docs/about/release-notes/current-release.mdx`, and return the coverage report without editing it.

If a required draft input is missing, ask for it before editing. Accept an explicit alternate notes path. Never publish a release, create a tag, or run the release workflow as part of this skill.

## 1. Load repository policy and examples

Read the root `AGENTS.md`, `AGENTS.local.md` when present, `docs/AGENTS.md`, and `RELEASING.md`. Read the current release note and the two most recent versioned notes in `docs/about/release-notes/` to preserve current terminology and release-level context.

Read [references/note-contract.md](references/note-contract.md) before classifying or writing entries.

Preserve unrelated worktree changes. In draft mode, inspect the notes file's working-tree diff before editing and reconcile existing authored content instead of replacing it blindly.

## 2. Resolve and characterize the comparison

Use task-specific shell variables rather than generic environment names:

```bash
NMP_RELEASE_REF='<required release ref>'
NMP_CANDIDATE_REF='<candidate ref, default HEAD>'

git rev-parse --verify "$NMP_RELEASE_REF^{commit}"
git rev-parse --verify "$NMP_CANDIDATE_REF^{commit}"
git merge-base "$NMP_RELEASE_REF" "$NMP_CANDIDATE_REF"
git rev-list --left-right --count "$NMP_RELEASE_REF...$NMP_CANDIDATE_REF"
```

Stop if either ref does not resolve or there is no merge base. Record both resolved SHAs in the final report. Do not claim the local refs are current with their remotes.

Collect complementary views of the change. No single view is sufficient:

```bash
git log --right-only --cherry-pick --no-merges \
  --format='%H%x09%cs%x09%s' \
  "$NMP_RELEASE_REF...$NMP_CANDIDATE_REF"

git log --left-right --cherry-mark --no-merges \
  --format='%m%x09%H%x09%cs%x09%s' \
  "$NMP_RELEASE_REF...$NMP_CANDIDATE_REF"

git diff --find-renames --name-status \
  "$NMP_RELEASE_REF...$NMP_CANDIDATE_REF"

git diff --find-renames --stat \
  "$NMP_RELEASE_REF...$NMP_CANDIDATE_REF"

git log --right-only --first-parent --merges \
  --format='%H%x09%P%x09%cs%x09%s' \
  "$NMP_RELEASE_REF...$NMP_CANDIDATE_REF"

git diff --find-renames "$NMP_RELEASE_REF..$NMP_CANDIDATE_REF" -- \
  docs/about/release-notes/

git log --left-right --cherry-mark \
  --format='%m%x09%H%x09%cs%x09%s' \
  "$NMP_RELEASE_REF...$NMP_CANDIDATE_REF" -- \
  docs/about/release-notes/
```

Use the candidate-only commit list to discover changes, the cherry-marked list to retain patch-equivalent/cherry-picked changes for coverage review, the tree diff to verify the resulting product state, and the first-parent history to analyze integration and forward merges. Use the release-note tree diff and history to establish what each ref actually documents. Inspect relevant patches, tests, configuration, docs, and generated surfaces before describing behavior. Do not derive release notes from filenames or subjects alone.

In `--cherry-mark` output, `=` means Git found a patch-equivalent change on both sides; it does not mean the change is documented. Add every user-visible equivalence group to the evidence ledger and check it against the note content at both refs. Treat `<` and `>` as side-specific evidence, not automatically as release-worthy or already covered.

## 3. Reconcile forward merges

Classify a merge by ancestry, not wording. For each candidate-side first-parent merge:

1. Read its ordered parents with `git show -s --format='%P' <merge>`.
2. Treat the first parent as the candidate-line parent.
3. A later parent is a release-line parent only when `git merge-base --is-ancestor <parent> "$NMP_RELEASE_REF"` succeeds.
4. Commit text such as `forward merge` or `from .../release/X.Y` may corroborate the classification but cannot establish it.

For a confirmed release-line parent, enumerate what that merge introduced relative to its first parent:

```bash
git log --cherry-pick --right-only --no-merges \
  --format='%H%x09%cs%x09%s' \
  '<first-parent>...<release-line-parent>'
```

Inspect the merge result against its first parent when conflict resolution may have changed behavior:

```bash
git diff --find-renames '<first-parent>..<merge>'
```

Do not create an entry for the merge wrapper. Add the underlying change to the evidence ledger only if it is user-visible and is not already described in the applicable current, patch, or prior release note. This second coverage pass is mandatory even when the release ref is an ancestor of the candidate and the ordinary three-dot diff contains none of the release-line commits.

Also audit user-visible cherry-equivalent groups from the comparison step even when they did not travel through a detectable forward merge. A patch being present on both refs is a code deduplication fact, not release-note coverage. Check the release-note diff and the actual note text before assigning `already documented`.

When several forward merges contain the same patch or outcome, deduplicate by patch equivalence, resulting code state, and user outcome. Do not deduplicate solely because titles look similar.

## 4. Build the evidence ledger

Cluster related commits into one user outcome. Maintain a working ledger with:

| Field | Required evidence |
| --- | --- |
| Outcome | Concise user-visible capability, behavior change, fix, or migration |
| Source | Commit SHAs and whether they are candidate-only, forward-merged, or cherry-equivalent |
| Product area | Agent Evaluation, Agents and Fabric, Studio, Customizer, Platform/CLI/Deployment, or another established area |
| Resulting behavior | Code, schema, configuration, tests, or UI evidence from the candidate tree |
| Documentation | Published Fern URL and source path, or `missing` |
| Interface | Verified CLI command, Studio navigation, another public surface, or an explicit reason neither applies |
| Disposition | Include, combine, maintenance summary, already documented, omit, or needs decision |

Include user-visible features, behavior changes, important fixes, migrations, compatibility changes, and material operational changes. Summarize internal refactors, tests, CI, dependency refreshes, and maintenance compactly unless they change installation, security, compatibility, performance, or operations for users.

Search the whole candidate tree for corroborating docs and interfaces; do not assume documentation changed in the comparison range. A local Fern page counts as published only if it is reachable from `docs/fern/versions/latest.yml`. A generated CLI reference may verify syntax, but a command listing alone does not replace conceptual or task documentation for a substantial feature. Official external documentation may supplement local docs; it substitutes for local product documentation only when the capability genuinely belongs to that external product.

Verify CLI syntax in command definitions, generated reference docs, or published task documentation. Verify Studio labels and navigation in the candidate UI source or published docs. Never invent a command, flag, route, menu label, limitation, or compatibility claim.

## 5. Resolve undocumented candidates

Before editing in draft mode, group every otherwise-includable outcome whose documentation field is `missing` and ask the user to choose which outcomes to:

- omit from this release;
- include while explicitly stating that documentation is unavailable; or
- defer until documentation is added.

Ask once for the group rather than interrupting for each outcome. If interaction is unavailable or the user requested a headless/non-interactive run, omit undocumented outcomes from the feature list and report them as release-readiness gaps. Do not silently treat an undocumented feature as complete.

Audit mode never needs this pause: mark each undocumented outcome as a gap and state whether the existing note currently includes it.

## 6. Draft or audit the note

Follow the structure and wording rules in [references/note-contract.md](references/note-contract.md).

In draft mode:

- Preserve accurate existing content and metadata, then reconcile it with the ledger.
- Use one structured block per user outcome, combining tightly related commits.
- Use exact canonical Fern URLs for documentation.
- Put internal-only items in a compact maintenance section; do not inflate them into highlights.
- Keep install, upgrade, compatibility, constraints, and known-issue sections only when supported by the candidate tree and relevant to the target version.
- Inspect the final diff and ensure every factual claim maps to ledger evidence.

In audit mode, compare every existing note entry to the ledger and report unsupported claims, missing user-visible outcomes, duplicate coverage, missing or gated documentation, unverified interfaces, and stale compatibility or constraint text.

Always reconcile the release-note files at the two refs. Flag a user-visible change present on both code lines but absent from both sets of notes, and flag note content present on only one ref when that difference does not match the intended release scope.

## 7. Validate and report

For a drafted note, follow `docs/AGENTS.md` and run:

```bash
make docs-check
make docs-broken-links
```

Do not claim a blocked or failed check passed. If the checks modify tracked files, inspect those changes before including them.

End both modes with:

1. the resolved release and candidate refs and SHAs;
2. a ledger summary showing outcome, source, documentation, interface, and disposition;
3. forward merges and cherry-equivalent groups examined, including any previously undocumented release-line changes retained;
4. documentation and release-readiness gaps;
5. exact validation commands and results;
6. in draft mode, the notes file changed.

Keep uncertainty visible. When evidence cannot establish whether separate changes form one feature or whether a behavior is intended for release, mark it `needs decision` instead of guessing.
