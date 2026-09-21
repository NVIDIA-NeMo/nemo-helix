<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Release-note contract

Use this contract for both drafting and auditing `docs/about/release-notes/current-release.mdx`.

## Audience and voice

Write for users deciding whether to adopt or upgrade NeMo Platform. Lead with what they can now accomplish or what behavior changed. Mention implementation details only when they affect setup, compatibility, performance, security, or operations.

- Use present tense and concrete product names.
- Do not expose private issue identifiers, internal project names, contributor workflow, or branch mechanics.
- Do not turn commit subjects into prose without verifying the resulting behavior.
- Do not promise availability beyond what the candidate tree, packaging, feature flags, and published docs support.

## Document shape

Preserve the repository license header and the existing frontmatter shape. In draft mode, set the title from the required inputs using the existing convention:

```yaml
title: "v<MAJOR.MINOR.PATCH> - <YYYY-MM-DD>"
description: ""
```

Do not infer a patch, minor, or major version from the branch name. Preserve a meaningful existing description or write a concise release summary when the repository begins using that field consistently.

A feature release normally uses:

```md
## Highlights

<A short list of the most consequential outcomes.>

## What's included

### <Established product area>

#### <User outcome>

**Description:** <What changed, who benefits, and any essential boundary.>

**Documentation:** [<Task or concept title>](/documentation/<canonical-path>)

**Use it:**

- **CLI:** `<verified command>`
- **Studio:** Open **<verified navigation labels>**.
```

Include only applicable interface bullets. When neither CLI nor Studio applies, keep the field explicit:

```md
**Use it:** This behavior applies automatically; there is no separate CLI or Studio entry point.
```

For an outcome the user explicitly chose to include without documentation, write:

```md
**Documentation:** Not yet available.
```

Do not use that marker without the explicit interactive decision described by the skill. Headless drafting omits such outcomes from the feature list.

After product areas, use only the release-level sections supported by evidence:

- `## Install`
- `## Upgrade from <version>`
- `## Compatibility`
- `## Current constraints`
- `## Known issues`
- `## Maintenance`
- `## Links`

Patch releases may use a shorter `## What's fixed` structure, but each substantial user-visible fix still needs its description, documentation status, and applicable interface recorded. Combine those fields into compact prose only when repeating them would make a short patch note harder to read.

## Documentation coverage

A repository documentation pointer must satisfy all of these:

1. The source exists in the candidate tree.
2. The page is included, directly or through its section, in `docs/fern/versions/latest.yml`.
3. The canonical `/documentation/...` URL matches the Fern navigation slug.
4. The page actually explains the released outcome or task, rather than merely mentioning its name.

Generated API or CLI reference pages can support exact syntax. Prefer a task, tutorial, concept, or deployment guide as the primary feature pointer.

## Inclusion rubric

Include an outcome when a user would need it to answer at least one of these questions:

- What can I do now that I could not do before?
- What behavior, result, or supported workflow changed?
- What must I change during installation or upgrade?
- Which compatibility, security, performance, or operational property changed materially?
- Which significant defect no longer affects me?

Normally omit individual entries for tests, formatting, code organization, generated-file refreshes, dependency movement without user impact, and CI-only behavior. If these changes collectively affect release reliability or supported environments, summarize the verified effect rather than listing their mechanics.

## Coverage report

Use a compact table after either mode:

| Outcome | Source | Documentation | Interface | Disposition |
| --- | --- | --- | --- | --- |
| <name> | `<short SHA>` ... and candidate-only, forward-merged, or cherry-equivalent | Published link or `missing` | CLI, Studio, automatic, API/SDK | Included, maintenance, already documented, omitted, or needs decision |

Follow the table with separate short lists for:

- forward merges and cherry-equivalent groups inspected;
- differences between release-note content at the release and candidate refs;
- undocumented or gated outcomes;
- claims that could not be verified;
- validation commands and their exact status.
