<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Local Ethos for Eval Author

This procedure owns locating, checking, creating, and reviewing the repository's
`ETHOS.md`. Read and write the document locally. The caller owns when this
prerequisite is required.
When called from a fresh guided audit, begin only after its opening has ended
and the user has replied to continue. Loading this procedure alongside the audit
skill does not authorize starting its reads, validation, or generation early.

## Bundled authoring skill

For new or revised intent, load the sibling [ethos skill](../../ethos/SKILL.md).
It combines repository exploration, intent questions, authoring, and review in
one file with an inline schema-v1 outline. Pass the selected agent, existing
answers, chosen local path, and review state. The procedure below supplies
Eval Author's introduction, write boundaries, and milestone checkpoints; use
one interview and preserve those caller-specific requirements. Return with the
saved path, gaps, checks, and review state before continuing Eval Author.
An already suitable Ethos can go directly through the locate-and-reuse checks.
The skill ships with Eval Author.

## Introduce Ethos

Before checking or creating the document, explain in ordinary language that
Ethos records what the agent should do, what it should avoid, and what a good
result looks like. Evals and audits use this as their target. Link the
[bundled Ethos skill](../../ethos/SKILL.md)
and make the explanation concrete for the selected agent. This introduction also
applies when reusing an existing file; do not assume familiarity with the term.
Reuse an introduction already given in the current flow.

## Locate and reuse

Use the caller's explicit path when supplied; otherwise use root `ETHOS.md`.
Read the file and apply the checks below. Resolve ambiguous agent identity with
the user. A suitable unchanged document can be returned without another
interview or content approval; still honor the caller's introduction and milestone
check-in, including a guided audit's linked summary and opportunity to edit it.
File presence, empty placeholders, README content,
code, traces, and intent notes do not substitute for a substantive Ethos.

When onboarding has not yet identified the agent, use the user's description and
a bounded look at the root README and directly named agent documentation or entry
point. Ask which agent is intended if ambiguous. Do not scan eval reports, extract
conversations, or run Harbor to establish the agent's identity and purpose.

For a new document, default to `<repo-root>/ETHOS.md` or the user's chosen in-repo
location under the core's narrow Ethos write exception. Preserve existing content,
custom sections, and edits. Reuse prior interview answers, including saved
`.eval-author/intent-notes.md`.
Do not commit automatically. Recording change permissions does not itself change
the agent, prompts, model, or runtime.

## Capture intended behavior

After the introduction, read source and documentation for implementation
facts; ask focused questions only for intent they cannot establish, reusing answers
already given. Keep the current agent, model, and runtime unless asked to change them.

Ground questions in actual workflows and desired outcomes. Avoid abstract choices
such as demo versus production accountability unless they affect the requested
work. End intent messages with the question. Ordinary intent gathering is not a
request for permission to follow a skill or a blocked-work report; do not append
procedural footnotes merely to justify it. Preserve any explicit host disclosure
requirement for a genuine approval or unavailable prerequisite.

Ethos describes the selected agent's overall intended behavior, including areas
outside the first eval set. Existing cases show what is tested, not the complete
intent. Preserve broader confirmed workflows and record unconfirmed areas in Open
Questions; do not infer approval from implementation or rewrite intent to match
old tests. Surface conflicts with existing scoring for the user's resolution.
A limited first eval set need not wait for unrelated details, but explain any gaps
instead of presenting a partial Ethos as a complete account of the agent.

Use [the local template](../templates/ETHOS.md). A concrete Role and Purpose &
Outcomes are required; other sections contain actual answers or an honest
`_(none)_`, with uncertainties recorded in Open Questions. Do not invent metrics
or leave template placeholders. Treat demo policies as fixtures when that is the
scope. Missing production approval, SLAs, or business metrics need not block a
demo Ethos; production claims require resolving the relevant policy authority.

## Save, check, and review

Write the complete document. Use `schema_version: 1`, the actual agent name and
author, and an ISO 8601 creation timestamp. On edits, retain creation metadata,
add `updated_timestamp`, preserve unknown frontmatter keys and custom sections,
and make targeted changes instead of replacing an existing file with the template.

Read back the exact file and check:

1. It is nonempty and at the selected in-repo path.
2. Frontmatter is a mapping with version 1, nonempty `name` and `author`, and valid
   ISO 8601 creation and optional update timestamps.
3. Each of the template's fifteen `##` headings occurs once. Extra headings are
   allowed. Role and Purpose & Outcomes express concrete intent; remaining sections
   have real content or honest `_(none)_` values, with no template placeholders.
4. Purpose, boundaries, and change permissions agree with the user's answers;
   implementation references do not establish user approval.

If the caller has execution tools and an existing local interpreter supports YAML,
use its safe loader for frontmatter; otherwise use structural inspection.
Report whether checks were structural inspection or parser-backed, and fix local
errors before calling the document complete. These checks do not prove evaluation
coverage.

Present newly authored or revised content for review: link the saved file,
summarize its role, outcome, boundaries, and open questions, then name the caller's
actual next step. For fresh onboarding, Harbor follows Ethos; for a later return,
use the next unfinished milestone; for a guided audit, it is **Confirm evals and
traces** unless that milestone and its check-in are already complete, then defining
the coverage specification. A scoped audit-generation request proceeds to its
requested drafting step. For example during onboarding:

> I've generated your `ETHOS.md` for review. It describes the
> agent's intended baggage-policy behavior and limits it to the demo policies.
> Once you confirm it, I'll introduce Harbor, the framework we'll use for the
> evals, and check its setup.
>
> Does ETHOS.md look right, or would you like to change anything before we move on?

Wait for review of the saved content before using it for cases or an audit;
interview answers alone are not document review. Apply requested revisions,
recheck, and present the revised document. Reuse an existing approval of the exact
saved content. Return the checked path, substantive intent, remaining gaps, and
review state to the caller, which handles its next milestone.

## Recovery

If writing fails, preserve confirmed answers in the conversation or an existing
writable `.eval-author/intent-notes.md`. Explain the local error, provide the
complete proposed content and the bundled Ethos skill link above, and ask for the
saved path if the user must save it themselves. A user who prefers to create the
file can also supply that path. Missing intent calls for a focused question;
filesystem problems call for local recovery.
