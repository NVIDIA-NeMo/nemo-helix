---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: nemo-explore
description: Captures what a NeMo Platform agent should do before any code or YAML. Explores the user's codebase and docs first, then asks one intent question at a time for what source cannot supply. Output feeds nemo-ethos. Use over generic brainstorming for any NeMo Platform agent design conversation.
triggers:
  - design my agent
  - what should my agent do
  - help me think through the agent
  - I want to build an agent
  - agent design
  - explore the agent
  - figure out what my agent needs
  - bootstrap AGENT_DESCRIPTION
  - onboard my existing agent
not-for:
  - nemo-skill-selection (use to dispatch when intent is unclear)
  - nemo-ethos (use to write the Ethos file once explore is done)
  - nemo-build-agent (use after the Ethos exists)
  - nemo-model-selection (use for the model question in step 5; explore delegates to it)
  - superpowers:brainstorming (use for design work unrelated to NeMo Platform)
preconditions:
  - nemo_cli_available
compatibility: nemo-platform >= 0.1.0; dialogue-driven with read-only pre-flight (`ls`, `find`, `Read`); one intent question per message after the codebase scan; at least three questions covering Purpose & Outcomes, Principles, and Vision; safe under any sandbox; works offline; output is a structured conversation handed to nemo-ethos.
maturity: active
license: Apache-2.0
user-invocable: true
allowed-tools: Read, Glob, Grep, Bash
metadata:
  author: NeMo Helix Team <nemo-helix@nvidia.com>
---

# NeMo Platform agent explore

Capture what the agent should do before any code or YAML. Product mission and
user goals matter more than implementation inventory. The output of this skill
is the data that `nemo-ethos` writes into `agents/<name>-ethos/ETHOS.md` —
the durable contract that downstream optimization agents read as their primary
context. Underspecified input here directly degrades the quality of generated
Insights and PRs downstream.

This skill is **explore first, then a mandatory intent interview.** Scan the
codebase and docs, infer implementation-shaped fields, then ask the user for
intent source cannot own. Do not skip the interview because the scan looked
complete, and do not skip a question because you think you already know the
answer. Do not dump the draft Ethos until that interview has asked at least
three questions and received a reply to each.

The division of labor is: **infer implementation, ask for intent.** Almost
everything about what the agent *is* can be read from source. Nothing about what
the developer *wants* can be. Constraints, trade-offs, principles, mission, and
change permissions live in the user's head, so they are the questions worth
spending attention on.

## The schema you are filling

The Ethos has five base front-matter fields, plus optional `updated_timestamp`,
and fifteen body sections. Every body section is required. One field is a
quality gate for handoff: `nemo-ethos` is blocked until `Role` is concrete.
For any other canonical section with nothing to say, write `_(none)_` rather
than dropping the heading.

The fifteen headings are a floor, not a ceiling. Extra `##` headings and extra
YAML front-matter keys are allowed. Keep them if the user adds them. Do not
strip custom sections to make the file look "strict."

**Front matter**

| Field | Required | Guidance |
| :---- | :---- | :---- |
| `schema_version` | yes | Always `1` for new files. `nemo-ethos` fills this at write time. |
| `name` | yes | Canonical agent name. Use the directory or workflow name if obvious; ask if not. |
| `created_timestamp` | yes | ISO 8601 timestamp for when the Ethos is created. `nemo-ethos` fills this at write time. |
| `author` | yes | Human or agent that created the Ethos. `nemo-ethos` fills this from the current author context when known; ask only if ambiguous. |
| `owner` | optional | Accountable human or team for the approvals named in `Constraints` or `Change Scope`. Ask only if those sections name an approval. |
| `updated_timestamp` | conditional | Set on edits. Omit on first write. `nemo-ethos` fills this. |

**Body sections** (in canonical order)

| # | Section | What "good" looks like |
| :---- | :---- | :---- |
| 1 | Role | One concrete sentence describing the role this agent plays. Example: "answer IT helpdesk questions about VPN, password reset, and software access." Vague answers ("help with stuff") are useless downstream. |
| 2 | Purpose & Outcomes | Two labeled parts. **Mission:** why the agent exists, what user value it provides, and the product or workflow context it serves — not a restatement of implementation mechanics. **Outcome:** the external result it is accountable for, with the measurable target and who owns that number. A mission with no outcome cannot be optimized; an outcome with no mission gets optimized in the wrong direction. Say so plainly when the agent is internal tooling with no business metric. |
| 3 | Scope | Audience, 3-6 task categories, expected in-scope work, and explicit out-of-scope work/non-goals. |
| 4 | Tools | Tools, APIs, and knowledge sources the agent can use, or "Prompt-only." Group related helpers by capability or source. Capture only behaviorally important purpose, credentials/scopes, side effects, freshness, and expected failures. |
| 5 | Harness | How this agent actually runs: the loop, tool use, and runtime. Write what is true of this agent. Do not pick a named platform harness, and do not treat a framework import as a requirement. Write `_(none)_` if you cannot describe how it runs. |
| 6 | Behavior | Behavioral rules and boundaries: refusal/escalation policy, tone, safety/compliance requirements, accepted limitations, and known non-goals. Hard external limits belong in `Constraints`. |
| 7 | Principles | How the agent should decide when no rule in `Behavior` covers the case: which way to err on an ambiguous request, what it protects even at some cost to the answer, and whose interest wins when the user and the business disagree. Two or three concrete judgment calls. "Helpful, harmless, and honest" is not an answer — it is true of every agent. Write `_(none)_` if there is no judgment call beyond `Behavior`. |
| 8 | Success Criteria | What good production behavior looks like, independent of current evals: mission-level outcomes, quality standards, escalation quality, accuracy expectations, and examples of success. Rank them when some matter more. |
| 9 | Trade-offs | How to choose when two improvements conflict. Needs three things: hard gates never traded away, a priority order over the rest (quality, latency, cost, reliability), and regressions that are unacceptable even alongside a headline win. "Balance quality and cost" is not an answer. Write `_(none)_` if the user cannot rank them. |
| 10 | Constraints | Hard external bounds no change may cross: approved providers/models/regions, data residency and handling, compliance obligations, production cost ceilings and latency SLOs, and changes that need human sign-off. Give the current measured figure next to a ceiling when you know it. Usually organizational, which is why the code cannot supply them. Write `_(none)_` if unconstrained. |
| 11 | Evaluation Setup | Current validation setup: how to run it, what datasets/checks it uses, what scorers/metrics measure, pass/fail thresholds, and known coverage gaps relative to the success criteria. If no eval suite exists, say so explicitly. |
| 12 | Metric Semantics | What ambiguous or load-bearing metric and telemetry field names actually mean, and which claims they do not support. Write `_(none)_` when every name means exactly what it says. |
| 13 | Change Scope | A permissions list — what may be modified. Each lever takes `yes`, `no`, or `with-approval`. Name levers that exist on this agent. Do not copy a platform catalog. The loop never edits the Ethos itself. |
| 14 | Vision | Where the agent is headed: an intention beyond today's job, plus one or two concrete use cases it should grow into but does not serve yet. That last part marks what `Scope` excludes *for now* rather than on principle. Write `_(none)_` rather than pasting a dated backlog. |
| 15 | Open Questions | Open facts that affect safe use, evaluation, or modification of the agent. Write `_(none)_` when there are none. Remove items once answered. |

Known issues / failure patterns are tracked as first-class Insight entities by
the insights plugin — do not duplicate them into the Ethos.

## Pre-flight

Check whether an Ethos already exists for this agent. If `agents/<name>-ethos/ETHOS.md`
is present, ask the user whether they want to edit the existing Ethos or start
over. If they want to edit, route to `nemo-ethos` directly.

```bash
ls agents/*-ethos/ETHOS.md 2>/dev/null || echo "no ethos yet"
```

If `agents/<name>-spec/AGENT-SPEC.md` exists, the agent still has a spec
package. Read that file as prior answers. Scan the codebase, then still run
the intent interview for anything the spec never answered. Hand those
answers to `nemo-ethos` so it writes `agents/<name>-ethos/ETHOS.md`.

```bash
ls agents/*-spec/AGENT-SPEC.md 2>/dev/null && echo "spec package present"
```

After the Ethos is uploaded and the user confirms it, copy remaining package
files such as `agent.yaml` into `agents/<name>-ethos/`. Do that when those
files still live only in the spec package. Confirm, then delete
`agents/<name>-spec/` and the `<name>-spec` Fileset:

```bash
nemo files filesets delete "${NAME}-spec"
```

Do not start a greenfield explore unless the spec is missing answers you
still need.

## Step 1 — Explore the codebase

Time-box this to ~5 minutes of tool use. Read first, ask second. Greenfield
projects will turn up nothing here, which is fine — move to the intent
interview and ask for Role first.

1. **Find agent definitions and entry points.** Look for Platform
   `agent.yaml`, NAT workflow YAMLs, Python agent builders, system prompts,
   skills, and tool definitions:

   ```bash
   find . -maxdepth 5 -type f -name "agent.yaml" 2>/dev/null
   find . -maxdepth 4 -type f \( -name "*.workflow.yaml" -o -name "*.workflow.yml" \) 2>/dev/null
   find . -maxdepth 4 -type d -name "agents" 2>/dev/null
   ```

   Then use `Glob` / `Grep` to find `nemo-agents-spec-v1`,
   `langgraph`, `StateGraph`, `create_react_agent`, `system_prompt`, skills,
   MCP servers, and tool definitions. Treat any harness or framework name you
   find as a clue about how the agent runs, not as a value the Ethos must pick.

2. **Find design context.** Look for `README.md`, `AGENTS.md`,
   product/design/planning docs, launch notes, and anything in `docs/`. Read
   docs that describe goals and user value before implementation details when
   they look agent-relevant.

3. **Map findings to schema fields.** As you scan, hold a running mental
   table of what you can fill from the code/docs. Be honest about confidence:
   "inferred from system prompt" is different from "confirmed by the user."

4. **Choose the model.** Hand off to `nemo-model-selection` after the
   code/docs scan. That skill profiles the agent on tool density, primary
   capability, and deployment, then recommends a specific NIM model with a
   plain-English explanation grounded in what the model is actually good at.
   Return here with the chosen model string, which `nemo-build-agent` writes
   into `agent.yaml`. The Ethos has no `Model` section — record the *permitted*
   providers and model families in `Constraints` instead, since the config
   already carries the model in use and it changes without touching the Ethos.
   If the user wants to skip the conversation, the default is cloud,
   `nvidia/llama-3.3-nemotron-super-49b-v1` — announce that and move on.
   Local NIMs require host-gpu mode.

   Typical inferences per field:

   - **name** — directory name, workflow name, or top-level package name.
   - **Role** — first paragraph of README, system prompt preamble, or
     top-level docstring. Often partial; usually needs user confirmation.
   - **Purpose & Outcomes** — mission from product docs, README motivation,
     system prompt preamble, or workflow context. Treat that as a draft to
     confirm, never as a skip. The outcome (the business objective) is rarely
     in code; even when a README names a metric, ask the user to confirm it.
   - **Scope** — audience from docs or prompts; categories from enumerated
     capabilities or named tool clusters; in/out boundaries from prompt rules.
   - **Tools** — from `@tool` decorators, NAT tool registry,
     `create_react_agent(tools=[...])`, retrieval/corpus config, or API clients.
     Group low-level helpers when they share credentials, side effects,
     freshness, and failure modes.
   - **Harness** — infer from how the agent actually runs: adapter
     configuration, workflow YAML, service entrypoints, CLI commands,
     Dockerfiles, notebooks, or deployment configs. Capture behaviorally
     relevant capabilities, not a catalog name. If you cannot see how it runs,
     write `_(none)_`.
   - **Behavior** — system prompt rules ("never give medical advice"),
     refusal/escalation policy, tone, accepted limitations, and non-goals.
   - **Success Criteria** — desired production outcomes, product goals,
     quality standards, escalation quality, accuracy expectations, and examples
     of successful behavior.
   - **Trade-offs** — not in the code; ask in the intent interview.
   - **Constraints** — mostly not in the code, but scan for partial evidence
     worth confirming: a pinned gateway base URL or provider allowlist, region
     settings, redaction or PII middleware, timeout and token ceilings, and
     compliance notes in docs. Treat findings as a starting draft to confirm,
     never as the complete list.
   - **Evaluation Setup** — Makefile targets, scripts, CI config, eval YAMLs,
     metric definitions, thresholds, and coverage notes.
   - **Metric Semantics** — from scorer definitions, metric names in eval
     configs, and telemetry field names. Fill only the entries whose meaning is
     genuinely ambiguous from the name. Write `_(none)_` when every name is
     obvious.
   - **Change Scope** — not in the code; ask the user.
   - **Principles** — not in the code; ask in the intent interview even if a
     prompt hints at one. A prompt is the implementation, not the intent
     behind it.
   - **Vision** — not in the code. Roadmap docs, design notes, or a README's
     future-work section are speculative until the user confirms them. Ask.
     Write `_(none)_` rather than guessing.
   - **Open Questions** — TODOs / FIXMEs in agent-adjacent code that
     affect safe use, evaluation, or modification.

## Step 2 — Ask intent questions

The scan told you what the agent is. It cannot tell you what the developer
wants. Run a real interview for that intent before you show a draft Ethos.

This step is a hard gate. Do not present the full draft, and do not hand off
to `nemo-ethos`, until you have asked at least three questions and received a
reply to each. "The scan filled everything" and "asking feels unnecessary"
are not reasons to skip it.

### What to ask

After the scan, split fields into two piles:

- **Inferred** — implementation-shaped fields you can draft from source:
  `Tools`, `Harness`, `Evaluation Setup`, `Behavior` copied from prompts,
  `Metric Semantics` from scorer names.
- **Intent** — fields source cannot own: `Purpose & Outcomes`, `Principles`,
  and `Vision` always; then `Constraints`, `Trade-offs`, `Change Scope`, and
  a ranked `Success Criteria` when the repo only has eval wiring.

Walk the intent pile. Skip a topic only when **this conversation** already
answered it. A README, prompt, or roadmap is not a substitute for a user
reply on the always-ask topics below. Keep every other unanswered intent
topic on the list. Greenfield work adds `Role`, `Scope`, and whether the
agent is prompt-only before that intent list.

Question count is at least three, then however many remaining intent topics
still need a reply. Prefer fewer sharp questions over a long checklist, but
never go below three.

### Always ask these three

Code and docs can look like they already answered them. They did not, until
the user says so. Ask even when you have a high-confidence draft. Confirming
an inference counts; silently filling the section does not.

1. **`Purpose & Outcomes`** (business objective). Confirm the mission and the
   result the agent is accountable for. If you inferred a metric from a
   README, show it and ask whether that is the target.
2. **`Principles`**. The judgment call when `Behavior` runs out. Reject
   generic virtues. Ask what this agent should do that a careless version
   of it would not.
3. **`Vision`**. Where the agent is headed, or an explicit `_(none)_`. Do
   not copy a future-work section into Vision without asking. Speculative
   vision is the usual failure.

If `Role` is missing or vague, ask that first so the rest of the interview
is grounded. It does not replace one of the three always-ask topics.

Ask about remaining topics in this order, skipping any this conversation
already answered:

1. `Constraints` — hard bounds no change may cross.
2. `Trade-offs` — hard gates, then a ranking.
3. `Change Scope` — which levers on this agent may move.
4. Ranked `Success Criteria` when the repo only has eval wiring.

Do not collect run limits. A per-experiment spend cap belongs to the
optimizer. If the user volunteers one, keep the standing policy and drop
the number.

### How to ask

Follow this Q&A pattern:

- **One question per message.** If a topic needs more depth, ask a
  follow-up in the next message. Do not batch four intent questions.
- **Prefer multiple choice.** Ground the options in what you found.
  Always include a way to reject the list (`Something else` or `I don't
  know`). Open-ended is fine when a lettered list would fake certainty.
- **Wait for the reply** before the next question.
- **Accept "I don't know"** and move on. Write `_(none)_` and record the
  gap in `Open Questions`. A fabricated constraint is worse than a missing
  one.
- **Confirm inferred bounds.** If the scan found a pinned gateway or
  redaction middleware, show it and ask whether it is a real boundary.
- **Confirm inferred intent.** If the scan produced a plausible `Purpose &
  Outcomes`, `Principles`, or `Vision`, show the draft as an option. Do not
  treat that draft as the answer.

Example (one message, then stop):

> The README says this agent exists to "cut ticket volume." Is that the
> business objective I should record in `Purpose & Outcomes`?
>
> A. Yes — ticket volume is the outcome (tell me the target if you have one)
> B. Close, but the real objective is … (tell me)
> C. No business metric — internal tooling
> D. I don't know

A later Constraints example:

> The scan shows a pinned NVIDIA gateway and no cost ceiling in docs.
>
> Which hard bounds should `Constraints` record?
>
> A. Keep the gateway pin; no other bounds
> B. Gateway pin plus a production cost or latency ceiling (tell me the number)
> C. No hard bounds — write `_(none)_`
> D. Something else

Reject generic virtues for `Principles`. "Helpful, harmless, and honest"
gives a downstream reader nothing. Ask for the judgment call: what does
this agent do that a careless version of it would not? If nothing comes
back, write `_(none)_`.

Push once on a non-answer for `Trade-offs`. "Balance quality and cost" is
not decidable. One concrete follow-up usually produces a ranking. If it
does not, record the ambiguity and move on.

If the user provides outside context, read it and update the inferred
draft before the next question.

### Red flags — stop and ask

These mean you skipped the interview:

- You are about to paste the full Ethos and have asked fewer than three
  questions
- You filled `Purpose & Outcomes`, `Principles`, or `Vision` from the scan
  without a user reply
- Several intent questions in one message
- Handing off to `nemo-ethos` with fewer than three Q&A replies in this
  conversation

| Excuse | Reality |
| --- | --- |
| The scan filled everything | Code never owns intent. Ask at least three questions. |
| I already know Purpose / Principles / Vision | Those answers belong to the user. Confirm the draft. |
| Asking feels unnecessary | Unnecessary-looking questions are the ones that catch speculative Vision. |
| Batching questions saves turns | One question gets a real answer. A dump gets shallow ones. |
| The draft review can collect intent | Review is for corrections. Ask intent first. |
| The user looks busy | One short multiple-choice question is enough to start. |

## Step 3 — Present the draft Ethos

After the interview, present the entire Ethos at once — every field, with
inferred values shown inline and remaining gaps as `_(none)_`. Then ask
one question:

> "Here's the full Ethos I'd write. Tell me what to change, and I need a
> concrete Role before I can hand off to `nemo-ethos`."

Show the rendered Ethos inline in markdown (one `##` section per field,
same shape as the on-disk file). For fields you defaulted, note the
default in parentheses so the user knows they can override:

- `Tools: Prompt-only.` *(default — say so if the agent needs tools)*
- `Purpose & Outcomes` / `Success Criteria` inferred from implementation
  *(say so if there is outside context to incorporate)*
- `Change Scope:` name the parts of this agent a change may touch
  *(default — call out anything you want to lock down, or mark it
  `with-approval`)*

When the user could not answer a section, write `_(none)_` and add the
gap to `Open Questions`. Say "`Constraints`: _(none)_ — a later optimizer
might treat every provider and cost as fair game" so the consequence is
visible while it is still cheap to fix.

Do not walk the schema field by field in this review. Do not restart the
intent interview here unless the reply surfaces a contradiction.

Do not use public-facing shorthand like `AUT` or "agent under test" in the
rendered Ethos. Use "this agent" for the agent being specified. Use
"target agent" only where the agent's purpose is explicitly to inspect or
modify another agent, and name optimizer helper agents only when they are
part of the actual product workflow.

Allowed follow-ups after the draft:

1. The **hard-required** `Role` is missing or vague — ask for it before
   handoff.
2. The user's reply surfaces a contradiction that needs one targeted
   clarification (for example they say "drop the search tool" but the
   codebase shows the agent depends on it).

## Step 4 — Hand off

After the user's reply, apply the corrections and check the one hard
precondition:

1. **Role** is a concrete one-sentence answer (not "help with stuff").

If it is still unresolved, ask for it in one final message and stop until the
user provides it. Do not hand off with `Role` blank — the artifact is useless
downstream even though the parser accepts it.

Every body section heading must be present before handoff. Write `_(none)_`
for honest gaps, and note them in `Open Questions` when they affect safe
optimization. Do not drop a heading.

If the Role quality gate is satisfied, announce the handoff in one line
("Handing off to `nemo-ethos` to write `agents/<name>-ethos/ETHOS.md` and upload
the canonical copy to Filesets") and trigger it.

## If the user pushes back

- **They want to change one or two fields.** Apply the edits, re-show the
  changed sections only, ask "good now?", proceed.
- **They want to redo the whole thing.** That usually means the codebase
  scan got something fundamentally wrong. Re-scan with their correction in
  mind, then re-present once.
- **They keep changing their mind on Role.** Stop. Tell them the agent will
  not be useful until they can write one concrete sentence and offer to
  come back later. Do not loop on rewording.
- **They skip remaining intent questions.** You already asked at least
  three, including `Purpose & Outcomes`, `Principles`, and `Vision`. Treat
  skipped topics as `I don't know`: write `_(none)_`, record them in
  `Open Questions`, and say once what it costs. Then present the draft. Do
  not re-open the full interview.
- **They say "just write it" before three questions.** Ask the next
  always-ask question anyway. After they answer or decline each of the
  three, continue.

## Gotchas

- **"You decide" means commit to the default and announce it.** Example:
  "I'll go with cloud and `nvidia/llama-3.3-nemotron-super-49b-v1`. Tell me
  to change if not." Never silently fill in. Prefer routing through
  `nemo-model-selection` so the user gets a plain-English reason, not just a
  name.
- **Tool over-spec is the most common error.** Users ask for a search tool
  when prompt-only would work. Probe: "Do you have evidence the model alone
  fails on these?" If no, drop the tool.
- **Tool and harness inventory should be compressed.** Do not create one row
  per helper method when several helpers share the same source, credential,
  side effect, freshness, and failure mode. Group them and call out only the
  differences an optimizer or evaluator needs to know.
- **Mission before mechanics.** An Ethos that only says how the current code is
  wired is not good enough. If goal context cannot be found in the codebase or
  docs, say the mission is inferred from implementation and still ask the user
  to confirm `Purpose & Outcomes`. A guessed business objective is worse than
  `_(none)_`.
- **Never infer a constraint you cannot verify.** A guessed provider allowlist
  or cost ceiling silently deletes good candidates and looks authoritative doing
  it. Ask, or write `_(none)_` and record the gap in `Open Questions`.
- **Trade-offs are the highest-leverage question you will ask.** Without a
  priority order, every candidate that improves one metric and regresses another
  is undecidable, and the optimizer either stalls or picks arbitrarily.
- **"No behavior constraints" usually means "I haven't thought about it."** Probe
  once: "Anything that should never appear — names, phone numbers, competitor
  mentions?" One probe, then move on.
- **Do not skip the codebase scan even when the user seems eager to
  answer questions.** Spending the first five minutes reading makes the
  interview shorter and sharper. Asking something the codebase already
  answers loses trust immediately.
- **Do not skip the intent interview even when the scan looks complete.**
  Ask at least three questions. Always ask `Purpose & Outcomes`,
  `Principles`, and `Vision`, even when you think you already know. One at
  a time. Multiple choice when you can.
- **Do not invent Vision from a backlog.** A future-work bullet is
  speculative until the user confirms it. Ask, or write `_(none)_`.
- **A framework import is not a harness.** Describe how the agent runs. Do not
  map an import onto a platform harness name. Imports such as `langchain`,
  `langgraph`, `crewai`, `autogen`, or `pydantic_ai` say nothing about that.
- **Keep Platform terminology at the design boundary.** Record the desired
  harness behavior and artifacts without exposing Fabric SDK types or asking
  the user to design a raw runtime config. `nemo-agent-config` owns the
  machine-readable Platform YAML after the Ethos is approved.
- **Change Scope is a permissions list, not a wishlist.** It controls what
  later optimization may edit. Walk the levers that exist on this agent so
  the user knows what they're consenting to. Offer `with-approval` for levers
  the user wants available but not automatic. Do not copy a platform catalog.
- **Do not invent Known Issues fields.** Known issues / recurring failure
  patterns live in the Insights plugin as first-class entities, not in the
  Ethos.
