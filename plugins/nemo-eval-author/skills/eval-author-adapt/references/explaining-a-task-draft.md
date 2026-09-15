<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Explaining a task draft

Use this at the first task-creation milestone, especially when the user has
never worked with Harbor. Explain the result in the reply itself; a README
link is useful for inspection but must not carry the entire explanation.
Scale later updates to what changed rather than repeating the introduction.

Refresh the core's shared checklist at this milestone. Keep its high-level labels;
explain the created files beneath **Prepare your Harbor evals** instead of replacing
the user's map with a technical checklist. For example, when task files exist but
grading is incomplete, keep preparation unchecked and say “Task files created;
finishing the grading checks.” If an external connection is unavailable, identify
what live execution needs and what preparation can continue now. A first case is
partial progress when the user requested a larger suite.

## Make the created files understandable

Start with the concrete outcome: “I created a draft of your first eval case.”
Explain that a Harbor task packages what the agent is asked, the setup it needs,
and the checks used to judge its response. State whether the artifact is one
conversation with several steps or several independent cases. For multi-step
tasks, explain whether retaining conversation history is implemented or still
required; a sequence of files does not prove the agent will remember prior turns.

Show the request and proposed expected behavior together, using a small table
or short list. Preserve the actual requests and criteria in the files; summaries
in the reply can be shorter. A count such as “nine criteria” tells the user
nothing about what is being tested. Explain enough of the criteria to support
the decision being requested, and link the rest when useful. Distinguish
criteria taken from source material, already confirmed intent, and new proposals.

Describe the role and actual state of the relevant created parts:

- **Requests:** the messages the agent will receive.
- **Grading criteria:** the written description of a good response. A saved
  rubric is not necessarily a working scorer.
- **Scoring code:** the checks that will read responses and produce scores.
  Say when only a placeholder exists.
- **Reference response or solution:** a known correct example used to check
  the scorer. Do not describe a placeholder or past response as a verified answer.
- **Environment and app connection:** the setup that will send the requests
  to the actual agent and collect its replies. Generated environment files do
  not establish that the user's application is installed or connected.

For a desktop or external dependency, name what stays outside the Harbor
container and how the task is expected to reach it. Explain the session/reset
requirement and what that means for rerunning the eval. “App connection missing”
alone hides this distinction; follow the parent skill's execution-dependency
guidance while keeping task creation moving.

Introduce file names only alongside their purpose and only when useful. Do not
recite the entire directory tree. Keep the technical inventory in the README.

## Explain what the completed check proves

Translate “Harbor loaded the structure” into “Harbor can read the task files.”
That establishes the format, not working grading or agent behavior. Describe
the unfinished work concretely: implement checks for the agreed criteria,
check them with reference responses, connect the app, and run the conversation.
Explain which of those can proceed now and what requires app access. Reuse
the execution and scoring boundaries in the parent skill.

## Make the next decision concrete

### Turn remaining gaps into an actionable handoff

At a draft handoff, explain **what you have now**, **what you can try now**, and
**what is needed for a live evaluation** in the reply itself. Keep the shared
checklist as an overview; open checklist items and a README link do not explain
how to move forward. Lead with the task drafts and their purpose, then describe
the actual validation. If the user requested a mock, report its outcome too and
connect it back to the remaining work on their evals.

For a replay, explain that it sends saved responses through the implemented
checks. It can test the replay interface and parts of the grading without invoking
the real agent. Say whether Harbor itself executed a job or only accepted the
task format. Label commands by what they actually run, such as **Replay saved
conversations**, rather than a broad “Run all scenarios” that suggests live evals.
Describe grading coverage by complete and partial source criteria when measured;
subchecks can cover parts of criteria, so their counts are not interchangeable.

For each relevant unresolved requirement, give the specific gap, what it blocks,
the smallest useful input from the user, and what you will do with that input.
Use a short list or table for the main remaining requirements. Keep the full
per-case inventory in the linked findings. Distinguish work the assistant can
finish now, decisions the user needs to make, and access needed later. For example:

| Remaining work | Why it matters | How we can finish it |
|---|---|---|
| A case has requests but no grading rules | We cannot yet decide whether its result passes | Draft criteria from the request and ask the user about the specific proposed behavior; then implement the agreed checks |
| Two source rules disagree | Either interpretation could give a different score | Show the conflicting expectations for that case, explain the consequence, and ask which behavior is intended |
| The agent connection is unknown | New requests cannot reach the real agent | Ask for a setup guide, runner/config path, or working request-and-response example; inspect it and build the supported connection |
| Starting data or reset instructions are absent | Repeated runs may start in different states | Identify the fixture or state referenced by the task and ask for that material or instructions; then prepare and verify repeatable setup |

These are examples, not requirements to impose on every user. Include only gaps
supported by the inspected material. For software or license access, explain
which dependency needs access and what task operation it enables; a setup guide
or pointer to the person who manages it can be enough to start. Never ask for
secret values or license keys in chat. Do not label implementation work as
something the user must supply when the evidence is sufficient to build it.

Respect what the user has already told you. If reports are all they have, do not
keep asking for the original scorer or dataset. Explain which historical scoring
details cannot be recovered, propose new grading decisions where needed, and
label their equivalence to the old scores as unproven. If the user lacks a real
endpoint and requested a mock, keep real integration as later work rather than
asking again for that endpoint. A mock does not fill missing grading rules.

End with one recommended next action and a focused question only when needed.
Do not ask the user to solve the whole inventory or choose among unexplained
technical tasks. For a grading decision, present the concrete proposed rule
before asking. If independent implementation can continue, do it; explain what
will be possible after the next missing input arrives. Save these actions beside
their evidence in the existing adaptation findings or task README for resumption.

### Explain the decision in terms of agent behavior

When behavior needs clarification, show what an answer would mean for this
test. For example, if the source leaves tool ordering ambiguous, explain whether
a correct result reached through a different sequence would pass or fail.
Ask which behavior the check should accept, rather than
asking the user to approve “the report's intended behavior” as a bundle.

Ask about one unresolved behavior at a time, preserve decisions already made,
and explain what you will implement after the answer. Do not turn every created
draft into a mandatory approval gate: when behavior is already clear and the
work is authorized, explain the milestone and continue implementing the checks.
Keep uncertainty local to the affected check. When the user answers, implement
that decision in the task and continue the conversion, rather than only saving
the answer and ending the turn. Missing app access does not prevent task files,
written criteria, and supported offline verifier checks from being created.

## Example: continuing after the user settles a sequence rule

Use only when the actual artifacts support these statements. Adapt the
explanation to the user's source material and the work actually completed.

The user has reports and recorded conversations, asked to preserve a report's
prescribed tool sequence, and clarified whether additional calls are allowed.
Apply that decision to the Harbor task and continue implementing the supported
checks. Explain the immediate work in a progress update:

> I’ll add that rule to the task, keeping your original requests and checks.
> We can test the supported tool-sequence checks with recorded data now.
> Running the eval against your agent will also need its application connection.

Then do the work. After creating files and performing the described checks, a
milestone explanation could be:

> I created a draft of your first eval case. It keeps the conversation together
> as one task, with your original requests in order and the checks from your
> report alongside them.
>
> Here’s what I put in the task:
>
> - **The conversation:** the messages the agent will receive, in their original order.
> - **The grading criteria:** the report's requirements, including the tool
>   sequence and the rule you confirmed for additional calls.
> - **The implemented checks:** the call counts and ordering rules supported
>   by the source. The remaining criteria are retained and marked as unfinished.
>
> Harbor can read the task files. I also checked the implemented rules with
> example traces that satisfy and violate them. That tests the scoring code;
> we still need a fresh agent run to measure its performance.
>
> You can inspect the messages and checks in the linked task draft. There are
> two pieces left before a live evaluation:
>
> - **Finish grading:** I'll implement the remaining rules supported by your
>   report. If a rule is missing or contradictory, I'll show you the proposed
>   check and the specific behavior we need to decide together.
> - **Connect your agent:** when available, a setup guide, runner script, or
>   example of sending a request and receiving a response will help me wire up
>   the real connection. I'll use that to check how conversations retain their
>   state and how each test starts fresh.
>
> We can continue the grading work while that connection is unavailable. We can
> test the implemented checks offline now; a live run will measure new
> responses from your agent once grading and setup are ready.

Name only checks that were actually implemented and run. If structural validation
or offline verifier testing is unavailable, substitute its observed status and
specific next action. Preserve the distinction between task files, working
grading, and an actual agent run without making runtime readiness a condition
for saying task files exist. Continue other requested conversions where their
inputs are available rather than ending at the first blocked case.
