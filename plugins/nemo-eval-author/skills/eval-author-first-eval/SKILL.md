---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: eval-author-first-eval
description: >-
  Help a user with no evals establish a required Ethos, plan evaluation cases,
  and set up a small working Harbor suite while explaining its parts. No prior
  traces or coverage reports are required. Missing Harbor blocks scaffolding
  and execution, not planning.
triggers:
  - help me build my first evals
  - my agent has no evals yet
  - create an evaluation suite from scratch
not-for:
  - eval-author (use for the shared standard and routing)
  - eval-author-task-create (use for measured audit coverage gaps)
  - eval-author-discover (use to check an existing suite)
compatibility: >-
  Ethos is required before evaluation design and is saved and checked locally
  in the user's repository. No NeMo service, account, CLI, or upload is needed.
  Planning needs no Harbor installation. Scaffolding requires an existing Harbor CLI;
  validation requires its Python environment. Execution may require Docker,
  an agent adapter, and provider credentials.
maturity: alpha
license: Apache-2.0
user-invocable: true
allowed-tools: [Bash, Read, Write, Grep, Glob]
---

# Eval Author: first eval

Read `eval-author` for the shared standard and boundaries. No evals is a normal
starting state. Do not require a previous run or manufacture a coverage report
to enter `eval-author-task-create`. The first milestone is a small working suite
the user understands and can rerun. Modest coverage is acceptable: do not require
comprehensive coverage, difficult tasks, repeated agent success, or trace-driven
optimization before delivering it. Explain what the starter eval cases measure and
where they are weak. A working suite and strong evaluation quality are separate
claims.

## 1. Establish Ethos and check the environment

### Ground the user before asking for intent

Before the first intent question, explain the outcome in ordinary language:
we will build a starter eval set for the selected agent: a few customer scenarios
with criteria for scoring its responses. The user can rerun these evals after
changes to see whether performance improves or regresses. Briefly inspect the agent's
README and entry point to make this explanation concrete. Lead with that value,
not an installation inventory, missing-file report, or skill requirement.

Explain **ETHOS.md** on first mention: it records what the agent is supposed
to do, its boundaries, and what success looks like, giving the evals a target.
If it exists, summarize the relevant intent instead of asking the user to
recreate it. If missing, explain that we will capture that intent first. The
opening need not describe the document as “local” or refer to the requested skill.
Use the shared local Ethos procedure below, carrying this user-facing context
into its questions and review. Keep questions concrete and rooted in the
agent's actual workflows; avoid abstract choices such as “demo versus production
accountability” unless the user's goal requires that distinction.

After the outcome sentence, show a short bulleted requirements list. For each
item, explain its purpose and observed status: Ethos defines the target behavior;
Harbor builds and runs the evals; a connection to the actual agent enables
scoring its responses. Distinguish requirements for creating tasks from those
for running the agent. Use statuses supported by inspection, such as found,
missing, or not yet checked. Installed Harbor does not establish that an agent
connection works. If Harbor is missing, say planning can proceed and setup is
needed before creating runnable tasks. Mention credentials or Docker only when
the selected execution path requires them. Keep commands and version details
out of this list unless needed to resolve setup.

For an airline demo with no Ethos and verified Harbor, an opening could be:

> We'll build a starter eval set for your airline customer-service agent: a few
> customer scenarios with criteria for scoring the agent's responses. You can
> rerun these evals after changes to see whether performance improves or regresses.
>
> Here's what we need:
>
> - **ETHOS.md — to create:** Records the agent's intended behavior, boundaries,
>   and success criteria, giving the evals a target. We'll capture that first.
> - **Harbor — installed:** Builds and runs the evals and records their results.
> - **Agent connection — not yet checked:** Lets the suite send requests to your
>   agent and score its responses. This is needed to measure the agent itself.
>
> Should the starter eval set focus on baggage-policy answers, seat changes,
> and disrupted-flight rebooking using the demo policies, or is another workflow
> more important?

Adapt the examples to repository evidence. These are proposed areas of intent,
not authored evaluation cases; case design still waits for Ethos. If the user
already supplied the intended scope, acknowledge it and ask only the next
missing intent question. Avoid repeating this introduction on resume.

Use “starter eval set” for the collection and “eval case” for each scenario in
user-facing explanations. Introduce an eval case as a request with criteria for
scoring the response. Reserve “sanity checks” for validation of the eval cases
themselves; use Harbor's technical term “task” when discussing its files or CLI.

The first message asking for intent must itself explain both the value of the
starter suite and Ethos, even if a preceding progress update mentioned them.
After reading the local Ethos procedure, return to this opening before composing
the first question. Keep the user's outcome in view while collecting intent.
Use the core skill's onboarding rules here; its verdict-first format
is reserved for validation and result reports.

An intent question is ordinary progress toward the requested evals. It gathers
the user's goals; it does not ask permission to follow the skill. End the message
with that question rather than a paragraph quoting the Ethos prerequisite or
explaining why questions come one at a time. For example: “What should a customer
be able to accomplish reliably with this airline agent—for example, change a
seat, understand baggage rules, or get help after a disrupted flight?” Adapt the
examples to the actual agent, without treating them as approved requirements.

Reserve blocker explanations for actual unavailable prerequisites or required
approval. Follow any explicit host requirement to disclose a skill-imposed pause
or permission request, but do not infer such a disclosure from ordinary intent
elicitation alone. Where a disclosure is required, keep it brief and separate;
it does not replace the practical explanation of what the user is answering.

**ETHOS.md is required before designing evaluation cases.** Use a user-supplied
path first; otherwise look for root `ETHOS.md`, then
`agents/<name>-ethos/ETHOS.md`. Read the file and confirm that it describes the
intended agent. If multiple candidates are ambiguous, ask which agent to evaluate.
An empty file, placeholder, or unrelated Ethos does not satisfy this prerequisite.

If missing, explain that Ethos records the agent's purpose, intended behavior,
constraints, success and failure criteria, and what may change. It is the source
of truth for what the evals should test; code only shows current implementation.
Share the [Ethos documentation](https://docs.nvidia.com/nemo-platform/documentation/agents/optimize-agents/ethos).

Read and follow [Local Ethos](../eval-author/references/local-ethos.md) to create,
check, and review the file in the user's repo, defaulting to root `ETHOS.md`.
Reuse saved interview answers, including `.eval-author/intent-notes.md`. This
procedure is bundled with Eval Author and does not depend on installed NeMo
skills. Do not invoke the platform Ethos workflow, probe NeMo services, require
a workspace, create a Fileset, or upload anything. A failed earlier upload does
not invalidate a local Ethos or require repeating the interview.

Return here once the local file checks and content review are complete. Do not
substitute README files, code, traces, or an intent-notes file for the real Ethos.
If a local write or validation issue prevents completion, preserve the answers,
explain the actual local problem, and include the Ethos documentation link above
so the user can save or correct the file. Do not ask them to restore a service.

Check Harbor early using the interpreter probe in
[`eval-author-discover`](../eval-author-discover/SKILL.md#before-you-start) and
`harbor --help`. Distinguish a missing installation from the wrong Python
environment; a CLI on PATH alone does not prove that discovery can import Harbor.
Do not run suite discovery just to report a missing config in an empty repository.

Run this prerequisite check without making its version or importability the
opening message. Introduce **Harbor** when explaining execution or a setup
blocker: it runs the eval cases from a fresh starting environment and records each
result. If available, say there is nothing to set up for Harbor at this step.
Keep version and interpreter details in the saved report unless troubleshooting
requires the user to act on them.

If Harbor is unavailable, explain: “Harbor is the framework Eval Author uses to
build and run repeatable evaluation tasks. We can establish Ethos and plan your
eval cases now; building and running them needs Harbor in your execution
environment.” Share [Harbor Getting Started](https://www.harborframework.com/docs/getting-started).
Provide `uv tool install harbor` as a command for the user to run, not an action
to execute automatically. Respect a repository's documented compatible version.
After installation, verify `harbor --help` and locate its Python environment
with the discovery probe before using its validators. A uv tool installation
is isolated from the project's Python environment; do not repeatedly probe only
the project interpreter or install a second copy to work around that distinction.

Missing Harbor does not block Ethos creation or step 2. Missing Ethos does block
step 2, even when Harbor is installed. Check Docker only for a task whose chosen
backend requires it; it is not a prerequisite for the design conversation.

## 2. Plan the first cases from Ethos

Read the agent entry point, tool definitions, and usage docs to understand how
the actual agent runs. Aim for two or three simple representative cases from Ethos,
adjusting to the user's scope and available resources rather than enforcing a quota:
the user request, initial fixture, expected observable outcome, a meaningful
failure example, and the Ethos requirement each case tests. Ask only for intent
or invocation details not already established. Do not make the user supply
Harbor YAML or choose a framework.

Save `.eval-author/first-eval.md` with the Ethos path and requirement references,
cases, agent invocation, planned verifiers, prerequisites,
and unresolved questions. Preserve existing artifacts on resumption. This is an
**evaluation plan**, not a coverage report or runnable evals.

Start with a reproducible happy path, then a useful variation or expected failure
when supported by Ethos. Grade an observable result rather than a specific tool
call or exact prose. Simple assertions are acceptable if their limits are clear.
Do not replace subjective quality with brittle string
matching or mock away the behavior under test. If the requested case cannot be
tested with available resources, explain the limitation and select a supported
case with the user.

If Harbor is missing, deliver the plan and installation next step here. Resume
at step 3 after verifying Harbor; do not restart the Ethos interview.

## 3. Build the starter tasks and explain their parts

Require a working Harbor CLI and Python environment. Read its
`harbor task init --help` and `harbor run --help` before using options. Use an
unused descriptive slug for each selected case under `.eval-author/task-drafts/`.
Build one task end to end before adding the others. For each task:

```bash
harbor task init <org>/<slug> --tasks-dir .eval-author/task-drafts \
  --description "<behavior being tested>" --author "<actual author>"
```

Verify that the expected directory was created. Complete Harbor's generated
`instruction.md`, `task.toml`, `environment/`, `tests/test.sh`, and
`solution/solve.sh` using its installed schema. Keep solutions and verifier-only
fixtures out of the agent's initial environment. Set executable permissions,
realistic timeouts, and deterministic rewards; leave no scaffold placeholders.
Before running, document each case's reward format, metric names and ranges,
and expected NOP and Oracle acceptance criteria in its README. Derive these
criteria from the intended outcome. For a binary completion metric, NOP should
score 0 and Oracle 1. For named or graded metrics, specify the expected values
or thresholds for each relevant metric instead of imposing a universal 0/1 pair.
Add a README with the Ethos requirement, fixtures, verifier, and run commands.

Teach each concept when it becomes concrete, using the files being created:

- The instruction is the request the agent receives.
- Fixtures and the environment provide a repeatable starting state.
- The verifier checks the output and writes the reward. Explain the actual
  assertion and an example it cannot distinguish yet.
- The Oracle is a reference solution used to check the task, not the user's agent.
- The suite config selects tasks and an agent; job results show what happened.

Keep explanations short and practical. Do not require a separate tutorial or
turn initial task design into an exhaustive evaluation methodology exercise.

For Docker-backed execution, check `docker info` first. Retain all jobs under
`.eval-author/jobs/`, with new names on reruns:

```bash
harbor run -p .eval-author/task-drafts/<slug> -a nop \
  --jobs-dir .eval-author/jobs --job-name <slug>-nop-1
harbor run -p .eval-author/task-drafts/<slug> -a oracle \
  --jobs-dir .eval-author/jobs --job-name <slug>-oracle-1
```

For each task, inspect Harbor trial results and recorded rewards. Require both
runs to complete without exceptions and meet the case's predeclared NOP/Oracle
criteria, including the expected reward shape. The reference solution should
satisfy the intended outcome; doing nothing should not earn completion credit
when the case requires an answer or action. If inaction is itself correct, NOP
success does not show that the verifier rejects incorrect behavior: exercise
an explicit incorrect response or action as a negative control for that case.
These are basic wiring and verifier sanity
checks, not evidence of broad coverage or a robust benchmark. Do not add repeated
proof runs or a separate negative-control campaign as an onboarding gate.
Investigate obvious unconditional rewards or leaked answers. Fix broken tasks
and rerun affected checks; do not weaken the intended assertion to force a pass.
Report blocked tasks separately and deliver the working subset without silently
dropping planned cases or claiming the whole suite passed.

## 4. Evaluate the user's agent and deliver

Establish a supported Harbor integration for the actual agent from its entry
point, installed adapter code, registry, and CLI help. Do not substitute another
agent or rewrite the application. If the adapter is missing, deliver the starter
tasks and their sanity-check results
and the specific integration requirement; agent performance remains unmeasured.

When supported, write `.eval-author/first-eval.yaml` using the installed Harbor
JobConfig schema: explicitly select the working starter tasks, the actual agent
and model settings, one attempt per task, and jobs under `.eval-author/jobs/`.
Do not include unrelated drafts merely because they share a parent directory.
Resolve paths from the repository
root and reference credential environment variables rather than embedding secrets.
Follow `eval-author-discover` to validate this config and report its per-config
verdict if other configs exist. Do not claim it is runnable from YAML presence.

Confirm that the resolved task set matches the selected starter tasks. Explain
the selected task count and where rewards and errors will appear.
Once validated, show the tasks, agent, model, and one-attempt-per-task command.
Run it only when the user asked for execution or approved model spend; preserve
existing authorization. From the repository root:

```bash
harbor job start -c .eval-author/first-eval.yaml
```

Inspect results for every selected task and update the plan with exact commands,
artifact paths, sanity-check results, per-task agent rewards and exceptions, and
remaining blockers. Show the user how to rerun the suite, inspect one result, and
add or modify a task. Distinguish a plan, tasks whose sanity checks passed, a
validated runnable config, and a completed agent run. If execution is blocked or
not authorized, say which milestone was reached instead of claiming the suite
ran. An agent failing a functioning task is a useful baseline, not a reason to
weaken the test or start repeated unrequested runs.

Deliver when the selected starter tasks execute and record rewards through a
validated config, even if the agent scores poorly or the checks are basic.
Document limitations such as narrow fixtures or assertions that only check part
of an outcome. Do not present successful setup as high-quality evaluation.

### Explain the completed milestone

Lead the completion message with what the user now has: name the created eval cases
and explain each in a bullet describing the customer behavior being evaluated,
rather than listing topic labels alone. Explain that each task pairs a customer request with
a rule for judging the response. Translate observed sanity results into their
meaning: doing nothing failed and the prepared reference solution passed, so
these examples exercise the scoring rules successfully. Do not describe NOP as an empty
answer unless that is what the task actually tested. These results do not prove
the rules judge every answer correctly or measure the user's agent. For graded
metrics or cases where inaction is correct, explain the actual acceptance
criteria and observed control results instead of using the binary example.

Then state whether the actual agent ran. If it did, summarize its observed
results. If integration is missing, explain the concrete next step in everyday
terms: connect the suite to the agent's actual entry point so the evals can send
it requests and score its responses. Name the entry point only when verified;
introduce “Harbor adapter” only if that technical detail helps the user act.
Distinguish configuration validation and an agent connection's setup check from
an actual agent run: neither provides performance scores. If another prerequisite
blocks execution, name the observed blocker and the action needed to resolve it,
then explain that the next run will score the agent's actual responses. For a
missing credential, name the required environment variable and tell the user to
configure it in the execution environment, without requesting its secret value.
Do not bury the next step solely in the linked report.
Explain practical limits, such as a wording assertion rejecting a correct
paraphrase. Link the saved eval cases and results, with rerun instructions clearly
labeled as either task sanity checks or an actual agent evaluation.

For example, when supported by the recorded results:

> Your **starter eval set is ready**, with three eval cases:
>
> - **Baggage limits:** Does the agent explain the allowance and size/weight limits
>   in the demo policy?
> - **Missing-bag guidance:** Does it tell the customer how to report a missing bag?
> - **Unspecified overweight fees:** Does it acknowledge that the policy doesn't
>   specify a fee, rather than inventing one?
>
> Each case includes a customer request and criteria for scoring the response.
> I ran them with no action and with prepared reference solutions: doing nothing
> failed, and the reference solutions passed. This confirms the cases run and
> distinguish those examples.
>
> Harbor validated the eval configuration, and the connection to your Triage
> agent passed its setup check. **We haven't measured the agent's responses yet**
> because `OPENAI_API_KEY` isn't set in the execution environment.
>
> Next, configure that key in your environment so we can run the three cases
> against your agent and get its first scores.
>
> These initial scoring rules look for specific wording, so a correct answer
> phrased differently might fail. The first agent run will help us identify
> where those rules need improvement.

Adapt this to the actual milestone and include the real artifact link. Keep raw
scores, interpreter details, and full commands in the saved report unless useful
in the message. Do not append a quotation of the skill to justify delivering
starter tasks; the observed results and concrete next step explain the handoff.
Preserve any explicit host disclosure requirement for a genuine approval or
blocker.

Treat traces and improvement as the next stage, not a prerequisite for setup.
At the final handoff, explain that the starter eval set covers only the selected
behaviors. Remind the user to review and keep ETHOS.md up to date as the agent
changes: it should capture the whole selected agent's intended workflows,
boundaries, and success criteria, including areas not yet evaluated. Link their
actual Ethos and call out known gaps without claiming those areas are covered by
the starter evals. For example:

> These first eval cases cover baggage answers. Your ETHOS.md should describe
> the whole airline agent, including its other intended workflows and boundaries.
> Review it for missing areas and keep it current as the agent changes; we can
> use that broader intent to expand the eval set over time.

Explain that traces reveal the agent's steps and tool calls, helping identify
failure patterns, missing coverage, and weak checks. Point to actual trace
artifacts when emitted; if absent, explain the adapter or instrumentation work
needed without blocking the working starter suite.
For coverage accounting, follow `eval-author-audit` with the established Ethos
and actual ATIF from completed runs, then `eval-author-task-create` for measured
actionable gaps. Do not fabricate traces or measurement reports when the agent
does not emit ATIF.
