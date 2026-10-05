<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Review and rerun a generated suite

Create the review and rerun guide as soon as the first cases are authored, using
`README.md` at the top level of each generated evaluation suite and preserving
task README filenames as described below. Link it when presenting the created
cases, without waiting for execution or a question about where they are saved.
Refresh it after validation or agent runs and when cases, configuration, setup,
or commands change. A conversation reply or run report does not replace this
entry point. Multi-task suites need a suite guide in addition to individual task
READMEs.

Use the suite's existing root under `.eval-author/`: for first-eval this is
`.eval-author/README.md`, beside `first-eval.md`; for a standalone task draft,
extend `.eval-author/task-drafts/<task-slug>/README.md`. For trace-derived work,
use `<task-dir>/README.md`, beside `task/`; a generated batch also needs a README
at its collection root. Keep the trace task's required `task/README.md` intact.
When an existing generated suite has an uppercase `README.md` at the same
location, keep its filename and content and add the review and rerun sections
there. Do not rename existing task READMEs or create files differing only in
case. Keep unrelated suite guides separate.
Proposal-only work does not create a suite or require this file.

## Review the created cases

Put a case inventory near the top of the README, before the run instructions.
Reuse the case names and requirement or coverage IDs from the agreed plan.
Include every selected case, with pending or blocked cases visible alongside
authored ones. One task may contain several examples; repeated attempts do not
add cases. The inventory can also serve as the coverage-to-task mapping below.

For each case, give a short readable description and links to the details:

- **Purpose and input:** what behavior it tests, the request and relevant
  starting state, and the exact task or dataset row. Link the instruction and
  fixtures. For Gym, identify the dataset and one-based physical JSONL line;
  describe the row's input in Markdown so the user need not format JSONL just
  to understand the case.
- **Expected outcome and grading:** what a correct response or action achieves,
  what the verifier actually checks, and any known limitation. Link the task's
  grading explanation and verifier; keep expected answers distinct from the
  agent's input. Reuse existing task documentation for detailed explanations.
- **Source evidence:** when a case was derived from a trace, link the reviewed
  source evidence and derivation notes as provenance. State when no source trace
  was used. A source trace is not evidence that the generated case has run.
- **Run evidence:** link available per-case results, logs, and execution traces,
  labeling task controls separately from actual-agent attempts. Include failed
  attempts and distinguish a case that has not run from a run whose trace is
  unavailable. State known blockers or missing-evidence reasons; do not infer
  execution from a draft, configuration, or source trace. Link existing run
  reports for detailed histories rather than duplicating them in the inventory.

Use clickable links resolved from the README's location. Keep source and run
evidence in their existing locations; this guide does not change what is private
or exported. An exported guide must work from its exported root and identify
evidence omitted from the product without depending on private links.

When the installed provider has a supported viewer, include its exact launch
command, working directory, and which case or run to select. Verify the options
and directory layout against that installation. For Harbor versions supporting
`harbor view`, task review uses the directory directly containing the tasks with
`--tasks`; run review uses the jobs directory with `--jobs`. A trace workspace's
collection root may not be the task directory's parent. Use the actual paths in
the guide, and distinct ports when both views run concurrently. Link available
readable reports when a format has no verified viewer, and label any remaining
raw-only trace inspection limitation rather than inventing a command or UI.

## What a returning user needs

Write the guide using the generated files and verified runtime information.
It must be usable without the authoring conversation or an agent reconstructing
the commands. Include:

- **Setup and working directory:** where to run every command; the provider and
  runtime version used; installation or environment activation commands;
  dependency manifests/locks; required backend, services, agent adapter, model
  configuration, and access. Name credential variables and how to supply them
  to the configured runtime, without saving secret values. Link local setup
  files and explain fixture/state reset before each run.
- **Run the full suite:** a copyable command using the saved configuration and
  exact intended case set, including attempts and any required service startup.
  Use fresh job names/output paths on each invocation so prior results survive.
  Label whether this evaluates the actual agent or only checks task wiring.
- **Run one coverage item:** a table mapping the suite's existing requirement or
  coverage IDs and descriptions to task names/paths or dataset rows. Include
  commands for the mapped selections and a concrete example from this suite.
  A coverage item may need multiple cases; preserve all of them. If it shares a
  task with other checks, explain that selection runs the whole task. An audit
  tool/requirement ID is not automatically a provider CLI selector.
- **Read results and troubleshoot:** where each new run saves rewards, errors,
  logs, and available traces; how to inspect one case's result; the meaning of
  the actual reward fields; and remedies for known setup failures. Link the
  saved run summary for observed results and limitations.
- **Open readable evidence:** use the installed Harbor viewer for native tasks
  and jobs. For selected Gym JSONL records or standalone ATIF, follow
  [Local review](local-review.md) to create and link a private `report.html`.
  Save its exact regeneration command, selected source and rows, evidence role,
  and any missing or unsupported content. Introduce that review path in the
  conversation using [Show the user how to review](#show-the-user-how-to-review).
  A rendered report is not validation or proof of execution.

Keep task design and verifier details in their existing documents and link them
from the guide. For an incomplete suite, save the instructions that are known,
identify blocked cases and missing setup explicitly, and label unverified or
pending commands. Do not invent an adapter, model, credential, or successful run
to complete the README. Replace authoring placeholders before describing a
command as ready to copy; explicit blockers are preferable to fake runnable
examples. Refresh the guide and link it in the final handoff even if execution
is blocked.

## Select cases without changing the experiment

Use the installed provider's help/schema to verify commands. Preserve the full
suite's agent, model, environment, verifier, attempts, and other run settings
when selecting a subset. Check the resolved selection includes exactly the
intended cases, excluding unrelated drafts and blocked cases unless explicitly
selected. Document any deliberate setting changes.

For Harbor, reuse the full-suite config. A CLI task/path override is suitable
only after checking that it replaces the selection as intended and preserves
task-specific settings. Dataset name filters do not filter an explicit `tasks`
list. When overrides cannot preserve the configuration, save a subset config
under the suite root: retain the selected task entries, remove other task and
dataset sources, and keep the non-selection settings. Do not replace the real
agent with Oracle or NOP to demonstrate an individual agent evaluation.

For example, after saving and validating a one-case config for a starter suite
whose case is `refund-policy`, its guide could show these commands from the
repository root (use the actual runtime and paths in the generated guide):

```bash
# Full suite: actual agent evaluation, with a new output directory each time.
harbor job start -c .eval-author/first-eval.yaml \
  --jobs-dir .eval-author/jobs --job-name "suite-$(date -u +%Y%m%dT%H%M%S)-$$"

# Only the refund-policy case, preserving the same agent and run settings.
harbor job start -c .eval-author/rerun/refund-policy.yaml \
  --jobs-dir .eval-author/jobs --job-name "refund-policy-$(date -u +%Y%m%dT%H%M%S)-$$"
```

For Gym, retain the selected composition, model configuration, dataset identity,
and reset procedure from `reproducibility.md`. Use a verified native selector or
save a subset JSONL with the complete original rows, including verifier fields,
in their original order. Show the exact selection and run commands, including
service startup when using `--no-serve`; do not invent a coverage-ID flag. Keep
full-suite and subset output paths distinct and fresh.

## Show the user how to review

At the first created-case checkpoint and again when handing back run results,
include a short review walkthrough in the user-facing reply. A general README
link or a rerun command alone does not introduce the review experience.

For Harbor, explain that its viewer is a local browser for the test cases and
saved attempts. Include the relevant launch command, working directory, and
browser URL in the reply, using the verified runtime and actual artifact paths.
Keep the same instructions in a clearly linked README section. Use the directory
directly containing the tasks with `--tasks`, and the parent of the saved job
directories with `--jobs`; verify options against the installed version. When
both views run concurrently, use separate available ports.

Give one concrete route through the relevant view:

- **Created cases:** open the task view, select a named case, read its
  **Instruction**, and use **Files** for its README, fixtures, and verifier.
  Explain that a task is one test case and that it can be reviewed before it runs.
- **Run results:** open the job view, select the actual agent's job, a named case,
  and one attempt when it exists. Explain that a job groups attempts and a trial
  is one attempt.
  Point to its recorded reward and grading evidence; show the trajectory when
  available. Keep control jobs separate from the actual-agent job.

If only controls ran or execution was blocked before any actual-agent run, say
the actual agent has not run and offer case review with any clearly labeled
control evidence instead of inventing an actual-agent job.

Use labels and navigation verified for the installed viewer. If a viewer is
already running and its content was checked, give its live URL. Otherwise say
“Run this command, then open this URL,” explain that the command stays running,
and label launch as untested when only CLI help was checked. Do not present a
planned localhost URL as a running service or require a new evaluation to review
saved artifacts. If the viewer is unavailable, state the concrete limitation
and point to the available readable report or case files.

An adapter that records custom action/reply JSON without ATIF can still supply
case files, trial results, and grading evidence. Explain that its native
trajectory is unavailable, and link the actual capture by name and path. Do not
promise that the Trajectory tab contains those actions or send unsupported
custom JSON through the Gym/ATIF reader. For supported standalone ATIF or Gym
JSONL, link the local reader's HTML report directly and explain its evidence role.

## Verify the handoff

Follow the inventory's links to each created case's input, grading, and available
evidence. Check that pending cases and missing traces remain visible and that
source evidence is not presented as a generated-case run. Check viewer commands
against the installed provider and the actual artifact roots.

Check the README's saved configurations from its stated working directory.
Use provider validation or configuration inspection to confirm both
full-suite and individual selections and their preserved settings. Record what
was validated versus actually executed; printing a config is not run evidence.
Reuse authorized run evidence where applicable. Do not spend model credentials
just to check documentation. A returning user should be able to review the cases,
follow setup, run the suite or a named item, and inspect results using this
README and the files it links. Link the refreshed README in the final
handoff, including when execution is blocked.
