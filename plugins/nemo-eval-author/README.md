<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NeMo Eval Author

Skills that an agent reads to work on the evaluation suites in a user's own
repository, derive private environments from trace evidence, and understand
traces from NeMo Intake. This directory provides no
installed public package or service, so a customer points their agent at
`skills/` and nothing gets installed.

The audit sub-flow also ships a bundled validator CLI at
[`skills/eval-author-audit/scripts/audit_spec/validate.py`](skills/eval-author-audit/scripts/audit_spec/validate.py)
and private helper modules under `scripts/audit_spec/`. They are invoked from the
copied skill tree, not published as a package API.

## Prerequisites

Discovery imports nothing beyond the standard library and Harbor itself, so it
runs on whatever Python the customer already has. Audit generation and
validation need PyYAML to read YAML and jsonschema to enforce
`schemas/audit.schema.json`; audit measurement and reporting use
`skills/eval-author-audit/requirements.txt`. Trace inspection requires the
supported `nemo` CLI, an explicit workspace, and read access to a configured
local or remote NeMo Platform instance.

`tests/test_skill_contract.py` holds to the same boundary and imports nothing
from the platform, so `pytest`, `pyyaml`, and `jsonschema` are enough to run it.
The five tests that make Harbor judge a fixture suite skip when Harbor is absent,
which is why this directory declares no dependencies and appears in no dependency
group.

| Skill | Role |
| --- | --- |
| [`eval-author`](skills/eval-author/SKILL.md) | Core. Owns the standard every sub-flow follows and routes to one. |
| [`eval-author-first-eval`](skills/eval-author-first-eval/SKILL.md) | Sub-flow. Requires Ethos, plans cases without Harbor, and builds a small working starter suite while explaining how to run and extend it. |
| [`eval-author-discover`](skills/eval-author-discover/SKILL.md) | Sub-flow. Records whether a repository's Harbor evals are ready to run. |
| [`eval-author-adapt`](skills/eval-author-adapt/SKILL.md) | Sub-flow. Guides existing non-Harbor evals into Harbor while preserving their cases and scoring rules. |
| [`eval-author-audit`](skills/eval-author-audit/SKILL.md) | Sub-flow. Validates an existing finite `audit.md` coverage denominator. |
| [`eval-author-inspect-trace`](skills/eval-author-inspect-trace/SKILL.md) | Sub-flow. Not user-invocable. Explains one Intake trace after `eval-author` selects it. |
| [`eval-author-task-create`](skills/eval-author-task-create/SKILL.md) | Sub-flow. Creates and proves one Harbor task from an actionable audit gap. |
| [`eval-author-trace-environment`](skills/eval-author-trace-environment/SKILL.md) | Sub-flow. Converts one canonicalized trace into a private candidate, inventories ground truth and software constraints, and builds a reproducible Harbor task environment when supported. |
| [`mlflow-to-atif`](skills/mlflow-to-atif/SKILL.md) | Utility. Normalizes bounded MLflow exports to canonical ATIF. |

## Where findings go

Repository eval onboarding starts with discovery. If no Harbor evals are found,
the skill explains Harbor and asks whether the user has evals in another form.
When inspection finds possible eval material, the skill explains what it found
and asks whether to use it or look elsewhere. It waits for that answer before
conversion, reusing an earlier answer or explicitly supplied eval source.
User-identified tests, scripts, datasets, notebooks, or rubrics go to `eval-author-adapt`,
which first explains Harbor, the proposed conversion, and execution needs, then
asks whether the user wants to proceed. Identifying the source does not start
conversion. After acceptance, it saves its mapping in `.eval-author/adaptation.md` and creates requested
drafts under `.eval-author/adapted-tasks/`. Original evals stay unchanged.
Conversion produces task files and the grading checks supported by the source;
unavailable app access blocks live execution, not task creation. Written
specifications support the tasks rather than replacing them.
Once the user says they are done adapting their evals, the skill offers an
optional coverage audit against their Ethos. On acceptance, `eval-author-audit`
generates or reconciles the audit specification and reviews the created tasks;
trace-based measured coverage is reported separately when evidence is available.
Confirmed absence routes to the bundled `eval-author-first-eval` flow to establish
Ethos and build a starter suite, carrying forward discovery and prior answers.

If Harbor is unavailable, the [setup guidance](skills/eval-author-discover/references/harbor-setup.md)
helps locate an existing installation or install and verify one. Source discovery,
requirements, and grading design can continue meanwhile. Installation is never
automatic; native task creation and execution resume after Harbor is verified.

`eval-author-discover` leaves a report at `.eval-author/discovery.md`, carrying the
JSON in an evidence section so a later model reads the verdict without Harbor. It is
visible and worth committing: a teammate who reads it skips the discovery pass.

`eval-author-inspect-trace` leaves one report per trace under
`.eval-author/traces/`. The front matter carries Intake source metadata and the
exact read commands. Findings use `behavior`, `issue`, `recovery`, and
`uncertainty` categories.

`eval-author-trace-environment` creates one owner-private, gitignored workspace
per task under `.eval-author/trace-environments/`. Each finalized workspace has
a `candidate` or `no_candidate` summary and keeps restricted source evidence
separate from its text-only scrubbed ATIF copy.

Discovery scripts write no files. Audit scripts write only the requested
`.eval-author/` artifacts and report JSON summaries to stdout. Trace inspection
contains instructions only.
The trace-environment helper reports to stdout and writes only its documented
artifacts under `.eval-author/`.
Capability and failure-case measurement can also consume local skill-authored
judgment sidecars for non-tool evidence; deterministic tool requirements and
prohibited-tool checks still come from ATIF traces.

## Why skills instead of an agent

Harbor tasks live in the customer's repository, so an agent that proposes changes
has to write to that repository. Customers were unwilling to grant that, sandboxed
or not. A skill inverts the arrangement: the customer's own agent does the work,
and this directory only supplies the instructions and the deterministic scripts.

The Eval Author agent that Experimentalist insight mode still uses lives in
[the Experimentalist plugin](../nemo-experimentalist/src/nemo_experimentalist_plugin/eval_author/README.md).

## Dependencies

Adding a runtime dependency to a bundled script is a breaking change for anyone who
copied the skill, so the contract test walks each script's imports and fails on
anything outside the standard library, a sibling module, or the explicitly allowed
third-party validators.

Trace inspection uses read-only `nemo intake` commands. The CLI handles its
contexts, authentication, transport, filters, pagination, and errors.

The standalone trace-environment helper requires Python 3.11 or newer. Proving
a candidate requires Harbor and Docker for the `harbor run -a nop` and
`harbor run -a oracle` checks; without them, the environment remains unproven.
The flow does not require model or provider configuration.

## Next Steps

- If your agent has no evals, start with
  [`eval-author-first-eval`](skills/eval-author-first-eval/SKILL.md). It requires
  Ethos, saves and checks it locally in the repo using the bundled
  [Local Ethos procedure](skills/eval-author/references/local-ethos.md), and
  reuses saved interview answers. No NeMo service or upload is involved.
  Missing Harbor leaves you with an evaluation plan and
  installation guidance; it is never installed automatically. Generated eval
  artifacts stay under `.eval-author/`; the Ethos itself defaults to root `ETHOS.md`.
  The first milestone is a few functioning tasks and a repeatable run command,
  with basic verifier checks and clear limitations. Coverage analysis and
  trace-driven improvement follow once the suite is working.
- Start with [`eval-author`](skills/eval-author/SKILL.md) to select the right
  sub-flow and apply the shared evaluation standard.
- Use [`eval-author-discover`](skills/eval-author-discover/SKILL.md) to check
  whether a Harbor suite can run.
- Use [`eval-author-audit`](skills/eval-author-audit/SKILL.md) to validate an
  existing finite `audit.md` coverage denominator.
- After `eval-author` selects it, follow
  [`eval-author-inspect-trace`](skills/eval-author-inspect-trace/SKILL.md) to
  explain one Intake trace.
- Use [`eval-author-trace-environment`](skills/eval-author-trace-environment/SKILL.md)
  to derive a private, evidence-backed environment from one trace.
