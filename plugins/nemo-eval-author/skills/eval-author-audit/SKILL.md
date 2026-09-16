---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: eval-author-audit
description: >-
  Generate, validate, measure, and report on an audit-spec coverage denominator
  for Eval Author. Use when the user wants a hand-editable audit.md file derived
  from Ethos, needs schema enforcement for declared tools, capabilities, failure
  cases, evidence, and references, wants to measure which audit items one ATIF
  trace covers, or wants to aggregate coverage across measured traces. Changes
  none of the user's source, and saves audit artifacts under `.eval-author/`.
triggers:
  - generate audit.md from ETHOS.md
  - validate audit.md coverage schema
  - measure audit.md coverage against a harbor trace
  - aggregate audit.md coverage reports
  - check audit.md coverage denominator
  - what should my evals cover from the agent ethos
  - review the audit coverage denominator
not-for:
  - eval-author (use for the standard, the boundaries, and to pick a sub-flow)
  - eval-author-discover (use to prove whether a Harbor suite is runnable)
  - eval-author-inspect-trace (use after eval-author selects an Intake trace)
  - nemo-experimentalist (use to optimize an agent from Insights or explicit datasets)
compatibility: >-
  Python 3.11 or later for generation and validation; Python 3.12 or later for
  ATIF measurement via Harbor's trajectory model. Dependencies are listed in requirements.txt.
  Generation, validation, measurement, and aggregation read local files only;
  they do not start Harbor jobs or call platform services.
maturity: alpha
license: Apache-2.0
user-invocable: true
allowed-tools: [Bash, Read, Write, Grep, Glob]
---

## Ethos Pre-flight

Before drafting audit items, locate the Ethos file that will serve as the source
of truth. If the user provides `--ethos <path>`, validate and use that path as
`<ethos_path>` before applying repository discovery. Otherwise, prefer
`ETHOS.md` at the repository root when it exists. If it is missing, check for one
existing local Ethos at
`agents/<name>-ethos/ETHOS.md` and use that path as `<ethos_path>`.

If no Ethos file exists, pause audit-item drafting and generation while completing
the local Ethos prerequisite below. Explain why it is needed: Eval
Author needs a source of truth for how the agent is supposed to behave before it
can decide what `audit.md` should cover. Code shows what the agent does today;
Ethos records intended behavior, mission, constraints, success and failure
criteria, and what may change. Link the user to
[ETHOS.md](https://docs.nvidia.com/nemo-platform/documentation/agents/optimize-agents/ethos)
for what Ethos is and how to create one.

Follow [Local Ethos](../eval-author/references/local-ethos.md) to capture and
validate intent locally. Save the document and wait for the user's review of its
contents before resuming audit-item drafting with the exact path as `<ethos_path>`.
The bundled procedure reuses saved interview answers and defaults to root
`ETHOS.md`. It does not require installed NeMo skills, a running platform,
a workspace, or any upload. Do not call the platform Ethos or model-selection
flows as prerequisites for local audit authoring.

If the user prefers to create the file themselves, or local filesystem access
prevents writing it, explain the local limitation, provide the Ethos documentation
link above, and ask for the saved file's path. An unreachable NeMo service is
not a blocker. A demo may explicitly use demo policies without airline approval;
record that scope instead of inventing production guarantees.

The local Ethos handoff creates a real document from user-confirmed intent; an
empty file or placeholder written merely to bypass this prerequisite does not
qualify. Do not substitute other repository material for Ethos. Contributor docs, operations docs, README
files, code, traces, or draft labels are not valid source-of-truth replacements
for a missing Ethos. Do not synthesize an audit denominator from those materials,
even if the output is marked as draft. First save and review the real local Ethos.

# Eval Author: audit

Read `eval-author` for the shared standard, vocabulary, and boundaries. This
sub-flow generates and validates a finite coverage denominator from `<ethos_path>`
and reviewed audit items, then can measure one ATIF trace and aggregate coverage
reports against it. It does not generate tasks yet.

The audit-spec approach has three item kinds in v1:

| Kind | Meaning |
|---|---|
| `tool` | A canonical tool name the agent may call |
| `capability` | A high-level behavior the agent should exercise |
| `failure_case` | Expected safe behavior when a capability cannot proceed normally |

Every item uses `name` as its stable coverage key. Names must be unique across
the whole file; do not add sequential numeric IDs. Tool references in
`required_tools`, `expected_tools`, and `evidence_required[].tool` must match the
`name` of a declared `tool` item. `prohibited_tools` may name any syntactically
valid tool name, including tools the agent must never call and therefore should
not declare as allowed tools.

Write audit artifacts under `.eval-author/`. Do not edit the customer's source,
existing evals, source-of-truth documents, or `ETHOS.md`.

## Scripts

Audit-spec mechanics live under `scripts/audit_spec/`:

Read `scripts/audit_spec/README.md` for the current measurement assumptions:
ATIF input, Harbor trajectory parsing, v1 `tool_calls`, `capabilities`, and
`failure_cases` coverage, and coverage aggregation from `coverage.json` files.

| Script | Use it to |
|---|---|
| `scripts/audit_spec/generate.py` | Create, reconcile, replace, or preview `.eval-author/audit.md` from `ETHOS.md` and reviewed item proposals |
| `scripts/audit_spec/measure.py` | Measure one ATIF trace or Harbor trial directory against `audit.md` and write coverage/details files for each selected method |
| `scripts/audit_spec/report.py` | Aggregate per-trace `coverage.json` files into one coverage report with uncovered audit items |
| `scripts/audit_spec/validate.py` | Validate the marked audit-spec block in `audit.md` |

Shared helpers, measurement method contracts, schemas, and examples are
documented in `scripts/audit_spec/README.md`.
Runtime dependencies are listed in `requirements.txt`.

## Step 1: Draft Or Update Audit Items

Before drafting or updating `.eval-author/audit-items.yaml`, read
`templates/audit.md` and `schemas/audit.schema.json`. Use the template as the
worked example and the JSON Schema descriptions as the field definitions. Do not
use validation as the primary way to discover the format; validation is the
enforcement and repair step after drafting.

Read `<ethos_path>` and draft audit items at the level between Ethos and runnable
tasks: canonical tools, high-level capabilities, and material failure cases. Keep
the list finite. Do not create separate items for prompt paraphrases, fixture
variants, or ordinary happy-path permutations.

For `tool` items, use the names that appear in the actual runtime traces or tool
registry, including eval-specific tools that may be more precise than product
tools named in Ethos prose. If Ethos describes a generic tool such as `sqlite`
but measurement traces expose `execute_sql` and `submit_sql`, declare the
runtime tool names and connect capabilities or failure cases to those names.
Do not invent tool names that will not appear in the measurement surface.

Save the reviewed item proposals as `.eval-author/audit-items.yaml`. The items
file may be either a mapping with an `items` key or the item list itself. It
should use the same item shape shown in `templates/audit.md` and enforced by
`schemas/audit.schema.json`.

For an initial audit, this file should contain the full proposed denominator. For
an update, it may contain only the proposed additions or edits. Existing reviewed
`audit.md` items remain the source of truth in reconcile mode.

Capabilities that do not need tools, such as policy refusals or out-of-scope
handling, should use `required_tools: []`. Failure cases attach to capability
names through `applies_to`; tool-level failure expectations stay on the tool item
as `expected_failure_behavior`.

For failure cases, make the trigger and safe response explicit in
`evidence_required`. Include prohibited output classes in an `output` evidence
description when their absence must gate coverage. Measurement uses
`prohibited_tools` as a deterministic gate; `applies_to`, `expected_tools`,
`trigger`, `expected_behavior`, and `prohibited_outputs` otherwise provide the
rubric and authoring context rather than separate hidden checks.

## Step 2: Generate Or Reconcile Audit.md

Create or update `.eval-author/audit.md` from `<ethos_path>` and the reviewed item
proposals:

```bash
uv run --with pyyaml --with jsonschema \
  <skill_dir>/scripts/audit_spec/generate.py \
  --ethos <ethos_path> \
  --items .eval-author/audit-items.yaml \
  --out .eval-author/audit.md
```

The default mode is `reconcile`. If `.eval-author/audit.md` does not exist, it
creates the file. If it already exists, the generator parses the existing marked
block, updates source metadata such as the Ethos digest, preserves existing item
bodies by stable `name`, appends new proposed items, and reports proposed edits
without silently rewriting them. Hand-authored prose outside the marked block is
preserved.

By default, the generator treats `.eval-author/audit-items.yaml` as a partial
update proposal. Missing existing items are not stale in that mode, because the
items file may contain only additions or edits. Use `--items-mode full` only when
the items file is intended to be the complete denominator; then existing items
omitted from the proposal are reported as `possibly_stale_items`.

Use the explicit modes when the default is not what the user wants:

```bash
uv run --with pyyaml --with jsonschema \
  <skill_dir>/scripts/audit_spec/generate.py \
  --ethos <ethos_path> \
  --items .eval-author/audit-items.yaml \
  --out .eval-author/audit.md \
  --mode suggest

uv run --with pyyaml --with jsonschema \
  <skill_dir>/scripts/audit_spec/generate.py \
  --ethos <ethos_path> \
  --items .eval-author/audit-items.yaml \
  --out .eval-author/audit.md \
  --mode reconcile \
  --items-mode full

uv run --with pyyaml --with jsonschema \
  <skill_dir>/scripts/audit_spec/generate.py \
  --ethos <ethos_path> \
  --items .eval-author/audit-items.yaml \
  --out .eval-author/audit.md \
  --mode replace
```

`suggest` performs the same comparison as `reconcile` but writes nothing.
`replace` rewrites the whole file from the item proposal file, including prose
outside the marked block, and should be used only when the user wants to discard
the existing generated audit file.

The generator prints a JSON summary containing `written`, `added_items`,
`conflicting_items`, `conflicting_items_applied`, and `possibly_stale_items`.
Treat `conflicting_items` as items where the proposal differs from the reviewed
audit item; reconcile mode preserves the reviewed item and reports
`conflicting_items_applied: false` so the user can accept the change manually or
through a future editor. If reconcile adds items, finds conflicts, reports stale
items in full mode, or detects an agent-name change, an approved audit is
demoted to `status: draft` unless the user passes `--status approved`.

The generator adds an optional `sources` entry for Ethos with `name: ethos`, a
path relative to `audit.md`, and a real `sha256` digest. It uses the frontmatter
`name` from `ETHOS.md` when `--agent` is omitted. If that inferred name differs
from the existing audit's `agent`, reconcile preserves the reviewed audit name
and reports `agent_change` unless the user passes `--agent` explicitly.
`source_refs` are advisory provenance notes in v1; the validator preserves them
but does not resolve them against `sources` until a future generator or grammar
defines that reference format.

## Step 3: Validate

Run validation after every generated or hand-edited audit file:

```bash
uv run --with pyyaml --with jsonschema \
  <skill_dir>/scripts/audit_spec/validate.py --audit .eval-author/audit.md
```

`schemas/audit.schema.json` is the canonical structural schema. The Python
validator applies that schema first, then checks any source digests that are
provided and cross-item references such as `required_tools` and `applies_to`.

Validation proves only structure and references, not that the denominator is
complete or correct.

## Step 4: Measure One ATIF Trace

After validation, measure one completed trial directory or one ATIF trajectory
file. Use `--measure` to select one or more comma-separated measurement methods;
the default is `tool_calls`.

```bash
uv run --with-requirements <skill_dir>/requirements.txt \
  <skill_dir>/scripts/audit_spec/measure.py \
  --audit .eval-author/audit.md \
  --trial-dir <harbor-job-dir>/<trial-dir> \
  --measure tool_calls \
  --out-dir .eval-author/audit-measurements
```

The current `--trial-dir` reader supports Harbor-style trial directories that
normally contain `agent/trajectory.json`; agents that do not emit ATIF may not
have that file. When the trace file is already known, pass it directly and stamp
the task explicitly:

```bash
uv run --with-requirements <skill_dir>/requirements.txt \
  <skill_dir>/scripts/audit_spec/measure.py \
  --audit .eval-author/audit.md \
  --trace <path-to>/trajectory.json \
  --task-id <task-id> \
  --run-id <run-id> \
  --measure tool_calls \
  --out-dir .eval-author/audit-measurements
```

`--measure` may be passed more than once or as CSV, for example
`--measure tool_calls,capabilities,failure_cases`. The default is `tool_calls`;
include the other methods when the user wants the same trace to count against
capability or failure-case items. The script loads the trajectory once, then runs
each selected method against the same parsed Harbor trajectory model. Unknown
method names fail before the trace is loaded.

When capability evidence contains non-tool kinds such as `user_intent`, `output`,
`outcome`, `policy_boundary`, or `verifier`, inspect the trace and write a
structured judgment file under `.eval-author/` using
`schemas/audit_capability_judgments.schema.json`. Each judgment must target the
capability `name` plus the zero-based `evidence_required` index, copy the
evidence `kind` and `description` exactly, and judge only non-tool evidence. Do
not write judgments for `tool_call`; the script measures those deterministically.
Set the sidecar's required `trace_sha256` to `sha256:` followed by the lowercase
SHA-256 digest of the exact ATIF trajectory file you inspected. Measurement
rejects the sidecar if those trace bytes have changed or another trace is used.
Use the capability description, evidence description, and concrete trace content
as the rubric: mark `satisfied` only when the trace clearly demonstrates the
requirement, `missing` when it clearly does not, and `unclear` when the trace is
ambiguous or insufficient. Include brief rationale and supporting trace
references when available. A subjective judgment can satisfy only the non-tool
evidence it targets; it cannot override missing required tools or missing
`tool_call` evidence.
Pass the sidecar when measuring capabilities:

```bash
uv run --with-requirements <skill_dir>/requirements.txt \
  <skill_dir>/scripts/audit_spec/measure.py \
  --audit .eval-author/audit.md \
  --trace <path-to>/trajectory.json \
  --task-id <task-id> \
  --run-id <run-id> \
  --measure capabilities \
  --capability-judgments .eval-author/capability-judgments.json \
  --out-dir .eval-author/audit-measurements
```

Capability coverage is conjunctive: every deterministic requirement must be
satisfied, and every judged evidence requirement must be satisfied. Missing
judgments leave the capability uncovered. Stale judgments fail measurement
before the script writes a coverage report, including judgments bound to a
different trace digest.

Failure-case coverage follows the same pattern. Inspect the trace and write
`schemas/audit_failure_case_judgments.schema.json`, targeting each non-tool
evidence requirement by failure-case `name`, zero-based index, exact `kind`, and
exact `description`. Judge only what `evidence_required` states, using the
failure case's trigger, expected behavior, and prohibited outputs as context.
Then measure it with:

```bash
uv run --with-requirements <skill_dir>/requirements.txt \
  <skill_dir>/scripts/audit_spec/measure.py \
  --audit .eval-author/audit.md \
  --trace <path-to>/trajectory.json \
  --task-id <task-id> \
  --run-id <run-id> \
  --measure failure_cases \
  --failure-case-judgments .eval-author/failure-case-judgments.json \
  --out-dir .eval-author/audit-measurements
```

A failure case is covered only when every evidence requirement is satisfied and
none of its `prohibited_tools` appears anywhere in the trace. Missing judgments
leave it uncovered, and a subjective judgment cannot override missing
`tool_call` evidence or an observed prohibited tool. The same trace-digest and
stale-target checks used for capability judgments apply.

The script writes one folder per task, run, and method. Task and run ids are
encoded as single path components so ids containing `/` cannot create nested or
escaping paths:

```text
.eval-author/audit-measurements/task=<encoded-task-id>/run=<encoded-run-id>/<method>/coverage.json
.eval-author/audit-measurements/task=<encoded-task-id>/run=<encoded-run-id>/<method>/details.json
```

`coverage.json` uses the shared coverage schema and contains only the stable
audit item names this trace covered plus provider-neutral subject identity
(`trace`, `trace_format`, `task_id`, and `run_id`). It also records
`item_kind_count`, the denominator for the measured item kind. Coverage
aggregation should consume this file and ignore method-specific debug details.
`details.json` is specific to the selected method and carries traceability data
for humans.

For current method semantics and details schemas, see
`scripts/audit_spec/README.md`.

The script validates `coverage.json` against `schemas/audit_coverage.schema.json`
and validates `details.json` against the selected method's details schema before
writing. Use the next step to union coverage across tasks and runs.

## Step 5: Aggregate Coverage Reports

After measuring one or more traces, aggregate the per-trace `coverage.json`
files into a coverage report:

```bash
uv run --with-requirements <skill_dir>/requirements.txt \
  <skill_dir>/scripts/audit_spec/report.py \
  --audit .eval-author/audit.md \
  --coverage-dir .eval-author/audit-measurements \
  --out .eval-author/audit-coverage-report.json
```

Use `--coverage <path-to-coverage.json>` for explicit files, or repeat
`--coverage-dir` and `--coverage` when the inputs are split across directories.
The script scans coverage directories recursively for files named
`coverage.json`, validates every input against
`schemas/audit_coverage.schema.json`, and rejects inputs whose audit metadata no
longer matches the current `audit.md`. A status-only mismatch, such as
measurements produced while the audit was `draft` and aggregated after it became
`approved`, is reported as a warning instead of forcing a re-measure.

The aggregate report uses `schemas/audit_coverage_report.schema.json`. It
contains overall and per-kind count summaries, the union of covered item names,
warnings, the measured item kinds, and `uncovered_items`. Each uncovered item
includes the original audit item plus generation-oriented context: a stable
`reason`, a one-sentence `focus`, likely `needed_tools`, and the item's
`evidence_required`.
Use `reason: not_measured_by_any_method` to distinguish gaps that no included
measurement method could close from `reason: not_covered_by_any_input_report`,
which means the item kind was measured but no input report covered that item.
Treat that list as evidence for the proposal step in `eval-author-task-create`.
The aggregate report unions coverage; it does not establish why an item is
uncovered or whether an existing task already exposes an agent failure.

## Next Steps

- For audit-generation inputs and reconciliation modes, return to
  [Step 2: Generate Or Reconcile Audit.md](#step-2-generate-or-reconcile-auditmd).
- Report coverage by kind, observed findings, measurement limits, and artifact
  links. Dataset recommendations belong to the proposal step in
  [`eval-author-task-create`](../eval-author-task-create/SKILL.md). Hand off there
  when the user requests proposals or continuation through the full workflow;
  an audit-only request ends with the findings.
- The proposal step considers tools, capabilities, and failure cases. Automatic
  task creation still accepts only uncovered tool items with
  `reason: not_covered_by_any_input_report`. Items with
  `reason: not_measured_by_any_method` remain unmeasured, even when the proposal
  step suggests a candidate scenario for them.
