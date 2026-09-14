---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: eval-author-trace-environment
description: >-
  Use MLflow, Intake, OpenTelemetry, or ATIF trace evidence to derive a private,
  reproducible Harbor environment candidate.
triggers:
  - create an evaluation environment from a trace
  - turn ATIF into a Harbor task environment
  - derive an eval task from MLflow Intake or OpenTelemetry data
not-for:
  - eval-author (use for the shared standard and routing)
  - mlflow-to-atif (use only to normalize MLflow traces into ATIF)
  - eval-author-task-create (use to close an actionable audit coverage gap)
  - eval-author-inspect-trace (use to explain an Intake trace without creating an environment)
compatibility: >-
  Python 3.11+; mlflow-to-atif for MLflow; nemo CLI for Intake; Harbor and
  Docker for proof. Run proof commands in Harbor's Python environment. No model
  or provider configuration is required.
metadata:
  author: NVIDIA NeMo Platform
  tags: [evaluation, harbor, traces]
maturity: alpha
license: Apache-2.0
user-invocable: true
allowed-tools: [Bash, Read, Write, Grep, Glob]
---

# Eval Author: trace to environment

## Requirements

Use the dependencies declared above. Harbor and Docker are needed only to prove
a candidate environment.

## Purpose

Read `eval-author` for the shared evidence standard and boundaries. This flow
turns one recorded interaction into a small, reproducible Harbor task without
requiring a particular coding agent or framework.

## Limitations

This flow handles one bounded trace and execution verification. It cannot prove
unrecorded side effects, subjective outcomes, unavailable software, or private
and live external state.

## Troubleshooting

On a contract failure, preserve the artifacts and record the exact failed
command with `finalize --did-not-work`; never bypass privacy, isolation, or proof
gates.

The ATIF file is the only handoff into candidate analysis. Keep exact source
exports restricted. Do not put trace payloads, credentials, or generated task
workspaces in Git.

## Artifact contract

Use one workspace per task. Paths below are generated workspace artifacts, not
bundled skill files; `extra.*` names elsewhere are ATIF fields, not file paths.

```text
.eval-author/trace-environments/
  .gitignore
  <task-id>/
    private/source.atif.json
    private/canonical.atif.json
    private/privacy-audit.json
    private/{tool-call-inventory,tool-call-plan,tool-access,tool-call-generation}.json
    private/publications/<digest>/
    private/publication-review.json
    private/ground-truth/
    safe/trace.atif.json
    safe/privacy.json
    candidate.json
    task/
      environment/tool-call-fixtures/
    reproducibility.json
    validation.json
    summary.json
    summary.md
```

Source, evidence, ground truth, and Harbor jobs stay ignored. Only the
whitelist-only `export` command publishes the declassified reproducibility
manifest.

`scripts/trace_environment.py init` writes the parent `.gitignore` so every
task directory is ignored, makes directories owner-only, and refuses to replace
an existing task workspace. The helper prints content-free JSON summaries.

## Step 1: create the private task workspace

Choose a stable lowercase kebab-case task ID. Derive it from a caller-supplied
case name or trace ID; do not put a person's name, email address, account number,
or other private value in it.

```bash
python <skill_dir>/scripts/trace_environment.py init \
  --task-id <task-id>
```

Keep every source export and intermediate conversion under the returned task
directory. Use `umask 077` before redirecting provider output there.

## Step 2: normalize exactly one source to ATIF

Do not merge unrelated traces. Preserve the source's trace and span identifiers
in `extra`, order steps by recorded time, and record every missing or lossy field
under `extra.normalization.uncertainties` or `extra.normalization.losses`.
Never invent an instruction, tool result, file, or final answer.

### Existing ATIF

Use the trajectory as the canonical input. Do not relabel its ATIF version.

### MLflow

Read and follow `mlflow-to-atif`. Put its owner-private output under the task
workspace and use the emitted `.atif.json` file as the canonical input.

### Intake

Resolve the CLI exactly as `eval-author-inspect-trace` specifies. Read one exact
trace and every detailed span into owner-private JSON files:

```bash
umask 077
nemo intake traces get --output-format=json --workspace="WORKSPACE" --mode=detailed "TRACE_ID" > <task-dir>/private/intake-trace.json
nemo intake spans list --output-format=json --workspace="WORKSPACE" --filter.trace-id="TRACE_ID" --mode=detailed --sort=started_at --all-pages > <task-dir>/private/intake-spans.json
```

Require every returned span's `trace_id` to match the selected trace. The coding
agent converts those records to ATIF: root input becomes the user instruction;
model or assistant output becomes agent steps; each tool span becomes a paired
tool call and observation; errors and unmapped attributes remain cited in
`extra.intake`. If there is no complete human instruction, record no_candidate.

### OpenTelemetry

Prefer an existing Intake trace produced from the telemetry and use the Intake
path above. For a bounded JSON OTLP export, the coding agent may map the root and
descendant spans directly using the same rules. Preserve trace IDs, span IDs,
parent IDs, timestamps, status, and semantic attributes under `extra.otel`.
Do not parse protobuf bytes, contact a collector, or ingest remote data in this
flow. If the export cannot establish parentage or a human instruction, record
no_candidate instead of guessing.

Before continuing, require one ATIF v1.0-v1.7 object with sequential one-based
steps, a user step, and resolvable tool-call references. Limits are 128 MiB for
source and 25 MiB for canonical or safe output. Only these normalizations are
allowed:

- insert a missing JSON escape when the parser-implicated quote immediately
  follows a provider-redaction placeholder; and
- convert string-encoded ATIF image objects into image parts while omitting
  encoded binary data, including image metadata embedded in text fields.

Every operation, offset, count, and loss is recorded in the canonical ATIF and
summary. Exact source bytes remain unchanged and hashed. Reject unrelated JSON
damage instead of repairing it heuristically.

## Step 3: make a text-only safe copy

```bash
python <skill_dir>/scripts/trace_environment.py prepare \
  --task-dir <task-dir> \
  --atif <canonical-atif-file> \
  --source-kind <atif|mlflow|intake|otel>
```

The helper retains an exact owner-private original, writes the bounded canonical
ATIF and a scrubbed safe copy, replaces image parts with omission markers, and
records only redaction counts in the safe report.
It recognizes common secret fields, bearer tokens, private keys, email addresses,
phone numbers, social-security numbers, IP addresses, internal Kubernetes hostnames,
and user home paths.

The helper also writes `private/privacy-audit.json`: a complete string-field and
character denominator, URL hosts, and candidate name, organization, and street
address findings. These are contextual leads, not automatic claims. Review every
text field in `safe/trace.atif.json`, every audit finding and host, then record
who performed this trace review. Generated task files do not exist yet; review
the complete publication separately after finalization.

Use the audit's field and character denominator to plan bounded reads. If a tool
truncates output, continue through the omitted fields or character ranges; a
truncated display does not establish missing trace evidence. If review cannot
finish, retain that limitation without issuing a complete review attestation.

```bash
python <skill_dir>/scripts/trace_environment.py review-privacy \
  --task-dir <task-dir> \
  --reviewer-kind <agent|human> \
  --note "<what was reviewed and any disposition>"
```

Never copy a redacted value into a verifier. An image-only user instruction is a
blocking reason and must remain no_candidate. The scanner cannot establish that
proprietary code is safe; the contextual reviewer owns that judgment.

### Resolve tool-call access

For traces with tool calls, read `../../docs/trace-derived-fixtures.md`, then run:

```bash
python <skill_dir>/scripts/trace_environment.py inventory-tool-calls --task-dir <task-dir>
python <skill_dir>/scripts/trace_environment.py plan-tool-call-access --task-dir <task-dir>
python <skill_dir>/scripts/trace_environment.py resolve-tool-call-access \
  --task-dir <task-dir> --decisions <decisions.json> --reviewer-kind <agent|human>
# After privacy review, when at least one decision is mock:
python <skill_dir>/scripts/trace_environment.py generate-mock-tool-calls --task-dir <task-dir>
```

Select `real`, `mock`, or `none` for every function; never substitute silently.
Copy generated fixtures into the task image and merge `integration.toml` into
`task.toml`; finalization checks the wiring. The stdio MCP adapter performs
exact-match replay. Fixtures are agent-visible and cannot hold verifier truth.

## Step 4: inventory ground truth and software requirements

Before deciding candidacy, look for ground truth that is distinct from the
agent's observed answer:

- a separate reference or successful trajectory;
- expected output, golden files, fixtures, or labeled data;
- verifier inputs or reference solutions; and
- explicit human or evaluator feedback that establishes correctness.

An agent answer is not ground truth merely because the trace succeeded. Record
ground truth as `available`, `partial`, `absent`, or `unknown`. Retain available
artifacts under `private/ground-truth/`, make them owner-only, record their
SHA-256 digests, and declare whether each artifact came from an ATIF step or an
external source. ATIF provenance requires real step IDs and no external locator.
External provenance requires an empty step list and at least one immutable URI,
revision, or source ID. Never cite an instruction step as evidence for a public
benchmark fixture merely because both describe the same task. Record whether
ground truth is used for `comparison_only` or `verification`; use `none` when it
is absent or unknown. Do not copy private ground-truth values into the generated
task; generalize only what the verifier needs.

Also inventory software needed to reproduce the work or verify the outcome.
Include libraries and CLIs, desktop applications such as CAD tools, services,
hardware, and proprietary or commercially licensed software. For each item,
record whether it is actually required, its version when known, license class,
local availability, redistributability, evidence steps, and a short note.
Use the same explicit ATIF-or-external provenance object for software facts.

Inventory the complete build, runtime, solution, and verifier dependency
closure. Pin images, packages, repositories, and tools where reproducibility
depends on them. The grading path must not download dependencies or contact a
live service. A dependency needed only by the verifier still belongs in the
inventory; common misses include fonts, compiler headers, package indexes, and
language registries.

Do not silently replace required software with a different application or mock
when the requested behavior depends on the real product. Required unavailable
software prevents candidacy. Unknown availability keeps the environment
unproven until resolved. Proprietary or non-redistributable software is usable
only when a legitimate, reproducible runtime path exists; never copy licensed
binaries into the task.

## Step 5: decide candidate or no_candidate

Read only `safe/trace.atif.json`. Later user corrections outrank earlier turns.
Every decision must cite real ATIF `step_id` values. Read
[references/candidate-record.md](references/candidate-record.md) for the
`candidate.json` shape before writing the decision.
Run its read-only `check-candidate` command before construction; it checks
metadata, not execution, privacy review or readiness.

Use `candidate` only when the request and expected outcome are complete,
reproducible without private or live external state, and objectively testable.
This basic flow supports execution verification only. Do not add a model judge
or convert a subjective, visual, or prose-quality outcome into a brittle string
check.

For no candidate, use the reference's null and empty fields and stable
`reason_codes`; retain the `ground_truth` and `software_requirements`
inventories even when empty. Ground truth may be absent without blocking a task,
but its absence must be explicit. Use
`required_software_unavailable` or `proprietary_runtime_unavailable` when the
software inventory blocks reproducibility.

For batch failures, use the reference's specific source or construction
`reason_codes` instead of `insufficient_trace_evidence`. Record the failed
command or contract check through Step 7's `finalize --did-not-work`, not an
extra `candidate.json` field.

## Step 6: author and prove a candidate environment

Skip this step for no_candidate. Under `<task-dir>/task/`, create the smallest
Harbor task that reproduces the initial state and objectively verifies the
generalized outcome:

- `task.toml` with realistic timeouts and resources;
- `instruction.md` matching `candidate.json` without leaking tool names or test logic;
- `environment/` containing prerequisites but never the solution;
- `tests/test.sh` that grades only the observable outcome;
- `solution/solve.sh` with the reference solution; and
- `README.md` containing reviewer-facing development context, not a copy of the
  agent instruction.

`task.toml` must explicitly set `[verifier].environment_mode = "separate"`,
`[verifier].network_mode = "no-network"`, and a
`[verifier.environment]` table whose `network_mode` is also `"no-network"`.
The helper rejects shared verification because it lets the grader observe or
alter the agent container and can expose hidden tests to the agent. When the
grader needs a different image, provide a verifier-owned `tests/Dockerfile` or
a pinned verifier image through `[verifier.environment]`; include its complete
grading dependency closure and `/tests` tree.

For a multi-step task, every step inherits the top-level separate verifier.
An explicit `[steps.verifier]` override must not select `shared`. When a step
defines `[steps.verifier.environment]`, that environment must also set
`network_mode = "no-network"`. Transfer only the declared agent-produced
artifacts into the verifier environment; do not mount or copy the agent
workspace wholesale.

Read and follow `references/environment-integrity.md` for agent-network,
contamination, portability, repeat-run, and negative-control requirements.

Write the reviewer-only task README using the integrity reference's required
sections. Human-supplied or reviewed `Relevant experience` is necessary for
readiness; never invent it.

Do not copy private trace payloads into the task. Include only the minimal files
needed to reproduce the starting state. Pin external source to an exact public
commit when it is truly required; otherwise prefer a small local fixture.

Before Harbor, record the exact task tree and its static integrity scan:

```bash
python <skill_dir>/scripts/trace_environment.py record-reproducibility \
  --task-dir <task-dir>
```

Execute the repeat-run protocol in `references/environment-integrity.md`.
For example, record the first NOP job's inputs and then run it:

```bash
python <skill_dir>/scripts/trace_environment.py record-run-inputs \
  --task-dir <task-dir> --arm nop --job-dir private/jobs/nop-1
harbor run -p <task-dir>/task -a nop \
  --jobs-dir <task-dir>/private/jobs --job-name nop-1
```

Repeat with `nop-2`, `oracle-1`, and `oracle-2`, selecting the matching arm and
agent. Follow the integrity reference to declare and run the negative control.
Retain every exact Harbor job and use the reference's `record-validation`
command. Never hand-write rewards, job IDs, exceptions, or checksums. Do not
weaken the verifier to make Oracle pass. Failed proof remains technical evidence.

If Harbor or Docker is missing, technical status is `not_run` and the environment
is `unproven`; do not describe it as ready. The integrity reference defines the
conditions for technical status `passed` and the limits of that claim; passing
proof does not establish human review.

## Step 7: finalize and verify the summary

Record concise facts about the conversion and construction. Do not include raw
payloads or redacted values in these flags.

For a ready candidate:

```bash
python <skill_dir>/scripts/trace_environment.py finalize \
  --task-dir <task-dir> \
  --status candidate \
  --human-reviewed \
  --worked-well "<evidence-backed success>" \
  --did-not-work "<evidence-backed limitation>"
```

For no_candidate:

```bash
python <skill_dir>/scripts/trace_environment.py finalize \
  --task-dir <task-dir> \
  --status no_candidate \
  --reason "<why no reproducible environment can be built>"
```

Then verify digests and required artifacts:

```bash
python <skill_dir>/scripts/trace_environment.py check \
  --task-dir <task-dir>
```

The helper derives environment status rather than accepting a claimed status:
failed technical proof becomes `failed`; passed proof with the required separate
no-network verification, `--human-reviewed`, and no required software whose
availability is `unknown` becomes `ready`; every other
candidate is `unproven`. A shared verifier is a contract error rather than an
unproven candidate. The human-review flag means a human supplied or reviewed
Relevant experience and the generalized task. It is distinct from the earlier
contextual privacy review, which records either an agent or human reviewer.

## Batch and publication

For a multi-task run, write a checked-in manifest with stable task IDs and
source paths. This makes the denominator explicit and reruns idempotent:

```json
{
  "schema": "nemo.eval_author.trace_environment_batch.v1",
  "members": [
    {"task_id": "stable-id", "atif": "private-source.atif.json", "source_kind": "atif"}
  ]
}
```

Keep a real manifest's source paths private if they reveal internal layout.
Prepare missing members, safely resume existing ones, and report every member:

```bash
python <skill_dir>/scripts/trace_environment.py batch-prepare \
  --manifest <manifest.json>
python <skill_dir>/scripts/trace_environment.py batch-status \
  --manifest <manifest.json>
```

`denominator` must equal the selected source set. Never report only candidates
or successes. A malformed workspace remains in the `batch-status` report with
status `invalid` and makes the report invalid without hiding other member rows.

Before any export, including `no_candidate`, read and follow
`references/publication-review.md`. Prepare the complete private publication
preview, review every exported file and path, and attest its exact digest before
exporting. Trace privacy review and `--human-reviewed` do not replace this gate.
Any change to the public product requires a new publication review; a successful
`check` alone is not publication approval.

Report the task ID, `candidate` or `no_candidate`, ground-truth availability and
artifact count, required software and licensing constraints, environment
status, technical status, review status, verifier isolation, privacy review
status, NOP and Oracle rewards when run, and paths to
`summary.md` and the generated task. A `valid: true` check proves the recorded
files are internally consistent; Harbor is the authority for whether the task
actually runs.
