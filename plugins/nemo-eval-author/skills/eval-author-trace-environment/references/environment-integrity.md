<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Environment integrity protocol

Apply this protocol to every candidate before finalization.

## Existing Harbor runtime preflight

Before building images or starting proof jobs, run this read-only check in the
existing Harbor Python environment. Run it again when that environment changes;
do not infer compatibility from a minimum version or a previous result:

```bash
python <skill_dir>/scripts/trace_environment.py check-runtime
python <skill_dir>/scripts/trace_environment.py check-runtime --task-dir <task-dir>
```

The first command probes provider-declared configuration and result fields; the
second also makes Harbor validate the authored task and compute its checksum.
Exit code 1 / `valid: false` means do not launch proof jobs with that runtime.
Keep the report private with construction evidence. Use the same interpreter's
Harbor installation for receipts and jobs, not an unrelated `harbor` on PATH.
The reported version is provenance, not an allowlist: development builds and
future versions must retain the required capabilities too. If unavailable, use
another already-installed compatible runtime when present; do not automatically
install, upgrade or patch the provider, or synthesize missing result fields.

This preflight does not start Docker, download anything, or prove execution.
It checks only the single-step proof shape understood by this helper. Multi-step
trial results may omit trial-level verifier mode even when their task config
accepts separate verification; the task-specific preflight fails closed there.
Do not flatten a multi-step task just to pass. Every actual job must still pass
the existing checksum, identity, reward and separate-mode result checks.

## Preserve the recorded task's scope

Minimal construction does not authorize narrowing the requested outcome or
permitted implementation choices. Preserve allowed languages, dependencies and
multi-file deliverables; declare their complete output closure for transfer
instead of imposing a single-file or standard-library-only solution for
convenience. Record any evidence-backed generalization explicitly. If necessary
dependencies or outputs cannot be supported within isolation, retain that
limitation rather than silently substituting an easier task.

## Network isolation

In addition to the separate no-network verifier contract, require the agent
environment to set `[environment].network_mode = "no-network"`. A step-local
agent environment may inherit that setting or repeat `network_mode =
"no-network"`; it must never enable public networking. Resolve and vendor the
complete build and runtime dependency closure before grading.
This is an authoring requirement, not a claim that the static helper verifies
the complete dependency closure.

### Configuration examples

Minimal isolated task configuration:

```toml
[verifier]
environment_mode = "separate"
network_mode = "no-network"

[verifier.environment]
network_mode = "no-network"

[environment]
network_mode = "no-network"
```

Add a step-local environment only when that step needs a different verifier
image or resource configuration:

```toml
[[steps]]
name = "grade"

[steps.verifier]
environment_mode = "separate"

[steps.verifier.environment]
network_mode = "no-network"
```

### Artifact handoff and verifier images

In Harbor's separate environment, explicitly declared artifacts are restored at
their original `source` paths. An artifact `destination` changes the host result
layout, not the path used by the verifier. For example:

```toml
# Top-level task.toml, before any section header:
artifacts = [{ source = "/app/result.txt", destination = "submission/result.txt" }]
```

The verifier reads `/app/result.txt`, not `/logs/artifacts/submission/result.txt`.
The conventional `/logs/artifacts` directory is a different publish mechanism;
do not mix these layouts. Check the installed Harbor artifact API when using a
different provider revision, and prove the actual handoff with NOP and Oracle.
Transfer only the declared outputs; do not broaden the transfer to make a path
mismatch disappear.

Choose one verifier image path and include its complete offline grading closure:

- **Build recipe:** provide `tests/Dockerfile` with the required runtime and
  `COPY . /tests/`; leave `[verifier.environment].docker_image` unset so Harbor
  builds that recipe. The resulting image must contain `/tests/test.sh` and every
  file it needs.
- **Prebuilt image:** set `docker_image` to an immutable image already containing
  the complete `/tests` tree and runtime. A vanilla language image does not
  contain the authored tests. Do not assume a sibling Dockerfile is built or
  tests are uploaded when a prebuilt image is selected.

Check inherited image entrypoints, commands, users and environment variables
against Harbor's startup and execution lifecycle. A successful image build does
not prove that the container stays available or its tools work under the
configured user. Retain build, startup and grading failures separately; do not
infer a verifier or reference-solution defect when execution never reached them.

Separate verification does not preserve the agent's running processes. For a
configuration deliverable, a fresh verifier-owned service can test the declared
configuration. This proves replay, not that the agent left a service running.
Do not substitute replay for a requested live-state outcome; if the provider
cannot test that outcome within the isolation contract, retain the limitation
and use `verifier_not_isolated` rather than weakening the task. Label agent-side
collected observations as untrusted evidence, not independent runtime proof.

## Reviewer documentation

The task README is not passed to the agent. Give it a level-one task title and
these substantive level-two sections:

- `Difficulty explanation`: why the task is difficult for agents and humans;
- `Environment and software requirements`: runtimes, services, hardware,
  versions, licensing, and availability constraints;
- `Ground-truth provenance`: what establishes correctness and where that
  evidence came from, without exposing private values;
- `Solution explanation`: the high-level reference approach without duplicating
  `solution/solve.sh`;
- `Verification explanation`: the observable outcomes and how the verifier
  distinguishes success from failure; and
- `Relevant experience`: human-supplied experience relevant to authoring or
  reviewing the task.

Keep each section concise and evidence-backed. Do not repeat `instruction.md`,
reveal verifier internals to the agent, or invent author experience. If a human
cannot supply and review `Relevant experience`, keep the environment `unproven`
rather than claiming it is ready.

## Reproducibility manifest

Record the exact task tree before running Harbor:

```bash
python <skill_dir>/scripts/trace_environment.py record-reproducibility \
  --task-dir <task-dir>
```

The manifest hashes every task path, file byte, directory, and executable bit.
It records a declared source revision, Dockerfile base and `# syntax=` frontend images, external
`COPY --from` and `RUN --mount=from=...` image dependencies, configured agent,
verifier and step `docker_image` references, network modes,
and one image-reference state: `local_only`, `image_pinned_recipe`, or
`immutable_image`. `image_pinned_recipe` means the agent Dockerfile exists
and every inventoried external image is pinned by digest. Local named or numeric
build stages are distinguished from external images. Unresolved variable-based
references keep the task `local_only`. A configured agent image is immutable
only when its reference contains a SHA-256 digest and all inventoried image
dependencies are immutable. Mutable or unresolved frontend references keep the
recipe `local_only`; pin an external frontend by digest too. Only active parser
directives before a blank line, ordinary comment, or build instruction count.
Use Harbor's `docker_image` field, not `image`.
These states describe image pinning only. The manifest and public result
explicitly report `dependency_closure: "unverified"`: arbitrary package
downloads, tool installations, and external build inputs are not proven
reproducible. Do not describe `image_pinned_recipe` as a rebuildability verdict.

The scan fails closed on Git metadata in `task/environment`, exact solution or
hidden-test files copied into the agent build context, private keys, bearer
credentials, credential-bearing URLs, and symlinks. It is a static task-tree
gate, not an inspection of opaque image layers. Keep image-construction
transcripts private and inspect pre-existing opaque images separately.

A duplicate-file finding establishes byte equality, not the file's semantic role:
public starting-state fixtures can also be retained by the verifier. Report that
distinction when supported by the instruction and file contents, while retaining
the failed gate. Do not reformat duplicates or weaken the scanner to evade it.

## Repeat and negative-control proof

Run every arm as an independent Harbor invocation with new task containers.
Require at least two NOP runs, two
Oracle runs, and one negative control. NOP must receive reward `0`, Oracle must
receive reward `1`, and every run must finish without an exception.

The negative control must be a deliberately incomplete or subtly incorrect,
non-crashing solution relevant to the task and must receive reward `0`. Do not
reuse NOP as the negative control. Record the mutation in private construction
notes without exposing verifier logic.

Check the installed Harbor custom-agent constructor and execution interface
before choosing a base class. Built-in Oracle may receive task paths and other
orchestration arguments that custom import-path agents do not; importing an
Oracle subclass alone does not prove it can be instantiated as a custom control.

Before **every** invocation, run `record-run-inputs` with its arm and future job
directory (see the skill's NOP example). This writes an owner-private receipt
under `private/run-inputs/` and refuses existing jobs or receipts. It binds the
full task-tree digest, including executable bits, and Harbor's directory
checksum to the future job. Proof commands use Harbor's `Task.checksum` API
directly; use the existing Harbor Python environment.

For the negative control, retain the custom agent source and supply its exact
`module:Class` identity and task-specific mutation rationale before execution:

```bash
python <skill_dir>/scripts/trace_environment.py record-run-inputs \
  --task-dir <task-dir> --arm negative --job-dir private/jobs/negative-1 \
  --negative-agent negative_agent:IncompleteSolution \
  --negative-source private/negative_agent.py \
  --negative-rationale "<specific incomplete behavior this control exercises>"
PYTHONPATH=<task-dir>/private harbor run -p <task-dir>/task \
  -a negative_agent:IncompleteSolution \
  --jobs-dir <task-dir>/private/jobs --job-name negative-1
```

Ensure that the import resolves to that retained source and retain any local
dependencies as construction evidence. The helper compares the declared source
digest and recorded agent identity; it does not import or execute the control
to infer its semantics. Both `config.agent.name` and `config.agent.import_path`
representations are supported, but conflicting nonempty fields are rejected.
Built-in NOP/Oracle import paths are recognized and cannot serve as the negative
control. NOP and Oracle proof arms must identify their respective built-in agent.

Retain each exact Harbor job directory. Repeat each job option once per result:

```bash
python <skill_dir>/scripts/trace_environment.py record-validation \
  --task-dir <task-dir> \
  --nop-job-dir private/jobs/nop-1 \
  --nop-job-dir private/jobs/nop-2 \
  --oracle-job-dir private/jobs/oracle-1 \
  --oracle-job-dir private/jobs/oracle-2 \
  --negative-job-dir private/jobs/negative-1 \
  --harbor-version "$(harbor --version)"
```

The helper requires unique Harbor job IDs, each result's task checksum matching
the current task and its pre-run receipt, separate verifier mode in every
result, and matching pre-run/current/reproducibility task-tree digests. Old
results cannot be attached to a changed task by regenerating its manifest.
The report records `distinct_jobs: true` only after checking unique job paths
and IDs. It explicitly reports `container_freshness: "unverified"`, including
in exported results. Neither job identity nor the pre-run task receipt proves
container identity or absence of reused container state. Fresh containers remain
an execution requirement, not a property established by this helper. A passing
report means the retained rewards and evidence are consistent, not that container
freshness has been verified. Failed rewards remain technical evidence: retain
them and report the candidate as failed.

Validation v5, reproducibility v3, and public product v3 replace the previous
contracts and their broader claim names. Preserve previous reports privately
before regenerating them from the unchanged task and retained evidence; do not
edit schema strings or claims in place. Jobs with genuine matching pre-run
receipts can be rechecked without claiming a new execution. Historical proof
without those receipts must be rerun; do not fabricate receipts for completed
jobs. Keep historical fixture results labeled with their original contract.
Run receipts and negative-control source/rationale stay private and are never
exported.
