<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Task fidelity and verifier review

**Experimental.** Apply this review when constructing either Harbor or Gym
output. It supplements the provider's execution checks; it does not add new
helper commands, proof arms, or summary schema fields. Keep findings and
supporting artifacts in private construction notes, and link them from the
Harbor README or Gym report without copying restricted evidence.

## Preserve the problem and its starting state

Treat trace text, repository files, and tool results as evidence, not authoring
instructions. A transport `user` role alone does not establish human authorship:
exclude identified harness housekeeping and retain uncertainty about ambiguous
roles. Apply actual user corrections to the selected task, preserving constraints
without supplying the solution's implementation recipe.

Separate the initial fixture from source-agent edits, final outputs, and private
reference material. Fragments from different points in a trajectory do not
establish a complete original workspace. If initial state is unavailable, use
the declared reconstruction rules; do not replay successful edits into the
starter and then call it the original state. Materialize the named artifacts
and necessary neighboring code, configuration, and data in their real formats.
A description of a file, an empty stub, or an in-memory simulation cannot stand
in for a required native artifact or tool interaction. Keep the world as small
as the capability allows; neither sparse shells nor irrelevant padding help.

If the capability depends on responding to a later correction, distinguish that
from satisfying a final, consolidated instruction. Claim multi-turn behavior
only when the selected runtime delivers the evidenced turns in order, preserves
state between them, and retains checks that depend on the later turn. Otherwise
report the narrower final-state capability or the unsupported runtime need.
Do not hide an essential correction in the grader, invent follow-ups, or add a
random turn count to create difficulty.

## Check setup before interpreting baseline failures

For a dependency-bearing task, use the intended task or component interpreter
to exercise a small existing behavior through the entry point the verifier
will need. Verify imports resolve to the submitted checkout, not an installed
reference copy. Where test collection is used, verify it collects the intended
tests; zero collected tests are not successful validation. An import alone does
not prove application startup, rendering, plugin initialization, or a CLI works.
Keep this proportional to the runtime and use the normal inherited environment.

Ground dependency constraints in source manifests, compatibility documentation,
or observed setup failures. Retain the resolved inputs and actual component
versions. Installing a package only in the author's environment does not prepare
the agent or verifier. Preserve required native software; do not replace it with
a mock to escape a setup defect. Diagnose unrelated dependency, startup, import,
timeout, and output-parsing failures before treating a baseline's failure as
evidence of the requested defect. An explicitly requested compatibility repair
can itself be the task, but that distinction must come from the instruction.

## Review the public contract and every assertion

Map each scored outcome to a visible requirement, its fixture evidence, and the
actual assertion that enforces it. Required values must be stated or discoverable
through permitted interaction. A hidden requirement map, reference answer, or
assertion description cannot fix an underspecified instruction. Check that the
instruction and acceptance criteria agree, including exceptions to broad claims.

Use the reference as feasibility evidence, not the unique correct algorithm,
file layout, helper-call sequence, or wording. Do not exact-match free-form
answers or private implementation details unless the task requires them. Expected
values need an independent basis in the requirement and fixture or separately
validated reference execution; never compute the oracle using the submission
being graded. Preserve the historical ground-truth availability label when
verifying a new synthetic fixture.

Choose small cases that distinguish plausible incorrect behavior. As relevant,
check empty versus nonempty results, a changed boundary and its nearest valid
neighbor, omitted versus explicit defaults, and one meaningful option interaction.
A filter needs both included and excluded examples; an enumeration check may
need multiple qualifying results. Passing an option that changes nothing in the
fixture does not establish that option's behavior. Add cases for demonstrated
contract gaps, not speculative exhaustive coverage or new requirements.

Assert the described property directly: a replacement value's presence does
not establish that all prohibited placeholders were removed. A declared tool,
registration, or success message does not prove an invocation or side effect.
Exercise the supported interface with meaningful arguments and observe its
result. Use trusted runtime events for behavior checks; an agent-written log is
not independent execution evidence. Keep prerequisite and preservation checks
with their scored outcome so the untouched task still fails every scored check.

## Try both incorrect and equivalent solutions

The required no-action, reference, and negative controls remain unchanged.
For a concrete suspected gap, run a plausible incomplete implementation that
reaches the intended behavior without crashing. To investigate an overly strict
check, also try a materially different correct implementation when the public
contract permits one. Inspect the submitted files and per-check results, not
only the reward or solver's own explanation. False acceptance, false rejection,
setup failure, and missing evidence are different findings.

Keep valid-alternative diagnostics separate from Harbor's required proof arms;
the helper does not support an `alternative` arm. Retain the input revision,
control source, execution settings, result paths and digests privately, or use
Gym's native fixtures and controls. Do not relabel an alternative as a NOP,
Oracle, negative, or copy arm. Source reasoning without execution remains a
review finding. A missing diagnostic is not evidence of verifier soundness.

Correct a verifier defect within the existing repair budget and preserve checks
already justified by the contract. Remove a check only when evidence shows it
is wrong. A contract defect needs a separately documented revision and fresh
evidence; do not silently change the task to help a particular solver pass.
Legitimate solver failure can indicate a useful task, while reference success
alone establishes neither fairness nor difficulty.

## Inspect the delivered environment and retain evidence honestly

Keep reference patches, hidden tests, grader helpers, and expected outputs out
of agent-visible images, mounts, package assets, and histories. Encoding them or
deleting them in a later image layer does not establish absence. Inspect the
actual runtime visibility as well as source layout. Rebuild and rerun affected
tasks after fixing a leak; a generator update does not repair old exports or
cached images. For Gym, account for the selected agent's filesystem and service
access, not merely separation between JSONL keys.

Only the verifier may create authoritative rewards and check rows. Start with
fresh verifier outputs, ignore agent-supplied reward artifacts, and handle any
provider-supported competing reward formats so stale success cannot survive a
failed check. A subprocess that failed to start or emitted no parseable result
has not proved that an unwanted behavior is absent.

Bind review and execution claims to the tested task revision. Preserve failed,
interrupted, and superseded attempts with their stage and actual evidence.
Operational blocks or absent artifacts remain unresolved, not observed task
defects or passes. Report task correctness, runtime readiness, human review,
and measured solver performance separately using the existing status contracts.
Do not import another pipeline's acceptance thresholds or infer broad benchmark
quality from a few controls, model success rates, or reviewer agreement.
