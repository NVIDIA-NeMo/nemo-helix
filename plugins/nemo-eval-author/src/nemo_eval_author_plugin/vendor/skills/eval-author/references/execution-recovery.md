<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Preserve evaluation evidence during recovery

Apply this guidance when executing or reporting an evaluation, including after
an endpoint, adapter, dependency, or environment failure. Preserve the selected
provider, target agent, tasks, fixtures, and original verifiers during recovery.
Completion requires the selected provider's native artifacts from that agent and
those verifiers: Harbor jobs and trials, or Gym rollouts with their verifier
rewards and health records. A run through the other provider is not a fallback
completion path.

## Repair the integration, then rerun

Retain failed trials or rollouts, raw results, logs, and the command and
configuration that produced them. Diagnose the failing boundary before changing
it. An endpoint rejecting an unsupported request parameter is an integration
failure, not an observed failure of the agent's task-solving ability.

Within the authorized scope, repair the integration while preserving the task,
fixture, and grading semantics. For example, removing an unsupported
`prompt_cache_key` from the affected endpoint request may be a compatibility
fix; replacing the agent, application, or verifier is not that fix. Record the
patch and affected source/configuration versions, including provider, adapter,
model/endpoint settings and task/verifier revisions or digests. Do not record
credentials. Run the selected target agent through the selected provider again,
with the original verifiers and fresh native outputs. Keep the failed attempt and
link it to the rerun instead of overwriting or relabeling it.

Approval for a temporary compatibility patch does not authorize a different
evaluation protocol. If repair cannot preserve the selected path within scope,
report the evaluation incomplete with the remaining blocker. A separately
requested protocol change produces a separately identified evaluation; its
results do not retroactively complete the original run.

## Keep diagnostic work separate

Local application runs, direct endpoint calls, and manual verifier probes may
help debug the integration. Never substitute assistant-authored task answers,
local diagnostic runs, reconstructed graders, or manufactured native artifacts
for the requested evaluation. Do not synthesize provider results from local
outputs, even when the real application ran and the workaround was disclosed.

When the requested run remains incomplete, start a diagnostic report with
**Diagnostic only — Harbor evaluation incomplete** or **Diagnostic only — Gym
evaluation incomplete**, naming the selected provider. Retain that qualification
in later summaries, comparisons, and checklist recaps. Label any diagnostic score
beside the score; do not lead with an unqualified pass rate or “comparison
complete.” Keep diagnostic outputs outside authoritative provider aggregates.
Never mark the requested run or whole pipeline complete based on those outputs.

## Report what actually executed

For each completed evaluation claim, identify the actual native run paths (Harbor
jobs and trials, or Gym rollout and health files), selected target agent/model
configuration, tested task/verifier versions, and the retained native results and
verifier outputs. Task controls such as Harbor's Oracle or Gym's known-correct
fixtures establish reference behavior, not results for the target agent. Missing
runs or verifier evidence remain explicit; report any completed subset separately
from the outstanding requested work.

Preserve raw rewards and failure details: Harbor exceptions and trial statuses,
or Gym failure reasons and health verdicts. Add an explanation that distinguishes
infrastructure failures, blocked or missing runs, and agent-quality failures
without rewriting those outcomes. Show requested and completed run counts and
the denominator of every score. An infrastructure failure is not evidence of poor
agent quality; a genuine low score from completed runs does not justify weakening
the task or its grader. Completion describes execution with evidence, not a
requirement that the agent pass.

Runs recorded with the [evidence recorder](task-validation.md#record-controls-and-agent-runs)
use the same categories. `infrastructure_error` is an infrastructure failure or
missing evidence. `behavior_failed` on an agent run is an agent-quality failure;
on a control, the task or verifier did not behave as declared. `diagnostic` cases
are diagnostic work and never count as acceptance evidence.
