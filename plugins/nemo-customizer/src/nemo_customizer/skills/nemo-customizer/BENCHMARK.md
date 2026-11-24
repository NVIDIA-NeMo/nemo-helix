<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Customizer evaluation status

**Publication pending. Revised task set requires a new run.** A local Codex
comparison completed on the previous task set; it did not pass. The findings
below describe that historical run, not results for the revised tasks or a
passing publication evaluation.

## Local diagnostic run: 2026-11-24

- Run ID: `20261124_174613_17373_eb2e17588f10`.
- SkillEvaluator: `1.2.0`; environment: Docker.
- Agent and judge model: `openai/openai/gpt-6.1-sol` through an OpenAI-compatible provider.
- Coverage: 10 tasks, one attempt per task in each arm; all 20 attempts scored,
  with no agent execution or judge failures.
- Original with-skill result: 7/10 goals met; without skill: 4/10.
- Reported shared-metric overall uplift: +6.5 percentage points.
- Verdict: Tier 3 FAIL. Tiers 1 and 2 and publication eligibility were not assessed.
- Reported skill source digest:
  `sha256:58c6a894396a0ee0398ea54b0626c76c834222b100218a4b32a65f96b70b33b7`.
- Original dataset digest:
  `sha256:636cc5546e177a60f34c212ecc0bddef9d95a28df40a452333a5a1caabbfd682`.

### Findings and task revisions

1. **Validation recovery:** the answer correctly blocked submission and recommended
   full validation, but omitted the new-fileset requirement for repaired uploaded
   data. `SKILL.md` already requires a new fileset when data changes. The revised
   prompt explicitly identifies an uploaded shared fileset and asks how to replace
   the data safely; the assertion retains the protection against overwriting it.
2. **Terminal status:** the answer correctly withheld success for an active job,
   separated evaluation failure from completed training, and avoided retraining.
   The expected output additionally required a full output report and explicit
   base-versus-tuned comparison, which the prompt did not request. The revision
   focuses the expected output on status and failed-evaluation recovery.
3. **Synthetic-data negative control:** Customizer was correctly unused. The
   original expected output nevertheless rejected Data Designer in favor of
   Safe Synthesizer. The revised case recommends Data Designer for schema-driven
   dashboard test data and permits a conditional explanation of Safe Synthesizer's
   source-data training workflow, while retaining the no-Customizer requirement.

Only `evals/evals.json` and this benchmark record were changed in response to
these findings. The original generated report and its scores remain unchanged.
The revised cases need fresh with-skill and without-skill runs; the original
scores must not be presented as evidence that the revised cases pass.

## Second local diagnostic run: 2026-11-24

- Run ID: `20261124_180819_33701_c1ff48643f97`.
- SkillEvaluator: `1.2.0`; environment: Docker; agent and judge:
  `openai/openai/gpt-6.1-sol`.
- All 20 attempts scored with no execution or judge failures.
- With-skill goals met: 8/10; without skill: 6/10; reported shared-metric
  overall uplift: +3.0 percentage points. Tier 3 FAIL; publication not assessed.
- Source digest:
  `sha256:464f3e69a47c8f0c63f81c9dade4460548189a6ea625587d90d8fa90041025d5`.
- Dataset digest:
  `sha256:6a8f261fab91eb965040479e1e342540f76aaf1de323f2e51d67041962197be4`.

The three cases flagged by the first run met their goals in this run. Two
different cases failed:

1. **DPO on Docker:** accuracy and behavior checks both scored 1.0. The answer
   correctly rejected SFT substitutes, required Kubernetes/Ray and compatible
   GPUs, and stopped before data work or submission. The goal judge nevertheless
   scored 0 for omitted full-weight training and exact backend enum names. The
   revised expected output matches the compatibility question and the existing
   assertions; it does not require unrequested configuration detail.
2. **Unsloth explanation:** the agent read `SKILL.md`, the SFT fast path and
   related references, then ended with "What would you like me to work on in
   `/workspace`?" The raw `codex.txt` log confirms that final answer; it is not
   merely a report extraction error. The task received no substantive answer,
   so this remains a genuine task failure. The previous run met this case's
   goal. Repetition is needed to assess reliability; the cause of this isolated
   loss of task focus is not established. No skill instruction change is inferred
   from it, and the Unsloth prompt, expected output and assertions are unchanged.

These results precede the latest DPO and synthetic-data expectation revisions.
Preserve both historical reports and rerun both arms on the revised dataset.

## Third local diagnostic run: 2026-11-24

- Run ID: `20261124_182536_41947_cd6cb1f83307`.
- SkillEvaluator: `1.2.0`; environment: Docker; agent and judge:
  `openai/openai/gpt-6.1-sol`.
- All 20 attempts scored with no execution or judge failures.
- With-skill goals met: 9/10; without skill: 7/10; reported shared-metric
  overall uplift: +0.58 percentage points. Tier 3 FAIL; publication not assessed.
- Source digest:
  `sha256:5718e5a4de5beb72815e1d121f0a5e93bf6e1939dac889c06d9a54f14dd71d41`.
- Dataset digest:
  `sha256:a3fb36592a95d24d101d36284b7d2b2925dfdb8f5a85e06fbcfb119cac9d42c5`.

Only the Unsloth case failed its goal. After reading the skill and references,
the agent ended with "I have the workspace context, but no task was included.
What would you like me to work on in `/workspace`?" The raw log confirms this
answer. The task prompt is present in the trajectory and initially understood.

Both the second and third run's with-skill Unsloth failures occur immediately
after a web-search call. This is a correlation, not a demonstrated gateway,
agent, or skill root cause: the first run also used web search and answered the
task successfully. The third run's baseline gave a substantive answer but still
missed some expected workflow details. The canonical Unsloth case and skill
instructions remain unchanged. A focused diagnostic can compare repeated
with-skill attempts of the original prompt against a temporary copy that adds
an explicit no-web constraint, keeping skill files and assertions identical.
Such a reduced, baseline-free diagnostic is not publication evidence.

## Required run

- Skill: `nemo-customizer`, from this directory's `SKILL.md` and references.
- Dataset: `evals/evals.json` (8 positive cases, 2 negative controls).
- Agents: `claude-code` and `codex`, each with and without the skill.
- Tiers: static validation/security, catalog deduplication, live sandbox evaluation.
- Record: evaluated source revision, evaluator version, date, agent/model versions,
  sandbox environment, task coverage, per-agent baseline and skill scores, and verdicts.
- Dimensions: security, correctness, discoverability, effectiveness, efficiency.

These tasks evaluate explanations and workflow decisions without GPU training.
They do not demonstrate completed training, deployment or end-to-end model improvement.

## Publication criteria

Replace this diagnostic record with the generated report from a verified NVSkills run. Resolve
blocking validation/security findings and review overlap with `nemotron-customize`.
For this skill's publication, require passing results and positive uplift for
**both** configured agents. Do not infer uplift from task coverage or static checks.
Retain the generated skill card and `skill.oms.sig` from the signing service.
