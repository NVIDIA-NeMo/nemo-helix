<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Customizer evaluation status

**Codex Tier 3 PASS; publication pending.** The latest run scored all 20
attempts: 10/10 goals with the skill versus 8/10 without it, with +6.08 percentage
points reported shared-metric uplift. It used one attempt per case per arm on
the revised dataset. This is a passing Codex sample, not proof of repeatability
or resolution of the historical task-loss issue. Publication remains unassessed.

## Passing single-attempt Codex comparison

- Run ID: `20261125_020204_14943_21b62c17f8ab`.
- SkillEvaluator: `1.2.0`; environment: Docker; agent and judge:
  `openai/openai/gpt-6.1-sol`.
- Coverage: 10 cases (8 positive, 2 negative), one attempt per case in each
  arm. All 20 attempts scored; no execution or judge failures; native jobs
  retained. The dataset snapshot's prompts, expected outputs, assertions and
  routing fields match the current authored dataset.
- Goals and required checks with skill: 10/10; baseline: 8/10. All 10 attempts
  in each arm were security-clean.
- Reported Codex completed-agent score: 97.42%; shared-metric uplift: +6.08
  percentage points. These are distinct report measures.
- Verdict: Codex Tier 3 PASS. Tiers 1 and 2 and publication eligibility were
  not assessed; Claude Code was not evaluated by this run.
- Source digest:
  `sha256:ace6c246164cc11779fd0d030255739dc27907983ec728c9405d14aa91e6cff7`.
- Dataset digest:
  `sha256:2221988158008f3e43723b8dae5da8c8a5195bd3cf8b8ded94dc6796f6bfdd08`.
- Source revision and evaluator container revision were not recorded by the
  orchestration input; the digests above identify the evaluated content.

This run evaluates the corrected negative control. Historical failures remain
attached to their original datasets and are not replaced by this pass. One
attempt per case does not establish a stable failure rate, a resolved task-loss
cause, or a reliable live MR gate. No further skill or grader change is inferred
from the passing sample. `SKILL.md` was not changed to obtain this result.

## User-risk review of the authored cases

This revision evaluates useful advice and observable execution boundaries,
rather than requiring proof of a particular reference-reading sequence or
wording. Expected outputs and assertions were revised together so that omitted
editorial details do not remain hidden goal requirements. This deliberately
changes the scope of some requirements; it is not evidence that an unchanged
rubric has been calibrated. The review itself provides no live grader
calibration or agent-quality evidence; the subsequent passing sample is
recorded separately above.

| Case | Concrete user risk | Revision and reason |
| --- | --- | --- |
| Default SFT | An unusable platform plan, leaked test data, or a false claim of improvement | Keep documented defaults, model resources, isolated labeled test data, validation, LoRA-capable serving and comparable base/tuned measurement. Remove mandatory file reads, a separate approval sentence and literal `deployment_config` in prose. Judge data isolation rather than recitation of the words "before upload". Supplied JSON must still be correct; no operational action is authorized. |
| User-data Unsloth | Invalid backend/schema, mishandled messages, or silently changed training duration | Check essential Unsloth fields, 4-bit LoRA, chat templating, the user's LR/epochs, GPU/image suitability and dry-run advice directly. Remove file-reading choreography and the unrequested full evaluation plan; do not infer quality from which references were read. |
| Gated model | Credential exposure or a platform download that cannot authenticate | Keep platform credential scope, gated access and fileset `token_secret`. Accept any valid secret name and an explanation of verification instead of requiring a particular secrets-list command. Explicitly prohibit reading token values or operating the platform. |
| DPO on Docker | Wasted data work or silently replacing preference learning with SFT | Keep the Docker blocker, `rl`, compatible Kubernetes GPU capacity and an actionable access request. Ray is implementation context, not a required word in this compatibility answer; incorrect infrastructure claims still fail. |
| No GPU | Preparing data for an unsupported CPU-only training path | Keep the capacity blocker and stop before data work. Accept a factual explanation instead of a prescribed `cpu/local (subprocess)` display format; local hardware is not evidence of platform GPU capacity. |
| Failed validation | Training on bad merged rows or destroying another job's shared data | Keep full-row validation, a new fileset, updated job reference and protection of the shared fileset. Require explaining sampling versus full validation, not memorizing the sample count. Explicitly separate proposed repair/upload commands from execution. |
| GRPO without environment | Optimizing meaningless reward or silently changing the task/data | Keep task-compatible inputs/reward and both choices: a matching environment for the user's data or an explicitly separate demo pair. Do not require an exact helper-skill or demo-dataset name; named examples must still be valid. |
| Terminal status | False success reports or unnecessary duplicate GPU training | Require top-level `completed` for training success, distinguish failed/cancelled terminals, and keep quality unverified after scoring failure. Retry existing-output evaluation rather than automatically retraining. Permit documentation reads while forbidding actual polling/retries. |
| Agent requirements negative | Unrequested training/deployment and failure to answer the requirements question | Retain negative routing and no mutations; additionally require a relevant checklist addressing policy grounding and unsupported answers. Tool inventories, acceptance thresholds and the other example topics are not individually mandatory. Avoiding Customizer alone is insufficient. |
| Schema-only tabular negative | Routing to model training that needs unavailable source rows | Keep Data Designer and no Customizer/job execution. Require explaining the schema-only fit, not an unrequested product comparison. Reject unsupported distribution/privacy claims without requiring a ritual disclaimer. |

All ten IDs, the eight-positive/two-negative split and routing labels are
unchanged. Explanation-only prompts now consistently distinguish documentation
reads from platform/data operations. Existing credential, shared-data,
compatibility, validation and truthful-outcome protections remain. Answers
which abandon the task still fail; these revisions do not fix or exempt the
historical unrelated workspace responses. `SKILL.md` is unchanged.

Fresh paired evaluation would measure this revised task set; old scores must
not be relabeled or selectively regraded as new passing evidence. Validation
of the JSON and offline contract checks establishes input integrity only,
not grader accuracy or model reliability.

### Correction after the first reported run of the reviewed cases

Run `20261125_012333_89934_6169e82605cd` scored 59/60 attempts. The missing
baseline attempt failed before agent execution: Docker could not reach
`registry-1.docker.io` while resolving `python:3.12-slim` (`no route to host`).
There is no complete baseline comparison or valid uplift for this run.

With-skill goals were 27/30, security was clean in all 30 attempts, and every
positive case met its goal in all three attempts. The three missed goals were
all `customizer-negative-agent-design`: the judge required an explicit allowed
tools item, and sometimes concrete acceptance criteria. Inspection of the saved
answers shows useful policy-grounded checklists with uncertainty/escalation and
no Customizer activation. The review had incorrectly turned illustrative
requirements categories into mandatory deliverables for a short outline.

The corrected expected output and assertion require policy grounding and
handling unsupported answers, retain the negative-routing/no-mutation boundary,
and explicitly treat a tool inventory, numeric targets and other example topics
as optional. This is a correction to task scope, not evidence that the live
grader has been calibrated. The three saved with-skill answers satisfy those
requirements on manual review; their original recorded failures remain intact.
No complete rerun is needed to establish that the original rubric demanded an
unrequested item. Live grading of the corrected case has not been verified.

All 59 retained native sessions were checked; none recorded compaction or the
historical unrelated workspace question. `SKILL.md` remains unchanged.

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

## Three-attempt local comparison: 2026-11-24

- Run ID: `20261124_194940_58040_cb8afd80a24c`.
- SkillEvaluator: `1.2.0`; environment: Docker; agent and judge:
  `openai/openai/gpt-6.1-sol`.
- Coverage: 10 cases, 3 attempts per arm; all 60 attempts scored with no
  execution or judge failures. Security was clean in all 30 with-skill attempts.
- With-skill goals met: 23/30; without skill: 18/30. Reported shared-metric
  overall uplift: +2.92 percentage points.
- Every with-skill case passed at least once (pass@3: 10/10), but only 4/10
  passed required checks in every attempt. Seven unmet goals across six cases
  produce Tier 3 FAIL. A success on another attempt does not erase a failed
  required check. Publication was not assessed.
- Source digest:
  `sha256:749075ed57d589ad055af73fa3596773154f9bfde6fe0b45a3293024c0e277d2`.
- Dataset digest:
  `sha256:97af0b68a6abc2c29e9834f62a67ec3725516d876d91229819db75b7331e4f3f`.

### Attempt findings and changes

| Case | Failed attempts | Evidence and response |
| --- | ---: | --- |
| No GPU | 2/3 | The only shell action was reading the installed skill. The original prompt prohibited all commands, conflicting with the intended reference-based workflow. The revised prompt explicitly permits local reference reads while prohibiting platform/setup commands, data processing and resource changes. |
| Default SFT | 1/3 | The plan omitted registration of the base model before serving it. This is a genuine omission from an end-to-end plan. The revised prompt explicitly asks for prerequisite platform resources, and an assertion now checks weights fileset/model registration before serving; the requirement remains. |
| Unsloth | 1/3 | A substantive explanation stopped at validation without submitting anything. The goal judge failed it solely for not restating that later submission needs permission. The revised expected output and assertion enforce the execution boundary and require authorization if the answer recommends proceeding to submission, without requiring an extra permission statement when it stops at explanation/validation. |
| DPO on Docker | 1/3 | The agent read the skill and RL references, then asked what to work on in `/workspace`. This remains a real missed task; all backend-selection and stop-condition requirements are retained. |
| GRPO environment | 1/3 | The agent acknowledged the explanation-only request, then asked what to work on in `/workspace`. This remains a real missed task; environment/dataset compatibility requirements are retained. |
| Synthetic-data negative | 1/3 | The answer led with Safe Synthesizer, then explained Data Designer fits schema-only dummy rows. The original prompt mentioned an existing CSV without explicitly ruling out source-data synthesis. The revised prompt says only column names/types are available and requests no model training or statistical reproduction. Data Designer remains the expected recommendation. |

The no-GPU grader followed the literal original no-command instruction; this
revision resolves the task's conflicting constraints rather than treating that
instruction as optional. All diagnosis is based on the saved trial trajectories
and rewards, not only the report's abbreviated recommendation list.

`SKILL.md` was not changed for these findings. Its workflow already includes model
registration and the DPO/GRPO requirements. A possible follow-up skill improvement
is a concise explanation-only path that skips executable preflight and provides
a plan checklist including resource registration, validation and evaluation.
The repeated unrelated workspace questions also need runtime/model investigation;
the DPO failure in this run followed local reference reads without a recorded web
search, so web search alone does not explain every lost-task response. Neither
cause nor benefit from a skill edit has been demonstrated by this run.

The revised dataset needs fresh paired evaluation. Historical scores above
remain attached to their original dataset digest and have not been recomputed.

## Subsequent three-attempt comparison: 2026-11-24

- Run ID: `20261124_202740_84042_cecbb1897091`.
- SkillEvaluator: `1.2.0`; environment: Docker; agent and judge:
  `openai/openai/gpt-6.1-sol`.
- All 60 attempts scored with no execution or judge failures. Security was
  clean in all 30 with-skill attempts.
- With-skill goals met: 22/30; without skill: 18/30. Reported shared-metric
  overall uplift: -3.67 percentage points; this differs from goal-count change
  because the report compares the scored dimensions, not just goals met.
- With-skill pass@3: 9/10 cases; only 5/10 cases passed required checks on every
  attempt. Eight unmet goals across five cases produce Tier 3 FAIL.
- Source digest:
  `sha256:ff4fb407c0c192d8b8f73c2763c56174fd280feac8116045e4339294745c3a2b`.
- Dataset digest:
  `sha256:4a55374c3b0342e909fa3a717d08450d6cf5249bd3b1b0f1f0286836833b1718`.

Six failures ended with the unrelated question "What would you like me to work
on in `/workspace`?": two default-SFT attempts, one gated-model attempt, one
GRPO attempt and two terminal-status attempts. These are genuine missed tasks,
not reasons to relax their expected outputs. Their runtime/model cause remains
unresolved. The other two failures were substantive plans that omitted the
pre-data confirmation checkpoint (default SFT) or an explicit next step to seek
a compatible platform/profile (DPO). The existing skill already describes both;
no skill edit or new passing score is inferred from this run.

## Unproxied task-history diagnostic: 2026-11-24

- Run ID: `20261124_215659_17408_e10a1f3c2d54`.
- SkillEvaluator: `1.2.0`; Codex: `0.161.0`; environment: Docker; agent and
  judge: `openai/openai/gpt-6.1-sol` through the configured gateway.
- Coverage: 10 cases, 3 with-skill attempts each, concurrency 4; baseline
  skipped. All 30 attempts scored, with native Harbor jobs retained.
- Goals met: 27/30. None of the 30 native sessions ended with the unrelated
  workspace question, and none recorded a native compaction event. Unsloth
  and terminal-status goals passed in all three attempts each.
- Dataset digest:
  `sha256:4a55374c3b0342e909fa3a717d08450d6cf5249bd3b1b0f1f0286836833b1718`.

The dataset digest and recorded model, evaluator, attempts, concurrency and
environment settings match the preceding comparison's with-skill arm. Native
job retention was enabled for this diagnostic. Earlier instrumented native and
Docker runs also did not reproduce task loss; their captured model requests
retained the original task. Those successful traces do not show what the
gateway received or returned during the historical failures. Neither a context
compaction bug nor a gateway/model root cause has been established, and the
absence of failures here is not evidence of a fix. A captured failing request
and response, or corresponding gateway-side trace, is needed to distinguish
request-history loss from model behavior. Another unchanged successful run
would not resolve that distinction.

The remaining missed goals were content omissions: two default-SFT plans
omitted confirmation before data work, with one also omitting the frozen test
split before upload and `deployment_config`; one Docker-blocked DPO answer
omitted Ray while correctly rejecting incompatible backends and requesting a
compatible Kubernetes GPU profile. These are separate from task loss. No
grading requirement or skill instruction was changed for this diagnostic.
The generated suggestions to edit `SKILL.md` require review against its existing
instructions; they are not automatic evidence that the skill is incorrect.
This baseline-free run does not establish uplift or publication readiness.

## Subsequent direct paired comparison: 2026-11-25

- Run ID: `20261125_001328_45357_aecc4d20e23b`.
- SkillEvaluator: `1.2.0`; environment: Docker; agent and judge:
  `openai/openai/gpt-6.1-sol`. Three attempts per case per arm, concurrency 4.
- All 60 attempts scored with no execution or judge failures. With-skill goals:
  27/30; baseline goals: 19/30. Reported shared-metric uplift: +8.25 percentage
  points. All 30 with-skill attempts were security-clean. Tier 3 remains FAIL.
- All 60 native sessions were retained. None recorded native compaction or the
  unrelated workspace question. This direct run did not reproduce task loss;
  no runtime fix is established.
- Source digest:
  `sha256:ff4fb407c0c192d8b8f73c2763c56174fd280feac8116045e4339294745c3a2b`.
- Dataset digest:
  `sha256:4a55374c3b0342e909fa3a717d08450d6cf5249bd3b1b0f1f0286836833b1718`.

Two default-SFT plans omitted confirmation before data work; one also omitted
explicitly freezing the test split before upload. These remain genuine workflow
omissions against requirements already present in the skill.

The validation-recovery failure was different: its only shell command read the
local `SKILL.md` with `cat`. The goal judge accepted its validation and shared-data
safety guidance but correctly applied the prompt's literal "do not run anything"
constraint. As in the earlier no-GPU correction, the revised prompt explicitly
permits local documentation reads while prohibiting platform commands, data
processing, file edits, uploads and submission. A matching assertion preserves
that execution boundary. Shared-fileset protections and full-validation
requirements are unchanged. `SKILL.md` was not edited. The original report
remains unchanged; its scores do not evaluate the revised prompt.

## MR checks and publication evidence

Live agent/judge results are reliability evidence and remain advisory for MRs.
The local request workflow is not part of `ci-status.needs`; the repository's
effective `main` rules checked during this investigation require `Lint all`,
`Python unit tests`, `DCO` and `CI status`, not `NVSkills CI`. No remote rules
were changed. Repository administrators must keep the optional live status out
of required checks during this pilot.

Required plugin unit tests cover deterministic skill paths/frontmatter,
backend validation of documented job JSON, and the authored dataset contract.
They do not establish agent quality or catalog readiness. With the 1.2.0
all-attempts required-check policy, increasing attempts increases exposure to
an individual missed goal; pass@3 does not override that quality gate. Model
sampling and LLM judging cannot be made deterministic merely by repeating runs.
Do not weaken safety checks or discard failed attempts to obtain a green run.

Catalog publication still needs the complete three-tier review, both configured
agents, signatures and verified baseline comparisons described below. No
publication pass is claimed. A future live MR gate needs a separately agreed
statistical regression policy, a stable pinned runtime and a calibrated task set.

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
