<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Customizer evaluation status

**Codex Tier 3 PASS; publication pending.** The latest verified comparison met
10/10 goals with the skill versus 8/10 without it, with +6.08 percentage points
reported shared-metric uplift. This is one attempt per case, not evidence of
repeatability or catalog readiness.

## Latest verified result

- Run ID: `20261125_020204_14943_21b62c17f8ab`.
- Evaluated at: `2026-11-25T02:13:22Z` (UTC).
- SkillEvaluator: `1.2.0`; Docker environment; Codex agent and judge model:
  `openai/openai/gpt-6.1-sol` through the OpenAI-compatible Inference Hub
  endpoint. Judge reasoning effort: `medium`.
- Dataset: `evals/evals.json`, 10 cases (8 positive, 2 negative controls).
  One attempt per case in each arm, concurrency 4, with a fresh baseline.
- All 20 attempts scored with no execution or judge failures. Native jobs were
  retained. The evaluated prompts, expected outputs, assertions and routing
  fields matched the authored dataset.

| Measure | With skill | Without skill |
| --- | ---: | ---: |
| Goals met | 10/10 | 8/10 |
| Cases passing required checks | 10/10 | 8/10 |
| Security-clean attempts | 10/10 | 10/10 |

The reported Codex completed-agent score is **97.42%**. The **+6.08 percentage
points uplift** compares shared metrics across the two arms; these are distinct
report measures. The verdict is **Codex Tier 3 PASS**. Publication eligibility
was not assessed.

- Evaluated source digest:
  `sha256:ace6c246164cc11779fd0d030255739dc27907983ec728c9405d14aa91e6cff7`.
- Dataset digest:
  `sha256:2221988158008f3e43723b8dae5da8c8a5195bd3cf8b8ded94dc6796f6bfdd08`.
- Source Git revision and evaluator container revision were not recorded by
  the orchestration input; the digests identify the evaluated content.

The generated report is stored locally at
`evals/results/20261125_020204_14943_21b62c17f8ab/report.html` (ignored by Git).

## Limitations

- One attempt per case does not establish a stable failure rate or a reliable
  live MR gate. Earlier runs exposed intermittent task loss; its cause remains
  unresolved, and this pass does not establish a fix.
- Only Codex was evaluated. Claude Code, Tiers 1 and 2, and publication/signing
  were not assessed by this run.
- The cases test explanations and workflow decisions without GPU training.
  They do not demonstrate completed training, deployment or model improvement.
- Results depend on the recorded dataset, runtime, agent model and judge.
  Changes require new evidence; this result does not validate grader calibration.

## Reproduce locally

Use SkillEvaluator `1.2.0` (`se`) with Docker running and an Inference Hub key
provided through `SKILL_EVAL_LLM_API_KEY`. No local GPUs or running Helix platform
are needed for these cases. From the repository root:

```bash
export SKILL_EVAL_LLM_PROVIDER=openai-compatible
export SKILL_EVAL_LLM_BASE_URL=https://inference-api.nvidia.com/v1
export SKILL_EVAL_LLM_MODEL=openai/openai/gpt-6.1-sol

skill="$PWD/plugins/nemo-customizer/src/nemo_customizer/skills/nemo-customizer"
se tier3 validate "$skill" --strict
se tier3 evaluate "$skill" --agents codex \
  --env-mode docker --n-attempts 1 --n-concurrent 4 --harbor-keep-jobs
se view "$skill"
```

This reproduces the evaluation configuration, not guaranteed scores. Preserve
all attempts; rerunning until green does not establish reliability.

## Publication requirements

- Evaluate the canonical skill and dataset through all three tiers: static
  validation/security, catalog deduplication and live sandbox evaluation.
- Obtain passing results and positive baseline uplift for both `codex` and
  `claude-code`, each evaluated with and without the skill.
- Record the evaluated source revision, evaluator and agent/model versions,
  environment, coverage, per-agent results and verdicts.
- Resolve blocking validation/security findings and review overlap with
  `nemotron-customize`.
- Retain the verified central NVSkills report, generated skill card and
  `skill.oms.sig` from the signing service before catalog publication.

Local evaluation provides evidence; it does not replace central validation and
signing. The CI request and publication procedure is documented in
[the CI README](../../../../../../.github/CI_README.md).
