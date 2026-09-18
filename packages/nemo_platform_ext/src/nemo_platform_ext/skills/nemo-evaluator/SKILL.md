---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: nemo-evaluator
description: >
  NeMo Evaluator SDK-first rubric-to-eval guide for BYOB (Bring Your Own
  Benchmark): parse domain expert rubrics, choose composable evaluation
  primitives, generate human-reviewable eval configs/artifacts, and run
  reproducible exact, numeric, LLM-as-judge, code/custom, composite,
  RAG/agentic, and tool-calling evaluations. Use when a task involves
  questions/rubrics/responses files, rubric criteria, benchmark scoring,
  evaluator primitive selection, or reusable evaluation artifacts.
preconditions:
  - evaluator_sdk_available
compatibility: Designed for installed NeMo Platform skill use; repo-relative SDK paths are developer fallbacks when a checkout is available.
allowed-tools: Bash, Read
metadata:
  author: NeMo Helix Team <nemo-helix@nvidia.com>
  user-invocable: true
---

# NeMo Evaluator

Use this skill to turn a domain-specific benchmark and expert-written rubric
into a reproducible evaluation. The agent should parse the rubric, choose the
simplest correct composable primitive for each criterion, generate
human-reviewable config/artifacts, run the evaluation, and explain both scores
and reasoning.

Use the NeMo Evaluator SDK as the source of truth for metric names, fields,
templates, execution modes, result shapes, and failure behavior. Keep this skill
focused on SDK guidance. Use CLI commands only when the user explicitly needs a
remote platform job or an existing platform resource.

## Runtime Context

This skill can run in two contexts:

- In the NeMo Platform repo, expect the SDK to be importable through the repo
  environment, for example with `uv run` commands from the repo root. Use the
  repo-relative SDK reference paths in `references/sdk-execution.md` when
  checking exact schemas or tests.
- As an installed skill outside the repo, do not assume the source tree is
  present. Use installed Python imports, package metadata, CLI help, or
  user-provided files first; treat repo paths as developer fallback references.

Prefer local SDK runs for iteration and smoke tests. Use remote platform jobs
only when the user needs platform-managed execution, existing platform
resources, or a job result from a deployed environment. For local SDK runs,
`api_key_secret` resolves through the local environment; for remote jobs, the
same name should refer to a platform secret in the job workspace.

## Reference Files

Load these files only when the task needs the detail:

- Metric selection: `references/metric-selection.md`
  - Read when mapping rubric criteria to SDK metrics, choosing deterministic vs
    LLM judges, or working with RAG/agentic/tool-calling metrics.
- SDK execution: `references/sdk-execution.md`
  - Read before guessing a metric schema, execution parameter, online/offline
    setup, parser, template, or SDK result shape.
- Benchmark reproduction: `references/benchmark-reproduction.md`
  - Read for BYOB protocol, long-running benchmark reproduction, artifact
    contracts, and external-safe output requirements.
- Troubleshooting: `references/troubleshooting.md`
  - Read when SDK runs fail, judge outputs do not parse, provider calls time
    out, or generated artifacts need recovery.

## Default Loop

1. Identify the BYOB inputs: questions, rubric criteria, responses or live
   model endpoints, weights, pass threshold, and desired summary breakdowns.
2. Clarify the evaluation target: model output quality, judge quality, RAG
   quality, tool use, benchmark regression, or bring-your-own benchmark
   reproduction.
   - Judge-quality evaluation checks whether a judge agrees with labels for
     existing responses.
   - Generator-quality evaluation creates fresh responses and scores those
     responses with a fixed judge or deterministic checker.
3. Choose the simplest correct SDK metric class or composed set of classes for
   each criterion. See `references/metric-selection.md`.
4. Generate a small, human-reviewable config or code artifact instead of
   burying criterion logic in one-off scripts.
5. Build a tiny inline dataset with at least one expected pass and one expected
   fail.
6. Run locally with `Evaluator().run_sync(...)` or `await Evaluator().run(...)`.
   See `references/sdk-execution.md`.
7. Inspect `result.print_summary()`, `row_scores`, and `aggregate_scores`.
8. When errors arise, fix dataset shape, Jinja templates, parser paths,
   prompts, model config, or secrets. See `references/troubleshooting.md`.
9. Move to remote platform jobs only after the local SDK run behaves as
   expected.

Do not stop at "the score is bad." Explain what failed and what evidence
supports that conclusion.

## Core Selection Rules

Prefer deterministic primitives before LLM calls. Use LLM judges for nuance,
not for checks that can be expressed as exact, string, numeric, or code logic.

Use online generation only when the evaluator should call a model or agent
before scoring. Pass `target=Model(...)` or `target=Agent(...)` and a
`prompt_template`; otherwise keep evaluation offline and put outputs in the
dataset rows.

For external, custom, or bring-your-own benchmarks, keep the protocol explicit
and reproducible.

Separate judge-quality evaluation from generation-quality evaluation. If the
target is report generation, generate fresh model responses, score them with a
fixed judge, and aggregate fulfilment. Do not treat human labels for existing
baseline responses as labels for new model outputs.

For public or broadly shared skills and reports, avoid internal endpoint names,
internal model IDs, authentication details, and secret paths. Use placeholders
such as `<provider-base-url>`, `<generator-model-id>`, `<judge-model-id>`, and
`<secret-reference>`. Report parameter categories and aliases instead of
revealing private infrastructure.

## Completion Criteria

A good evaluator answer includes:

- The evaluation goal and chosen SDK metric class or benchmark shape.
- The dataset shape and any templates or parser paths.
- The exact SDK snippet or evaluation spec used.
- Row-level and aggregate evidence for local runs, or job status/result
  evidence for remote runs.
- Any caveats about reproducibility, provider config, or calibration.
