<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NeMo Customizer skill card

## Description

Fine-tune open models through the NeMo Helix platform using Automodel, Unsloth,
or NeMo-RL. The skill coordinates platform execution profiles, dataset and model
registration, job validation and monitoring, output deployment, and base-versus-tuned
evaluation. It supports SFT, LoRA, DPO and GRPO as permitted by each backend.

## Owner and license

NVIDIA NeMo Helix Team (`nemo-helix@nvidia.com`). Apache-2.0.
Intended for commercial and non-commercial use globally, subject to the selected
model and dataset licenses and applicable organizational requirements.

## Requirements and dependencies

- An installed NeMo Helix CLI with the Customizer plugin and selected backend contributor.
- A reachable platform exposing jobs, files and models APIs, authentication where enabled,
  a suitable GPU execution profile, and the backend's training image on the executor.
- A Kubernetes-runtime platform with a compatible GPU execution profile for DPO/GRPO.
  Ray is included in the backend training image; users do not install it separately.
  Automodel and Unsloth can also use Docker GPU executors.
- Platform model-serving and Evals capabilities for SFT/LoRA quality comparison.
- Hugging Face credentials in a platform secret for gated/private models. A provider
  API key may be needed for evaluation. Never put raw credentials in prompts or logs.

## Outputs

Confirmed training plans, prepared dataset files, platform filesets and model entities,
validated job specifications, job status and registered outputs, deployments when
requested, and evaluation summaries. Model quality is measured, never guaranteed.

## Known risks and mitigations

- Training and serving consume GPU resources. Confirm the plan and verify execution
  profiles, GPU compatibility and images before preparing data or submitting jobs.
- Private training data can contain sensitive information. Use approved datasets,
  workspaces and credentials; review data access before uploading.
- Reusing a dataset fileset can reuse stale content or replace files used elsewhere.
  Prefer a new fileset when data changes; require explicit confirmation for replacement.
- Malformed data or incompatible backends can fail training. Run dry-run validation,
  use full validation for merged/edited datasets, and stop on failures.
- Training can regress quality. Freeze a held-out test split, compare base and tuned
  results with identical settings, and report evaluation failures separately.
- GRPO with a mismatched reward environment can optimize a meaningless reward.
  Require an environment paired with the dataset.

## Evaluation

The authored [task set](evals/evals.json) contains eight positive workflow cases
and two negative routing controls. It covers default SFT, user-data Unsloth,
platform credentials, DPO backend constraints, unavailable GPUs, dataset validation,
GRPO environment selection, and job completion/reporting.

These are explanation and decision tasks that can run without training GPUs or live
platform credentials. They measure workflow correctness and routing, not successful
GPU training. Run both `claude-code` and `codex` with and without the skill using
SkillEvaluator; record security, correctness, discoverability, effectiveness and
efficiency, including baseline uplift. See [BENCHMARK.md](BENCHMARK.md) for status.

## References

- [Workflow](SKILL.md)
- [SFT job commands](references/fast-path-sft.md)
- [DPO and GRPO commands](references/fast-path-rl.md)
- [Base versus tuned evaluation](references/post-training-eval.md)
- [Troubleshooting and platform prerequisites](references/troubleshooting.md)
