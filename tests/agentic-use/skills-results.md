<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Skills Impact on Agentic Evals

Tracking the impact of Claude Code skills on non-easy CLI eval performance.

## Results

Results across two runs. First batch had 7 infra failures (agent ran as root) which were fixed with `runuser -u harbor` in entrypoints, then rerun.

| Eval | Baseline | With Skills | Time | Skills Invoked |
|------|----------|-------------|------|----------------|
| garak-plugin-config-crud-cli | 0.0 (1:12) | **1.0** | 5:08 | garak-plugin |
| garak-plugin-default-job-cli | - | **1.0** | 2:29 | garak-plugin |
| garak-plugin-target-crud-cli | - | **1.0** | 3:19 | garak-plugin |
| auth-authorization-cli | - | **1.0** | 4:40 | nemo-auth |
| data-designer-config-cli | - | **1.0** | 2:28 | nemo-secrets, nemo-inference-providers |
| entities-basic-cli | - | **1.0** | 2:24 | nemo-entities |
| evaluator-academic-benchmark-cli | - | 0.0 | 10:28 | (genuine failure) |
| evaluator-llm-judge-cli | - | **1.0** | 7:43 | nemo-evaluator, nemo-files, nemo-secrets |
| evaluator-simple-job-cli | 0.0 (1:10) | **1.0** | 4:11 | nemo-evaluator, nemo-files, nemo-auth |
| files-crud-cli | 0.0 (3:05) | **1.0** | 4:34 | nemo-files |
| files-upload-dataset-cli | - | **1.0** | 3:07 | nemo-files |
| guardrails-content-safety-cli | 1.0 (3:24) | **1.0** | 2:08 | nemo-guardrails |
| guardrails-custom-config-cli | 0.0 (13:17) | **1.0** | 7:26 | nemo-secrets, nemo-inference-providers, nemo-guardrails |
| inference-chat-completions-cli | - | **1.0** | 3:09 | nemo-inference-gateway |
| inference-igw-provider-cli | - | 0.0 | 0:31 | (infra: root user, fixed) |
| inference-mockllm-cli | - | **1.0** | 1:59 | nemo-inference-providers, nemo-inference-gateway |
| inference-provider-reg-cli | 0.0 (3:07) | **1.0** | 3:57 | nemo-secrets, nemo-inference-providers |
| secrets-crud-cli | 0.0 (3:05) | **1.0** | 3:54 | nemo-secrets |
| workspace-basic-cli | - | **1.0** | 1:24 | nemo-auth |

**Summary**: 17/19 passed (89%). 1 genuine failure (evaluator-academic-benchmark), 1 infra failure (inference-igw-provider, fixed with runuser).

## Key Findings

1. **Skills dramatically improve pass rate**: Where we have baselines, skills turned 0.0 failures into 1.0 passes (garak-plugin-config-crud, evaluator-simple-job, files-crud, guardrails-custom-config, inference-provider-reg, secrets-crud)
2. **Skill invocation requires instruction hints**: The agent reliably invokes skills when the instruction.md says "You have skills available for X. Use them before exploring --help."
3. **Root user bug**: Harbor's agent setup re-installs Claude Code as root. Fixed by adding `exec runuser -u harbor` to all ENTRYPOINT scripts.

## Skills Created

| Skill | Description | Evals Using It |
|-------|------------|----------------|
| nemo-secrets | Secrets CRUD | secrets-crud, evaluator-llm-judge, data-designer-config, guardrails-custom-config, inference-provider-reg |
| nemo-inference-providers | Provider setup + update-status | data-designer-config, guardrails-custom-config, inference-provider-reg, inference-mockllm |
| nemo-inference-gateway | Chat completions, provider gateway, mock providers | inference-chat-completions, inference-mockllm |
| nemo-guardrails | Guardrail config + self-check rails | guardrails-content-safety, guardrails-custom-config |
| garak-plugin | Scan configs, targets, jobs, probes | garak-plugin-config-crud, garak-plugin-default-job, garak-plugin-target-crud |
| nemo-evaluator | Metrics, sync/async evals, benchmarks | evaluator-llm-judge, evaluator-simple-job |
| nemo-files | Filesets, upload/download, datasets | files-crud, files-upload-dataset, evaluator-llm-judge, evaluator-simple-job |
| nemo-entities | Entity CRUD (model, dataset types) | entities-basic |
| nemo-auth | Workspaces + member management | auth-authorization, workspace-basic, evaluator-simple-job |
