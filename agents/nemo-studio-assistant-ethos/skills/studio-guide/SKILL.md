---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: studio-guide
description: "Map of NeMo Studio: what each area is for, which studio_link destination opens it, and what to suggest next. Use for orientation questions such as where do I start, what can I do here, where is X, or what should I do next."
metadata:
  author: NeMo Helix Team <nemo-helix@nvidia.com>
---
# Finding your way around Studio

Use this skill when the user wants orientation rather than a specific action.
Start from the page in the Studio context: explain what that page is for,
then suggest the most useful next step.

## How to answer

1. Say what the user is looking at, or what the feature they asked about does,
   in one or two sentences.
2. Check the workspace with a read-only `nemo_api` call when the answer depends
   on what exists, for example `models` `list` or `inference.providers` `list`.
3. Suggest one concrete next step and link to it with `studio_link`.

## Studio areas

| Area | What it is for | `studio_link` destination |
|---|---|---|
| Dashboard | Workspace overview and starting point | `dashboard` |
| Inference providers | Connect a model endpoint such as build.nvidia.com or OpenAI | `inference_providers` |
| Base models | Models registered in the workspace | `base_models` |
| Model chat | Try and compare available models | `model_chat` |
| Virtual models | Route traffic to models, optionally through guardrails | `virtual_models` |
| Guardrails | Safety policies such as content safety and topic control | `guardrails` |
| Agents | Register, deploy, chat with, and evaluate agents | `agents` |
| Agent evaluations | Score an agent against a test set | `agent_evaluations` |
| Evaluation | Metrics, benchmarks, and evaluation results for models | `evaluation` |
| Customizations | Fine-tune a model on your own data | `customizations` |
| Data Designer | Generate synthetic datasets | `data_designer` |
| Safe Synthesizer | Produce privacy-safe synthetic copies of a dataset | `safe_synthesizer` |
| Filesets | Upload and browse datasets and other files | `filesets` |
| Secrets | Store API keys and other credentials by reference | `secrets` |
| Jobs | Track long-running platform jobs | `jobs` |
| Intake | Traces and telemetry from agents | `intake` |
| Settings and members | Workspace settings and access | `settings`, `members` |

Some areas are behind feature flags. If `studio_link` reports that a
destination is unavailable, say the feature is not enabled in this Studio
rather than sending the user to a broken page.

## Common starting paths

- **Nothing is set up yet.** Suggest adding an inference provider first, then
  trying a model in Model chat.
- **"I want to build an agent."** Point to Agents, and explain that agents use
  models from this workspace, so a working model comes first.
- **"Is my agent any good?"** Point to Agent evaluations, and to Intake for
  traces of real runs.
- **"Make my model safer."** Explain guardrails and VirtualModels; read the
  `guardrails` skill before going further.
- **"I need data."** Filesets for data the user has, Data Designer to generate
  new data, Safe Synthesizer to make a shareable copy of sensitive data.
