<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Embedding recipe (Platform)

## Stage map

| Stage | Platform command | Output |
|---|---|---|
| 0 SDG | `nemo data-designer retrieval-generate` | stage0 fileset |
| 1 Prep | `nemo data-designer retrieval-prepare` | `training.jsonl` + `eval_beir/` |
| 0+1 | `nemo data-designer retrieval-run` | both |
| 2 Finetune | `nemo customization automodel submit` `recipe: bi_encoder` | model entity |
| 3 Eval | `nemo evaluator retrieve-eval submit` | `eval_results.json` |
| 4 Export | Automodel ONNX export (`embeddings`) + `alternates/hf/` | Retriever NIM layout |
| 5 Deploy | ModelDeployment + Retriever NIM 2.2.0 | `/v1/embeddings` |

Default stop at Stage 3. Stage 4 is part of the Automodel job, not a separate Nemotron exporter.

## Commands

Chain Stage 0+1 (preferred):

```bash
nemo data-designer retrieval-run --workspace default --spec '{
  "generate": {
    "corpus": "default/my-docs",
    "provider": "default/nvidia-build",
    "artifact_extraction_model": "nvidia/nemotron-3-nano-30b-a3b",
    "qa_generation_model": "nvidia/nemotron-3-nano-30b-a3b",
    "quality_judge_model": "nvidia/nemotron-3-nano-30b-a3b",
    "embed_model": "nvidia/nemotron-3-embed-1b"
  },
  "prepare": {"enable_mining": false}
}'
```

Skip SDG:

```bash
nemo data-designer retrieval-prepare --spec '{
  "sdg_input": "hf://nvidia/Retrieval-Synthetic-NVDocs-v1@<revision>",
  "enable_mining": false
}'
```

Mining needs `enable_mining: true` and `model` as a platform entity with an encoder fileset. Do not mine when convert produced an empty train split.

Stage 2 (dataset fileset holds `training.jsonl` at the root):

```json
{
  "model": "default/nemotron-3-embed-1b",
  "dataset": {"training": "default/stage1-prep"},
  "training": {
    "recipe": "bi_encoder",
    "training_type": "sft",
    "finetuning_type": "lora_merged"
  },
  "output": {"name": "nemotron-3-embed-1b-tuned"}
}
```

Leave batch/LR unset to take Nemotron retrieval defaults. Do not set `max_steps` with `epochs`.

Stage 3:

```bash
nemo evaluator retrieve-eval submit --spec '{
  "dataset": "default/eval-beir",
  "target": "default/nemotron-3-embed-1b-tuned",
  "baseline": "default/nemotron-3-embed-1b",
  "k": [1, 5, 10, 100]
}'
```

## Stage 4 / deploy

Automodel post-processing exports embedding ONNX (`input_ids`, `attention_mask`
→ `embeddings`, plus a Matryoshka `dimensions` input when
`training.retrieval.export.dimensions: true`) and moves HF weights under
`alternates/hf/`. Set
`training.retrieval.export.primary: hf` when the target NIM loads PyTorch weights.
Deploy the **output** model entity (full weights or merged LoRA) with
`nvcr.io/nim/nvidia/nemotron-3-embed-1b:2.2.0`. Pass `input_type` query vs document.
Do not run `nemotron embed export`. Unmerged LoRA cannot be served.

## Invariants

- Same prefixes, max length, pooling, and `eval_beir` for base and tuned.
- Do not use CHAT `evaluate submit` or RAGAS for this path.
