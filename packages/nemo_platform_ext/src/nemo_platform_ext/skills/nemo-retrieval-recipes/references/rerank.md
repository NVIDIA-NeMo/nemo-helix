<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Rerank recipe (Platform)

Keep the first-stage retriever fixed while comparing base and tuned rerankers.

## Stage map

| Stage | Platform command | Output |
|---|---|---|
| 0–1 | Same Data Designer jobs as embed | `training.jsonl` + `eval_beir/` |
| 2 | `nemo customization automodel submit` `recipe: cross_encoder` | HF sequence-classification entity |
| 3 | `retrieve-eval` with `target.embeddings` + `target.reranker`, `first_stage_k: 100` | `eval_results.json` |
| 4 | Automodel ONNX export (`logits`) + `alternates/hf/` | Ranking NIM layout |
| 5 | Ranking NIM `llama-nemotron-rerank-1b-v2:1.10.0` `/v1/ranking` | Deploy the **output** model entity |

Default stop at Stage 3 (checkpoint / IGW model-ref eval). Stage 4 is part of the Automodel job, not a separate Nemotron exporter.

## Commands

Stage 2:

```json
{
  "model": "default/llama-nemotron-rerank-1b-v2",
  "dataset": {"training": "default/stage1-prep"},
  "training": {
    "recipe": "cross_encoder",
    "training_type": "sft",
    "finetuning_type": "lora_merged"
  },
  "output": {"name": "llama-nemotron-rerank-1b-v2-tuned"}
}
```

Prompt template must stay `question:{query} \n \n passage:{passage}` (Automodel collator). LoRA merge must use the cross-encoder merge path, not causal-LM merge.

Stage 3 (two-stage):

```bash
nemo evaluator retrieve-eval submit --spec '{
  "dataset": "default/eval-beir",
  "target": {
    "embeddings": "default/llama-nemotron-embed-1b-v2",
    "reranker": "default/llama-nemotron-rerank-1b-v2-tuned",
    "first_stage_k": 100
  },
  "baseline": {
    "embeddings": "default/llama-nemotron-embed-1b-v2",
    "reranker": "default/llama-nemotron-rerank-1b-v2",
    "first_stage_k": 100
  },
  "k": [1, 5, 10, 100]
}'
```

Do not change `first_stage_k` or the embedding model between baseline and target.

## Stage 4 / deploy

Automodel post-processing exports sequence-classification ONNX (`input_ids`,
`attention_mask` → `logits`) and moves HF weights under `alternates/hf/`. Deploy
the output entity with `nvcr.io/nim/nvidia/llama-nemotron-rerank-1b-v2:1.10.0`.
Do not run `nemotron rerank export`. Unmerged LoRA cannot be served.

## Invariants

Primary KPI is nDCG@10. Use Recall@100 on the embedder to decide whether to tune embed instead.
