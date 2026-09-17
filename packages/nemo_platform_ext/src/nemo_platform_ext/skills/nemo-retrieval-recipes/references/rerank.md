<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Rerank recipe (Platform)

Keep the first-stage retriever fixed while comparing base and tuned rerankers.

## Prerequisites

- A Stage 1 `artifacts` fileset from `sdg.md` whose every `training.jsonl` row
  has a non-empty `neg_doc` list.
- A fileset-backed reranker model entity for training.
- A fixed embedding model and base reranker served through Inference Gateway
  for the baseline and target evaluation.

## Stage map

| Stage | Platform command | Output |
|---|---|---|
| 0–1 | Same Data Designer jobs as embed | `training.jsonl` + `eval_beir/` |
| 2 | `nemo customization automodel submit` `recipe: cross_encoder` | HF sequence-classification entity |
| 3 | `retrieve-eval` with `target.embeddings` + `target.reranker`, `first_stage_k: 100` | `eval_results.json` |
| 4 | Produced automatically by the successful Automodel job: ONNX (`logits`) + `alternates/hf/` | Ranking NIM layout |
| 5 | Ranking NIM `llama-nemotron-rerank-1b-v2:1.10.0` `/v1/ranking` | Deploy the **output** model entity |

Default stop after `retrieve-eval` (checkpoint / IGW model-ref eval). Fileset-backed
rerankers need the same Deployment Manager path (`deploy.md`) before submit.
Do not run a separate export step: a successful Automodel job already writes
the Stage 4 ONNX/HF layout.

## Commands

Stage 0+1 is identical to embed; see `sdg.md`. Run the `sdg.md` pre-submit
`neg_doc` check before Stage 2 — convert-only JSONL makes `cross_encoder` fail
the same way as `bi_encoder`. Both stages below read the Stage 1 `artifacts`
fileset directly.

Register a fileset-backed reranker checkpoint if it does not already exist; an
Inference Gateway endpoint entity has no fileset and cannot be trained:

`<model-revision>` means the Hugging Face model revision to pin: use a commit
SHA or tag from the model repository. For a non-reproducible latest-revision
fileset, remove the `"revision"` field instead of leaving the placeholder.

```bash
nemo files filesets create llama-nemotron-rerank-1b-v2 \
  --workspace default --purpose model --exist-ok \
  --storage '{
    "type":"huggingface",
    "repo_id":"nvidia/llama-nemotron-rerank-1b-v2",
    "repo_type":"model",
    "revision":"<model-revision>"
  }'
nemo models create llama-nemotron-rerank-1b-v2 \
  --workspace default --exist-ok \
  --fileset default/llama-nemotron-rerank-1b-v2 \
  --custom-fields '{"hf_model_id":"nvidia/llama-nemotron-rerank-1b-v2"}'
nemo models get llama-nemotron-rerank-1b-v2 --workspace default
```

Do not submit until the model entity reports a non-null fileset.

Stage 2:

```json
{
  "model": "default/llama-nemotron-rerank-1b-v2",
  "dataset": {"training": "default/retrieval-stage1-artifacts"},
  "training": {
    "recipe": "cross_encoder",
    "training_type": "sft",
    "finetuning_type": "lora_merged"
  },
  "output": {"name": "llama-nemotron-rerank-1b-v2-tuned"}
}
```

Prompt template must stay `question:{query} \n \n passage:{passage}`. Do not
merge LoRA with a causal-LM merge path.

Stage 3 (two-stage):

```bash
nemo evaluator retrieve-eval submit --spec '{
  "dataset": "default/retrieval-stage1-artifacts",
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
Every embeddings and reranker ref in that spec needs an IGW provider (`deploy.md`)
before submit.

## Stage 4 / deploy

Automodel post-processing exports ONNX at the fileset root and HF weights under
`alternates/hf/`. Create Deployment Manager configs and wait for `READY` using
`deploy.md` (Ranking NIM `llama-nemotron-rerank-1b-v2:1.10.0`).
Unmerged LoRA cannot be served.

## Invariants

Primary KPI is nDCG@10. Use Recall@100 on the embedder to decide whether to tune embed instead.

## Next Steps

- Compare base and tuned nDCG@10 in `eval_results.json`.
- If reranking improves, use `deploy.md` to serve the tuned output model.
- If Recall@100 is the bottleneck, keep the reranker fixed and follow
  `embed.md`.
