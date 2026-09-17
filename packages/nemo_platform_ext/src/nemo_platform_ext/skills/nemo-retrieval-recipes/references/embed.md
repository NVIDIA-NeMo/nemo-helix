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
| 4 Export | Automodel job writes both layouts. Set `training.retrieval.export.primary` to `onnx` or `hf`. Newer Retriever NIMs (2.2.0+) need `hf` at the fileset root. | Retriever NIM layout |
| 5 Deploy | ModelDeployment + Retriever NIM 2.2.0 | `/v1/embeddings` |

Default stop after `retrieve-eval`. Fileset-backed models must be served first
(`deploy.md`). Do not run a separate export step. Primary vs alternate is a
job setting: `training.retrieval.export.primary: hf` puts Hugging Face weights
at the fileset root (ONNX under `alternates/onnx`). Retriever NIM 2.2.0 and
newer require that HF-primary layout.

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
  "prepare": {
    "enable_mining": true,
    "model": "default/nemotron-3-embed-1b"
  }
}'
```

Corpus, generation, and split control live in `sdg.md`, including how to reuse a published Stage 0 dump.

Mining needs `enable_mining: true` and `model` as a platform entity with an encoder fileset. Do not mine when convert produced an empty train split. Convert-only filesets leave `neg_doc: []` and Automodel crashes sampling negatives — run the `sdg.md` pre-submit check before Stage 2.

### Register the trainable base

A model entity auto-discovered from Inference Gateway is an endpoint: its
`fileset` is null and it cannot be trained. Register the checkpoint as a
fileset-backed entity and confirm the fileset before submitting.

`<model-revision>` means the Hugging Face model revision to pin: use a commit
SHA or tag from the model repository. For a non-reproducible latest-revision
fileset, remove the `"revision"` field instead of leaving the placeholder.

```bash
nemo files filesets create nemotron-3-embed-1b \
  --workspace default --purpose model --exist-ok \
  --storage '{
    "type":"huggingface",
    "repo_id":"nvidia/Nemotron-3-Embed-1B-BF16",
    "repo_type":"model",
    "revision":"<model-revision>"
  }'
nemo models create nemotron-3-embed-1b \
  --workspace default --exist-ok \
  --fileset default/nemotron-3-embed-1b \
  --custom-fields '{"hf_model_id":"nvidia/Nemotron-3-Embed-1B-BF16"}'
nemo models get nemotron-3-embed-1b --workspace default
```

Do not submit until the last command reports
`"fileset": "default/nemotron-3-embed-1b"`.

Stage 2 (`dataset.training` is the Stage 1 `artifacts` fileset):

```json
{
  "model": "default/nemotron-3-embed-1b",
  "dataset": {"training": "default/retrieval-stage1-artifacts"},
  "training": {
    "recipe": "bi_encoder",
    "training_type": "sft",
    "finetuning_type": "lora_merged",
    "retrieval": {"export": {"primary": "hf"}}
  },
  "output": {"name": "nemotron-3-embed-1b-tuned"}
}
```

Leave batch/LR unset to take Nemotron retrieval defaults. Do not set `max_steps` with `epochs`.
Pass the Stage 1 artifacts fileset as-is: `training.jsonl` and `eval_beir/` sit
at the result root; wrapped `train.json` and mining caches are under `additional/`.

Stage 3 reads the same fileset — the BEIR loader accepts a root containing `eval_beir`.
Both `target` and `baseline` must already have IGW providers (`deploy.md`) before submit:

```bash
nemo evaluator retrieve-eval submit --spec '{
  "dataset": "default/retrieval-stage1-artifacts",
  "target": "default/nemotron-3-embed-1b-tuned",
  "baseline": "default/nemotron-3-embed-1b",
  "k": [1, 5, 10, 100]
}'
```

## Stage 4 / deploy

Confirm the fileset layout after the job; do not re-export. Newer Retriever
NIMs load PyTorch weights, so set `training.retrieval.export.primary: hf` on
the Automodel job (HF at the root, ONNX under `alternates/onnx`). Leave
`primary: onnx` only for NIMs that still expect ONNX at the fileset root.
Set `training.retrieval.export.dimensions: true` for Matryoshka.

Create Deployment Manager configs and wait for `READY` using `deploy.md`
(baked base `model_spec: {}`, tuned fileset mount, Retriever NIM 2.2.0).
Unmerged LoRA cannot be served.

## Invariants

- Same prefixes, max length, pooling, and `eval_beir` for base and tuned.
- Do not use CHAT `evaluate submit` or RAGAS for this path.
