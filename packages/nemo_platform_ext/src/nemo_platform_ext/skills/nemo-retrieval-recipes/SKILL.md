---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: nemo-retrieval-recipes
description: >-
  End-to-end NeMo Platform recipe for domain embedding and reranking fine-tuning:
  Data Designer retrieval-generate/prepare, Automodel bi_encoder/cross_encoder,
  Evaluator retrieve-eval on frozen eval_beir, then optional NIM deploy. Use when
  the user has a document corpus, wants better retrieval nDCG/recall, or asks to
  fine-tune Nemotron embed or rerank models on Platform.
triggers:
  - fine-tune embedding
  - fine-tune rerank
  - embedding recipe
  - retrieval recipe
  - domain retrieval
  - nDCG embedding
  - retrieve-eval
  - Nemotron embed
  - Nemotron rerank
  - bi_encoder
  - cross_encoder
  - retrieval-sdg
not-for:
  - nemo-customizer (chat/SFT/LoRA/DPO/GRPO on language models)
  - nemo-data-designer-plugin Autopilot create (tabular synthetic columns)
  - public BEIR/MTEB leaderboard eval unrelated to a domain corpus
  - generic vector-database selection
preconditions:
  - nemo_cli_available
  - platform_running
  - workspace_exists
compatibility: NeMo Platform with data-designer [retrieval-sdg], customizer automodel, and evaluator retrieve-eval plugins.
maturity: active
license: Apache-2.0
user-invocable: true
allowed-tools: [Bash, Read, Grep]
version: "0.1.0"
---

# NeMo Platform retrieval recipes

Conduct the Nemotron embed/rerank pipeline **on NeMo Platform plugin CLIs**. Do not
run `uv run nemotron embed|rerank`. Do not use `nemo data-designer create`.

Plugin skills (`nemo-data-designer-plugin`, `nemo-customizer`, `nemo-evaluator-plugin`)
are stage manuals. This skill owns family routing, artifact IDs, frozen eval, and
the 15% nDCG@10 / Recall@10 README bar.

## Resolve CLI

Use `nemo`, or `uv run nemo` from the nemo-platform root. If neither works, route
to `setup`. Confirm `http://localhost:8080/health/ready` before submitting jobs.

## Choose a family

Read only the matching recipe after this routing step:

- Low Recall@k, missing relevant documents, first-stage retrieval → `references/embed.md`
- Recall@100 OK but nDCG@10 poor, relevant docs buried in top-k → `references/rerank.md`
- User asks both or a two-stage stack → embed first (candidate coverage), then rerank.

Use one workspace. Stage 1 must produce `training.jsonl` and a frozen BEIR fileset
whose root contains `corpus.jsonl`, `queries.jsonl`, and `qrels/test.tsv`. Keep
`query:` / `passage:` prefixes for embed and
`question:{query} \n \n passage:{passage}` for rerank.

## Safe workflow

1. Confirm corpus governance: Stage 0 sends chunk text to Inference Gateway chat
   models. Do not ask users to paste API keys. Use `provider` / `chat_provider` /
   `embed_provider`, not `NVIDIA_API_KEY` on the job.
2. Dry-run schemas first: `nemo data-designer retrieval-generate --help`,
   `nemo customization automodel explain`, `nemo evaluator retrieve-eval explain`.
3. Prefer skip-SDG (`hf://nvidia/Retrieval-Synthetic-NVDocs-v1@<rev>` or an existing
   `generation_result.json` fileset) for a first eval. Live SDG needs a corpus of
   **50+ documents**; one file can dump every query into the test split and leave
   train empty (mining then crashes).
4. Run Stage 0+1 once (`retrieval-run` or generate then prepare). **Freeze**
   `eval_beir`. Never regenerate it for base vs fine-tuned comparisons.
5. Fine-tune with explicit `training.recipe: bi_encoder` or `cross_encoder` and
   `finetuning_type` `all_weights` or `lora_merged` only.
6. Default stop after `retrieve-eval` with `baseline` + tuned target, `k: [1,5,10,100]`.
7. Deploy is opt-in. Automodel writes ONNX at the fileset root (`alternates/hf/`
   for the HF checkpoint) for both `bi_encoder` and `cross_encoder`. Embed: Retriever
   NIM 2.2.0. Rerank: Ranking NIM `llama-nemotron-rerank-1b-v2:1.10.0`. Do not shell
   out to `nemotron embed|rerank export`.
8. Poll jobs at 60–300s. Parse JSON from **stdout only** (never `2>&1` into `json.load`).

Report absolute and relative nDCG@10, Recall@10, and Recall@100 on the same frozen
eval set. Prefer at least 100 queries; warn below 50. Treat 15% relative nDCG@10
and Recall@10 uplift as an indicative embed target, not a hard small-corpus gate.

References: `references/embed.md`, `references/rerank.md`.

For stage-specific failures, hand off to `nemo-data-designer-plugin`,
`nemo-customizer`, `nemo-evaluator-plugin`, or `inference`.
